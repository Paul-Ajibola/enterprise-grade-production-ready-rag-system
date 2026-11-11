import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embeddings import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise_ingestion-service")

PROCESSED_DATA_DIR = "processed_data"

_locked_dim: int | None = None

# Initialize the Qdrant client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

# function to save processed file locally
def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    # Replaced trailing comma-space with a dot for clean extensions
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest

# function to process file
def process_file(file_path: str, filename: str, source_type: str) -> str:
    """Parse -> Chunk -> Save Locally -> Embed -> Index in Qdrant"""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            ext = filename.split(".")[-1].lower() if "." in filename else ""

            # 1. extraction of the txt based on file extension
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported extension {filename}")
                return "skipped"

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} - skipping.")
                return "skipped"

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return "skipped"

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,            
            }

            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"saved processed data -> {local_path}")

            with logfire.span("Vectorizing and Indexing"):
                active_dim = get_embedding_dim()
                if _locked_dim is not None and active_dim != _locked_dim:
                    raise RuntimeError(
                        f"Active embedding model is {active_dim}-dim but the existing Qdrant "
                        f"collection is {_locked_dim}-dim. Rebuild the collection with --wipe "
                        "before ingesting with a different embedding model."
                    )

                embeddings = embed_texts(chunks)
                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        f"Created {len(embeddings)} embeddings for {len(chunks)} chunks."
                    )
                if _locked_dim is not None and any(
                    len(vector) != _locked_dim for vector in embeddings
                ):
                    raise RuntimeError(
                        f"Embedding output does not match the {_locked_dim}-dim Qdrant collection. "
                        "Rebuild the collection with --wipe using the active embedding model."
                    )
                
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Ingested {len(points)} points to Qdrant from {filename}.")
                return "processed"

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return "failed"


# function to process directory
def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        counts = {"processed": 0, "skipped": 0, "failed": 0}
        for root, _, filenames in os.walk(dir_path):
            logfire.info(f"Found {len(filenames)} files in {root}.")
            for filename in filenames:
                result = process_file(os.path.join(root, filename), filename, source_type)
                counts[result] += 1
        return counts


# function to run universal ingested data
def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    global _locked_dim

    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested explicitly
        if wipe and qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
            logfire.info(f"Wiped existing collection '{settings.QDRANT_COLLECTION}'")

        # Recreate collection -> Dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection {settings.QDRANT_COLLECTION} "
                f"({dim}-dim, Cosine)."
            )
        else:
            # ADJUSTMENT 1: Collection already exists (e.g. no --wipe this run) -
            # read its actual configured vector size instead of assuming it
            # matches whatever embedding model happens to be active right now.
            existing_info = qdrant_client.get_collection(settings.QDRANT_COLLECTION)
            dim = existing_info.config.params.vectors.size

        # ADJUSTMENT 2: Record the dimension this run's collection is locked to.
        # process_file() checks every batch's embedding dimension against this
        # value so a mid-run Gemini -> sentence-transformers fallback switch
        # gets caught and skipped instead of silently corrupting the collection
        # with mixed 3072-dim / 768-dim vectors.
        _locked_dim = dim

        totals = {"processed": 0, "skipped": 0, "failed": 0}
        if explicit_source_type:
            counts = process_directory(base_dir, explicit_source_type)
            for key in totals:
                totals[key] += counts[key]
        else:
            subdirs = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]

            if not subdirs:
                fallback_type = os.path.basename(os.path.normpath(base_dir))
                counts = process_directory(base_dir, fallback_type)
                for key in totals:
                    totals[key] += counts[key]
            else:
                for d in subdirs:
                    counts = process_directory(os.path.join(base_dir, d), d)
                    for key in totals:
                        totals[key] += counts[key]

        logfire.info("Ingestion summary", **totals)
        if totals["failed"]:
            raise RuntimeError(f"Ingestion had {totals['failed']} failed or unsupported files.")
        return totals


if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None
    
    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    try:
        run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
        logfire.info("Ingestion job completed!")
    except Exception as e:
        logfire.error(f"Ingestion job failed: {e}")
        sys.exit(1)
