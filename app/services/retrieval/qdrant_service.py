import logfire
from qdrant_client import QdrantClient
from qdrant_client import models
from app.config import settings
from app.services.retrieval.embeddings import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def search_enterprise_knowledge(query: str, limit: int=8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)
        collection_info = client.get_collection(settings.QDRANT_COLLECTION)
        vector_config = collection_info.config.params.vectors
        if isinstance(vector_config, dict):
            raise RuntimeError("Named Qdrant vectors are not supported by this retrieval service.")
        if len(query_vector) != vector_config.size:
            raise RuntimeError(
                f"Query embeddings are {len(query_vector)}-dim but the Qdrant collection is "
                f"{vector_config.size}-dim. Re-embed the collection with the active model."
            )

        # Using query_points -- the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True #JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": (res.payload or {}).get("text", ""),
                "source": (res.payload or {}).get("source", "Unknown"),
                "source_type": (res.payload or {}).get("source_type", "Unknown"),
                "score": res.score
            })
        return results

    except Exception as e:
        logfire.error(f"Qdrant Search Failed: {e}")
        raise RuntimeError("Qdrant search failed; check the collection, credentials, and embedding dimensions.") from e

