import time
import logfire
from flashrank import Ranker, RerankRequest


# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() runs first
_ranker = None

def get_ranker() -> Ranker:
    """
    Initializes the FlashRank Engine Lazily.
    FlashRank uses a local ONNX model (ms-macro-MiniLM-L-6-v2)
    for ultra reranking
    """
    global _ranker
    if _ranker is None:
        logfire.info("Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """
    Refines retrieval results by re-scoring documents against the query semantically
    """
    if not documents:
        return []
    
    start_time = time.time()
    logfire.info(f"[Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoders")

    try:
        ranker = get_ranker()

        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(documents)]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        # Results are returned sorted by highest semantic score first
        reranked_docs = []
        for res in results[:top_n]:
            document = dict(documents[int(res["id"])])
            document["rerank_score"] = res.get("score")
            reranked_docs.append(document)

        duration = time.time() - start_time
        top_score = results[0]["score"] if results else "N/A"
        logfire.info(f"[Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")

        return reranked_docs

    except Exception as e:
        logfire.error(f"[Reranker] Semantic Reranking Failed: {e}.")
        
        return documents[:top_n]

