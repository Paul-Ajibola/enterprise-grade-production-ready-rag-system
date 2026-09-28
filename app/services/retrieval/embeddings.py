import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from langchain_cohere import CohereEmbeddings


BATCH_SIZE = 50
_GEMINI_DIM = 3072   # gemini's dimension
_COHERE_DIM = 1024     # cohere v3 default dimension
_FALLBACK_DIM = 768   # all-mpnet-base-v2


_active_model = None
_model_type: str | None = None   # "gemini" or "fallback"


def _probe_cohere():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=settings.COHERE_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Cohere embeddings ready (cohere-english-v3.0, 1024-dim)")
        return model
    except Exception as e:
        logfire.warning(f"Cohere API probing failed: {e}. Using the sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    global _active_model, _model_type

    if _active_model is not None:
        return

    cohere_model = _probe_cohere()
    if cohere_model:
        _active_model = cohere_model
        _model_type = "cohere"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def get_embedding_dim() -> int:
    """Returns the vector dimension for the currently active model.
    IMPORTANT: because the active model can switch mid-run (gemini -> fallback),
    callers that need to pick a Qdrant collection for a given file's vectors
    should call this AFTER embed_texts() returns for that file, not before —
    otherwise you may size/select the wrong collection if a switch happened
    during that call.
    """
    _init()
    return _COHERE_DIM if _model_type == "cohere" else _FALLBACK_DIM


def _switch_to_fallback():
    """Permanently switch to sentence-transformers for the remainder of this process."""
    global _active_model, _model_type
    logfire.warning(
        "Gemini rate limit persisted after max retries - "
        "switching to sentence-transformers fallback for the remainder of this run."
    )
    _active_model = _load_fallback()
    _model_type = "fallback"


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "cohere":
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "limit exceeded", "blocked"))

                if not is_rate_limit:
                    logfire.error(f"Cohere embedding failed: {e}")
                    raise

                if attempt < max_attempts - 1:
                    wait = 2 ** (attempt + 1)
                    logfire.warning(
                        f"Cohere rate limit hit - retrying in {wait}s "
                        f"(attempt {attempt + 1}/{max_attempts})."
                    )
                    time.sleep(wait)
                else:
                    _switch_to_fallback()

        # Only reached if the loop above just switched to fallback.
        return _active_model.encode(batch, show_progress_bar=False).tolist()

    return _active_model.encode(batch, show_progress_bar=False).tolist()


def _reembed_all_with_fallback(texts: list[str]) -> list[list[float]]:
    """
    Re-embed an ENTIRE text list from scratch using only the fallback model.
    Used when a mid-call switch (cohere -> fallback) happens partway through
    embed_texts(), so we never return a list mixing 1024-dim and 768-dim
    vectors. Any partial Cohere results already produced in that call are
    discarded and redone here for dimensional consistency.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch (fallback re-embed)", model="fallback", start_idx=i, size=len(batch)):
            all_embeddings.extend(_active_model.encode(batch, show_progress_bar=False).tolist())
    return all_embeddings


def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "cohere":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start_idx=i, size=len(batch)):
            model_before = _model_type
            batch_embeddings = _embed_batch(batch)

            if _model_type != model_before:
                # Switched from gemini -> fallback partway through this call.
                # Everything embedded so far in all_embeddings is 3072-dim;
                # the fallback model produces 768-dim. Discard the partial
                # results and re-embed this WHOLE file's texts with the
                # fallback model so embed_texts() always returns a single,
                # dimensionally-consistent list.
                logfire.warning(
                    f"Model switched from '{model_before}' to '{_model_type}' mid-file. "
                    "Discarding partial results and re-embedding the full input "
                    "with the fallback model for dimensional consistency."
                )
                return _reembed_all_with_fallback(texts)

            all_embeddings.extend(batch_embeddings)

            if _model_type == "cohere":
                time.sleep(1.5)    # rest the model for 1.5 seconds before hitting the API again

    return all_embeddings


