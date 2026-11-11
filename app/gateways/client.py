import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import settings



# Production gateway config:
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant
#   - Cache: semantic mode (requires )
#   - Retry: 2 attempts on rate limit


# set up the gateway configuration
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
    ]
}


portkey_client = (
    Portkey(api_key=settings.PORTKEY_API_KEY, config=GATEWAY_CONFIG)
    if settings.PORTKEY_API_KEY
    else None
)


# "rag" is the name of the virtual key on Portkey
def get_langchain_llm(feature: str = "rag"):
    """
    Returns a Portkey-backend ChatOpenAI - a drop-in for ChatGroq

    Why ChatOpenAI and not ChatGroq:
        Portkey is a proxy. It exposes an OpenAI-compartible endpoint at PORTKEY_GATEWAY_URL.
        ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
        ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
        auth + config). The @rag/model-name format is Portkey-specific -- Groq's own client
        does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    if not settings.PORTKEY_API_KEY:
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0,
        )

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=GATEWAY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "rag_system-today",
                "environment": "dev"
            }
        )
    )


# model format is from the documentation


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively - returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
            return "MISS"
    return "MISS"    # added: fallback when none of the attrs exist

