import logfire
from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig


from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup
    Uses llama-3.1-8b-instant for fast intent classification at 
    the heavier llama-3.3-70b-versatile is reserved for the RAG processing 
    (this is where the guardrails check the logs)
    """

    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0
    )


    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    # so, we just pass the rules into the LLM
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("NeMo Guardrails initialized (llama-3.1-8b-instant)")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.
    Returns:
        (True, rail_response) - a rail fired, return this repsonse
                                skip the RAG pipeline entirely.
        (False, None)         - message is clean; proceed to LangGraph
    """
    if _rails is None:
        logfire.warning("Guardrails not initialized - skipping gateway")
        return False, None

    with logfire.span("Guardrails Check"):
        results = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {"role": "assistant", "content": "..."}
        content = results.get("content", "") if isinstance(results, dict) else ""

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"Guardrails fired | query='{message}' | response='{content}'")
            return True, content

        logfire.info(f"Guardrails passed.")
        return False, None


