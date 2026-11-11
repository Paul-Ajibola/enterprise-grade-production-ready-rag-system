import logfire
from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq

# Direct Groq response node used when Portkey is not configured or unavailable.
llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.1)
#  lower temperature because it is preferably to be deterministic, than probabilistic


# function to generate node
def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if isinstance(doc, dict):
                source = doc.get("source", "Unknown source")
                content = doc.get("content", "")
                rendered_doc = f"Source: {source}\n{content}"
            else:
                rendered_doc = str(doc)
            if len(full_context) + len(rendered_doc) < max_context_chars:
                full_context += rendered_doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits")
                break

        if not full_context:
            prompt = f"""
            You are a Senior Technical Architect. The knowledge base returned no relevant context.
            Tell the user you could not find supporting documentation and ask them to refine the question.
            Do not invent technical facts.

            CONVERSATION HISTORY:
            {history_str}

            LATEST USER MESSAGE:
            {user_msg}
            """
        else:
            prompt = f"""
            You are a Senior Technical Architect. Answer using the supplied technical context.
            Include the source filename when it helps identify where a fact came from.
            If the context does not support an answer, say so instead of guessing.

            TECHNICAL CONTEXT:
            {full_context}

            CONVERSATION HISTORY:
            {history_str}

            LATEST USER MESSAGE:
            {user_msg}
            """

    with logfire.span("LLM Synthesis"):
        try:
            content_parts = []
            for chunk in llm.stream(prompt):
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    content_parts.append(chunk_content)
                elif isinstance(chunk_content, list):
                    content_parts.extend(
                        block.get("text", "")
                        for block in chunk_content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
            content = "".join(content_parts)
            logfire.info("Response synthesized via LLM.")

            return {
                "final_answer": content,
                "status": "Response generated",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            return {
                "final_answer": "I apologize, but I encountered an internal error while processing.",
                "status": "error",
                "plan": state["plan"] + ["LLM generation failed"],
                "messages": [{"role": "assistant", "content": "I apologize, but I encountered an internal error while processing."}]
            }


