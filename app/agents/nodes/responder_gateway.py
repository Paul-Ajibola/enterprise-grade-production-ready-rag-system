import logfire
from app.agents.state import AgentState
from app.gateways.client import portkey_client, extract_cache_status, get_langchain_llm
from app.agents.nodes.responder import generate_node as generate_with_groq


# function to generate node
def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History
    Uses the native Portkey client (not LangChain) so we can read the 
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    if portkey_client is None:
        return generate_with_groq(state)

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
                rendered_doc = f"Source: {doc.get('source', 'Unknown source')}\n{doc.get('content', '')}"
            else:
                rendered_doc = str(doc)
            if len(full_context) + len(rendered_doc) < max_context_chars:
                full_context += rendered_doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits")
                break

        # Fixed: Moved out of the loop so it executes correctly
        if full_context:
            prompt = f"""
            You are a Senior Technical Architect. Answer using the supplied technical context.
            If the context does not support an answer, say so instead of guessing.

            TECHNICAL CONTEXT:
            {full_context}

            CONVERSATION HISTORY:
            {history_str}

            LATEST USER MESSAGE:
            {user_msg}
            """
        else:
            prompt = f"""
            You are a Senior Technical Architect. The knowledge base returned no relevant context.
            Tell the user you could not find supporting documentation and ask them to refine the question.
            Do not invent technical facts.

            CONVERSATION HISTORY:
            {history_str}

            LATEST USER MESSAGE:
            {user_msg}
            """

    with logfire.span("LLM Synthesis"):
        try:
            if state.get("streaming"):
                streaming_llm = get_langchain_llm(feature="responder")
                content_parts = []
                for chunk in streaming_llm.stream(prompt):
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
                return {
                    "final_answer": content,
                    "status": "Response generated.",
                    "plan": state["plan"],
                    "messages": [{"role": "assistant", "content": content}],
                }

            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("Portkey cache hit; response served from cache.")
                plan_update = state["plan"] + ["Cache: Hit"]
                status = "Cache hit instant response."
            else:
                logfire.info("Response synthesised through LLM.")
                plan_update = state["plan"]
                status = "Response generated."
            
            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }


        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            return generate_with_groq(state)

