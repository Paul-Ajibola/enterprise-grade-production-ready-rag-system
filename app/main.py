#--------------------------------------------------------------
# CRITICAL: Logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the starts

import logfire
import os
import json
from dotenv import load_dotenv

load_dotenv()

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from app.agents.graphs import checkpointer, rag_agent
from pydantic import BaseModel
# for the guardrails
from app.guardrails.rail import guard, initialize_rails


# initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")

# if on_event is deprecated. Use `.lifespan()`
@app.on_event("startup")
def startup_event():
    initialize_rails()


class QueryRequest(BaseModel):
    q: str
    thread_id: str


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live with lovely audience"}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error(f"Could not generate graph image: {e}")
        raise HTTPException(status_code=500, detail="Could not generate the graph image.") from e


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Delete the persisted conversation checkpoint for a UI session."""
    try:
        checkpointer.delete_thread(thread_id)
        return {"deleted": True, "thread_id": thread_id}
    except Exception as e:
        logfire.error(f"Could not delete conversation thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not delete conversation memory.") from e


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using POST request
    """
    query = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "documents": [],
        "plan": ["start"],
        "status": "Initializing Graph...",
        "streaming": False,
    }

    # Configuration for Memory (Thread ID) - Now properly indented!
    config = {"configurable": {"thread_id": thread_id}}

    try:
        rail_fired, rail_response = guard(query)
        if rail_fired:
            return {
                "question": query,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "blocked",
                "sources": [],
            }

        final_output = rag_agent.invoke(initial_state, config=config)
        if final_output.get("status") == "error":
            raise HTTPException(status_code=502, detail=final_output.get("final_answer"))
        return {
            "question": query,  # Fixed: Changed from 'q' to 'query'
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        raise HTTPException(status_code=502, detail="The RAG backend failed to process the request.") from e


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Stream answer tokens as server-sent events, followed by the final response payload."""
    query_text = request.q
    thread_id = request.thread_id
    initial_state = {
        "messages": [{"role": "user", "content": query_text}],
        "current_query": query_text,
        "documents": [],
        "plan": ["start"],
        "status": "Initializing Graph...",
        "streaming": True,
    }
    config = {"configurable": {"thread_id": thread_id}}

    def emit(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            rail_fired, rail_response = guard(query_text)
            if rail_fired:
                yield emit({
                    "type": "final",
                    "question": query_text,
                    "answer": rail_response,
                    "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                    "status": "blocked",
                    "sources": [],
                })
                return

            for chunk, metadata in rag_agent.stream(
                initial_state, config=config, stream_mode="messages"
            ):
                if metadata.get("langgraph_node") != "responder":
                    continue
                text = chunk.content
                if isinstance(text, str) and text:
                    yield emit({"type": "token", "token": text})
                elif isinstance(text, list):
                    for block in text:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield emit({"type": "token", "token": block.get("text", "")})

            final_output = rag_agent.get_state(config).values
            if final_output.get("status") == "error":
                yield emit({"type": "error", "error": final_output.get("final_answer")})
                return
            yield emit({
                "type": "final",
                "question": query_text,
                "answer": final_output.get("final_answer", ""),
                "thought_process": final_output.get("plan", []),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", []),
            })
        except Exception as e:
            logfire.error(f"Streaming backend execution failed: {e}")
            yield emit({"type": "error", "error": "The RAG backend failed to process the request."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# use uvicorn to test the api


