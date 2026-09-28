#--------------------------------------------------------------
# CRITICAL: Logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the starts

import logfire
import os
from dotenv import load_dotenv

load_dotenv()

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graphs import rag_agent
from pydantic import BaseModel
from typing import Optional
# for the guardrails
from app.guardrails.rails import initialize_rails, guard


# initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")

# if on_event is deprecated. Use `.lifespan()`
@app.on_event("startup")
def startup_event():
    initialize_rails()


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


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
        return {"error": f"Could not generate graph image: {e}"}


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
        "status": "Initializing Graph..."
    }

    # Configuration for Memory (Thread ID) - Now properly indented!
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_output = rag_agent.invoke(initial_state, config=config)
        return {
            "question": query,  # Fixed: Changed from 'q' to 'query'
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }

    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": query,
            "answer": "I apologize, I encountered an internal error while processing your request",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }

# use uvicorn to test the api


