from langgraph.graph import StateGraph, END
import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.planner_gateway import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder_gateway import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)

# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

# 3. Define the Edges and Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

# This entry point. Graph starts from the planner node here
workflow.set_entry_point("planner")

# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# Persist thread state across API restarts.
checkpoint_path = Path(
    os.path.expanduser(settings.CHECKPOINT_DB_PATH)
)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
checkpoint_connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_connection)
checkpointer.setup()


# compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)

