import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

# Before importing the retriever, you import the reranker
# Retriever node only works when the query is technical in nature


# function to retrieve node
def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    reranked_contents = []

    with logfire.span("Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        with logfire.span("Semantic Reranking"):
            reranked_contents = rerank_documents(query, raw_results, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

    return {
        "documents": reranked_contents,
        "status": "Found technical context." if reranked_contents else "No relevant context found.",
        "plan": state["plan"] + (["Context Retrieved"] if reranked_contents else ["No context found"]),
    }

