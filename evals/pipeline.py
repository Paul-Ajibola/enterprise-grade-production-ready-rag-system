"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: full actual_response, actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import copy
import json
import os
import uuid
import requests
import logfire

API_URL = f"{os.getenv('BACKEND_URL', 'http://localhost:8000').rstrip('/')}/query"
DELAY_BETWEEN_CALLS = 10   # seconds — stays within Groq RPM on the main key
REQUEST_TIMEOUT = 120      # seconds — guardrails + LangGraph + Groq can take >60s




def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)  # deepcopy to maintain strict data isolation and prevent state pollution
    samples = dataset["rag_samples"]
    n = len(samples)
    run_id = uuid.uuid4().hex

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate (samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_run_{run_id}_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("status") == "error":
                        raise RuntimeError(data.get("answer") or "Backend returned an error status.")

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer
                    sample["actual_contexts"] = [
                        source.get("content", "") if isinstance(source, dict) else str(source)
                        for source in sources[:5]
                    ]
                    sample["actual_error"] = None
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["actual_tools_called"] = ["unknown"]
                    sample["actual_error"] = str(e)

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)

