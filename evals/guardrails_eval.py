"""
Guardrails binary evaluation
Sends each test input to the live / query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""


import copy
import os
import time
import uuid
import requests
import logfire


API_URL = f"{os.getenv('BACKEND_URL', 'http://localhost:8000').rstrip('/')}/query"

def _is_blocked(response_json: dict) -> bool:
    tp = response_json.get("thought_process") or []
    return any("guardrails fired" in step.lower() for step in tp)


def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked and result (TP/TN/FP/FN) to each sample in place.
    Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)
    run_id = uuid.uuid4().hex

    with logfire.span("Eval - Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, n, sample["input"])

            expected = sample["expected_blocked"]
            error = None

            with logfire.span(
                f"Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=expected,
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": sample["input"], "thread_id": f"guardrail_eval_{run_id}_{i}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    blocked = _is_blocked(resp.json())

                except Exception as e:
                    logfire.error(f"Guardrails test error: {e}")
                    error = str(e)
                    blocked = None

                sample["actual_blocked"] = blocked
                sample["actual_error"] = error

                if error:
                    sample["result"] = "ERROR"
                elif expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                logfire.info(
                    f" {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=blocked,
                    input_preview=sample["input"][:60],
                )
            time.sleep(2)

    return samples



def compute_guardrails_metrics(results: list) -> dict:
    tp = sum(1 for r in results if r["result"] == "TP")
    tn = sum(1 for r in results if r.get("result") == "TN")
    fp = sum(1 for r in results if r.get("result") == "FP")
    fn = sum(1 for r in results if r.get("result") == "FN")
    errors = sum(1 for r in results if r.get("result") == "ERROR")
    evaluated = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "evaluated": evaluated,
        "errors": errors,
        "correct": tp + tn,
    }




