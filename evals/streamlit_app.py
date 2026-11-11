import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import logfire
import streamlit as st

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

from evals.guardrails_eval import compute_guardrails_metrics, run_guardrails_eval
from evals.metrics import run_all_metrics
from evals.pipeline import load_golden_dataset, run_pipeline


RESULTS_PATH = Path(__file__).with_name("results.json")


def save_results(dataset: dict) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as result_file:
        json.dump(dataset, result_file, ensure_ascii=False, indent=2)


st.set_page_config(page_title="RAG Evaluation", page_icon="🧪", layout="wide")
st.title("RAG Evaluation")
st.caption("Run live RAG queries, check guardrails, and score responses against the golden dataset.")

if "eval_dataset" not in st.session_state:
    try:
        st.session_state.eval_dataset = (
            json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if RESULTS_PATH.exists()
            else load_golden_dataset()
        )
    except Exception as exc:
        st.error(f"Could not load the golden dataset: {exc}")
        st.stop()

dataset = st.session_state.eval_dataset
tab_pipeline, tab_guardrails, tab_metrics = st.tabs(
    ["Live RAG pipeline", "Guardrails", "RAG metrics"]
)

with tab_pipeline:
    rag_samples = dataset.get("rag_samples", [])
    st.write(f"Golden questions: **{len(rag_samples)}**")
    if st.button("Run live pipeline", type="primary", disabled=not rag_samples):
        progress = st.progress(0.0)
        status = st.empty()

        def update_pipeline_progress(index, total, question, stage, response=""):
            progress.progress(min((index + (stage == "done")) / max(total, 1), 1.0))
            status.write(f"{stage.title()}: {question}")

        try:
            st.session_state.eval_dataset = run_pipeline(dataset, update_pipeline_progress)
            dataset = st.session_state.eval_dataset
            rag_samples = dataset.get("rag_samples", [])
            save_results(dataset)
            st.success(f"Pipeline run complete. Results saved to {RESULTS_PATH.name}.")
        except Exception as exc:
            st.error(f"Pipeline evaluation failed: {exc}")

    if rag_samples:
        st.dataframe(
            [
                {
                    "Question": sample.get("question", ""),
                    "Tool": (sample.get("actual_tools_called") or [""])[0],
                    "Answer": sample.get("actual_response", ""),
                    "Error": sample.get("actual_error") or "",
                }
                for sample in rag_samples
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_guardrails:
    guardrail_samples = dataset.get("guardrails_samples", [])
    st.write(f"Guardrail cases: **{len(guardrail_samples)}**")
    if st.button("Run guardrails evaluation", disabled=not guardrail_samples):
        progress = st.progress(0.0)
        status = st.empty()

        def update_guardrail_progress(index, total, input_text):
            progress.progress(min(index / max(total, 1), 1.0))
            status.write(f"Checking {index + 1}/{total}: {input_text}")

        try:
            results = run_guardrails_eval(guardrail_samples, update_guardrail_progress)
            dataset["guardrails_samples"] = results
            guardrail_samples = results
            st.session_state.eval_dataset = dataset
            save_results(dataset)
            st.session_state.guardrail_metrics = compute_guardrails_metrics(results)
            st.success(f"Guardrails results saved to {RESULTS_PATH.name}.")
        except Exception as exc:
            st.error(f"Guardrails evaluation failed: {exc}")

    if "guardrail_metrics" in st.session_state:
        st.json(st.session_state.guardrail_metrics)
    if guardrail_samples:
        st.dataframe(guardrail_samples, use_container_width=True, hide_index=True)

with tab_metrics:
    st.write("LLM metrics use `JUDGE_GROQ` (or `GROQ_API_KEY`) and the captured live pipeline results.")
    if st.button("Run RAG metrics", type="primary"):
        status = st.empty()

        try:
            metric_frames = asyncio.run(
                run_all_metrics(dataset, status_cb=lambda message: status.info(message))
            )
            st.session_state.metric_frames = metric_frames
            st.success("Metric scoring complete.")
        except Exception as exc:
            st.error(f"RAG metric scoring failed: {exc}")

    for metric_name, frame in st.session_state.get("metric_frames", {}).items():
        with st.expander(metric_name.replace("_", " ").title(), expanded=True):
            st.metric("Average", f"{frame[metric_name].mean():.3f}")
            st.dataframe(frame, use_container_width=True, hide_index=True)
