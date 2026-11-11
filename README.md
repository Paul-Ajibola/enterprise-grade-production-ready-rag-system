# 🤖 Enterprise Agentic RAG

A development/demo **agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **FastAPI**, and **Streamlit**. The planner chooses between a conversational response and document retrieval; technical answers are generated from retrieved context. Logfire and LangSmith tracing are optional.

---

## ✨ Key Features

- **🧠 Agentic Planning** — A LangGraph-powered planner node inspects the full conversation history and decides, on every turn, whether the user's message is *conversational* (answerable from memory) or *technical* (requires a fresh knowledge-base search).
- **🔍 Hybrid Retrieval Pipeline** — Vector search against **Qdrant** followed by **semantic reranking** with FlashRank cross-encoders to surface only the most relevant chunks.
- **🔁 Resilient Embeddings** — Uses **Cohere** when its key and API are available, otherwise loads a local **sentence-transformers** model. Ingestion checks the active model dimension against the Qdrant collection and stops with rebuild instructions when they differ.
- **💾 Persistent Conversational Memory** — LangGraph's SQLite checkpointer stores per-session/thread conversation state across backend restarts. The UI's clear-memory action removes the saved thread.
- **🛡️ NeMo Guardrails** — Configured Colang flows handle greetings, capability questions, farewells, selected off-topic prompts, and selected jailbreak phrases before a request enters the RAG graph. These configured flows are not a comprehensive security filter.
- **📄 Multi-Format Ingestion** — Built-in text extraction for PDFs (pypdf with a pdfplumber retry), HTML, plain text, and Office documents (`.docx`, `.pptx`) via `unstructured`. Image-only PDF pages are not OCR-processed.
- **✂️ Smart Chunking** — Recursive character-based text splitting tuned for RAG retrieval quality.
- **📊 Optional Observability** — Logfire spans are emitted by the ingestion pipeline, API, retrieval, and UI. LangSmith tracing can also be enabled with credentials.
- **💬 Streaming Chat UI** — A Streamlit front-end shows a working status while the request runs, streams answer tokens, and displays retrieved source chunks with their filenames and Qdrant scores. The plan is shown after the response completes.
- **🗂️ Universal Ingestion CLI** — Point the ingestion script at a directory and it auto-discovers sub-folders as source types, chunks + embeds + upserts everything into Qdrant, with an optional `--wipe` flag to rebuild the collection from scratch.

---

## 🏗️ Architecture

### Agentic Workflow (LangGraph)

```
                     ┌─────────────┐
                     │   planner   │
                     └──────┬──────┘
                            │
                 route_planner (conditional)
                            │
              ┌─────────────┴─────────────┐
              │                           │
      "CONVERSATIONAL"              (search query)
              │                           │
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │   responder   │  ◄────────│   retriever   │
      └───────┬───────┘           └───────────────┘
              │
              ▼
             END
```

1. **`planner`** — Calls the configured LLM (Groq directly, or Portkey when enabled) with conversation history to classify the query as `CONVERSATIONAL` or produce a refined search query.
2. **`retriever`** *(conditionally invoked)* — Searches Qdrant for the top 15 candidate chunks, then reranks down to the top 5 using FlashRank's cross-encoder.
3. **`responder`** — Synthesizes the answer from retrieved context or conversational history, then appends it to persistent thread state. Technical answers with no retrieved context are prompted to say supporting documentation was not found.

Before LangGraph runs, NeMo Guardrails evaluates the request against the configured Colang flows. The guardrail model uses Groq's OpenAI-compatible API with `llama-3.1-8b-instant`.

### Request Flow (Application)

```
┌────────────┐     POST /query/stream     ┌─────────────┐
│ Streamlit  │ ─────────────────────────► │   FastAPI   │
│    UI      │ ◄───────────────────────── │   (main.py) │
└────────────┘   answer + sources + plan  └──────┬──────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │  LangGraph Agent   │
                                        │   (rag_agent)      │
                                        └─────────┬──────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                 ┌────────────────┐     ┌──────────────────┐    ┌──────────────────┐
                 │ Groq / Portkey  │     │  Qdrant Vector DB  │    │  FlashRank (ONNX) │
                 │ (planner/answer)│     │   (semantic search)│    │  local reranker   │
                 └────────────────┘     └──────────────────┘    └──────────────────┘
```

### Ingestion Pipeline

```
Raw Files (PDF / HTML / TXT / DOCX / PPTX)
         │
         ▼
   Format-specific Loader   (parse_pdf / parse_html / parse_text / parse_office)
         │
         ▼
   Recursive Text Splitter  (chunk_text — 1500 chars, 200 overlap)
         │
         ▼
   Embedding Service         (Cohere primary → sentence-transformers fallback)
         │
         ▼
   Qdrant Upsert             (dimension-locked per collection run)
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, SQLite checkpointer) |
| LLM Inference | [Groq](https://groq.com/) — planner/responder use `llama-3.3-70b-versatile`; guardrails use `llama-3.1-8b-instant` |
| Optional LLM Gateway | Portkey routing for planner/responder when `PORTKEY_API_KEY` is configured; direct Groq otherwise, with direct Groq fallback if Portkey response generation fails |
| Guardrails | NeMo Guardrails with configured Colang flows |
| Vector Database | [Qdrant](https://qdrant.tech/) (cloud cluster) |
| Embeddings | [Cohere](https://cohere.com/) `embed-english-v3.0` (primary) → `sentence-transformers/all-mpnet-base-v2` (local fallback) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (local ONNX cross-encoder) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend UI | [Streamlit](https://streamlit.io/) |
| Observability | [Logfire](https://logfire.pydantic.dev/) and optional LangSmith tracing |
| Document Parsing | `pypdf`, `pdfplumber`, `BeautifulSoup4`, `unstructured` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Package/Env Management | `uv` (`pyproject.toml` + `uv.lock`) |

---

## 📁 Project Structure

```
<project-root>/
├── app/
│   ├── agents/
│   │   ├── nodes/
│   │   │   ├── planner_gateway.py    # Planner using Portkey or direct Groq
│   │   │   ├── retriever.py          # Vector search + reranking node
│   │   │   ├── responder_gateway.py  # Response generation and streaming
│   │   │   └── responder.py          # Direct Groq response fallback
│   │   ├── graphs.py             # LangGraph wiring + SQLite checkpointing
│   │   └── state.py              # Shared AgentState (TypedDict) definition
│   ├── gateways/
│   │   └── client.py             # Optional Portkey client and direct-Groq factory
│   ├── guardrails/
│   │   ├── colang_rules.py       # NeMo Colang flows and Groq model config
│   │   └── rail.py               # NeMo Guardrails initialization and request gate
│   ├── ingestion/
│   │   ├── chunking/
│   │   │   └── splitter.py       # RecursiveCharacterTextSplitter wrapper
│   │   ├── loaders/
│   │   │   ├── pdf.py            # pypdf + pdfplumber fallback parser
│   │   │   ├── html.py           # BeautifulSoup parser
│   │   │   ├── text.py           # Plain text parser
│   │   │   └── office.py         # .docx / .pptx via `unstructured`
│   │   └── processor.py          # Orchestrates parse → chunk → embed → upsert
│   ├── services/
│   │   └── retrieval/
│   │       ├── embeddings.py     # Cohere + sentence-transformers fallback logic
│   │       ├── qdrant_service.py # Qdrant client + query_points search
│   │       └── ranking_service.py# FlashRank semantic reranker
│   ├── main.py                   # FastAPI app and API routes
│   ├── main_rails.py             # Compatibility alias for app.main:app
│   └── config.py                 # Environment-driven settings
├── ui/
│   └── app.py                    # Streamlit chat interface
├── evals/
│   ├── golden_dataset.json        # Evaluation questions, references, and guardrail cases
│   ├── pipeline.py                # Live API evaluation and captured outputs
│   ├── guardrails_eval.py         # Guardrails checks and confusion-matrix metrics
│   ├── metrics.py                 # RAGAS and inferred tool-correctness metrics
│   └── streamlit_app.py           # Evaluation dashboard
├── DATA/                          # User-provided ingestion documents (not included)
├── processed_data/                # Local JSON snapshots produced during ingestion
├── .env                           # Environment variables (not committed)
├── .env.example                   # Environment variable template
├── .gitignore
├── .python-version
├── commands.txt                   # Handy CLI command reference
├── pyproject.toml                 # Project metadata & dependencies
├── requirements.txt               # Pip-compatible dependency list
├── uv.lock                        # uv dependency lockfile
└── README.md
```

The default graph uses `planner_gateway.py` and `responder_gateway.py`. `planner.py` remains as an unused direct-Groq planner implementation.

---

## ⚙️ Environment Variables

Create a `.env` file at the project root with the following keys:

```env
# Required by the API and guardrails
GROQ_API_KEY=
QDRANT_CLUSTER_ENDPOINT=
QDRANT_API_KEY=

# Optional: if absent or unavailable, local sentence-transformers is used
COHERE_API_KEY=

# Optional observability and evaluation
LOGFIRE_TOKEN=
JUDGE_GROQ=
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=enterprise-grade-rag

# UI, evaluation dashboard, and persistent thread storage
BACKEND_URL=http://localhost:8000
CHECKPOINT_DB_PATH=~/.enterprise-rag/checkpoints.sqlite

# Optional Portkey routing (virtual keys default to rag and brag)
PORTKEY_API_KEY=
PORTKEY_PRIMARY_SLUG=rag
PORTKEY_FALLBACK_SLUG=brag
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Required | Used by guardrails and direct-Groq fallback, including when Portkey is enabled |
| `COHERE_API_KEY` | Optional | Preferred embedding provider; local sentence-transformers is the fallback |
| `QDRANT_CLUSTER_ENDPOINT` | Required | Qdrant URL. The application uses the fixed collection name `enterprise_grade_rag` |
| `QDRANT_API_KEY` | Required | Qdrant authentication |
| `LOGFIRE_TOKEN` | Optional | Enables Logfire reporting; the app can run without it |
| `LANGSMITH_TRACING` | Optional | Enables LangSmith tracing only when `LANGSMITH_API_KEY` is also set; default is off without a key |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing credential |
| `LANGSMITH_PROJECT` | Optional | LangSmith project name; defaults to `enterprise-grade-rag` |
| `LANGSMITH_ENDPOINT` | Optional | LangSmith API endpoint; defaults to `https://api.smith.langchain.com` |
| `BACKEND_URL` | Optional | Base URL for the API, used by the chat UI and evaluation dashboard; defaults to `http://localhost:8000` |
| `CHECKPOINT_DB_PATH` | Optional | SQLite file for graph checkpoints; defaults to `~/.enterprise-rag/checkpoints.sqlite` |
| `PORTKEY_API_KEY` | Optional | Enables Portkey routing for planner/responder; omit to call Groq directly |
| `PORTKEY_PRIMARY_SLUG` | Optional | Portkey primary virtual-key slug; defaults to `rag` |
| `PORTKEY_FALLBACK_SLUG` | Optional | Portkey fallback virtual-key slug; defaults to `brag` |
| `JUDGE_GROQ` | Optional | Separate Groq key for evaluation metrics; falls back to `GROQ_API_KEY` |

Do not commit `.env` or put credentials in source control. In GitHub Codespaces, store credentials as Codespaces secrets; they are exposed to the process as environment variables.

---

## 🚀 Getting Started

### 1. Clone & Install

Use Python 3.11 (the project declares `>=3.11,<3.12`). This project uses [`uv`](https://github.com/astral-sh/uv) for dependency and environment management. Run commands from the directory containing `pyproject.toml`.

```bash
git clone https://github.com/Paul-Ajibola/enterprise-grade-production-ready-rag-system.git
cd enterprise-grade-production-ready-rag-system

# Regenerate the lock after dependency changes, then install
uv python install 3.11
uv lock --python 3.11
uv sync --python 3.11

# ...or with pip
python -m pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials. In GitHub Codespaces, you can instead add the required credentials as Codespaces secrets and restart the Codespace so the environment receives them.

### 3. Ingest Your Knowledge Base

Place your source documents inside the `DATA/` directory (optionally organized into sub-folders, e.g. `DATA/kubernetes/`, `DATA/networking/` — each sub-folder becomes a `source_type` tag in Qdrant).

```bash
# Ingest everything under DATA/, auto-detecting sub-folders as source types
uv run python -m app.ingestion.processor DATA

# Ingest a specific folder as a single source type
uv run python -m app.ingestion.processor DATA/kubernetes kubernetes

# Destructive: delete and rebuild the Qdrant collection from scratch
uv run python -m app.ingestion.processor DATA --wipe
```

Supported file types: `.pdf`, `.html` / `.htm`, `.txt`, `.docx`, `.pptx`. The ingestion command creates the `enterprise_grade_rag` collection if needed. Add documents before starting the app; an empty or missing collection cannot serve retrieval queries. Keep the embedding provider consistent between ingestion and querying because Cohere and the local fallback produce vectors with different dimensions.

### 4. Start the FastAPI Backend

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` — Basic liveness response; it does not probe Groq, Qdrant, or the embedding provider
- `POST /query` — Main agent endpoint (`{ "q": "...", "thread_id": "..." }`)
- `POST /query/stream` — Server-sent events: token events followed by a final payload, or an error event
- `DELETE /threads/{thread_id}` — Deletes the saved conversation for a session
- `GET /graph` — Returns a PNG diagram of the compiled LangGraph workflow
- `GET /docs` — Interactive FastAPI API documentation

Both query endpoints require `q` and `thread_id`. A successful JSON response contains `question`, `answer`, `thought_process`, `status`, and `sources`. Each source includes its content, source filename, source type, and Qdrant score; the reranker score is included when reranking succeeds. `/query` returns HTTP 502 for backend processing errors; `/query/stream` emits an `error` event if processing fails after streaming begins.

Example non-streaming request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q":"How do Kubernetes jobs work?","thread_id":"demo-session"}'
```

### 5. Launch the Streamlit UI

```bash
uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

In Codespaces, open port `8501` from the **Ports** panel. Forwarded ports are private by default. Forward port `8000` too only if you want to inspect the API directly. The UI calls the backend at `BACKEND_URL` (default `http://localhost:8000`).

The UI will:
- Show a working status while the request runs, then display the agent's plan
- Stream answer tokens as the backend generates them
- Let you expand and inspect each retrieved source chunk
- Let you clear the current browser session and delete its saved backend thread via the sidebar

### 6. Run the Evaluation Dashboard

Launch the separate evaluation UI:

```bash
uv run streamlit run evals/streamlit_app.py --server.address 0.0.0.0 --server.port 8502
```

Forward port `8502` in Codespaces. The dashboard has three sections: live pipeline queries, guardrail checks, and RAG metrics. The backend must be running for the first two. Run the live pipeline before scoring RAG metrics; metric scoring requires `JUDGE_GROQ` or `GROQ_API_KEY` and also downloads a local sentence-transformers model for relevancy metrics. Live pipeline and guardrail case results are saved to `evals/results.json`. Metric score tables are held in the current Streamlit session and are not persisted there.

The six reported metrics are faithfulness, answer relevancy, context precision, context recall, answer correctness, and tool correctness. Context metrics use contexts returned by Qdrant (missing-context samples score `0`); they do not use the golden dataset's `relevant_contexts` as retrieved evidence. Tool correctness is inferred from the planner's reported thought process, not measured from instrumented tool calls.

---

## 🔍 Design Notes & Limitations

- **Dimension-safe embedding fallback**: If Cohere rate-limits mid-batch, the system discards partial results and re-embeds the whole file with the local fallback model. If that model does not match the collection dimension, ingestion stops with instructions to rebuild the collection using `--wipe`.
- **Collection dimension locking**: When re-running ingestion against an existing Qdrant collection (without `--wipe`), the pipeline reads the *actual* configured vector size from Qdrant rather than assuming it matches whichever embedding model happens to be active, preventing silent corruption.
- **Graceful reranker fallback**: If FlashRank reranking fails for any reason, the system falls back to returning the original top-N retrieved documents rather than failing the whole request.
- **Conversational memory boundary**: The planner explicitly separates "answerable from memory" vs. "needs fresh retrieval" queries, avoiding unnecessary vector search calls for greetings or follow-up questions about prior turns.
- **Optional tracing**: Logfire spans are added at key points in ingestion, API, retrieval, generation, and UI flows. They are not a substitute for application health checks or request-level error monitoring.

This is a development/demo application, not a hardened multi-user production service. The API has no authentication or authorization, all users share one Qdrant collection, and conversation checkpoints live in a local SQLite file. Keep forwarded Codespaces ports private and do not expose the API publicly with sensitive data. Scale-out deployments need shared checkpoint storage and tenant access controls.

Ingestion parses PDF text but does not OCR scanned/image-only pages. The evaluation dashboard is an LLM-as-judge workflow, not a deterministic test suite; its tool-correctness score is inferred from planner output as described above.


---

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a PR or an issue on GitHub.
