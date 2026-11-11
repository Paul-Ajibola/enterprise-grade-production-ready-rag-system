# 🤖 Enterprise Agentic RAG

A production-grade, **agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **FastAPI**, and **Streamlit**. The system intelligently decides whether an incoming query needs fresh document retrieval or can be answered conversationally from memory, then synthesizes a grounded answer using an LLM — all while being fully observable via **Logfire** distributed tracing.

---

## ✨ Key Features

- **🧠 Agentic Planning** — A LangGraph-powered planner node inspects the full conversation history and decides, on every turn, whether the user's message is *conversational* (answerable from memory) or *technical* (requires a fresh knowledge-base search).
- **🔍 Hybrid Retrieval Pipeline** — Vector search against **Qdrant** followed by **semantic reranking** with FlashRank cross-encoders to surface only the most relevant chunks.
- **🔁 Resilient Embeddings** — Uses **Cohere** when its key and API are available, otherwise loads a local **sentence-transformers** model. Ingestion checks the active model dimension against the Qdrant collection and stops with rebuild instructions when they differ.
- **💾 Persistent Conversational Memory** — LangGraph's SQLite checkpointer stores per-session/thread conversation state across backend restarts. The UI's clear-memory action removes the saved thread.
- **📄 Multi-Format Ingestion** — Built-in text extraction for PDFs (pypdf with a pdfplumber retry), HTML, plain text, and Office documents (`.docx`, `.pptx`) via `unstructured`. Image-only PDF pages are not OCR-processed.
- **✂️ Smart Chunking** — Recursive character-based text splitting tuned for RAG retrieval quality.
- **📊 Full Observability** — Every step (planning, retrieval, reranking, generation, embedding) is wrapped in **Logfire** spans for end-to-end distributed tracing across the ingestion pipeline, the FastAPI backend, and the Streamlit UI.
- **💬 Streaming Chat UI** — A Streamlit front-end with live "agent thinking" status, streamed answer tokens, and collapsible source chunks labeled with their source files.
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

1. **`planner`** — Calls Groq (Llama 3.3 70B) with the conversation history to classify the query as `CONVERSATIONAL` or to produce a refined search query.
2. **`retriever`** *(conditionally invoked)* — Searches Qdrant for the top 15 candidate chunks, then reranks down to the top 5 using FlashRank's cross-encoder.
3. **`responder`** — Synthesizes the final answer using either the retrieved technical context or pure conversational memory, and appends the response to persistent thread state.

### Request Flow (Application)

```
┌────────────┐        POST /query        ┌─────────────┐
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
                 │  Groq LLM API   │     │  Qdrant Vector DB  │    │  FlashRank (ONNX) │
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
| LLM Inference | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` via `langchain-groq` |
| Optional LLM Gateway | Portkey routing and fallback when `PORTKEY_API_KEY` is configured; direct Groq otherwise |
| Vector Database | [Qdrant](https://qdrant.tech/) (cloud cluster) |
| Embeddings | [Cohere](https://cohere.com/) `embed-english-v3.0` (primary) → `sentence-transformers/all-mpnet-base-v2` (local fallback) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (local ONNX cross-encoder) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend UI | [Streamlit](https://streamlit.io/) |
| Observability | [Logfire](https://logfire.pydantic.dev/) |
| Document Parsing | `pypdf`, `pdfplumber`, `BeautifulSoup4`, `unstructured` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Package/Env Management | `uv` (`pyproject.toml` + `uv.lock`) |

---

## 📁 Project Structure

```
data/
├── .logfire/                     # Local Logfire cache/config
├── .venv/                        # Virtual environment (uv-managed)
├── app/
│   ├── agents/
│   │   ├── nodes/
│   │   │   ├── planner.py        # Decides CONVERSATIONAL vs. technical search
│   │   │   ├── retriever.py      # Vector search + reranking node
│   │   │   └── responder.py      # Final answer synthesis node
│   │   ├── graphs.py             # LangGraph StateGraph wiring + compilation
│   │   └── state.py              # Shared AgentState (TypedDict) definition
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
│   └── config.py                 # Centralized Settings (env-driven)
├── ui/
│   └── app.py                    # Streamlit chat interface
├── DATA/                          # Raw source documents for ingestion (input)
├── processed_data/                # JSON snapshots of parsed/chunked docs (output)
├── app/main.py                    # FastAPI app entrypoint (/, /query, /query/stream, /graph)
├── .env                           # Environment variables (not committed)
├── .gitignore
├── .python-version
├── commands.txt                   # Handy CLI command reference
├── pyproject.toml                 # Project metadata & dependencies
├── requirements.txt               # Pip-compatible dependency list
├── uv.lock                        # Locked dependency versions (uv)
└── README.md
```

> 💡 The tree above reflects the logical module layout used throughout the codebase (e.g. `app.agents.nodes.planner`, `app.services.retrieval.embeddings`, `app.ingestion.loaders.pdf`).

---

## ⚙️ Environment Variables

Create a `.env` file at the project root with the following keys:

```env
# LLM
GROQ_API_KEY=your_groq_api_key

# Embeddings
COHERE_API_KEY=your_cohere_api_key

# Vector Database
QDRANT_CLUSTER_ENDPOINT=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Observability
LOGFIRE_TOKEN=your_logfire_write_token

# Streamlit UI -> FastAPI backend
BACKEND_URL=http://localhost:8000
CHECKPOINT_DB_PATH=~/.enterprise-rag/checkpoints.sqlite
PORTKEY_API_KEY= # Optional; omit to call Groq directly
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Powers planner, responder, and guardrails when Portkey is not configured |
| `COHERE_API_KEY` | Optional | Preferred embedding provider; local sentence-transformers is the fallback |
| `QDRANT_CLUSTER_ENDPOINT` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud authentication |
| `LOGFIRE_TOKEN` | ⚠️ Recommended | Enables distributed tracing/observability |
| `BACKEND_URL` | ✅ (UI only) | Base URL for the FastAPI backend |
| `CHECKPOINT_DB_PATH` | Optional | SQLite file used to persist conversation threads |
| `PORTKEY_API_KEY` | Optional | Enables Portkey routing; direct Groq is used when it is omitted |
| `JUDGE_GROQ` | Optional | Separate Groq key for RAG metric evaluations |

---

## 🚀 Getting Started

### 1. Clone & Install

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency and environment management.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies with uv (uses pyproject.toml + uv.lock)
uv sync

# ...or with pip
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example above into a `.env` file at the project root and fill in your credentials.

### 3. Ingest Your Knowledge Base

Place your source documents inside the `DATA/` directory (optionally organized into sub-folders, e.g. `DATA/kubernetes/`, `DATA/networking/` — each sub-folder becomes a `source_type` tag in Qdrant).

```bash
# Ingest everything under DATA/, auto-detecting sub-folders as source types
python -m app.ingestion.processor DATA

# Ingest a specific folder as a single source type
python -m app.ingestion.processor DATA/kubernetes kubernetes

# Wipe and rebuild the Qdrant collection from scratch
python -m app.ingestion.processor DATA --wipe
```

Supported file types: `.pdf`, `.html` / `.htm`, `.txt`, `.docx`, `.pptx`.

### 4. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` — Health check
- `POST /query` — Main agent endpoint (`{ "q": "...", "thread_id": "..." }`)
- `POST /query/stream` — Server-sent event stream with answer tokens and final sources
- `DELETE /threads/{thread_id}` — Deletes the saved conversation for a session
- `GET /graph` — Returns a PNG diagram of the compiled LangGraph workflow

### 5. Launch the Streamlit UI

```bash
streamlit run ui/app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`) and start chatting. The UI will:
- Show live "Agent is thinking…" status with the agent's plan steps
- Stream answer tokens as the backend generates them
- Let you expand and inspect each retrieved source chunk
- Let you clear memory/session state via the sidebar

### 6. Run the Evaluation Dashboard

With the backend running, launch the separate evaluation UI:

```bash
streamlit run evals/streamlit_app.py
```

It can run the live query pipeline, guardrail checks, and RAG quality metrics. Metric scoring requires `JUDGE_GROQ` or `GROQ_API_KEY`.

---

## 🔍 Design Notes & Resilience Patterns

- **Dimension-safe embedding fallback**: If Cohere rate-limits mid-batch, the system discards partial results and re-embeds the whole file with the local fallback model. If that model does not match the collection dimension, ingestion stops with instructions to rebuild the collection using `--wipe`.
- **Collection dimension locking**: When re-running ingestion against an existing Qdrant collection (without `--wipe`), the pipeline reads the *actual* configured vector size from Qdrant rather than assuming it matches whichever embedding model happens to be active, preventing silent corruption.
- **Graceful reranker fallback**: If FlashRank reranking fails for any reason, the system falls back to returning the original top-N retrieved documents rather than failing the whole request.
- **Conversational memory boundary**: The planner explicitly separates "answerable from memory" vs. "needs fresh retrieval" queries, avoiding unnecessary vector search calls for greetings or follow-up questions about prior turns.
- **Full-stack tracing**: Logfire spans are nested consistently — from the Streamlit chat interaction, through the FastAPI request, into every LangGraph node, and down to individual embedding batches and reranking calls — enabling true end-to-end trace visualization.


---

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a PR or an issue on GitHub.
