# RAG_Document_QA_MultiCollection

 Phase 1 Baseline System Design: 
 - Ingestion PDFs/DOCX, chunk, embed, store in Qdrant.
 - Answer with **OpenAI API**, or **Ollama** (local).
 -  Streamlit UI with configurable embeddings, answer with selected model, session history, and collection (folder) isolation.


---

## Project scope

### Features

- **Ingestion:** Extract text from PDF/DOCX → hierarchical chunking → embed (OpenAI Ada or BGE-M3) → upsert to Qdrant with full metadata.
- **Retrieval:** Semantic search over one or all collections; top-k chunks with score, source, and page range.
- **Answering:** Context-aware answers via OpenAI GPT or Ollama (Qwen, Llama) with citation rules.
- **UI:** Streamlit app — select embedding model, answer model, and collection; chat with session persistence; ingest via upload; view inventory and reports.

### Components

| Component        | Role |
|-----------------|------|
| **Qdrant**      | Vector store (one collection per "folder"). |
| **rag-api**     | Streamlit app + ingestion/retrieval logic; runs embeddings (OpenAI or BGE-M3 in-process). |
| **Ollama**      | Local LLM server (Qwen, Llama); optional. |
| **Scripts**     | CLI for ingest, retrieval, inventory, preview, numeric check, local answer. |

### Scripts (CLI)

| Script | Purpose |
|--------|--------|
| `scripts/index_text.py` | Ingest: extract → chunk → embed → upsert to Qdrant. |
| `scripts/report_inventory.py` | Build ingestion report from state logs → `state/reports/inventory.json`. |
| `scripts/preview_extract.py` | Preview extracted text (PDF pages or DOCX head). |
| `scripts/query_chunks.py` | Retrieve top-k chunks for a query (no LLM). |
| `scripts/check_numbers.py` | Numeric preservation check → `state/reports/numeric_issues.json`. |
| `scripts/export_sample_qdrant_payload.py` | Export one point’s payload as JSON (client sample artifact). |
| `scripts/azure_openai_env.py` | Shared Azure OpenAI client factory (`set_env`) for Streamlit. |
| `scripts/answer_local.py` | **CLI only:** retrieve chunks → call Ollama → answer with citations. |
| `scripts/run_script_tests_in_order.py` | Run all scripts in order and append a test report. |

### RAG entry points (canonical)

| Path | Role |
|------|------|
| **`frontend/app.py`** | **Primary UI:** `retrieve_context_with_folder()` (dense, leaf chunks) + `generate_answer()`. |
| **`scripts/answer_local.py`** | Offline / Ollama CLI; not used by Streamlit chat. |

### Data model (chunk payload — canonical)

Values are written by **`scripts/index_text.py`** (`embed_and_upsert`). Retrieval and inventory should treat this as the source of truth.

| Field | Type (logical) | Description |
|-------|----------------|-------------|
| `collection` | string | Qdrant collection id (includes `QDRANT_COLLECTION_PREFIX` when set). |
| `source_path` | string | Logical path `"{collection}/{file_name}"` or disk path after CLI ingest. |
| `doc_id` | string | MD5 of stable key (path or `collection\|name\|size` for uploads). |
| `file_name` | string | Original filename (preserved for Streamlit uploads). |
| `page_number`, `page_start`, `page_end` | int | Page alignment for the chunk. |
| `chunk_index` | int | Index within the document. |
| `chunk_id` | string | `{doc_id}:{page_start}:{chunk_index}`. |
| `text` | string | Chunk text. |
| `is_leaf` | bool | **true** for child chunks used in retrieval filters. |
| `parent_id`, `parent_text` | optional | Hierarchical chunking (LlamaIndex). |
| `chunk_total` | int | Total chunks for the document. |
| `ingest_source_path` | string | Same as logical `source_path` prefix for uploads. |

**Sample payload:** with Qdrant running, `python scripts/export_sample_qdrant_payload.py --collection YOUR_COLLECTION --out sample_payload.json`

**Scanned PDFs:** extraction yielding &lt; `MIN_EXTRACTED_TEXT_CHARS` (default 40) is recorded as `status: skipped`, `ingest_flag: zero_text_pdf` in ingestion logs.

---

## Prerequisites

- **Docker & Docker Compose** (for recommended setup).
- **Python 3.11+** (if running locally).
- **OpenAI.com API** (Path C — recommended): `OPENAI_API_KEY` from [platform.openai.com](https://platform.openai.com) — for embeddings and chat.
- **Azure OpenAI** (optional alternative): endpoint, API key, deployment — for embeddings and/or chat.
- **Ollama** (optional): local answer models; pull `qwen2.5:7b-instruct` and `llama3.1:8b`.

---

## Setup

### 1. Clone and enter project

```bash
git clone https://github.com/asifejjaz/RAG_Document_QA_MultiCollection.git
cd RAG_Document_QA_MultiCollection
```

### 2. Environment file

Copy **[`.env.example`](.env.example)** to `.env` in the project root (or create `.env` manually). Required and common variables:



```env
# OpenAI.com API (recommended — Path C)
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Embedding: openai_small (1536) | openai_ada (1536) | azure_ada (1536) | bge_m3 (1024, local)
EMBED_MODEL=openai_small

# Default LLM for answering: openai | ollama_qwen2.5 | ollama_llama3.1
DEFAULT_LLM=openai

# Chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# Vector DB (use host when running scripts on host)
VECTOR_DB_URL=http://localhost:6333
QDRANT_COLLECTION_PREFIX=rag_

# Local LLM (Ollama) — optional
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL_PRIMARY=qwen2.5:7b-instruct
LOCAL_LLM_MODEL_SECONDARY=llama3.1:8b-instruct

# Paths
DATA_ROOT=./data
STATE_ROOT=./state
```



### 3. Docker setup

```bash
# Build and start all services
docker compose -f docker_compose.yaml build
docker compose -f docker_compose.yaml up -d

# Check
docker compose -f docker_compose.yaml ps
```

Services:

- **qdrant** — port 6333 (and 6334 gRPC); volume `qdrant_data`.
- **rag-api** — Streamlit on 8501, app code and `./data` mounted; uses `rag_state` and `.env`.
- **ollama** — port 11434; volume `ollama_models`.

Pull Ollama models (If you want to test locally):

```bash
docker exec -it rag-ollama ollama pull qwen2.5:7b-instruct
docker exec -it rag-ollama ollama pull llama3.1:8b
```

### 4. Local setup (Without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Ensure **Qdrant** is running (e.g. only start the qdrant service with Docker, or run Qdrant elsewhere). Set in `.env`:

- `QDRANT_URL=http://localhost:6333`
- `OLLAMA_BASE_URL=http://localhost:11434` if using Ollama on the host.

---

## Run

### Run the Streamlit app

**With Docker (after `up -d`):**

- Open **http://localhost:8501**.

**Locally:**

```bash
# From project root; ensure PYTHONPATH includes project root
set PYTHONPATH=%CD%   # Windows
# export PYTHONPATH=$(pwd)  # Linux/macOS
streamlit run frontend/app.py
```

Then open http://localhost:8501. In the sidebar: choose **Embedding model**, **Answer model**, and **Filter by collection** (or “All Folders”). Ask questions; you can also upload files to ingest into a collection.

### Run local scripts (ingest, retrieve, report)

From the **project root**, with `PYTHONPATH` set to the project root and `.env` in place:

```bash
# Ingest one collection (example: hydrogen_books under ./data)
python scripts/index_text.py --data-root ./data --collection hydrogen_books

# Ingest all collections under data root
python scripts/index_text.py --data-root ./data --all-collections

# Inventory report (writes state/reports/inventory.json)
python scripts/report_inventory.py --state ./state

# Preview PDF/DOCX
python scripts/preview_extract.py --file "./data/hydrogen_books/sample.pdf" --pages 1-2
python scripts/preview_extract.py --file "./data/biofuels_books/doc.docx" --head 2000

# Retrieve top-k chunks (no LLM)
python scripts/query_chunks.py --q "electrolyzer efficiency" --collection hydrogen_books --topk 8

# Check numbers (writes state/reports/numeric_issues.json)
python scripts/check_numbers.py --collection hydrogen_books --state ./state

# Answer with Ollama (retrieve + LLM)
python scripts/answer_local.py --q "What is hydrogen bunkering?" --collection hydrogen_books --model qwen2.5:7b-instruct
python scripts/answer_local.py --q "What is hydrogen bunkering?" --collection hydrogen_books --model llama3.1:8b
```

When using Docker for Qdrant/Ollama but running scripts on the host, use `QDRANT_URL=http://localhost:6333` and `OLLAMA_BASE_URL=http://localhost:11434` in `.env`.

---

## Test

### Automated test run (all scripts in order)

From project root, with Qdrant (and optionally Ollama) running:

```bash
set PYTHONPATH=%CD%
set QDRANT_URL=http://localhost:6333
set OLLAMA_BASE_URL=http://localhost:11434
set STATE_ROOT=state

python scripts/run_script_tests_in_order.py --data-root ./data --collection hydrogen_books --state state
```

- **Without Ollama:** add `--skip-ollama` to skip `answer_local.py`.
- **Without re-ingesting:** add `--skip-ingestion` to use existing collection.

Results are appended to `scripts/test_report_ollama.txt` with a **FINAL TEST REPORT** (pass/fail per script). Exit code 0 = all passed.

Test order:

1. Ingestion (`index_text.py`)
2. Retrieval: `query_chunks.py`, then `answer_local.py` (unless `--skip-ollama`)
3. Inventory (`report_inventory.py`)
4. Preview (`preview_extract.py` on first PDF/DOCX under data/collection)
5. Check numbers (`check_numbers.py`)

### Manual checks

- **Qdrant:** `curl http://localhost:6333/health` (or from host if Qdrant in Docker).
- **Ollama:** `curl http://localhost:11434/api/tags` or `docker exec rag-ollama ollama list`.
- **Streamlit:** Open http://localhost:8501, pick a collection, ask a question and confirm an answer or “I couldn’t find relevant information.”

### Sample data

Place at least one PDF or DOCX under `data/<collection_name>/` (e.g. `data/hydrogen_books/sample.pdf`) before running ingestion or the full test suite.

---

## Configuration reference

| Variable | Description |
|----------|-------------|
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking for ingestion. |
| `EMBED_MODEL` | `openai_small` (default), `openai_ada`, `azure_ada`, or `bge_m3`. Must match collections you query (dimension: 1536 vs 1024). |
| `VECTOR_DB_URL` / `QDRANT_URL` | Qdrant HTTP URL. |
| `QDRANT_COLLECTION_PREFIX` | Prefix for Qdrant collection names (e.g. `rag_` → `rag_hydrogen_books`). |
| `OLLAMA_BASE_URL` | Ollama API URL. |
| `DEFAULT_LLM` | `openai` (default), `azure`, `ollama_qwen2.5`, or `ollama_llama3.1`. |
| `OPENAI_API_KEY` | OpenAI.com secret key (required for OpenAI path). |
| `OPEN_AI_KEY` | Alternative variable name for the OpenAI.com API key. |
| `OPENAI_CHAT_MODEL` | Chat model id, e.g. `gpt-4o-mini`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, e.g. `text-embedding-3-small` (used when `EMBED_MODEL=openai_small`). |
| `STATE_ROOT` | Directory for ingestion logs and reports (e.g. `state` or `/state` in Docker). |
| `DATA_ROOT` | Root for collection folders (e.g. `data` or `/data` in Docker). |

Embedding and LLM options are defined in `scripts/embed_config.py`; you can add or change models there.

---

## Project structure (main paths)

```
Eva_Rsearch_AI/
├── .env                    # Config (not committed)
├── docker_compose.yaml     # qdrant, rag-api, ollama
├── Dockerfile              # rag-api image
├── requirements.txt
├── frontend/
│   └── app.py              # Streamlit app
├── scripts/
│   ├── index_text.py       # Ingest
│   ├── embed_config.py    # Embedding & LLM registry
│   ├── report_inventory.py
│   ├── preview_extract.py
│   ├── query_chunks.py
│   ├── check_numbers.py
│   ├── answer_local.py
│   └── run_script_tests_in_order.py
├── data/                   # Corpus (e.g. data/hydrogen_books/*.pdf)
├── state/                  # Ingestion logs, state/reports/
├── sessions/               # Streamlit session data
└── docs/
    ├── User_Guide.md          # Step-by-step setup (Git, Python, VS Code, Docker, .env)
    └── M1_Requirements.md
```

---

## One folder = one Qdrant collection

Each “folder” in the UI is one Qdrant collection (e.g. `rag_hydrogen_books`). Benefits: search isolation, smaller retriever/LLM context, and clear UX. **“All Folders”** searches every collection and merges by score. Choose a single folder for focused answers; use All Folders to search everything. See `docs/M1_Requirements.md` for full requirements and status.
