# M1 Requirements Complete Implementation

This document responds to the M1 requirements: what was built, where to find it, and how to test it.



## A) Docker Compose: Services + Volumes

### 1. Service: qdrant
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Image: qdrant/qdrant:latest | Service `qdrant` in compose uses that image. | `docker_compose.yaml` (lines 4–22) | `docker compose -f docker_compose.yaml up -d qdrant` then `curl http://localhost:6333/health` | Volume: qdrant_data:/qdrant/storage | Volume mounted under `/qdrant/storage`. Named volume `qdrant_data` with exact name. | `docker_compose.yaml` lines 11–12, 72–73 | After ingest, restart container; data persists. |
### 2. Service: rag-api
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Build: repo Dockerfile | `build: context: ., dockerfile: Dockerfile`. | `docker_compose.yaml` lines 26–29 | `docker compose -f docker_compose.yaml build rag-api` | Mount corpus read-only: ./data:/data:ro | Volume `./data:/data:ro`. | `docker_compose.yaml` line 36 | Place a PDF in `./data/hydrogen_books/`, run ingest from inside container; file is read. | Mount state/output: rag_state:/state | Volume `rag_state:/state`. | `docker_compose.yaml` line 37 | Run report_inventory; output appears under `/state` (e.g. `/state/reports/inventory.json`). | Env: DATA_ROOT=/data, STATE_ROOT=/state, VECTOR_DB_URL=http://qdrant:6333 | Set in `environment` (and via `env_file: .env`). | `docker_compose.yaml` lines 30, 44–48 | Scripts use these; ingestion and reports use /data and /state. |
### 3. Service: ollama
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Image: ollama/ollama:latest | Service `ollama` uses that image. | `docker_compose.yaml` lines 56–69 | `docker compose -f docker_compose.yaml up -d ollama` | Volume: ollama_models:/root/.ollama | Mounted; named volume `ollama_models` with exact name. | `docker_compose.yaml` lines 63–64, 76–77 | Pull a model, restart container; model persists. | Port: 11434:11434 | Port mapping defined. | `docker_compose.yaml` lines 60–61 | `curl http://localhost:11434/api/tags` from host. | Env: OLLAMA_HOST=0.0.0.0 | Set in environment. | `docker_compose.yaml` line 65 | Ollama accepts connections from host/network. |
### Volumes (exact names)
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| qdrant_data | Named volume `qdrant_data`. | `docker_compose.yaml` lines 71–73 | `docker volume ls \| findstr qdrant_data` | rag_state | Named volume `rag_state`. | `docker_compose.yaml` lines 73–74 | Same. | ollama_models | Named volume `ollama_models`. | `docker_compose.yaml` lines 75–77 | Same. |
---

## B) Scripts (M1 deliverables)

### 1) scripts/index_text.py
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Action: ingest → extract text → chunk → embed → upsert to Qdrant | PDF/DOCX extraction (PyMuPDF, python-docx), recursive chunking (LangChain), configurable embeddings (Azure or BGE-M3), upsert to Qdrant with required payload. | `scripts/index_text.py` | From repo root (with PYTHONPATH set): `python scripts/index_text.py --data-root ./data --collection hydrogen_books` | CLI: --data-root /data --collection hydrogen_books | Args: `--data-root`, `--collection`. | `scripts/index_text.py` (argparse, ~lines 839–847) | `python scripts/index_text.py --data-root /data --collection hydrogen_books` (e.g. inside container with /data mounted)  CLI: --data-root /data --all-collections | Arg: `--all-collections`. | Same file. | `python scripts/index_text.py --data-root /data --all-collections` |
### 2) scripts/report_inventory.py
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Action: produce ingestion report (docs/chunks/skips) | Scans STATE_ROOT for ingestion_*.json logs; aggregates files, chunks, success/fail/skip. | `scripts/report_inventory.py` | `python scripts/report_inventory.py --state /state` (or `./state` on host)  Output: /state/reports/inventory.json  Writes to `<state>/reports/inventory.json`. | `scripts/report_inventory.py` (~lines 305–306, 336) | Check file after run. | CLI: python scripts/report_inventory.py --state /state | `--state` required. | Same file. | As above. |
### 3) scripts/preview_extract.py
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Action: show extracted text for inspection (page-aware for PDF) | Thin CLI calling `preview.preview_file()`; PDF uses page ranges, DOCX uses --head/--all. | `scripts/preview_extract.py`; logic in `scripts/preview.py` | `python scripts/preview_extract.py --file "<path>" --pages 1-2` or `--head 2000` for DOCX | CLI: --file "/data/hydrogen_books/book1.pdf" --pages 1-2 | `--file`, `--pages`. | `scripts/preview_extract.py` (e.g. lines 40–41) | With real path.  CLI: --file "/data/biofuels_books/doc.docx" --head 2000 | `--head` for DOCX. | Same. | With real path. |
### 4) scripts/query_chunks.py
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Action: retrieve top-k chunks (no answering) | Embeds query, searches Qdrant, prints hits. | `scripts/query_chunks.py` | `python scripts/query_chunks.py --q "electrolyzer efficiency" --collection hydrogen_books --topk 8`  CLI: --q "..." --collection hydrogen_books --topk 8  Args implemented. | Same file. | As above. | Print per hit: score, chunk_id, source_path, page_start-page_end, text_preview | All five printed for each hit. | `scripts/query_chunks.py` (lines 54–68) | Inspect stdout. |
### 5) scripts/check_numbers.py
| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Action: numeric preservation check (flags missing numeric tokens) | Scrolls collection, checks chunk text for numeric patterns, writes issues. | `scripts/check_numbers.py` | `python scripts/check_numbers.py --collection hydrogen_books --state /state`  CLI: python scripts/check_numbers.py --collection hydrogen_books | We also support `--state` for output path (default /state). | Same file. | As above. | Output: /state/reports/numeric_issues.json  Writes to `<state>/reports/numeric_issues.json`. | `scripts/check_numbers.py` (lines 103–105) | Check file after run. |
---

## C) Config: Single .env + Variables

| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| Create .env, read in docker-compose  `env_file: .env` on rag-api. Scripts load via python-dotenv. | `.env`; `docker_compose.yaml` line 30 | Set a var, run script or container; value used. | CHUNK_SIZE=1200 | In .env; read in index_text. | `.env` line 11 | Change and re-ingest; chunk sizes reflect. | CHUNK_OVERLAP=150 | In .env; read in index_text. | `.env` line 12 | Same. | EMBED_MODEL=bge-m3 | We support `bge_m3` (underscore). .env currently has `EMBED_MODEL=azure_ada`; set to `bge_m3` for BGE-M3. Client name "bge-m3" maps to our id `bge_m3`. | `.env` line 15; `scripts/embed_config.py` | Set `EMBED_MODEL=bge_m3`, run ingest. | VECTOR_DB=qdrant | In .env. | `.env` line 18 | Used for config. | QDRANT_COLLECTION_PREFIX=rag_ | In .env; used for collection naming. | `.env` line 20 | Collections created with prefix. | LOCAL_LLM_PROVIDER=ollama | In .env. | `.env` line 23 | answer_local uses Ollama. | LOCAL_LLM_MODEL_PRIMARY=qwen2.5:7b-instruct | In .env. | `.env` line 24 | Used by UI/default. | LOCAL_LLM_MODEL_SECONDARY=llama3.1:8b-instruct | In .env. Ollama tag for second model is `llama3.1:8b` (no -instruct); CLI accepts either where we pass model name. | `.env` line 25 | Use `--model llama3.1:8b` for answer_local (see D). |
---

## D) Models: Install Exact Names (Local LLM + Embeddings)

| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| ollama pull qwen2.5:7b-instruct | Run inside ollama container or host. | N/A (Ollama CLI) | `docker exec rag-ollama ollama pull qwen2.5:7b-instruct` | ollama pull llama3.1:8b-instruct | Ollama registry uses tag `llama3.1:8b` (no "-instruct"). We pulled `llama3.1:8b`; use that tag in CLI. | N/A | `docker exec rag-ollama ollama pull llama3.1:8b` then `--model llama3.1:8b` in answer_local | Embedding: bge-m3 (sentence-transformers / HF) | Option in embed_config; id `bge_m3`, model BAAI/bge-m3. Can run in rag-api container. | `scripts/embed_config.py`; set `EMBED_MODEL=bge_m3` in .env | Set EMBED_MODEL=bge_m3, run index_text. |
---

## E) Data Model: Exact Metadata Keys on Every Chunk

| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| collection | Set in payload in index_text. | `scripts/index_text.py` (e.g. payload dict ~553–561) | query_chunks or scroll; inspect payload. | source_path | Set from file path. | Same. | Same. | doc_id | Generated per document. | Same. | Same. | page_start | Set per chunk. | Same. | Same. | page_end | Set per chunk. | Same. | Same. | chunk_index | Index within page. | Same. | Same. | chunk_id (stable: doc_id:page_start:chunk_index) | Format exactly that. | Same (e.g. chunk_id = f"{doc_id}:{page_start}:{idx}"). | Same. | text | Chunk text. | Same. | Same. | No optional – all keys for all chunks | Payload built with all keys for every point. | Same. | Scroll collection; every point has all keys. |
---

## F) Local Answer Generation (answer_local.py)

| Requirement | Implementation | Where to find | How to test |
|-------------|-----------------|---------------|--------------|
| File: scripts/answer_local.py | Implemented. | `scripts/answer_local.py` | Run CLI below. | Action: retrieve chunks → call Ollama → return answer with citations | Embeds query, searches Qdrant, builds context, POSTs to Ollama /api/chat, prints message content. | Same file. | As below. | CLI: --q "..." --collection hydrogen_books --model qwen2.5:7b-instruct | Implemented. | Same file. | `python scripts/answer_local.py --q "What is hydrogen bunkering?" --collection hydrogen_books --model qwen2.5:7b-instruct` | CLI: --model llama3.1:8b-instruct | Use `--model llama3.1:8b` (Ollama tag). | Same. | `python scripts/answer_local.py --q "..." --collection hydrogen_books --model llama3.1:8b` | Hard rule in prompt (exact): "Use only the provided context. If not found, say NOT FOUND. Cite as (source_path p.page_start–page_end)." | PROMPT_RULE in code matches that text; injected into system message. | `scripts/answer_local.py` lines 26–29, 70 | Run and check model follows rule. |
---

## How to Run Full Test Suite

From project root (with Qdrant and Ollama running).

**PowerShell:**
```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:QDRANT_URL = "http://localhost:6333"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:STATE_ROOT = "state"
python scripts/run_script_tests_in_order.py --data-root ./data --collection hydrogen_books --state state
```

**CMD:**
```cmd
set PYTHONPATH=%CD%
set QDRANT_URL=http://localhost:6333
set OLLAMA_BASE_URL=http://localhost:11434
set STATE_ROOT=state
python scripts/run_script_tests_in_order.py --data-root ./data --collection hydrogen_books --state state
```

To skip Ollama (answer_local): add `--skip-ollama`.

Results are appended to `scripts/test_report_ollama.txt` with a **FINAL TEST REPORT** (pass/fail per script). A short summary report is in `scripts/TEST_RUN_REPORT.txt`.


---

## Working commands (How to test)

Run all commands from the **project root**. Set env once per session, then run the script.

**PowerShell (set env once):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:QDRANT_URL = "http://localhost:6333"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:STATE_ROOT = "state"
```

**CMD (set env once):**
```cmd
set PYTHONPATH=%CD%
set QDRANT_URL=http://localhost:6333
set OLLAMA_BASE_URL=http://localhost:11434
set STATE_ROOT=state
```

---

| What to test | PowerShell | CMD |
|--------------|-------------|-----|
| **1. index_text.py** (ingest) | `python scripts/index_text.py --data-root ./data --collection hydrogen_books` | Same, after env above. |
| **2. query_chunks.py** (retrieve chunks) | `python scripts/query_chunks.py --q "electrolyzer efficiency" --collection hydrogen_books --topk 5` | Same. |
| **3. answer_local.py** (Ollama answer) | `python scripts/answer_local.py --q "What is hydrogen bunkering?" --collection hydrogen_books --model qwen2.5:7b-instruct` | Same. For llama3.1: `--model llama3.1:8b` |
| **4. report_inventory.py** | `python scripts/report_inventory.py --state state` | Same. |
| **5. preview_extract.py** (PDF) | `python scripts/preview_extract.py --file "data\hydrogen_books\test_doc.pdf" --pages 1` | Same (use backslash in path). |
| **6. check_numbers.py** | `python scripts/check_numbers.py --collection hydrogen_books --state state` | Same. |
| **Docker: Qdrant** | `docker compose -f docker_compose.yaml up -d qdrant` then `curl http://localhost:6333/health` | Same. |
| **Docker: Ollama** | `docker compose -f docker_compose.yaml up -d ollama` then `curl http://localhost:11434/api/tags` | Same. |

---

---

# Comprehensive Project Report

This section summarizes the **Eva Research AI** project: scope, architecture, codebase, requirements compliance, and how it aligns with the README and M1 requirements.

---

## 1. Executive Summary

**Eva Research AI** is a RAG-based document Q&A system that ingests PDF/DOCX documents, chunks and embeds them, stores vectors in Qdrant, and answers questions using **Azure OpenAI (GPT)** or **Ollama** (Qwen, Llama). A Streamlit UI provides configurable embeddings, answer models, session history, and collection (folder) isolation. All M1 deliverables (Docker Compose services, five CLI scripts, single .env config, data model, and local answer generation) are implemented and testable via the commands in this document.

---

## 2. Project Overview

| Item | Description |
|------|--------------|
| **Name** | Eva Research AI |
| **Purpose** | RAG-based document Q&A: ingest, chunk, embed, store, retrieve, and answer with citations. |
| **Deployment** | Docker Compose (qdrant, rag-api, ollama) or local Python with Qdrant/Ollama. |
| **Key docs** | `README.md` (setup, run, test), `docs/M1_Requirements.md` (this document). |

### Features (from README)

- **Ingestion:** PDF/DOCX → extract text → hierarchical chunking (LangChain) → embed (Azure Ada or BGE-M3) → upsert to Qdrant with full metadata.
- **Retrieval:** Semantic search over one or all collections; top-k chunks with score, source, page range.
- **Answering:** Context-aware answers via Azure GPT or Ollama (Qwen, Llama) with citation rules.
- **UI:** Streamlit — select embedding model, answer model, collection; chat with session persistence; ingest via upload; view inventory and reports.

---

## 3. Architecture

### 3.1 Services (Docker Compose)

| Service | Image / build | Ports | Role |
|---------|----------------|-------|------|
| **qdrant** | `qdrant/qdrant:latest` | 6333 (HTTP), 6334 (gRPC) | Vector store; one collection per “folder”. |
| **rag-api** | Build from repo `Dockerfile` | 8501 (Streamlit), 8000 (optional) | Streamlit app + ingestion/retrieval; runs embeddings in-process. |
| **ollama** | `ollama/ollama:latest` | 11434 | Local LLM server (Qwen, Llama); optional. |

### 3.2 Volumes

- **qdrant_data** → `/qdrant/storage` (persistent vectors).
- **rag_state** → `/state` (ingestion logs, reports: `inventory.json`, `numeric_issues.json`).
- **ollama_models** → `/root/.ollama` (pulled models).

Bind mounts: `./data` → `/data` (read-only corpus), `./sessions` → session persistence for Streamlit.

### 3.3 Data Flow

1. **Ingest:** `data/<collection>/` (PDF/DOCX) → extract → hierarchical chunk → embed (Azure or BGE-M3) → upsert to Qdrant collection `rag_<collection>`; ingestion log under `state/`.
2. **Retrieve:** Query → embed → Qdrant search → top-k chunks with payload (source_path, page_start, page_end, text, etc.).
3. **Answer:** Retrieved context + question → Azure OpenAI chat or Ollama `/api/chat` → response with citation rule (source_path, page range).
4. **UI:** User picks embedding, LLM, and collection; asks questions; optional file upload → ingest into chosen collection; sessions and history via `SessionManager`.

### 3.4 Technology Stack

- **Language:** Python 3.11+.
- **Vector DB:** Qdrant (HTTP client).
- **Embeddings:** Azure OpenAI (text-embedding-ada-002, 1536 dim) or BGE-M3 (sentence-transformers, 1024 dim) via `embed_config.py`.
- **Chunking:** LangChain-style recursive/hierarchical (LlamaIndex `HierarchicalNodeParser`) in `index_text.py`.
- **Document parsing:** PyMuPDF (PDF), python-docx / docx2txt (DOCX) in `index_text.py` and `preview.py`.
- **LLM:** Azure OpenAI (chat completions) or Ollama (REST `/api/chat`).
- **UI:** Streamlit; session persistence via `sessions/sessionManager.py`.
- **Config:** Single `.env`; `python-dotenv`; Docker `env_file` for rag-api.

---

## 4. Codebase Structure

```
Eva_Rsearch_AI/
├── .env                          # Config (not committed)
├── docker_compose.yaml           # qdrant, rag-api, ollama; volumes; network
├── Dockerfile                    # Python 3.11 slim; requirements; Streamlit
├── requirements.txt
├── frontend/
│   └── app.py                    # Streamlit app: embeddings/LLM/collection selectors,
│                                  # retrieve + generate_answer (Azure/Ollama), sessions, upload
├── scripts/
│   ├── index_text.py             # Ingest: extract, chunk, embed, upsert; Qdrant setup; Config
│   ├── embed_config.py           # Embedding & LLM registry (Azure, BGE-M3, Ollama models)
│   ├── report_inventory.py       # Aggregate ingestion_*.json → state/reports/inventory.json
│   ├── preview_extract.py         # CLI wrapper for preview.py (PDF/DOCX text preview)
│   ├── preview.py                # PDF/DOCX extraction and page/head preview logic
│   ├── query_chunks.py           # Embed query, search Qdrant, print top-k hits
│   ├── check_numbers.py          # Scroll collection, detect numeric issues → numeric_issues.json
│   ├── answer_local.py           # Retrieve chunks → Ollama /api/chat → answer with citations
│   ├── run_script_tests_in_order.py  # Run all scripts in sequence; append FINAL TEST REPORT
│   └── rg_pipeline.py            # retrieve_context (hybrid RRF), RAGRetriever, AutoGen agent (optional)
├── data/                         # Corpus: e.g. data/hydrogen_books/*.pdf
├── state/                        # Ingestion logs, state/reports/
├── sessions/
│   └── sessionManager.py         # Session create/load/list; conversation history; JSON persistence
└── docs/
    └── M1_Requirements.md       # This document (requirements + working commands + report)
```

---

## 5. Requirements Compliance (M1)

| M1 section | Delivered | Location / notes |
|------------|-----------|------------------|
| **A) Docker Compose** | Yes | `docker_compose.yaml`: qdrant, rag-api, ollama; volumes qdrant_data, rag_state, ollama_models; env and mounts as specified. |
| **B) Scripts** | Yes | `index_text.py`, `report_inventory.py`, `preview_extract.py`, `query_chunks.py`, `check_numbers.py`; CLIs and behavior match requirements. |
| **C) Config** | Yes | Single `.env`; `env_file` in compose; scripts use `python-dotenv`; CHUNK_*, EMBED_MODEL, VECTOR_DB_*, QDRANT_*, LOCAL_LLM_*, OLLAMA_BASE_URL. |
| **D) Models** | Yes | Ollama: qwen2.5:7b-instruct, llama3.1:8b; Embedding: azure_ada, bge_m3 via `embed_config.py`. |
| **E) Data model** | Yes | Every chunk: collection, source_path, doc_id, page_start, page_end, chunk_index, chunk_id (doc_id:page_start:chunk_index), text; in `index_text.py` payload construction. |
| **F) answer_local.py** | Yes | Retrieve → build context → Ollama /api/chat; CLI --q, --collection, --model; prompt rule (NOT FOUND, cite source_path p.page_start–page_end). |

---

## 6. Key Components (Implementation Summary)

### 6.1 Ingestion (`scripts/index_text.py`)

- **Config:** `Config` class reads QDRANT_URL (prefer over VECTOR_DB_URL for host runs), CHUNK_SIZE, CHUNK_OVERLAP, STATE_DIR, embedding batch/delay.
- **Extraction:** PDF via PyMuPDF, DOCX via Docx2txtLoader; page-level text.
- **Chunking:** `HierarchicalNodeParser` (parent/child); chunk size and overlap from env.
- **Embedding:** `embed_config.get_embeddings_model()` → Azure or BGE-M3; batch embed with retry and rate-limit handling.
- **Qdrant:** `get_qdrant_client()`, `setup_collection()` (vector size from embedding dimension); payload includes all required keys; upsert in batches; ingestion log written under STATE_DIR.
- **CLI:** `--data-root`, `--collection` or `--all-collections`.

### 6.2 Retrieval and Answer

- **query_chunks.py:** Embeds query, searches Qdrant, prints score, chunk_id, source_path, page_start–page_end, text_preview.
- **answer_local.py:** Same retrieval; builds context string; POST to OLLAMA_BASE_URL/api/chat with system/user messages; prompt rule for citations and NOT FOUND.
- **UI (app.py):** `retrieve_context_with_folder()` (single or all collections); `generate_answer()` branches on `embed_config.is_ollama(model_id)` → Ollama or Azure chat completions.

### 6.3 Config and Models (`scripts/embed_config.py`)

- **Embedding registry:** azure_ada (1536), bge_m3 (1024); `get_embeddings_model()`, `get_embedding_dimension()`.
- **LLM registry:** azure (deployment from env), ollama_qwen2.5, ollama_llama3.1; `get_llm_options()`, `is_ollama()`, `get_ollama_model_name()`.

### 6.4 Reports and Utilities

- **report_inventory.py:** Finds `state/ingestion_*.json`, aggregates files/chunks/success/fail/skip; writes `state/reports/inventory.json` and timestamped copy.
- **preview_extract.py / preview.py:** `--file`, `--pages` (PDF) or `--head`/`--all` (DOCX); prints extracted text for inspection.
- **check_numbers.py:** Scrolls collection, checks chunk text for numeric patterns; writes `state/reports/numeric_issues.json`.

### 6.5 Session Management (`sessions/sessionManager.py`)

- Create/load/list sessions; conversation history; JSON files under `sessions/`; used by Streamlit app for multi-session chat.

### 6.6 Frontend (`frontend/app.py`)

- Sidebar: embedding model, answer model (Azure / Ollama), collection filter (folder list from `get_collection_names_for_dimension()`).
- Chat: user message → retrieve context → build messages → `generate_answer()` (Azure or Ollama) → display; session and history via SessionManager.
- Optional file upload and ingest into selected collection; inventory/report references.
- Theme and layout (e.g. law-firm-style header) for consistent UX.

---

## 7. Testing and Validation

- **Automated:** `run_script_tests_in_order.py` runs ingestion, query_chunks, answer_local (unless `--skip-ollama`), report_inventory, preview_extract (first PDF/DOCX in data/collection), check_numbers; appends to `scripts/test_report_ollama.txt` with FINAL TEST REPORT (pass/fail per script).
- **Manual:** Qdrant `curl http://localhost:6333/health`; Ollama `curl http://localhost:11434/api/tags`; Streamlit at http://localhost:8501; run individual scripts with commands in the “Working commands” table above.
- **Host vs Docker:** On host, set `QDRANT_URL=http://localhost:6333` and `OLLAMA_BASE_URL=http://localhost:11434` so scripts target localhost; connection-refused handling in `index_text.py` suggests starting Qdrant and overriding URL when needed.

---

## 8. Deployment and Run Modes

| Mode | When to use | Notes |
|------|----------------|------|
| **Docker Compose** | Full stack (Qdrant + Streamlit + Ollama) | `docker compose -f docker_compose.yaml up -d`; use VECTOR_DB_URL=http://qdrant:6333, OLLAMA_BASE_URL=http://ollama:11434 in .env. |
| **Host scripts** | Run CLIs on Windows/Linux with Qdrant/Ollama elsewhere | Set PYTHONPATH, QDRANT_URL, OLLAMA_BASE_URL, STATE_ROOT; see Working commands. |
| **Streamlit local** | UI on host, Qdrant/Ollama on host or Docker | `streamlit run frontend/app.py`; ensure .env has correct URLs for host. |

---

## 9. References

- **README.md** — Project scope, prerequisites, setup (Docker + local), run (Streamlit + scripts), test, configuration reference, project structure.
- **docs/M1_Requirements.md** — This document: M1 requirements response, working commands (How to test), and comprehensive project report.
- **scripts/embed_config.py** — Single source of truth for embedding and LLM options; extend here for new models or providers.

---