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