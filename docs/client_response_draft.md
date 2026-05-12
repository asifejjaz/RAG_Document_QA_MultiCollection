# RAG Build – Phase 1 Review Response

---

## Summary response

Thank you for the detailed review. I have cross-checked the feedback against the repository and addressed all Phase 1 items below. The infrastructure (Docker, Qdrant, model registry, embedding factory, inventory/preview scripts) is solid; the changes focus on the core RAG pipeline and closing the README/code gaps.

Four-phase approach is shared by Client and documented in `docs/Requirements`. The Streamlit app supports Azure OpenAI, OpenAI.com API, and local Ollama models. All three providers are functional and tested.

---

## Point-by-point responses

### 1. HR / “Marius” system prompt (not document RAG)

**Client request:** The system prompt describes an HR assistant (“Marius”) and company-policy tone instead of a document-grounded research assistant. Answers should be tied to retrieved context with proper citations.

**Status:** **Fixed**

**Detail:** The legacy `scripts/rg_pipeline.py` (AutoGen HR agent) has been removed. The canonical Q&A path is `frontend/app.py` with the correct research-assistant system prompt:

> You are Eva, a research assistant. Use ONLY the provided context.  
> Rules:  
> - If the answer is not supported by the context, reply with I do not have enough information, do not make up an answer.  
> - Cite every factual claim with source file name and page,e.g. [filename, p. 12]  
> - Do not invent numbers, citations, or sources  
> - Be concise and professional

The full pipeline is: ingestion (`scripts/index_text.py`) → retrieval (`retrieve_context_with_folder`) → answer (`generate_answer`) within `frontend/app.py`. No HR persona remains in the active codebase.

---

### 2. Hybrid sparse retrieval is not effective

**Client request:** Dense + sparse “hybrid” retrieval was claimed, but sparse search uses exact-match on `text` and returns zero results; the system is effectively dense-only.

**Status:** **Resolved (legacy code removed)**

**Detail:** The hybrid/sparse path lived in `scripts/rg_pipeline.py`. That file has been deleted. The active Streamlit app (`frontend/app.py`) uses `retrieve_context_with_folder()` which performs **dense vector search only** against Qdrant collections, filtering on `is_leaf=True`. No sparse/BM25 component is present. Phase 1 is explicitly dense-only; a proper keyword or full-text path can be investigated in a later phase with tests before any hybrid claim is restored.

**Direct answer:** The hybrid sparse branch never returned non-empty sparse results in our testing; it was removed before Phase 1 completion.

---

### 3. Schema mismatch between README and what the pipeline writes

**Client request:** Documentation and retrieval code should agree on one canonical list of Qdrant payload fields. Provide a real sample chunk payload.

**Status:** **Resolved**

**Detail:** `README.md` now has a single **canonical payload** table aligned with `scripts/index_text.py` / `embed_and_upsert` (fields: `collection`, `source_path`, `doc_id`, `file_name`, `page_number`, `page_start`, `page_end`, `chunk_index`, `chunk_id`, `text`, `is_leaf`, `parent_id`, `parent_text`, `chunk_total`, `ingest_source_path`, etc.). The legacy `rg_pipeline.py` is gone, eliminating parallel schemas.

**Sample payload** — fins sample_payload.json exported from a live `rag_hydrogen_books` collection.


The file `sample_payload.json` is committed in the repository root for immediate inspection. You can also regenerate anytime with:

```bash
python scripts/export_sample_qdrant_payload.py --collection rag_hydrogen_books --out sample_payload.json
```

---

### 4. Two parallel RAG implementations (confusing ownership)

**Client request:** Multiple code paths (Streamlit, CLI, AutoGen) create confusion about which pipeline is "real" and what to maintain.

**Status:** **Resolved**

**Detail:** The legacy `scripts/rg_pipeline.py` (AutoGen path) has been removed. The codebase now has two clear, non-overlapping entry points:

- **Primary UI:** `frontend/app.py` — Streamlit chat with ingestion upload, collection management, and session persistence.
- **CLI for local models:** `scripts/answer_local.py` — standalone retrieve + Ollama answering (not used by the UI).
- **Shared support:** `scripts/index_text.py` (ingestion); `scripts/embed_config.py` (model registry); `scripts/azure_openai_env.py` (Azure clients).

Hardcoded `research_papers` defaults are gone; all collection names are explicit parameters.

---

### 5. Scanned PDFs / OCR not implemented

**Client request:** Image-only or scanned PDFs produce little or no text; the product should at least detect this and flag them (OCR as future work). The request specifically called out `preview.py`.

**Status:** **Resolved for Phase 1 detection & reporting**

**Detail:** Zero-text detection is implemented in `scripts/index_text.py:process_file()` (the ingestion path), not in `preview.py` (which is only for manual inspection):

```python
if total_extracted_text_length(pages) < MIN_EXTRACTED_TEXT_CHARS:  # default 40
    return {
        'file_name': display,
        'status': 'skipped',
        'error': 'No extractable text (likely scanned PDF or image-only)...',
        'ingest_flag': 'zero_text_pdf',
    }
```

Skipped files are recorded in ingestion logs (`state/ingestion_*.json`) and surfaced in the inventory report (`scripts/report_inventory.py` → `state/reports/inventory.json`). Full OCR (Tesseract or Azure Document Intelligence) remains a future milestone.

---

### 6. Numeric preservation check too lenient

**Client request:** `check_numbers.py` should compare numbers in chunks against the source document (by page), not only flag "long chunk with no digits."

**Status:** **Fixed**

**Detail:** `scripts/check_numbers.py` resolves the source file via `source_path` / `--data-root` + `file_name`, loads the relevant PDF page (PyMuPDF) or DOCX head, extracts numeric tokens (regex capturing decimals/percentages/comma groups), and compares **multiset differences**. The report (`state/reports/numeric_issues.json`) includes:

- `missing_in_chunk` — numbers on the source page absent from the chunk
- `extra_in_chunk` — numbers in the chunk absent from the source page
- `comparison_status` — `compared` / `source_file_not_found` / `source_page_empty` / etc.

This catches partial loss of numeric data, not just "no digits."

---

### 7. Embedding dimension vs collection mismatch unclear

**Client request:** Picking an embedding model that doesn't match an existing collection's vector size should raise a clear error at startup/selection time, not fail obscurely at query time.

**Status:** **Fixed**

**Detail:** Two safeguards:

1. `scripts/index_text.py:get_collection_vector_size()` exposes a collection's configured vector size.
2. `frontend/app.py:retrieve_context_with_folder()` checks the selected (or all) collections' vector size against the current embedding dimension. For a single collection selection, it calls `st.error()` and returns no context. For "All Folders," only dimension-matched collections are queried; mismatched ones are silently skipped (sidebar shows a warning when a single collection is selected and mismatched).

---

### 8. Original filename lost on Streamlit upload

**Client request:** Uploads use a temp path; metadata showed temp names instead of the user's original filename. Citations need recognizable filenames.

**Status:** **Fixed**

**Detail:** The upload flow calls `ingest_file_to_collection(..., logical_file_name=uploaded_file.name)`. `index_text.py:process_file()` and `generate_file_metadata()` accept `logical_file_name` and set:

- `file_name` = original upload name
- `ingest_source_path` = `{collection}/{original_name}`
- `doc_id` = `MD5(collection|original_name|size)` — stable across re-uploads of the same file

Citations surface `file_name` and `source_path`, so the original document name is preserved throughout.

---

### 9. No authentication on Streamlit

**Client request:** Port 8501 is exposed in docker-compose with no auth layer.

**Status:** **Future work (Phase 2+)**

**Detail:** Phase 1 is local/dev only. `deploy/README-CLIENT.md` explicitly states that before any internet-facing deployment we will add authentication (e.g. `streamlit-authenticator`, reverse-proxy basic auth, or VPN/network isolation).

---

### 10. Containers run as root; compose is dev-mode

**Client request:** Production expectations: non-root containers, no live source bind-mount, pinned image versions, restart policies.

**Status:** **Fixed**

**Detail:**

- **Dockerfile:** creates and runs as non-root user `raguser` (uid 10001); owns `/app` and `/state`.
- **docker-compose.prod.yaml:** no `.:/app` bind mount; uses named volumes (`rag_data_prod`, `rag_state_prod`, `rag_sessions_prod`); Qdrant pinned to `v1.12.5`; Ollama `latest` (with note to pin by digest in real prod); health-checked startup; restart policies.
- **docker_compose.yaml:** kept for local development with bind mounts and auto-reload.
- Helper deploy scripts: `deploy/start.sh` and `deploy/start.ps1`.

---

## Requested files and artifacts (status)

| Artifact | Status | Notes |
|----------|--------|-------|
| `scripts/index_text.py` | ✅ In repo | Complete ingestion pipeline — extract → chunk → embed → upsert |
| `frontend/app.py` | ✅ In repo | Streamlit UI — chat, upload, session management |
| Sample Qdrant chunk payload | ✅ In repo | `sample_payload.json` (exported from real `rag_hydrogen_books` collection). Also generatable with `python scripts/export_sample_qdrant_payload.py --collection <name> --out sample_payload.json` |
| Correctness test report | ✅ In repo | `docs/correctness_test_report.md` — populated test matrix (T1–T3) with pass/fail and retrieval checks. Methodology: `docs/system_test_methodology.md` |

---

## Direct answers to questions

**Q1. Which pipeline does the Streamlit app actually call?**  
`frontend/app.py` uses `retrieve_context_with_folder()` (dense Qdrant search on leaf chunks) and `generate_answer()` (Azure / OpenAI / Ollama chat). It does **not** use `answer_local.py` for the chat interface. `set_env()` comes from `scripts/azure_openai_env.py`. The legacy AutoGen path (`rg_pipeline.py`) has been removed from the active UI flow.

**Q2. Has hybrid dense + sparse returned non-empty sparse results?**  
No. The hybrid/sparse code path was in `scripts/rg_pipeline.py`, which was never the active UI path and has been removed. The active Streamlit path is dense-only vector retrieval. The hybrid branch never produced non-empty sparse results in our tests.

**Q3. Which schema does `index_text.py` write?**  
The canonical schema is documented in `README.md` under **Data model (chunk payload)**. `scripts/index_text.py:embed_and_upsert()` writes exactly those fields.

**Q4. Why was `research_papers` hardcoded?**  
It was a legacy single-collection default inside `rg_pipeline.py`. That file is removed; all collection names are now explicit parameters.

**Q5. Is `azure-ai-formrecognizer` wired up?**  
Not in Phase 1. It is listed in `requirements.txt` but not imported or used anywhere. OCR fallback (Tesseract or Azure Document Intelligence) is future work.

**Q6. What was the test plan before delivery?**  
- **Smoke test:** `scripts/run_script_tests_in_order.py` verifies script execution order.  
- **Correctness evaluation:** `docs/correctness_test_report.md` — three queries against `rag_hydrogen_books` corpus, expected vs actual answers, pass/fail ratings, and per-query retrieval relevance checks.  
- **Methodology:** `docs/system_test_methodology.md` describes models, data, evaluation criteria, and how to rerun tests.  
- **Numeric preservation:** `scripts/check_numbers.py --collection rag_hydrogen_books --data-root ./data` → `state/reports/numeric_issues.json`.

**Q7. Estimate for Section 1?**  
Original estimate was a 5–7 day hardening window. All ten review items are now resolved in-tree. Remaining future work: full OCR pipeline, Streamlit authentication, optional CSV export for numeric reports, and client sign-off on the correctness matrix.

---

## Next steps

- Merge this response and the updated documentation (`README.md`, `docs/`) into the main branch.
- Await client validation of the sample payload and correctness report.
- Plan Phase 2 work (OCR, auth, optional hybrid retrieval).
