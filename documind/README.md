# DocuMind — productized RAG (FastAPI)

A multi-user, productized layer built on top of this repo's document-QA baseline.
Where the baseline (`../scripts`, `../frontend`) is a Streamlit prototype for ingestion +
retrieval, DocuMind wraps the same RAG approach in a deployable product:

- **Auth**: email/password signup + login (bcrypt, signed session cookies)
- **Multi-format ingestion**: PDF, DOCX, XLSX/CSV, TXT/MD, and **images** (via Gemini vision)
- **Per-user isolation**: embedded Qdrant, filtered by `user_id`
- **Grounded, cited answers**: retrieval → Gemini with inline `[n]` citations, honest refusal
- **Report generation**: any answer → downloadable PDF (reportlab)
- **Token usage stats** per user (SQLite)
- Landing page + login/signup + chat app UI (Jinja + Tailwind)

## Stack
FastAPI · embedded Qdrant (`QdrantClient(path=…)`) · Voyage embeddings (voyage-3-lite, 512d) ·
Google Gemini (answers + image vision) · SQLite.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=... VOYAGE_API_KEY=... SESSION_SECRET=$(openssl rand -hex 32)
uvicorn app.main:app --host 0.0.0.0 --port 8600
```
Open http://localhost:8600. Rename the product in `app/config.py` (`PRODUCT_NAME`).

## Relationship to the baseline
Chunking, retrieval, and citation concepts follow this repo's `scripts/` design
(one collection per source, leaf-chunk retrieval, numeric-preservation mindset). The
embedded vector store replaces the external Qdrant service for a self-contained deploy;
swap `app/rag/store.py` back to a hosted Qdrant for scale.
