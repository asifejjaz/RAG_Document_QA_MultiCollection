# Client deploy notes (Eva Research RAG)

## Dev vs production Compose

| File | Use |
|------|-----|
| `docker_compose.yaml` | Local development: bind-mounts `.:/app` so code edits hot-reload. |
| `docker-compose.prod.yaml` | Production-style: **no** source bind-mount; named volumes for data, state, sessions; pinned service images. |

The **Dockerfile** runs the app as non-root user **`raguser` (uid 10001)**. On Linux, if you bind-mount the repo into the container, ensure files are readable by that uid or adjust ownership.

## Quick start (production compose)

1. Copy **`.env.example`** → **`.env`** and set secrets and `QDRANT_URL` / `VECTOR_DB_URL`.
2. Load your corpus into the **`rag_data_prod`** volume (or mount a host directory by replacing the named volume with a bind path in the compose file).
3. Run:

```bash
docker compose -f docker-compose.prod.yaml --env-file .env up -d --build
```

Streamlit: **http://localhost:8501**

## Helper scripts

- **`deploy/start.sh`** — check Docker, optional `.env` copy from example, `compose up`.
- **`deploy/start.ps1`** — same for Windows PowerShell.

## Before exposing to a network

- Add **authentication** (e.g. Streamlit Authenticator, reverse proxy basic auth, or VPN). See Phase 1 review / `docs/RAG_Phase1_Review_Fix_Plan.md`.

## Artifacts for acceptance

- `scripts/index_text.py`, `frontend/app.py`
- Sample payload: `python scripts/export_sample_qdrant_payload.py --collection <name> --out sample_payload.json`
- Correctness / regression: `docs/correctness_test_report_template.md` and `scripts/run_script_tests_in_order.py`
