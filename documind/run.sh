#!/bin/bash
cd /root/documind
source .venv/bin/activate
source /root/.secrets/cognitionsync-apis
export SESSION_SECRET="${SESSION_SECRET:-$(cat /root/.secrets/documind_session 2>/dev/null || echo change-me-in-prod)}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8600 "$@"
