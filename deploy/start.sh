#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH."
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  if [[ -f "$ROOT/.env.example" ]]; then
    echo "No .env found; copy from .env.example"
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "Edit .env before production use."
  else
    echo "No .env in project root."
    exit 1
  fi
fi

COMPOSE="${COMPOSE_FILE:-docker-compose.prod.yaml}"
echo "Using compose file: $COMPOSE"
docker compose -f "$COMPOSE" --env-file "$ROOT/.env" up -d --build "$@"
