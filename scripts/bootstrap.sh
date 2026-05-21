#!/usr/bin/env bash
# Local development bootstrap.
set -euo pipefail

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — fill in real values before running."
fi

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install

echo "Bootstrap complete. Next:"
echo "  docker compose up -d postgres qdrant"
echo "  alembic upgrade head"
echo "  uvicorn app.main:app --reload"
