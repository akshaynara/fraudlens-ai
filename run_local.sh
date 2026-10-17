#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] && set -a && source .env && set +a
uvicorn app.main:app --reload --port 8080
