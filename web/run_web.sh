#!/usr/bin/env bash
# Launch the swarm web UI. Creates a local venv with Flask on first run
# (system Python on CachyOS is externally-managed, so we don't touch it).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-web"

if [ ! -x "$VENV/bin/flask" ]; then
  echo "[setup] creating venv + installing Flask at $VENV ..."
  python -m venv "$VENV"
  "$VENV/bin/pip" -q install --upgrade pip >/dev/null
  "$VENV/bin/pip" -q install "flask>=3.0"
fi

echo "[run] http://127.0.0.1:${PORT:-5000}   (Ctrl-C to stop)"
cd "$ROOT"
exec "$VENV/bin/python" web/app.py
