#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo "Starting AI Project..."
echo "====================================="

# Move to project directory (directory of this script).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "[1] Activating venv..."
if [[ ! -f "venv/bin/activate" ]]; then
  echo "ERROR: venv not found!"
  exit 1
fi
# shellcheck disable=SC1091
source "venv/bin/activate"

echo
echo "[2] Starting Ollama..."
ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

sleep 4

echo
echo "[3] Checking model..."
if ! ollama list | grep -q "qwen2.5:1.5b-instruct"; then
  echo "Downloading model..."
  ollama pull qwen2.5:1.5b-instruct
fi

echo
echo "[4] Building index..."
python build_clean_index.py

echo
echo "[5] Starting Flask (IMPORTANT)..."
python -m app.main >/tmp/flask.log 2>&1 &
FLASK_PID=$!

sleep 3

echo
echo "[6] Opening browser..."
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:5000" >/dev/null 2>&1 || true
else
  echo "Open this URL manually: http://127.0.0.1:5000"
fi

echo
echo "====================================="
echo "DONE"
echo "====================================="

echo
if command -v wait >/dev/null 2>&1; then
  wait "$FLASK_PID"
fi
