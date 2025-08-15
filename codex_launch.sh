#!/usr/bin/env bash
set -euo pipefail

# 1) make sure the code in src/ is importable
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# 2) rebuild sentence-embedding cache if missing
if [ ! -f data/embeddings.npy ]; then
  echo "[codex] no cache – rebuilding"
  python - <<'PY'
from wba.local_rag import LocalRAG
LocalRAG()            # instantiation triggers cache build
print("[codex] cache built")
PY
fi

# 3) hand control to the chat CLI
exec python app/local_cli.py
