#!/usr/bin/env bash
set -euo pipefail

# ensure modules import
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Build sentence-embedding cache if missing
if [ ! -f data/embeddings.npy ]; then
  echo "[codex setup] building sentence-embedding cache"
  python - <<'PY'
from wba.local_rag import load_pages, _build_embeddings
pages = load_pages("extracted_text.json")
texts = [p["content"] for p in pages]
_build_embeddings(texts)
print("[codex setup] cache built")
PY
fi

