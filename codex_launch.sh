#!/usr/bin/env bash
set -euo pipefail

# 1) make sure the code in src/ is importable
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# 2) hand control to the chat CLI (embedding cache prebuilt at setup)
exec python app/local_cli.py
