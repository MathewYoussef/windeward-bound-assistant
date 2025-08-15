#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"
python -m app.local_cli
