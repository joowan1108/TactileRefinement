#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src" TOKENIZERS_PARALLELISM=false
exec "$ROOT/.venv-train/bin/python" -u -m tactile_refinement.train "$@"
