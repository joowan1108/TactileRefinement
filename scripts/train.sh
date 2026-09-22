#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec "$ROOT/.venv-train/bin/python" -u -m smolvla_baseline.train "$@"
