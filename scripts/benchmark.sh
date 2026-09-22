#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
exec "$ROOT/.venv-train/bin/python" -u -m smolvla_baseline.benchmark "$@"
