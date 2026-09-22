#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
exec "$ROOT/.venv-train/bin/python" -u -m tactile_refinement.benchmark "$@"
