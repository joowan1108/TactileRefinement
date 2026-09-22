#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv-train/bin/python" ]]; then
    uv venv --python 3.12 "$ROOT/.venv-train"
fi
uv pip install --python "$ROOT/.venv-train/bin/python" -r configs/training_freeze.txt
uv pip check --python "$ROOT/.venv-train/bin/python"
