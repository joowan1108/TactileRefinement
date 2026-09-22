#!/usr/bin/env bash
# Start a detached run; no process is launched merely by setting up this project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
if [[ -f logs/training.pid ]] && kill -0 "$(cat logs/training.pid)" 2>/dev/null; then
    echo "Training process already running: $(cat logs/training.pid)" >&2
    exit 1
fi
python3 - "$@" <<'PY'
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
with (root/'logs/training.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(
        ['bash', str(root/'scripts/train.sh'), *sys.argv[1:]],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
(root/'logs/training.pid').write_text(str(process.pid) + '\n')
print(f'Started detached PID {process.pid}; follow logs/training.log')
PY
