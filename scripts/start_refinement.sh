#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
test -s runs/smolvla_full_dense_b16_ga4_72h/final.pt || { echo 'Missing trained SmolVLA baseline checkpoint.'; exit 1; }
exec 9>logs/tactile_refinement.lock
flock -n 9 || { echo 'A refinement training launcher is already running.'; exit 1; }
nohup setsid bash scripts/train_refinement.sh "$@" </dev/null >>logs/tactile_refinement.log 2>&1 &
printf '%s\n' "$!" >logs/tactile_refinement.pid
echo "Started refinement training (PID $!); log: $ROOT/logs/tactile_refinement.log"
