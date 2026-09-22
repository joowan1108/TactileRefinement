#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="runs/benchmark/final_100eps_seed1000000"
CHECKPOINT="runs/smolvla_full_dense_b16_ga4_72h/final.pt"
HF_REPO="sungkyunner/univtac-smolvla-baseline-eval"

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p logs
exec 9>logs/univtac_baseline_eval_100eps.lock
flock -n 9 || { echo "A baseline evaluation supervisor is already running."; exit 1; }
printf '%s\n' "$$" >logs/univtac_baseline_eval_100eps.pid

# UniVTAC's motion planner can intermittently reject a task's pre-move even
# with the same seed. Each successful task/seed is persisted atomically enough
# for --resume, so restart the simulator process until all 800 records exist.
# The publisher below remains the strict completion gate and never uploads a
# partial benchmark.
attempt=1
until scripts/benchmark.sh \
    --checkpoint "$CHECKPOINT" \
    --episodes 100 \
    --start-seed 1000000 \
    --video-episodes 3 \
    --video-fps 30 \
    --resume \
    --output "$RUN_DIR"; do
    echo "Benchmark attempt ${attempt} failed; resuming completed task/seed records."
    attempt=$((attempt + 1))
    sleep 10
done

"$PROJECT_ROOT/.venv-train/bin/python" -u scripts/publish_benchmark.py \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --repo-id "$HF_REPO" \
    --episodes-per-task 100 \
    --videos-per-task 3
