#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="runs/benchmark/baseline_full50_two_step_100eps_seed1000000"
CHECKPOINT="runs/smolvla_full_dense_b16_ga4_72h/final.pt"
HF_REPO="sungkyunner/univtac-smolvla-baseline-full50-two-step-eval"

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p logs
exec 9>logs/univtac_baseline_full50_eval.lock
flock -n 9 || { echo "A matched full-50 baseline evaluation supervisor is already running."; exit 1; }
printf '%s\n' "$$" >logs/univtac_baseline_full50_eval.pid

# Reset/pre-move occasionally fails for a valid deterministic seed. Successful
# task/seed records are retained, and a fresh simulator process resumes them.
attempt=1
until bash scripts/benchmark_refinement.sh \
    --baseline \
    --episodes 100 \
    --start-seed 1000000 \
    --video-episodes 3 \
    --resume \
    --output "$RUN_DIR"; do
    echo "Matched benchmark attempt ${attempt} failed; resuming completed task/seed records."
    attempt=$((attempt + 1))
    sleep 10
done

"$PROJECT_ROOT/.venv-train/bin/python" -u scripts/publish_benchmark.py \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --repo-id "$HF_REPO" \
    --episodes-per-task 100 \
    --videos-per-task 3
