# Checking whether Stage 2 uses tactile information

These tools load saved `expert` checkpoints without updating any weights or
changing the training dataset. They restore the checkpoint's Stage 2 prefix-mask
setting. A joint/grounding checkpoint is rejected rather than silently testing a
different training stage. The existing refinement benchmark also restores the
saved stage before inference.

## Offline checkpoint comparison

From `/tactile/new`:

```bash
PYTHONPATH=src .venv-train/bin/python -m tactile_refinement.ablation \
  --checkpoints runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_1.pt \
  --samples 256 \
  --output runs/diagnostics/stage2_tactile_ablation_33333
```

Pass multiple paths after `--checkpoints` to compare saved updates. The sample
indices, donor bank, initial noise and flow timesteps are identical across
checkpoints. Configurations must match apart from output paths, device and update
budgets. This uses training-corpus observations, not a new train/validation split.

For each observation the tool encodes RGB/language/state once, computes one frozen
SmolVLA handoff, then tests three compact conditions:

1. `correct`: current left/right sensor depth with the observation's VL context.
2. `mismatched`: a same-task donor's paired depth maps, recomputed with the
   **recipient's unchanged VL context**. Donors exclude the entire recipient
   episode file. Selection uses nearest robot pose after per-joint scaling within
   a deterministic donor bank, not future targets or prediction error.
3. `disabled`: zero the actual compact condition **after** resampling, including
   learned resampler biases. Cross-attention/LoRA modules remain enabled.

Every branch starts from the same handoff. Reports include normalized arm,
gripper and combined action MSE against demonstrated actions; action RMS change
relative to correct tactile; and Stage 2 flow MSE at four fixed interior bridge
times (0.05, 0.15, 0.25, 0.35). Padded action slots never contribute.

Outputs are `samples.jsonl` and `summary.json` per checkpoint, plus a combined
`comparison.json`. Positive **control-minus-correct error gaps** favor real tactile.
95% bootstrap intervals resample source episodes, rather than treating overlapping
frames as independent. Single-episode intervals are omitted. Reports stratify by
task and above/below median current depth activity; that activity is only a proxy,
not a calibrated physical contact label. Donor pose distance and depth difference
are recorded so near-identical substitutions and poor matches can be identified.

Action changes alone show dependence, not usefulness. Lower error with correct
tactile supports useful adaptation on these training examples; it does not establish
generalization. Retrieval accuracy is not substituted for this action-level test.

## Matched closed-loop interventions

```bash
OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 PYTHONPATH=src \
  .venv-train/bin/python -m tactile_refinement.ablation_rollout \
  --checkpoint runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_1.pt \
  --tasks insert_HDMI pull_out_key \
  --conditions correct mismatched disabled \
  --episodes 20 --start-seed 2000000 --video-episodes 3 \
  --schedule matched60 \
  --output runs/benchmark/stage2_tactile_ablation_33333
```

Each task/seed runs all three conditions in a deterministic, counterbalanced order.
The normal prefix context always comes from the current rollout observation. The
mismatched condition substitutes only the sensor maps, selecting a same-task
training donor near the current robot pose. The policy noise uses the same seed
for corresponding model calls across conditions. Observation trajectories can,
and should, diverge after the intervention changes actions.

The default `matched60` schedule obtains a new observation and action chunk once
per recorded-data interval, applies UniVTAC's temporal aggregation at that 60 Hz
index, and executes the aggregated endpoint over two 120 Hz physics substeps. The
first substep receives the midpoint from the preceding endpoint and the second
receives the new endpoint; both receive explicit zero velocity targets. There is
no EMA. The action trace includes all commanded and observed substep positions.

The diagnostic `official` schedule instead replans and advances aggregation every
120 Hz physics step; it retains the known mismatch with 60 Hz training. The
alternative `full50` schedule preserves two physics steps per target but executes
all 50 targets before replanning. Never compare outcomes across schedules as
evidence for a tactile effect.

The output contains metadata/checkpoint hashes, incrementally saved episode rows,
action traces, videos for the first requested seeds in **each** condition, and
success/early-stop/duration summaries. Success differences are **correct minus
control**. Rates use Wilson intervals; contrasts provide paired bootstrap intervals
and conservative bounds combining marginal 97.5% Wilson intervals. The latter avoid
claiming zero uncertainty when all observed outcomes are identical. Duration is separated by successful/failed
outcomes; it is not an unconditional measure of policy speed. Official task success
and early-stop predicates are used; the tool does not invent task-specific slip or
insertion error classifiers.

Simulator reset IDs are checked when making paired comparisons. Reports separate
same-seed contrasts from contrasts with matching recorded reset IDs. These IDs
cover rounded robot/actor poses, not the full deformable simulator state; even the
matching subset is not claimed to be an exact restored-state experiment. A small
sample or mismatched resets can make results inconclusive. Zero and donor inputs
can be unfamiliar, so agreement between both controls is more informative than
the zero control alone.

Use `--resume` with the identical command/output to continue completed episode
records after interruption. Failed simulator runs are not recorded as policy
failures. Use a different directory for each checkpoint or changed setting.
`--smoke-steps 1 --episodes 1 --video-episodes 0` checks all three integration paths
without running full episodes; smoke rows are excluded from success statistics.

Both CLIs can be launched detached from SSH by prefixing the command with
`nohup env` (before `PYTHONPATH=src`) and appending a log redirection and `&`.
They do not pause, resume or otherwise control existing training/evaluation jobs.

## Verification

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_tactile_ablation.py
```

Tests cover stage-specific inference masking, shared handoff/flow states,
padding exclusion, raw-depth-only interventions, zeroing after compression,
same-task/different-episode donors, episode-clustered uncertainty, reset mismatch
reporting, smoke exclusion and both rollout schedules.
