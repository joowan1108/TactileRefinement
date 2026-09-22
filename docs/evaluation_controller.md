# Evaluation controller

## Ten-action replanning with cached visual-language context

Use `tactile_refinement.ablation_rollout --schedule tactile10_vl50` with an existing
Stage 2 checkpoint. At executed-action indices 0, 10, 20, ... the policy receives
current tactile and normalized joints, generates a new baseline flow handoff and
a refined 50-action chunk, and executes the first 10 actions. The remaining 40
are discarded. At indices 0, 50, 100, ... it also recomputes the image/language
prefix. There is no temporal aggregation or EMA in this mode.

The cached prefix is not treated as an immutable whole: its state token is
recomputed through every text-transformer layer using cached visual-language
keys and values, with the original padding mask and rotary position. The block
attention mask prevents VL tokens attending to state, so their cached values
remain valid. Current state also feeds the refinement state adapter. Stage 2's
direct-prefix mask is preserved; the predictor and baseline handoff use the
cached visual-language context. Only predictor outputs reach tactile attention.

Each executed action retains the `linear_position` midpoint/endpoint controller
at two 120 Hz physics steps (unless terminated early), zero velocity targets,
and the 60 Hz action index. Ten actions span 20 physics steps, approximately
0.167 seconds; 50 span approximately 0.833 seconds. Videos use 60 fps.

Episode results record replan, state/tactile refresh and VL refresh indices.
Action traces additionally save every predicted 50-action chunk and observed
state at replanning, allowing verification that only slots 0–9 were executed.
Caches and queues reset on each episode. This mode uses existing weights and
does not change the training objective; pairing fresh tactile with up to 40
actions-old VL context is an inference distribution change to evaluate.

```bash
PYTHONPATH=src .venv-train/bin/python -m tactile_refinement.ablation_rollout \
  --checkpoint runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt \
  --tasks pull_out_key --conditions correct --episodes 1 --video-episodes 1 \
  --start-seed 2000000 --schedule tactile10_vl50 \
  --output runs/benchmark/stage2_checkpoint_66666_tactile10_vl50_key
```

`scripts/run_tactile10_vl50_probe_background.py --wait-pid PID` can wait for the
existing checkpoint-66666 eight-task supervisor, then run a 55-action Grasp and
Classify smoke test followed by one complete Pull Out Key episode. Both use
seed 2000000. If the smoke ends early, its refresh indices show that the 50-action
boundary was not reached; smoke outcomes are never reported as success scores.
Each completed video is decoded to verify frame count before uploading, together
with metadata and action traces, under `checkpoint_66666/tactile10_vl50/` in the
existing private Stage 2 dataset. The runner records waiting, evaluation, upload,
completion or failure in `logs/stage2_checkpoint_66666_tactile10_vl50_status.json`.

## Official UniVTAC schedule probe

`smolvla_baseline.official_benchmark` is the separate official-schedule baseline entry point, pinned to upstream commit `0dafa10262e22f486f160a55d6f11aeab12d8e7b`.
It reuses `_SmolVLALocalPolicy._reset_temporal_aggregation` and `_aggregate_action_chunk` directly from UniVTAC, with parameters from `policy/smolvla/deploy.yml`.
Every step predicts a new 50-action chunk from the latest observation, aggregates predictions aligned to the current timestep (exp(-0.01*i), oldest first), and calls native `task.take_action(qpos)` exactly once.
`decimation=1` and physics dt=1/120 mean 120 Hz; there is no extra hold, interpolation, EMA, or velocity-target override. Temporal aggregation averages different predictions of the same timestep; it is not the earlier EMA over successive executed actions.
The HDMI action limit remains 600, so the maximum rollout is five simulated seconds. The video uses 120 fps.
Our checkpoint's original input features, normalization, and bfloat16 inference remain unchanged. This reproduces the execution schedule, not published policy scores. The training dataset's two-physics-step recording interval is not rescaled by the upstream schedule; that remains a limitation of applying the official indexing directly to this baseline.
Main evaluation and training processes are independent; previous results are retained under their original schedules.

```bash
.venv-train/bin/python -u scripts/run_official_hdmi_probe.py
```

This one-episode HDMI runner verifies logged aggregation independently, checks inference/action/physics-step counts, validates the video, and uploads to a separate experiment folder in the existing private Hugging Face repository. It also archives the exact upstream sources used.

The full baseline benchmark uses `scripts/run_official_baseline_eval_background.py`: 100 episodes for each of the eight tasks, starting at seed 1000000, with the first three videos per task. Its output directory is `runs/benchmark/baseline_official_100eps_seed1000000` and its private Hugging Face dataset is `sungkyunner/univtac-smolvla-baseline-official-schedule-eval`.
Completed episode identities are validated on resume; model, temporal buffer, and simulator episode state reset for every seed. The supervisor retries failed processes up to five attempts without counting simulator exceptions as policy failures. Completed videos upload immediately for the first three episodes of each task, with partial success totals refreshed every ten episodes and on completion.
The former full-50/two-step process was stopped on 2026-09-16 at the user's request; its completed artifacts remain in their original directory. Training continues independently.

## Earlier full-chunk controller experiments

New `tactile_refinement.benchmark` runs default to `--controller linear_position` with no EMA. Model endpoints are passed through unchanged, with midpoint/endpoint commands at physics cadence and explicit zero velocity targets.
The completed `linear` HDMI probe showed observed arm second differences of 0.001597 rad versus 0.000972 rad in filtered targets, and endpoint error correlated 0.967 with endpoint increments. Derived velocity targets combined with forced position updates are a suspected source of overshoot. Do not treat this mode as a validated improvement.
The completed EMA + `linear_position` HDMI probe reduced arm endpoint mean absolute error to 1.55e-8 rad (maximum 8.94e-6 rad), versus 3.09e-4 rad for EMA + `linear`. Observed second differences matched the filtered targets (0.00088138 rad); the episode still failed. This supports removing controller-induced overshoot, not a claim that the model's raw trajectory is smooth or that task success improves. The new no-EMA probe isolates raw model endpoints from endpoint smoothing and records observed joints after both substeps.
Resuming a run retains its recorded controller; older metadata without a controller means `hold`.
An explicit controller change fails the metadata comparison, so use a new output directory.
Existing running evaluations keep the hold controller, including newly spawned workers: the worker default remains `hold`.

The policy still predicts 50 absolute-position actions sampled at 60 Hz. Each action consumes two 120 Hz physics steps.
Let s be the previous commanded endpoint (observed joint positions immediately after reset for the first action), e the next endpoint, and dt=1/120 second.
Both interpolated controllers issue position (s+e)/2 then e. `linear` sets velocity (e-s)/(2*dt) on both steps; `linear_position` sets zero velocity targets on both steps.
This applies to seven arm joints and the shared gripper target; UniVTAC broadcasts the first finger prediction to both fingers as before.
The position/velocity setters retain demonstration collection's force=True behavior.
This removes the immediate endpoint jump and explicitly replaces stale pre-move velocity targets.
Linear interpolation does not recover the original planner trajectory and does not guarantee velocity continuity at action boundaries or better task success.

The policy action counter advances once, and success/early-stop is checked after each physical substep.
Controller state resets each episode and continues across chunk boundaries. Reset and scripted pre-move are unchanged.
Training code, weights, normalization, model inputs, and chunk inference frequency are unchanged.

Optional `--arm-ema-alpha 0.5` operates on arm endpoints before interpolation. Gripper endpoints are not EMA filtered but are interpolated at physics cadence.
The video label and metadata identify the controller. Action traces record raw/filtered endpoints, observed joint positions, and commanded substep positions/velocities.

Run a separate single-episode HDMI controller test without EMA and upload it to the existing private Hugging Face repository:

```bash
.venv-train/bin/python -u scripts/run_ema_hdmi_probe.py --controller linear_position --no-ema
```

The publisher verifies exact equality between raw model endpoints and controller endpoints when EMA is disabled, both substep commands, zero velocity targets, physics timing, decoded video frame count, and uploaded video hash. Tracking error is measured separately rather than assumed from commanded positions. The historical probe script name is retained for reproducing earlier EMA experiments.
