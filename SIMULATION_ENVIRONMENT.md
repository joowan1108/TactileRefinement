# UniVTAC simulation and evaluation settings

Last verified: 2026-09-18. This document records the simulator and controller settings used for the SmolVLA baseline and tactile-refinement evaluations. The two evaluation schedules are intentionally different and must be reported separately.

## Reproducibility boundary

- UniVTAC repository: [`univtac/UniVTAC`](https://github.com/univtac/UniVTAC)
- Pinned UniVTAC commit: `0dafa10262e22f486f160a55d6f11aeab12d8e7b`
- Simulator: Isaac Sim `4.5.0.0`
- Isaac Lab: `0.41.3`
- Simulator Python: `3.10.12`
- Simulator PyTorch: `2.5.1+cu124`
- cuRobo: `0.7.7`
- Simulator NumPy: `1.26.4`
- GPU used for the reported runs: NVIDIA GeForce RTX 3090

The evaluator checks that the pinned UniVTAC schedule files are unmodified before an official-schedule run. The simulator runs in a separate Python 3.10 process because the policy environment and Isaac Sim use different Python and NumPy versions. Communication uses a private Unix socket; no simulator service is exposed over TCP.

## Simulator configuration

| Setting | Value |
|---|---|
| Isaac application | Headless, cameras enabled, livestream disabled |
| Device | `cuda:0` |
| Parallel environments | 1 |
| Physics timestep | `1/120 s` (120 Hz) |
| Task decimation | 1 |
| UIPC timestep | Same as physics timestep |
| UIPC ground height | `0.001 m` |
| UIPC contact distance `d_hat` | `0.0005 m` |
| UIPC friction | Enabled |
| UIPC epsilon velocity | `0.1` |
| PhysX continuous collision detection | Enabled |
| Rigid-body friction combine mode | Multiply |
| Rigid-body restitution combine mode | Multiply |
| Restitution | 0 |
| Random textures | Disabled |
| Tactile sensor type | GelSight Mini (`gsmini`) |
| Scene query support | Enabled, so the two deformable gel pads attach correctly in headless mode |
| Reset time limit | 120 seconds |

Internal UniVTAC saving and video recording are disabled during evaluation (`save_frequency = video_frequency = render_frequency = 0`). The external evaluator records frames from returned observations instead. Physics and tactile simulation remain enabled in every run, including the tactile-free baseline, because task success, early-stop logic, and contact dynamics depend on them.

## Cameras and observations

The simulator provides two RGB cameras:

| Camera | Simulator resolution | Update period |
|---|---:|---:|
| Head | 480 × 270 | `1/120 s` |
| Wrist | 480 × 270 | `1/120 s` |

The policy receives the instruction string, head RGB, wrist RGB, and a nine-value joint state. The joint state contains seven arm joints and two finger positions. Actor state is retained inside the simulator for reset verification but is not a policy input.

For the **SmolVLA baseline**, tactile arrays are deliberately omitted from policy IPC. The two tactile sensors are still simulated and checked after reset and every step for finite gel-pad, marker-motion, and height-map state.

For the **tactile-refinement policy**, the evaluator additionally sends ordered left/right depth maps with shape `2 × 240 × 320`, in millimetres. Their order is `left_gsmini`, then `right_gsmini`. The policy normalizes each valid pixel as

$$
D = \mathrm{clip}((34-H_{\mathrm{mm}})/10, 0, 1).
$$

Invalid pixels become zero after masking. This value is treated as normalized tactile depth, not calibrated indentation or force.

## Action interface

The policy predicts absolute joint-position chunks with shape `50 × 9`. The nine values are seven arm targets plus two finger targets. UniVTAC's evaluation API consumes an eight-value target: the evaluator passes the seven arm values and the first finger value, and UniVTAC broadcasts that shared gripper target to both fingers.

Actions are denormalized by the policy postprocessor before entering the simulator. The reported evaluations use bfloat16 autocast for policy inference and float32 physical action arrays. No action EMA is enabled in the current official baseline or matched tactile runs.

## Schedule A: official SmolVLA baseline

This schedule follows the released UniVTAC SmolVLA deployment logic.

1. Read a new observation at every 120 Hz physics step.
2. Predict a new 50-action SmolVLA chunk at every step.
3. Add the chunk to the temporal-aggregation buffer.
4. Execute the weighted aggregate for the current timestep.
5. Advance exactly one 120 Hz physics step.

Temporal aggregation uses a capacity of 1,000 timesteps and `k = 0.01`. If predictions that began at indices $s$ cover the current index $t$, their current actions are combined in oldest-to-newest order using weights

$$
w_i=\exp(-0.01i),\qquad i=0,1,\ldots,n-1.
$$

The executed action is the weighted mean. The evaluator independently reconstructs every aggregated action from the saved chunks and checks it numerically.

The controller is UniVTAC's native `task.take_action(qpos)` path with `force=True`. It sets arm and gripper position targets, directly updates the corresponding DOF positions through the robot PhysX view, and advances one physics step. Existing velocity targets are left untouched. There is no interpolation and no EMA. Videos are encoded at 120 fps.

The baseline policy inputs are only RGB, language, and joints. This schedule has a known timing caveat: the demonstration data were recorded every two physics steps (60 Hz), while the official deployment schedule advances the 50-action chunk index at 120 Hz.

Current baseline evaluation identity:

- Eight tasks
- 100 requested episodes per task
- Per-task seeds `1000000` through `1000099`
- First three episodes per task recorded as video
- Resume is keyed by `(task, seed)` and never overwrites completed episodes

## Schedule B: matched tactile-refinement evaluation

This is the default schedule for the Stage 2 tactile ablations and attention visualizations. It matches the 60 Hz cadence of recorded UniVTAC demonstrations while retaining 120 Hz physics.

1. Read current RGB, joints, and tactile depth at a 60 Hz policy decision.
2. Predict a fresh 50-action chunk.
3. Advance temporal aggregation once at 60 Hz and select one endpoint action.
4. Execute that endpoint through two 120 Hz physics substeps.
5. Read a new observation and repeat.

The two-substep controller reconstructs a linear position path from the previous commanded endpoint to the new endpoint:

$$
q_{1/2}=\tfrac12(q_{\mathrm{previous}}+q_{\mathrm{target}}),
\qquad q_1=q_{\mathrm{target}}.
$$

The first action starts from the observed post-reset robot position. Both substeps use explicit zero velocity targets and `force=True`. Arm and shared-gripper positions are interpolated. Success and early-stop predicates are checked after each substep, so an episode may advance only one of the two substeps on its final action. Each pair counts as one policy action. Videos are encoded at 60 fps.

Stage 2 final-checkpoint attention runs use:

- Schedule name: `matched60`
- Controller name: `linear_position`
- Policy rate: 60 Hz
- Physics rate: 120 Hz
- Two physics substeps per complete policy action
- New visual-language, proprioceptive, and tactile observation every policy action
- Temporal aggregation enabled at the 60 Hz policy index
- No EMA
- Seed `2000000` for the one-episode-per-task diagnostic

The attention evaluation's one rollout per task is qualitative and must not be reported as a task success-rate estimate.

## Other implemented schedules

These modes exist for controlled experiments and should not be mixed into the primary baseline/refinement comparison.

| Schedule | Replanning | Execution | Status |
|---|---|---|---|
| `official` | New 50-action chunk every action | One 120 Hz native action | Official baseline schedule |
| `matched60` | New 50-action chunk every action | Midpoint and endpoint at 120 Hz | Default tactile diagnostic schedule |
| `full50` | One prediction per full chunk | All 50 actions, two physics substeps each | Open-loop reference mode |
| `tactile10_vl50` | New 50-action chunk every 10 actions; execute first 10 | Two substeps per action | Implemented inference experiment; not used in the reported runs |

In `tactile10_vl50`, tactile and proprioception refresh every 10 actions, while RGB/language context refreshes every 50. It uses no temporal aggregation or EMA. Because training did not include stale visual context or variable prediction horizons, this mode requires separate labeling and validation.

## Tasks and termination

| Task | Policy-action limit |
|---|---:|
| `grasp_classify` | 300 |
| `pull_out_key` | 300 |
| `insert_HDMI` | 600 |
| `lift_bottle` | 500 |
| `lift_can` | 300 |
| `insert_tube` | 300 |
| `put_bottle_in_shelf` | 300 |
| `insert_hole` | 300 |

Each task uses UniVTAC's own reset, pre-move planner, success predicate, and early-stop predicate. A reset is accepted only when UniVTAC reports `plan_success`. Episodes end on task success, task-specific early stop, or the policy-action limit. The custom two-substep controller is used only after reset; reset and pre-move retain UniVTAC's original controller.

## Reset, seeds, and validation

- The simulator is reset with an explicit episode seed and the task instruction.
- PyTorch policy noise is seeded with the same episode seed.
- Temporal-aggregation and policy state are cleared at every reset.
- The evaluator hashes rounded robot-joint and actor poses to detect reset mismatches. This is not a checksum of hidden deformable/UIPC state.
- Both tactile sensors must exist and have at least three attachment points. All checked tactile tensors must be finite.
- Completed rows record policy actions, model calls, physics-step counts, simulated time, success, early stop, and tactile health.
- Resume validates task/seed uniqueness and the complete schedule metadata before continuing.

## Installation and launch requirements

Accept the NVIDIA Omniverse EULA before starting Isaac Sim:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
```

Prepare the simulator environment from the project root:

```bash
bash scripts/setup_benchmark.sh
```

The setup script verifies the exact simulator package versions listed above. Evaluation creates one simulator subprocess per active task and writes its logs under `logs/univtac_workers/`.

## Reporting checklist

When sharing a result, always state:

1. UniVTAC commit and task.
2. Checkpoint hash and training stage.
3. Schedule (`official`, `matched60`, `full50`, or `tactile10_vl50`).
4. Policy rate, physics rate, replanning interval, and physics substeps per action.
5. Temporal-aggregation parameters.
6. Controller position and velocity behavior.
7. Whether tactile input and action EMA were enabled.
8. Episode seeds, episode count, and video count.
9. Any reset or simulator failures, reported separately from policy failures.

The official 120 Hz baseline and matched 60 Hz tactile evaluation are useful for different questions. Their success rates are not a controlled head-to-head comparison unless both policies are rerun with the same schedule and controller.
