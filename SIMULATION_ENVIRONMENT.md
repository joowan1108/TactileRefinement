# UniVTAC simulation setup specification

Last verified: 2026-09-18. This document specifies how to reproduce the UniVTAC simulation environment used by this project. It contains simulator setup only; it does not define a policy, training procedure, action schedule, checkpoint, or experiment.

## Source and dependency versions

| Component | Version |
|---|---|
| UniVTAC | commit `0dafa10262e22f486f160a55d6f11aeab12d8e7b` |
| Isaac Sim | `4.5.0.0` |
| Isaac Lab | `0.41.3` |
| Python | `3.10.12` |
| PyTorch | `2.5.1+cu124` |
| NVIDIA cuRobo | `0.7.7` |
| NumPy | `1.26.4` |

Repository: [`univtac/UniVTAC`](https://github.com/univtac/UniVTAC)

Use an NVIDIA CUDA GPU supported by Isaac Sim 4.5. The current machine uses an NVIDIA GeForce RTX 3090, but the simulation code does not depend on that exact model.

## Required environment variables

Accept the NVIDIA Omniverse EULA before launching Isaac Sim:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
```

In this workspace, the simulator environment is prepared with:

```bash
bash scripts/setup_benchmark.sh
```

The setup script creates `.venv-benchmark`, exposes the installed simulator packages to it, and verifies the exact versions listed above.

## Isaac application initialization

Isaac Sim must be launched before importing UniVTAC task modules:

```python
from isaaclab.app import AppLauncher

launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    livestream=0,
    device="cuda:0",
)
simulation_app = launcher.app
```

Use one environment on `cuda:0`. Headless mode still requires `enable_cameras=True` because UniVTAC observations include rendered RGB images.

## Task configuration

After starting the Isaac application, import the selected task module and override its evaluation settings:

```python
import importlib
from pathlib import Path

task_name = "insert_tube"
module = importlib.import_module(f"envs.{task_name}")
cfg = module.TaskCfg()

cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.sim.enable_scene_query_support = True
cfg.decimation = 1
cfg.tactile_sensor_type = "gsmini"
cfg.random_texture = False

# Disable UniVTAC's internal dataset/video writers.
cfg.save_frequency = 0
cfg.video_frequency = 0
cfg.render_frequency = 0
cfg.save_dir = Path("eval_result") / task_name

# Minimal observation set for RGB, robot state, tactile depth, and reset checks.
cfg.obs_data_type = {
    "camera": ["rgb"],
    "embodiment": ["joint"],
    "tactile": ["depth"],
    "actor": True,
}

task = module.Task(cfg, mode="eval")
```

`enable_scene_query_support=True` is required in headless mode. TacEx uses scene queries to find the finger-case colliders and attach both deformable gel pads. Without it, the simulator may start while the tactile sensors remain invalid.

Setting a field in `obs_data_type` controls which arrays are returned. It does not disable the underlying physics. For example, the `tactile` entry may be omitted from returned observations while tactile contact simulation remains active.

## Physics configuration

The following values come from UniVTAC's base task configuration:

| Setting | Value |
|---|---|
| Physics timestep | `1/120 s` |
| Physics rate | 120 Hz |
| Task decimation | 1 |
| Render interval | 1 physics step |
| PhysX continuous collision detection | Enabled |
| Friction combine mode | Multiply |
| Restitution combine mode | Multiply |
| Restitution | 0 |
| UIPC timestep | `1/120 s` |
| UIPC ground height | `0.001 m` |
| UIPC contact distance `d_hat` | `0.0005 m` |
| UIPC friction | Enabled |
| UIPC epsilon velocity | `0.1` |

Do not change the timestep or decimation without revalidating contact stability, tactile deformation, controller timing, and task success predicates.

## Camera configuration

UniVTAC's base task provides two cameras:

| Camera | Resolution | Update period |
|---|---:|---:|
| Head | 480 × 270 | `1/120 s` |
| Wrist | 480 × 270 | `1/120 s` |

RGB observations are unsigned 8-bit arrays in HWC layout with three channels. Some tasks override camera placement while preserving the observation interface.

The base head camera uses:

- Position offset: `(0.554, 1.0, 0.150)`
- Rotation: `(0, 0, 0.707, 0.707)` in OpenGL convention
- Focal length: `1.94`
- Focus distance: `1.0`
- Horizontal aperture: `2.688`
- Clipping range: `(0.01, 100.0)`

The wrist camera uses the camera already defined on the robot asset.

## Tactile configuration

Use two GelSight Mini sensors, one on each gripper finger:

```python
cfg.tactile_sensor_type = "gsmini"
```

Each returned tactile depth map has shape `240 × 320`. UniVTAC also simulates marker motion, a height map, gel-pad nodal positions, and the attachment transform even when only depth is requested as an observation.

After reset, validate both tactile sensors:

1. Exactly two tactile sensors exist.
2. Each gel pad has at least three attachment-point indices.
3. The attachment transform is finite.
4. Gel-pad nodal positions are finite and nonempty.
5. Marker-motion output is finite and nonempty.
6. Height-map output is finite and nonempty.

This catches the common failure where a headless simulator starts successfully but a gel pad fails to attach to its case collider.

## Reset procedure

Reset with an explicit seed and one task instruction:

```python
task.reset(seed=episode_seed, instructions=[instruction])

if not task.plan_success:
    raise RuntimeError("UniVTAC pre-move planning failed")

observation = task._get_observations()
```

UniVTAC uses its original reset and pre-move planner. The base reset time limit is 120 seconds. Treat pre-move failure or reset timeout as a simulator/reset failure, not as a policy failure.

For reproducibility, record:

- Task name
- Reset seed
- Language instruction
- UniVTAC commit
- Final task configuration
- Initial robot joint state
- Initial actor poses
- Tactile attachment counts and finite-state checks

Robot joints and actor poses can detect obvious reset mismatches, but they are not a complete checksum of hidden deformable or UIPC state.

## Observation structure

With the minimal configuration above, `task._get_observations()` contains:

```text
observation/head/rgb          uint8 [270, 480, 3]
observation/wrist/rgb         uint8 [270, 480, 3]
embodiment/joint              9 finite values
tactile/<left sensor>/depth   [240, 320]
tactile/<right sensor>/depth  [240, 320]
actor                         task-specific actor state
```

The nine joint values contain seven arm joints and two finger joints. Preserve the left/right tactile ordering explicitly when converting the observation to another format.

## Action interface

UniVTAC's joint-position action API consumes eight values:

```text
[arm_joint_0, ..., arm_joint_6, shared_gripper_position]
```

Apply one action with:

```python
import torch

qpos = torch.as_tensor(action_8d, device=task.device)
executed, success = task.take_action(qpos, action_type="qpos")
```

The shared gripper value is applied to both finger joints. `take_action` sets arm and gripper position targets with `force=True` by default and advances the simulator. Check the task's action counter, physics-step counter, success predicate, and early-stop predicate after stepping.

If an external policy emits nine values containing two finger targets, define the reduction explicitly. The current wrapper uses the first finger target as the shared gripper command; this is a wrapper convention rather than a requirement of Isaac Sim.

## Tasks and default action limits

| Task module | Default action limit |
|---|---:|
| `grasp_classify` | 300 |
| `pull_out_key` | 300 |
| `insert_HDMI` | 600 |
| `lift_bottle` | 500 |
| `lift_can` | 300 |
| `insert_tube` | 300 |
| `put_bottle_in_shelf` | 300 |
| `insert_hole` | 300 |

Use each task module's own `check_success()` and `check_early_stop()` implementation. Do not replace these with video-based manual judgments when calculating benchmark success.

## Process isolation

The simulator should run in its Python 3.10 environment, separate from model or analysis code that may require a different Python, PyTorch, or NumPy version. This project launches one simulator worker subprocess per active task and exchanges arrays through a private Unix socket.

When serializing arrays across environments, send explicit dtype, shape, and contiguous bytes. Do not depend on pickled NumPy module paths because NumPy 1.x and 2.x environments are not pickle-compatible in all cases.

Recommended worker lifecycle:

1. Start Isaac Sim.
2. Import and construct one task.
3. Send a ready message only after task construction succeeds.
4. Handle reset and step commands sequentially.
5. Close the task.
6. Close the Isaac application.
7. Terminate or kill only that worker if graceful shutdown times out.

## Minimal cleanup

Always close both the task and Isaac application:

```python
try:
    # reset/step loop
    ...
finally:
    if task is not None:
        task.close()
    simulation_app.close()
```

Leaving an Isaac worker alive can retain several gigabytes of GPU memory and interfere with other simulator or training processes.

## Implementation checklist

- [ ] Checkout the pinned UniVTAC commit.
- [ ] Install and verify the pinned simulator dependencies.
- [ ] Accept the Omniverse EULA.
- [ ] Launch Isaac before importing task modules.
- [ ] Use one environment on `cuda:0` with cameras enabled.
- [ ] Set physics to 120 Hz with decimation 1.
- [ ] Enable scene-query support.
- [ ] Select `gsmini` and verify both gel-pad attachments.
- [ ] Disable random textures for deterministic evaluation setup.
- [ ] Request only the observation fields needed by the caller.
- [ ] Reset with an explicit seed and reject failed pre-move plans.
- [ ] Preserve the task's native success and early-stop logic.
- [ ] Record the full task configuration and reset metadata.
- [ ] Close the task and Isaac application on every exit path.
