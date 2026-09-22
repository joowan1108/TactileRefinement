"""Run only through univtac_bridge after GPU availability/EULA preflight."""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import traceback
from pathlib import Path

from .sim_bridge import (
    send_message, receive_message, convert_observation, action_to_qpos, initial_state_digest,
)


def tactile_health(task):
    """Validate simulator state locally; no tactile values enter policy inputs."""
    import torch
    counts = {}
    for name, tactile in task._tactile_manager.tactiles.items():
        count = len(tactile.attachment.attachment_points_idx)
        if count < 3:
            raise RuntimeError(f"Invalid {name} attachment count: {count}")
        tensors = {
            "attachment_transform": tactile.attach_to_init,
            "gelpad_positions": tactile.gelpad.data.nodal_pos_w,
            "marker_motion": tactile.sensor.data.output["marker_motion"],
            "height_map": tactile.sensor.data.output["height_map"],
        }
        for label, value in tensors.items():
            if value.numel() == 0 or not torch.isfinite(value).all().item():
                raise RuntimeError(f"Invalid tactile simulator state: {name}/{label}")
        counts[name] = count
    if len(counts) != 2:
        raise RuntimeError(f"Expected two tactile sensors, got {list(counts)}")
    return {"finite": True, "attachment_counts": counts}


def main(*, refinement=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument('--controller', choices=('hold', 'linear', 'linear_position', 'official'), default='hold')
    args = parser.parse_args()
    if args.controller != 'hold' and not refinement:
        parser.error('Linear controller requires the two-step refinement benchmark worker')
    controller = None
    converter = convert_observation
    if refinement:
        from tactile_refinement.sim_bridge import convert_tactile_observation
        from tactile_refinement.execution import held_position_step
        if args.controller in ('linear', 'linear_position'):
            from tactile_refinement.execution import LinearPositionController
            controller = LinearPositionController(velocity_mode='zero' if args.controller == 'linear_position' else 'derived')
        converter = convert_tactile_observation
    channel = socket.socket(fileno=args.socket_fd)
    task = app = None
    try:
        # Isaac app must be initialized before envs imports.
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, enable_cameras=True, livestream=0, device="cuda:0")
        app = launcher.app
        import torch
        from .instructions import TASK_INSTRUCTIONS
        module = importlib.import_module(f"envs.{args.task}")
        cfg = module.TaskCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        # Headless Isaac Lab defaults this off; TacEx sphere sweeps need it
        # to find the case collider and attach the deformable gelpad.
        cfg.sim.enable_scene_query_support = True
        cfg.decimation = 1
        cfg.tactile_sensor_type = "gsmini"
        cfg.random_texture = False
        cfg.save_frequency = 0
        cfg.video_frequency = 0
        cfg.render_frequency = 0
        cfg.save_dir = Path(os.environ.get('UNIVTAC_EVAL_SAVE_ROOT', 'eval_result/smolvla_baseline')) / args.task
        # Physics and tactile sensors remain enabled for task success/early-stop
        # predicates. Only their observation payload is omitted from policy IPC.
        cfg.obs_data_type = {"camera": ["rgb"], "embodiment": ["joint"], "actor": True}
        if refinement:
            cfg.obs_data_type['tactile'] = ['depth']
        task = module.Task(cfg, mode="eval")
        send_message(channel, {"ok": True, "value": {"ready": True, "step_limit": cfg.step_lim}})
        while True:
            request = receive_message(channel)
            command = request["command"]
            if command == "close":
                break
            if command == "reset":
                task.reset(seed=request["seed"], instructions=[TASK_INSTRUCTIONS[args.task]])
                if not task.plan_success:
                    raise RuntimeError(f"UniVTAC pre-move failed for seed {request['seed']}; invalid initial state")
                raw = task._get_observations()
                if controller is not None:
                    controller.reset(torch.from_numpy(action_to_qpos(converter(raw, task.instruction)['state'])).to(task.device))
                value = (converter(raw, task.instruction), {
                    "initial_state_id": initial_state_digest(raw),
                    "initial_state_signature": "robot_joint_and_actor_pose_round_1e-5_not_full_uipc",
                    "tactile_health": tactile_health(task),
                })
            elif command == "step":
                qpos = torch.from_numpy(action_to_qpos(request["action"])).to(task.device)
                if refinement and args.controller != 'official':
                    step_controller = controller if controller is not None else held_position_step
                    executed, success, early_stop, truncated, physics_steps = step_controller(task, qpos)
                else:
                    before = task.step_count
                    executed, succeeded = task.take_action(qpos, action_type="qpos")
                    physics_steps = task.step_count-before
                    success = bool(succeeded)
                    early_stop = bool(task.check_early_stop()) if not success else False
                    truncated = task.take_action_cnt >= task.cfg.step_lim and not success
                raw = task._get_observations()
                value = (converter(raw, task.instruction), float(success),
                         success or early_stop, truncated,
                         {"success": success, "execution_success": bool(executed), "early_stop": early_stop,
                          "tactile_health": tactile_health(task)})
                if refinement:
                    value[-1].update(physics_steps=physics_steps,
                                     simulated_seconds=physics_steps*float(task.physics_dt),
                                     policy_action_count=int(task.take_action_cnt))
                    if controller is not None:
                        value[-1]['controller_commands'] = controller.last_commands
                else:
                    value[-1].update(physics_steps=physics_steps,
                                     simulated_seconds=physics_steps*float(task.physics_dt),
                                     policy_action_count=int(task.take_action_cnt))
            else:
                raise ValueError(f"unknown IPC command: {command}")
            send_message(channel, {"ok": True, "value": value})
    except BaseException:
        try:
            send_message(channel, {"ok": False, "error": traceback.format_exc()})
        except OSError:
            pass
        raise
    finally:
        if task is not None:
            task.close()
        if app is not None:
            app.close()
        channel.close()


if __name__ == "__main__":
    main()
