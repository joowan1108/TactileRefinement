"""Evaluation controllers; reset/pre-move always keeps its original controller."""


class LinearPositionController:
    """Interpolate 60 Hz endpoints at 120 Hz using explicit position/velocity targets.

    The start is the observed reset pose, then the preceding commanded endpoint.
    This reconstructs a linear path, not the unavailable original planner path.
    """
    def __init__(self, velocity_mode='derived'):
        if velocity_mode not in ('derived', 'zero'):
            raise ValueError('Unknown velocity target mode')
        self.velocity_mode = velocity_mode

    def reset(self, initial_qpos):
        import torch
        if initial_qpos.shape != (8,) or not torch.isfinite(initial_qpos).all():
            raise ValueError('Expected seven arm joints and one shared gripper position')
        self.previous = initial_qpos.detach().clone()
        self.last_commands = []

    def __call__(self, task, qpos):
        import math
        import torch
        if task.cfg.decimation != 1 or not math.isfinite(task.physics_dt) or task.physics_dt <= 0:
            raise ValueError('Expected decimation one and a positive finite physics timestep')
        if qpos.shape != (8,) or not torch.isfinite(qpos).all():
            raise ValueError('Expected eight finite joint targets')
        self.last_commands = []
        success = bool(task.eval_success)
        early = bool(task.check_early_stop()) if not success else False
        if success or early or task.take_action_cnt >= task.cfg.step_lim:
            return True, success, early, task.take_action_cnt >= task.cfg.step_lim and not success, 0
        if not task.plan_success:
            return False, False, True, False, 0
        start = self.previous.to(qpos)
        target = qpos.detach().clone()
        velocity = ((target-start)/(2*float(task.physics_dt)) if self.velocity_mode == 'derived'
                    else torch.zeros_like(target))
        task.take_action_cnt += 1  # Two substeps still consume exactly one policy action.
        before = task.step_count
        for fraction in (0.5, 1.0):
            position = (start+target)*0.5 if fraction == 0.5 else target
            # Same setters/force behavior as demonstration take_dense_action.
            task._robot_manager.set_arm(position[:-1], velocity[:-1], force=True)
            task._robot_manager.set_gripper(position[-1], velocity[-1], force=True)
            prior = task.step_count
            task._step()
            if task.step_count != prior+1:
                raise RuntimeError('Interpolated controller failed to advance exactly one physics step')
            self.last_commands.append({'position': position.cpu().numpy().copy(),
                                       'velocity': velocity.cpu().numpy().copy(),
                                       'observed_position': task._robot_manager.get_qpos()[0, :8].cpu().numpy().copy()})
            self.previous = position.detach().clone()
            success = bool(task.check_success())
            if success:
                task.eval_success = True
            early = bool(task.check_early_stop()) if not success else False
            if success or early:
                break
        return True, success, early, task.take_action_cnt >= task.cfg.step_lim and not success, task.step_count-before


def held_position_step(task, qpos, hold_steps=2):
    if hold_steps != 2 or task.cfg.decimation != 1:
        raise ValueError('Expected two physical steps with task decimation one')
    before = task.step_count
    executed, success = task.take_action(qpos, action_type='qpos')
    early = bool(task.check_early_stop()) if not success else False
    if not success and not early and executed and task.step_count > before:
        # Keep existing position targets; do not consume another policy action.
        task._step()
        success = bool(task.check_success())
        if success:
            task.eval_success = True
        early = bool(task.check_early_stop()) if not success else False
    physics_steps = task.step_count-before
    if physics_steps not in (0, 1, 2):
        raise RuntimeError(f'Unexpected physics advancement: {physics_steps}')
    truncated = task.take_action_cnt >= task.cfg.step_lim and not success
    return bool(executed), bool(success), early, truncated, physics_steps
