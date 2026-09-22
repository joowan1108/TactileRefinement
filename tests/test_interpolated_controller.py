from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tactile_refinement.execution import LinearPositionController


class Robot:
    def __init__(self):
        self.position = torch.zeros(8)
        self.velocity = torch.full((8,), 99.)  # Simulate stale pre-move velocities.
    def set_arm(self, pos, vel, force):
        assert force is True
        self.position[:7], self.velocity[:7] = pos, vel
    def set_gripper(self, pos, vel, force):
        assert force is True
        self.position[7], self.velocity[7] = pos, vel
    def get_qpos(self):
        return self.position[None].clone()


class Task:
    def __init__(self, success_at=None, early_at=None):
        self.cfg = SimpleNamespace(decimation=1, step_lim=600)
        self.physics_dt = 1/120
        self.step_count = self.take_action_cnt = 0
        self.eval_success = False
        self.plan_success = True
        self._robot_manager = Robot()
        self.history = []
        self.success_at, self.early_at = success_at, early_at
    def _step(self):
        self.history.append((self._robot_manager.position.clone(), self._robot_manager.velocity.clone()))
        self.step_count += 1
    def check_success(self):
        return self.step_count == self.success_at
    def check_early_stop(self):
        return self.step_count == self.early_at


def test_midpoint_endpoint_velocities_and_chunk_continuity():
    task = Task()
    controller = LinearPositionController()
    initial = torch.arange(8)*0.001
    controller.reset(initial)
    previous = initial.clone()
    # Cross a full chunk boundary, with direction changes and gripper motion.
    for i in range(51):
        target = initial+0.001*(1 if i % 2 else -1)
        result = controller(task, target)
        assert result == (True, False, False, False, 2)
        midpoint, endpoint = task.history[-2:]
        torch.testing.assert_close(midpoint[0], (previous+target)/2)
        torch.testing.assert_close(endpoint[0], target)
        for _, velocity in (midpoint, endpoint):
            torch.testing.assert_close(velocity, (target-previous)*60)
        previous = target.clone()
    assert task.take_action_cnt == 51 and task.step_count == 102
    controller.reset(initial)
    controller(task, initial)
    torch.testing.assert_close(task.history[-1][1], torch.zeros(8))


@pytest.mark.parametrize('event,substep', [('success', 1), ('success', 2), ('early', 1), ('early', 2)])
def test_terminal_checks_at_each_substep(event, substep):
    task = Task(**{event+'_at': substep})
    controller = LinearPositionController(); controller.reset(torch.zeros(8))
    result = controller(task, torch.ones(8))
    assert result[1] == (event == 'success')
    assert result[2] == (event == 'early')
    assert result[-1] == substep
    assert task.take_action_cnt == 1
    assert controller(task, torch.ones(8))[-1] == 0
    assert task.take_action_cnt == 1


def test_action_budget_and_invalid_input():
    task = Task(); task.cfg.step_lim = 1
    controller = LinearPositionController(); controller.reset(torch.zeros(8))
    assert controller(task, torch.ones(8))[3] is True
    assert controller(task, torch.ones(8))[-1] == 0
    assert task.step_count == 2 and task.take_action_cnt == 1
    with pytest.raises(ValueError):
        controller(task, torch.full((8,), float('nan')))
    task.cfg.decimation = 2
    with pytest.raises(ValueError):
        controller(task, torch.ones(8))


def test_zero_velocity_mode_replaces_stale_targets_on_both_substeps():
    task = Task()
    controller = LinearPositionController(velocity_mode='zero')
    controller.reset(torch.zeros(8))
    controller(task, torch.full((8,), 0.002))
    torch.testing.assert_close(task.history[0][0], torch.full((8,), 0.001))
    torch.testing.assert_close(task.history[1][0], torch.full((8,), 0.002))
    for _, velocity in task.history:
        torch.testing.assert_close(velocity, torch.zeros(8))
    # No EMA: even a direction reversal reaches the raw endpoint exactly.
    raw_endpoint = torch.full((8,), -0.003)
    original = raw_endpoint.clone()
    controller(task, raw_endpoint)
    torch.testing.assert_close(task.history[-1][0], original)
    torch.testing.assert_close(raw_endpoint, original)
    np.testing.assert_array_equal(controller.last_commands[-1]['observed_position'], original.numpy())
