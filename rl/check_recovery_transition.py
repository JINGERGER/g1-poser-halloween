import tempfile
from pathlib import Path

import numpy as np
import torch

from recovery_transition_task import RecoveryTransitionTask, paired_action
from stand_transition_task import UnsafeState


def main():
    torch.set_num_threads(1)
    task = RecoveryTransitionTask()
    state = task.make_states([0])[0]
    raw = np.zeros(23)
    left = task.names.index("left_hip_roll_joint")
    right = task.names.index("right_hip_roll_joint")
    raw[left] = 1.
    common = paired_action(raw, task.names)
    np.testing.assert_allclose(common[left], -common[right])
    raw[right] = 1.
    independent = paired_action(raw, task.names)
    assert abs(independent[left]) != abs(independent[right])
    observation = task.reset(state)
    assert observation.shape == (139,) and not task.training_mode
    original_measure = task.measure
    row, gates, penetration = task.measure()
    gates["non_support_load"] = False
    row[12] = 2.
    task.measure = lambda: (row, gates, penetration)
    try:
        task._observe_step(task.model, task.data)
    except UnsafeState:
        pass
    else:
        raise AssertionError("Strict evaluation did not stop on violation")
    task.training_mode = True
    task._observe_step(task.model, task.data)
    assert "non_support_load" in task.violations and task.violation_steps == 2
    assert task.observation()[-1] == 1.
    task.measure = original_measure
    task.reset(state)
    assert not task.violations and task.observation()[-1] == 0.
    task.violations.add("non_support_load")
    actual_ready, actual_finish = task.ready, task.finish
    task.ready = lambda: True
    task.finish = lambda: {"success": True, "failure": []}
    unused_obs, unused_reward, done, result = task.step(np.zeros(23))
    assert done and not result["success"] and result["recovery_only"]
    task.ready, task.finish = actual_ready, actual_finish
    for training in (False, True):
        task.training_mode = training
        task.reset(state)
        for unused in range(task.horizon):
            observation, reward, done, result = task.step(np.ones(23) * .5)
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if done:
                break
        assert done
        if task.violations:
            assert not result["success"]
        with tempfile.TemporaryDirectory() as directory:
            task.save_and_replay(Path(directory) / "trajectory.npz")
    print("Recovery learning: bilateral freedom, observable prefix violations, strict stop, recovery != success and replay passed")


if __name__ == "__main__":
    main()
