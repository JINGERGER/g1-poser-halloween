import tempfile
from pathlib import Path

import numpy as np
import torch

from closed_loop_transition_task import ClosedLoopTransitionTask


def main():
    torch.set_num_threads(1)
    for prior in (True, False):
        task = ClosedLoopTransitionTask(prior)
        state = task.make_states([0])[0]
        observation = task.reset(state)
        assert observation.shape == (161,)
        task.residual[0] = .2
        assert task.observation()[111] != observation[111]
        task.raw_previous[0] = .3
        assert task.observation()[134] != observation[134]
        np.testing.assert_array_equal(task.reset(state), observation)
        for unused in range(task.horizon):
            observation, reward, done, result = task.step(np.zeros(23))
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if done:
                break
        assert done
        if task.failure:
            assert not result["success"]
        with tempfile.TemporaryDirectory() as directory:
            task.save_and_replay(Path(directory) / "trajectory.npz")
    print("Closed-loop: 161 observations, filter/history visibility, reset, finite rollout and physical replay passed")


if __name__ == "__main__":
    main()
