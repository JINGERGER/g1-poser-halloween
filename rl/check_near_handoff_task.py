import tempfile
from pathlib import Path

import numpy as np
import torch

from near_handoff_task import NearHandoffTask
from train_near_handoff import ActorCritic, advantages
from transition_execution import physical_state


def main():
    torch.set_num_threads(1)
    np.testing.assert_allclose(advantages([1., 2.], [0., 0.], [0., 1.], 100.), [1. + .99 * .95 * 2., 2.])
    np.testing.assert_allclose(advantages([1., 2.], [0., 0.], [1., 1.], 100.), [1., 2.])
    network = ActorCritic()
    distribution = network.distribution(torch.zeros(88))
    assert distribution.sample().shape == (23,) and torch.isfinite(distribution.log_prob(torch.zeros(23))).all()
    task = NearHandoffTask()
    states = task.make_states([0, 1000])
    assert states[0]["seed"] != states[1]["seed"]
    hand_states = task.make_states(range(3), "hands")
    assert {state["mode"] for state in hand_states} == {"left", "right", "left+right"}
    for state in hand_states:
        np.testing.assert_array_equal(np.array(state["qpos"])[:19], task.nominal[:19])
        assert all(.0095 <= gap <= .0305 for gap in state["actual_hand_gaps_m"].values())
    observation = task.reset(states[0])
    assert observation.shape == (88,)
    initial = physical_state(task.model, task.data)
    for bad in (np.zeros(12), np.full(23, np.nan)):
        try:
            task.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action accepted")
        np.testing.assert_array_equal(physical_state(task.model, task.data), initial)
    result = task.finish()
    assert task.execution.data is task.data
    assert task.execution.model is task.model
    assert len(task.positions) == len(task.controls) + 1
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "direct.npz")
    task.reset(states[0])
    for unused in range(20):
        observation, reward, done, result = task.step(np.zeros(23))
        assert np.isfinite(reward) and np.isfinite(observation).all()
        if done:
            break
    assert done
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "transition.npz")
    print("Near-handoff: GAE terminal masks, observation/action contract, physical continuity and replay checks passed")


if __name__ == "__main__":
    main()
