import tempfile
from pathlib import Path

import numpy as np
import torch

from stand_transition_task import StandTransitionTask, progress_reward
from train_near_handoff import ActorCritic


def main():
    torch.set_num_threads(1)
    for path in ([0., .1, .2], [0., .5, 2.], [0., -.2, -.4]):
        rewards = [progress_reward(path[0], path[1]), progress_reward(path[1], path[2]),
                   progress_reward(path[2], 0., True, -5.)]
        np.testing.assert_allclose(sum(.99 ** index * reward for index, reward in enumerate(rewards)),
                                   -5. * .99 ** 2, atol=1e-12)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    observation = task.reset(state)
    assert observation.shape == (111,)
    np.testing.assert_array_equal(observation[-23:], np.zeros(23))
    task.integrated[0] += .1
    changed = task.observation()
    np.testing.assert_array_equal(changed[:88], observation[:88])
    np.testing.assert_allclose(changed[-23], .1)
    observation = task.reset(state)
    np.testing.assert_array_equal(observation[-23:], np.zeros(23))
    for size in (88, 111):
        network = ActorCritic(size)
        assert network.actor(torch.zeros(size)).shape == (23,)
        assert network.critic(torch.zeros(size)).shape == (1,)
        optimizer = torch.optim.Adam(network.parameters(), lr=.0003)
        distribution = network.distribution(torch.zeros((2, size)))
        actions = distribution.sample()
        loss = -distribution.log_prob(actions).mean() + network.critic(torch.zeros((2, size))).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "network.pt"
            torch.save(network.state_dict(), checkpoint)
            restored = ActorCritic(size)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            torch.testing.assert_close(restored.actor(torch.zeros(size)), network.actor(torch.zeros(size)))
    assert task.data.qpos[2] > .7 and observation[55] > .7 and observation[56] > .7
    assert all(task.measure()[1].values()) and not task.ready()
    result = task.run_baseline()
    assert not result["success"] and result["failure"] == ["timeout_without_readiness"]
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "standing_hold.npz")
    task.reset(state)
    for unused in range(300):
        observation, reward, done, result = task.step(np.zeros(23))
        assert np.isfinite(reward) and np.isfinite(observation).all()
        if done:
            break
    assert done and not result["success"] and not result["any_hand_loaded"]
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "lowering_seed.npz")
    task.reset(state)
    synthetic = np.zeros((51, 15))
    synthetic[:, 0] = np.arange(51) * .002
    synthetic[:, 3] = .3
    synthetic[:, 4:8] = 10.
    task.samples = synthetic.tolist()
    task.data.qvel[:] = 0.
    task.data.qpos[task.addresses] = task.crawl_target
    assert task.ready()
    task.samples[-1][4] = 0.
    assert not task.ready()
    task.samples = synthetic[:50].tolist()
    assert not task.ready()
    task.samples = synthetic.tolist()
    task.data.qvel[0] = 1.
    assert not task.ready()
    print("Standing task: true standing reset, visible hand heights, no false hold/lowering success, continuous load gate and replay passed")


if __name__ == "__main__":
    main()
