import argparse
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry

from ik_assets import PERIOD, foot_path, motion_time, reference_at
from ik_task import bounded_reward
from train_ik import configuration


def make_env(assets, count=4, seconds=20):
    config = configuration(assets, Path.home() / "UniLab")
    values = OmegaConf.to_container(config.env, resolve=True)
    values["reward_config"] = OmegaConf.to_container(config.reward, resolve=True)
    values["max_episode_seconds"] = seconds
    return registry.make("G1IKTracking23", num_envs=count, sim_backend="mujoco", env_cfg_override=values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    assert motion_time(0) == 0 and motion_time(1) == 0 and motion_time(2) == 0.5
    for left, right in zip(foot_path(0), foot_path(1)):
        np.testing.assert_allclose(left, right)
    rewards = bounded_reward(np.array([1, 0.5, 1]), np.array([1, 1, 1]), np.array([False, False, True]), 0.02)
    np.testing.assert_allclose(rewards, [0.02, 0.01, 0])
    for stopping in range(3):
        sequence = np.array([0.02, 0.01, 0.001]) * 0.99 ** np.arange(3)
        assert sequence[:stopping].sum() <= sequence.sum()
    env = make_env(args.assets)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (4, 178)
        anchor_ids = state.info["anchor_id"][:, 0]
        np.testing.assert_allclose(env.get_dof_pos(), env.anchor_qpos[anchor_ids, 7:], atol=2e-7)
        np.testing.assert_allclose(env.get_dof_vel(), env.anchor_qvel[anchor_ids, 6:], atol=2e-7)
        np.testing.assert_allclose(env.targets(state.info, np.zeros((4, 23)), advance=True), env.anchor_ctrl[anchor_ids], atol=1e-7)
        model = mujoco.MjModel.from_xml_path(str(args.assets / "flat.xml"))
        data = mujoco.MjData(model)
        maximum_error = 0.0
        for step in range(120):
            actions = np.zeros((4, 23), dtype=np.float32)
            before = env.get_physics_state_snapshot().copy()
            targets = env.targets(state.info, actions, advance=True)
            state = env.step(actions)
            assert np.isfinite(state.obs["obs"]).all() and np.all((state.reward >= 0) & (state.reward <= 0.020001))
            for episode in range(4):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, before[episode].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                data.ctrl[:] = targets[episode]
                for substep in range(10):
                    mujoco.mj_step(model, data)
                if not (state.terminated[episode] or state.truncated[episode]):
                    snapshot = np.empty(before.shape[1])
                    mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    maximum_error = max(maximum_error, float(np.max(np.abs(snapshot - env.get_physics_state_snapshot()[episode]))))
            if state.terminated.any():
                assert np.all(state.reward[state.terminated] == 0)
        assert maximum_error < 2e-4
        saved = env.get_physics_state_snapshot().copy()
        env._dr_manager.reset(np.array([0]))
        np.testing.assert_array_equal(saved[1:], env.get_physics_state_snapshot()[1:])
        invalid_pose = env.anchor_qpos[:1].copy()
        invalid_pose[:, 2] = 0.9
        env._backend.set_state(np.array([0]), invalid_pose, env.anchor_qvel[:1].copy())
        failed = env.step(np.zeros((4, 23)))
        assert failed.terminated[0] and failed.reward[0] == 0 and not failed.truncated[0]
        assert failed.final_observation is not None and failed.info["elapsed"][0, 0] == 0
        try:
            env.step(np.full((4, 23), np.nan))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN action accepted")
        print(f"PASS: loaded qpos/qvel reset, preload targets, 178d observation, nonnegative return counterexample, partial reset isolation, NaN rejection; native max error {maximum_error:.3g}")
    finally:
        env.close()
    env = make_env(args.assets, seconds=0.04)
    try:
        env.init_state()
        env.step(np.zeros((4, 23)))
        state = env.step(np.zeros((4, 23)))
        assert state.truncated.all() and not state.terminated.any() and state.final_observation is not None
        assert state.final_observation["obs"].shape == (4, 178)
        print("PASS: time limit is truncated with final observation, not a physical failure")
    finally:
        env.close()


if __name__ == "__main__":
    main()
