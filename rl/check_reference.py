import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry

from prepare import ROOT
from reference_assets import joint_mapping, prepare_reference
from reference_task import body_to_world, world_to_body
from servo_force import position_servo_force


def make_env(assets, count=4, **overrides):
    config = OmegaConf.load(Path(__file__).with_name("reference.yaml"))
    config.env.seed = 123
    env_config = OmegaConf.to_container(config.env, resolve=True)
    env_config.update(scene={"model_file": str(assets / "flat.xml")}, reward_config=OmegaConf.to_container(config.reward), **overrides)
    return registry.make("G1CrawlReference23", num_envs=count, sim_backend="mujoco", env_cfg_override=env_config)


def audit_animation_geometry(env):
    frames = []
    model = env.force_model
    data = mujoco.MjData(model)
    for index, frame in enumerate(env.reset_frames):
        data.qpos[:] = frame
        data.qpos[7:] = np.clip(data.qpos[7:], env.joint_limits[:, 0], env.joint_limits[:, 1])
        mujoco.mj_forward(model, data)
        minimum = min(mujoco.mj_geomDistance(model, data, geom, env.floor_id, 1, None) for geom in env.floor_geoms)
        projection = max(0, 0.002 - minimum)
        data.qpos[2] += projection
        mujoco.mj_forward(model, data)
        self_penetration = max((-float(contact.dist) for contact in data.contact
                                if env.floor_id not in (contact.geom1, contact.geom2)), default=0.0)
        frames.append({"frame": index, "floor_projection_m": projection,
                       "joint_clip_rad": float(np.max(np.abs(frame[7:] - data.qpos[7:]))),
                       "self_penetration_m": max(0, self_penetration)})
    return {"frames": frames, "max_floor_projection_m": max(row["floor_projection_m"] for row in frames),
            "max_joint_clip_rad": max(row["joint_clip_rad"] for row in frames),
            "max_self_penetration_m": max(row["self_penetration_m"] for row in frames),
            "frames_with_self_penetration_over_3mm": sum(row["self_penetration_m"] > 0.003 for row in frames),
            "note": "3mm is a diagnostic flag, not a physical validity standard; no noisy-reset or dynamic feasibility guarantee"}


def main():
    paths = {variant: prepare_reference(variant) for variant in ("current", "reference")}
    models = [mujoco.MjModel.from_xml_path(str(path / "flat.xml")) for path in paths.values()]
    changed = [name for name in dir(models[0]) if isinstance(getattr(models[0], name), np.ndarray)
               and not np.array_equal(getattr(models[0], name), getattr(models[1], name))]
    assert set(changed) == {"actuator_gainprm", "actuator_biasprm"}, changed
    assert np.array_equal(models[0].actuator_gainprm[:, 1:], models[1].actuator_gainprm[:, 1:])
    assert np.array_equal(models[0].actuator_biasprm[:, [0, *range(3, 10)]], models[1].actuator_biasprm[:, [0, *range(3, 10)]])
    assert np.array_equal(np.load(paths["current"] / "reset_frames.npy"), np.load(paths["reference"] / "reset_frames.npy"))
    try:
        joint_mapping(["bad"] * 23, ["bad"] * 23)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate mapping accepted")
    quaternion = np.array([[np.sqrt(0.5), 0, np.sqrt(0.5), 0]])
    assert np.allclose(world_to_body(quaternion, [[1, 0, 0]]), [[0, 0, 1]])
    assert np.allclose(body_to_world(quaternion, [[-1, 0, 0]]), [[0, 0, 1]])
    report = {"changed_model_arrays": changed, "variants": {}}
    initial = []
    for variant, assets in paths.items():
        env = make_env(assets, actor_noise=False, reset_noise=False)
        try:
            state = env.init_state()
            initial.append(env.get_physics_state_snapshot().copy())
            assert list(state.obs) == ["obs", "critic"]
            assert state.obs["obs"].shape == (4, 75) and state.obs["critic"].shape == (4, 81)
            assert np.array_equal(state.obs["obs"], state.obs["critic"][:, 6:])
            assert "phase" not in state.info
            if variant == "current":
                report["animation_geometry"] = audit_animation_geometry(env)
            assert np.any(np.abs(env.get_dof_pos() - env._init_qpos[7:]) > 0.1)
            actions = np.linspace(-0.5, 0.5, 23, dtype=np.float32)[None].repeat(4, axis=0)
            targets = env.apply_action(actions * 20, state)
            assert np.max(np.abs(state.info["current_actions"])) == 5
            expected = env.default_angles.copy().astype(float)
            expected[env.policy_to_actuator] += np.clip(actions[0] * 20, -5, 5) * 0.5
            assert np.allclose(targets[0], np.clip(expected, env.ctrl_range[:, 0], env.ctrl_range[:, 1]))
            state = env.init_state()
            snapshot = env.get_physics_state_snapshot().copy()
            control = env.action_targets(actions)
            data = mujoco.MjData(env.force_model)
            mujoco.mj_setState(env.force_model, data, snapshot[0].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = control[0]
            for _ in range(env.cfg.sim_substeps):
                mujoco.mj_step(env.force_model, data)
            mujoco.mj_forward(env.force_model, data)
            state = env.step(actions)
            assert not state.terminated[0] and not state.truncated[0]
            native = np.empty(snapshot.shape[1])
            mujoco.mj_getState(env.force_model, data, native, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            state_error = float(np.max(np.abs(native - env.get_physics_state_snapshot()[0])))
            assert state_error < 1e-4, state_error
            forces = position_servo_force(env.force_model, control[0], data.qpos[7:], data.qvel[6:])
            force_error = float(np.max(np.abs(forces - data.qfrc_actuator[6:])))
            assert force_error < 1e-6
            assert np.allclose(state.obs["critic"][0, :3], world_to_body(data.qpos[None, 3:7], data.qvel[None, :3])[0], atol=1e-5)
            assert np.allclose(state.obs["critic"][0, 3:6], data.qvel[3:6], atol=1e-5)
            state.info["command_remaining"][:] = 1
            state.info["command_remaining"][0] = 0
            previous_commands = state.info["commands"].copy()
            state = env.step(np.zeros((4, 23), dtype=np.float32))
            assert np.array_equal(state.info["commands"][1:], previous_commands[1:])
            assert state.info["command_remaining"][0, 0] >= env.cfg.command_interval_min
            before = env.get_physics_state_snapshot()[1:].copy()
            observation, info = env.reset(np.array([0], dtype=np.int32))
            assert observation["obs"].shape == (1, 75) and observation["critic"].shape == (1, 81)
            assert np.array_equal(before, env.get_physics_state_snapshot()[1:])
            env.cfg.actor_noise = True
            clean = env.observations(info, np.array([0]))
            noisy = env.observations(info, np.array([0]))
            assert np.array_equal(clean["critic"], noisy["critic"])
            assert not np.array_equal(clean["obs"], noisy["obs"])
            assert np.array_equal(clean["obs"][:, 3:6], noisy["obs"][:, 3:6])
            assert np.array_equal(clean["obs"][:, 52:], noisy["obs"][:, 52:])
            env.cfg.max_episode_seconds = env.cfg.ctrl_dt
            env.init_state()
            final = env.step(np.zeros((4, 23), dtype=np.float32))
            assert final.truncated.all()
            assert final.final_observation["obs"].shape == (4, 75)
            assert final.final_observation["critic"].shape == (4, 81)
            assert not np.shares_memory(final.obs["critic"], final.final_observation["critic"])
            env.cfg.max_episode_seconds = 20.0
            env.cfg.reset_noise = True
            state = env.init_state()
            random = np.random.default_rng(9)
            resets = 0
            for _ in range(100):
                state = env.step(random.normal(0, 1, (4, 23)).astype(np.float32))
                assert all(np.isfinite(value).all() for value in state.obs.values())
                assert np.isfinite(state.reward).all()
                resets += int(np.sum(state.terminated))
            try:
                env.step(np.full((4, 23), np.nan))
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite actions accepted")
            report["variants"][variant] = {"native_state_error": state_error, "native_force_error_nm": force_error,
                "random_transitions": 400, "safety_resets": resets, "reset_isolation": "passed", "timeout_final_observation": "passed",
                "last_reset_floor_projection_m": state.info["reset_floor_projection"].reshape(-1).tolist()}
        finally:
            env.close()
    assert np.array_equal(*initial)
    report["paired_reset_states_identical"] = True
    path = ROOT / "output/rl/reference_contract_check.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**report, "animation_geometry": {key: value for key, value in report["animation_geometry"].items() if key != "frames"}}, indent=2))


if __name__ == "__main__":
    main()
