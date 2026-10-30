import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry

import anchored_foot
import crawl_task
import direct_task
import single_foot


def audit(task, assets, force_source=None):
    config = OmegaConf.to_container(OmegaConf.load(Path(__file__).with_name("flat.yaml")), resolve=True)
    overrides = {**config["env"], "scene": {"model_file": str(assets / "flat.xml")}, "reset_joint_noise": 0.0, "reward_config": config["reward"]}
    if force_source is not None:
        overrides["force_source"] = force_source
    env = registry.make(task, num_envs=1, sim_backend="mujoco", env_cfg_override=overrides)
    model = mujoco.MjModel.from_xml_path(str(assets / "flat.xml"))
    data = mujoco.MjData(model)
    report = {"task": task, "observation_size": env.obs_groups_spec["obs"], "samples": 0,
              "force_source": force_source or ("analytic_pd" if task == "G1AnchoredFoot23V5" else "not_used" if task == "G1CrawlFlat23" else "legacy_sensor"),
              "max_state_error": 0.0, "max_joint_observation_error": 0.0,
              "max_velocity_observation_error": 0.0,
              "max_gravity_observation_error": 0.0, "contact_flag_mismatches": 0,
              "max_force_error_nm": None, "ended_early": False}
    joint_ids = model.actuator_trnid[:, 0]
    assert model.nu == 23 and model.nq == 30 and model.nv == 29
    assert np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 30))
    assert np.array_equal(model.jnt_dofadr[joint_ids], np.arange(6, 29))
    assert np.isclose(env.cfg.ctrl_dt, model.opt.timestep * env.cfg.sim_substeps)
    try:
        state = env.init_state()
        random = np.random.default_rng(2026)
        for step in range(50):
            actions = random.uniform(-0.15, 0.15, (1, 23)).astype(np.float32)
            before = np.asarray(env.get_physics_state_snapshot()[0], dtype=np.float64)
            mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, before, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = np.clip(env.default_angles + env.cfg.control_config.action_scale * actions[0], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
            for substep in range(env.cfg.sim_substeps):
                mujoco.mj_step(model, data)
            state = env.step(actions)
            if state.terminated[0] or state.truncated[0]:
                report["ended_early"] = True
                break
            native = np.zeros_like(before)
            mujoco.mj_getState(model, data, native, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            actual = env.get_physics_state_snapshot()[0]
            report["max_state_error"] = max(report["max_state_error"], float(np.max(np.abs(native - actual))))
            mujoco.mj_setState(model, data, np.asarray(actual, dtype=np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            report["max_joint_observation_error"] = max(report["max_joint_observation_error"], float(np.max(np.abs(state.obs["obs"][0, 9:32] - (data.qpos[7:] - env.default_angles)))))
            expected_velocity = np.r_[data.qvel[:3], data.sensor("gyro_torso").data * 0.25, data.qvel[6:] * 0.05]
            actual_velocity = np.r_[state.obs["obs"][0, :6], state.obs["obs"][0, 32:55]]
            report["max_velocity_observation_error"] = max(report["max_velocity_observation_error"], float(np.max(np.abs(expected_velocity - actual_velocity))))
            assert np.array_equal(state.obs["obs"][0, 55:78], actions[0])
            assert np.allclose(state.obs["obs"][:, 78:79], state.info["commands"])
            assert np.allclose(state.obs["obs"][:, 79:81], np.concatenate((np.sin(state.info["phase"] * 2 * np.pi), np.cos(state.info["phase"] * 2 * np.pi)), axis=1))
            expected_gravity = direct_task.projected_gravity(data.qpos[None, 3:7])[0] if isinstance(env, direct_task.DirectEnv) else data.sensor("upvector_torso").data
            report["max_gravity_observation_error"] = max(report["max_gravity_observation_error"], float(np.max(np.abs(state.obs["obs"][0, 6:9] - expected_gravity))))
            expected_contacts = np.array([data.sensor(f"crawl_contact_{name}").data[0] for name in crawl_task.SITES]) > 0
            report["contact_flag_mismatches"] += int(np.count_nonzero(expected_contacts != (state.obs["obs"][0, 81:85] > 0)))
            if isinstance(env, direct_task.DirectEnv):
                expected_force = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
                error = float(np.max(np.abs(env.joint_forces(state)[0] - expected_force)))
                report["max_force_error_nm"] = max(report["max_force_error_nm"] or 0, error)
            report["samples"] += 1
        report["kinematics_dynamics_passed"] = bool(report["samples"] == 50 and report["max_state_error"] < 2e-4 and report["max_joint_observation_error"] < 1e-5 and report["max_gravity_observation_error"] < 1e-5 and report["max_velocity_observation_error"] < 1e-5 and report["contact_flag_mismatches"] == 0)
        report["force_path_passed"] = report["max_force_error_nm"] < 1e-3 if report["max_force_error_nm"] is not None else None
        return report
    finally:
        env.close()


def audit_force_only(assets):
    config = OmegaConf.to_container(OmegaConf.load(Path(__file__).with_name("flat.yaml")), resolve=True)
    config["reward"]["scales"].update(torque_utilization=-0.05, torque_near_limit=-2.0, non_support_contact=-5.0)
    overrides = {**config["env"], "scene": {"model_file": str(assets / "flat.xml")},
                 "reset_joint_noise": 0.0, "reward_config": config["reward"], "control_config": {"action_scale": 0.5}}
    environments = []
    reward_error = 0.0
    try:
        for source in ("legacy_sensor", "analytic_pd"):
            environments.append(registry.make("G1CrawlDirect23V2", num_envs=1, sim_backend="mujoco",
                                               env_cfg_override={**overrides, "force_source": source}))
        states = [env.init_state() for env in environments]
        random = np.random.default_rng(20261015)
        for step in range(50):
            actions = random.uniform(-0.15, 0.15, (1, 23)).astype(np.float32)
            states = [env.step(actions) for env in environments]
            assert np.array_equal(environments[0].get_physics_state_snapshot(), environments[1].get_physics_state_snapshot())
            assert np.array_equal(states[0].obs["obs"], states[1].obs["obs"])
            assert not any(state.terminated.any() or state.truncated.any() for state in states)
            terms = [env.additional_terms(state)[0] for env, state in zip(environments, states)]
            expected = sum(config["reward"]["scales"][name] * (terms[1][name] - terms[0][name])
                           for name in ("torque_utilization", "torque_near_limit")) * environments[0].cfg.ctrl_dt
            error = float(np.max(np.abs(states[1].reward - states[0].reward - expected)))
            reward_error = max(reward_error, error)
            assert error < 1e-6
        return {"samples": 50, "physics_and_observations_identical": True,
                "max_force_only_reward_delta_error": reward_error}
    finally:
        for env in environments:
            env.close()


def main():
    assets = direct_task.prepare_direct(3)
    reports = [audit(task, crawl_task.ASSETS if task == "G1CrawlFlat23" else assets) for task in (
        "G1CrawlFlat23", "G1CrawlDirect23V2", "G1CrawlBalanced23V3", "G1SingleFoot23V4", "G1AnchoredFoot23V5")]
    reports.append(audit("G1CrawlDirect23V2", assets, "analytic_pd"))
    output = crawl_task.ROOT / "output/rl/integrity_check.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if not all(report["kinematics_dynamics_passed"] for report in reports) or not all(report["force_path_passed"] for report in reports if report["force_source"] == "analytic_pd"):
        raise AssertionError("Core integrity check failed; inspect report before training")
    paired = audit_force_only(assets)
    (output.with_name("force_only_comparison.json")).write_text(json.dumps(paired, indent=2) + "\n")
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
