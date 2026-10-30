from pathlib import Path
import argparse
import json
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry

import crawl_task
import direct_task
import single_foot
import anchored_foot


def main():
    parser = argparse.ArgumentParser(description="Validate crawl physics, batched reset, and zero-action baseline")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--foot", choices=("left", "right"))
    parser.add_argument("--anchored", action="store_true")
    args = parser.parse_args()
    if args.anchored and not args.foot:
        parser.error("--anchored requires --foot left or right")
    if args.foot:
        args.balanced = True
    if args.balanced:
        args.direct = True
    assets = direct_task.prepare_direct(3 if args.balanced else 2) if args.direct else crawl_task.ASSETS
    model = mujoco.MjModel.from_xml_path(str(assets / "flat.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    contacts = [{
        "geom1": model.geom(int(contact.geom1)).name,
        "geom2": model.geom(int(contact.geom2)).name,
        "distance": float(contact.dist),
    } for contact in data.contact]
    config = OmegaConf.to_container(OmegaConf.load(Path(__file__).with_name("flat.yaml")), resolve=True)
    if args.direct:
        config["env"]["scene"] = {"model_file": str(assets / "flat.xml")}
        config["reward"]["scales"].update(torque_utilization=-0.05, torque_near_limit=-2.0, non_support_contact=-5.0)
        for quaternion in ([1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]):
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, np.array(quaternion, dtype=float))
            assert np.allclose(direct_task.projected_gravity(np.array([quaternion]))[0], -rotation.reshape(3, 3)[2])
        original = mujoco.MjModel.from_xml_path(str(crawl_task.ASSETS / "flat.xml"))
        for attribute in ("body_mass", "body_inertia", "geom_size", "jnt_actfrcrange", "actuator_gainprm"):
            assert np.array_equal(getattr(original, attribute), getattr(model, attribute))
        probe = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, probe, model.key("home").id)
        probe.ctrl[:] += 0.1
        mujoco.mj_forward(model, probe)
        for index, joint_id in enumerate(model.actuator_trnid[:, 0]):
            sensor_force = probe.sensor(f"direct_torque_{index}").data[0]
            assert np.isclose(sensor_force, probe.qfrc_actuator[model.jnt_dofadr[joint_id]])
        if args.balanced:
            for index, name in enumerate(direct_task.SUPPORTS):
                gap = mujoco.mj_geomDistance(model, probe, model.geom(name).id, model.geom("floor").id, 0.5, None)
                assert np.isclose(probe.sensor(f"balanced_gap_{index}").data[0], gap)
            initial_gaps = np.array([probe.sensor(f"balanced_gap_{index}").data[0] for index in range(4)])
            probe.qpos[2] += 0.05
            mujoco.mj_forward(model, probe)
            raised_gaps = np.array([probe.sensor(f"balanced_gap_{index}").data[0] for index in range(4)])
            assert np.allclose(raised_gaps - initial_gaps, 0.05)
            for limb in range(4):
                phase = np.full((1, 4), 0.4)
                phase[0, limb] = 0.875
                gaps = np.zeros((1, 4))
                gaps[0, limb] = 0.02
                sample_contacts = np.ones((1, 4), dtype=bool)
                sample_contacts[0, limb] = False
                ideal = direct_task.balanced_terms(sample_contacts, gaps, phase, 0.02, np.array([True]))
                grounded = direct_task.balanced_terms(np.ones_like(sample_contacts), np.zeros_like(gaps), phase, 0.02, np.array([True]))
                assert ideal["swing_gap"][0] == 1 and ideal["stance_contact"][0] == 1
                assert ideal["swing_gap"][0] > grounded["swing_gap"][0]
                assert ideal["swing_contact"][0] == 0 and grounded["swing_contact"][0] == 1
                hover = direct_task.balanced_terms(sample_contacts, gaps, np.full((1, 4), 0.05), 0.02, np.array([True]))
                assert hover["stance_hover"][0] > 0 and hover["landing_contact"][0] == 3
    task = "G1CrawlDirect23V2" if args.direct else "G1CrawlFlat23"
    if args.balanced:
        task = "G1CrawlBalanced23V3"
        config["reward"]["scales"].update(support=0.0, clearance=0.0, swing_gap=4.0, swing_contact=-2.0, stance_contact=2.0, stance_hover=-1.0, landing_contact=1.0)
    if args.foot:
        task = "G1SingleFoot23V4"
        config["env"]["target_foot"] = 2 if args.foot == "left" else 3
        for selected in (2, 3):
            phase = single_foot.single_foot_phase(np.array([[0.875], [0.05]]), selected)
            assert phase[0, selected] >= 0.75 and phase[1, selected] < 0.1
            assert np.all(phase[:, [index for index in range(4) if index != selected]] == np.float32(0.4))
    if args.anchored:
        task = "G1AnchoredFoot23V5"
        config["reward"]["scales"].update(anchor_drift=-0.5, base_drift=-1.0)
    env = registry.make(task, num_envs=args.envs, sim_backend="mujoco", env_cfg_override={**config["env"], "reward_config": config["reward"]})
    failures = np.zeros(args.envs, dtype=int)
    timeouts = np.zeros(args.envs, dtype=int)
    start = time.perf_counter()
    try:
        state = env.init_state()
        dimensions = env.obs_groups_spec["obs"]
        assert state.obs["obs"].shape == (args.envs, dimensions)
        if args.anchored:
            assert np.allclose(state.obs["obs"][:, 85:], 0)
        if args.foot:
            assert env.gait_active(state.info).all()
            assert np.all(state.info["commands"] == 0)
        actions = np.zeros((args.envs, 23), dtype=np.float32)
        reward_sum = 0.0
        for step in range(args.steps):
            state = env.step(actions)
            if args.anchored and step % 10 == 0 and not state.terminated[0] and not state.truncated[0]:
                mujoco.mj_setState(model, probe, np.asarray(env.get_physics_state_snapshot()[0], dtype=np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                probe.ctrl[:] = env.default_angles
                mujoco.mj_forward(model, probe)
                expected = probe.qfrc_actuator[model.jnt_dofadr[model.actuator_trnid[:, 0]]]
                assert np.allclose(env.joint_forces(state)[0], expected, atol=1e-3)
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.reward).all()
            failures += state.terminated
            timeouts += state.truncated
            reward_sum += float(np.mean(state.reward))
        report = {
            "task": task,
            "target_foot": args.foot,
            "initial_contacts": contacts,
            "num_envs": args.envs,
            "steps_per_env": args.steps,
            "simulated_seconds_per_env": args.steps * env.cfg.ctrl_dt,
            "failures_per_env": failures.tolist(),
            "timeouts_per_env": timeouts.tolist(),
            "mean_total_reward": reward_sum,
            "transitions_per_second": args.steps * args.envs / (time.perf_counter() - start),
            "last_metrics": state.info["log"],
        }
        untouched = env.get_physics_state_snapshot()[1:].copy()
        observation, info = env.reset(np.array([0], dtype=np.int32))
        assert observation["obs"].shape == (1, dimensions)
        if args.anchored:
            assert np.allclose(observation["obs"][:, 85:], 0)
            assert np.allclose(info["support_anchors"], env.site_xy()[[0]])
        assert np.array_equal(untouched, env.get_physics_state_snapshot()[1:])
        report["batch_reset_check"] = "passed"
        env.cfg.max_episode_seconds = env.cfg.ctrl_dt
        env.init_state()
        final = env.step(actions)
        assert final.truncated.all() and not final.terminated.any()
        assert final.final_observation["obs"].shape == (args.envs, dimensions)
        report["timeout_and_final_observation_check"] = "passed"
        try:
            env.step(np.full_like(actions, np.nan))
        except ValueError:
            report["nonfinite_action_rejected"] = True
        else:
            raise AssertionError("Nonfinite actions must be rejected before stepping physics")
        output = crawl_task.ROOT / "output" / "rl" / ("balanced_zero_action_check.json" if args.balanced else "direct_zero_action_check.json" if args.direct else "zero_action_check.json")
        if args.foot:
            output = output.with_name(f"{'anchored_' if args.anchored else ''}{args.foot}_foot_zero_action_check.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        output.with_name(f"{task}_{args.foot or 'all'}_zero_action_{args.envs}env_{args.steps}steps.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
