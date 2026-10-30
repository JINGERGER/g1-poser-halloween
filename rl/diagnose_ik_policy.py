import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cancellation(reference, residual):
    if len(reference) < 3:
        return None
    reference = np.asarray(reference) - np.mean(reference, axis=0)
    residual = np.asarray(residual) - np.mean(residual, axis=0)
    energy = np.sum(reference ** 2)
    if energy < 1e-10:
        return None
    return {"opposing_projection": float(-np.sum(reference * residual) / energy),
            "combined_ac_ratio": float(np.sqrt(np.sum((reference + residual) ** 2) / energy))}


def run(checkpoint, mode, episodes):
    import mlx.core as mx
    from evaluate_ik import load, sha256
    from direct_task import projected_gravity
    from ik_assets import PERIOD, foot_path, motion_time
    from prepare import SITES
    from servo_force import position_servo_force

    env, policy, config = load(checkpoint, episodes, 1001)
    try:
        state = env.init_state()
        initial = env.get_physics_state_snapshot().copy()
        anchors = state.info["anchor_id"][:, 0].copy()
        origins = env._backend.get_base_pos()[:, :2].copy()
        observation = mx.array(state.obs["obs"])
        eval_mean = np.asarray(policy.policy(observation)).copy()
        policy.train()
        train_mean = np.asarray(policy.policy(observation)).copy()
        np.testing.assert_array_equal(eval_mean, train_mean)
        policy.eval()
        mx.random.seed(91001)
        if mode == "hold":
            env.targets = lambda info, actions, advance=False: np.clip(env.anchor_ctrl[info["anchor_id"][:, 0]], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
        captured = {}
        original = env.update_state

        def capture(current):
            updated = original(current)
            info = current.info
            anchor_ids = info["anchor_id"][:, 0]
            elapsed = info["elapsed"][:, 0]
            reference = env.references(info)
            residual = .5 * info["current_actions"]
            target = env.targets(info, info["current_actions"])
            positions = np.stack([env._backend.get_sensor_data(f"crawl_pos_{name}") for name in SITES], axis=1)
            velocities = np.stack([env._backend.get_sensor_data(f"crawl_vel_{name}") for name in SITES], axis=1)
            contacts = np.column_stack([env._backend.get_sensor_data(f"crawl_contact_{name}").reshape(-1) > 0 for name in SITES])
            bad = np.column_stack([env._backend.get_sensor_data(name).reshape(-1) > 0 for name in env.bad_contacts])
            gravity_error = np.sum((projected_gravity(env._backend.get_base_quat()) - projected_gravity(env.anchor_qpos[anchor_ids, 3:7])) ** 2, axis=1)
            height = env._backend.get_base_pos()[:, 2]
            _, desired_height, swing = foot_path(motion_time(elapsed) / PERIOD % 1)
            pose = np.exp(-np.mean((env.get_dof_pos() - env.anchor_qpos[anchor_ids, 7:] - reference) ** 2, axis=1) / .3 ** 2)
            clearance = np.exp(-np.mean((positions[:, :, 2] - env.site_heights[anchor_ids] - desired_height) ** 2, axis=1) / .015 ** 2)
            contact = np.mean(contacts == ~(swing & (elapsed[:, None] > 1)), axis=1)
            linear = env._backend.get_base_lin_vel()[:, :2]
            velocity = np.exp(-np.sum((linear - env.heading[anchor_ids] * env.command(info)[:, None]) ** 2, axis=1) / .03 ** 2)
            forces = position_servo_force(env.force_model, target, env.get_dof_pos(), env.get_dof_vel())
            ratios = np.abs(forces) / np.where(forces >= 0, env.force_limits[:, 1], -env.force_limits[:, 0])
            slip = np.mean(np.sum(velocities[:, :, :2] ** 2, axis=2) * contacts, axis=1)
            quality = np.exp(-gravity_error / .25 - (height - env.anchor_qpos[anchor_ids, 2]) ** 2 / .05 ** 2 - slip / .05 ** 2 - np.mean(np.maximum(ratios - .8, 0) ** 2, axis=1) / .1 ** 2)
            terms = np.column_stack((pose, clearance, contact, velocity))
            contributions = terms * np.array([.1, .2, .2, .5]) * quality[:, None] * (~updated.terminated)[:, None]
            np.testing.assert_allclose(contributions.sum(axis=1) * env.cfg.ctrl_dt, updated.reward, atol=1e-7, rtol=1e-5)
            captured.update(reference=reference.copy(), residual=residual.copy(), target_delta=(target - env.anchor_ctrl[anchor_ids]).copy(),
                            joint_delta=(env.get_dof_pos() - env.anchor_qpos[anchor_ids, 7:]).copy(), terms=terms.copy(),
                            contributions=contributions.copy(), quality=quality.copy(), reward=updated.reward.copy(),
                            height=height.copy(), gravity_error=gravity_error.copy(), bad=bad.copy(), contacts=contacts.copy(),
                            progress=np.sum((env._backend.get_base_pos()[:, :2] - origins) * env.heading[anchors], axis=1),
                            ctrl_clipped=np.mean(np.abs(target - (env.anchor_ctrl[anchor_ids] if mode == "hold" else env.anchor_ctrl[anchor_ids] + reference + residual)) > 1e-6, axis=1))
            return updated

        env.update_state = capture
        active = np.ones(episodes, dtype=bool)
        rows = [[] for episode in range(episodes)]
        summaries = [None] * episodes
        for step in range(env.cfg.max_episode_steps):
            observation = mx.array(state.obs["obs"])
            mean, std, _ = policy.distribution_params(observation)
            actions = np.asarray(policy.act(observation)[0] if mode == "sampled" else mean, dtype=np.float32)
            if mode in ("zero", "hold"):
                actions = np.zeros_like(actions)
            state = env.step(actions)
            for episode in np.flatnonzero(active):
                row = {name: np.asarray(value[episode]).tolist() for name, value in captured.items()}
                row.update(elapsed_s=(step + 1) * env.cfg.ctrl_dt, action_clip_fraction=float(np.mean(np.abs(actions[episode]) > 1)), std=float(np.mean(np.asarray(std)[episode])))
                rows[episode].append(row)
                if state.terminated[episode] or state.truncated[episode]:
                    active[episode] = False
                    summaries[episode] = {"anchor_id": int(anchors[episode]), "duration_s": row["elapsed_s"], "progress_m": row["progress"],
                                          "survived": bool(state.truncated[episode] and not state.terminated[episode]),
                                          "initial_state_sha256": hashlib.sha256(initial[episode].tobytes()).hexdigest(),
                                          "failure": {"height": row["height"] < .16 or row["height"] > .65, "orientation": row["gravity_error"] > .8,
                                                      "contacts": [name for name, touched in zip(env.bad_contacts, row["bad"]) if touched]},
                                          "cancellation_after_2s": cancellation([item["reference"] for item in rows[episode] if item["elapsed_s"] > 2], [item["residual"] for item in rows[episode] if item["elapsed_s"] > 2])}
            if not active.any():
                break
        if active.any():
            raise AssertionError("Unfinished first episodes")
        paired = False
        if mode == "mean":
            baseline = json.loads(checkpoint.with_name(f"{checkpoint.stem}_ik_learned_{episodes}episodes.json").read_text())
            for actual, expected in zip(summaries, baseline["episodes"]):
                assert actual["initial_state_sha256"] == expected["initial_state_sha256"]
                np.testing.assert_allclose(actual["duration_s"], expected["duration_s"], atol=1e-6)
                np.testing.assert_allclose(actual["progress_m"], expected["progress_m"], atol=2e-5)
            paired = True
        windows = {}
        for label, lower, upper in (("hold_0_1s", 0, 1), ("ramp_1_2s", 1, 2), ("moving_2_20s", 2, 20)):
            selected = [row for trace in rows for row in trace if lower < row["elapsed_s"] <= upper]
            windows[label] = {"transitions": len(selected), "means": None if not selected else {
                name: np.mean([row[name] for row in selected], axis=0).tolist()
                for name in ("terms", "contributions", "quality", "reward", "action_clip_fraction", "ctrl_clipped", "std", "contacts")}}
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "mode": mode,
                "mean_reproduces_original_evaluation": paired, "train_eval_mean_identical": True,
                "summary": {"survived": sum(row["survived"] for row in summaries), "mean_duration_s": float(np.mean([row["duration_s"] for row in summaries])),
                            "mean_progress_m": float(np.mean([row["progress_m"] for row in summaries])),
                            "mean_episode_return": float(np.mean([sum(row["reward"] for row in trace) for trace in rows]))},
                "windows": windows, "episodes": summaries, "first_episode_trace": rows[0]}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Frozen checkpoint diagnosis; no optimizer or training")
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    output = args.output or args.matrix.with_name("policy_diagnosis.json")
    if output.exists():
        raise FileExistsError(output)
    report = {"diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "noise_seed": 91001,
              "term_order": ["pose", "clearance", "contact", "velocity"], "runs": [],
              "limitations": "100 paired first episodes. Sampled final policy is not historical training replay. Reward terms are proxies, not native contact loads. Centered cancellation is descriptive, not causal. No training or checkpoint selection."}
    for entry in matrix["runs"]:
        for mode in (("mean", "sampled", "zero", "hold") if entry["seed"] == 1 else ("mean", "sampled")):
            result = run(Path(entry["checkpoint"]), mode, 100)
            result["seed"] = entry["seed"]
            report["runs"].append(result)
            print(json.dumps({"seed": entry["seed"], "mode": mode, **result["summary"]}), flush=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    report["training_history"] = []
    for entry in matrix["runs"]:
        events = EventAccumulator(str(Path(entry["checkpoint"]).parent / "tb")).Reload()
        learner = json.loads(Path(entry["training_result"]).read_text())["learner_iterations"]
        report["training_history"].append({"seed": entry["seed"], "first_learner": learner[0], "last_learner": learner[-1],
                                           "scalar_tail": {tag: [{"step": event.step, "value": event.value} for event in events.Scalars(tag)[-3:]]
                                                           for tag in events.Tags()["scalars"]}})
    hashes = [[episode["initial_state_sha256"] for episode in run["episodes"]] for run in report["runs"]]
    assert all(value == hashes[0] for value in hashes)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
