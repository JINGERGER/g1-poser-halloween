from datetime import datetime
import json
from pathlib import Path

import numpy as np

from component_unload_return import envelope
from evaluate_short_step import episode
from short_step_assets import prepare
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def verify_replay(bundle, run):
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle), reference_gain=run["reference_gain"]))
    env._autoreset = False
    try:
        env.init_state()
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            for index, action in enumerate(saved["actions"]):
                state = env.step(action[None])
                np.testing.assert_allclose(env.datas[0].qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(env.datas[0].qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(state.reward[0], saved["rewards"][index], atol=1e-10, rtol=0)
                assert state.info["evaluation"][0] == run["trace"][index]
                assert bool(state.terminated[0]) == (index == len(saved["actions"]) - 1)
        return True
    finally:
        env.close()


def main():
    bundle = prepare()
    payload = json.loads(bundle.read_text())
    recovery_path = ROOT / "output/rl/component_unload_return/2026-10-19_20-50-12-737895/report.json"
    recovery = json.loads(recovery_path.read_text())
    assert recovery["status"] == "completed" and recovery["model_sha256"] == payload["model_sha256"]
    direction = .25 * (np.asarray(recovery["candidate_ctrl"]) - payload["initial_ctrl"])
    directions = {"zero": np.zeros(23), "coordinated": direction, "opposite": -direction}
    for seed in (1, 2, 3, 4):
        directions[f"permuted_{seed}"] = np.random.default_rng(seed).permutation(direction)
    for name, values in directions.items():
        assert np.max(abs(values)) <= payload["protocol"]["residual_scale_rad"]
        if name != "zero":
            np.testing.assert_allclose(np.linalg.norm(values), np.linalg.norm(direction), atol=1e-12)
    destination = ROOT / "output/rl/coordinated_exploration" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "recovery_report": str(recovery_path), "recovery_report_sha256": digest(recovery_path),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "evaluate_short_step.py", "short_step_task.py", "component_unload_return.py", "component_unload_ramps.py")},
              "directions_rad": {name: values.tolist() for name, values in directions.items()},
              "protocol": {"reference_gains": [0., 1.], "ramp_s": .6, "return_start_s": 1., "return_end_s": 2., "duration_s": 4., "planned_episodes": 14},
              "limitations": "Matched residual joint-angle norms and envelope, not matched torque, mechanical work or post-clipping control. Four preselected permutations are diagnostic controls, not a policy search or success-rate estimate. Contact statistics are control-step endpoints; guards still run at every physics substep. No changed model/reset/threshold, no stochastic policy, no training.", "runs": []}
    report_path = destination / "report.json"
    try:
        env = ShortStepEnv(ShortStepCfg(bundle=str(bundle)))
        try:
            env.init_state()
            report["initial_metrics"] = env.metrics[0]
        finally:
            env.close()
        for gain in (0., 1.):
            for name, values in directions.items():
                control_stats = []

                def policy(env, state):
                    action = envelope(env.elapsed[0] + .02, 1.) * values / env.protocol["residual_scale_rad"]
                    frame = min(round(env.elapsed[0] / .02) + 1, len(env.offsets) - 1)
                    desired = env.initial_ctrl + gain * env.offsets[frame] + env.protocol["residual_scale_rad"] * action
                    clipped = np.clip(desired, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
                    control_stats.append({"range_clipped_fraction": float(np.mean(abs(desired - clipped) > 1e-12)),
                                          "rate_limited_fraction": float(np.mean(abs(clipped - env.datas[0].ctrl) > env.protocol["max_ctrl_delta_rad"]))})
                    return action

                label = f"reference_{int(gain)}_{name}"
                run = episode(bundle, label, policy, gain, destination / f"{label}.npz")
                run["direction"] = name
                run["control_stats"] = control_stats
                run["native_replay_passed"] = verify_replay(bundle, run)
                run["summary"] = {
                    "min_support_loads_n": np.min([row["loads_n"] for row in run["trace"]], axis=0).tolist(),
                    "peak_endpoint_penetration_m": max(row["penetration_m"] for row in run["trace"]),
                    "max_left_gap_m": max(row["gaps_m"][0] for row in run["trace"]),
                    "max_other_load_n": max(max(row["other_loads_n"].values(), default=0.) for row in run["trace"]),
                    "mean_rate_limited_fraction": float(np.mean([row["rate_limited_fraction"] for row in control_stats])),
                    "max_range_clipped_fraction": max(row["range_clipped_fraction"] for row in control_stats)}
                report["runs"].append(run)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: run[key] for key in ("label", "elapsed_s", "failure", "max_stage", "summary")}), flush=True)
        assert len(report["runs"]) == 14
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
