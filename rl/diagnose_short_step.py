from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from short_step_assets import prepare
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest


def probe(bundle, label, gain, noise, seed, episode):
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle), reference_gain=gain))
    env._autoreset = False
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode]))
    rows = []
    try:
        env.init_state()
        initial_depth = env.metrics[0]["penetration"]["depth_m"]
        for unused in range(200):
            action = rng.normal(0, noise, (1, 23))
            data = env.datas[0]
            previous = data.ctrl.copy()
            frame = min(round(env.elapsed[0] / .02) + 1, len(env.offsets) - 1)
            raw = env.initial_ctrl + gain * env.offsets[frame] + .25 * np.clip(action[0], -1, 1)
            bounded = np.clip(raw, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
            desired = previous + np.clip(bounded - previous, -.01, .01)
            state = env.step(action)
            np.testing.assert_allclose(data.ctrl, desired, atol=1e-12, rtol=0)
            outcome = state.info["evaluation"][0]
            deepest = min(data.contact, key=lambda contact: contact.dist, default=None)
            rows.append({**outcome,
                         "rate_limited_fraction": float(np.mean(abs(bounded - previous) > .01)),
                         "range_clipped_fraction": float(np.mean(abs(raw - bounded) > 1e-12)),
                         "ctrl_delta_max_rad": float(np.max(abs(data.ctrl - previous))),
                         "joint_displacement_max_rad": float(np.max(abs(data.qpos[7:] - env.initial_qpos[7:]))),
                         "deepest_contact": None if deepest is None else
                         [env.model.geom(int(geom)).name for geom in (deepest.geom1, deepest.geom2)]})
            if state.terminated[0]:
                break
        assert state.terminated[0]
        return {"label": label, "seed": seed, "episode": episode, "noise_std": noise,
                "reference_gain": gain, "initial_penetration_m": initial_depth,
                "elapsed_s": rows[-1]["elapsed_s"], "failure": rows[-1]["failure"],
                "success": rows[-1]["success"], "max_stage": max(row["stage"] for row in rows),
                "min_left_load_n": min(row["loads_n"][0] for row in rows),
                "max_left_gap_m": max(row["gaps_m"][0] for row in rows),
                "peak_endpoint_penetration_m": max(row["penetration_m"] for row in rows),
                "mean_rate_limited_fraction": float(np.mean([row["rate_limited_fraction"] for row in rows])),
                "trace": rows}
    finally:
        env.close()


def main():
    bundle = prepare()
    runs = [probe(bundle, "hold", 0., 0., 1, 0), probe(bundle, "ik_zero_residual", 1., 0., 1, 0)]
    for noise in (.15, .03):
        for seed in (1, 2):
            for episode in range(4):
                runs.append(probe(bundle, f"zero_mean_noise_{noise}", 1., noise, seed, episode))
    summary = {}
    for label in dict.fromkeys(run["label"] for run in runs):
        selected = [run for run in runs if run["label"] == label]
        summary[label] = {
            "episodes": len(selected), "failure_counts": dict(Counter(run["failure"] or "deadline" for run in selected)),
            "median_duration_s": float(np.median([run["elapsed_s"] for run in selected])),
            "reached_lift_reference_count": sum(run["elapsed_s"] >= .8 for run in selected),
            "max_stage": max(run["max_stage"] for run in selected),
            "min_left_load_n": min(run["min_left_load_n"] for run in selected),
            "mean_rate_limited_fraction": float(np.mean([run["mean_rate_limited_fraction"] for run in selected]))}
    destination = Path(__file__).resolve().parents[1] / "output/rl/short_step_diagnostics" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"bundle": str(bundle), "bundle_sha256": digest(bundle),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "short_step_task.py", "component_unload_ramps.py")},
              "limitations": "Zero-mean independent Gaussian residual probes, NOT samples from the initialized or trained PPO actor. Small nominal diagnostic, not feasibility proof. Endpoint contact statistics may miss substep peaks; task guards still run every substep. Noise 0.03 is a diagnostic comparison, not a training configuration change.",
              "summary": summary, "runs": runs}
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(destination / "report.json")


if __name__ == "__main__":
    main()
