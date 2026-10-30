from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from check_coordinated_exploration import verify_replay
from component_unload_return import envelope
from evaluate_short_step import episode
from short_step_assets import prepare
from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def perturbation(seed, amplitude):
    knots = np.random.default_rng(seed).uniform(-1., 1., (11, 23))
    knots[0] = 0.
    knots[-1] = 0.
    times = np.arange(201) * .02
    coordinate = times / .4
    indices = np.minimum(np.floor(coordinate).astype(int), 9)
    fraction = coordinate - indices
    weight = fraction ** 2 * (3 - 2 * fraction)
    values = amplitude * ((1 - weight[:, None]) * knots[indices] + weight[:, None] * knots[indices + 1])
    assert np.isfinite(values).all() and np.max(abs(values)) <= amplitude + 1e-12
    np.testing.assert_array_equal(values[0], np.zeros(23))
    np.testing.assert_allclose(values[-1], 0., atol=1e-12)
    return values


def main():
    bundle = prepare()
    payload = json.loads(bundle.read_text())
    source_path = ROOT / "output/rl/coordinated_exploration/2026-10-19_21-29-39-652172/report.json"
    source = json.loads(source_path.read_text())
    assert source["status"] == "completed" and source["bundle_sha256"] == digest(bundle)
    direction = np.asarray(source["directions_rad"]["coordinated"])
    destination = ROOT / "output/rl/exploration_neighborhood" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "source_report": str(source_path), "source_report_sha256": digest(source_path),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "check_coordinated_exploration.py", "evaluate_short_step.py", "component_unload_return.py", "short_step_task.py", "component_unload_ramps.py")},
              "protocol": {"amplitudes_rad": [.001, .003], "seeds": [1, 2, 3, 4], "knot_interval_s": .4,
                           "reference_gain": 1., "planned_episodes": 18, "direction_rad": direction.tolist()},
              "limitations": "Bounded smooth open-loop joint-target perturbations, not PPO action samples, reset randomization or a robust success rate. Identical perturbations paired between IK-zero and IK-coordinated references. Noise fades to zero at four seconds. Endpoint statistics may miss substep peaks; original substep guards unchanged. No parameter search or training.", "runs": []}
    report_path = destination / "report.json"
    try:
        cases = [(0., 0)] + [(amplitude, seed) for amplitude in (.001, .003) for seed in (1, 2, 3, 4)]
        for amplitude, seed in cases:
            disturbance = perturbation(seed, amplitude)
            for coordinated in (False, True):
                label = f"{'coordinated' if coordinated else 'zero'}_{amplitude}_{seed}"
                offsets = np.array([envelope(frame * .02, 1.) * direction if coordinated else np.zeros(23) for frame in range(201)]) + disturbance
                action_table = offsets / payload["protocol"]["residual_scale_rad"]
                assert np.max(abs(action_table)) <= 1.
                action_path = destination / f"{label}_planned.npz"
                np.savez_compressed(action_path, perturbation_rad=disturbance, actions=action_table)
                stats = []

                def policy(env, state):
                    frame = min(round(env.elapsed[0] / .02) + 1, 200)
                    desired = env.initial_ctrl + env.offsets[frame] + offsets[frame]
                    bounded = np.clip(desired, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
                    stats.append({"range_clipped_fraction": float(np.mean(abs(desired - bounded) > 1e-12)),
                                  "rate_limited_fraction": float(np.mean(abs(bounded - env.datas[0].ctrl) > env.protocol["max_ctrl_delta_rad"]))})
                    return action_table[frame]

                run = episode(bundle, label, policy, 1., destination / f"{label}.npz")
                run.update(coordinated=coordinated, amplitude_rad=amplitude, seed=seed,
                           planned_actions=str(action_path), planned_actions_sha256=digest(action_path),
                           native_replay_passed=verify_replay(bundle, run), control_stats=stats)
                run["summary"] = {
                    "reached_lift_window": run["elapsed_s"] >= .8,
                    "completed_without_guard_failure": run["failure"] is None and run["deadline"],
                    "min_left_load_n": min(row["loads_n"][0] for row in run["trace"]),
                    "peak_endpoint_penetration_m": max(row["penetration_m"] for row in run["trace"]),
                    "max_left_gap_m": max(row["gaps_m"][0] for row in run["trace"]),
                    "max_rate_limited_fraction": max(row["rate_limited_fraction"] for row in stats),
                    "max_range_clipped_fraction": max(row["range_clipped_fraction"] for row in stats)}
                report["runs"].append(run)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = []
        for amplitude in (0., .001, .003):
            for coordinated in (False, True):
                runs = [run for run in report["runs"] if run["amplitude_rad"] == amplitude and run["coordinated"] == coordinated]
                summary.append({"amplitude_rad": amplitude, "coordinated": coordinated, "episodes": len(runs),
                                "reached_0_8s": sum(run["summary"]["reached_lift_window"] for run in runs),
                                "completed_4s": sum(run["summary"]["completed_without_guard_failure"] for run in runs),
                                "failures": dict(Counter(run["failure"] or "deadline" for run in runs)),
                                "duration_range_s": [min(run["elapsed_s"] for run in runs), max(run["elapsed_s"] for run in runs)],
                                "max_stage": max(run["max_stage"] for run in runs)})
        assert len(report["runs"]) == 18
        report.update(status="completed", summary=summary)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
