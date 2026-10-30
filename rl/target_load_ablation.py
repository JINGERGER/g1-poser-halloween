import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from directional_feedback import DirectionalPredictor, hand_envelope
from load_feedback import DT, EXECUTE, UPDATES, constraints
from load_transfer import physical_metrics, smooth
from prepare import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_target(matrix, model_path, model_hash):
    assert matrix["status"] == "completed"
    selected = next(run for run in matrix["runs"] if run["anchor"] == 2 and run["swing"] == 2)
    assert selected["model"] == model_path and selected["model_sha256"] == model_hash
    hold = next(run for run in selected["replays"] if run["branch"] == "hold")
    assert hold["passed"] and hold["duration_s"] == 3
    rows = [row for row in hold["trace"] if row["elapsed_s"] > 1]
    assert len(rows) == 100
    measured = np.mean([row["loads_n"] for row in rows], axis=0)
    assert abs(measured[2]) < 1e-9
    target = measured.copy()
    target[2] = 5
    target[3] -= 5
    assert np.all(target > 0)
    np.testing.assert_allclose(sum(target), sum(measured), atol=1e-10, rtol=0)
    return measured, target


def targeted_run(model, initial_state, initial_ctrl, output, final_loads):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    first = physical_metrics(model, data)
    initial_loads = np.array(first["loads_n"])
    predictor = DirectionalPredictor(model, float(data.qpos[2]), initial_loads)
    final_loads = np.asarray(final_loads, dtype=float)
    assert final_loads.shape == (4,) and np.isfinite(final_loads).all() and np.all(final_loads >= 0)
    rows, controls, qpos, qvel, plans = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    peak, failure, start = first["penetration"]["depth_m"], None, float(data.time)
    for update in range(UPDATES):
        state = np.empty(len(initial_state))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        blend = smooth((update + 1) / UPDATES)
        desired_loads = initial_loads * (1 - blend) + final_loads * blend
        desired = np.r_[desired_loads / 100 * [1, 1, 3, 1], np.zeros(7)]
        target, plan = predictor.choose(state, data.ctrl.copy(), desired)
        plan.update(update=update, desired_loads_n=desired_loads.tolist())
        plans.append(plan)
        if not plan["safe_candidate_found"]:
            failure = "no_directionally_safe_predicted_action"
            break
        for step in range(EXECUTE):
            data.ctrl[:] = np.clip(data.ctrl + np.clip(target - data.ctrl, -.01, .01), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            controls.append(data.ctrl.copy())
            assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
            for substep in range(predictor.substeps):
                mujoco.mj_step(model, data)
                peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - start - len(controls) * DT) > 1e-7:
                raise AssertionError("Invalid live state or implicit reset")
            row = physical_metrics(model, data)
            row["elapsed_s"] = len(controls) * DT
            rows.append(row)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            peak = max(peak, row["penetration"]["depth_m"])
            if not constraints(row, peak):
                failure = "physical_guard"
            elif not hand_envelope(row["loads_n"], initial_loads):
                failure = "hand_direction_guard"
            if failure:
                break
        if failure:
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    mean_slip = float(np.mean(slips)) if slips else None
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows])) if rows else None
    physical = bool(rows and all(constraints(row, peak) for row in rows) and mean_slip is not None and mean_slip <= .02 and near_limit <= .1)
    directional = bool(rows and all(hand_envelope(row["loads_n"], initial_loads) for row in rows))
    unloaded = bool(len(rows) == UPDATES * EXECUTE and all(row["loads_n"][2] <= 5 for row in rows[-10:]))
    np.savez_compressed(output, qpos=qpos, qvel=qvel, ctrl=np.array(controls).reshape(-1, model.nu), initial_physics_state=initial_state, initial_ctrl=initial_ctrl, dt=DT)
    return {"branch": "directional", "duration_s": len(rows) * DT, "failure": failure, "physical_safety_passed": physical,
            "direction_guard_passed": directional, "unload_passed": physical and directional and unloaded,
            "initial_loads_n": first["loads_n"], "final_loads_n": rows[-1]["loads_n"] if rows else first["loads_n"],
            "max_penetration_m": peak, "mean_loaded_slip_mps": mean_slip, "near_limit_fraction": near_limit,
            "prediction_rollouts": predictor.evaluations, "plans": plans, "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Frozen directional controller, paired load-target ablation")
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("matrix_report", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    baseline = json.loads(args.baseline_report.read_text())
    matrix = json.loads(args.matrix_report.read_text())
    assert baseline["status"] == "completed"
    sources = dict(baseline["source_sha256"])
    for name, expected in matrix["source_sha256"].items():
        assert name not in sources or sources[name] == expected
        sources[name] = expected
    for name, expected in sources.items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(baseline["model"]) == baseline["model_sha256"]
    measured, target = terminal_target(matrix, baseline["model"], baseline["model_sha256"])
    model = mujoco.MjModel.from_xml_path(baseline["model"])
    state, ctrl = np.array(baseline["initial_physics_state"]), np.array(baseline["initial_ctrl"])
    arbitrary = np.array([70, 80, 5, sum(model.body_mass) * np.linalg.norm(model.opt.gravity) - 155])
    np.testing.assert_allclose(sum(target), sum(arbitrary), atol=1e-5, rtol=0)
    destination = ROOT / "output/rl/target_load_ablation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources[Path(__file__).name] = digest(__file__)
    report = {
        "status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__,
        "model": baseline["model"], "model_sha256": baseline["model_sha256"],
        "initial_physics_state": state.tolist(), "initial_ctrl": ctrl.tolist(), "source_sha256": sources,
        "baseline_report": str(args.baseline_report.resolve()), "baseline_report_sha256": digest(args.baseline_report),
        "matrix_report": str(args.matrix_report.resolve()), "matrix_report_sha256": digest(args.matrix_report),
        "measured_tripod_loads_n": measured.tolist(),
        "protocol": dict(baseline["protocol"], changed_variable="terminal supporting-limb load distribution",
                         target_window_s=[1, 3], left_foot_target_n=5),
        "limitations": "One initial state. Tripod hold starts from a hypothetical static state, not a reachable path. No gain, duration, horizon or hand-bound changes; no RL, lift or crawl claim."
    }
    started = time.monotonic()
    try:
        for branch, desired in (("arbitrary", arbitrary), ("tripod_derived", target)):
            result = targeted_run(model, state, ctrl, destination / f"{branch}.npz", desired)
            result.update(branch=branch, terminal_target_n=desired.tolist())
            report["runs"].append(result)
            if branch == "arbitrary":
                old = next(run for run in baseline["runs"] if run["branch"] == "directional")
                assert result["duration_s"] == old["duration_s"]
                np.testing.assert_allclose(result["final_loads_n"], old["final_loads_n"], atol=1e-9, rtol=0)
                with np.load(result["trajectory"]) as actual, np.load(old["trajectory"]) as expected:
                    for key in ("qpos", "qvel", "ctrl"):
                        np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=0)
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key not in ("trace", "plans")}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
