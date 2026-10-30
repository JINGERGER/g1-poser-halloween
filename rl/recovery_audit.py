import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from com_feedback import ComPredictor, constraints
from load_transfer import physical_metrics
from prepare import ROOT
from target_load_ablation import digest


class RecordedPredictor(ComPredictor):
    def __init__(self, model, target_com):
        super().__init__(model, target_com)
        self.calls = []

    def rollout(self, state, initial_ctrl, target):
        result = super().rollout(state, initial_ctrl, target)
        self.calls.append({"target_ctrl": target.tolist(), "safe": result[1], "peak_m": result[2], "features": result[0].tolist()})
        return result


def inspect_rollout(model, state, initial_ctrl, target, target_com, duration, stop_on_guard, output=None):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    peak = physical_metrics(model, data)["penetration"]["depth_m"]
    failure, start = None, float(data.time)
    minimum = {"prediction_penetration_m": .00495 - peak, "physical_penetration_m": .005 - peak,
               "torque_ratio": 1., "support_load_n": float("inf"), "other_load_n": float("inf"),
               "height_lower_m": float("inf"), "height_upper_m": float("inf")}
    first_violation = {}
    for step in range(round(duration / .02)):
        data.ctrl[:] = np.clip(data.ctrl + np.clip(target - data.ctrl, -.01, .01), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        controls.append(data.ctrl.copy())
        assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
        step_peak = 0.
        for substep in range(round(.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            step_peak = max(step_peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and abs(data.time - start - (step + 1) * .02) < 1e-7
        row = physical_metrics(model, data)
        step_peak = max(step_peak, row["penetration"]["depth_m"])
        peak = max(peak, step_peak)
        margins = {"prediction_penetration_m": .00495 - peak, "physical_penetration_m": .005 - peak,
                   "torque_ratio": .95 - max(row["torque_ratios"]), "support_load_n": min(row["loads_n"]) - 5,
                   "other_load_n": 1 - row["other_total_n"], "height_lower_m": row["base_height_m"] - .16,
                   "height_upper_m": .65 - row["base_height_m"]}
        for name, value in margins.items():
            minimum[name] = min(minimum[name], value)
            if value < 0 or (name in ("torque_ratio", "support_load_n") and value == 0):
                first_violation.setdefault(name, (step + 1) * .02)
        if not all(row["floor_contacts"]):
            first_violation.setdefault("four_support_contact", (step + 1) * .02)
        row.update(elapsed_s=(step + 1) * .02, step_peak_m=step_peak,
                   com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - target_com)), velocity_max=float(np.max(np.abs(data.qvel))))
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        if stop_on_guard and not constraints(row, peak):
            failure = "physical_guard"
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    safe = bool(rows and all(constraints(row, peak) for row in rows) and slips and np.mean(slips) <= .02
                and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
    predicted = bool(rows and all(constraints(row, peak, True) and max(row["torque_ratios"]) < .95 for row in rows))
    completed = len(rows) == round(duration / .02)
    settled = bool(completed and len(rows) >= 50 and all(row["step_peak_m"] <= .00495 and row["velocity_max"] <= .05 for row in rows[-50:]))
    final_state = np.empty(len(state))
    mujoco.mj_getState(model, data, final_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    if output is not None:
        np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=state, initial_ctrl=initial_ctrl, dt=.02)
    return {"duration_s": len(rows) * .02, "failure": failure, "target_ctrl": target.tolist(),
            "physical_safety_passed": safe, "prediction_safe": predicted, "settled_with_margin": bool(safe and settled),
            "com_reached": bool(safe and settled and all(row["com_error_m"] <= .005 for row in rows[-50:])),
            "minimum_margins": minimum, "first_violation_s": first_violation, "max_penetration_m": peak,
            "final_com_error_m": rows[-1]["com_error_m"], "final_loads_n": rows[-1]["loads_n"],
            "final_physics_state": final_state.tolist(), "final_ctrl": data.ctrl.tolist(), "trace": rows,
            "trajectory": str(output) if output is not None else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feedback_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.feedback_report.read_text())
    assert source["status"] == "completed"
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["model"]) == source["model_sha256"] and mujoco.__version__ == source["mujoco"]
    stopped = next(run for run in source["runs"] if run["branch"] == "feedback")
    assert stopped["failure"] == "no_safe_predicted_action"
    model, data = mujoco.MjModel.from_xml_path(source["model"]), None
    data = mujoco.MjData(model)
    with np.load(stopped["trajectory"], allow_pickle=False) as saved:
        mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = saved["initial_ctrl"]
        mujoco.mj_forward(model, data)
        for index, ctrl in enumerate(saved["ctrl"]):
            data.ctrl[:] = ctrl
            for substep in range(round(.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    current, rollback, target_com = data.ctrl.copy(), np.array(source["initial_ctrl"]), np.array(source["target_com_xy_m"])
    predictor = RecordedPredictor(model, target_com)
    selected, plan = predictor.choose(state, current, np.zeros(8))
    assert not plan["safe_candidate_found"] and predictor.evaluations == 51
    for key in ("selected_scale", "baseline_prediction_safe", "safe_candidate_found", "least_squares_success"):
        assert plan[key] == stopped["plans"][-1][key]
    candidates = [("hold_current", predictor.calls[0])] + [(f"rejected_scale_{scale}", call) for scale, call in zip((1., .5, .25, .125), predictor.calls[-4:])]
    candidates.append(("rollback_previous", {"target_ctrl": rollback.tolist()}))
    destination = ROOT / "output/rl/recovery_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "candidates": [], "recoveries": [],
              "feedback_report": str(args.feedback_report.resolve()), "feedback_report_sha256": digest(args.feedback_report),
              "source_trajectory": stopped["trajectory"], "source_trajectory_sha256": digest(stopped["trajectory"]),
              "source_sha256": {**source["source_sha256"], Path(__file__).name: digest(__file__)},
              "initial_physics_state": state.tolist(), "initial_ctrl": current.tolist(), "target_com_xy_m": target_com.tolist(),
              "protocol": {"reproduced_prediction_calls": 51, "diagnostic_candidates": 6, "diagnostic_s": .2,
                           "recovery_candidates": ["hold_current", "rollback_previous"], "recovery_s": 3, "terminal_prediction_s": .2, "restarts": 0},
              "limitations": "Rejected actions are diagnostic scratch predictions only. Only hold/previous target passing prediction gate may execute a bounded native recovery. Recovery is not COM arrival or crawling; terminal prediction is not indefinite stability or robust safety."}
    try:
        for name, candidate in candidates:
            result = inspect_rollout(model, state, current, np.array(candidate["target_ctrl"]), target_com, .2, False)
            result["branch"] = name
            if "safe" in candidate:
                assert result["prediction_safe"] == candidate["safe"]
                np.testing.assert_allclose(result["max_penetration_m"], candidate["peak_m"], atol=1e-12, rtol=0)
            report["candidates"].append(result)
            if name in ("hold_current", "rollback_previous") and result["prediction_safe"]:
                recovery = inspect_rollout(model, state, current, np.array(candidate["target_ctrl"]), target_com, 3, True, destination / f"{name}.npz")
                recovery["branch"] = name
                tail = inspect_rollout(model, np.array(recovery["final_physics_state"]), np.array(recovery["final_ctrl"]), np.array(candidate["target_ctrl"]), target_com, .2, False)
                recovery["terminal_prediction_safe"] = tail["prediction_safe"]
                recovery["terminal_prediction"] = tail
                recovery["recovered"] = bool(recovery["settled_with_margin"] and tail["prediction_safe"])
                report["recoveries"].append(recovery)
            print(json.dumps({"branch": name, "safe": result["prediction_safe"], "peak_m": result["max_penetration_m"], "violations": result["first_violation_s"]}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps([{key: run[key] for key in ("branch", "duration_s", "physical_safety_passed", "recovered", "com_reached", "final_com_error_m", "max_penetration_m")} for run in report["recoveries"]]), flush=True)
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
