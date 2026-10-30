import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from load_transfer import physical_metrics, smooth
from prepare import ROOT


DT = .02
HORIZON = 10
EXECUTE = 5
UPDATES = 30


def features(model, data, row, height):
    return np.r_[np.array(row["loads_n"]) / 100 * [1, 1, 3, 1], data.qvel[:3] / .2,
                 data.qvel[3:6], (data.qpos[2] - height) / .1]


def constraints(row, peak, predictive=False):
    return bool(peak <= (.00495 if predictive else .005) and row["other_total_n"] <= 1
                and .16 <= row["base_height_m"] <= .65
                and all(row["loads_n"][limb] > 10 for limb in (0, 1, 3)))


class Predictor:
    def __init__(self, model, height):
        self.model, self.height = model, height
        self.data = mujoco.MjData(model)
        self.substeps = round(DT / model.opt.timestep)
        assert abs(self.substeps * model.opt.timestep - DT) < 1e-10
        self.evaluations = 0

    def rollout(self, state, initial_ctrl, target):
        self.evaluations += 1
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(self.model, self.data)
        peak = physical_metrics(self.model, self.data)["penetration"]["depth_m"]
        feasible = True
        for step in range(HORIZON):
            self.data.ctrl[:] = np.clip(self.data.ctrl + np.clip(target - self.data.ctrl, -.01, .01), self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            for substep in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
                peak = max(peak, max((-contact.dist for contact in self.data.contact), default=0))
            mujoco.mj_forward(self.model, self.data)
            row = physical_metrics(self.model, self.data)
            peak = max(peak, row["penetration"]["depth_m"])
            feasible = feasible and constraints(row, peak, True) and max(row["torque_ratios"]) < .95
        result = features(self.model, self.data, row, self.height)
        if not np.isfinite(result).all():
            raise AssertionError("Nonfinite prediction")
        return result, bool(feasible), peak

    def choose(self, state, ctrl, desired):
        baseline, baseline_safe, _ = self.rollout(state, ctrl, ctrl)
        jacobian = np.zeros((len(baseline), self.model.nu))
        for joint in range(self.model.nu):
            plus, minus = ctrl.copy(), ctrl.copy()
            plus[joint] = min(ctrl[joint] + .003, self.model.actuator_ctrlrange[joint, 1])
            minus[joint] = max(ctrl[joint] - .003, self.model.actuator_ctrlrange[joint, 0])
            if plus[joint] > minus[joint]:
                jacobian[:, joint] = (self.rollout(state, ctrl, plus)[0] - self.rollout(state, ctrl, minus)[0]) / (plus[joint] - minus[joint])
        lower = np.maximum(-.015, self.model.actuator_ctrlrange[:, 0] - ctrl)
        upper = np.minimum(.015, self.model.actuator_ctrlrange[:, 1] - ctrl)
        fitted = lsq_linear(np.vstack((jacobian, np.eye(self.model.nu) * (.1 / .015))),
                            np.r_[desired - baseline, np.zeros(self.model.nu)], bounds=(lower, upper), max_iter=100)
        target = ctrl.copy()
        best_cost = float(np.sum((baseline - desired) ** 2)) if baseline_safe else float("inf")
        selected = 0.0
        for scale in (1., .5, .25, .125):
            candidate = ctrl + scale * fitted.x
            predicted, feasible, peak = self.rollout(state, ctrl, candidate)
            cost = float(np.sum((predicted - desired) ** 2))
            if feasible and cost < best_cost - 1e-6:
                target, best_cost, selected = candidate, cost, scale
        return target, {"selected_scale": selected, "baseline_prediction_safe": baseline_safe,
                        "safe_candidate_found": bool(baseline_safe or selected > 0),
                        "predicted_cost": best_cost if np.isfinite(best_cost) else None,
                        "least_squares_success": bool(fitted.success)}


def run(model, initial_state, initial_ctrl, branch, output):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    first = physical_metrics(model, data)
    height = float(data.qpos[2])
    predictor = Predictor(model, height)
    initial_loads = np.array(first["loads_n"])
    final_loads = np.array([70, 80, 5, sum(model.body_mass) * np.linalg.norm(model.opt.gravity) - 155])
    rows, controls, qpos, qvel, plans = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    peak = first["penetration"]["depth_m"]
    failure = None
    start = float(data.time)
    for update in range(UPDATES):
        state = np.empty(len(initial_state))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        blend = smooth((update + 1) / UPDATES)
        desired_loads = initial_loads * (1 - blend) + final_loads * blend
        desired = np.r_[desired_loads / 100 * [1, 1, 3, 1], np.zeros(7)]
        if branch == "feedback":
            target, plan = predictor.choose(state, data.ctrl.copy(), desired)
            plan.update(update=update, desired_loads_n=desired_loads.tolist())
            plans.append(plan)
            if not plan["safe_candidate_found"]:
                failure = "no_safe_predicted_action"
                break
        else:
            target = initial_ctrl
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
                break
        if failure:
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    mean_slip = float(np.mean(slips)) if slips else None
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows])) if rows else None
    physical_safe = bool(rows and peak <= .005 and all(constraints(row, peak) for row in rows)
                         and mean_slip is not None and mean_slip <= .02 and near_limit <= .1)
    tail = rows[-10:]
    unloaded = bool(len(tail) == 10 and all(row["loads_n"][2] <= 5 for row in tail))
    np.savez_compressed(output, qpos=qpos, qvel=qvel, ctrl=np.array(controls).reshape(-1, model.nu), initial_physics_state=initial_state, initial_ctrl=initial_ctrl, dt=DT)
    return {"branch": branch, "duration_s": len(rows) * DT, "failure": failure, "physical_safety_passed": physical_safe,
            "unload_passed": bool(physical_safe and unloaded and len(rows) == UPDATES * EXECUTE),
            "initial_loads_n": first["loads_n"], "final_loads_n": rows[-1]["loads_n"] if rows else first["loads_n"],
            "max_penetration_m": peak, "mean_loaded_slip_mps": mean_slip, "near_limit_fraction": near_limit,
            "prediction_rollouts": predictor.evaluations, "plans": plans, "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Paired short-horizon load-feedback test from a saved real transfer endpoint")
    parser.add_argument("transfer_report", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    source = json.loads(args.transfer_report.read_text())
    for name, digest in source["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    matrix_path = Path(source["matrix"])
    assert hashlib.sha256(matrix_path.read_bytes()).hexdigest() == source["matrix_sha256"]
    matrix = json.loads(matrix_path.read_text())
    entry = next(row for row in matrix["runs"] if row["anchor"] == 2 and row["swing"] == 2)
    assert hashlib.sha256(Path(entry["model"]).read_bytes()).hexdigest() == entry["model_sha256"]
    model = mujoco.MjModel.from_xml_path(entry["model"])
    saved_path = Path(source["replay"]["trajectory"])
    saved = np.load(saved_path, allow_pickle=False)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for control in saved["ctrl"]:
        data.ctrl[:] = control
        for substep in range(round(DT / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos, saved["qpos"][-1], atol=1e-9, rtol=0)
    np.testing.assert_allclose(data.qvel, saved["qvel"][-1], atol=1e-9, rtol=0)
    initial_state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_ctrl = data.ctrl.copy()
    assert constraints(physical_metrics(model, data), physical_metrics(model, data)["penetration"]["depth_m"])
    destination = ROOT / "output/rl/load_feedback" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources = dict(source["source_sha256"])
    sources["load_feedback.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__, "model": entry["model"], "model_sha256": entry["model_sha256"],
              "source_sha256": sources, "transfer_report": str(args.transfer_report.resolve()), "transfer_report_sha256": hashlib.sha256(args.transfer_report.read_bytes()).hexdigest(),
              "source_trajectory_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(), "initial_physics_state": initial_state.tolist(), "initial_ctrl": initial_ctrl.tolist(),
              "protocol": {"duration_s": 3, "prediction_s": .2, "replan_s": .1, "updates": UPDATES, "trust_rad": .015,
                           "finite_difference_rad": .003, "backtracking": [1, .5, .25, .125], "max_prediction_rollouts": UPDATES * 51},
              "limitations": "One paired endpoint, no robustness claim. Local load-response feedback with bounded target steps and nonlinear prediction check, not full MPC trajectory optimization. Unload gate is <=5N, not no-contact/clearance/landing/crawl. Predictions are scratch simulations only; live state changes through motors only."}
    started = time.monotonic()
    try:
        for branch in ("hold", "feedback"):
            result = run(model, initial_state, initial_ctrl, branch, destination / f"{branch}.npz")
            report["runs"].append(result)
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
