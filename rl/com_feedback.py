import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from load_feedback import Predictor, DT, HORIZON, EXECUTE, UPDATES
from load_transfer import physical_metrics
from prepare import ROOT
from target_load_ablation import digest


def constraints(row, peak, predictive=False):
    return bool(peak <= (.00495 if predictive else .005) and row["other_total_n"] <= 1
                and .16 <= row["base_height_m"] <= .65
                and all(load > 5 for load in row["loads_n"]) and all(row["floor_contacts"]))


def com_features(data, target_com):
    return np.r_[(data.subtree_com[1, :2] - target_com) / .01, data.qvel[:3] / .2, data.qvel[3:6]]


class ComPredictor(Predictor):
    def __init__(self, model, target_com):
        super().__init__(model, 0)
        self.target_com = np.array(target_com)

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
        result = com_features(self.data, self.target_com)
        if not np.isfinite(result).all():
            raise AssertionError("Nonfinite prediction")
        return result, bool(feasible), peak


def run(model, initial_state, initial_ctrl, branch, output, target_com):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    first = physical_metrics(model, data)
    height = float(data.qpos[2])
    predictor = ComPredictor(model, target_com)

    rows, controls, qpos, qvel, plans = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    peak = first["penetration"]["depth_m"]
    failure = None
    start = float(data.time)
    for update in range(UPDATES):
        state = np.empty(len(initial_state))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        desired = np.zeros(8)
        if branch == "feedback":
            target, plan = predictor.choose(state, data.ctrl.copy(), desired)
            plan.update(update=update)
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
            row.update(elapsed_s=len(controls) * DT, com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - target_com)), velocity_max=float(np.max(np.abs(data.qvel))))
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
    tail = rows[-50:]
    reached = bool(len(rows) == UPDATES * EXECUTE and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05 for row in tail))
    np.savez_compressed(output, qpos=qpos, qvel=qvel, ctrl=np.array(controls).reshape(-1, model.nu), initial_physics_state=initial_state, initial_ctrl=initial_ctrl, dt=DT)
    return {"branch": branch, "duration_s": len(rows) * DT, "failure": failure, "physical_safety_passed": physical_safe,
            "reached": bool(physical_safe and reached),
            "initial_com_error_m": float(np.linalg.norm(np.array(first["com_xy"]) - target_com)),
            "final_com_error_m": rows[-1]["com_error_m"] if rows else float(np.linalg.norm(np.array(first["com_xy"]) - target_com)),
            "initial_loads_n": first["loads_n"], "final_loads_n": rows[-1]["loads_n"] if rows else first["loads_n"],
            "max_penetration_m": peak, "mean_loaded_slip_mps": mean_slip, "near_limit_fraction": near_limit,
            "prediction_rollouts": predictor.evaluations, "plans": plans, "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transfer_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.transfer_report.read_text())
    assert source["status"] == "completed" and source["solution"]["passed"]
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    chain = source
    for key in ("audit_report", "source_report", "baseline_report", "geometry_report", "source_report"):
        assert digest(chain[key]) == chain[key + "_sha256"]
        chain = json.loads(Path(chain[key]).read_text())
    assert digest(chain["model"]) == chain["model_sha256"] and mujoco.__version__ == chain["mujoco"]
    model = mujoco.MjModel.from_xml_path(chain["model"])
    transfer = next(run for run in source["runs"] if run["branch"] == "real_transfer")
    assert transfer["physical_safety_passed"] and transfer["duration_s"] == 6
    data = mujoco.MjData(model)
    with np.load(transfer["trajectory"], allow_pickle=False) as saved:
        mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = saved["initial_ctrl"]
        mujoco.mj_forward(model, data)
        for index, control in enumerate(saved["ctrl"]):
            data.ctrl[:] = control
            for substep in range(round(DT / model.opt.timestep)):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ctrl = data.ctrl.copy()
    target = mujoco.MjData(model)
    target.qpos[:] = source["solution"]["qpos"]
    mujoco.mj_forward(model, target)
    target_com = target.subtree_com[1, :2].copy()
    destination = ROOT / "output/rl/com_feedback" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0,
              "transfer_report": str(args.transfer_report.resolve()), "transfer_report_sha256": digest(args.transfer_report),
              "source_trajectory": transfer["trajectory"], "source_trajectory_sha256": digest(transfer["trajectory"]),
              "model": chain["model"], "model_sha256": chain["model_sha256"], "mujoco": mujoco.__version__,
              "source_sha256": {**source["source_sha256"], "load_feedback.py": digest(Path(__file__).with_name("load_feedback.py")), Path(__file__).name: digest(__file__)},
              "initial_physics_state": state.tolist(), "initial_ctrl": ctrl.tolist(), "target_com_xy_m": target_com.tolist(),
              "protocol": {"duration_s": 3, "prediction_s": .2, "replan_s": .1, "updates": 30, "max_prediction_rollouts": 1530,
                           "trust_rad": .015, "finite_difference_rad": .003, "com_feature_scale_m": .01,
                           "com_gate_m": .005, "velocity_gate": .05, "tail_s": 1, "restarts": 0},
              "limitations": "One true endpoint paired hold/feedback. Target is frozen four-support equilibrium COM, not lift or gait. Four-support endpoint guards, substep penetration guard; local finite-difference prediction, not full horizon optimization or robust control."}
    started = time.monotonic()
    try:
        for branch in ("hold", "feedback"):
            result = run(model, state, ctrl, branch, destination / f"{branch}.npz", target_com)
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
