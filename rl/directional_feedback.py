import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from load_feedback import DT, EXECUTE, HORIZON, UPDATES, Predictor, constraints, features, run as baseline_run
from load_transfer import physical_metrics, smooth
from prepare import ROOT


def hand_envelope(loads, initial):
    return bool(loads[0] <= initial[0] + .5 + 1e-6 and loads[1] >= initial[1] - .5 - 1e-6)


class DirectionalPredictor(Predictor):
    def __init__(self, model, height, initial_loads):
        super().__init__(model, height)
        self.initial_loads = np.array(initial_loads)
        self.current_loads = self.initial_loads.copy()

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
            feasible = (feasible and constraints(row, peak, True) and max(row["torque_ratios"]) < .95
                        and hand_envelope(row["loads_n"], self.initial_loads))
        feasible = feasible and row["loads_n"][0] <= self.current_loads[0] + .25 + 1e-6 and row["loads_n"][1] >= self.current_loads[1] - .25 - 1e-6
        result = features(self.model, self.data, row, self.height)
        if not np.isfinite(result).all():
            raise AssertionError("Nonfinite prediction")
        return result, bool(feasible), peak

    def choose(self, state, ctrl, desired):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        self.current_loads = np.array(physical_metrics(self.model, self.data)["loads_n"])
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
        matrix = np.vstack((jacobian, np.eye(self.model.nu) * (.1 / .015)))
        residual = np.r_[desired - baseline, np.zeros(self.model.nu)]
        left_cap = min(self.initial_loads[0] + .5, self.current_loads[0] + .25) / 100
        right_floor = max(self.initial_loads[1] - .5, self.current_loads[1] - .25) / 100
        linear = LinearConstraint(jacobian[:2], [-np.inf, right_floor - baseline[1]], [left_cap - baseline[0], np.inf])
        fitted = minimize(lambda delta: .5 * np.sum((matrix @ delta - residual) ** 2), np.zeros(self.model.nu),
                          jac=lambda delta: matrix.T @ (matrix @ delta - residual), method="SLSQP",
                          bounds=list(zip(lower, upper)), constraints=[linear], options={"maxiter": 100, "ftol": 1e-10})
        target = ctrl.copy()
        best_cost = float(np.sum((baseline - desired) ** 2)) if baseline_safe else float("inf")
        selected = 0.0
        for scale in (1., .5, .25, .125):
            candidate = np.clip(ctrl + scale * fitted.x, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            predicted, feasible, _ = self.rollout(state, ctrl, candidate)
            cost = float(np.sum((predicted - desired) ** 2))
            if feasible and cost < best_cost - 1e-6:
                target, best_cost, selected = candidate, cost, scale
        return target, {"selected_scale": selected, "baseline_prediction_safe": baseline_safe,
                        "safe_candidate_found": bool(baseline_safe or selected > 0), "qp_success": bool(fitted.success),
                        "qp_iterations": int(fitted.nit), "qp_message": str(fitted.message),
                        "predicted_cost": best_cost if np.isfinite(best_cost) else None,
                        "current_loads_n": self.current_loads.tolist()}


def guarded_run(model, initial_state, initial_ctrl, output):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    first = physical_metrics(model, data)
    initial_loads = np.array(first["loads_n"])
    predictor = DirectionalPredictor(model, float(data.qpos[2]), initial_loads)
    final_loads = np.array([70, 80, 5, sum(model.body_mass) * np.linalg.norm(model.opt.gravity) - 155])
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
    parser = argparse.ArgumentParser(description="Paired hard hand-load constraints versus the unchanged load-feedback baseline")
    parser.add_argument("baseline_report", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    source = json.loads(args.baseline_report.read_text())
    assert source["status"] == "completed"
    for name, digest in source["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(source["model"]).read_bytes()).hexdigest() == source["model_sha256"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    initial_state, initial_ctrl = np.array(source["initial_physics_state"]), np.array(source["initial_ctrl"])
    destination = ROOT / "output/rl/directional_feedback" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources = dict(source["source_sha256"])
    sources["directional_feedback.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__, "model": source["model"], "model_sha256": source["model_sha256"],
              "initial_physics_state": initial_state.tolist(), "initial_ctrl": initial_ctrl.tolist(), "source_sha256": sources,
              "baseline_report": str(args.baseline_report.resolve()), "baseline_report_sha256": hashlib.sha256(args.baseline_report.read_bytes()).hexdigest(),
              "protocol": {"duration_s": 3, "prediction_s": .2, "replan_s": .1, "updates": 30, "trust_rad": .015,
                           "hand_origin_envelope_n": .5, "hand_endpoint_direction_tolerance_n": .25, "qp_maxiter": 100,
                           "max_prediction_rollouts_per_controller": 1530, "restarts": 0},
              "limitations": "One paired initial state; hard hand bounds do not prove useful COM transfer. Same objective, horizon, budget and physical model as baseline; only constrained local fit and nonlinear hand-direction acceptance added. No time extension, RL, lift/landing/crawl or hardware claim."}
    started = time.monotonic()
    try:
        for branch in ("hold", "feedback", "directional"):
            output = destination / f"{branch}.npz"
            result = guarded_run(model, initial_state, initial_ctrl, output) if branch == "directional" else baseline_run(model, initial_state, initial_ctrl, branch, output)
            if branch != "directional":
                old = next(run for run in source["runs"] if run["branch"] == branch)
                np.testing.assert_allclose(result["final_loads_n"], old["final_loads_n"], atol=1e-9, rtol=0)
                assert result["duration_s"] == old["duration_s"]
            result["direction_guard_passed"] = bool(result["trace"] and all(hand_envelope(row["loads_n"], result["initial_loads_n"]) for row in result["trace"]))
            report["runs"].append(result)
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key not in ("trace", "plans", "trajectory")}), flush=True)
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
