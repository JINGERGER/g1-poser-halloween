import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from load_transfer import physical_metrics, smooth
from prepare import ROOT, SUPPORTS
from servo_force import validate_position_servos
from target_load_ablation import digest


class ShiftEquilibrium:
    def __init__(self, model, qpos, ctrl):
        self.model, self.data = model, mujoco.MjData(model)
        self.qpos, self.ctrl = np.array(qpos), np.array(ctrl)
        self.ids = [model.geom(name).id for name in SUPPORTS]
        self.scale = np.r_[[.01, .01, .005], [.05 / np.sqrt(3)] * 3, [.1] * 23]
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(model, self.data)
        self.xy = self.data.geom_xpos[self.ids, :2].copy()
        self.normals = self.data.geom_xmat[self.ids[2:]].reshape(2, 3, 3)[:, :, 2].copy()
        self.com = self.data.subtree_com[1, :2].copy()
        self.initial = np.r_[np.zeros(model.nv), ctrl]
        self.deadline = float("inf")
        self.calls, self.cached = 0, None

    def evaluate(self, values):
        if time.monotonic() > self.deadline:
            raise TimeoutError("60s joint-equilibrium budget exhausted")
        if self.cached is not None and np.array_equal(values, self.cached[0]):
            return self.cached[1]
        self.calls += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos
        mujoco.mj_integratePos(self.model, self.data.qpos, values[:29] * self.scale, 1)
        self.data.ctrl[:] = values[29:]
        mujoco.mj_forward(self.model, self.data)
        row = physical_metrics(self.model, self.data)
        xy = self.data.geom_xpos[self.ids, :2]
        normals = self.data.geom_xmat[self.ids[2:]].reshape(2, 3, 3)[:, :, 2]
        joints = self.model.actuator_trnid[:, 0]
        floor = self.model.geom("floor").id
        unexpected = min((contact.dist for contact in self.data.contact
                          if not (floor in (contact.geom1, contact.geom2) and (contact.geom1 in self.ids or contact.geom2 in self.ids))), default=0)
        geometry = np.r_[(.00495 - row["penetration"]["depth_m"]) / .005,
                         (np.array(row["loads_n"]) - 5) / 100, -np.array(row["gaps_m"]) / .005,
                         1 - np.linalg.norm(xy - self.xy, axis=1) / .002,
                         1 - np.linalg.norm(normals - self.normals, axis=1) / .02,
                         1 - np.linalg.norm(self.data.subtree_com[1, :2] - self.com) / .005,
                         .95 - np.array(row["torque_ratios"]), (1 - row["other_total_n"]) / 100,
                         (unexpected + .001) / .001,
                         self.data.qpos[7:] - self.model.jnt_range[joints, 0],
                         self.model.jnt_range[joints, 1] - self.data.qpos[7:],
                         self.data.qpos[2] - .16, .65 - self.data.qpos[2]]
        limits = np.r_[[.005 / np.sqrt(3)] * 3, [.05] * 26]
        inequalities = np.r_[geometry, 1 - self.data.qacc / limits, 1 + self.data.qacc / limits]
        result = {"objective": float(np.mean(values[:29] ** 2) + .01 * np.mean((values[29:] - self.ctrl) ** 2)),
                  "ineq": inequalities, "geometry": geometry, "qpos": self.data.qpos.copy(), "ctrl": self.data.ctrl.copy(),
                  "qacc": self.data.qacc.copy(), "metrics": row, "com_xy": self.data.subtree_com[1, :2].copy()}
        self.cached = values.copy(), result
        return result

    def solve(self):
        self.deadline = time.monotonic() + 60
        last, history = self.initial.copy(), []

        def callback(values):
            nonlocal last
            last = values.copy()
            measured = self.evaluate(values)
            history.append(float(np.min(measured["ineq"])))

        try:
            result = minimize(lambda values: self.evaluate(values)["objective"], self.initial, method="SLSQP",
                              bounds=[(-1, 1)] * 29 + list(map(tuple, self.model.actuator_ctrlrange)),
                              constraints=[{"type": "ineq", "fun": lambda values: self.evaluate(values)["ineq"]}],
                              callback=callback, options={"maxiter": 100, "ftol": 1e-8, "eps": 1e-6})
            values, success, message = result.x, bool(result.success), str(result.message)
        except TimeoutError as error:
            values, success, message = last, False, str(error)
        self.deadline = float("inf")
        measured = self.evaluate(values)
        linear, angular = float(np.linalg.norm(measured["qacc"][:3])), float(np.max(np.abs(measured["qacc"][3:])))
        passed = linear <= .01 and angular <= .1 and np.min(measured["geometry"]) >= -1e-5
        return {"passed": bool(passed), "solver_success": success, "message": message, "values": values.tolist(),
                "qpos": measured["qpos"].tolist(), "ctrl": measured["ctrl"].tolist(), "native_metrics": measured["metrics"],
                "base_linear_mps2": linear, "angular_joint_max_radps2": angular,
                "geometry_min_scaled": float(np.min(measured["geometry"])), "optimizer_min_scaled": float(np.min(measured["ineq"])),
                "com_shift_from_reference_m": float(np.linalg.norm(measured["com_xy"] - self.com)),
                "evaluations": self.calls, "iterations": len(history), "history": history}


def replay(model, original, solution, branch, output):
    data = mujoco.MjData(model)
    if branch == "hypothetical_hold":
        data.qpos[:], data.ctrl[:] = solution["qpos"], solution["ctrl"]
        steps = 150
    else:
        mujoco.mj_setState(model, data, np.array(original["initial_physics_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = original["initial_ctrl"]
        steps = 300
    mujoco.mj_forward(model, data)
    initial_ctrl = data.ctrl.copy()
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    target = mujoco.MjData(model)
    target.qpos[:] = solution["qpos"]
    mujoco.mj_forward(model, target)
    desired_com = target.subtree_com[1, :2].copy()
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    peak, failure, start = physical_metrics(model, data)["penetration"]["depth_m"], None, data.time
    for step in range(steps):
        elapsed = (step + 1) * .02
        blend = 1 if branch == "hypothetical_hold" else smooth(elapsed / 3)
        desired = (1 - blend) * initial_ctrl + blend * np.array(solution["ctrl"])
        data.ctrl[:] += np.clip(desired - data.ctrl, -.01, .01)
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        for substep in range(round(.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and abs(data.time - start - elapsed) < 1e-7
        row = physical_metrics(model, data)
        row.update(elapsed_s=elapsed, com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - desired_com)),
                   velocity_max=float(np.max(np.abs(data.qvel))))
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        peak = max(peak, row["penetration"]["depth_m"])
        if peak > .005 or row["other_total_n"] > 1 or not .16 <= row["base_height_m"] <= .65:
            failure = "physical_guard"
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    safe = bool(failure is None and slips and np.mean(slips) <= .02 and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
    reached = bool(safe and len(rows) == steps and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05
                   and all(load > 5 for load in row["loads_n"]) and all(row["floor_contacts"]) for row in rows[-50:]))
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=state, initial_ctrl=initial_ctrl, dt=.02)
    return {"branch": branch, "duration_s": len(rows) * .02, "failure": failure, "physical_safety_passed": safe,
            "reached": reached, "max_penetration_m": peak, "mean_loaded_slip_mps": float(np.mean(slips)) if slips else None,
            "final_loads_n": rows[-1]["loads_n"], "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_report", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit_report.read_text())
    assert audit["status"] == "completed"
    for name, expected in audit["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(audit["source_report"]) == audit["source_report_sha256"]
    source = json.loads(Path(audit["source_report"]).read_text())
    baseline = json.loads(Path(source["baseline_report"]).read_text())
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    original = json.loads(Path(geometry["source_report"]).read_text())
    for container, key in ((source, "baseline_report"), (baseline, "geometry_report"), (geometry, "source_report"), (original, "model")):
        assert digest(container[key]) == container[key + "_sha256"]
    assert mujoco.__version__ == original["mujoco"]
    model = mujoco.MjModel.from_xml_path(original["model"])
    validate_position_servos(model)
    fixed = next(fit for fit in audit["fixed_pose_fits"] if fit["time_s"] == 3)
    destination = ROOT / "output/rl/joint_shift_equilibrium" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "runs": [], "solution": None,
              "audit_report": str(args.audit_report.resolve()), "audit_report_sha256": digest(args.audit_report),
              "source_sha256": {**audit["source_sha256"], Path(__file__).name: digest(__file__)},
              "protocol": {"reference_time_s": 3, "maxiter": 100, "timeout_s": 60, "restarts": 0,
                           "com_tolerance_m": .005, "contact_xy_tolerance_m": .002,
                           "hold_s": 3, "transfer_s": 6, "transfer_ramp_s": 3},
              "limitations": "One local four-support equilibrium near the geometric shift node, not unloading or crawling. Static initial hold and real-state motor transfer are separate. Pose/contact gaps and ctrl vary jointly; no force injection or relaxed hardware limits."}
    started = time.monotonic()
    try:
        report["solution"] = ShiftEquilibrium(model, fixed["qpos"], fixed["ctrl"]).solve()
        if report["solution"]["passed"]:
            for branch in ("hypothetical_hold", "real_transfer"):
                result = replay(model, original, report["solution"], branch, destination / f"{branch}.npz")
                report["runs"].append(result)
                if not result["reached"]:
                    break
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "solution": {key: value for key, value in (report["solution"] or {}).items() if key not in ("values", "qpos", "ctrl", "history", "native_metrics")},
                          "runs": [{key: value for key, value in run.items() if key != "trace"} for run in report["runs"]]}), flush=True)
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
