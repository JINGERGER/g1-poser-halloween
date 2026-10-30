import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from load_transfer import physical_metrics, smooth
from prepare import ROOT, SITES, SUPPORTS
from servo_force import validate_position_servos
from target_load_ablation import digest


def schedule(branch, elapsed):
    if branch == "hold":
        return 0., 0.
    if branch == "shift_then_lift":
        return smooth(elapsed / 3), smooth((elapsed - 3) / 2)
    if branch == "joint_shift_lift":
        return smooth(elapsed / 5), smooth(elapsed / 5)
    raise ValueError(branch)


class GeometricPath:
    def __init__(self, model, initial, shift):
        self.model, self.data = model, mujoco.MjData(model)
        self.initial = np.array(initial)
        self.data.qpos[:] = initial
        mujoco.mj_forward(model, self.data)
        self.sites = np.array([self.data.site(name).xpos.copy() for name in SITES])
        self.com = self.data.subtree_com[1, :2].copy()
        self.shift = np.array(shift)
        self.columns = np.r_[np.arange(3), np.arange(6, model.nv)]
        self.origin = np.r_[self.initial[:3], self.initial[7:]]
        joints = model.actuator_trnid[:, 0]
        self.lower = np.r_[self.initial[:3] - [.08, .08, .02], np.maximum(model.jnt_range[joints, 0], self.initial[7:] - .5)]
        self.upper = np.r_[self.initial[:3] + [.08, .08, .02], np.minimum(model.jnt_range[joints, 1], self.initial[7:] + .5)]
        self.target_sites, self.target_com = self.sites.copy(), self.com.copy()

    def place(self, values):
        self.data.qpos[:] = self.initial
        self.data.qpos[:3], self.data.qpos[7:] = values[:3], values[3:]
        mujoco.mj_forward(self.model, self.data)

    def residual(self, values):
        self.place(values)
        sites = np.array([self.data.site(name).xpos for name in SITES])
        return np.r_[(sites - self.target_sites).ravel() / .001,
                     (self.data.subtree_com[1, :2] - self.target_com) / .001,
                     (values - self.origin) * .01]

    def jacobian(self, values):
        self.place(values)
        blocks = []
        jacobian = np.zeros((3, self.model.nv))
        for name in SITES:
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.model.site(name).id)
            blocks.append(jacobian[:, self.columns].copy() / .001)
        mujoco.mj_jacSubtreeCom(self.model, self.data, jacobian, 1)
        blocks.append(jacobian[:2, self.columns].copy() / .001)
        blocks.append(np.eye(len(values)) * .01)
        return np.vstack(blocks)

    def solve(self, branch):
        previous = self.origin.copy()
        nodes = []
        for elapsed in np.linspace(.5, 5, 10):
            shift, lift = schedule(branch, elapsed)
            self.target_sites = self.sites.copy()
            self.target_sites[2, 2] += .022 * lift
            self.target_com = self.com + self.shift * shift
            result = least_squares(self.residual, previous, jac=self.jacobian, bounds=(self.lower, self.upper), max_nfev=100,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            self.place(result.x)
            sites = np.array([self.data.site(name).xpos for name in SITES])
            site_error = float(np.max(np.linalg.norm(sites - self.target_sites, axis=1)))
            com_error = float(np.linalg.norm(self.data.subtree_com[1, :2] - self.target_com))
            metrics = physical_metrics(self.model, self.data)
            support_ids = {self.model.geom(name).id for name in SUPPORTS}
            floor = self.model.geom("floor").id
            bad_contacts = [[self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name, float(contact.dist)]
                            for contact in self.data.contact if contact.dist < -.001
                            and not (floor in (contact.geom1, contact.geom2) and (contact.geom1 in support_ids or contact.geom2 in support_ids))]
            passed = bool(site_error <= .001 and com_error <= .001 and metrics["penetration"]["depth_m"] <= .005
                          and not bad_contacts and all(metrics["floor_contacts"][limb] for limb in (0, 1, 3))
                          and (elapsed < 5 or metrics["gaps_m"][2] >= .015))
            nodes.append({"time_s": float(elapsed), "passed": passed, "solver_success": bool(result.success), "message": result.message,
                          "nfev": result.nfev, "site_error_m": site_error, "com_error_m": com_error,
                          "penetration_m": metrics["penetration"]["depth_m"], "gaps_m": metrics["gaps_m"],
                          "unexpected_penetrations": bad_contacts, "qpos": self.data.qpos.tolist()})
            if not passed:
                break
            previous = result.x
        return {"branch": branch, "passed": len(nodes) == 10 and all(node["passed"] for node in nodes), "nodes": nodes}


def replay(model, source, branch, path, output):
    data = mujoco.MjData(model)
    state, initial_ctrl = np.array(source["initial_physics_state"]), np.array(source["initial_ctrl"])
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    bias = initial_ctrl - initial_qpos[7:]
    times = np.r_[0, [node["time_s"] for node in path["nodes"]]] if path else np.array([0., 5.])
    poses = np.array([initial_qpos] + [node["qpos"] for node in path["nodes"]]) if path else np.array([initial_qpos, initial_qpos])
    targets = poses[:, 7:] + bias
    if np.any(targets < model.actuator_ctrlrange[:, 0]) or np.any(targets > model.actuator_ctrlrange[:, 1]):
        return {"branch": branch, "executed": False, "reason": "biased_targets_exceed_ctrlrange"}
    rows, controls, positions, velocities = [], [], [initial_qpos], [data.qvel.copy()]
    peak, failure = physical_metrics(model, data)["penetration"]["depth_m"], None
    start = data.time
    for step in range(300):
        elapsed = (step + 1) * .02
        desired = np.array([np.interp(elapsed, times, targets[:, joint]) for joint in range(model.nu)])
        data.ctrl[:] += np.clip(desired - data.ctrl, -.01, .01)
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        for substep in range(round(.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and abs(data.time - start - elapsed) < 1e-7
        row = physical_metrics(model, data)
        row["elapsed_s"] = elapsed
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        peak = max(peak, row["penetration"]["depth_m"])
        if peak > .005 or row["other_total_n"] > 1 or not .16 <= row["base_height_m"] <= .65:
            failure = "physical_guard"
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows]))
    safe = bool(failure is None and slips and np.mean(slips) <= .02 and near_limit <= .1)
    tail = rows[-50:]
    supported = all(all(row["loads_n"][limb] > 5 and row["floor_contacts"][limb] for limb in (0, 1, 3)) for row in tail)
    unloaded = bool(safe and len(rows) == 300 and supported and all(row["loads_n"][2] <= 5 for row in tail))
    lifted = bool(unloaded and all(row["loads_n"][2] <= 1 and row["gaps_m"][2] >= .015 and not row["floor_contacts"][2] for row in tail))
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=state, initial_ctrl=initial_ctrl, dt=.02)
    return {"branch": branch, "executed": True, "duration_s": len(rows) * .02, "failure": failure,
            "physical_safety_passed": safe, "unload_passed": unloaded, "lift_passed": lifted,
            "max_penetration_m": peak, "mean_loaded_slip_mps": float(np.mean(slips)) if slips else None,
            "near_limit_fraction": near_limit, "final_loads_n": rows[-1]["loads_n"], "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_report", type=Path)
    args = parser.parse_args()
    geometry = json.loads(args.geometry_report.read_text())
    assert geometry["status"] == "completed" and geometry["geometric_target"]["passed"]
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    source = json.loads(Path(geometry["source_report"]).read_text())
    for name, expected in geometry["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["model"]) == source["model_sha256"] and mujoco.__version__ == source["mujoco"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    validate_position_servos(model)
    destination = ROOT / "output/rl/local_shift_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "paths": [], "runs": [],
              "geometry_report": str(args.geometry_report.resolve()), "geometry_report_sha256": digest(args.geometry_report),
              "source_sha256": {**geometry["source_sha256"], Path(__file__).name: digest(__file__)},
              "protocol": {"nodes_per_path": 10, "max_nfev_per_node": 100, "replay_s": 6, "tail_s": 1,
                           "root_rotation": "fixed", "site_error_m": .001, "com_error_m": .001,
                           "site_lift_m": .022, "required_final_gap_m": .015, "retries": 0},
              "limitations": "New geometric-path experiment, not single-variable load ablation. Fixed initial PD position offset is approximate, not inverse-dynamics torque allocation. IK is geometry only; intermediate joint interpolation and all dynamics require native replay. No imposed hand-direction bounds, no full crawl claim."}
    started = time.monotonic()
    try:
        for branch in ("hold", "shift_then_lift", "joint_shift_lift"):
            path = None
            if branch != "hold":
                planner = GeometricPath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"])
                path = planner.solve(branch)
                report["paths"].append(path)
            result = replay(model, source, branch, path, destination / f"{branch}.npz") if path is None or path["passed"] else {"branch": branch, "executed": False, "reason": "geometry_gate_failed"}
            report["runs"].append(result)
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"path": path, "run": {key: value for key, value in result.items() if key != "trace"}}), flush=True)
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
