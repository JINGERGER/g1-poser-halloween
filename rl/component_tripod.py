import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from joint_shift_equilibrium import ShiftEquilibrium
from load_transfer import physical_metrics
from prepare import ROOT
from target_load_ablation import digest


class TripodEquilibrium(ShiftEquilibrium):
    def __init__(self, model, qpos, ctrl):
        super().__init__(model, qpos, ctrl)
        self.scale = np.r_[[.05, .05, .04], [.15 / np.sqrt(3)] * 3, [.4] * 23]

    def evaluate(self, values):
        if time.monotonic() > self.deadline:
            raise TimeoutError("60s tripod equilibrium budget exhausted")
        if self.cached is not None and np.array_equal(values, self.cached[0]):
            return self.cached[1]
        self.calls += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos
        mujoco.mj_integratePos(self.model, self.data.qpos, values[:29] * self.scale, 1)
        self.data.ctrl[:] = values[29:]
        mujoco.mj_forward(self.model, self.data)
        row = physical_metrics(self.model, self.data)
        floor = self.model.geom("floor").id
        unexpected = min((contact.dist for contact in self.data.contact
                          if not (floor in (contact.geom1, contact.geom2) and any(geom in self.ids[1:] for geom in (contact.geom1, contact.geom2)))), default=0)
        gaps, loads = np.array(row["gaps_m"]), np.array(row["loads_n"])
        normals = self.data.geom_xmat[self.ids[2:]].reshape(2, 3, 3)[:, :, 2]
        joints = self.model.actuator_trnid[:, 0]
        groups = {
            "left_clearance": np.array([(gaps[0] - .002) / .01, (.02 - gaps[0]) / .01]),
            "left_unloaded": np.array([(1 - loads[0]) / 100]),
            "remaining_loads": (loads[1:] - 5) / 100,
            "remaining_gaps": -gaps[1:] / .005,
            "penetration": np.array([(.00495 - row["penetration"]["depth_m"]) / .005]),
            "support_xy": 1 - np.linalg.norm(self.data.geom_xpos[self.ids[1:], :2] - self.xy[1:], axis=1) / .005,
            "foot_orientation": 1 - np.linalg.norm(normals - self.normals, axis=1) / .03,
            "com_motion": np.array([1 - np.linalg.norm(self.data.subtree_com[1, :2] - self.com) / .04]),
            "torque": .95 - np.array(row["torque_ratios"]),
            "other_load": np.array([(1 - row["other_total_n"]) / 100]),
            "unexpected_contact": np.array([(unexpected + .001) / .001]),
            "joint_lower": self.data.qpos[7:] - self.model.jnt_range[joints, 0],
            "joint_upper": self.model.jnt_range[joints, 1] - self.data.qpos[7:],
            "base_height": np.array([self.data.qpos[2] - .16, .65 - self.data.qpos[2]])}
        geometry = np.concatenate(list(groups.values()))
        limits = np.r_[[.005 / np.sqrt(3)] * 3, [.05] * 26]
        result = {"objective": float(np.mean(values[:29] ** 2) + .01 * np.mean((values[29:] - self.ctrl) ** 2)),
                  "geometry": geometry, "ineq": np.r_[geometry, 1 - self.data.qacc / limits, 1 + self.data.qacc / limits],
                  "qpos": self.data.qpos.copy(), "ctrl": self.data.ctrl.copy(), "qacc": self.data.qacc.copy(),
                  "metrics": row, "com_xy": self.data.subtree_com[1, :2].copy(),
                  "constraint_group_minima": {name: float(min(values)) for name, values in groups.items()}}
        self.cached = values.copy(), result
        return result


def hold(model, solution, output):
    data = mujoco.MjData(model)
    data.qpos[:] = solution["qpos"]
    data.ctrl[:] = solution["ctrl"]
    mujoco.mj_forward(model, data)
    target_com = data.subtree_com[1, :2].copy()
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    peak = physical_metrics(model, data)["penetration"]["depth_m"]
    failure = None
    for step in range(150):
        controls.append(data.ctrl.copy())
        for unused in range(round(.02 / model.opt.timestep)):
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            mujoco.mj_step(model, data)
            peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert abs(data.time - (step + 1) * .02) < 1e-8
        row = physical_metrics(model, data)
        row.update(com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - target_com)), velocity_max=float(max(abs(data.qvel))))
        peak = max(peak, row["penetration"]["depth_m"])
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        if peak > .005 or row["other_total_n"] > 1 or not .16 <= row["base_height_m"] <= .65:
            failure = "physical_guard"
        elif row["loads_n"][0] > 1 or row["floor_contacts"][0] or row["gaps_m"][0] < .001:
            failure = "left_hand_clearance_or_unload_lost"
        elif min(row["loads_n"][1:]) <= 5 or not all(row["floor_contacts"][1:]):
            failure = "remaining_support_lost"
        if failure:
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    passed = bool(failure is None and len(rows) == 150 and slips and np.mean(slips) <= .02
                  and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1
                  and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05 for row in rows[-50:]))
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, dt=.02)
    return {"passed": passed, "failure": failure, "duration_s": len(rows) * .02, "peak_penetration_m": peak,
            "trace": rows, "trajectory": str(output), "trajectory_sha256": digest(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("margin_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.margin_report.read_text())
    assert source["status"] == "completed" and source["optimization"]["success"]
    assert source["optimization"]["solution"]["normalized_margin"] > 0
    path = Path(source["model"])
    assert digest(path) == source["model_sha256"]
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    problem = TripodEquilibrium(model, source["qpos"], source["candidate_ctrl"])
    destination = ROOT / "output/rl/component_tripod" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.margin_report.resolve()), "source_report_sha256": digest(args.margin_report),
              "model": str(path), "model_sha256": digest(path), "training_updates": 0,
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "joint_shift_equilibrium.py", "load_transfer.py", "diagnose_short_crawl.py")},
              "protocol": {"seed_count": 1, "restarts": 0, "maxiter": 100, "budget_s": 60,
                           "pose_delta_scale": problem.scale.tolist(), "left_gap_m": [.002, .02],
                           "support_xy_tolerance_m": .005, "com_shift_max_m": .04, "conditional_hold_s": 3},
              "limitations": "Single local native tripod endpoint attempt, not dynamic reachability, a step or hardware proof. Frozen contact parameters and adjacent hull filtering retained; sensitivity failure unresolved.",
              "solution": None, "hold": None}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        solution = problem.solve()
        solution["constraint_group_minima"] = problem.evaluate(np.array(solution["values"]))["constraint_group_minima"]
        report["solution"] = solution
        if solution["passed"]:
            report["hold"] = hold(model, solution, destination / "hold.npz")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)
        if report["solution"]:
            print(json.dumps({key: value for key, value in report["solution"].items() if key not in ("values", "qpos", "ctrl", "history", "native_metrics")}))
            print("loads", report["solution"]["native_metrics"]["loads_n"], "gaps", report["solution"]["native_metrics"]["gaps_m"])
        if report["hold"]:
            print("hold", report["hold"]["passed"], report["hold"]["duration_s"])


if __name__ == "__main__":
    main()
