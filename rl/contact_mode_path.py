import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from local_shift_path import replay, schedule
from load_transfer import physical_metrics
from prepare import ROOT, SUPPORTS
from servo_force import validate_position_servos
from target_load_ablation import digest


class ContactModePath:
    def __init__(self, model, initial, shift, rotating):
        self.model, self.data = model, mujoco.MjData(model)
        self.initial, self.shift = np.array(initial), np.array(shift)
        self.rotating = rotating
        self.columns = np.arange(model.nv) if rotating else np.r_[np.arange(3), np.arange(6, model.nv)]
        self.geom_ids = [model.geom(name).id for name in SUPPORTS]
        assert list(model.geom_type[self.geom_ids]) == [2, 2, 6, 6]
        self.place(np.zeros(len(self.columns)))
        self.xy = self.data.geom_xpos[self.geom_ids, :2].copy()
        self.normals = self.data.geom_xmat[self.geom_ids[2:]].reshape(2, 3, 3)[:, :, 2].copy()
        self.com = self.data.subtree_com[1, :2].copy()
        self.gaps = self.contact_values()[1]
        self.target_gaps, self.target_com = self.gaps.copy(), self.com.copy()
        limits = np.r_[[.08, .08, .02], [.15 / np.sqrt(3)] * 3, [.5] * 23]
        lower, upper = -limits, limits.copy()
        joints = model.actuator_trnid[:, 0]
        lower[6:] = np.maximum(lower[6:], model.jnt_range[joints, 0] - self.initial[7:])
        upper[6:] = np.minimum(upper[6:], model.jnt_range[joints, 1] - self.initial[7:])
        self.lower, self.upper = lower[self.columns], upper[self.columns]

    def configuration(self, values):
        displacement = np.zeros(self.model.nv)
        displacement[self.columns] = values
        qpos = self.initial.copy()
        mujoco.mj_integratePos(self.model, qpos, displacement, 1)
        return qpos

    def place(self, values):
        self.data.qpos[:] = self.configuration(values)
        mujoco.mj_forward(self.model, self.data)

    def contact_values(self):
        centers = self.data.geom_xpos[self.geom_ids]
        rotations = self.data.geom_xmat[self.geom_ids].reshape(4, 3, 3)
        sizes = self.model.geom_size[self.geom_ids]
        assert abs(self.data.geom("floor").xpos[2]) < 1e-12
        np.testing.assert_allclose(self.data.geom("floor").xmat.reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-12)
        gaps = centers[:, 2].copy()
        gaps[:2] -= sizes[:2, 0]
        gaps[2:] -= np.sum(np.abs(rotations[2:, 2, :]) * sizes[2:], axis=1)
        return centers, gaps, rotations

    def residual(self, values):
        self.place(values)
        centers, gaps, rotations = self.contact_values()
        return np.r_[(centers[:, :2] - self.xy).ravel() / .001,
                     (gaps - self.target_gaps) / .001,
                     (rotations[2:, :, 2] - self.normals).ravel() / .001,
                     (self.data.subtree_com[1, :2] - self.target_com) / .001, values * .01]

    def jacobian(self, values):
        self.place(values)
        transform = np.eye(self.model.nv)[:, self.columns]
        if self.rotating:
            for index in range(3, 6):
                offset = np.zeros_like(values)
                offset[index] = 1e-7
                mujoco.mj_differentiatePos(self.model, transform[:, index], 2e-7,
                                         self.configuration(values - offset), self.configuration(values + offset))
        centers, gaps, rotations = self.contact_values()
        horizontal, gap_rows, normal_rows = [], [], []
        linear, angular = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for limb, geom in enumerate(self.geom_ids):
            mujoco.mj_jacGeom(self.model, self.data, linear, angular, geom)
            horizontal.append(linear[:2].copy())
            gap_row = linear[2].copy()
            if limb >= 2:
                derivatives = np.stack([np.cross(angular.T, rotations[limb, :, axis]).T for axis in range(3)])
                gap_row -= np.sum((np.sign(rotations[limb, 2]) * self.model.geom_size[geom])[:, None] * derivatives[:, 2, :], axis=0)
                normal_rows.append(derivatives[2])
            gap_rows.append(gap_row)
        mujoco.mj_jacSubtreeCom(self.model, self.data, linear, 1)
        return np.vstack((np.vstack((*horizontal, gap_rows, *normal_rows, linear[:2])) @ transform / .001, np.eye(len(values)) * .01))

    def inspect(self, values, elapsed, branch):
        shift, lift = schedule(branch, elapsed)
        self.target_com = self.com + self.shift * shift
        self.target_gaps = self.gaps.copy()
        self.target_gaps[2] = self.gaps[2] * (1 - lift) + .02 * lift
        residual = self.residual(values)
        row = physical_metrics(self.model, self.data)
        floor = self.model.geom("floor").id
        unexpected = [[self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name, float(contact.dist)]
                      for contact in self.data.contact if contact.dist < -.001
                      and not (floor in (contact.geom1, contact.geom2) and (contact.geom1 in self.geom_ids or contact.geom2 in self.geom_ids))]
        stance = (0, 1, 2, 3) if lift == 0 else (0, 1, 3)
        errors = {"xy_error_m": float(np.max(np.linalg.norm(residual[:8].reshape(4, 2) * .001, axis=1))),
                  "gap_error_m": float(np.max(np.abs(residual[8:12])) * .001),
                  "normal_error": float(np.max(np.linalg.norm(residual[12:18].reshape(2, 3) * .001, axis=1))),
                  "com_error_m": float(np.linalg.norm(residual[18:20]) * .001)}
        passed = (errors["xy_error_m"] <= .001 and errors["gap_error_m"] <= .0001 and errors["normal_error"] <= .001
                  and errors["com_error_m"] <= .001 and row["penetration"]["depth_m"] <= .005 and not unexpected
                  and all(row["floor_contacts"][limb] for limb in stance) and (elapsed < 5 or row["gaps_m"][2] >= .015))
        return {**errors, "passed": bool(passed), "time_s": float(elapsed), "stance": list(stance), "values": values.tolist(),
                "qpos": self.data.qpos.tolist(), "gaps_m": row["gaps_m"], "target_gaps_m": self.target_gaps.tolist(),
                "penetration_m": row["penetration"]["depth_m"], "unexpected_penetrations": unexpected}

    def solve(self, branch):
        previous = np.zeros(len(self.columns))
        nodes = []
        for elapsed in np.linspace(.5, 5, 10):
            self.inspect(previous, elapsed, branch)
            fitted = least_squares(self.residual, previous, jac=self.jacobian, bounds=(self.lower, self.upper), max_nfev=100,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            row = self.inspect(fitted.x, elapsed, branch)
            row.update(nfev=fitted.nfev, solver_success=bool(fitted.success), message=str(fitted.message))
            nodes.append(row)
            if not row["passed"]:
                break
            previous = fitted.x
        return {"branch": branch, "rotating": self.rotating, "passed": len(nodes) == 10 and all(row["passed"] for row in nodes), "nodes": nodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_report", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text())
    assert baseline["status"] == "completed"
    for name, expected in baseline["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(baseline["geometry_report"]) == baseline["geometry_report_sha256"]
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    source = json.loads(Path(geometry["source_report"]).read_text())
    assert digest(source["model"]) == source["model_sha256"] and mujoco.__version__ == source["mujoco"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    validate_position_servos(model)
    destination = ROOT / "output/rl/contact_mode_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "paths": [], "runs": [], "training_updates": 0,
              "baseline_report": str(args.baseline_report.resolve()), "baseline_report_sha256": digest(args.baseline_report),
              "source_sha256": {**baseline["source_sha256"], Path(__file__).name: digest(__file__)},
              "protocol": {"schedules": ["shift_then_lift", "joint_shift_lift"], "root_rotation": [False, True],
                           "rotation_component_bound_rad": .15 / np.sqrt(3), "nodes_per_path": 10, "max_nfev_per_node": 100,
                           "replay_s": 6, "restarts": 0, "clearance_target_m": .02},
              "limitations": "Geometry first; sphere centers and box floor gaps/normals, not fixed full foot rotations. Root rotation is a scratch planning freedom, never a live pose command. Constant initial PD bias remains approximate. Residual tolerances are independently gated; finite nodes do not prove continuous feasibility or dynamics."}
    started = time.monotonic()
    try:
        hold = replay(model, source, "hold", None, destination / "hold.npz")
        report["runs"].append(hold)
        old = next(run for run in baseline["runs"] if run["branch"] == "hold")
        with np.load(hold["trajectory"]) as actual, np.load(old["trajectory"]) as expected:
            for key in ("ctrl", "qpos", "qvel"):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=0)
        for rotating in (False, True):
            for branch in report["protocol"]["schedules"]:
                planner = ContactModePath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"], rotating)
                path = planner.solve(branch)
                report["paths"].append(path)
                name = f"{branch}_rotation{int(rotating)}"
                run = replay(model, source, name, path, destination / f"{name}.npz") if path["passed"] else {"branch": name, "executed": False, "reason": "geometry_gate_failed"}
                if run["executed"]:
                    phase = all(all(row["floor_contacts"][limb] for limb in ((0, 1, 2, 3) if schedule(branch, row["elapsed_s"])[1] == 0 else (0, 1, 3))) for row in run["trace"])
                    run["contact_phase_passed"] = phase
                    run["lift_passed"] = bool(run["lift_passed"] and phase)
                report["runs"].append(run)
                print(json.dumps({"branch": name, "geometry_passed": path["passed"], "last_node": path["nodes"][-1], "run": {key: value for key, value in run.items() if key != "trace"}}), flush=True)
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
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
