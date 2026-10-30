import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from local_shift_path import GeometricPath, replay
from load_transfer import physical_metrics
from prepare import ROOT
from servo_force import validate_position_servos
from target_load_ablation import digest


class ContactPosePath(GeometricPath):
    def __init__(self, model, initial, shift):
        super().__init__(model, initial, shift)
        self.foot_id = model.geom("right_foot").id
        self.rotation = self.data.geom_xmat[self.foot_id].reshape(3, 3).copy()
        self.initial_gap = physical_metrics(model, self.data)["gaps_m"][3]

    def residual(self, values):
        base = super().residual(values)
        current = self.data.geom_xmat[self.foot_id].reshape(3, 3)
        return np.r_[base, (current - self.rotation).ravel() / .001]

    def jacobian(self, values):
        base = super().jacobian(values)
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, self.data, None, angular, self.foot_id)
        current = self.data.geom_xmat[self.foot_id].reshape(3, 3)
        rotation_jacobian = np.empty((9, len(values)))
        for index, column in enumerate(self.columns):
            rotation_jacobian[:, index] = np.column_stack(
                [np.cross(angular[:, column], current[:, axis]) for axis in range(3)]).ravel() / .001
        return np.vstack((base, rotation_jacobian))

    def solve(self, branch):
        result = super().solve(branch)
        for node in result["nodes"]:
            self.data.qpos[:] = node["qpos"]
            mujoco.mj_forward(self.model, self.data)
            current = self.data.geom_xmat[self.foot_id].reshape(3, 3)
            angle = float(np.arccos(np.clip((np.trace(self.rotation.T @ current) - 1) / 2, -1, 1)))
            drift = abs(node["gaps_m"][3] - self.initial_gap)
            node.update(right_foot_orientation_error_rad=angle, right_foot_gap_drift_m=drift)
            node["passed"] = bool(node["passed"] and angle <= .001 and drift <= .0001)
        result["passed"] = bool(result["passed"] and all(node["passed"] for node in result["nodes"]))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text())
    assert baseline["status"] == "completed"
    for name, expected in baseline["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert baseline["geometry_report_sha256"] == digest(args.geometry_report)
    geometry = json.loads(args.geometry_report.read_text())
    assert geometry["status"] == "completed" and geometry["geometric_target"]["passed"]
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    source = json.loads(Path(geometry["source_report"]).read_text())
    for name, expected in geometry["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["model"]) == source["model_sha256"] and mujoco.__version__ == source["mujoco"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    validate_position_servos(model)
    destination = ROOT / "output/rl/contact_pose_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "paths": [], "runs": [],
              "geometry_report": str(args.geometry_report.resolve()), "geometry_report_sha256": digest(args.geometry_report),
              "baseline_report": str(args.baseline_report.resolve()), "baseline_report_sha256": digest(args.baseline_report),
              "source_sha256": {**baseline["source_sha256"], Path(__file__).name: digest(__file__)},
              "protocol": {"nodes_per_path": 10, "max_nfev_per_node": 100, "replay_s": 6, "tail_s": 1,
                           "root_rotation": "fixed", "site_error_m": .001, "com_error_m": .001,
                           "right_foot_orientation_error_rad": .001, "right_foot_gap_drift_m": .0001,
                           "site_lift_m": .022, "required_final_gap_m": .015, "retries": 0},
              "limitations": "New geometric-path experiment, not single-variable load ablation. Fixed initial PD position offset is approximate, not inverse-dynamics torque allocation. IK is geometry only; intermediate joint interpolation and all dynamics require native replay. No imposed hand-direction bounds, no full crawl claim."}
    started = time.monotonic()
    try:
        for branch in ("hold", "shift_then_lift", "joint_shift_lift"):
            path = None
            if branch != "hold":
                planner = ContactPosePath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"])
                path = planner.solve(branch)
                report["paths"].append(path)
            result = replay(model, source, branch, path, destination / f"{branch}.npz") if path is None or path["passed"] else {"branch": branch, "executed": False, "reason": "geometry_gate_failed"}
            report["runs"].append(result)
            if branch == "hold":
                old = next(run for run in baseline["runs"] if run["branch"] == "hold")
                with np.load(old["trajectory"]) as expected, np.load(result["trajectory"]) as actual:
                    for key in ("qpos", "qvel", "ctrl"):
                        np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=0)
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
