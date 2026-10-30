import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from contact_mode_path import ContactModePath
from local_shift_path import replay, schedule
from prepare import ROOT
from servo_force import validate_position_servos
from target_load_ablation import digest


class ControlFeasiblePath(ContactModePath):
    def __init__(self, model, initial, shift, rotating, initial_ctrl):
        super().__init__(model, initial, shift, rotating)
        initial_ctrl = np.array(initial_ctrl)
        offset = 6 if rotating else 3
        self.lower[offset:] = np.maximum(self.lower[offset:], model.actuator_ctrlrange[:, 0] - initial_ctrl + 1e-8)
        self.upper[offset:] = np.minimum(self.upper[offset:], model.actuator_ctrlrange[:, 1] - initial_ctrl - 1e-8)
        assert np.all(self.lower <= 0) and np.all(self.upper >= 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("geometry_comparison", type=Path)
    args = parser.parse_args()
    comparison = json.loads(args.geometry_comparison.read_text())
    assert comparison["status"] == "completed" and comparison["baseline_report_sha256"] == digest(args.baseline_report)
    for name, expected in comparison["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
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
    destination = ROOT / "output/rl/control_feasible_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "paths": [], "runs": [], "training_updates": 0,
              "baseline_report": str(args.baseline_report.resolve()), "baseline_report_sha256": digest(args.baseline_report),
              "geometry_comparison": str(args.geometry_comparison.resolve()), "geometry_comparison_sha256": digest(args.geometry_comparison),
              "source_sha256": {**comparison["source_sha256"], Path(__file__).name: digest(__file__)},
              "protocol": {"schedules": ["shift_then_lift", "joint_shift_lift"], "root_rotation": [False, True],
                           "rotation_component_bound_rad": .15 / np.sqrt(3), "nodes_per_path": 10, "max_nfev_per_node": 100,
                           "replay_s": 6, "restarts": 0, "clearance_target_m": .02, "ctrl_bound_margin_rad": 1e-8},
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
                planner = ControlFeasiblePath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"], rotating, source["initial_ctrl"])
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
