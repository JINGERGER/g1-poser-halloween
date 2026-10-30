import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from component_tripod import TripodEquilibrium, hold
from prepare import ROOT
from target_load_ablation import digest


GEOMETRY_GROUPS = ("left_clearance", "remaining_gaps", "penetration", "support_xy", "foot_orientation",
                   "com_motion", "unexpected_contact", "joint_lower", "joint_upper", "base_height")


def solve_geometry(problem):
    problem.deadline = time.monotonic() + 30
    last = np.zeros(problem.model.nv)
    iterations = 0

    def evaluate(values):
        measured = problem.evaluate(np.r_[values, problem.ctrl])
        return np.array([measured["constraint_group_minima"][name] for name in GEOMETRY_GROUPS])

    def callback(values):
        nonlocal last, iterations
        last = values.copy()
        iterations += 1

    try:
        result = minimize(lambda values: float(np.mean(values ** 2)), last, method="SLSQP",
                          bounds=[(-1, 1)] * problem.model.nv, constraints=[{"type": "ineq", "fun": evaluate}],
                          callback=callback, options={"maxiter": 100, "ftol": 1e-8, "eps": 1e-6})
        last, success, message = result.x, bool(result.success), str(result.message)
    except TimeoutError:
        success, message = False, "30s geometry budget exhausted"
    problem.deadline = float("inf")
    measured = problem.evaluate(np.r_[last, problem.ctrl])
    groups = {name: measured["constraint_group_minima"][name] for name in GEOMETRY_GROUPS}
    return {"passed": bool(min(groups.values()) >= -1e-5), "solver_success": success, "message": message,
            "values": last.tolist(), "qpos": measured["qpos"].tolist(), "constraint_group_minima": groups,
            "gaps_m": measured["metrics"]["gaps_m"], "iterations": iterations, "evaluations": problem.calls}


def warm_start(problem, geometry):
    positions = np.array(geometry["qpos"])[7:]
    ctrl = np.clip(problem.ctrl + positions - problem.qpos[7:], problem.model.actuator_ctrlrange[:, 0], problem.model.actuator_ctrlrange[:, 1])
    problem.initial = np.r_[geometry["values"], ctrl]
    return problem.initial.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tripod_report", type=Path)
    args = parser.parse_args()
    previous = json.loads(args.tripod_report.read_text())
    assert previous["status"] == "completed"
    assert digest(previous["source_report"]) == previous["source_report_sha256"]
    source = json.loads(Path(previous["source_report"]).read_text())
    path = Path(previous["model"])
    assert digest(path) == previous["model_sha256"] == source["model_sha256"]
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    problem = TripodEquilibrium(model, source["qpos"], source["candidate_ctrl"])
    destination = ROOT / "output/rl/component_tripod_staged" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.tripod_report.resolve()), "source_report_sha256": digest(args.tripod_report),
              "model": str(path), "model_sha256": digest(path), "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
              (Path(__file__).name, "component_tripod.py", "joint_shift_equilibrium.py", "load_transfer.py", "diagnose_short_crawl.py")},
              "protocol": {"geometry_groups": GEOMETRY_GROUPS, "geometry_maxiter": 100, "geometry_budget_s": 30,
                           "native_maxiter": 100, "native_budget_s": 60, "restarts": 0, "conditional_hold_s": 3,
                           "reference_pose_unchanged": True, "bounds_unchanged": True},
              "limitations": "Geometry phase omits force/acceleration conditions, never proof of support. Native phase restores all original constraints and reference frame/bounds; no recentering around geometry result. One conditional initialized hold, not dynamic transition or hardware proof.",
              "geometry": None, "solution": None, "hold": None, "training_updates": 0}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["geometry"] = solve_geometry(problem)
        if report["geometry"]["passed"]:
            report["native_initial_values"] = warm_start(problem, report["geometry"]).tolist()
            problem.calls, problem.cached = 0, None
            report["solution"] = problem.solve()
            report["solution"]["constraint_group_minima"] = problem.evaluate(np.array(report["solution"]["values"]))["constraint_group_minima"]
            if report["solution"]["passed"]:
                report["hold"] = hold(model, report["solution"], destination / "hold.npz")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)
        for name in ("geometry", "solution", "hold"):
            print(name, json.dumps({key: value for key, value in (report[name] or {}).items()
                                   if key not in ("values", "qpos", "ctrl", "native_metrics", "history", "trace")}))


if __name__ == "__main__":
    main()
