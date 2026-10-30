import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from component_tripod import TripodEquilibrium, hold
from component_tripod_staged import GEOMETRY_GROUPS, warm_start
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    for key in ("source_report", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    previous = json.loads(Path(report["source_report"]).read_text())
    assert digest(previous["source_report"]) == previous["source_report_sha256"]
    source = json.loads(Path(previous["source_report"]).read_text())
    path = Path(report["model"])
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    problem = TripodEquilibrium(model, source["qpos"], source["candidate_ctrl"])
    geometry = report["geometry"]
    assert report["protocol"]["geometry_groups"] == list(GEOMETRY_GROUPS)
    assert report["protocol"]["restarts"] == 0
    assert geometry["iterations"] <= report["protocol"]["geometry_maxiter"]
    values = np.array(geometry["values"])
    assert max(abs(values)) <= 1 + 1e-8
    measured = problem.evaluate(np.r_[values, problem.ctrl])
    np.testing.assert_allclose(measured["qpos"], geometry["qpos"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured["metrics"]["gaps_m"], geometry["gaps_m"], atol=1e-12, rtol=0)
    groups = {name: measured["constraint_group_minima"][name] for name in GEOMETRY_GROUPS}
    assert groups == geometry["constraint_group_minima"]
    assert geometry["passed"] == bool(min(groups.values()) >= -1e-5)
    if not geometry["passed"]:
        assert report["solution"] is None and report["hold"] is None
    else:
        reference = {name: getattr(problem, name).copy() for name in ("qpos", "ctrl", "xy", "normals", "com", "scale")}
        initial = warm_start(problem, geometry)
        np.testing.assert_array_equal(initial, report["native_initial_values"])
        for name, value in reference.items():
            np.testing.assert_array_equal(getattr(problem, name), value)
        assert max(abs(initial[:29])) <= 1 + 1e-8
        assert np.all(initial[29:] >= model.actuator_ctrlrange[:, 0]) and np.all(initial[29:] <= model.actuator_ctrlrange[:, 1])
        solution = report["solution"]
        assert solution["iterations"] <= report["protocol"]["native_maxiter"]
        values = np.array(solution["values"])
        assert max(abs(values[:29])) <= 1 + 1e-8
        assert np.all(values[29:] >= model.actuator_ctrlrange[:, 0]) and np.all(values[29:] <= model.actuator_ctrlrange[:, 1])
        measured = problem.evaluate(values)
        np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(measured["ctrl"], solution["ctrl"], atol=1e-12, rtol=0)
        assert measured["constraint_group_minima"] == solution["constraint_group_minima"]
        linear, angular = np.linalg.norm(measured["qacc"][:3]), np.max(np.abs(measured["qacc"][3:]))
        np.testing.assert_allclose(linear, solution["base_linear_mps2"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(angular, solution["angular_joint_max_radps2"], atol=1e-9, rtol=0)
        passed = bool(linear <= .01 and angular <= .1 and min(measured["geometry"]) >= -1e-5)
        assert solution["passed"] == passed
        if passed:
            with TemporaryDirectory() as folder:
                replayed = hold(model, solution, Path(folder) / "hold.npz")
                for key in ("passed", "failure", "duration_s", "peak_penetration_m", "trace"):
                    assert replayed[key] == report["hold"][key]
        else:
            assert report["hold"] is None and not (args.report.parent / "hold.npz").exists()
    print("PASS: geometric feasibility, unchanged original references/bounds, deterministic warm start, native failure gates and conditional hold isolation")


if __name__ == "__main__":
    main()
