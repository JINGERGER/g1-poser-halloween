import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from component_tripod import TripodEquilibrium, hold
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
    source = json.loads(Path(report["source_report"]).read_text())
    path = Path(report["model"])
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    problem = TripodEquilibrium(model, source["qpos"], source["candidate_ctrl"])
    np.testing.assert_array_equal(problem.scale, report["protocol"]["pose_delta_scale"])
    assert report["protocol"]["seed_count"] == 1 and report["protocol"]["restarts"] == 0
    solution = report["solution"]
    assert solution["iterations"] <= report["protocol"]["maxiter"]
    values = np.array(solution["values"])
    assert max(abs(values[:29])) <= 1 + 1e-8
    assert np.all(values[29:] >= model.actuator_ctrlrange[:, 0]) and np.all(values[29:] <= model.actuator_ctrlrange[:, 1])
    measured = problem.evaluate(values)
    np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured["ctrl"], solution["ctrl"], atol=1e-12, rtol=0)
    assert measured["constraint_group_minima"] == solution["constraint_group_minima"]
    np.testing.assert_allclose(measured["metrics"]["loads_n"], solution["native_metrics"]["loads_n"], atol=1e-9, rtol=0)
    linear = float(np.linalg.norm(measured["qacc"][:3]))
    angular = float(max(abs(measured["qacc"][3:])))
    np.testing.assert_allclose(linear, solution["base_linear_mps2"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(angular, solution["angular_joint_max_radps2"], atol=1e-9, rtol=0)
    passed = bool(linear <= .01 and angular <= .1 and min(measured["geometry"]) >= -1e-5)
    assert passed == solution["passed"]
    problem.evaluate(problem.initial)
    np.testing.assert_array_equal(problem.evaluate(values)["qacc"], measured["qacc"])
    problem.deadline = 0
    try:
        problem.evaluate(values)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Deadline ignored")
    if not passed:
        assert report["hold"] is None
        assert not (args.report.parent / "hold.npz").exists()
    else:
        recorded = report["hold"]
        assert digest(recorded["trajectory"]) == recorded["trajectory_sha256"]
        with TemporaryDirectory() as folder:
            replayed = hold(model, solution, Path(folder) / "hold.npz")
            for key in ("passed", "failure", "duration_s", "peak_penetration_m", "trace"):
                assert replayed[key] == recorded[key]
            with np.load(replayed["trajectory"], allow_pickle=False) as actual, np.load(recorded["trajectory"], allow_pickle=False) as saved:
                for key in ("qpos", "qvel", "ctrl"):
                    np.testing.assert_allclose(actual[key], saved[key], atol=1e-9, rtol=0)
    print("PASS: frozen model/seed, bounded candidate, native constraint and acceleration recomputation, timeout, scratch isolation and conditional hold gating")


if __name__ == "__main__":
    main()
