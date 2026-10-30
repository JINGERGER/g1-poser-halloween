import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from component_unload_return import BRANCHES, envelope, run
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    np.testing.assert_allclose([envelope(stamp, 1) for stamp in (0, .6, 1, 1.5, 2, 4)], [0, 1, 1, .5, 0, 0], atol=1e-12)
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["protocol"]["restarts"] == 0
    assert report["protocol"]["branches"] == [list(branch) for branch in BRANCHES]
    for key in ("source_report", "model", "source_trajectory"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    source = json.loads(Path(report["source_report"]).read_text())
    np.testing.assert_array_equal(report["candidate_ctrl"], source["candidate_ctrl"])
    path = Path(report["model"])
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    steps = 0
    assert [entry["branch"] for entry in report["runs"]] == [branch[0] for branch in BRANCHES]
    with TemporaryDirectory() as folder, np.load(report["source_trajectory"], allow_pickle=False) as initial:
        for branch, recorded in zip(BRANCHES, report["runs"]):
            assert digest(recorded["trajectory"]) == recorded["trajectory_sha256"]
            actual = run(model, initial, np.array(report["candidate_ctrl"]), branch, Path(folder) / f"{branch[0]}.npz")
            for key in ("completed", "failure", "duration_s", "peak_penetration_m", "trace", "initial_metrics"):
                assert actual[key] == recorded[key]
            prefix = next(entry for entry in source["runs"] if entry["amplitude"] == branch[1])
            assert digest(prefix["trajectory"]) == prefix["trajectory_sha256"]
            with np.load(recorded["trajectory"], allow_pickle=False) as saved, np.load(actual["trajectory"], allow_pickle=False) as replayed, np.load(prefix["trajectory"], allow_pickle=False) as old:
                np.testing.assert_array_equal(saved["initial_physics_state"], initial["initial_physics_state"])
                np.testing.assert_array_equal(saved["initial_ctrl"], initial["initial_ctrl"])
                for key in ("qpos", "qvel", "ctrl"):
                    np.testing.assert_allclose(saved[key], replayed[key], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(saved[key][:len(old[key])], old[key], atol=1e-9, rtol=0)
                changes = np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)
                assert max(abs(changes).ravel()) <= .0100000001
                if recorded["completed"]:
                    np.testing.assert_allclose(saved["ctrl"][-1], initial["initial_ctrl"], atol=1e-12, rtol=0)
                steps += len(saved["ctrl"])
            rows, baseline = recorded["trace"], report["runs"][0]["trace"]
            start = branch[2]
            hold_complete = recorded["duration_s"] >= start and report["runs"][0]["duration_s"] >= start
            assert hold_complete == recorded["pre_return_window_completed"]
            if hold_complete:
                indices = [index for index, row in enumerate(rows) if start - .2 < row["elapsed_s"] <= start]
                reduction = 1 - np.mean([rows[index]["loads_n"][0] for index in indices]) / np.mean([baseline[index]["loads_n"][0] for index in indices])
                np.testing.assert_allclose(reduction, recorded["pre_return_reduction_fraction"], atol=1e-12, rtol=0)
            if not recorded["completed"]:
                assert recorded["recovery_passed"] is False
                continue
            indices = [index for index, row in enumerate(rows) if row["elapsed_s"] > branch[3] - .5]
            passed = all(np.max(np.abs(np.array(rows[index]["loads_n"]) - baseline[index]["loads_n"])) <= 5
                         and np.linalg.norm(np.array(rows[index]["com_xy"]) - baseline[index]["com_xy"]) <= .005
                         and rows[index]["velocity_max"] <= .05 and all(rows[index]["floor_contacts"])
                         and min(rows[index]["loads_n"]) > 5 and rows[index]["loaded_slip_mps"] is not None
                         and rows[index]["loaded_slip_mps"] <= .02 for index in indices)
            assert recorded["recovery_passed"] == passed
    print(f"PASS: fixed return schedules, matched prior prefixes, slew limits, {steps} native steps and independent recovery gates")


if __name__ == "__main__":
    main()
