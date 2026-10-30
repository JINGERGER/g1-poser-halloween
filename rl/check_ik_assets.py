import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from ik_assets import FRAMES, PERIOD, foot_path, motion_time, reference_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.assets / "flat.xml"))
    with np.load(args.assets / "ik_reference.npz", allow_pickle=False) as source:
        table = source["offsets"]
        assert table.shape == (2, FRAMES, 23) and np.isfinite(table).all()
        np.testing.assert_allclose(table[:, 0], 0, atol=1e-10)
        anchors = np.array([0, 1])
        for elapsed in (0, 0.2, 0.9, 1):
            np.testing.assert_allclose(reference_at(table, anchors, np.array([elapsed, elapsed])), 0, atol=1e-10)
        for elapsed in (2.0, 3.7, 5.5):
            np.testing.assert_allclose(reference_at(table, anchors, elapsed), reference_at(table, anchors, elapsed + PERIOD), atol=1e-10)
        np.testing.assert_allclose(reference_at(table, anchors, 5.5 - 1e-7), reference_at(table, anchors, 5.5 + 1e-7), atol=1e-6)
        limits = model.jnt_range[model.actuator_trnid[:, 0]]
        joints = source["qpos"][:, None, 7:] + table
        assert np.all(joints >= limits[:, 0] - 1e-9) and np.all(joints <= limits[:, 1] + 1e-9)
        assert np.any(source["qvel"] != 0)
    assert np.all(np.diff(motion_time(np.linspace(0, 20, 2001))) >= 0)
    for boundary in (0, 0.25, 0.5, 0.75, 1):
        for left, right in zip(foot_path(boundary - 1e-8)[:2], foot_path(boundary + 1e-8)[:2]):
            np.testing.assert_allclose(left, right, atol=1e-7)
    plan = json.loads(subprocess.check_output([sys.executable, "-B", str(Path(__file__).with_name("run_ik_screen.py")), str(args.assets), "--dry-run"]))
    assert plan["total_planned_transitions"] == 614400 and plan["transitions_per_seed"] == 307200
    assert plan["seeds"] == [1, 2] and plan["episodes_per_evaluation"] == 100
    print("PASS: exact loaded start, nonzero reset velocities, periodic/continuous reference, joint limits, monotonic startup clock and frozen two-seed budget")
    if args.matrix:
        matrix = json.loads(args.matrix.read_text())
        assert matrix["status"] == "completed" and len(matrix["runs"]) == 2 and len(matrix["evaluations"]) == 4
        for path, digest in matrix["source_and_asset_hashes"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        paired = None
        for row in matrix["runs"]:
            result = json.loads(Path(row["training_result"]).read_text())
            assert result["status"] == "completed" and result["recorded_transitions"] == 307200
            assert len(result["learner_iterations"]) == 400
            for iteration in result["learner_iterations"]:
                assert iteration["updates_applied"] > 0
                assert not any(value for key, value in iteration.items() if key.startswith(("skipped_nonfinite", "rolled_back")))
        for row in matrix["evaluations"]:
            result = json.loads(Path(row["report"]).read_text())
            actual = [episode["initial_state_sha256"] for episode in result["episodes"]]
            assert len(actual) == 100 and (paired is None or paired == actual)
            paired = actual
            assert result["max_native_state_error"] < 2e-4
        print("PASS: completed budgets, all finite optimizer updates, frozen files and four paired 100-episode native evaluations")


if __name__ == "__main__":
    main()
