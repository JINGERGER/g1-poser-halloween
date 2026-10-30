import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from load_feedback import Predictor, constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    row = {"loads_n": [11, 11, 0, 11], "other_total_n": 0, "base_height_m": .25}
    assert constraints(row, .0049, True)
    assert not constraints(row, .00496, True) and constraints(row, .00496)
    assert not constraints(row, .00501)
    row["loads_n"][0] = 10
    assert not constraints(row, .004)
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and [run["branch"] for run in report["runs"]] == ["hold", "feedback"]
        source_path = Path(report["transfer_report"])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["transfer_report_sha256"]
        source = json.loads(source_path.read_text())
        source_trajectory = Path(source["replay"]["trajectory"])
        assert hashlib.sha256(source_trajectory.read_bytes()).hexdigest() == report["source_trajectory_sha256"]
        source_saved = np.load(source_trajectory, allow_pickle=False)
        assert hashlib.sha256(Path(report["model"]).read_bytes()).hexdigest() == report["model_sha256"]
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        model = mujoco.MjModel.from_xml_path(report["model"])
        for run in report["runs"]:
            assert run["prediction_rollouts"] <= 1530 and len(run["plans"]) <= 30
            saved = np.load(run["trajectory"], allow_pickle=False)
            np.testing.assert_allclose(saved["qpos"][0], source_saved["qpos"][-1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(saved["qvel"][0], source_saved["qvel"][-1], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(saved["initial_physics_state"], report["initial_physics_state"])
            np.testing.assert_array_equal(saved["initial_ctrl"], report["initial_ctrl"])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            if run["branch"] == "hold":
                np.testing.assert_array_equal(saved["ctrl"], np.broadcast_to(saved["initial_ctrl"], saved["ctrl"].shape))
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            for index, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
            if run["trace"]:
                assert run["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in run["trace"])
            if run["unload_passed"]:
                assert run["physical_safety_passed"] and run["duration_s"] == 3
                assert all(row["loads_n"][2] <= 5 for row in run["trace"][-10:])
        initial = np.array(report["initial_physics_state"])
        ctrl = np.array(report["initial_ctrl"])
        predictor = Predictor(model, .25)
        baseline = predictor.rollout(initial, ctrl, ctrl)
        predictor.rollout(initial, ctrl, ctrl + .001)
        repeated = predictor.rollout(initial, ctrl, ctrl)
        np.testing.assert_array_equal(baseline[0], repeated[0])
        assert baseline[1:] == repeated[1:]
        print("Verified paired initial state, finite budgets, native replay and scratch isolation")
    print("PASS: load-feedback safety gates")


if __name__ == "__main__":
    main()
