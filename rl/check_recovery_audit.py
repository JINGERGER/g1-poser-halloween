import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from com_feedback import constraints
from load_transfer import physical_metrics
from recovery_audit import inspect_rollout
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for key in ("feedback_report", "source_trajectory"):
        assert digest(report[key]) == report[key + "_sha256"]
    source = json.loads(Path(report["feedback_report"]).read_text())
    assert digest(source["model"]) == source["model_sha256"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    data = mujoco.MjData(model)
    with np.load(report["source_trajectory"], allow_pickle=False) as saved:
        mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = saved["initial_ctrl"]
        mujoco.mj_forward(model, data)
        for index, control in enumerate(saved["ctrl"]):
            data.ctrl[:] = control
            for substep in range(round(.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
    state = np.empty(len(report["initial_physics_state"]))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.testing.assert_array_equal(state, report["initial_physics_state"])
    np.testing.assert_array_equal(data.ctrl, report["initial_ctrl"])
    np.testing.assert_array_equal(report["target_com_xy_m"], source["target_com_xy_m"])
    assert len(report["candidates"]) == 6
    assert [candidate["branch"] for candidate in report["candidates"]] == ["hold_current", "rejected_scale_1.0", "rejected_scale_0.5", "rejected_scale_0.25", "rejected_scale_0.125", "rollback_previous"]
    for candidate in report["candidates"]:
        assert candidate["duration_s"] == .2 and len(candidate["trace"]) == 10
        target = np.array(candidate["target_ctrl"])
        assert np.all(target >= model.actuator_ctrlrange[:, 0]) and np.all(target <= model.actuator_ctrlrange[:, 1])
        if candidate["branch"] == "hold_current":
            np.testing.assert_array_equal(target, report["initial_ctrl"])
        if candidate["branch"] == "rollback_previous":
            np.testing.assert_array_equal(target, source["initial_ctrl"])
        replay = inspect_rollout(model, state, np.array(report["initial_ctrl"]), target, np.array(report["target_com_xy_m"]), .2, False)
        for key in ("prediction_safe", "physical_safety_passed", "first_violation_s", "minimum_margins"):
            assert replay[key] == candidate[key]
        np.testing.assert_array_equal(replay["final_physics_state"], candidate["final_physics_state"])
        peak = candidate["max_penetration_m"]
        independently_safe = all(constraints(row, peak, True) and max(row["torque_ratios"]) < .95 for row in candidate["trace"])
        assert candidate["prediction_safe"] == independently_safe
        for name in candidate["first_violation_s"]:
            if name in candidate["minimum_margins"]:
                assert candidate["minimum_margins"][name] <= 0
        for actual, expected in zip(replay["trace"], candidate["trace"]):
            np.testing.assert_allclose(actual["loads_n"], expected["loads_n"], atol=1e-9, rtol=0)
    eligible = {candidate["branch"] for candidate in report["candidates"] if candidate["branch"] in ("hold_current", "rollback_previous") and candidate["prediction_safe"]}
    assert {run["branch"] for run in report["recoveries"]} == eligible
    for run in report["recoveries"]:
        assert run["recovered"] == bool(run["settled_with_margin"] and run["terminal_prediction_safe"])
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["initial_physics_state"], state)
            np.testing.assert_array_equal(saved["initial_ctrl"], report["initial_ctrl"])
            assert len(saved["ctrl"]) <= 150
    print("PASS: frozen inputs, true stopped state, six fixed candidates, native diagnostic reproduction, margins and guarded recovery eligibility")


if __name__ == "__main__":
    main()
