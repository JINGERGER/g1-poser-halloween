import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from com_feedback import ComPredictor, constraints
from load_transfer import physical_metrics
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for key in ("transfer_report", "source_trajectory", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    source = json.loads(Path(report["transfer_report"]).read_text())
    model = mujoco.MjModel.from_xml_path(report["model"])
    target = mujoco.MjData(model)
    target.qpos[:] = source["solution"]["qpos"]
    mujoco.mj_forward(model, target)
    np.testing.assert_array_equal(target.subtree_com[1, :2], report["target_com_xy_m"])
    data = mujoco.MjData(model)
    with np.load(report["source_trajectory"], allow_pickle=False) as saved:
        mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = saved["initial_ctrl"]
        mujoco.mj_forward(model, data)
        for control in saved["ctrl"]:
            data.ctrl[:] = control
            for substep in range(round(.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        state = np.empty(len(report["initial_physics_state"]))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(state, report["initial_physics_state"])
        np.testing.assert_array_equal(data.ctrl, report["initial_ctrl"])
        np.testing.assert_allclose(data.qvel, saved["qvel"][-1], atol=1e-9, rtol=0)
    assert [run["branch"] for run in report["runs"]] == ["hold", "feedback"]
    for run in report["runs"]:
        assert run["prediction_rollouts"] <= 1530 and len(run["plans"]) <= 30
        assert run["prediction_rollouts"] == 51 * len(run["plans"])
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["initial_physics_state"], state)
            np.testing.assert_array_equal(saved["initial_ctrl"], report["initial_ctrl"])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            if run["branch"] == "hold":
                np.testing.assert_array_equal(saved["ctrl"], np.tile(saved["initial_ctrl"], (len(saved["ctrl"]), 1)))
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos, saved["qpos"][0])
            np.testing.assert_array_equal(data.qvel, saved["qvel"][0])
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows = []
            for index, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                row = physical_metrics(model, data)
                row.update(com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - report["target_com_xy_m"])), velocity_max=float(np.max(np.abs(data.qvel))))
                np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(row["com_error_m"], run["trace"][index]["com_error_m"], atol=1e-12, rtol=0)
                peak = max(peak, row["penetration"]["depth_m"])
                rows.append(row)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12, rtol=0)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            safe = bool(rows and all(constraints(row, peak) for row in rows) and slips and np.mean(slips) <= .02
                        and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            reached = bool(safe and len(rows) == 150 and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05 for row in rows[-50:]))
            assert run["physical_safety_passed"] == safe and run["reached"] == reached
            assert run["duration_s"] == len(rows) * .02
    predictor = ComPredictor(model, report["target_com_xy_m"])
    ctrl = np.array(report["initial_ctrl"])
    before = predictor.rollout(state, ctrl, ctrl)
    predictor.rollout(state, ctrl, np.clip(ctrl + .001, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]))
    after = predictor.rollout(state, ctrl, ctrl)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1:] == after[1:]
    print("PASS: source chain, true endpoint continuation, frozen target, paired full state, budgets, controls, native replay, independent safety/reach verdicts and prediction isolation")


if __name__ == "__main__":
    main()
