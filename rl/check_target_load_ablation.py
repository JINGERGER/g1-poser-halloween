import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from directional_feedback import hand_envelope
from load_feedback import constraints
from load_transfer import physical_metrics
from target_load_ablation import digest, terminal_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert [run["branch"] for run in report["runs"]] == ["arbitrary", "tripod_derived"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for key in ("baseline_report", "matrix_report", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    baseline = json.loads(Path(report["baseline_report"]).read_text())
    matrix = json.loads(Path(report["matrix_report"]).read_text())
    measured, target = terminal_target(matrix, report["model"], report["model_sha256"])
    np.testing.assert_array_equal(measured, report["measured_tripod_loads_n"])
    np.testing.assert_array_equal(target, report["runs"][1]["terminal_target_n"])
    for key in ("initial_physics_state", "initial_ctrl"):
        np.testing.assert_array_equal(report[key], baseline[key])
    for key, value in baseline["protocol"].items():
        assert report["protocol"][key] == value
    model = mujoco.MjModel.from_xml_path(report["model"])
    for run in report["runs"]:
        assert run["prediction_rollouts"] <= 1530 and len(run["plans"]) <= 30
        assert all(plan["qp_iterations"] <= 100 for plan in run["plans"])
        assert run["terminal_target_n"][2] == 5
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            for key in ("initial_physics_state", "initial_ctrl"):
                np.testing.assert_array_equal(saved[key], report[key])
            assert len(saved["ctrl"]) == len(run["trace"])
            assert run["duration_s"] == len(saved["ctrl"]) * .02
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0])
            assert np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos, saved["qpos"][0])
            np.testing.assert_array_equal(data.qvel, saved["qvel"][0])
            rows = []
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
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
                rows.append(row)
                peak = max(peak, row["penetration"]["depth_m"])
                np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12, rtol=0)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            physical = bool(rows and slips and all(constraints(row, peak) for row in rows)
                            and np.mean(slips) <= .02 and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            directional = bool(rows and all(hand_envelope(row["loads_n"], run["initial_loads_n"]) for row in rows))
            unloaded = bool(physical and directional and len(rows) == 150 and all(row["loads_n"][2] <= 5 for row in rows[-10:]))
            assert run["physical_safety_passed"] == physical
            assert run["direction_guard_passed"] == directional
            assert run["unload_passed"] == unloaded
            if run["branch"] == "arbitrary":
                old = next(item for item in baseline["runs"] if item["branch"] == "directional")
                with np.load(old["trajectory"], allow_pickle=False) as original:
                    for key in ("qpos", "qvel", "ctrl"):
                        np.testing.assert_allclose(saved[key], original[key], atol=1e-9, rtol=0)
    print("PASS: source chain, target provenance, unchanged protocol, paired state, historical reproduction, budgets, native replay and independent safety/task verdicts")


if __name__ == "__main__":
    main()
