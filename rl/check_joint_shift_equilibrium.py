import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from joint_shift_equilibrium import ShiftEquilibrium
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
    assert digest(report["audit_report"]) == report["audit_report_sha256"]
    audit = json.loads(Path(report["audit_report"]).read_text())
    assert digest(audit["source_report"]) == audit["source_report_sha256"]
    source = json.loads(Path(audit["source_report"]).read_text())
    baseline = json.loads(Path(source["baseline_report"]).read_text())
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    original = json.loads(Path(geometry["source_report"]).read_text())
    for container, key in ((source, "baseline_report"), (baseline, "geometry_report"), (geometry, "source_report"), (original, "model")):
        assert digest(container[key]) == container[key + "_sha256"]
    model = mujoco.MjModel.from_xml_path(original["model"])
    fixed = next(fit for fit in audit["fixed_pose_fits"] if fit["time_s"] == 3)
    problem = ShiftEquilibrium(model, fixed["qpos"], fixed["ctrl"])
    solution = report["solution"]
    assert solution["iterations"] <= 100
    values = np.array(solution["values"])
    assert np.all(np.abs(values[:29]) <= 1 + 1e-9)
    assert np.all(values[29:] >= model.actuator_ctrlrange[:, 0]) and np.all(values[29:] <= model.actuator_ctrlrange[:, 1])
    measured = problem.evaluate(values)
    np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured["ctrl"], solution["ctrl"], atol=1e-12, rtol=0)
    linear, angular = np.linalg.norm(measured["qacc"][:3]), np.max(np.abs(measured["qacc"][3:]))
    np.testing.assert_allclose(linear, solution["base_linear_mps2"], atol=1e-9)
    np.testing.assert_allclose(angular, solution["angular_joint_max_radps2"], atol=1e-9)
    assert solution["passed"] == bool(linear <= .01 and angular <= .1 and np.min(measured["geometry"]) >= -1e-5)
    problem.evaluate(problem.initial)
    np.testing.assert_array_equal(problem.evaluate(values)["qacc"], measured["qacc"])
    if not solution["passed"]:
        assert not report["runs"]
    target = mujoco.MjData(model)
    target.qpos[:] = solution["qpos"]
    mujoco.mj_forward(model, target)
    for run in report["runs"]:
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            if run["branch"] == "real_transfer":
                assert report["runs"][0]["reached"]
                np.testing.assert_array_equal(saved["initial_physics_state"], original["initial_physics_state"])
                np.testing.assert_array_equal(saved["initial_ctrl"], original["initial_ctrl"])
            else:
                np.testing.assert_array_equal(saved["qpos"][0], solution["qpos"])
                assert not saved["qvel"][0].any()
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows = []
            for index, ctrl in enumerate(saved["ctrl"]):
                data.ctrl[:] = ctrl
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                row = physical_metrics(model, data)
                peak = max(peak, row["penetration"]["depth_m"])
                row.update(com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - target.subtree_com[1, :2])), velocity_max=float(np.max(np.abs(data.qvel))))
                np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                rows.append(row)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12, rtol=0)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            safe = bool(peak <= .005 and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                        and slips and np.mean(slips) <= .02 and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            assert run["physical_safety_passed"] == safe
            steps = 150 if run["branch"] == "hypothetical_hold" else 300
            reached = bool(safe and len(rows) == steps and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05
                           and all(load > 5 for load in row["loads_n"]) and all(row["floor_contacts"]) for row in rows[-50:]))
            assert run["reached"] == reached
    print("PASS: frozen inputs, bounded solve, independent equilibrium gates, scratch isolation, conditional execution and native hold/transfer replay")


if __name__ == "__main__":
    main()
