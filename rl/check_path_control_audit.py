import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import physical_metrics
from path_control_audit import FixedPoseFit, acceleration
from servo_force import position_servo_force
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert digest(report["source_report"]) == report["source_report_sha256"]
    assert digest(report["trajectory"]) == report["trajectory_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    source = json.loads(Path(report["source_report"]).read_text())
    baseline = json.loads(Path(source["baseline_report"]).read_text())
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    original = json.loads(Path(geometry["source_report"]).read_text())
    assert digest(source["baseline_report"]) == source["baseline_report_sha256"]
    assert digest(baseline["geometry_report"]) == baseline["geometry_report_sha256"]
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    assert digest(original["model"]) == original["model_sha256"]
    model = mujoco.MjModel.from_xml_path(original["model"])
    path = next(path for path in source["paths"] if path["branch"] == "shift_then_lift" and not path["rotating"])
    bias = np.array(original["initial_ctrl"]) - np.array(geometry["initial"]["qpos"])[7:]
    assert len(report["tracking"]) == 10 and [fit["time_s"] for fit in report["fixed_pose_fits"]] == [0, 3, 5]
    with np.load(report["trajectory"], allow_pickle=False) as saved:
        for row, node in zip(report["tracking"], path["nodes"]):
            index = round(row["time_s"] / .02)
            actual, desired = mujoco.MjData(model), mujoco.MjData(model)
            actual.qpos[:], actual.qvel[:], actual.ctrl[:] = saved["qpos"][index], saved["qvel"][index], saved["ctrl"][index - 1]
            desired.qpos[:], desired.ctrl[:] = node["qpos"], np.array(node["qpos"])[7:] + bias
            mujoco.mj_forward(model, actual)
            mujoco.mj_forward(model, desired)
            error = np.linalg.norm(actual.subtree_com[1, :2] - desired.subtree_com[1, :2])
            np.testing.assert_allclose(error, row["com_xy_error_m"], atol=1e-12)
            np.testing.assert_allclose(actual.qpos[7:] - desired.qpos[7:], row["joint_errors_rad"], atol=1e-12)
            reconstructed = position_servo_force(model, actual.ctrl, actual.qpos[7:], actual.qvel[6:])
            np.testing.assert_allclose(reconstructed, actual.qfrc_actuator[6:], atol=1e-9)
            np.testing.assert_allclose(reconstructed, row["actual_torques_nm"], atol=1e-9)
            np.testing.assert_allclose(physical_metrics(model, actual)["loads_n"], row["actual_metrics"]["loads_n"], atol=1e-7)
    for fit in report["fixed_pose_fits"]:
        assert fit["nfev"] <= 100
        ctrl = np.array(fit["ctrl"])
        assert np.all(ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(ctrl <= model.actuator_ctrlrange[:, 1])
        initial_ctrl = np.array(fit["qpos"])[7:] + bias
        problem = FixedPoseFit(model, fit["qpos"], initial_ctrl)
        before = problem.residual(initial_ctrl)
        after = problem.residual(ctrl)
        assert np.sum(after ** 2) <= np.sum(before ** 2) + 1e-8
        assert not problem.data.qvel.any() and not problem.data.xfrc_applied.any() and not problem.data.qfrc_applied.any()
        for key, value in acceleration(model, problem.data).items():
            np.testing.assert_allclose(value, fit["after"][key], atol=1e-9)
        metrics = physical_metrics(model, problem.data)
        passed = (fit["after"]["base_linear_mps2"] <= .01 and fit["after"]["angular_joint_max_radps2"] <= .1
                  and metrics["penetration"]["depth_m"] <= .005 and metrics["other_total_n"] <= 1 and max(metrics["torque_ratios"]) < .95)
        assert fit["equilibrium_passed"] == passed
        problem.residual(initial_ctrl)
        np.testing.assert_array_equal(problem.residual(ctrl), after)
    print("PASS: source chain, fixed path, native PD forces, COM/joint tracking, bounded fits, scratch reset isolation and equilibrium verdicts")


if __name__ == "__main__":
    main()
