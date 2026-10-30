import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from component_load_capacity import formulate
from component_unload_margin import margin_problem, native_probe, optimize
from servo_force import position_servo_force
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["dynamic_rollouts"] == 0
    for key in ("source_report", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    model_path = Path(report["model"])
    assert digest(model_path.with_name("manifest.json")) == report["manifest_sha256"]
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    capacity = json.loads(Path(report["source_report"]).read_text())
    np.testing.assert_array_equal(report["qpos"], capacity["qpos"])
    chain = capacity
    for unused in range(2):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    original = next(entry for entry in chain["seeds"] if entry["seed"] == "old_equilibrium")
    np.testing.assert_array_equal(report["baseline_ctrl"], original["equilibrium"]["ctrl"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    problem = formulate(model, report["qpos"])
    actual = optimize(problem)
    assert actual["success"] == report["optimization"]["success"]
    assert native_probe(model, report["qpos"], report["baseline_ctrl"]) == report["baseline_probe"]
    if actual["success"]:
        solution = report["optimization"]["solution"]
        setup = margin_problem(problem)
        values = np.r_[solution["values"], solution["normalized_margin"]]
        np.testing.assert_allclose(setup["equalities"] @ values, setup["target"], atol=1e-6, rtol=0)
        assert max(setup["inequalities"] @ values - setup["rhs"]) < 1e-6
        assert 0 <= values[-1] <= 1
        np.testing.assert_allclose(values[-1], actual["solution"]["normalized_margin"], atol=1e-8, rtol=0)
        forces = np.array(solution["forces_world_n"])
        torque = np.array(solution["joint_torques_nm"])
        np.testing.assert_allclose(values[:forces.size], forces.ravel(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(values[forces.size:-1], torque, atol=1e-12, rtol=0)
        for index, point in enumerate(problem["points"]):
            if point["support"] == "left_hand_collision":
                np.testing.assert_allclose(forces[index], 0, atol=1e-8, rtol=0)
            else:
                slack = point["mu"] * forces[index, 2] - abs(forces[index, 0]) - abs(forces[index, 1])
                assert slack >= values[-1] * setup["force_scale_n"] - 1e-7
        margin = np.minimum(torque - problem["torque_lower"], problem["torque_upper"] - torque)
        assert np.all(margin >= values[-1] * setup["half_width"] - 1e-7)
        np.testing.assert_allclose(forces.sum(axis=0), -problem["mass_kg"] * np.array(problem["gravity_mps2"]), atol=1e-6, rtol=0)
        arms = np.array([point["position_m"] for point in problem["points"]]) - problem["com_m"]
        np.testing.assert_allclose(np.cross(arms, forces).sum(axis=0), 0, atol=1e-6, rtol=0)
        ctrl = np.array(report["candidate_ctrl"])
        assert np.all(ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(ctrl <= model.actuator_ctrlrange[:, 1])
        np.testing.assert_allclose(position_servo_force(model, ctrl, np.array(report["qpos"])[7:], np.zeros(model.nu)), torque, atol=1e-6, rtol=0)
        candidate = native_probe(model, report["qpos"], ctrl)
        assert candidate == report["candidate_probe"] and candidate["time_s"] == 0
        np.testing.assert_allclose(candidate["native_actuator_torque_nm"], torque, atol=1e-6, rtol=0)
        row = candidate["metrics"]
        passed = bool(row["loads_n"][0] <= 1 and min(row["loads_n"][1:]) > 5
                      and np.linalg.norm(candidate["qacc"][:3]) <= .01 and max(abs(np.array(candidate["qacc"])[3:])) <= .1
                      and row["penetration"]["depth_m"] <= .005 and row["other_total_n"] <= 1)
        assert candidate["native_static_unload_passed"] == passed
    print("PASS: normalized-margin LP, independent force/moment checks, attainable torques, paired native probes and failed-unload gate; no simulation steps")


if __name__ == "__main__":
    main()
