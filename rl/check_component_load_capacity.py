import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from component_load_capacity import formulate, solve
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
    sensitivity = json.loads(Path(report["source_report"]).read_text())
    for path, expected in sensitivity["mesh_sha256"].items():
        assert digest(path) == expected
    assert digest(sensitivity["source_report"]) == sensitivity["source_report_sha256"]
    initial = json.loads(Path(sensitivity["source_report"]).read_text())
    entry = next(entry for entry in initial["seeds"] if entry["seed"] == "old_equilibrium")
    np.testing.assert_array_equal(report["qpos"], entry["equilibrium"]["qpos"])
    model = mujoco.MjModel.from_xml_path(report["model"])
    problem = formulate(model, report["qpos"])
    assert problem["points"] == report["contact_points"]
    assert len(report["runs"]) == 5
    for target, run in zip((None, 0, 1, 2, 3), report["runs"]):
        actual = solve(problem, target)
        assert actual["target_support"] == run["target_support"]
        assert actual["solver_success"] == run["solver_success"]
        if not run["solver_success"]:
            continue
        solution = run["solution"]
        values = np.array(solution["values"])
        forces = np.array(solution["forces_world_n"])
        torque = np.array(solution["joint_torques_nm"])
        np.testing.assert_allclose(problem["matrix"] @ values, problem["target"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(values[:forces.size], forces.ravel(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(values[forces.size:], torque, atol=1e-12, rtol=0)
        assert min(forces[:, 2]) >= -1e-8
        for force, point in zip(forces, problem["points"]):
            assert abs(force[0]) + abs(force[1]) <= point["mu"] * force[2] + 1e-7
        loads = problem["loads"] @ values
        np.testing.assert_allclose(loads, solution["loads_n"], atol=1e-8, rtol=0)
        for index in range(4):
            if index != target:
                assert loads[index] >= 5 - 1e-7
        if target is not None:
            np.testing.assert_allclose(loads[target], actual["solution"]["minimum_target_load_n"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(forces.sum(axis=0), -problem["mass_kg"] * np.array(problem["gravity_mps2"]), atol=1e-6, rtol=0)
        arms = np.array([point["position_m"] for point in problem["points"]]) - problem["com_m"]
        np.testing.assert_allclose(np.cross(arms, forces).sum(axis=0), np.zeros(3), atol=1e-6, rtol=0)
        assert np.all(torque >= problem["torque_lower"] - 1e-7) and np.all(torque <= problem["torque_upper"] + 1e-7)
        positions = np.array(report["qpos"])[7:]
        ctrl = (torque - model.actuator_biasprm[:, 0] - model.actuator_biasprm[:, 1] * positions) / model.actuator_gainprm[:, 0]
        ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        np.testing.assert_allclose(position_servo_force(model, ctrl, positions, np.zeros(model.nu)), torque, atol=1e-6, rtol=0)
    print("PASS: five LPs, fixed pose, minimum-load objectives, world force/moment balance, friction and attainable servo torques; no dynamic lift claimed")


if __name__ == "__main__":
    main()
