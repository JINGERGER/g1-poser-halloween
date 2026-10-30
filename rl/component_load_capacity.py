import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog

from prepare import ROOT, SUPPORTS
from servo_force import position_servo_force, validate_position_servos
from target_load_ablation import digest


def formulate(model, qpos):
    validate_position_servos(model)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    support_ids = [model.geom(name).id for name in SUPPORTS]
    points, jacobians = [], []
    for contact in data.contact:
        if floor not in (contact.geom1, contact.geom2):
            continue
        geom = contact.geom2 if contact.geom1 == floor else contact.geom1
        if geom not in support_ids:
            raise ValueError("Unexpected floor contact at fixed pose")
        assert contact.dim == 3
        np.testing.assert_allclose(np.abs(contact.frame[:3]), [0, 0, 1], atol=1e-10)
        assert abs(contact.friction[0] - contact.friction[1]) < 1e-12
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact.pos, int(model.geom_bodyid[geom]))
        jacobians.append(jacobian.T)
        points.append({"support": SUPPORTS[support_ids.index(geom)], "position_m": contact.pos.tolist(),
                       "mu": float(contact.friction[0]), "gap_m": float(contact.dist)})
    assert set(point["support"] for point in points) == set(SUPPORTS)
    force_columns = 3 * len(points)
    matrix = np.zeros((model.nv, force_columns + model.nu))
    matrix[:, :force_columns] = np.hstack(jacobians)
    joint_ids = model.actuator_trnid[:, 0]
    matrix[model.jnt_dofadr[joint_ids], force_columns + np.arange(model.nu)] = 1
    target = data.qfrc_bias - data.qfrc_passive
    lower = position_servo_force(model, model.actuator_ctrlrange[:, 0], data.qpos[7:], np.zeros(model.nu))
    upper = position_servo_force(model, model.actuator_ctrlrange[:, 1], data.qpos[7:], np.zeros(model.nu))
    assert np.all(lower <= upper)
    inequalities = []
    for index, point in enumerate(points):
        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            row = np.zeros(matrix.shape[1])
            row[3 * index:3 * index + 3] = [first, second, -point["mu"]]
            inequalities.append(row)
    loads = np.zeros((4, matrix.shape[1]))
    for index, point in enumerate(points):
        loads[SUPPORTS.index(point["support"]), 3 * index + 2] = 1
    bounds = [(None, None), (None, None), (0, None)] * len(points) + list(zip(lower, upper))
    return {"matrix": matrix, "target": target, "friction": np.array(inequalities), "loads": loads,
            "bounds": bounds, "points": points, "torque_lower": lower, "torque_upper": upper,
            "com_m": data.subtree_com[1].tolist(), "mass_kg": float(sum(model.body_mass)),
            "gravity_mps2": model.opt.gravity.tolist()}


def solve(problem, target_support):
    objective = np.zeros(problem["matrix"].shape[1])
    retained = list(range(4))
    if target_support is not None:
        retained.remove(target_support)
        objective = problem["loads"][target_support].copy()
    inequalities = np.vstack((problem["friction"], -problem["loads"][retained]))
    right = np.r_[np.zeros(len(problem["friction"])), [-5.] * len(retained)]
    result = linprog(objective, A_ub=inequalities, b_ub=right, A_eq=problem["matrix"], b_eq=problem["target"],
                     bounds=problem["bounds"], method="highs", options={"time_limit": 20, "maxiter": 10000})
    output = {"target_support": None if target_support is None else SUPPORTS[target_support],
              "solver_success": bool(result.success), "status": int(result.status), "message": result.message,
              "iterations": int(result.nit), "solution": None}
    if result.success:
        values = result.x
        residual = float(np.max(np.abs(problem["matrix"] @ values - problem["target"])))
        violation = float(max(0, np.max(inequalities @ values - right)))
        assert residual < 1e-6 and violation < 1e-6
        loads = problem["loads"] @ values
        forces = values[:3 * len(problem["points"])].reshape(-1, 3)
        torque = values[3 * len(problem["points"]):]
        output["solution"] = {"values": values.tolist(), "loads_n": loads.tolist(), "forces_world_n": forces.tolist(),
                              "joint_torques_nm": torque.tolist(), "equilibrium_residual_max": residual,
                              "inequality_violation_max": violation,
                              "minimum_target_load_n": None if target_support is None else float(loads[target_support]),
                              "zero_load_feasible": None if target_support is None else bool(loads[target_support] < 1e-6),
                              "torque_margin_min_nm": float(np.min(np.minimum(torque - problem["torque_lower"], problem["torque_upper"] - torque)))}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sensitivity_report", type=Path)
    args = parser.parse_args()
    sensitivity = json.loads(args.sensitivity_report.read_text())
    assert sensitivity["status"] == "completed"
    assert digest(sensitivity["source_report"]) == sensitivity["source_report_sha256"]
    source = json.loads(Path(sensitivity["source_report"]).read_text())
    path = Path(source["model"])
    assert digest(path) == source["model_sha256"] == sensitivity["model_sha256"]
    for mesh, expected in sensitivity["mesh_sha256"].items():
        assert digest(mesh) == expected
    entry = next(entry for entry in source["seeds"] if entry["seed"] == "old_equilibrium")
    assert entry["hold"]["reached"]
    model = mujoco.MjModel.from_xml_path(str(path))
    problem = formulate(model, entry["equilibrium"]["qpos"])
    destination = ROOT / "output/rl/component_load_capacity" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.sensitivity_report.resolve()), "source_report_sha256": digest(args.sensitivity_report),
              "model": str(path), "model_sha256": digest(path), "qpos": entry["equilibrium"]["qpos"], "dynamic_rollouts": 0,
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (Path(__file__).name, "servo_force.py")},
              "protocol": {"pose_count": 1, "baseline_count": 1, "unload_count": 4, "time_limit_s_per_lp": 20,
                           "maxiter": 10000, "retained_min_load_n": 5, "reoptimization_of_pose": False},
              "contact_points": problem["points"], "torque_lower_nm": problem["torque_lower"].tolist(),
              "torque_upper_nm": problem["torque_upper"].tolist(), "com_m": problem["com_m"], "mass_kg": problem["mass_kg"],
              "limitations": "Ideal static screening at the initialized equilibrium pose, not dynamically reached state or lift trajectory. Existing floor contact manifold points frozen; no contact moments. Inner diamond friction cone |fx|+|fy|<=mu*fz. No dry-friction assistance. Soft contact does not necessarily realize the optimized forces. Zero force is not foot clearance. Contact sensitivity failure remains in force.", "runs": []}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for target in (None, 0, 1, 2, 3):
            result = solve(problem, target)
            report["runs"].append(result)
            print(json.dumps({"target": result["target_support"], "success": result["solver_success"],
                              "loads_n": result["solution"] and result["solution"]["loads_n"]}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
