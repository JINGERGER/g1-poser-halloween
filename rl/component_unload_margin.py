import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog

from component_load_capacity import formulate
from load_transfer import physical_metrics
from prepare import ROOT
from servo_force import position_servo_force
from target_load_ablation import digest


def margin_problem(problem):
    count = problem["matrix"].shape[1]
    force_count = 3 * len(problem["points"])
    weight = problem["mass_kg"] * np.linalg.norm(problem["gravity_mps2"])
    half_width = (problem["torque_upper"] - problem["torque_lower"]) / 2
    assert np.all(half_width > 0)
    equalities = np.column_stack((problem["matrix"], np.zeros(problem["matrix"].shape[0])))
    equalities = np.vstack((equalities, np.r_[problem["loads"][0], 0]))
    target = np.r_[problem["target"], 0]
    rows, rhs, labels = [], [], []
    for index, row in enumerate(problem["friction"]):
        retained = problem["points"][index // 4]["support"] != "left_hand_collision"
        rows.append(np.r_[row, weight / 3 if retained else 0])
        rhs.append(0)
        labels.append(f"friction_point_{index // 4}_face_{index % 4}")
    for index in range(3):
        rows.append(np.r_[-problem["loads"][index + 1], 0])
        rhs.append(-5)
        labels.append(f"retained_support_{index + 1}")
    for index in range(len(half_width)):
        for sign, bound in ((1, problem["torque_upper"][index]), (-1, -problem["torque_lower"][index])):
            row = np.zeros(count + 1)
            row[force_count + index], row[-1] = sign, half_width[index]
            rows.append(row)
            rhs.append(bound)
            labels.append(f"torque_{index}_{'upper' if sign == 1 else 'lower'}")
    objective = np.zeros(count + 1)
    objective[-1] = -1
    return {"objective": objective, "equalities": equalities, "target": target,
            "inequalities": np.array(rows), "rhs": np.array(rhs), "bounds": problem["bounds"] + [(0, 1)],
            "labels": labels, "force_scale_n": float(weight / 3), "half_width": half_width}


def optimize(problem):
    setup = margin_problem(problem)
    result = linprog(setup["objective"], A_ub=setup["inequalities"], b_ub=setup["rhs"],
                     A_eq=setup["equalities"], b_eq=setup["target"], bounds=setup["bounds"], method="highs",
                     options={"time_limit": 20, "maxiter": 10000})
    output = {"success": bool(result.success), "status": int(result.status), "message": result.message, "solution": None}
    if result.success:
        values, margin = result.x[:-1], float(result.x[-1])
        force_count = 3 * len(problem["points"])
        torque = values[force_count:]
        residual = float(np.max(np.abs(setup["equalities"] @ result.x - setup["target"])))
        slack = setup["rhs"] - setup["inequalities"] @ result.x
        assert residual < 1e-6 and min(slack) > -1e-6
        output["solution"] = {"values": values.tolist(), "normalized_margin": margin,
                              "forces_world_n": values[:force_count].reshape(-1, 3).tolist(),
                              "loads_n": (problem["loads"] @ values).tolist(), "joint_torques_nm": torque.tolist(),
                              "torque_min_margin_nm": float(np.min(np.minimum(torque - problem["torque_lower"], problem["torque_upper"] - torque))),
                              "retained_friction_face_slack_floor_n": margin * setup["force_scale_n"],
                              "equilibrium_residual_max": residual,
                              "active_constraints": [label for label, value in zip(setup["labels"], slack) if value < 1e-6]}
    return output


def native_probe(model, qpos, ctrl):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    row = physical_metrics(model, data)
    linear, angular = float(np.linalg.norm(data.qacc[:3])), float(np.max(np.abs(data.qacc[3:])))
    passed = bool(row["loads_n"][0] <= 1 and all(load > 5 for load in row["loads_n"][1:])
                  and linear <= .01 and angular <= .1 and row["penetration"]["depth_m"] <= .005 and row["other_total_n"] <= 1)
    return {"metrics": row, "qacc": data.qacc.tolist(), "base_linear_mps2": linear,
            "angular_joint_max_radps2": angular, "native_static_unload_passed": passed,
            "native_actuator_torque_nm": data.qfrc_actuator[6:].tolist(), "time_s": float(data.time)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capacity_report", type=Path)
    args = parser.parse_args()
    capacity = json.loads(args.capacity_report.read_text())
    assert capacity["status"] == "completed"
    assert digest(capacity["model"]) == capacity["model_sha256"]
    chain = capacity
    for unused in range(2):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    entry = next(entry for entry in chain["seeds"] if entry["seed"] == "old_equilibrium")
    assert entry["hold"]["reached"]
    np.testing.assert_array_equal(capacity["qpos"], entry["equilibrium"]["qpos"])
    model_path = Path(capacity["model"])
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(model_path))
    problem = formulate(model, capacity["qpos"])
    destination = ROOT / "output/rl/component_unload_margin" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.capacity_report.resolve()), "source_report_sha256": digest(args.capacity_report),
              "model": str(model_path), "model_sha256": digest(model_path), "manifest_sha256": digest(model_path.with_name("manifest.json")),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "component_load_capacity.py", "servo_force.py", "load_transfer.py", "diagnose_short_crawl.py")},
              "qpos": capacity["qpos"], "baseline_ctrl": entry["equilibrium"]["ctrl"], "dynamic_rollouts": 0,
              "protocol": {"lp_count": 1, "time_limit_s": 20, "maxiter": 10000, "left_hand_load_n": 0,
                           "margin": "t in [0,1]; torque inward t*half attainable interval; retained friction faces inward t*weight/3",
                           "native_unload_max_n": 1, "native_linear_acceleration_max_mps2": .01, "native_other_acceleration_max_radps2": .1},
              "limitations": "Fixed-pose static LP and zero-velocity native forward probes only; no motor trajectory or lift. Margin is chosen normalized slack, not a hardware safety factor. Unloaded friction faces excluded from margin. No reoptimization of pose or contact parameters; previous contact-sensitivity failure retained."}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["optimization"] = optimize(problem)
        report["baseline_probe"] = native_probe(model, report["qpos"], report["baseline_ctrl"])
        if report["optimization"]["success"]:
            solution = report["optimization"]["solution"]
            torque = np.array(solution["joint_torques_nm"])
            positions = np.array(report["qpos"])[7:]
            ctrl = (torque - model.actuator_biasprm[:, 0] - model.actuator_biasprm[:, 1] * positions) / model.actuator_gainprm[:, 0]
            assert np.all(ctrl >= model.actuator_ctrlrange[:, 0] - 1e-9) and np.all(ctrl <= model.actuator_ctrlrange[:, 1] + 1e-9)
            np.testing.assert_allclose(position_servo_force(model, ctrl, positions, np.zeros(model.nu)), torque, atol=1e-6, rtol=0)
            report["candidate_ctrl"] = ctrl.tolist()
            report["candidate_probe"] = native_probe(model, report["qpos"], ctrl)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)
        for key in ("optimization", "baseline_probe", "candidate_probe"):
            if key in report:
                print(key, json.dumps(report[key]))


if __name__ == "__main__":
    main()
