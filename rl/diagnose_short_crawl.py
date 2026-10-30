import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare import SUPPORTS


def observe(model, data, target):
    floor = model.geom("floor").id
    supports = {model.geom(name).id: index for index, name in enumerate(SUPPORTS)}
    loads = np.zeros(4)
    other = {}
    deepest = {"depth_m": 0.0, "pair": None}
    for index, contact in enumerate(data.contact):
        if -contact.dist > deepest["depth_m"]:
            deepest = {"depth_m": float(-contact.dist),
                       "pair": [model.geom(int(geom)).name for geom in (contact.geom1, contact.geom2)]}
        if floor not in (contact.geom1, contact.geom2):
            continue
        geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = max(0.0, float(wrench[0]))
        if geom in supports:
            loads[supports[geom]] += normal
        else:
            name = model.geom(geom).name
            other[name] = other.get(name, 0.0) + normal
    joints = model.actuator_trnid[:, 0]
    force = data.qfrc_actuator[model.jnt_dofadr[joints]]
    limits = model.jnt_actfrcrange[joints]
    ratios = np.abs(force) / np.where(force >= 0, limits[:, 1], -limits[:, 0])
    errors = target - data.qpos[model.jnt_qposadr[joints]]
    worst = int(np.argmax(np.abs(errors)))
    return {"time_s": float(data.time), "base_height_m": float(data.qpos[2]),
            "gaps_m": [float(mujoco.mj_geomDistance(model, data, model.geom(name).id, floor, 2, None)) for name in SUPPORTS],
            "base_acceleration": data.qacc[:6].tolist(),
            "loads_n": loads.tolist(), "other_loads_n": other, "other_total_n": sum(other.values()),
            "penetration": deepest, "tracking_error_max_rad": float(abs(errors[worst])),
            "worst_joint": model.joint(int(joints[worst])).name,
            "tracking_error_rad": errors.tolist(), "torque_ratios": ratios.tolist(),
            "near_limit_joints": [model.joint(int(joints[index])).name for index in np.flatnonzero(ratios >= 0.95)]}


def summarize(rows):
    predicates = {"penetration_over_5mm": lambda row: row["penetration"]["depth_m"] > 0.005,
                  "non_support_over_1N": lambda row: row["other_total_n"] > 1,
                  "joint_error_over_0_1rad": lambda row: row["tracking_error_max_rad"] > 0.1,
                  "near_torque_limit": lambda row: bool(row["near_limit_joints"])}
    return {name: next((row for row in rows if predicate(row)), None) for name, predicate in predicates.items()}


def replay(entry, hold=False):
    path = Path(entry["model"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["model_sha256"]:
        raise ValueError("Model snapshot changed")
    model = mujoco.MjModel.from_xml_path(str(path))
    saved = np.load(entry["trajectory"], allow_pickle=False)
    data = mujoco.MjData(model)
    data.qpos[:] = saved["qpos"][0]
    data.qvel[:] = saved["qvel"][0]
    data.ctrl[:] = saved["ctrl"][0]
    mujoco.mj_forward(model, data)
    rows = [observe(model, data, data.ctrl)]
    maximum_error = 0.0
    joints = model.actuator_trnid[:, 0]
    limits = model.jnt_actfrcrange[joints]
    startup_peak_ratio = 0.0
    controls = np.repeat(saved["ctrl"][:1], 100, axis=0) if hold else saved["ctrl"]
    for step, control in enumerate(controls):
        data.ctrl[:] = control
        for substep in range(round(0.02 / model.opt.timestep)):
            if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise AssertionError("Unexpected external assistance")
            mujoco.mj_step(model, data)
            if data.time <= 0.5000001:
                forces = data.qfrc_actuator[model.jnt_dofadr[joints]]
                startup_peak_ratio = max(startup_peak_ratio, float(np.max(np.abs(forces) / np.where(forces >= 0, limits[:, 1], -limits[:, 0]))))
        mujoco.mj_forward(model, data)
        if not hold or step < 25:
            error = max(float(np.max(np.abs(data.qpos - saved["qpos"][step + 1]))),
                        float(np.max(np.abs(data.qvel - saved["qvel"][step + 1]))))
            maximum_error = max(maximum_error, error)
            if error > 1e-8:
                raise AssertionError("Replay or paired initial hold does not match saved experiment")
        rows.append(observe(model, data, control))
    return {"pd": entry["pd"], "period_s": entry["period_s"], "feedforward": entry["velocity_feedforward"],
            "intervention": "hold_initial_ctrl_2s" if hold else "original_controls",
            "verified_state_error": maximum_error, "first_events": summarize(rows),
            "trajectory_sha256": hashlib.sha256(Path(entry["trajectory"]).read_bytes()).hexdigest(),
            "startup_substep_peak_torque_ratio": startup_peak_ratio,
            "first_hold_events": summarize([row for row in rows if row["time_s"] <= 0.5000001]),
            "snapshots": [row for row in rows if any(abs(row["time_s"] - stamp) < 1e-7 for stamp in (0, 0.1, 0.2, 0.5, 1, 2))],
            "trace": rows}


def main():
    parser = argparse.ArgumentParser(description="Localize observed failures; paired fixed-control intervention, no search or learning")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    original = json.loads(args.report.read_text())
    if original["status"] != "completed":
        raise ValueError("Expected a completed short-motion experiment")
    result = {"source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_updates": 0, "runs": [], "hold_controls": [],
              "limitations": "20ms endpoint localization, not exact continuous event times. 0.1rad tracking error is a diagnostic marker, not a hardware limit. Fixed-control intervention tests whether motion is necessary for startup failures; it does not prove why equilibrium is lost or whether another controller is feasible."}
    for entry in original["runs"]:
        result["runs"].append(replay(entry))
    for variant in ("current", "reference"):
        entry = next(row for row in original["runs"] if row["pd"] == variant and not row["velocity_feedforward"])
        result["hold_controls"].append(replay(entry, hold=True))
    output = args.report.with_name("failure_diagnosis.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["hold_controls"]:
        print(json.dumps({"pd": row["pd"], "startup_peak_torque_ratio": row["startup_substep_peak_torque_ratio"],
                          "first_event_s": {name: event["time_s"] if event else None for name, event in row["first_events"].items()}}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
