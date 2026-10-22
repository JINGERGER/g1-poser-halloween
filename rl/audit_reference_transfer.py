import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from reference_policy_baseline import materialize
from artifact_utils import digest


def audit(trial, folder):
    path = Path(trial["model_path"])
    assert digest(path) == trial["model_sha256"]
    assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = trial.get("physics_timestep", .002)
    if trial.get("contact_time_constant") is not None:
        model.geom_solref[:, 0] = trial["contact_time_constant"]
    if trial["gain_profile"] == "reference":
        reference = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        mapping = {reference.joint(int(joint)).name: index for index, joint in enumerate(reference.actuator_trnid[:, 0])}
        for index, joint in enumerate(model.actuator_trnid[:, 0]):
            name = model.joint(int(joint)).name
            if name in mapping:
                source = mapping[name]
                model.actuator_gainprm[index, 0] = reference.actuator_gainprm[source, 0] * 1.4
                model.actuator_biasprm[index, 1:3] = reference.actuator_biasprm[source, 1:3] * 1.4
    saved = np.load(trial["trajectory"], allow_pickle=False)
    data = mujoco.MjData(model)
    assert model.na == 0
    data.qpos[:] = saved["qpos"][0]
    data.qvel[:] = saved["qvel"][0]
    mujoco.mj_forward(model, data)
    floor = next(index for index in range(model.ngeom) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE)
    names = ["left_hand_collision", "right_hand_collision", "left_foot", "right_foot"]
    supports = [model.geom(name).id for name in names]
    joint_ids = model.actuator_trnid[:, 0]
    limits = np.max(np.abs(model.jnt_actfrcrange[joint_ids]), axis=1)
    assert np.all(model.jnt_actfrclimited[joint_ids]) and np.all(limits > 0)
    contact_maxima = {}
    events = [[] for unused in supports]
    airborne_start = [None] * 4
    lift_positions = [None] * 4
    clearance_time = np.zeros(4)
    maximum_clearance_time = np.zeros(4)
    maximum_gap = np.zeros(4)
    samples = []
    support_positions, penetrations, self_penetrations = [], [], []
    initial_penetration = max((-contact.dist for contact in data.contact), default=0.)
    initial_self_penetration = max((-contact.dist for contact in data.contact
                                    if floor not in (contact.geom1, contact.geom2)), default=0.)
    torque_ratio, limit_violation = 0., 0.
    for control_index, control in enumerate(saved["ctrl"]):
        data.ctrl[:] = control
        steps = round((trial["trace"][control_index]["time_s"] - data.time) / model.opt.timestep)
        assert 1 <= steps <= trial["control_substeps"]
        for unused in range(steps):
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            loads = np.zeros(4)
            other = 0.
            for contact_index, contact in enumerate(data.contact):
                pair = "|".join(sorted(model.geom(int(geom)).name or f"geom_{geom}" for geom in (contact.geom1, contact.geom2)))
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, wrench)
                force = max(0., float(wrench[0]))
                record = contact_maxima.setdefault(pair, {"depth_m": 0., "force_n": 0., "max_depth_time_s": 0., "over_5mm_steps": 0})
                if -contact.dist > record["depth_m"]:
                    record.update(depth_m=float(-contact.dist), max_depth_time_s=float(data.time))
                record["force_n"] = max(record["force_n"], force)
                record["over_5mm_steps"] += int(contact.dist < -.005)
                if floor in (contact.geom1, contact.geom2):
                    geom = contact.geom2 if contact.geom1 == floor else contact.geom1
                    if geom in supports:
                        loads[supports.index(geom)] += force
                    else:
                        other += force
            gaps = [float(mujoco.mj_geomDistance(model, data, geom, floor, 2., None)) for geom in supports]
            current_ratio = float(np.max(np.abs(data.qfrc_actuator[model.jnt_dofadr[joint_ids]]) / limits))
            torque_ratio = max(torque_ratio, current_ratio)
            angles = data.qpos[model.jnt_qposadr[joint_ids]]
            excess = float(max(0., np.max(model.jnt_range[joint_ids, 0] - angles), np.max(angles - model.jnt_range[joint_ids, 1])))
            limit_violation = max(limit_violation, excess)
            for index, geom in enumerate(supports):
                if gaps[index] > .001 and loads[index] <= 1:
                    if airborne_start[index] is None:
                        airborne_start[index] = float(data.time)
                        lift_positions[index] = data.geom_xpos[geom].copy()
                        maximum_clearance_time[index] = 0.
                        maximum_gap[index] = 0.
                    maximum_gap[index] = max(maximum_gap[index], gaps[index])
                    clearance_time[index] = clearance_time[index] + model.opt.timestep if gaps[index] >= .01 else 0.
                    maximum_clearance_time[index] = max(maximum_clearance_time[index], clearance_time[index])
                elif loads[index] > 5:
                    if airborne_start[index] is not None:
                        duration = float(data.time - airborne_start[index])
                        events[index].append({"lift_time_s": airborne_start[index], "land_time_s": float(data.time),
                                              "duration_s": duration,
                                              "max_gap_m": float(maximum_gap[index]),
                                              "continuous_10mm_clearance_s": float(maximum_clearance_time[index]),
                                              "meets_10mm_80ms": bool(maximum_clearance_time[index] >= .08 - 1e-10),
                                              "world_displacement_m": (data.geom_xpos[geom] - lift_positions[index]).tolist()})
                    airborne_start[index] = None
                    clearance_time[index] = 0.
                else:
                    clearance_time[index] = 0.
            samples.append([float(data.time), *data.qpos[:3], *loads, *gaps, other, current_ratio, excess])
            support_positions.append(data.geom_xpos[supports].copy())
            penetrations.append(max((-contact.dist for contact in data.contact), default=0.))
            self_penetrations.append(max((-contact.dist for contact in data.contact
                                          if floor not in (contact.geom1, contact.geom2)), default=0.))
        np.testing.assert_allclose(data.qpos, saved["qpos"][control_index + 1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(data.qvel, saved["qvel"][control_index + 1], atol=1e-10, rtol=0)
    sample_path = Path(trial["trajectory"]).with_suffix(".audit_v3.npz")
    np.savez_compressed(sample_path, samples=np.array(samples), support_positions=np.array(support_positions),
                        penetrations=np.array(penetrations), self_penetrations=np.array(self_penetrations))
    return {"trajectory": trial["trajectory"], "replay_passed": True,
            "physics_steps": len(samples), "max_torque_limit_ratio": torque_ratio,
            "max_joint_limit_excess_rad": limit_violation, "contact_maxima": contact_maxima,
            "initial_penetration_m": float(initial_penetration),
            "initial_self_penetration_m": float(initial_self_penetration),
            "events_by_support": dict(zip(names, events)), "samples": str(sample_path),
            "samples_sha256": digest(sample_path),
            "sample_columns": ["time", "base_x", "base_y", "base_z", "left_hand_load", "right_hand_load", "left_foot_load", "right_foot_load",
                               "left_hand_gap", "right_hand_gap", "left_foot_gap", "right_foot_gap", "other_load", "torque_ratio", "joint_limit_excess"],
            "limitations": "Event candidates start with gap>1mm and load<=1N and end with load>5N. Continuous 10mm/80ms qualification is separately recorded, not assumed. No certified alternating crawl or relaxed success gate."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    folder, unused = materialize()
    assert digest(folder / "policy_crawl.pt") == report["policy_sha256"]
    model_folder = Path(report["runs"][0]["model_path"]).parent
    for name, expected in report["target_assets"].items():
        assert digest(model_folder / name) == expected
    results = []
    for trial in report["runs"]:
        result = audit(trial, folder)
        results.append(result)
        worst = max(result["contact_maxima"].items(), key=lambda pair: pair[1]["depth_m"])
        print(json.dumps({"trajectory": trial["trajectory"], "worst_contact": worst,
                          "torque_ratio": result["max_torque_limit_ratio"],
                          "joint_excess": result["max_joint_limit_excess_rad"],
                          "event_counts": [len(value) for value in result["events_by_support"].values()]}), flush=True)
    path = args.report.with_name("physics_audit_v3.json")
    path.write_text(json.dumps({"source_sha256": digest(__file__), "report_sha256": digest(args.report), "runs": results}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
