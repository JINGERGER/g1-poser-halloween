import hashlib
import json

import mujoco
import numpy as np

from audit_geometry import gaps_and_points, unwanted_contacts
from prepare import ASSETS, ROOT, SUPPORTS
from servo_force import position_servo_force, validate_position_servos


def trial(model, seed):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    targets = data.ctrl.copy()
    joint_ids = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    if seed is not None:
        data.qpos[addresses] += np.random.default_rng(seed).uniform(-0.01, 0.01, model.nu)
        data.qpos[addresses] = np.clip(data.qpos[addresses], model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    initial_hash = hashlib.sha256(data.qpos.tobytes() + data.qvel.tobytes() + data.ctrl.tobytes()).hexdigest()
    origin = data.qpos[:2].copy()
    floor = model.geom("floor").id
    support_ids = [model.geom(name).id for name in SUPPORTS]
    limits = model.jnt_actfrcrange[joint_ids]
    traces = []
    force_error = 0.0
    for step in range(round(10 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise AssertionError("Nonfinite native state")
        if step % 10 != 9:
            continue
        mujoco.mj_forward(model, data)
        force = position_servo_force(model, targets, data.qpos[addresses], data.qvel[dofs])
        force_error = max(force_error, float(np.max(np.abs(force - data.qfrc_actuator[dofs]))))
        loads = np.zeros(4)
        weighted_slip = np.zeros(4)
        non_support_load = 0.0
        for index, contact in enumerate(data.contact):
            if floor not in (contact.geom1, contact.geom2):
                continue
            geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = max(0.0, float(wrench[0]))
            if geom not in support_ids:
                non_support_load += normal
                continue
            support = support_ids.index(geom)
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos, int(model.geom_bodyid[geom]))
            loads[support] += normal
            weighted_slip[support] += normal * np.linalg.norm((jacobian @ data.qvel)[:2])
        gaps, _ = gaps_and_points(model, data)
        bad = unwanted_contacts(model, data)
        ratio = np.abs(force) / np.where(force >= 0, limits[:, 1], -limits[:, 0])
        traces.append({"time_s": float(data.time), "base_height_m": float(data.qpos[2]),
                       "base_drift_m": float(np.linalg.norm(data.qpos[:2] - origin)),
                       "base_speed_mps": float(np.linalg.norm(data.qvel[:3])),
                       "max_joint_speed_radps": float(np.max(np.abs(data.qvel[dofs]))),
                       "normal_loads_n": loads.tolist(), "gaps_m": gaps.tolist(),
                       "non_support_normal_load_n": non_support_load,
                       "loaded_contact_slip_mps": (weighted_slip / np.maximum(loads, 1e-12)).tolist(),
                       "foot_normal_z": [float(data.geom(name).xmat.reshape(3, 3)[2, 2]) for name in SUPPORTS[2:]],
                       "max_torque_ratio": float(ratio.max()),
                       "unwanted_contacts": [[model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name, float(contact.dist)] for contact in bad]})
        if data.qpos[2] < 0.10:
            break
    tail = [row for row in traces if row["time_s"] >= 8.0]
    def all_loaded(row):
        return all(load > 1.0 for load in row["normal_loads_n"])

    passed = bool(data.time >= 9.99 and tail and all(all_loaded(row) and not row["unwanted_contacts"]
                  and row["base_speed_mps"] < 0.01 and row["max_joint_speed_radps"] < 0.1
                  and max(row["loaded_contact_slip_mps"]) < 0.01 and row["max_torque_ratio"] < 0.95 for row in tail))
    assert force_error < 1e-6
    return {"reset_seed": seed, "duration_s": float(data.time), "max_force_error_nm": force_error,
            "initial_state_control_sha256": initial_hash,
            "settled_support_screen_passed": passed,
            "first_unwanted_contact_s": next((row["time_s"] for row in traces if row["unwanted_contacts"]), None),
            "tail_summary": {"samples": len(tail),
                "all_four_loaded_fraction": float(np.mean([all_loaded(row) for row in tail])) if tail else None,
                "unwanted_contact_fraction": float(np.mean([bool(row["unwanted_contacts"]) for row in tail])) if tail else None,
                "mean_non_support_load_n": float(np.mean([row["non_support_normal_load_n"] for row in tail])) if tail else None,
                "max_base_speed_mps": max((row["base_speed_mps"] for row in tail), default=None),
                "max_joint_speed_radps": max((row["max_joint_speed_radps"] for row in tail), default=None),
                "max_contact_slip_mps": max((max(row["loaded_contact_slip_mps"]) for row in tail), default=None),
                "min_gap_m": min((min(row["gaps_m"]) for row in tail), default=None),
                "max_torque_ratio": max((row["max_torque_ratio"] for row in tail), default=None)},
            "trace": traces}


def main():
    path = ASSETS / "flat.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    validate_position_servos(model)
    results = [trial(model, seed) for seed in (None, 101, 102, 103, 104)]
    report = {"model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mujoco_version": mujoco.__version__,
              "controller": "Unchanged home-keyframe position targets; no base pinning, resets during rollout, policy or optimization",
              "limitations": "Five local initial conditions, 10s each. Last 2s screening thresholds are diagnostics, not hardware standards. Load >1N; no unwanted contact; base speed <0.01m/s; joint speed <0.1rad/s; loaded contact slip <0.01m/s; torque <95%. Penetration and foot tilt reported without claiming calibrated contact accuracy. Not a locomotion, disturbance-recovery or single-limb-unloading test.",
              "trials": results}
    output = ROOT / "output/rl/support_baseline_check.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: value for key, value in row.items() if key != "trace"} for row in results], indent=2))
    print(output)


if __name__ == "__main__":
    main()
