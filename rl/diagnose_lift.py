from pathlib import Path
import argparse
import hashlib
import json

import mujoco
import numpy as np

from prepare import ASSETS, ROOT, SITES, SUPPORTS


def smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3 - 2 * value)


def triangle_margin(point, vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = point - vertices
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    area = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                  - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    return float(np.min(np.sign(area) * cross / np.maximum(np.linalg.norm(edges, axis=1), 1e-9)))


def solve_reference(model, reference, targets, nominal, joint_ids):
    addresses = model.jnt_qposadr[joint_ids]
    velocities = model.jnt_dofadr[joint_ids]
    jacobian = np.zeros((3, model.nv))
    for iteration in range(30):
        mujoco.mj_forward(model, reference)
        matrices, errors = [], []
        for site, target in zip(SITES, targets):
            mujoco.mj_jacSite(model, reference, jacobian, None, model.site(site).id)
            matrices.append(jacobian[:, velocities] * 20)
            errors.append((target - reference.site(site).xpos) * 20)
        matrices.append(np.eye(model.nu) * 0.1)
        errors.append((nominal[addresses] - reference.qpos[addresses]) * 0.1)
        delta = np.linalg.lstsq(np.vstack(matrices), np.concatenate(errors), rcond=None)[0]
        reference.qpos[addresses] = np.clip(reference.qpos[addresses] + np.clip(delta, -0.04, 0.04),
                                            model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    mujoco.mj_forward(model, reference)
    return reference.qpos[addresses].copy()


def load_compensation(model, reference, stance, joint_ids):
    jacobian = np.zeros((3, model.nv))
    columns = []
    for index in stance:
        mujoco.mj_jacSite(model, reference, jacobian, None, model.site(SITES[index]).id)
        columns.append(jacobian[2].copy())
    support_map = np.array(columns).T
    forces = np.zeros(len(stance))
    residual = float(np.linalg.norm(reference.qfrc_bias[:6]))
    for mask in range(1, 1 << len(stance)):
        selected = [index for index in range(len(stance)) if mask & (1 << index)]
        candidate = np.linalg.lstsq(support_map[:6, selected], reference.qfrc_bias[:6], rcond=None)[0]
        error = np.linalg.norm(support_map[:6, selected] @ candidate - reference.qfrc_bias[:6])
        if np.all(candidate >= 0) and error < residual:
            forces[:] = 0
            forces[selected] = candidate
            residual = float(error)
    torque = reference.qfrc_bias - support_map @ forces
    return torque[model.jnt_dofadr[joint_ids]], float(residual)


def trial(model, limb, shift, height, controller="nominal"):
    data, reference = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    nominal = data.qpos.copy()
    reference.qpos[:] = nominal
    targets = np.array([data.site(name).xpos.copy() for name in SITES])
    joint_ids = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joint_ids]
    floor = model.geom("floor").id
    supports = [model.geom(name).id for name in SUPPORTS]
    stance = [index for index in range(4) if index != limb]
    vertices = targets[stance, :2]
    center = vertices.mean(axis=0)
    displacement = center - data.subtree_com[model.body("pelvis").id, :2]
    displacement *= min(1.0, 0.08 / max(np.linalg.norm(displacement), 1e-9)) if shift else 0.0
    records = []
    failure = None
    integral = np.zeros(model.nu)
    joint_limited = model.jnt_actfrclimited[joint_ids].astype(bool)
    joint_bounds = np.maximum(np.abs(model.jnt_actfrcrange[joint_ids]).max(axis=1), 1e-9)
    for step in range(250):
        elapsed = step * 0.02
        phase = "settle" if elapsed < 0.5 else "shift" if elapsed < 1.5 else "lift" if elapsed < 2.5 else "hold" if elapsed < 3.5 else "lower" if elapsed < 4.5 else "recover"
        desired = targets.copy()
        lift = height * smooth((elapsed - 1.5) / 1.0) * (1 - smooth((elapsed - 3.5) / 1.0)) if limb >= 0 else 0.0
        if limb >= 0:
            desired[limb, 2] += lift
        reference.qpos[:7] = nominal[:7]
        reference.qpos[:2] += displacement * smooth((elapsed - 0.5) / 1.0)
        command = solve_reference(model, reference, desired, nominal, joint_ids) if limb >= 0 else nominal[addresses]
        compensation = np.zeros(model.nu)
        wrench_residual = 0.0
        if controller == "load-compensated":
            mujoco.mj_forward(model, reference)
            all_torque, all_residual = load_compensation(model, reference, list(range(4)), joint_ids)
            stance_torque, stance_residual = load_compensation(model, reference, stance, joint_ids)
            unload = smooth((elapsed - 0.5) / 1.0) * (1 - smooth((elapsed - 3.5) / 1.0)) if limb >= 0 else 0.0
            torque = (1 - unload) * all_torque + unload * stance_torque
            wrench_residual = (1 - unload) * all_residual + unload * stance_residual
            integral = np.clip(integral + 0.02 * (command - data.qpos[addresses]), -0.15, 0.15)
            compensation = np.clip(torque / model.actuator_gainprm[:, 0] + integral, -0.5, 0.5)
        data.ctrl[:] = np.clip(command + compensation, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        saturation = 0.0 if model.actuator_forcelimited.any() else None
        peak_force = 0.0
        joint_saturation = 0.0 if joint_limited.any() else None
        saturated_joints = set()
        undesired = set()
        for substep in range(round(0.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak_force = max(peak_force, float(np.max(np.abs(data.actuator_force))))
            if joint_limited.any():
                ratios = np.abs(data.qfrc_actuator[model.jnt_dofadr[joint_ids]]) / joint_bounds
                joint_saturation = max(joint_saturation, float(np.max(ratios[joint_limited])))
                saturated_joints.update(model.joint(int(joint_ids[index])).name for index in np.flatnonzero(joint_limited & (ratios >= 0.99)))
            limited = model.actuator_forcelimited.astype(bool)
            bounds = np.maximum(np.abs(model.actuator_forcerange).max(axis=1), 1e-9)
            if limited.any():
                saturation = max(saturation, float(np.max(np.abs(data.actuator_force[limited]) / bounds[limited])))
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if floor in pair:
                    for geom in pair - {floor} - set(supports):
                        undesired.add(model.geom(geom).name or str(geom))
        mujoco.mj_forward(model, data)
        positions = np.array([data.site(name).xpos.copy() for name in SITES])
        contacts = [any(floor in (contact.geom1, contact.geom2) and geom in (contact.geom1, contact.geom2)
                        for contact in data.contact) for geom in supports]
        normal_forces = np.zeros(4)
        contact_force = np.zeros(6)
        for contact_index, contact in enumerate(data.contact):
            if floor in (contact.geom1, contact.geom2):
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                for support_index, geom in enumerate(supports):
                    if geom in (contact.geom1, contact.geom2):
                        normal_forces[support_index] += max(0.0, contact_force[0])
        tilt = float(2 * np.arccos(np.clip(abs(np.dot(data.qpos[3:7], nominal[3:7])), 0, 1)))
        gap = float(mujoco.mj_geomDistance(model, data, supports[limb], floor, 1.0, None)) if limb >= 0 else None
        record = {
            "time": float(data.time), "phase": phase, "target_lift_m": float(lift),
            "site_lift_m": float(positions[limb, 2] - targets[limb, 2]) if limb >= 0 else None,
            "swing_geom_floor_gap_m": gap, "contacts": contacts,
            "com_triangle_margin_m": triangle_margin(data.subtree_com[model.body("pelvis").id, :2], positions[stance, :2]) if limb >= 0 else None,
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean((command - data.qpos[addresses]) ** 2))),
            "peak_joint_force_limit_ratio": joint_saturation,
            "saturated_joints": sorted(saturated_joints),
            "support_normal_forces_n": normal_forces.tolist(),
            "compensation_max_rad": float(np.max(np.abs(compensation))),
            "reference_wrench_residual": wrench_residual,
            "peak_actuator_force_limit_ratio": saturation,
            "peak_actuator_force": peak_force,
            "root_orientation_error_deg": float(np.degrees(tilt)),
            "stance_xy_drift_m": float(np.max(np.linalg.norm(positions[stance, :2] - targets[stance, :2], axis=1))),
            "reference_site_error_m": float(np.max(np.linalg.norm(np.array([reference.site(name).xpos for name in SITES]) - desired, axis=1))) if limb >= 0 else 0.0,
            "reference_min_contact_distance_m": min((float(contact.dist) for contact in reference.contact), default=0.0) if limb >= 0 else None,
            "undesired_floor_contacts": sorted(undesired),
        }
        records.append(record)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite physics state")
        if data.qpos[2] < 0.16 or tilt > np.radians(35) or undesired:
            failure = "height, orientation, or non-support floor contact"
            break
    held = [record for record in records if record["phase"] == "hold"]
    return {
        "limb": SITES[limb] if limb >= 0 else "all_support_control",
        "shift_com": shift, "reference_root_shift_xy_m": displacement.tolist(),
        "duration_s": float(data.time), "failure": failure,
        "hold_mean_lift_m": float(np.mean([record["site_lift_m"] for record in held])) if held and limb >= 0 else None,
        "hold_swing_contact_fraction": float(np.mean([record["contacts"][limb] for record in held])) if held and limb >= 0 else None,
        "hold_mean_geom_gap_m": float(np.mean([record["swing_geom_floor_gap_m"] for record in held])) if held and limb >= 0 else None,
        "minimum_com_triangle_margin_m": min(record["com_triangle_margin_m"] for record in records) if limb >= 0 else None,
        "max_joint_tracking_rmse_rad": max(record["joint_tracking_rmse_rad"] for record in records),
        "max_actuator_force_limit_ratio": max(record["peak_actuator_force_limit_ratio"] for record in records) if model.actuator_forcelimited.any() else None,
        "max_actuator_force": max(record["peak_actuator_force"] for record in records),
        "max_joint_force_limit_ratio": max(record["peak_joint_force_limit_ratio"] for record in records) if joint_limited.any() else None,
        "saturated_joints": sorted({name for record in records for name in record["saturated_joints"]}),
        "trace": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Physics-only single-limb lift diagnostic; no learning or state teleportation")
    parser.add_argument("--height", type=float, default=0.02)
    parser.add_argument("--model", type=Path, default=ASSETS / "flat.xml")
    parser.add_argument("--controller", choices=("nominal", "load-compensated"), default="nominal")
    parser.add_argument("--output", type=Path, default=ROOT / "output/rl/lift_diagnostic.json")
    args = parser.parse_args()
    if not np.isfinite(args.height) or not 0 < args.height <= 0.08:
        parser.error("--height must be in (0, 0.08] metres")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if not np.isclose(round(0.02 / model.opt.timestep) * model.opt.timestep, 0.02):
        parser.error("Model timestep must divide 0.02 seconds")
    report = {
        "model": str(args.model.resolve()), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__, "height_m": args.height,
        "force_limited_actuators": int(model.actuator_forcelimited.sum()),
        "force_limited_joints": int(model.jnt_actfrclimited.sum()),
        "controller": args.controller,
        "caveat": "COM margin is a quasi-static heuristic, not a dynamic stability certificate. A failed all-support control confounds lift attribution.",
        "trials": [],
    }
    for limb, shift in [(-1, False)] + [(limb, shift) for limb in range(4) for shift in (False, True)]:
        result = trial(model, limb, shift, args.height, args.controller)
        report["trials"].append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "trace"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
