import argparse
import hashlib
import itertools
import json

import mujoco
import numpy as np

from diagnose_lift import solve_reference
from prepare import ASSETS, ROOT, SITES, SUPPORTS


def gaps_and_points(model, data):
    gaps, points = [], []
    for name in SUPPORTS:
        segment = np.zeros(6)
        gaps.append(mujoco.mj_geomDistance(model, data, model.geom(name).id, model.geom("floor").id, 1.0, segment))
        points.append(segment[3:].copy())
    return np.array(gaps), np.array(points)


def unwanted_contacts(model, data):
    floor = model.geom("floor").id
    supports = {model.geom(name).id for name in SUPPORTS}
    return [contact for contact in data.contact if not (floor in (contact.geom1, contact.geom2) and (contact.geom1 in supports or contact.geom2 in supports))]


def refine(model, data, nominal, xy_targets, limb):
    addresses = np.r_[2, np.arange(7, model.nq)]
    lower = np.r_[0.16, model.jnt_range[1:, 0]]
    upper = np.r_[0.40, model.jnt_range[1:, 1]]
    target_gap = np.full(4, -0.0001)
    target_gap[limb] = 0.02

    def residual(values):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        gaps, points = gaps_and_points(model, data)
        xy = np.array([data.site(name).xpos[:2] for name in SITES])
        penetration = min((float(contact.dist) for contact in unwanted_contacts(model, data)), default=0.0)
        return np.r_[(gaps - target_gap) * 50, (xy - xy_targets).ravel() * 10,
                     (values - nominal[addresses]) * 0.03, min(penetration, 0) * 100]

    values = data.qpos[addresses].copy()
    for iteration in range(30):
        error = residual(values)
        jacobian = np.empty((len(error), len(values)))
        for column in range(len(values)):
            shifted = values.copy()
            shifted[column] += 1e-5
            jacobian[:, column] = (residual(shifted) - error) / 1e-5
        delta = np.linalg.lstsq(np.vstack((jacobian, 0.03 * np.eye(len(values)))), np.r_[-error, np.zeros(len(values))], rcond=None)[0]
        delta *= min(1.0, 0.04 / max(np.max(np.abs(delta)), 1e-9))
        for scale in (1, 0.5, 0.1):
            candidate = np.clip(values + scale * delta, lower, upper)
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                values = candidate
                break
        else:
            break
    residual(values)


def statics(model, data, limb):
    floor = model.geom("floor").id
    supports = {model.geom(name).id for index, name in enumerate(SUPPORTS) if index != limb}
    contacts = [contact for contact in data.contact if floor in (contact.geom1, contact.geom2) and (contact.geom1 in supports or contact.geom2 in supports) and contact.dist <= 0.001]
    joint_ids = model.actuator_trnid[:, 0]
    columns = []
    jacobian = np.zeros((3, model.nv))
    for contact in contacts:
        geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
        mujoco.mj_jac(model, data, jacobian, None, contact.pos, int(model.geom_bodyid[geom]))
        columns.append(jacobian[2].copy())
    if not columns:
        return {"equilibrium_found": False, "contact_points": 0}
    mapping = np.array(columns).T
    bias = data.qfrc_bias - data.qfrc_passive
    weighted = mapping[:6].copy()
    weighted[3:] /= 0.3
    target = bias[:6].copy()
    target[3:] /= 0.3
    best = None
    for mask in range(1, 1 << len(columns)):
        selected = [index for index in range(len(columns)) if mask & (1 << index)]
        active = np.linalg.lstsq(weighted[:, selected], target, rcond=None)[0]
        if np.any(active < -1e-8):
            continue
        force = np.zeros(len(columns))
        force[selected] = np.maximum(active, 0)
        residual = bias - mapping @ force
        torque = residual[model.jnt_dofadr[joint_ids]]
        limits = model.jnt_actfrcrange[joint_ids]
        ratios = np.abs(torque) / np.where(torque >= 0, limits[:, 1], -limits[:, 0])
        ctrl = (torque - model.actuator_biasprm[:, 0] - model.actuator_biasprm[:, 1] * data.qpos[model.jnt_qposadr[joint_ids]]) / model.actuator_gainprm[:, 0]
        equilibrium = np.linalg.norm(residual[:3]) < 0.2 and np.linalg.norm(residual[3:6]) < 0.05
        score = (not equilibrium, float(np.max(ratios)) if equilibrium else float(np.linalg.norm(weighted @ force - target)))
        if best is None or score < best[0]:
            best = (score, {
                "equilibrium_found": bool(equilibrium), "contact_points": len(columns),
                "linear_residual_n": float(np.linalg.norm(residual[:3])),
                "angular_residual_nm": float(np.linalg.norm(residual[3:6])),
                "max_torque_limit_ratio": float(np.max(ratios)),
                "most_loaded_joint": model.joint(int(joint_ids[np.argmax(ratios)])).name,
                "ctrl_in_range": bool(np.all(ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(ctrl <= model.actuator_ctrlrange[:, 1])),
                "ctrl": ctrl.tolist(), "normal_forces_n": force.tolist(),
            })
    return best[1] if best else {"equilibrium_found": False, "contact_points": len(columns)}


def self_check():
    for offset, expected in ((0, True), (1, False)):
        spheres = "".join(f'<geom name="{name}" type="sphere" size="0.1" pos="{position} 0.099"/>' for name, position in zip(SUPPORTS, ("-0.2 -0.2", "0.2 -0.2", "0 0.2", "0 -0.3")))
        model = mujoco.MjModel.from_xml_string(f'''
        <mujoco><worldbody><geom name="floor" type="plane" size="2 2 0.1"/>
          <body><freejoint/><inertial pos="{offset} 0 0.1" mass="1" diaginertia="0.1 0.1 0.1"/>
            {spheres}
            <body pos="0 0 0.2"><joint name="rotor" axis="0 0 1" range="-1 1" actuatorfrcrange="-10 10"/>
              <geom type="sphere" size="0.01" mass="0.01" contype="0" conaffinity="0"/>
            </body>
          </body></worldbody><actuator><position joint="rotor" kp="40" ctrlrange="-1 1"/></actuator>
        </mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        result = statics(model, data, 3)
        assert result["equilibrium_found"] == expected
        if expected:
            assert np.isclose(sum(result["normal_forces_n"]), model.body_mass.sum() * 9.81)
            assert result["ctrl_in_range"] and result["max_torque_limit_ratio"] < 1e-8
    print("Static solver positive/negative control checks passed", flush=True)


def hold(model, candidate, limb):
    data = mujoco.MjData(model)
    data.qpos[:] = candidate["qpos"]
    data.ctrl[:] = candidate["statics"]["ctrl"]
    mujoco.mj_forward(model, data)
    initial_linear_acceleration = float(np.linalg.norm(data.qacc[:3]))
    initial_joint_acceleration = float(np.max(np.abs(data.qacc[6:])))
    origin = data.qpos[:2].copy()
    traces = []
    floor = model.geom("floor").id
    support_ids = [model.geom(name).id for name in SUPPORTS]
    joint_ids = model.actuator_trnid[:, 0]
    limits = model.jnt_actfrcrange[joint_ids]
    ended = "duration reached"
    for step in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite hold state")
        if step % 10 == 9:
            mujoco.mj_forward(model, data)
            gaps, points = gaps_and_points(model, data)
            stance = [index for index in range(4) if index != limb]
            bad = any(contact.dist < -0.0005 for contact in unwanted_contacts(model, data))
            loads = np.zeros(4)
            wrench = np.zeros(6)
            for contact_index, contact in enumerate(data.contact):
                if floor in (contact.geom1, contact.geom2):
                    mujoco.mj_contactForce(model, data, contact_index, wrench)
                    for support_index, geom in enumerate(support_ids):
                        if geom in (contact.geom1, contact.geom2):
                            loads[support_index] += max(0.0, wrench[0])
            torque = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
            ratio = np.abs(torque) / np.where(torque >= 0, limits[:, 1], -limits[:, 0])
            traces.append({"time": float(data.time), "gap_m": float(gaps[limb]),
                           "stance_near_floor": bool(np.all(np.abs(gaps[stance]) <= 0.002)),
                           "stance_loaded": bool(np.all(loads[stance] > 1.0)),
                           "loads_n": loads.tolist(), "max_torque_ratio": float(ratio.max()),
                           "drift_m": float(np.linalg.norm(data.qpos[:2] - origin)), "bad_contact": bad,
                           "unwanted_contacts": [[model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name, float(contact.dist)] for contact in unwanted_contacts(model, data) if contact.dist < -0.0005]})
            if bad or data.qpos[2] < 0.16:
                ended = "unwanted penetration" if bad else "base height below 0.16 m"
                break
    return {"duration_s": float(data.time), "ended": ended,
            "initial_base_acceleration_mps2": initial_linear_acceleration,
            "initial_max_joint_acceleration_radps2": initial_joint_acceleration,
            "clear_fraction": float(np.mean([row["gap_m"] >= 0.015 for row in traces])),
            "stance_near_floor_fraction": float(np.mean([row["stance_near_floor"] for row in traces])),
            "stance_loaded_fraction": float(np.mean([row["stance_loaded"] for row in traces])),
            "max_torque_limit_ratio": max(row["max_torque_ratio"] for row in traces),
            "max_drift_m": max(row["drift_m"] for row in traces),
            "trace": traces}


def main():
    parser = argparse.ArgumentParser(description="Read-only geometry and bounded static support audit")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check_only:
        return
    source = mujoco.MjModel.from_xml_path(str(ROOT / "g1_description/g1.xml"))
    model_path = ASSETS / "flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    nominal = data.qpos.copy()
    targets = np.array([data.site(name).xpos.copy() for name in SITES])
    differences = []
    for index in range(source.ngeom):
        name = source.geom(index).name
        if not name:
            continue
        other = model.geom(name).id
        changes = {attribute: {"source": getattr(source, attribute)[index].tolist(), "training": getattr(model, attribute)[other].tolist()}
                   for attribute in ("geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_condim", "geom_contype", "geom_conaffinity")
                   if not np.allclose(getattr(source, attribute)[index], getattr(model, attribute)[other])}
        if changes:
            differences.append({"geom": name, "changes": changes})
    assert np.allclose(source.body_mass, model.body_mass)
    assert np.allclose(source.body_inertia, model.body_inertia)
    assert np.allclose(source.body_ipos, model.body_ipos)
    assert np.allclose(source.body_iquat, model.body_iquat)
    assert np.array_equal(source.jnt_range, model.jnt_range)
    assert np.array_equal(source.jnt_actfrcrange, model.jnt_actfrcrange)
    assert np.array_equal(source.actuator_gainprm, model.actuator_gainprm)
    assert np.array_equal(source.actuator_biasprm, model.actuator_biasprm)
    report = {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "mass_kg": float(model.body_mass.sum()), "mass_inertia_limits_gains_unchanged": True,
              "geometry_differences": differences, "initial_gap_m": gaps_and_points(model, data)[0].tolist(),
              "support_site_minus_geom_center_m": {name: (data.site(site).xpos - data.geom(name).xpos).tolist() for name, site in zip(SUPPORTS, SITES)},
              "foot_normal_z": {name: float(data.geom(name).xmat.reshape(3, 3)[2, 2]) for name in SUPPORTS[2:]},
              "initial_contact_count": data.ncon, "trials": []}
    original_pose = mujoco.MjData(source)
    original_pose.qpos[:] = nominal
    mujoco.mj_forward(source, original_pose)
    report["source_robot_contacts_at_training_pose"] = [{"geoms": [source.geom(int(contact.geom1)).name, source.geom(int(contact.geom2)).name], "distance_m": float(contact.dist)} for contact in original_pose.contact]
    probe = mujoco.MjData(model)
    probe.qpos[:] = nominal
    probe.qpos[2] -= 0.003
    mujoco.mj_forward(model, probe)
    report["ground_probe_contacts"] = [{"geoms": [model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name], "dim": int(contact.dim), "friction": contact.friction.tolist(), "distance_m": float(contact.dist)} for contact in probe.contact]
    for limb in (2, 3):
        for forward, lateral, vertical in itertools.product((0, 0.04, 0.08, 0.12, 0.16), (0, 0.04, 0.08, 0.12), (-0.02, 0, 0.02)):
            data.qpos[:] = nominal
            data.qvel[:] = 0
            shift = [forward, lateral * (-1 if limb == 2 else 1), vertical]
            data.qpos[:3] += shift
            desired = targets.copy()
            desired[limb, 2] += 0.02
            solve_reference(model, data, desired, nominal, model.actuator_trnid[:, 0])
            refine(model, data, nominal, targets[:, :2], limb)
            gaps, points = gaps_and_points(model, data)
            stance = [index for index in range(4) if index != limb]
            penetration = min((float(contact.dist) for contact in unwanted_contacts(model, data)), default=0.0)
            geometry_ok = bool(np.max(np.abs(gaps[stance])) < 0.001 and gaps[limb] >= 0.018 and penetration >= -0.0005)
            static = statics(model, data, limb) if geometry_ok else {"equilibrium_found": False}
            passed = bool(geometry_ok and static["equilibrium_found"] and static["max_torque_limit_ratio"] <= 0.95 and static["ctrl_in_range"])
            row = {"limb": SITES[limb], "root_shift_seed_m": shift, "gaps_m": gaps.tolist(), "unwanted_min_dist_m": penetration,
                   "geometry_ok": geometry_ok, "static_screen_passed": passed, "statics": static, "qpos": data.qpos.tolist()}
            report["trials"].append(row)
        choices = [row for row in report["trials"] if row["limb"] == SITES[limb] and row["static_screen_passed"]]
        choices.sort(key=lambda row: row["statics"]["max_torque_limit_ratio"])
        unique = []
        for row in choices:
            if all(np.linalg.norm(np.array(row["qpos"]) - other["qpos"]) > 0.01 for other in unique):
                unique.append(row)
        for row in unique[:2]:
            row["physical_hold"] = hold(model, row, limb)
        print(json.dumps({"limb": SITES[limb], "searched": 60, "geometry_ok": sum(row["geometry_ok"] for row in report["trials"] if row["limb"] == SITES[limb]), "static_candidates": len(choices), "distinct_candidates": len(unique)}, indent=2), flush=True)
    report["limitations"] = "Bounded fixed-orientation local search, unchanged stance XY targets, vertical-only nonnegative contact forces, subset least-squares candidates. Not a full friction-cone feasibility solver or proof of impossibility. Static screen uses contact geometry and joint limits; only up to two best candidates per side get a 3-second physical fixed-PD hold. Initialization of hypothetical poses is diagnostic only, not a trained rollout. Hardware parameters remain unverified."
    output = ROOT / "output/rl/geometry_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
