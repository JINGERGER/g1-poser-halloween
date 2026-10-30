import json
from datetime import datetime

import mujoco
import numpy as np

from artifact_utils import digest
from reference_policy_baseline import ROOT, materialize


def standing_pose(model, knee):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0., 0., 1., 1., 0., 0., 0.]
    for side, sign in (("left", 1.), ("right", -1.)):
        angles = {"hip_pitch": -knee / 2, "knee": knee, "ankle_pitch": -knee / 2,
                  "shoulder_roll": sign * .25, "elbow": .3}
        for suffix, value in angles.items():
            joint = model.joint(f"{side}_{suffix}_joint").id
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    bottoms = []
    for name in ("left_foot", "right_foot"):
        geom = model.geom(name).id
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        bottoms.append(data.geom_xpos[geom, 2] - np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom])
    assert abs(bottoms[0] - bottoms[1]) < 1e-8
    data.qpos[2] -= min(bottoms)
    mujoco.mj_forward(model, data)
    return data


def sample(model, data, feet, floor, joints, start_feet):
    loads = np.zeros(2)
    other, penetration, self_penetration = 0., 0., 0.
    for index, contact in enumerate(data.contact):
        penetration = max(penetration, -contact.dist)
        if floor not in (contact.geom1, contact.geom2):
            self_penetration = max(self_penetration, -contact.dist)
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0., float(wrench[0]))
        geom = contact.geom2 if contact.geom1 == floor else contact.geom1
        if geom in feet:
            loads[feet.index(geom)] += force
        else:
            other += force
    angles = data.qpos[model.jnt_qposadr[joints]]
    ranges = model.jnt_range[joints]
    excess = max(0., float(np.max(ranges[:, 0] - angles)), float(np.max(angles - ranges[:, 1])))
    limits = np.max(np.abs(model.jnt_actfrcrange[joints]), axis=1)
    ratio = float(np.max(np.abs(data.qfrc_actuator[model.jnt_dofadr[joints]]) / limits))
    tilt = float(np.arccos(np.clip(data.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2], -1., 1.)))
    foot_drift = float(np.max(np.linalg.norm(data.geom_xpos[feet, :2] - start_feet[:, :2], axis=1)))
    return [data.time, data.qpos[2], tilt, np.linalg.norm(data.qvel[:3]),
            np.linalg.norm(data.qvel[3:6]), *loads, other, penetration,
            self_penetration, ratio, excess, foot_drift]


def evaluate(samples, completed, replay):
    settled = samples[samples[:, 0] >= .5]
    tail = samples[samples[:, 0] >= 4.]
    gates = {
        "completed_5s": completed,
        "replay": replay,
        "height_055_to_095m": bool(np.all((samples[:, 1] >= .55) & (samples[:, 1] <= .95))),
        "tilt_below_015rad": bool(np.all(samples[:, 2] <= .15)),
        "penetration_5mm": bool(np.all(samples[:, 8] <= .005)),
        "self_penetration_1mm": bool(np.all(samples[:, 9] <= .001)),
        "nonfoot_load_1n": bool(np.all(samples[:, 7] <= 1.)),
        "torque_below_95percent": bool(np.all(samples[:, 10] < .95)),
        "joint_limits_1mrad": bool(np.all(samples[:, 11] <= .001)),
        "foot_drift_1cm": bool(np.all(samples[:, 12] <= .01)),
        "both_feet_loaded_after_05s": bool(len(settled) and np.all(settled[:, 5:7] > 5.)),
        "settled_linear_speed_003": bool(len(tail) and np.all(tail[:, 3] <= .03)),
        "settled_angular_speed_005": bool(len(tail) and np.all(tail[:, 4] <= .05)),
    }
    return gates


def main():
    folder, unused = materialize()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    reference = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    mapping = {reference.joint(int(joint)).name: index for index, joint in enumerate(reference.actuator_trnid[:, 0])}
    output = ROOT / "output/rl/standing_endpoint" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(__file__),
              "model_sha256": digest(model_path), "reference_model_sha256": digest(folder / "scene.xml"),
              "assets": {str(path.relative_to(model_path.parent)): digest(path)
                         for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
              "protocol": "Six predeclared 5s constant-position PD holds: knee .2/.4/.6rad crossed with native/reference gains. "
                          "Reference gains multiplied by 1.4 as crawl; unmatched wrists unchanged. "
                          "All cases solref .005, dt .002. Upright base, symmetric bent legs, shoulders +/- .25, elbows .3. "
                          "Foot bottoms initialized at plane; zero velocity. No external force or intermediate state reset. "
                          "Gates fixed before execution; full transient audited, load gate after .5s and speed gate after 4s.",
              "sample_columns": ["time", "height", "tilt", "linear_speed", "angular_speed", "left_load", "right_load",
                                 "nonfoot_load", "penetration", "self_penetration", "torque_ratio", "joint_excess", "foot_drift"],
              "limitations": "Initialized holding only, not standing recovery or transition feasibility. "
                             "Existing convex-hull and adjacent-body collision filtering assumptions remain. "
                             "Foot-origin drift is a proxy, not contact-point slip velocity. No calibrated impact safety threshold.",
              "runs": []}
    try:
        for knee in (.2, .4, .6):
            for profile in ("native", "reference"):
                model = mujoco.MjModel.from_xml_path(str(model_path))
                model.geom_solref[:, 0] = .005
                joints = model.actuator_trnid[:, 0]
                assert model.opt.timestep == .002 and model.na == 0
                assert np.all(model.jnt_limited[joints]) and np.all(model.jnt_actfrclimited[joints])
                if profile == "reference":
                    for actuator, joint in enumerate(joints):
                        name = model.joint(int(joint)).name
                        if name in mapping:
                            source = mapping[name]
                            model.actuator_gainprm[actuator, 0] = reference.actuator_gainprm[source, 0] * 1.4
                            model.actuator_biasprm[actuator, 1:3] = reference.actuator_biasprm[source, 1:3] * 1.4
                data = standing_pose(model, knee)
                target = data.qpos[model.jnt_qposadr[joints]].copy()
                assert np.all(target >= model.actuator_ctrlrange[:, 0]) and np.all(target <= model.actuator_ctrlrange[:, 1])
                data.ctrl[:] = target
                mujoco.mj_forward(model, data)
                feet = [model.geom(name).id for name in ("left_foot", "right_foot")]
                floor = model.geom("floor").id
                start_feet = data.geom_xpos[feet].copy()
                positions, velocities = [data.qpos.copy()], [data.qvel.copy()]
                samples = [sample(model, data, feet, floor, joints, start_feet)]
                for unused_step in range(2500):
                    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise ValueError("Nonfinite standing state")
                    positions.append(data.qpos.copy())
                    velocities.append(data.qvel.copy())
                    samples.append(sample(model, data, feet, floor, joints, start_feet))
                    if data.qpos[2] < .4 or samples[-1][2] > .8:
                        break
                replay = mujoco.MjData(model)
                replay.qpos[:], replay.qvel[:], replay.ctrl[:] = positions[0], velocities[0], target
                mujoco.mj_forward(model, replay)
                for position, velocity in zip(positions[1:], velocities[1:]):
                    mujoco.mj_step(model, replay)
                    mujoco.mj_forward(model, replay)
                    np.testing.assert_allclose(replay.qpos, position, atol=1e-10, rtol=0)
                    np.testing.assert_allclose(replay.qvel, velocity, atol=1e-10, rtol=0)
                samples = np.array(samples)
                gates = evaluate(samples, bool(data.time >= 5. - 1e-8), True)
                path = output / f"knee_{knee}_{profile}.npz"
                np.savez_compressed(path, qpos=positions, qvel=velocities, ctrl=target, samples=samples,
                                    gainprm=model.actuator_gainprm, biasprm=model.actuator_biasprm)
                result = {"knee_rad": knee, "gain_profile": profile, "passed": all(gates.values()), "gates": gates,
                          "elapsed_s": float(data.time), "peak_total_foot_load_n": float(samples[:, 5:7].sum(axis=1).max()),
                          "max_torque_ratio": float(samples[:, 10].max()), "max_tilt_rad": float(samples[:, 2].max()),
                          "max_penetration_m": float(samples[:, 8].max()), "max_self_penetration_m": float(samples[:, 9].max()),
                          "max_foot_drift_m": float(samples[:, 12].max()), "trajectory": str(path), "sha256": digest(path)}
                report["runs"].append(result)
                print(json.dumps(result), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
