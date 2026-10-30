from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "output" / "rl" / "assets"
SUPPORTS = ("left_hand_collision", "right_hand_collision", "left_foot", "right_foot")
SITES = ("left_palm", "right_palm", "left_foot", "right_foot")


def prepare():
    ASSETS.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(ROOT / "g1_description" / "g1.xml").getroot()
    robot.find("compiler").set("meshdir", str(ROOT / "g1_description" / "meshes"))
    robot.find("option").attrib.update(timestep="0.002", iterations="50", ls_iterations="10", integrator="implicitfast")
    contact_section = robot.find("contact")
    if contact_section is None:
        contact_section = ET.SubElement(robot, "contact")
    for side in ("left", "right"):
        robot.find(f".//geom[@name='{side}_upper_arm']").attrib.update(fromto="0 0 -0.015 0.015 0 -0.065", size="0.035")
        robot.find(f".//geom[@name='{side}_forearm']").attrib.update(fromto="-0.08 0 0 0.145 0 0", size="0.025")
        robot.find(f".//geom[@name='{side}_hand_collision']").set("size", "0.035")
        ET.SubElement(contact_section, "exclude", body1=f"{side}_shoulder_yaw_link", body2=f"{side}_wrist_roll_rubber_hand")
    world = robot.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 0.1", friction="1 0.005 0.0001", rgba="0.25 0.25 0.3 1")
    sensors = robot.find("sensor")
    for name, site in zip(SUPPORTS, SITES):
        ET.SubElement(sensors, "contact", name=f"crawl_contact_{site}", geom1=name, geom2="floor", data="found", num="1", reduce="mindist")
        ET.SubElement(sensors, "framepos", name=f"crawl_pos_{site}", objtype="site", objname=site)
        ET.SubElement(sensors, "framelinvel", name=f"crawl_vel_{site}", objtype="site", objname=site)
    ET.SubElement(sensors, "contact", name="crawl_torso_contact", geom1="torso_collision_main", geom2="floor", data="found", num="1", reduce="mindist")
    path = ASSETS / "flat.pending.xml"
    ET.ElementTree(robot).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    pose = json.loads((ROOT / "crawl-pose.json").read_text())["poses"][0]
    pitch = float(pose["base_rpy"][1])
    data.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    for name, value in pose["joints"].items():
        joint = model.joint(name)
        if not joint.range[0] <= value <= joint.range[1]:
            raise ValueError(f"Pose violates joint limit: {name}")
        data.qpos[joint.qposadr[0]] = value
    data.qpos[2] = 0.29
    initial = data.qpos.copy()
    mujoco.mj_forward(model, data)
    targets = np.array([data.site(name).xpos.copy() for name in SITES])
    targets[:, 2] = [0.045, 0.045, 0.012, 0.012]
    active = np.r_[2, np.arange(6, model.nv)]
    jacobian = np.zeros((3, model.nv))
    for iteration in range(200):
        mujoco.mj_forward(model, data)
        matrices = []
        errors = []
        for site, target in zip(SITES, targets):
            mujoco.mj_jacSite(model, data, jacobian, None, model.site(site).id)
            matrices.append(jacobian[:, active] * 20)
            errors.append((target - data.site(site).xpos) * 20)
        matrices.append(np.eye(len(active)) * 0.3)
        errors.append(np.r_[initial[2] - data.qpos[2], initial[7:] - data.qpos[7:]] * 0.3)
        delta = np.linalg.lstsq(np.vstack(matrices), np.concatenate(errors), rcond=None)[0]
        velocity = np.zeros(model.nv)
        velocity[active] = np.clip(delta, -0.02, 0.02)
        mujoco.mj_integratePos(model, data.qpos, velocity, 1.0)
        data.qpos[7:] = np.clip(data.qpos[7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    mujoco.mj_forward(model, data)
    nominal = data.qpos.copy()
    support_ids = [model.geom(name).id for name in SUPPORTS]
    floor_id = model.geom("floor").id
    torso_id = model.geom("torso_collision_main").id
    arm_ids = [model.geom(name).id for name in ("left_upper_arm", "right_upper_arm")]
    arm_pairs = [(geom_id, torso_id) for geom_id in arm_ids]
    positions = np.r_[2, np.arange(7, model.nq)]
    lower = np.r_[0.22, model.jnt_range[1:, 0]]
    upper = np.r_[0.4, model.jnt_range[1:, 1]]

    def residual(values):
        data.qpos[positions] = values
        mujoco.mj_forward(model, data)
        gaps = [mujoco.mj_geomDistance(model, data, geom_id, floor_id, 1.0, None) - 0.002 for geom_id in support_ids]
        collisions = [min(mujoco.mj_geomDistance(model, data, first, second, 1.0, None) - 0.003, 0) for first, second in arm_pairs]
        horizontal = np.concatenate([data.site(name).xpos[:2] - target[:2] for name, target in zip(SITES, targets)])
        return np.r_[np.array(gaps) * 30, np.array(collisions) * 50, horizontal * 3, (values - nominal[positions]) * 0.1]

    values = nominal[positions].copy()
    for iteration in range(1000):
        error = residual(values)
        numerical_jacobian = np.empty((len(error), len(values)))
        for column in range(len(values)):
            shifted = values.copy()
            shifted[column] += 1e-5
            numerical_jacobian[:, column] = (residual(shifted) - error) / 1e-5
        delta = np.linalg.lstsq(np.vstack([numerical_jacobian, 0.03 * np.eye(len(values))]), np.r_[-error, np.zeros(len(values))], rcond=None)[0]
        delta *= min(1.0, 0.05 / max(np.max(np.abs(delta)), 1e-9))
        for scale in (1.0, 0.5, 0.25, 0.1, 0.01):
            candidate = np.clip(values + scale * delta, lower, upper)
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                values = candidate
                break
        else:
            break
    residual(values)
    minimum_distance = min((float(contact.dist) for contact in data.contact), default=0.0)
    if minimum_distance < -0.002:
        pairs = [(model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name, float(contact.dist)) for contact in data.contact]
        raise ValueError(f"Initial pose still penetrates: {pairs}; residual={residual(values).tolist()}")
    keyframes = ET.SubElement(robot, "keyframe")
    ET.SubElement(keyframes, "key", name="home", qpos=" ".join(map(str, data.qpos)), ctrl=" ".join(map(str, data.qpos[7:])))
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding="unicode")
    manifest = {
        "collision_variant": "training-only upper-arm and forearm capsules; 3.5 cm hand pads; replay with this scene, not the unmodified poser model",
        "source": "g1_description/g1.xml",
        "joint_names": [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]],
        "qpos": data.qpos.tolist(),
        "reference_up": data.sensordata[model.sensor("upvector_torso").adr[0]:model.sensor("upvector_torso").adr[0] + 3].tolist(),
        "site_heights": {name: float(data.site(name).xpos[2]) for name in SITES},
        "sim_dt": 0.002,
        "control_dt": 0.02,
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    path.replace(ASSETS / "flat.xml")
    nominal = data.qpos.copy()
    base_targets = np.array([data.site(name).xpos.copy() for name in SITES])
    reference = np.zeros((3, 128, model.nu), dtype=np.float32)
    speeds = np.array([0.05, 0.10, 0.15])
    for speed_index, speed in enumerate(speeds):
        for frame in range(128):
            phase = (frame / 128 + np.array([0, 0.5, 0.75, 0.25])) % 1
            desired = base_targets.copy()
            for limb, local_phase in enumerate(phase):
                stride = speed * 2.0 * 0.75
                if local_phase < 0.75:
                    desired[limb, 0] += stride * (0.5 - local_phase / 0.75)
                else:
                    swing_phase = (local_phase - 0.75) / 0.25
                    desired[limb, 0] += stride * (-0.5 + swing_phase ** 2 * (3 - 2 * swing_phase))
                    desired[limb, 2] += 0.08 * np.sin(np.pi * swing_phase)
            data.qpos[:] = nominal
            for iteration in range(60):
                mujoco.mj_forward(model, data)
                matrices = []
                errors = []
                for site, target in zip(SITES, desired):
                    mujoco.mj_jacSite(model, data, jacobian, None, model.site(site).id)
                    matrices.append(jacobian[:, 6:] * 20)
                    errors.append((target - data.site(site).xpos) * 20)
                matrices.append(np.eye(model.nu) * 0.15)
                errors.append((nominal[7:] - data.qpos[7:]) * 0.15)
                delta = np.linalg.lstsq(np.vstack(matrices), np.concatenate(errors), rcond=None)[0]
                data.qpos[7:] = np.clip(data.qpos[7:] + np.clip(delta, -0.05, 0.05), model.jnt_range[1:, 0], model.jnt_range[1:, 1])
            reference[speed_index, frame] = data.qpos[7:] - nominal[7:]
    np.savez(ASSETS / "reference.npz", speeds=speeds, offsets=reference)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
