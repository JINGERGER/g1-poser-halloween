from dataclasses import dataclass
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from unilab.base import registry

from crawl_task import ASSETS, SITES, CrawlCfg, CrawlEnv
from prepare import SUPPORTS
from servo_force import position_servo_force, validate_position_servos


def projected_gravity(quaternions):
    scalar, axis_x, axis_y, axis_z = np.asarray(quaternions).T
    return np.stack((2 * (scalar * axis_y - axis_x * axis_z),
                     -2 * (axis_y * axis_z + scalar * axis_x),
                     2 * (axis_x ** 2 + axis_y ** 2) - 1), axis=-1)


def prepare_direct(version=2):
    if version not in (2, 3):
        raise ValueError("Unsupported direct task version")
    destination = ASSETS / f"direct_v{version}"
    destination.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(ASSETS / "flat.xml").getroot()
    for index, geom in enumerate(robot.find("worldbody").iter("geom")):
        if not geom.get("name"):
            geom.set("name", f"direct_geom_{index}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    sensors = robot.find("sensor")
    if version == 3:
        for index, name in enumerate(SUPPORTS):
            ET.SubElement(sensors, "distance", name=f"balanced_gap_{index}", geom1=name, geom2="floor", cutoff="0.5")
    bad_contacts = []
    for index in range(model.ngeom):
        name = model.geom(index).name
        if name not in (*SUPPORTS, "floor") and (model.geom_contype[index] or model.geom_conaffinity[index]):
            sensor = f"direct_contact_{name}"
            ET.SubElement(sensors, "contact", name=sensor, geom1=name, geom2="floor", data="found", num="1", reduce="mindist")
            bad_contacts.append(sensor)
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    force_limits = []
    for index, joint_id in enumerate(model.actuator_trnid[:, 0]):
        if not model.jnt_actfrclimited[joint_id]:
            raise ValueError("Direct task requires joint-level force limits")
        ET.SubElement(sensors, "jointactuatorfrc", name=f"direct_torque_{index}", joint=model.joint(int(joint_id)).name)
        force_limits.append(model.jnt_actfrcrange[joint_id].tolist())
    manifest["direct_v2"] = {
        "bad_contacts": bad_contacts, "force_limits": force_limits,
        "gravity": projected_gravity(np.array([manifest["qpos"][3:7]]))[0].tolist(),
        "observation_protocol": "85d-v2-pelvis-projected-gravity",
    }
    if version == 3:
        manifest["balanced_v3"] = {"clearance": "signed collision geom distance to floor", "reward_protocol": "separate swing and stance normalization"}
    xml = ET.tostring(robot, encoding="unicode")
    mujoco.MjModel.from_xml_string(xml)
    (destination / "flat.xml").write_text(xml)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


@registry.envcfg("G1CrawlDirect23V2")
@dataclass
class DirectCfg(CrawlCfg):
    phase_period: float = 4.0
    lift_height: float = 0.02
    force_source: str = "legacy_sensor"


@registry.env("G1CrawlDirect23V2", sim_backend="mujoco")
class DirectEnv(CrawlEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if cfg.reference_motion:
            raise ValueError("Direct V2 must not load IK reference trajectories")
        if cfg.force_source not in ("legacy_sensor", "analytic_pd"):
            raise ValueError("Unknown force_source")
        if cfg.force_source == "analytic_pd":
            self.force_model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
            validate_position_servos(self.force_model)
        metadata = json.loads(Path(cfg.scene.model_file).with_name("manifest.json").read_text())["direct_v2"]
        self.gravity_target = np.array(metadata["gravity"], dtype=np.float32)
        self.bad_contacts = metadata["bad_contacts"]
        self.force_limits = np.array(metadata["force_limits"], dtype=np.float32)
        super().__init__(cfg, num_envs, backend_type)

    def observations(self, info, indices=None):
        observation = super().observations(info, indices)
        selection = slice(None) if indices is None else indices
        observation["obs"][:, 6:9] = projected_gravity(self._backend.get_base_quat()[selection])
        return observation

    def orientation_error(self):
        gravity = projected_gravity(self._backend.get_base_quat())
        return np.sum((gravity - self.gravity_target) ** 2, axis=1)

    def joint_forces(self, state):
        if self.cfg.force_source == "analytic_pd":
            targets = self.default_angles + self.cfg.control_config.action_scale * state.info["current_actions"]
            targets = np.clip(targets, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            return position_servo_force(self.force_model, targets, self.get_dof_pos(), self.get_dof_vel())
        return np.concatenate([self._backend.get_sensor_data(f"direct_torque_{index}").reshape(self.num_envs, 1) for index in range(23)], axis=1)

    def additional_terms(self, state):
        forces = self.joint_forces(state)
        limits = np.where(forces >= 0, self.force_limits[:, 1], -self.force_limits[:, 0])
        ratios = np.abs(forces) / limits
        bad = np.stack([self._backend.get_sensor_data(name).reshape(self.num_envs) > 0 for name in self.bad_contacts], axis=1).any(axis=1)
        return {
            "torque_utilization": np.mean(ratios ** 2, axis=1),
            "torque_near_limit": np.mean(np.maximum(ratios - 0.8, 0) ** 2, axis=1),
            "non_support_contact": bad.astype(float),
        }, bad


def balanced_terms(contacts, gaps, phase, lift_height, moving):
    swing = (phase >= 0.75) & moving[:, None]
    peak = (phase > 0.825) & (phase < 0.925) & moving[:, None]
    target = lift_height * np.sin(np.pi * np.clip((phase - 0.75) / 0.25, 0, 1))
    swing_count = np.maximum(swing.sum(axis=1), 1)
    stance_count = np.maximum((~swing).sum(axis=1), 1)
    return {
        "swing_gap": np.sum(np.exp(-((gaps - target) / 0.008) ** 2) * swing, axis=1) / swing_count,
        "swing_contact": np.sum(contacts & peak, axis=1),
        "stance_contact": np.sum(contacts & ~swing, axis=1) / stance_count,
        "stance_hover": np.sum(np.clip(np.maximum(gaps, 0) / lift_height, 0, 2) * ~swing, axis=1) / stance_count,
        "landing_contact": np.sum(contacts & (phase < 0.1) & moving[:, None], axis=1),
    }


@registry.envcfg("G1CrawlBalanced23V3")
@dataclass
class BalancedCfg(DirectCfg):
    pass


@registry.env("G1CrawlBalanced23V3", sim_backend="mujoco")
class BalancedEnv(DirectEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        manifest = json.loads(Path(cfg.scene.model_file).with_name("manifest.json").read_text())
        if "balanced_v3" not in manifest:
            raise ValueError("Balanced V3 requires distance sensors in its own model snapshot")
        if not np.isfinite(cfg.lift_height) or cfg.lift_height <= 0:
            raise ValueError("lift_height must be positive and finite")
        super().__init__(cfg, num_envs, backend_type)

    def limb_state(self, state):
        contacts = np.concatenate([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(self.num_envs, 1) for name in SITES], axis=1) > 0
        gaps = np.concatenate([self._backend.get_sensor_data(f"balanced_gap_{index}").reshape(self.num_envs, 1) for index in range(4)], axis=1)
        phase = self.limb_phase(state.info)
        return contacts, gaps, phase

    def additional_terms(self, state):
        terms, bad = super().additional_terms(state)
        contacts, gaps, phase = self.limb_state(state)
        terms.update(balanced_terms(contacts, gaps, phase, self.cfg.lift_height, self.gait_active(state.info)))
        return terms, bad

    def update_state(self, state):
        result = super().update_state(state)
        contacts, gaps, phase = self.limb_state(state)
        moving = self.gait_active(state.info)
        peak = (phase > 0.825) & (phase < 0.925) & moving[:, None]
        landing = (phase < 0.1) & moving[:, None]
        result.info["evaluation"].update({
            "limb_peak_geom_gap": gaps * peak,
            "limb_peak_clear": ((gaps >= 0.015) & ~contacts & peak).astype(float),
            "limb_landing_count": landing.astype(float),
            "limb_landing_contact": (contacts & landing).astype(float),
        })
        return result
