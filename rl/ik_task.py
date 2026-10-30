from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from unilab.base import registry
from unilab.dr import DomainRandomizationProvider, ResetPlan

from direct_task import DirectCfg, DirectEnv, projected_gravity
from ik_assets import PERIOD, SPEED, foot_path, motion_time, reference_at
from prepare import SITES
from servo_force import position_servo_force


def bounded_reward(quality, score, failed, dt):
    return np.where(failed, 0, np.clip(quality, 0, 1) * np.clip(score, 0, 1) * dt).astype(np.float32)


@registry.envcfg("G1IKTracking23")
@dataclass
class IKCfg(DirectCfg):
    force_source: str = "analytic_pd"
    reference_motion: bool = False


class IKReset(DomainRandomizationProvider):
    def validate(self, env, capabilities):
        if env._num_action != 23:
            raise ValueError("Expected 23 residual actions")

    def build_reset_plan(self, env, env_ids):
        count = len(env_ids)
        anchor_ids = env.rng.integers(2, size=count)
        positions = env.anchor_qpos[anchor_ids].copy()
        positions[:, :2] += env.rng.uniform(-0.03, 0.03, (count, 2))
        info = {"anchor_id": anchor_ids[:, None], "elapsed": np.zeros((count, 1)),
                "current_actions": np.zeros((count, 23), dtype=np.float32),
                "last_actions": np.zeros((count, 23), dtype=np.float32)}
        return ResetPlan(env_ids=env_ids, qpos=positions, qvel=env.anchor_qvel[anchor_ids].copy(), info_updates=info)

    def build_reset_observation(self, env, env_ids, info_updates):
        return env.observations(info_updates, env_ids)


@registry.env("G1IKTracking23", sim_backend="mujoco")
class IKEnv(DirectEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        path = Path(cfg.scene.model_file)
        metadata = json.loads(path.with_name("manifest.json").read_text())["ik_tracking"]
        if metadata["protocol"] != "ik-soft-reference-v1" or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise ValueError("IK snapshot model mismatch")
        if cfg.control_config.action_scale != 0.5 or cfg.force_source != "analytic_pd" or cfg.reference_motion:
            raise ValueError("Require full-body .5rad residuals, corrected forces and no legacy reference")
        with np.load(path.with_name("ik_reference.npz"), allow_pickle=False) as source:
            self.anchor_qpos = source["qpos"].copy()
            self.anchor_qvel = source["qvel"].copy()
            self.anchor_ctrl = source["ctrl"].copy()
            self.table = source["offsets"].copy()
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        self.site_heights, self.heading = [], []
        for anchor in self.anchor_qpos:
            data.qpos[:] = anchor
            mujoco.mj_forward(model, data)
            self.site_heights.append([data.site(name).xpos[2] for name in SITES])
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, anchor[3:7])
            heading = rotation.reshape(3, 3)[:2, 2].copy()
            self.heading.append(heading / np.linalg.norm(heading))
        self.site_heights = np.array(self.site_heights)
        self.heading = np.array(self.heading)
        super().__init__(cfg, num_envs, backend_type)

    def reset_provider(self):
        return IKReset()

    @property
    def obs_groups_spec(self):
        return {"obs": 178}

    def references(self, info, future=0):
        return reference_at(self.table, info["anchor_id"][:, 0], info["elapsed"][:, 0] + future)

    def command(self, info):
        ramp = np.clip(info["elapsed"][:, 0] - 1, 0, 1)
        return SPEED * (3 * ramp ** 2 - 2 * ramp ** 3)

    def observations(self, info, indices=None):
        selection = slice(None) if indices is None else indices
        anchor_ids = info["anchor_id"][:, 0]
        phase = motion_time(info["elapsed"]) / PERIOD * 2 * np.pi
        velocity = self._backend.get_base_lin_vel()[selection]
        heading = self.heading[anchor_ids]
        local_velocity = np.column_stack((np.sum(velocity[:, :2] * heading, axis=1),
                                          -velocity[:, 0] * heading[:, 1] + velocity[:, 1] * heading[:, 0], velocity[:, 2]))
        contacts = np.column_stack([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(-1)[selection] > 0 for name in SITES])
        observation = np.concatenate((local_velocity, self._backend.get_sensor_data("gyro_torso")[selection] * 0.25,
                                      projected_gravity(self._backend.get_base_quat()[selection]), self.get_dof_pos()[selection],
                                      self.get_dof_vel()[selection] * 0.1, info["current_actions"], contacts,
                                      np.sin(phase), np.cos(phase), info["elapsed"] / 20,
                                      self.anchor_ctrl[anchor_ids], self.references(info), self.references(info, 0.1),
                                      self.references(info, 0.3), self.command(info)[:, None]), axis=1).astype(np.float32)
        if observation.shape != (len(anchor_ids), 178) or not np.isfinite(observation).all():
            raise ValueError("Invalid IK observation")
        return {"obs": observation}

    def targets(self, info, actions, advance=False):
        elapsed = info["elapsed"][:, 0] + (self.cfg.ctrl_dt if advance else 0)
        anchor_ids = info["anchor_id"][:, 0]
        return np.clip(self.anchor_ctrl[anchor_ids] + reference_at(self.table, anchor_ids, elapsed)
                       + 0.5 * np.clip(actions, -1, 1), self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def apply_action(self, actions, state):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 23) or not np.isfinite(actions).all():
            raise ValueError("Invalid IK residual")
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = np.clip(actions, -1, 1)
        state.info["elapsed"] += self.cfg.ctrl_dt
        return self.targets(state.info, state.info["current_actions"])

    def update_state(self, state):
        info = state.info
        anchor_ids = info["anchor_id"][:, 0]
        elapsed = info["elapsed"][:, 0]
        phase = motion_time(elapsed) / PERIOD % 1
        _, desired_height, swing = foot_path(phase)
        positions = np.stack([self._backend.get_sensor_data(f"crawl_pos_{name}") for name in SITES], axis=1)
        velocities = np.stack([self._backend.get_sensor_data(f"crawl_vel_{name}") for name in SITES], axis=1)
        contacts = np.column_stack([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(-1) > 0 for name in SITES])
        bad = np.column_stack([self._backend.get_sensor_data(name).reshape(-1) > 0 for name in self.bad_contacts]).any(axis=1)
        gravity_error = np.sum((projected_gravity(self._backend.get_base_quat()) - projected_gravity(self.anchor_qpos[anchor_ids, 3:7])) ** 2, axis=1)
        height = self._backend.get_base_pos()[:, 2]
        failed = (height < 0.16) | (height > 0.65) | (gravity_error > 0.8) | bad
        target_pose = self.anchor_qpos[anchor_ids, 7:] + self.references(info)
        pose = np.exp(-np.mean((self.get_dof_pos() - target_pose) ** 2, axis=1) / 0.3 ** 2)
        clearance = np.exp(-np.mean((positions[:, :, 2] - self.site_heights[anchor_ids] - desired_height) ** 2, axis=1) / 0.015 ** 2)
        matching = np.mean(contacts == ~(swing & (elapsed[:, None] > 1)), axis=1)
        linear = self._backend.get_base_lin_vel()[:, :2]
        desired = self.heading[anchor_ids] * self.command(info)[:, None]
        tracking = np.exp(-np.sum((linear - desired) ** 2, axis=1) / 0.03 ** 2)
        forces = position_servo_force(self.force_model, self.targets(info, info["current_actions"]), self.get_dof_pos(), self.get_dof_vel())
        ratios = np.abs(forces) / np.where(forces >= 0, self.force_limits[:, 1], -self.force_limits[:, 0])
        slip = np.mean(np.sum(velocities[:, :, :2] ** 2, axis=2) * contacts, axis=1)
        quality = np.exp(-gravity_error / 0.25 - (height - self.anchor_qpos[anchor_ids, 2]) ** 2 / 0.05 ** 2
                         -slip / 0.05 ** 2 - np.mean(np.maximum(ratios - 0.8, 0) ** 2, axis=1) / 0.1 ** 2)
        score = 0.1 * pose + 0.2 * clearance + 0.2 * matching + 0.5 * tracking
        reward = bounded_reward(quality, score, failed, self.cfg.ctrl_dt)
        if not np.isfinite(reward).all():
            raise ValueError("Nonfinite IK reward")
        info["log"] = {f"ik/{name}": float(np.mean(value)) for name, value in
                       (("pose", pose), ("clearance", clearance), ("contact", matching), ("velocity", tracking), ("quality", quality), ("failure", failed))}
        return state.replace(obs=self.observations(info), reward=reward, terminated=failed)
