from dataclasses import dataclass
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from unilab.base import registry
from unilab.dr import DomainRandomizationProvider, ResetPlan

from crawl_task import CrawlCfg, CrawlEnv, SITES
from direct_task import projected_gravity
from prepare import SUPPORTS
from reference_assets import joint_mapping
from servo_force import position_servo_force, validate_position_servos


def world_to_body(quaternions, vectors):
    inverse = np.asarray(quaternions).copy()
    inverse[:, 1:] *= -1
    return body_to_world(inverse, vectors)


def body_to_world(quaternions, vectors):
    quaternions = np.asarray(quaternions)
    vectors = np.asarray(vectors)
    cross = 2 * np.cross(quaternions[:, 1:], vectors)
    return vectors + quaternions[:, :1] * cross + np.cross(quaternions[:, 1:], cross)


def euler_quaternion(angles):
    roll_cos, pitch_cos, yaw_cos = np.cos(angles / 2).T
    roll_sin, pitch_sin, yaw_sin = np.sin(angles / 2).T
    return np.column_stack((
        roll_cos * pitch_cos * yaw_cos + roll_sin * pitch_sin * yaw_sin,
        roll_sin * pitch_cos * yaw_cos - roll_cos * pitch_sin * yaw_sin,
        roll_cos * pitch_sin * yaw_cos + roll_sin * pitch_cos * yaw_sin,
        roll_cos * pitch_cos * yaw_sin - roll_sin * pitch_sin * yaw_cos,
    ))


@registry.envcfg("G1CrawlReference23")
@dataclass
class ReferenceCfg(CrawlCfg):
    speed_min: float = -1.0
    speed_max: float = 2.5
    yaw_rate_max: float = 1.0
    standing_fraction: float = 0.05
    command_interval_min: float = 4.0
    command_interval_max: float = 8.0
    actor_noise: bool = True
    reset_noise: bool = True


class ReferenceReset(DomainRandomizationProvider):
    def validate(self, env, capabilities):
        if env._num_action != 23:
            raise ValueError("Reference task requires 23 joints")

    def build_reset_plan(self, env, env_ids):
        count = len(env_ids)
        indices = env.rng.integers(len(env.reset_frames), size=count)
        qpos = env.reset_frames[indices].copy()
        qvel = np.zeros((count, 29))
        if env.cfg.reset_noise:
            qpos[:, :3] += env.rng.uniform(-0.05, 0.05, (count, 3))
            rotation = euler_quaternion(env.rng.uniform(-0.1, 0.1, (count, 3)))
            for index in range(count):
                quaternion = np.empty(4)
                mujoco.mju_mulQuat(quaternion, rotation[index], qpos[index, 3:7])
                qpos[index, 3:7] = quaternion
            qpos[:, 7:] *= env.rng.uniform(0.9, 1.1, (count, 23))
            qvel[:, :3] = env.rng.uniform(-0.2, 0.2, (count, 3))
            qvel[:, 3:6] = world_to_body(qpos[:, 3:7], env.rng.uniform(-0.3, 0.3, (count, 3)))
        unclipped = qpos[:, 7:].copy()
        qpos[:, 7:] = np.clip(qpos[:, 7:], env.joint_limits[:, 0], env.joint_limits[:, 1])
        projection = np.zeros(count)
        for index in range(count):
            env.reset_data.qpos[:] = qpos[index]
            mujoco.mj_forward(env.force_model, env.reset_data)
            minimum = min(mujoco.mj_geomDistance(env.force_model, env.reset_data, geom, env.floor_id, 1.0, None) for geom in env.floor_geoms)
            projection[index] = max(0.0, 0.002 - minimum)
            qpos[index, 2] += projection[index]
        commands, remaining = env.sample_commands(count)
        info = {
            "commands": commands, "command_remaining": remaining,
            "current_actions": np.zeros((count, 23), dtype=np.float32),
            "last_actions": np.zeros((count, 23), dtype=np.float32),
            "start_xy": qpos[:, :2].astype(np.float32),
            "reset_frame": indices[:, None], "reset_floor_projection": projection[:, None],
            "reset_joint_clip": np.max(np.abs(unclipped - qpos[:, 7:]), axis=1)[:, None],
        }
        return ResetPlan(env_ids=env_ids, qpos=qpos, qvel=qvel, info_updates=info)

    def build_reset_observation(self, env, env_ids, info_updates):
        return env.observations(info_updates, env_ids)


@registry.env("G1CrawlReference23", sim_backend="mujoco")
class ReferenceEnv(CrawlEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        from unilab.base.backend import create_backend
        from unilab.envs.locomotion.common.base import LocomotionBaseEnv

        if cfg.reference_motion or cfg.control_config.action_scale != 0.5:
            raise ValueError("Reference contract requires scale .5 and no IK reference")
        if not cfg.reward_config.scales:
            raise ValueError("Provide reference reward scales")
        if not (cfg.speed_min <= cfg.speed_max and cfg.yaw_rate_max >= 0 and 0 <= cfg.standing_fraction <= 1
                and 0 < cfg.command_interval_min <= cfg.command_interval_max):
            raise ValueError("Invalid command distribution")
        model_path = Path(cfg.scene.model_file)
        manifest = json.loads(model_path.with_name("manifest.json").read_text())
        metadata = manifest["reference"]
        if metadata["protocol"] != "reference-inspired-75-81-v1":
            raise ValueError("Unknown reference protocol")
        self.force_model = mujoco.MjModel.from_xml_path(str(model_path))
        validate_position_servos(self.force_model)
        names = [self.force_model.joint(int(joint)).name for joint in self.force_model.actuator_trnid[:, 0]]
        self.policy_to_actuator = joint_mapping(metadata["policy_order"], names)
        self.actuator_to_policy = np.argsort(self.policy_to_actuator)
        self.reset_frames = np.load(model_path.with_name("reset_frames.npy"), allow_pickle=False)
        if self.reset_frames.shape != (metadata["frame_count"], 30) or not np.isfinite(self.reset_frames).all():
            raise ValueError("Invalid reset frame snapshot")
        self.joint_limits = self.force_model.jnt_range[self.force_model.actuator_trnid[:, 0]]
        self.reset_data = mujoco.MjData(self.force_model)
        self.floor_id = self.force_model.geom("floor").id
        self.floor_geoms = [index for index in range(self.force_model.ngeom) if index != self.floor_id and
                            (self.force_model.geom_contype[index] or self.force_model.geom_conaffinity[index])]
        self.bad_contacts = manifest["direct_v2"]["bad_contacts"]
        self.bad_body_groups = {}
        support_bodies = {int(self.force_model.geom_bodyid[self.force_model.geom(name).id]) for name in SUPPORTS}
        for name in self.bad_contacts:
            body = int(self.force_model.geom_bodyid[self.force_model.geom(name.removeprefix("direct_contact_")).id])
            if body not in support_bodies:
                self.bad_body_groups.setdefault(body, []).append(name)
        self.rng = np.random.default_rng(cfg.seed)
        self.obs_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
        backend = create_backend(backend_type, cfg.scene, num_envs, cfg.sim_dt, base_name="pelvis", post_step_forward_sensor=True)
        LocomotionBaseEnv.__init__(self, cfg, backend, num_envs)
        self.default_angles = np.asarray(metadata["default_angles"], dtype=np.float32)
        self.ctrl_range = backend.get_actuator_ctrl_range()
        self._init_domain_randomization(ReferenceReset())

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(-5.0, 5.0, shape=(23,), dtype=np.float32)

    @property
    def obs_groups_spec(self):
        return {"obs": 75, "critic": 81}

    def sample_commands(self, count):
        commands = np.zeros((count, 3), dtype=np.float32)
        commands[:, 0] = self.rng.uniform(self.cfg.speed_min, self.cfg.speed_max, count)
        commands[:, 2] = self.rng.uniform(-self.cfg.yaw_rate_max, self.cfg.yaw_rate_max, count)
        commands[self.rng.random(count) < self.cfg.standing_fraction] = 0
        remaining = self.rng.uniform(self.cfg.command_interval_min, self.cfg.command_interval_max, (count, 1))
        return commands, remaining

    def observations(self, info, indices=None):
        selection = slice(None) if indices is None else indices
        quaternion = self._backend.get_base_quat()[selection]
        clean = np.concatenate((
            projected_gravity(quaternion), info["commands"],
            (self.get_dof_pos()[selection] - self.default_angles)[:, self.policy_to_actuator],
            self.get_dof_vel()[selection][:, self.policy_to_actuator], info["current_actions"],
        ), axis=1).astype(np.float32)
        critic = np.concatenate((world_to_body(quaternion, self._backend.get_base_lin_vel()[selection]),
                                 self._backend.get_base_ang_vel()[selection], clean), axis=1).astype(np.float32)
        actor = clean.copy()
        if self.cfg.actor_noise:
            for start, stop, amount in ((0, 3, 0.05), (6, 29, 0.01), (29, 52, 1.5)):
                actor[:, start:stop] += self.obs_rng.uniform(-amount, amount, actor[:, start:stop].shape)
        return {"obs": actor, "critic": critic}

    def action_targets(self, actions):
        return np.clip(self.default_angles + 0.5 * actions[:, self.actuator_to_policy], self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def apply_action(self, actions, state):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 23) or not np.isfinite(actions).all():
            raise ValueError("Invalid reference action")
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = np.clip(actions, -5, 5)
        return self.action_targets(state.info["current_actions"])

    def update_state(self, state):
        quaternion = self._backend.get_base_quat()
        velocity = world_to_body(quaternion, self._backend.get_base_lin_vel())
        angular_world = body_to_world(quaternion, self._backend.get_base_ang_vel())
        position = self.get_dof_pos()
        targets = self.action_targets(state.info["current_actions"])
        forces = position_servo_force(self.force_model, targets, position, self.get_dof_vel())
        raw_force = self.force_model.actuator_gainprm[:, 0] * targets + self.force_model.actuator_biasprm[:, 0]
        raw_force += self.force_model.actuator_biasprm[:, 1] * position + self.force_model.actuator_biasprm[:, 2] * self.get_dof_vel()
        contacts = np.column_stack([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(-1) > 0 for name in SITES])
        bad_contact = np.column_stack([np.any(np.column_stack([self._backend.get_sensor_data(name).reshape(-1) > 0 for name in names]), axis=1) for names in self.bad_body_groups.values()])
        support_vel = np.stack([self._backend.get_sensor_data(f"reference_support_vel_{index}") for index in range(4)], axis=1)
        gravity_error = np.sum((projected_gravity(quaternion) - [1, 0, 0]) ** 2, axis=1)
        height = self._backend.get_base_pos()[:, 2]
        terms = {
            "track_linear": np.exp(-np.sum((velocity[:, [2, 1]] - state.info["commands"][:, :2]) ** 2, axis=1) / 0.25 ** 2),
            "track_turn": np.exp(-(angular_world[:, 2] - state.info["commands"][:, 2]) ** 2 / 0.25 ** 2),
            "gravity": 1 - 0.5 * gravity_error,
            "height": (height - 0.22) ** 2,
            "joint_deviation": np.sum(np.abs(position - self.default_angles), axis=1),
            "joint_limits": np.sum(np.maximum(self.joint_limits[:, 0] - position, 0) + np.maximum(position - self.joint_limits[:, 1], 0), axis=1),
            "torque_limits": np.sum(np.abs(raw_force - forces), axis=1),
            "action_rate": np.sum((state.info["current_actions"] - state.info["last_actions"]) ** 2, axis=1),
            "torques": np.sum(forces ** 2, axis=1),
            "bad_contact": np.sum(bad_contact, axis=1),
            "slide": np.sum(np.linalg.norm(support_vel[:, :, :2], axis=2) * contacts, axis=1),
            "feet_air": ~np.any(contacts[:, 2:4], axis=1),
            "hands_air": ~np.any(contacts[:, :2], axis=1),
            "left_air": ~np.any(contacts[:, [0, 2]], axis=1),
            "right_air": ~np.any(contacts[:, [1, 3]], axis=1),
        }
        if set(self.cfg.reward_config.scales) != set(terms):
            raise ValueError("Reference reward configuration must specify exactly the declared terms")
        reward = sum(self.cfg.reward_config.scales[name] * value for name, value in terms.items()) * self.cfg.ctrl_dt
        if not np.isfinite(reward).all():
            raise ValueError("Nonfinite reference reward")
        terminated = (height < 0.08) | (height > 1.0) | (gravity_error > 3.5)
        state.info["log"] = {f"reference/{name}": float(np.mean(value)) for name, value in terms.items()}
        state.info["evaluation"] = {
            "velocity_yz": velocity[:, [2, 1]].copy(), "commands": state.info["commands"].copy(),
            "limb_contact": contacts.copy(), "bad_contact_count": terms["bad_contact"].copy(),
            "signed_gaps": np.column_stack([self._backend.get_sensor_data(f"balanced_gap_{index}").reshape(-1) for index in range(4)]),
        }
        state.info["command_remaining"] -= self.cfg.ctrl_dt
        due = np.flatnonzero(state.info["command_remaining"][:, 0] <= 0)
        if len(due):
            commands, remaining = self.sample_commands(len(due))
            state.info["commands"][due] = commands
            state.info["command_remaining"][due] = remaining
        return state.replace(obs=self.observations(state.info), reward=reward.astype(np.float32), terminated=terminated)
