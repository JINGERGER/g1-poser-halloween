from dataclasses import dataclass, field
from pathlib import Path
import json

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationProvider, ResetPlan
from unilab.envs.locomotion.common.base import LocomotionBaseCfg, LocomotionBaseEnv


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "output" / "rl" / "assets"
SITES = ("left_palm", "right_palm", "left_foot", "right_foot")


@dataclass
class CrawlReward:
    scales: dict[str, float] = field(default_factory=dict)


@registry.envcfg("G1CrawlFlat23")
@dataclass
class CrawlCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=str(ASSETS / "flat.xml")))
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 20.0
    post_step_forward_sensor: bool = True
    reward_config: CrawlReward = field(default_factory=CrawlReward)
    speed_min: float = 0.0
    speed_max: float = 0.0
    phase_period: float = 2.0
    lift_height: float = 0.08
    reset_joint_noise: float = 0.01
    seed: int = 1
    reference_motion: bool = False
    reference_scale: float = 1.0


class CrawlReset(DomainRandomizationProvider):
    def validate(self, env, capabilities):
        if env._num_action != 23:
            raise ValueError("This task requires the poser 23-actuator G1")

    def build_reset_plan(self, env, env_ids):
        count = len(env_ids)
        qpos = np.tile(env._init_qpos, (count, 1))
        qpos[:, 7:] += env.rng.uniform(-env.cfg.reset_joint_noise, env.cfg.reset_joint_noise, (count, 23))
        qpos[:, 7:] = np.clip(qpos[:, 7:], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
        info = {
            "commands": env.rng.uniform(env.cfg.speed_min, env.cfg.speed_max, (count, 1)).astype(np.float32),
            "phase": env.rng.uniform(0, 1, (count, 1)).astype(np.float32),
            "current_actions": np.zeros((count, 23), dtype=np.float32),
            "last_actions": np.zeros((count, 23), dtype=np.float32),
            "start_xy": qpos[:, :2].astype(np.float32),
        }
        qpos[:, 7:] += env.reference_offsets(info)
        qpos[:, 7:] = np.clip(qpos[:, 7:], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
        return ResetPlan(env_ids=env_ids, qpos=qpos, qvel=np.zeros((count, len(env._init_qvel))), info_updates=info)

    def build_reset_observation(self, env, env_ids, info_updates):
        return env.observations(info_updates, env_ids)


@registry.env("G1CrawlFlat23", sim_backend="mujoco")
class CrawlEnv(LocomotionBaseEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if not cfg.reward_config.scales:
            raise ValueError("Provide reward_config.scales through the task configuration")
        if cfg.speed_min < 0 or cfg.speed_max < cfg.speed_min:
            raise ValueError("This first task supports ordered, nonnegative forward-speed bounds")
        if cfg.reference_motion and (cfg.phase_period != 2.0 or cfg.lift_height != 0.08):
            raise ValueError("Regenerate matching IK references before changing cycle period or lift height")
        manifest = json.loads(Path(cfg.scene.model_file).with_name("manifest.json").read_text())
        self.reference_up = np.array(manifest["reference_up"], dtype=np.float32)
        self.reference_height = float(manifest["qpos"][2])
        self.reference_site_heights = np.array([manifest["site_heights"][name] for name in SITES], dtype=np.float32)
        self.reference = None
        if cfg.reference_motion:
            with np.load(Path(cfg.scene.model_file).with_name("reference.npz")) as reference:
                self.reference = reference["offsets"].copy()
                self.reference_speeds = reference["speeds"].copy()
        self.rng = np.random.default_rng(cfg.seed)
        backend = create_backend(backend_type, cfg.scene, num_envs, cfg.sim_dt, base_name="pelvis", post_step_forward_sensor=True)
        super().__init__(cfg, backend, num_envs)
        self.ctrl_range = backend.get_actuator_ctrl_range()
        self._init_domain_randomization(self.reset_provider())

    def reset_provider(self):
        return CrawlReset()

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=(23,), dtype=np.float32)

    @property
    def obs_groups_spec(self):
        return {"obs": 85}

    def observations(self, info, indices=None):
        selection = slice(None) if indices is None else indices
        contacts = np.concatenate([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(self.num_envs, 1) for name in SITES], axis=1)[selection]
        phase = info["phase"] * (2 * np.pi)
        observation = np.concatenate([
            self._backend.get_base_lin_vel()[selection],
            self._backend.get_sensor_data("gyro_torso")[selection] * 0.25,
            self._backend.get_sensor_data("upvector_torso")[selection],
            self.get_dof_pos()[selection] - self.default_angles,
            self.get_dof_vel()[selection] * 0.05,
            info["current_actions"], info["commands"],
            np.sin(phase), np.cos(phase), contacts,
        ], axis=1).astype(np.float32)
        if observation.shape[1] != 85 or not np.isfinite(observation).all():
            raise ValueError("Invalid crawl observation")
        return {"obs": observation}

    def reference_offsets(self, info):
        if self.reference is None:
            return np.zeros_like(info["current_actions"])
        frame = (info["phase"][:, 0] % 1.0) * self.reference.shape[1]
        lower_frame = np.floor(frame).astype(int)
        alpha = (frame - lower_frame)[:, None]
        speed = np.clip(info["commands"][:, 0], self.reference_speeds[0], self.reference_speeds[-1])
        lower_speed = np.clip(np.searchsorted(self.reference_speeds, speed) - 1, 0, len(self.reference_speeds) - 2)
        beta = ((speed - self.reference_speeds[lower_speed]) / (self.reference_speeds[lower_speed + 1] - self.reference_speeds[lower_speed]))[:, None]
        upper_frame = (lower_frame + 1) % self.reference.shape[1]
        low = (1 - alpha) * self.reference[lower_speed, lower_frame] + alpha * self.reference[lower_speed, upper_frame]
        high = (1 - alpha) * self.reference[lower_speed + 1, lower_frame] + alpha * self.reference[lower_speed + 1, upper_frame]
        return ((1 - beta) * low + beta * high) * self.cfg.reference_scale

    def orientation_error(self):
        up = self._backend.get_sensor_data("upvector_torso")
        return np.sum(np.square(up - self.reference_up), axis=1)

    def additional_terms(self, state):
        return {}, np.zeros(self.num_envs, dtype=bool)

    def gait_active(self, info):
        return info["commands"][:, 0] > 0.01

    def limb_phase(self, info):
        return (info["phase"] + np.array([0.0, 0.5, 0.75, 0.25])) % 1.0

    def apply_action(self, actions, state):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 23) or not np.isfinite(actions).all():
            raise ValueError("Invalid crawl action")
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = np.clip(actions, -1, 1)
        state.info["phase"] = (state.info["phase"] + self.cfg.ctrl_dt / self.cfg.phase_period) % 1.0
        targets = self.default_angles + self.reference_offsets(state.info) + self.cfg.control_config.action_scale * state.info["current_actions"]
        return np.clip(targets, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def update_state(self, state):
        velocity = self._backend.get_base_lin_vel()
        height = self._backend.get_base_pos()[:, 2]
        gyro = self._backend.get_sensor_data("gyro_torso")
        contacts = np.concatenate([self._backend.get_sensor_data(f"crawl_contact_{name}").reshape(self.num_envs, 1) for name in SITES], axis=1) > 0
        positions = np.stack([self._backend.get_sensor_data(f"crawl_pos_{name}") for name in SITES], axis=1)
        site_velocities = np.stack([self._backend.get_sensor_data(f"crawl_vel_{name}") for name in SITES], axis=1)
        torso_contact = self._backend.get_sensor_data("crawl_torso_contact").reshape(-1) > 0
        moving = self.gait_active(state.info)
        phase = self.limb_phase(state.info)
        swing = (phase >= 0.75) & moving[:, None]
        lift = self.cfg.lift_height * np.sin(np.pi * np.clip((phase - 0.75) / 0.25, 0, 1))
        lift *= moving[:, None]
        target_z = lift + self.reference_site_heights
        orientation_error = self.orientation_error()
        terms = {
            "alive": np.ones(self.num_envs),
            "velocity": np.exp(-((velocity[:, 0] - state.info["commands"][:, 0]) ** 2 + velocity[:, 1] ** 2) / 0.01),
            "orientation": np.exp(-orientation_error / 0.15),
            "height": np.exp(-np.square(height - self.reference_height) / 0.004),
            "support": np.mean(contacts == ~swing, axis=1),
            "clearance": np.mean(np.exp(-np.square(positions[:, :, 2] - target_z) / 0.002), axis=1),
            "slip": np.sum(np.sum(site_velocities[:, :, :2] ** 2, axis=2) * contacts, axis=1),
            "action_rate": np.mean((state.info["current_actions"] - state.info["last_actions"]) ** 2, axis=1),
            "joint_velocity": np.mean(self.get_dof_vel() ** 2, axis=1),
            "pose": np.mean((self.get_dof_pos() - self.default_angles) ** 2, axis=1),
            "angular_velocity": np.sum(gyro ** 2, axis=1),
            "torso_contact": torso_contact.astype(float),
        }
        extra_terms, extra_terminated = self.additional_terms(state)
        terms.update(extra_terms)
        unknown = set(self.cfg.reward_config.scales) - terms.keys()
        if unknown:
            raise ValueError(f"Unknown reward terms: {unknown}")
        reward = sum(self.cfg.reward_config.scales[name] * value for name, value in terms.items() if name in self.cfg.reward_config.scales) * self.cfg.ctrl_dt
        if not np.isfinite(reward).all():
            raise ValueError("Nonfinite crawl reward")
        terminated = (height < 0.16) | (height > 0.65) | (orientation_error > 0.8) | torso_contact | extra_terminated
        state.info["log"] = {f"crawl/{name}": float(np.mean(value)) for name, value in terms.items()}
        peak_swing = (phase > 0.825) & (phase < 0.925) & moving[:, None]
        state.info["evaluation"] = {
            "limb_contact": contacts.astype(float),
            "speed_error": np.abs(velocity[:, 0] - state.info["commands"][:, 0]).copy(),
            "progress": (self._backend.get_base_pos()[:, 0] - state.info["start_xy"][:, 0]).copy(),
            "horizontal_drift": np.linalg.norm(self._backend.get_base_pos()[:, :2] - state.info["start_xy"], axis=1),
            "support_fraction": contacts.mean(axis=1),
            "slip": terms["slip"].copy(),
            "clearance_error": np.mean(np.abs(positions[:, :, 2] - target_z), axis=1),
            "peak_swing_contact": np.sum(contacts & peak_swing, axis=1),
            "peak_swing_count": peak_swing.sum(axis=1),
            "peak_swing_clearance": np.sum((positions[:, :, 2] - self.reference_site_heights) * peak_swing, axis=1),
            "limb_peak_count": peak_swing.astype(float),
            "limb_peak_contact": (contacts & peak_swing).astype(float),
            "limb_peak_site_rise": (positions[:, :, 2] - self.reference_site_heights) * peak_swing,
        }
        return state.replace(obs=self.observations(state.info), reward=np.asarray(reward, dtype=np.float32), terminated=terminated)
