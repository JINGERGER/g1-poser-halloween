from dataclasses import dataclass, field
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from unilab.base import registry
from unilab.base.base import ABEnv, EnvCfg
from unilab.base.np_env import NpEnvState

from component_unload_ramps import guard
from load_transfer import physical_metrics
from target_load_ablation import digest


@dataclass
class StepEvents:
    stage: int = 0
    air_time: float = 0.
    reload_time: float = 0.
    lift_x: float = 0.

    def update(self, row, position, initial, velocity, dt, protocol):
        before = self.stage
        if self.stage == 0 and row["loads_n"][0] <= protocol["unload_max_n"]:
            self.stage = 1
        airborne = row["gaps_m"][0] >= protocol["air_gap_m"] and row["loads_n"][0] <= protocol["unload_max_n"]
        self.air_time = self.air_time + dt if airborne else 0.
        if self.stage == 1 and self.air_time >= protocol["air_dwell_s"] - 1e-10:
            self.stage = 2
            self.lift_x = float(position[0])
        if self.stage == 2 and (row["gaps_m"][0] < .005 or row["loads_n"][0] > protocol["unload_max_n"]):
            self.lift_x = float(position[0])
        if (self.stage == 2 and position[0] - initial[0] >= protocol["advance_min_m"]
                and position[0] - self.lift_x >= protocol["advance_min_m"] and row["gaps_m"][0] >= .005
                and row["loads_n"][0] <= protocol["unload_max_n"]):
            self.stage = 3
        target_xy = initial[:2] + [protocol["step_length_m"], 0]
        landed = (self.stage == 3 and row["loads_n"][0] > protocol["reload_min_n"] and all(row["floor_contacts"])
                  and row["gaps_m"][0] <= protocol["landing_gap_max_m"] and np.linalg.norm(position[:2] - target_xy) <= protocol["landing_xy_tolerance_m"]
                  and velocity <= protocol["settled_velocity_max"] and row["loaded_slip_mps"] is not None
                  and row["loaded_slip_mps"] <= protocol["settled_slip_max_mps"])
        self.reload_time = self.reload_time + dt if landed else 0.
        if self.stage == 3 and self.reload_time >= protocol["reload_dwell_s"] - 1e-10:
            self.stage = 4
        return self.stage - before


@registry.envcfg("G1ComponentShortStepV1")
@dataclass
class ShortStepCfg(EnvCfg):
    bundle: str = ""
    seed: int = 1
    sim_dt: float = .002
    ctrl_dt: float = .02
    max_episode_seconds: float = 4.
    reference_gain: float = 1.
    reward_config: dict = field(default_factory=dict)


@registry.env("G1ComponentShortStepV1", sim_backend="mujoco")
class ShortStepEnv(ABEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if backend_type != "mujoco" or not 1 <= num_envs <= 32:
            raise ValueError("Native MuJoCo diagnostic task supports 1..32 environments")
        self._cfg, self._num_envs, self._state = cfg, num_envs, None
        self.bundle = json.loads(Path(cfg.bundle).read_text())
        self.protocol = self.bundle["protocol"]
        if cfg.ctrl_dt != .02 or cfg.sim_dt != .002 or cfg.max_episode_seconds != self.protocol["duration_s"] or not 0 <= cfg.reference_gain <= 1:
            raise ValueError("Frozen short-step timing/reference contract violated")
        if cfg.reward_config and cfg.reward_config.get("scales", {}) != self.protocol["reward"]:
            raise ValueError("Reward configuration must match frozen protocol")
        if digest(self.bundle["model"]) != self.bundle["model_sha256"]:
            raise ValueError("Model changed")
        for path, expected in self.bundle["mesh_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Mesh changed")
        self.model = mujoco.MjModel.from_xml_path(self.bundle["model"])
        if self.model.opt.timestep != cfg.sim_dt:
            raise ValueError("Model timestep differs from task")
        self.datas = [mujoco.MjData(self.model) for unused in range(num_envs)]
        self.initial_state = np.array(self.bundle["initial_physics_state"])
        self.initial_qpos = np.array(self.bundle["initial_qpos"])
        self.initial_ctrl = np.array(self.bundle["initial_ctrl"])
        self.offsets = np.array(self.bundle["joint_offsets"])
        self.targets = np.array(self.bundle["hand_targets"])
        self.hand = self.model.geom("left_hand_collision").id
        self.initial_hand = np.array(self.bundle["initial_support_positions"])[0]
        self.elapsed = np.zeros(num_envs)
        self.actions = np.zeros((num_envs, 23))
        self.events = [StepEvents() for unused in range(num_envs)]
        self.metrics = [None] * num_envs
        self._autoreset = True

    @property
    def cfg(self):
        return self._cfg

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def state(self):
        return self._state

    @property
    def obs_groups_spec(self):
        return {"obs": 148}

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (148,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1., 1., (23,), dtype=np.float32)

    def observations(self):
        rows = []
        for index, data in enumerate(self.datas):
            frame = min(round(self.elapsed[index] / .02), len(self.offsets) - 1)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            metric, event = self.metrics[index], self.events[index]
            rows.append(np.r_[data.qpos[7:] - self.initial_qpos[7:], data.qvel * .1,
                              rotation.reshape(3, 3).T @ [0., 0, -1], data.qpos[2], self.elapsed[index] / 4,
                              data.ctrl - self.initial_ctrl, self.actions[index], np.array(metric["loads_n"]) / 100,
                              np.array(metric["gaps_m"]) * 20, data.geom_xpos[self.hand] - self.initial_hand,
                              self.offsets[frame], self.targets[frame] - data.geom_xpos[self.hand],
                              np.eye(5)[event.stage], event.air_time, event.reload_time,
                              event.lift_x - self.initial_hand[0] if event.stage >= 2 else 0.])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != (self.num_envs, 148) or not np.isfinite(result).all():
            raise ValueError("Invalid short-step observations")
        return {"obs": result}

    def reset(self, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=int)
        for index in ids:
            data = self.datas[index]
            mujoco.mj_resetData(self.model, data)
            mujoco.mj_setState(self.model, data, self.initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = self.initial_ctrl
            mujoco.mj_forward(self.model, data)
            self.elapsed[index], self.actions[index], self.events[index] = 0., 0., StepEvents()
            self.metrics[index] = physical_metrics(self.model, data)
            assert guard(self.model, data, self.metrics[index], self.metrics[index]["penetration"]["depth_m"]) is None
        if all(metric is not None for metric in self.metrics):
            observation = self.observations()
        else:
            raise ValueError("Initialize all environments before partial resets")
        if self._state is not None:
            self._state.obs["obs"][ids] = observation["obs"][ids]
        return {"obs": observation["obs"][ids].copy()}, {}

    def init_state(self):
        observation, unused = self.reset()
        self._state = NpEnvState(observation, np.zeros(self.num_envs, dtype=np.float32),
                                 np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), {})
        return self._state

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, 23) or not np.isfinite(actions).all():
            raise ValueError("Invalid residual action")
        if self._state is None:
            self.init_state()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        outcomes = []
        for index, data in enumerate(self.datas):
            if not self._autoreset and self._state.terminated[index]:
                raise RuntimeError("Reset terminated evaluation environment before stepping")
            action = np.clip(actions[index], -1, 1)
            previous = self.actions[index].copy()
            self.actions[index] = action
            frame = min(round(self.elapsed[index] / .02) + 1, len(self.offsets) - 1)
            desired = self.initial_ctrl + self.cfg.reference_gain * self.offsets[frame] + self.protocol["residual_scale_rad"] * action
            desired = np.clip(desired, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            data.ctrl[:] += np.clip(desired - data.ctrl, -self.protocol["max_ctrl_delta_rad"], self.protocol["max_ctrl_delta_rad"])
            failure, event_count = None, 0
            for unused in range(self.cfg.sim_substeps):
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                mujoco.mj_step(self.model, data)
                intermediate_peak = max((-contact.dist for contact in data.contact), default=0)
                mujoco.mj_forward(self.model, data)
                self.elapsed[index] += self.cfg.sim_dt
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("Nonfinite physics; do not silently reset")
                row = physical_metrics(self.model, data)
                self.metrics[index] = row
                failure = guard(self.model, data, row, max(intermediate_peak, row["penetration"]["depth_m"]))
                floor = self.model.geom("floor").id
                if any(floor not in (contact.geom1, contact.geom2) and contact.dist < -.001 for contact in data.contact):
                    failure = "self_collision"
                if failure:
                    break
                event_count += self.events[index].update(row, data.geom_xpos[self.hand], self.initial_hand,
                                                        float(max(abs(data.qvel))), self.cfg.sim_dt, self.protocol)
                if self.events[index].stage == 4:
                    break
            weights = self.protocol["reward"]
            distance = np.linalg.norm(data.geom_xpos[self.hand] - self.targets[frame])
            terms = {"alive": 1., "reference": float(np.exp(-(distance / .03) ** 2)),
                     "unload": float(np.clip(1 - row["loads_n"][0] / 100, 0, 1)) if .4 <= self.elapsed[index] < 2.6 else 0.,
                     "action_rate": float(np.mean((action - previous) ** 2)),
                     "slip": float(min((row["loaded_slip_mps"] or 0) / .02, 5))}
            reward = .02 * sum(weights[name] * value for name, value in terms.items()) + weights["event"] * event_count
            success = self.events[index].stage == 4 and failure is None
            if failure:
                reward += weights["failure"]
            elif success:
                reward += weights["success"]
            deadline = self.elapsed[index] >= self.protocol["duration_s"] - 1e-9
            terminated[index] = bool(failure or success or deadline)
            rewards[index] = reward
            outcomes.append({"success": success, "stage": self.events[index].stage, "failure": failure,
                             "deadline": bool(deadline), "elapsed_s": float(self.elapsed[index]), "loads_n": row["loads_n"],
                             "gaps_m": row["gaps_m"], "hand_position_m": data.geom_xpos[self.hand].tolist(),
                             "penetration_m": row["penetration"]["depth_m"], "other_loads_n": row["other_loads_n"],
                             "reward_terms": terms})
        observation = self.observations()
        final = {name: values.copy() for name, values in observation.items()} if np.any(terminated) else None
        info = {"evaluation": outcomes, "log": {"short_step/success": float(np.mean([row["success"] for row in outcomes])),
                                                "short_step/stage": float(np.mean([row["stage"] for row in outcomes]))}}
        self._state = NpEnvState(observation, rewards, terminated, np.zeros(self.num_envs, dtype=bool), info, final)
        if self._autoreset and np.any(terminated):
            self.reset(np.flatnonzero(terminated))
        return self._state

    def close(self):
        self.datas.clear()
