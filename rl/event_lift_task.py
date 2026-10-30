from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from unilab.base import registry

from unload_objective_task import UnloadObjectiveCfg, UnloadObjectiveEnv


WEIGHTS = {"height": 3., "horizontal": 1., "unload": .5, "action_rate": -.02, "slip": -.1,
           "event": 1., "success": 10., "failure": -10.}


def target_position(initial_hand, radius, stage):
    target = np.array(initial_hand, dtype=float).copy()
    target[2] = radius + (.015 if stage < 3 else -.002)
    if stage >= 2:
        target[0] += .03
    return target


def dense_terms(gap, position, target, radius, stage, old_terms):
    return {"height": float(np.exp(-((gap - (target[2] - radius)) / .02) ** 2)),
            "horizontal": float(np.exp(-(np.linalg.norm(np.asarray(position)[:2] - target[:2]) / .03) ** 2)),
            "unload": old_terms["unload"] if stage < 3 else 0.,
            "action_rate": old_terms["action_rate"], "slip": old_terms["slip"]}


@registry.envcfg("G1EventLiftV1")
@dataclass
class EventLiftCfg(UnloadObjectiveCfg):
    reference_gain: float = 0.
    telemetry_path: str = ""


@registry.env("G1EventLiftV1", sim_backend="mujoco")
class EventLiftEnv(UnloadObjectiveEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if cfg.reference_gain != 0:
            raise ValueError("Event-driven course does not use the clock-driven IK action reference")
        if cfg.reward_config and cfg.reward_config.get("scales", {}) != WEIGHTS:
            raise ValueError("Event-lift reward configuration mismatch")
        super().__init__(replace(cfg, reward_config={}), num_envs, backend_type)
        self._cfg = cfg
        for key in ("event", "success", "failure"):
            assert self.protocol["reward"][key] == WEIGHTS[key]
        assert self.protocol["step_length_m"] == .03
        self.control_steps = 0
        self.completed = []
        self.active = [None] * num_envs

    def observations(self):
        observation = super().observations()
        for index, event in enumerate(self.events):
            observation["obs"][index, 114:137] = 0.
            target = target_position(self.initial_hand, self.model.geom_size[self.hand, 0], event.stage)
            observation["obs"][index, 137:140] = target - self.datas[index].geom_xpos[self.hand]
        return observation

    def step(self, actions):
        state = super().step(actions)
        self.control_steps += 1
        for index, row in enumerate(state.info["evaluation"]):
            radius = self.model.geom_size[self.hand, 0]
            target = target_position(self.initial_hand, radius, row["stage"])
            old_terms = row["reward_terms"]
            terms = dense_terms(row["gaps_m"][0], row["hand_position_m"], target, radius, row["stage"], old_terms)
            original = sum(self.objective_weights[key] * value for key, value in old_terms.items())
            replacement = sum(WEIGHTS[key] * value for key, value in terms.items())
            state.reward[index] += .02 * (replacement - original)
            row.update(reward_terms=terms, objective="event_lift_v1", target_position_m=target.tolist())
            if self.active[index] is None:
                self.active[index] = {"env_index": index, "start_control_step": self.control_steps, "elapsed_s": 0.,
                                      "max_stage": 0, "min_left_load_n": float("inf"), "max_left_gap_m": -float("inf"),
                                      "endpoint_unload_duration_s": 0., "return": 0.}
            episode = self.active[index]
            dt = row["elapsed_s"] - episode["elapsed_s"]
            assert 0 < dt <= .02 + 1e-9
            episode.update(elapsed_s=row["elapsed_s"], max_stage=max(episode["max_stage"], row["stage"]),
                           min_left_load_n=min(episode["min_left_load_n"], row["loads_n"][0]),
                           max_left_gap_m=max(episode["max_left_gap_m"], row["gaps_m"][0]))
            episode["return"] += float(state.reward[index])
            if row["loads_n"][0] <= 1:
                episode["endpoint_unload_duration_s"] += dt
            if state.terminated[index]:
                episode.update(end_control_step=self.control_steps, success=row["success"], failure=row["failure"], deadline=row["deadline"])
                self.completed.append(episode)
                self.active[index] = None
        return state

    def close(self):
        if self.cfg.telemetry_path:
            Path(self.cfg.telemetry_path).write_text(json.dumps({"control_steps": self.control_steps,
                "transitions": self.control_steps * self.num_envs, "completed_episodes": self.completed,
                "unfinished_episodes": [entry for entry in self.active if entry is not None],
                "limitations": "Control-endpoint load/gap/dwell summaries; stages use original physical-substep events."
            }, indent=2, allow_nan=False) + "\n")
        super().close()
