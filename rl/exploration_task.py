from dataclasses import dataclass

import mujoco
import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from component_unload_ramps import guard
from load_transfer import physical_metrics
from short_step_task import ShortStepCfg, ShortStepEnv


def training_guard(model, data, row, peak):
    if peak > .02:
        return "diagnostic_penetration_over_20mm"
    if row["other_total_n"] > 30:
        return "body_support_over_30N"
    if not .16 <= row["base_height_m"] <= .65:
        return "base_height"
    joints = model.actuator_trnid[:, 0]
    angles = data.qpos[model.jnt_qposadr[joints]]
    if np.any(angles < model.jnt_range[joints, 0] - .001) or np.any(angles > model.jnt_range[joints, 1] + .001):
        return "joint_limit"
    return None


@registry.envcfg("G1ExplorationShortStepV1")
@dataclass
class ExplorationCfg(ShortStepCfg):
    pass


@registry.env("G1ExplorationShortStepV1", sim_backend="mujoco")
class ExplorationEnv(ShortStepEnv):
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
            strict_violations = set()
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
                peak = max(intermediate_peak, row["penetration"]["depth_m"])
                strict_failure = guard(self.model, data, row, peak)
                if strict_failure:
                    strict_violations.add(strict_failure)
                failure = training_guard(self.model, data, row, peak)
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
                             "reward_terms": terms, "strict_violations": sorted(strict_violations)})
        observation = self.observations()
        final = {name: values.copy() for name, values in observation.items()} if np.any(terminated) else None
        info = {"evaluation": outcomes, "log": {"short_step/success": float(np.mean([row["success"] for row in outcomes])),
                                                "short_step/stage": float(np.mean([row["stage"] for row in outcomes])),
                                                "short_step/strict_violation": float(np.mean([bool(row["strict_violations"]) for row in outcomes]))}}
        self._state = NpEnvState(observation, rewards, terminated, np.zeros(self.num_envs, dtype=bool), info, final)
        if self._autoreset and np.any(terminated):
            self.reset(np.flatnonzero(terminated))
        return self._state
