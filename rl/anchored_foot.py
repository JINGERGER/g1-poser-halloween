from dataclasses import dataclass

import mujoco
import numpy as np
from unilab.base import registry

from crawl_task import CrawlReset, SITES
from single_foot import SingleFootCfg, SingleFootEnv
from servo_force import position_servo_force, validate_position_servos


class AnchorReset(CrawlReset):
    def build_reset_observation(self, env, env_ids, info_updates):
        info_updates["support_anchors"] = env.site_xy()[env_ids].copy()
        return env.observations(info_updates, env_ids)


@registry.envcfg("G1AnchoredFoot23V5")
@dataclass
class AnchoredCfg(SingleFootCfg):
    pass


@registry.env("G1AnchoredFoot23V5", sim_backend="mujoco")
class AnchoredEnv(SingleFootEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self.force_model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
        validate_position_servos(self.force_model)
        self.stance = [index for index in range(4) if index != cfg.target_foot]
        super().__init__(cfg, num_envs, backend_type)

    def reset_provider(self):
        return AnchorReset()

    @property
    def obs_groups_spec(self):
        return {"obs": 93}

    def site_xy(self):
        return np.stack([self._backend.get_sensor_data(f"crawl_pos_{name}")[:, :2] for name in SITES], axis=1)

    def observations(self, info, indices=None):
        result = super().observations(info, indices)
        selection = slice(None) if indices is None else indices
        base_error = self._backend.get_base_pos()[selection, :2] - info["start_xy"]
        anchor_error = self.site_xy()[selection][:, self.stance] - info["support_anchors"][:, self.stance]
        extra = np.concatenate((base_error, anchor_error.reshape(len(base_error), 6)), axis=1)
        result["obs"] = np.concatenate((result["obs"], np.clip(extra / 0.1, -5, 5)), axis=1).astype(np.float32)
        return result

    def joint_forces(self, state):
        targets = np.clip(self.default_angles + self.cfg.control_config.action_scale * state.info["current_actions"], self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        return position_servo_force(self.force_model, targets, self.get_dof_pos(), self.get_dof_vel())

    def additional_terms(self, state):
        terms, bad = super().additional_terms(state)
        error = self.site_xy()[:, self.stance] - state.info["support_anchors"][:, self.stance]
        drift = np.linalg.norm(self._backend.get_base_pos()[:, :2] - state.info["start_xy"], axis=1)
        terms["anchor_drift"] = np.minimum(np.mean(np.sum(error ** 2, axis=2), axis=1) / 0.03 ** 2, 25)
        terms["base_drift"] = np.minimum((np.maximum(drift - 0.08, 0) / 0.05) ** 2, 25)
        return terms, bad

    def update_state(self, state):
        result = super().update_state(state)
        error = self.site_xy()[:, self.stance] - state.info["support_anchors"][:, self.stance]
        result.info["evaluation"]["stance_anchor_drift"] = np.max(np.linalg.norm(error, axis=2), axis=1)
        return result
