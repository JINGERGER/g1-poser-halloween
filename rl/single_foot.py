from dataclasses import dataclass

import numpy as np
from unilab.base import registry

from direct_task import BalancedCfg, BalancedEnv


def single_foot_phase(phase, foot):
    result = np.full((len(phase), 4), 0.4, dtype=np.float32)
    result[:, foot] = phase[:, 0] % 1.0
    return result


@registry.envcfg("G1SingleFoot23V4")
@dataclass
class SingleFootCfg(BalancedCfg):
    target_foot: int = 2


@registry.env("G1SingleFoot23V4", sim_backend="mujoco")
class SingleFootEnv(BalancedEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if cfg.target_foot not in (2, 3):
            raise ValueError("Single-foot curriculum requires left foot (2) or right foot (3)")
        if cfg.speed_min != 0 or cfg.speed_max != 0:
            raise ValueError("Single-foot curriculum has no forward velocity command")
        super().__init__(cfg, num_envs, backend_type)

    def gait_active(self, info):
        return np.ones(len(info["phase"]), dtype=bool)

    def limb_phase(self, info):
        return single_foot_phase(info["phase"], self.cfg.target_foot)
