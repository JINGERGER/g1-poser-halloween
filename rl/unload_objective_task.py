from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from unilab.base import registry

from short_step_task import ShortStepCfg, ShortStepEnv


def objective_weights(original):
    return {**original, "alive": 0., "reference": 0., "unload": 2.}


@registry.envcfg("G1UnloadObjectiveV1")
@dataclass
class UnloadObjectiveCfg(ShortStepCfg):
    pass


@registry.env("G1UnloadObjectiveV1", sim_backend="mujoco")
class UnloadObjectiveEnv(ShortStepEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        original = json.loads(Path(cfg.bundle).read_text())["protocol"]["reward"]
        expected = objective_weights(original)
        if cfg.reward_config and cfg.reward_config.get("scales", {}) != expected:
            raise ValueError("Unload objective reward configuration mismatch")
        super().__init__(replace(cfg, reward_config={"scales": original}), num_envs, backend_type)
        self._cfg = cfg
        self.objective_weights = expected

    def step(self, actions):
        state = super().step(actions)
        for index, row in enumerate(state.info["evaluation"]):
            original_terms = row["reward_terms"]
            terms = {**original_terms, "unload": float(np.clip(1 - row["loads_n"][0] / 100, 0, 1))}
            old_dense = sum(self.protocol["reward"][name] * value for name, value in original_terms.items())
            new_dense = sum(self.objective_weights[name] * value for name, value in terms.items())
            state.reward[index] += .02 * (new_dense - old_dense)
            row["reward_terms"] = terms
            row["objective"] = "continuous_unload_v1"
        state.info["log"]["unload/mean_left_load_n"] = float(np.mean([row["loads_n"][0] for row in state.info["evaluation"]]))
        return state
