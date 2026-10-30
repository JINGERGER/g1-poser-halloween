from dataclasses import dataclass
import json
from pathlib import Path

from unilab.base import registry

from unload_objective_task import UnloadObjectiveCfg, UnloadObjectiveEnv


@registry.envcfg("G1UnloadCurveV1")
@dataclass
class UnloadCurveCfg(UnloadObjectiveCfg):
    telemetry_path: str = ""


@registry.env("G1UnloadCurveV1", sim_backend="mujoco")
class UnloadCurveEnv(UnloadObjectiveEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs, backend_type)
        self.control_steps = 0
        self.completed = []
        self.active = [None] * num_envs

    def step(self, actions):
        state = super().step(actions)
        self.control_steps += 1
        for index, outcome in enumerate(state.info["evaluation"]):
            if self.active[index] is None:
                self.active[index] = {"env_index": index, "start_control_step": self.control_steps,
                                      "elapsed_s": 0., "max_stage": 0, "min_left_load_n": float("inf"),
                                      "endpoint_unload_duration_s": 0., "return": 0.}
            episode = self.active[index]
            duration = outcome["elapsed_s"] - episode["elapsed_s"]
            assert 0 < duration <= self.cfg.ctrl_dt + 1e-9
            episode["elapsed_s"] = outcome["elapsed_s"]
            episode["max_stage"] = max(episode["max_stage"], outcome["stage"])
            episode["min_left_load_n"] = min(episode["min_left_load_n"], outcome["loads_n"][0])
            episode["return"] += float(state.reward[index])
            if outcome["loads_n"][0] <= self.protocol["unload_max_n"]:
                episode["endpoint_unload_duration_s"] += duration
            if state.terminated[index]:
                episode.update(end_control_step=self.control_steps, success=outcome["success"],
                               failure=outcome["failure"], deadline=outcome["deadline"])
                self.completed.append(episode)
                self.active[index] = None
        return state

    def close(self):
        if self.cfg.telemetry_path:
            Path(self.cfg.telemetry_path).write_text(json.dumps({
                "control_steps": self.control_steps, "transitions": self.control_steps * self.num_envs,
                "completed_episodes": self.completed, "unfinished_episodes": [entry for entry in self.active if entry is not None],
                "limitations": "Endpoint unloading duration estimate, not substep-confirmed dwell. Event stages are the original substep task events. Completed and unfinished episodes are separated."
            }, indent=2, allow_nan=False) + "\n")
        super().close()
