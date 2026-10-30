import json
from pathlib import Path
import tempfile

import numpy as np

from short_step_assets import prepare
from unload_objective_task import UnloadObjectiveCfg, UnloadObjectiveEnv
from unload_curve_task import UnloadCurveCfg, UnloadCurveEnv


def main():
    bundle = str(prepare())
    with tempfile.TemporaryDirectory() as directory:
        telemetry = Path(directory) / "episodes.json"
        baseline = UnloadObjectiveEnv(UnloadObjectiveCfg(bundle=bundle), 2)
        measured = UnloadCurveEnv(UnloadCurveCfg(bundle=bundle, telemetry_path=str(telemetry)), 2)
        terminals = 0
        try:
            baseline.init_state()
            measured.init_state()
            random = np.random.default_rng(7)
            for index in range(220):
                actions = np.zeros((2, 23)) if index < 200 else random.normal(0, .15, (2, 23))
                original = baseline.step(actions)
                actual = measured.step(actions)
                for key in ("reward", "terminated", "truncated"):
                    np.testing.assert_array_equal(getattr(original, key), getattr(actual, key))
                np.testing.assert_array_equal(original.obs["obs"], actual.obs["obs"])
                assert original.info == actual.info
                if original.final_observation is not None:
                    np.testing.assert_array_equal(original.final_observation["obs"], actual.final_observation["obs"])
                terminals += int(np.sum(actual.terminated))
        finally:
            baseline.close()
            measured.close()
        report = json.loads(telemetry.read_text())
        assert report["transitions"] == 440 and len(report["completed_episodes"]) == terminals > 0
        episodes = report["completed_episodes"] + report["unfinished_episodes"]
        steps = sum((entry.get("end_control_step", 220) - entry["start_control_step"] + 1) for entry in episodes)
        assert steps == 440
        assert all(0 <= entry["endpoint_unload_duration_s"] <= entry["elapsed_s"] for entry in episodes)
    print("PASS: telemetry preserves physics/rewards/events/termination and accounts for all 440 transitions, including terminal and unfinished episodes")


if __name__ == "__main__":
    main()
