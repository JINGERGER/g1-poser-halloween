import argparse
import json
from pathlib import Path

import numpy as np

from exploration_task import ExplorationCfg, ExplorationEnv
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest


def load_record(record):
    assert digest(record["path"]) == record["sha256"]
    payload = json.loads(Path(record["path"]).read_text())
    assert payload["status"] == "completed"
    for name, expected in payload["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and len(report["runs"]) == 4
    assert {(run["seed"], run["training_mode"]) for run in report["runs"]} == {(seed, mode) for seed in (1, 2) for mode in ("strict", "relaxed")}
    for run in report["runs"]:
        training = load_record(run["training_report"])
        assert training["planned_transitions"] == 12288 and len(training["updates"]) == 64
        for key in ("config", "checkpoint", "bundle", "runner"):
            assert digest(training[key]) == training[key + "_sha256"]
        for update in training["updates"]:
            assert update["updates_applied"] > 0
            assert all(np.isfinite(value) for value in update.values() if isinstance(value, (int, float)))
            assert not any(value for name, value in update.items() if name.startswith(("skipped_nonfinite", "rolled_back")))
        for mode, evidence in run["evaluations"].items():
            evaluation = load_record(evidence["report"])
            assert evaluation["training_report_sha256"] == run["training_report"]["sha256"]
            cfg_class, env_class = (ShortStepCfg, ShortStepEnv) if mode == "strict" else (ExplorationCfg, ExplorationEnv)
            for trial in evaluation["runs"]:
                assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
                env = env_class(cfg_class(bundle=evaluation["bundle"], reference_gain=trial["reference_gain"]))
                env._autoreset = False
                try:
                    env.init_state()
                    with np.load(trial["trajectory"], allow_pickle=False) as saved:
                        for index, action in enumerate(saved["actions"]):
                            state = env.step(action[None])
                            np.testing.assert_allclose(env.datas[0].qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
                            np.testing.assert_allclose(env.datas[0].qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
                            np.testing.assert_allclose(state.reward[0], saved["rewards"][index], atol=1e-10, rtol=0)
                            assert state.info["evaluation"][0] == trial["trace"][index]
                            assert bool(state.terminated[0]) == (index == len(saved["actions"]) - 1)
                            assert not state.truncated[0]
                finally:
                    env.close()
    print("PASS: four bounded training runs, finite PPO updates, provenance and native replay of all 32 strict/relaxed evaluation trajectories")


if __name__ == "__main__":
    main()
