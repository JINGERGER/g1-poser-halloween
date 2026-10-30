import argparse
import json
from pathlib import Path

import numpy as np

from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    assert digest(report["bundle"]) == report["bundle_sha256"]
    assert digest(report["recovery_report"]) == report["recovery_report_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    if report["training_report"]:
        assert digest(report["training_report"]) == report["training_report_sha256"]
        training = json.loads(Path(report["training_report"]).read_text())
        assert training["status"] == "completed"
        for key in ("bundle", "config", "checkpoint", "runner"):
            assert digest(training[key]) == training[key + "_sha256"]
        for name, expected in training["source_sha256"].items():
            assert digest(Path(__file__).with_name(name)) == expected
        expected_updates = 4 if training["stage"] == "smoke" else 64
        assert len(training["updates"]) == expected_updates
        assert [entry["iteration"] for entry in training["updates"]] == list(range(expected_updates))
        for update in training["updates"]:
            assert update["updates_applied"] > 0
            for name, value in update.items():
                if isinstance(value, (float, int)):
                    assert np.isfinite(value)
                if name.startswith(("skipped_nonfinite", "rolled_back")):
                    assert value == 0
    labels = [run["label"] for run in report["runs"]]
    assert labels[:3] == ["hold", "ik_zero_residual", "known_unload_return"]
    for run in report["runs"]:
        assert digest(run["trajectory"]) == run["trajectory_sha256"]
        env = ShortStepEnv(ShortStepCfg(bundle=report["bundle"], reference_gain=run["reference_gain"]))
        env._autoreset = False
        env.init_state()
        try:
            total = 0.
            with np.load(run["trajectory"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(env.datas[0].qpos, saved["qpos"][0])
                np.testing.assert_array_equal(env.datas[0].qvel, saved["qvel"][0])
                for index, action in enumerate(saved["actions"]):
                    state = env.step(action[None])
                    np.testing.assert_allclose(env.datas[0].qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(env.datas[0].qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(state.reward[0], saved["rewards"][index], atol=1e-10, rtol=0)
                    assert state.info["evaluation"][0] == run["trace"][index]
                    assert not state.truncated[0]
                    if state.terminated[0]:
                        assert index == len(saved["actions"]) - 1
                    total += float(state.reward[0])
                assert state.terminated[0]
                last = state.info["evaluation"][0]
                assert last["success"] == run["success"] == (last["stage"] == 4 and last["failure"] is None)
                assert last["failure"] == run["failure"] and last["deadline"] == run["deadline"]
                np.testing.assert_allclose(total, run["return"], atol=1e-10, rtol=0)
        finally:
            env.close()
    assert not next(run for run in report["runs"] if run["label"] == "hold")["success"]
    assert not next(run for run in report["runs"] if run["label"] == "known_unload_return")["success"]
    print("PASS: checkpoint/config/source provenance, finite PPO updates, baseline non-success and native replay of every evaluated action/event/termination")


if __name__ == "__main__":
    main()
