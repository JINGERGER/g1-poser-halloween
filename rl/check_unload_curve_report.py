import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from check_coordinated_exploration import verify_replay
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
    assert report["status"] == "completed" and report["planned_transitions"] == 98304
    assert [run["seed"] for run in report["runs"]] == [1, 2]
    statistics = []
    trajectory_count = 0
    for run in report["runs"]:
        training = load_record(run["training"])
        assert training["planned_transitions"] == 49152 and len(training["updates"]) == 256
        assert [row["iteration"] for row in training["updates"]] == list(range(256))
        for key in ("bundle", "config", "runner", "telemetry"):
            assert digest(training[key]) == training[key + "_sha256"]
        for update in training["updates"]:
            assert update["updates_applied"] > 0
            assert all(np.isfinite(value) for value in update.values() if isinstance(value, (int, float)))
            assert not any(value for name, value in update.items() if name.startswith(("skipped_nonfinite", "rolled_back")))
        assert [entry["updates"] for entry in run["evaluations"]] == [64, 128, 256]
        for snapshot, checkpoint_eval in zip(training["snapshots"], run["evaluations"]):
            saved = load_record(snapshot)
            updates = checkpoint_eval["updates"]
            assert saved["completed_updates"] == updates and saved["observed_transitions"] == updates * 192
            assert saved["updates"] == training["updates"][:updates]
            assert digest(saved["checkpoint"]) == saved["checkpoint_sha256"]
            evaluation = load_record(checkpoint_eval["report"])
            assert evaluation["training_report_sha256"] == snapshot["sha256"]
            assert evaluation["bundle_sha256"] == training["bundle_sha256"]
            for trial in evaluation["runs"]:
                assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
                assert verify_replay(Path(evaluation["bundle"]), trial)
                trajectory_count += 1
                last = trial["trace"][-1]
                assert trial["success"] == last["success"] == (last["stage"] == 4 and last["failure"] is None)
                assert trial["failure"] == last["failure"]
            policy = next(trial for trial in evaluation["runs"] if trial["label"] == "ppo_deterministic")
            for key in ("success", "max_stage", "failure", "elapsed_s", "return"):
                assert checkpoint_eval["summary"][key] == policy[key]
        telemetry = json.loads(Path(training["telemetry"]).read_text())
        assert telemetry["transitions"] == 49152 and telemetry["control_steps"] == 256 * 24
        all_episodes = telemetry["completed_episodes"] + telemetry["unfinished_episodes"]
        for env_index in range(8):
            episodes = sorted((episode for episode in all_episodes if episode["env_index"] == env_index), key=lambda episode: episode["start_control_step"])
            cursor = 1
            for episode in episodes:
                assert episode["start_control_step"] == cursor
                cursor = episode.get("end_control_step", 6144) + 1
                assert 0 <= episode["endpoint_unload_duration_s"] <= episode["elapsed_s"]
            assert cursor == 6145
        for start, end in ((0, 64), (64, 128), (128, 256)):
            episodes = [episode for episode in telemetry["completed_episodes"] if start * 24 < episode["end_control_step"] <= end * 24]
            statistics.append({"seed": run["seed"], "update_window": [start, end], "completed_episodes": len(episodes),
                               "failure_counts": dict(Counter(episode["failure"] or ("success" if episode["success"] else "deadline") for episode in episodes)),
                               "stage_histogram": dict(Counter(episode["max_stage"] for episode in episodes)),
                               "median_duration_s": float(np.median([episode["elapsed_s"] for episode in episodes])) if episodes else None,
                               "endpoint_unload_duration_s": sum(episode["endpoint_unload_duration_s"] for episode in episodes)})
    summary_path = args.report.with_name("verified_training_statistics.json")
    summary_path.write_text(json.dumps({"source_report": str(args.report.resolve()), "source_sha256": digest(args.report),
                                       "windows": statistics, "limitations": "Completed episodes assigned by their end update, not whole-window substep statistics. Unfinished episodes excluded from these counts but included in transition accounting."}, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))
    print(f"PASS: 98304 transitions accounted for, finite PPO updates, snapshot provenance and all {trajectory_count} evaluation trajectories replayed")


if __name__ == "__main__":
    main()
