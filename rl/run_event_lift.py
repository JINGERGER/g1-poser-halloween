from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "output/rl/event_lift_curve" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "planned_transitions": 98304, "runs": [],
              "protocol": "Event-driven lift/advance/land course, no clock IK action reference. Two fresh seeds, 256 updates each. Fixed strict evaluations at 64/128/256 updates AFTER training; no best checkpoint selection, tuning, extra budget or curriculum change.",
              "source_sha256": digest(__file__)}
    report_path = destination / "report.json"

    def execute(script, arguments, label):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "rl" / script), *arguments], cwd=ROOT, capture_output=True, text=True)
        (destination / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed: see log")
        path = Path(result.stdout.strip().splitlines()[-1])
        payload = json.loads(path.read_text())
        assert payload["status"] == "completed"
        return {"path": str(path), "sha256": digest(path)}, payload

    try:
        for seed in (1, 2):
            training, trained = execute("train_event_lift.py", ["--seed", str(seed)], f"train_{seed}")
            entry = {"seed": seed, "training": training, "evaluations": []}
            report["runs"].append(entry)
            for snapshot in trained["snapshots"]:
                assert digest(snapshot["path"]) == snapshot["sha256"]
                saved = json.loads(Path(snapshot["path"]).read_text())
                updates = saved["completed_updates"]
                evidence, evaluation = execute("evaluate_event_lift.py", ["--training-report", snapshot["path"]], f"eval_{seed}_{updates}")
                policy = next(run for run in evaluation["runs"] if run["label"] == "ppo_deterministic")
                rows = policy["trace"]
                window = [row for row in rows if .4 <= row["elapsed_s"] < 2.6]
                summary = {key: policy[key] for key in ("success", "max_stage", "failure", "elapsed_s", "return")}
                summary.update(min_left_load_n=min(row["loads_n"][0] for row in rows),
                               max_left_gap_m=max(row["gaps_m"][0] for row in rows),
                               window_complete=policy["elapsed_s"] >= 2.6,
                               mean_window_left_load_n=sum(row["loads_n"][0] for row in window) / len(window) if window else None)
                entry["evaluations"].append({"updates": updates, "transitions": updates * 192,
                                              "report": evidence, "summary": summary})
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"seed": seed, "updates": updates, **summary}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
