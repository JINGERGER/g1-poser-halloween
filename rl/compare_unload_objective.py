from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline_path = ROOT / "output/rl/exploration_comparison/2026-10-19_21-39-17-849460/report.json"
    baseline = json.loads(baseline_path.read_text())
    assert baseline["status"] == "completed"
    destination = ROOT / "output/rl/unload_objective_comparison" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "planned_screen_transitions": 24576, "planned_smoke_transitions": 384,
              "source_sha256": digest(__file__), "baseline_report": str(baseline_path), "baseline_sha256": digest(baseline_path),
              "protocol": "Reward-only ablation; fresh seeds 1/2, 12288 transitions each, archived strict-training controls at equal budget; no resume, no tuning, strict evaluation only.", "runs": []}
    report_path = destination / "report.json"

    def execute(script, arguments, label):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "rl" / script), *arguments], cwd=ROOT, capture_output=True, text=True)
        (destination / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see log")
        path = Path(result.stdout.strip().splitlines()[-1])
        payload = json.loads(path.read_text())
        assert payload["status"] == "completed"
        return {"path": str(path), "sha256": digest(path)}, payload

    try:
        smoke, payload = execute("train_unload_objective.py", ["--stage", "smoke", "--seed", "1"], "smoke")
        assert len(payload["updates"]) == 4
        report["smoke"] = smoke
        for seed in (1, 2):
            control = next(run for run in baseline["runs"] if run["seed"] == seed and run["training_mode"] == "strict")
            provenance, training = execute("train_unload_objective.py", ["--stage", "screen", "--seed", str(seed)], f"train_{seed}")
            assert len(training["updates"]) == 64 and training["planned_transitions"] == 12288
            original = json.loads(Path(control["training_report"]["path"]).read_text())
            assert digest(control["training_report"]["path"]) == control["training_report"]["sha256"]
            assert original["bundle_sha256"] == training["bundle_sha256"] and original["runner_sha256"] == training["runner_sha256"]
            evaluation, evaluated = execute("evaluate_short_step.py", ["--training-report", provenance["path"]], f"eval_{seed}")
            check = subprocess.run([sys.executable, "-B", str(ROOT / "rl/check_short_step_evaluation.py"), evaluation["path"]], cwd=ROOT, capture_output=True, text=True)
            (destination / f"check_{seed}.log").write_text(check.stdout + check.stderr)
            if check.returncode:
                raise RuntimeError("Native evaluation replay failed")
            policy = next(run for run in evaluated["runs"] if run["label"] == "ppo_deterministic")
            summary = {key: policy[key] for key in ("success", "max_stage", "failure", "elapsed_s", "return")}
            summary["min_left_load_n"] = min(row["loads_n"][0] for row in policy["trace"])
            summary["max_left_gap_m"] = max(row["gaps_m"][0] for row in policy["trace"])
            report["runs"].append({"seed": seed, "training_report": provenance, "evaluation": evaluation,
                                   "control": control, "native_replay_passed": True, "summary": summary})
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"seed": seed, **summary}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
