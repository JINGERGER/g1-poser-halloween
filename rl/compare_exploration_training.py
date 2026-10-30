from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "output/rl/exploration_comparison" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "planned_transitions": 49152,
              "protocol": "Two seeds, strict versus relaxed termination; 12288 transitions each, no resume, no selection or extension. Evaluate each checkpoint in both environments.",
              "source_sha256": digest(__file__), "runs": []}
    report_path = destination / "report.json"

    def execute(script, arguments, label):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "rl" / script), *arguments], cwd=ROOT,
                                capture_output=True, text=True)
        (destination / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed: see saved log")
        path = Path(result.stdout.strip().splitlines()[-1])
        payload = json.loads(path.read_text())
        assert payload["status"] == "completed"
        return {"path": str(path), "sha256": digest(path)}, payload

    try:
        for seed in (1, 2):
            for mode in ("strict", "relaxed"):
                train_script = "train_short_step.py" if mode == "strict" else "train_exploration.py"
                provenance, trained = execute(train_script, ["--stage", "screen", "--seed", str(seed)], f"{mode}_{seed}_train")
                assert trained["planned_transitions"] == 12288 and len(trained["updates"]) == 64
                entry = {"seed": seed, "training_mode": mode, "training_report": provenance, "evaluations": {}}
                report["runs"].append(entry)
                for evaluation, script in (("strict", "evaluate_short_step.py"), ("relaxed", "evaluate_exploration.py")):
                    evidence, evaluated = execute(script, ["--training-report", provenance["path"]], f"{mode}_{seed}_{evaluation}_eval")
                    policy = next(run for run in evaluated["runs"] if run["label"] == "ppo_deterministic")
                    summary = {key: policy[key] for key in ("success", "max_stage", "failure", "elapsed_s", "return")}
                    entry["evaluations"][evaluation] = {"report": evidence, **summary}
                    print(json.dumps({"seed": seed, "trained": mode, "evaluated": evaluation, **summary}), flush=True)
                    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
