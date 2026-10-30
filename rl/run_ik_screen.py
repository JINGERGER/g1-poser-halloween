import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from prepare import ROOT
from run_reference_matrix import bounded_process, remaining_budget, save


def hashes(assets):
    names = ("ik_task.py", "ik_model.py", "ik_assets.py", "ik_tracking.yaml", "ik_eval.json", "train_ik.py", "evaluate_ik.py",
             "run_ik_screen.py", "reference_gait.py", "reference_model.py", "train_reference.py", "run_reference_matrix.py",
             "crawl_task.py", "direct_task.py", "servo_force.py", "evaluate_support.py")
    paths = [Path(__file__).with_name(name) for name in names] + [assets / name for name in ("flat.xml", "manifest.json", "ik_reference.npz")]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description="Two 307,200-transition runs and four paired 100-episode evaluations; no automatic extension")
    parser.add_argument("assets", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.assets = args.assets.resolve()
    plan = {"status": "planned", "total_planned_transitions": 614400, "transitions_per_seed": 307200,
            "seeds": [1, 2], "episodes_per_evaluation": 100, "total_timeout_s": 1800,
            "per_train_timeout_s": 600, "per_eval_timeout_s": 300, "runs": [], "evaluations": [],
            "source_and_asset_hashes": hashes(args.assets), "selection": "final checkpoints only; same 100 first episodes; no resume/extension"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output = ROOT / "output/rl/ik_screen_matrix" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    path = output / "matrix.json"
    started, started_wall = time.monotonic(), time.time()
    plan.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
    initial_hashes = None

    def execute(command, log, cap):
        if hashes(args.assets) != plan["source_and_asset_hashes"]:
            raise RuntimeError("Frozen experiment files changed")
        remaining = remaining_budget(started, started_wall, plan["total_timeout_s"])
        if remaining <= 0:
            raise TimeoutError("Overall budget exhausted")
        bounded_process(command, log, min(cap, remaining))
        if hashes(args.assets) != plan["source_and_asset_hashes"]:
            raise RuntimeError("Files changed during child execution")

    try:
        save(path, plan)
        print(path, flush=True)
        for seed in plan["seeds"]:
            previous = set((ROOT / "output/rl/ik_screen").glob(f"seed{seed}_*/result.json"))
            execute([sys.executable, "-B", str(ROOT / "rl/train_ik.py"), str(args.assets), "--stage", "screen", "--seed", str(seed)],
                    output / f"seed{seed}_training.log", plan["per_train_timeout_s"])
            new = set((ROOT / "output/rl/ik_screen").glob(f"seed{seed}_*/result.json")) - previous
            if len(new) != 1:
                raise RuntimeError("Expected exactly one new training result")
            training_path = next(iter(new))
            training = json.loads(training_path.read_text())
            if training["status"] != "completed" or training["recorded_transitions"] != plan["transitions_per_seed"] or "checkpoint" not in training:
                raise RuntimeError("Training budget or checkpoint incomplete")
            checkpoint = Path(training["checkpoint"])
            plan["runs"].append({"seed": seed, "training_result": str(training_path), "checkpoint": str(checkpoint)})
            save(path, plan)
            for controller in (("hold", "zero", "learned") if seed == 1 else ("learned",)):
                execute([sys.executable, "-B", str(ROOT / "rl/evaluate_ik.py"), str(checkpoint), "--controller", controller],
                        output / f"seed{seed}_{controller}_evaluation.log", plan["per_eval_timeout_s"])
                evaluation_path = checkpoint.with_name(f"{checkpoint.stem}_ik_{controller}_100episodes.json")
                evaluation = json.loads(evaluation_path.read_text())
                actual = [row["initial_state_sha256"] for row in evaluation["episodes"]]
                if len(actual) != 100 or initial_hashes is not None and actual != initial_hashes:
                    raise AssertionError("First-episode starts are not paired")
                initial_hashes = actual
                plan["evaluations"].append({"seed": seed, "controller": controller, "report": str(evaluation_path),
                                            "summary": evaluation["summary"], "screen_passed": evaluation["screen_passed"]})
                save(path, plan)
                print(json.dumps(plan["evaluations"][-1]), flush=True)
        plan["status"] = "completed"
        plan["both_seeds_passed"] = all(row["screen_passed"] for row in plan["evaluations"] if row["controller"] == "learned")
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        plan.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                    calendar_elapsed_seconds=time.time() - started_wall)
        save(path, plan)
        print(path, flush=True)


if __name__ == "__main__":
    main()
