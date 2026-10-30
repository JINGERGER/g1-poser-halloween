import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from prepare import ROOT


RUNS = (("current", 1), ("reference", 1), ("current", 2), ("reference", 2))


def source_hashes():
    directory = Path(__file__).parent
    names = ("reference_task.py", "reference_model.py", "reference_assets.py", "reference.yaml", "reference_eval.json",
             "reference_gait.py", "evaluate_reference.py", "evaluate_support.py", "train_reference.py", "run_reference_matrix.py",
             "servo_force.py", "crawl_task.py", "direct_task.py", "prepare.py")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def remaining_budget(started, started_wall, seconds):
    return min(seconds - (time.monotonic() - started), seconds - (time.time() - started_wall))


def bounded_process(command, log_path, seconds):
    started = time.monotonic()
    started_wall = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = remaining_budget(started, started_wall, seconds)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds)
                try:
                    code = process.wait(timeout=min(1.0, remaining))
                    if remaining_budget(started, started_wall, seconds) <= 0:
                        raise subprocess.TimeoutExpired(command, seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Child exited {code}; see {log_path}")
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description="Four frozen runs, 3,000,576 transitions each, with fixed first-episode evaluations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination = ROOT / "output/rl/reference_matrix" / stamp
    plan = {"status": "planned", "runs": [], "transitions_per_run": 3907 * 32 * 24,
            "total_planned_transitions": 4 * 3907 * 32 * 24,
            "per_training_timeout_s": 1800, "per_evaluation_timeout_s": 300, "total_timeout_s": 5400,
            "source_sha256": source_hashes(), "selection": "final checkpoint only, no best-reward selection"}
    commands = []
    for pd_variant, seed in RUNS:
        label = f"{pd_variant}_seed{seed}"
        slot = destination / label
        command = [sys.executable, "-B", str(ROOT / "rl/train_reference.py"), "--stage", "screen", "--pd", pd_variant,
                   "--seed", str(seed), "--result-path", str(slot / "training.json"), "--log-root", str(slot / "models")]
        commands.append(command)
        plan["runs"].append({"pd": pd_variant, "seed": seed, "status": "pending", "directory": str(slot), "command": command})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    destination.mkdir(parents=True)
    report_path = destination / "matrix.json"
    started = time.monotonic()
    started_wall = time.time()
    plan["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["status"] = "running"
    save(report_path, plan)
    print(report_path, flush=True)
    try:
        for entry, command in zip(plan["runs"], commands):
            if source_hashes() != plan["source_sha256"]:
                raise RuntimeError("Source/config changed during the frozen matrix")
            slot = Path(entry["directory"])
            slot.mkdir()
            entry["status"] = "training"
            save(report_path, plan)
            remaining = remaining_budget(started, started_wall, plan["total_timeout_s"])
            if remaining <= 0:
                raise TimeoutError("Overall experiment wall-clock budget exhausted")
            print(f"training {entry['pd']} seed={entry['seed']}", flush=True)
            entry["training_wall_seconds"] = bounded_process(command, slot / "training.log", min(remaining, plan["per_training_timeout_s"]))
            result = json.loads((slot / "training.json").read_text())
            if result["status"] != "completed" or result.get("completed_transitions") != plan["transitions_per_run"] or "checkpoint" not in result:
                raise RuntimeError("Training did not complete the frozen budget with a final checkpoint")
            checkpoint = Path(result["checkpoint"])
            entry["checkpoint"] = str(checkpoint)
            entry["completed_transitions"] = result["completed_transitions"]
            entry["status"] = "evaluating"
            save(report_path, plan)
            if source_hashes() != plan["source_sha256"]:
                raise RuntimeError("Source/config changed before evaluation")
            remaining = remaining_budget(started, started_wall, plan["total_timeout_s"])
            if remaining <= 0:
                raise TimeoutError("Overall experiment wall-clock budget exhausted")
            evaluation = [sys.executable, "-B", str(ROOT / "rl/evaluate_reference.py"), str(checkpoint), "--record-first"]
            entry["evaluation_wall_seconds"] = bounded_process(evaluation, slot / "evaluation.log", min(remaining, plan["per_evaluation_timeout_s"]))
            protocol = json.loads((ROOT / "rl/reference_eval.json").read_text())
            evaluation_path = checkpoint.with_name(f"{checkpoint.stem}_reference_eval_seed{protocol['seed']}_{protocol['episodes']}episodes.json")
            assessment = json.loads(evaluation_path.read_text())
            if assessment["protocol_sha256"] != plan["source_sha256"]["reference_eval.json"]:
                raise RuntimeError("Evaluation protocol changed")
            entry["evaluation"] = str(evaluation_path)
            entry["summary"] = assessment["summary"]
            reset_hashes = [row["initial_state_sha256"] for row in assessment["episodes"]]
            if "evaluation_initial_states" not in plan:
                plan["evaluation_initial_states"] = reset_hashes
            elif reset_hashes != plan["evaluation_initial_states"]:
                raise RuntimeError("Held-out initial states are not paired across runs")
            entry["status"] = "completed"
            plan["elapsed_seconds"] = time.monotonic() - started
            save(report_path, plan)
            print(f"completed {entry['pd']} seed={entry['seed']} passed={assessment['summary']['passed']}/100", flush=True)
        plan["both_seeds_passed"] = {variant: all(entry["summary"]["screen_passed"] for entry in plan["runs"] if entry["pd"] == variant)
                                     for variant in ("current", "reference")}
        plan["automatic_budget_extension"] = False
        plan["status"] = "completed"
    except BaseException as error:
        plan["status"] = "failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        for entry in plan["runs"]:
            if entry["status"] in ("training", "evaluating"):
                entry["status"] = "failed"
        raise
    finally:
        plan["elapsed_seconds"] = time.monotonic() - started
        plan["wall_elapsed_seconds"] = time.time() - started_wall
        plan["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(report_path, plan)
        print(report_path, flush=True)


if __name__ == "__main__":
    main()
