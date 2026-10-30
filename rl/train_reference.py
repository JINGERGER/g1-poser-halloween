import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf

from prepare import ROOT
from reference_assets import prepare_reference
import reference_task


def load_runner(unilab):
    path = unilab / "scripts/train_mlx_ppo.py"
    spec = importlib.util.spec_from_file_location("g1_reference_mlx_runner", path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    from reference_model import build_reference_model
    runner.build_model = build_reference_model
    return runner


def guard_learner(runner, records, progress_path=None):
    runtime = runner._require_mlx_runtime()
    base_trainer = runtime.PPOTrainer
    started = time.monotonic()

    class CheckedTrainer(base_trainer):
        def update(self, buffer, iteration=-1):
            metrics = super().update(buffer, iteration)
            records.append({"iteration": iteration, **metrics})
            errors = {name: value for name, value in metrics.items() if name.startswith(("skipped_nonfinite", "rolled_back")) and value}
            failed = bool(errors or metrics.get("updates_applied", 0) <= 0)
            if progress_path is not None and (iteration % 50 == 0 or failed):
                temporary = progress_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"status": "failed" if failed else "training", "iteration": iteration,
                                                 "elapsed_seconds": time.monotonic() - started, "metrics": metrics}, indent=2) + "\n")
                temporary.replace(progress_path)
            if failed:
                raise FloatingPointError(f"Stage-0 learner failed: {records[-1]}")
            return metrics

    runtime.PPOTrainer = CheckedTrainer


def main():
    parser = argparse.ArgumentParser(description="Bounded reference-inspired smoke or frozen A/B screen")
    parser.add_argument("--stage", choices=("smoke", "screen"), default="smoke")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--unilab", type=Path, default=Path.home() / "UniLab")
    parser.add_argument("--pd", choices=("current", "reference"), default="current")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = 10 if args.stage == "smoke" else 3907
    if args.stage == "smoke" and not (1 <= args.iterations <= 100 and 1 <= args.envs <= 128 and args.seed >= 0):
        parser.error("Stage-0 bounds: 1–100 iterations, 1–128 envs, nonnegative seed")
    if args.stage == "screen" and (args.iterations != 3907 or args.envs != 32 or args.seed not in (1, 2)):
        parser.error("Frozen screen: 3907 iterations, 32 envs, seed 1 or 2")
    assets = prepare_reference(args.pd)
    config = OmegaConf.load(args.unilab / "conf/ppo/config_mlx.yaml")
    del config["defaults"]
    config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("reference.yaml")))
    config.algo.seed = args.seed
    config.algo.num_envs = args.envs
    config.algo.max_iterations = args.iterations
    config.algo.save_interval = args.iterations if args.stage == "smoke" else 500
    config.algo.resume = False
    config.algo.load_run = "-1"
    config.env.scene = {"model_file": str(assets / "flat.xml")}
    log_root = args.log_root or ROOT / f"output/rl/reference_{args.stage}" / args.pd
    config.training.log_root = str(log_root.resolve())
    if args.stage == "screen":
        config.training.log_interval = 50
    config.training.seed = args.seed
    source_files = ("reference_task.py", "reference_model.py", "reference_assets.py", "reference.yaml", "train_reference.py")
    config.experiment = {
        "stage": "stage-0-smoke-only" if args.stage == "smoke" else "stage-1-frozen-screen", "pd_variant": args.pd,
        "observation_protocol": "reference-inspired-75-81-v1",
        "deviations": ["no symmetry augmentation/mirror loss", "no physical DR", "endpoint contact flags instead of force history",
                       "raw-minus-applied PD torque penalty", "current hard joint limits", "2ms x 10 integration", "safety termination",
                       "implemented animation reset intent with floor projection and no animation velocities"],
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files},
        "runner_sha256": hashlib.sha256((args.unilab / "scripts/train_mlx_ppo.py").read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((assets / "flat.xml").read_bytes()).hexdigest(),
        "evaluation_protocol_sha256": hashlib.sha256(Path(__file__).with_name("reference_eval.json").read_bytes()).hexdigest(),
    }
    directory = ROOT / "output/rl/configs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    config_path = directory / f"reference_{args.stage}_{args.pd}_{stamp}.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    runner = load_runner(args.unilab)
    learner_records = []
    if args.result_path is not None:
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
    guard_learner(runner, learner_records, args.result_path.with_name("progress.json") if args.result_path else None)
    previous_runs = set(log_root.glob("G1CrawlReference23/*/run_config.json"))
    started = time.perf_counter()
    result = {"config": str(config_path), "status": "failed", "planned_transitions": args.envs * 24 * args.iterations}
    try:
        runner.main.__wrapped__(config)
        result["status"] = "completed"
        result["completed_transitions"] = result["planned_transitions"]
    finally:
        new_runs = set(log_root.glob("G1CrawlReference23/*/run_config.json")) - previous_runs
        if len(new_runs) == 1:
            run_path = next(iter(new_runs)).parent
            result["run_directory"] = str(run_path.resolve())
            final_checkpoint = run_path / f"model_{args.iterations - 1}.safetensors"
            if final_checkpoint.exists():
                result["checkpoint"] = str(final_checkpoint.resolve())
        result["learner_iterations"] = learner_records
        result["transitions_in_recorded_rollouts"] = len(learner_records) * args.envs * 24
        result["wall_seconds"] = time.perf_counter() - started
        if result["status"] == "completed":
            result["transitions_per_second_including_training"] = result["completed_transitions"] / result["wall_seconds"]
        report = args.result_path or config_path.with_suffix(".result.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")
        if args.result_path is not None:
            progress_path = args.result_path.with_name("progress.json")
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"status": result["status"],
                                             "iteration": learner_records[-1]["iteration"] if learner_records else None,
                                             "elapsed_seconds": result["wall_seconds"], "result": str(report.resolve())}, indent=2) + "\n")
            temporary.replace(progress_path)
        print(report, flush=True)


if __name__ == "__main__":
    main()
