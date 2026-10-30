import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf

from prepare import ROOT
from short_step_assets import prepare
import short_step_task
from target_load_ablation import digest


def load_runner(unilab):
    spec = importlib.util.spec_from_file_location("g1_short_step_mlx_runner", unilab / "scripts/train_mlx_ppo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Bounded native short-step residual PPO; smoke is not evidence of learning")
    parser.add_argument("--stage", choices=("smoke", "screen"), default="smoke")
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--unilab", type=Path, default=Path.home() / "UniLab")
    args = parser.parse_args()
    bundle = prepare()
    iterations, envs = (4, 4) if args.stage == "smoke" else (64, 8)
    config = OmegaConf.load(args.unilab / "conf/ppo/config_mlx.yaml")
    del config["defaults"]
    config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("short_step.yaml")))
    config.algo.seed = config.training.seed = args.seed
    config.env.seed = args.seed
    config.env.bundle = str(bundle)
    config.reward = {"scales": json.loads(bundle.read_text())["protocol"]["reward"]}
    config.algo.num_envs = envs
    config.algo.max_iterations = iterations
    config.algo.save_interval = iterations
    config.algo.resume = False
    config.algo.load_run = "-1"
    folder = ROOT / "output/rl/short_step_training" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder.mkdir(parents=True)
    config.training.log_root = str(folder / "logs")
    config.experiment = {"stage": args.stage, "bundle_sha256": digest(bundle), "no_curriculum": True,
                         "limitations": "Nominal model only; smoke verifies software, not learned locomotion. No scene/default model changes."}
    OmegaConf.save(config, folder / "config.yaml", resolve=True)
    result = {"status": "running", "stage": args.stage, "seed": args.seed, "planned_transitions": iterations * envs * 24,
              "config": str(folder / "config.yaml"), "config_sha256": digest(folder / "config.yaml"),
              "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "runner": str(args.unilab / "scripts/train_mlx_ppo.py"), "runner_sha256": digest(args.unilab / "scripts/train_mlx_ppo.py"),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "short_step_task.py", "short_step_assets.py", "short_step_protocol.json", "short_step.yaml")},
              "updates": []}
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    try:
        runner = load_runner(args.unilab)
        runtime = runner._require_mlx_runtime()
        original = runtime.PPOTrainer

        class CheckedTrainer(original):
            def update(self, buffer, iteration=-1):
                metrics = super().update(buffer, iteration)
                result["updates"].append({"iteration": iteration, **metrics})
                report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                if metrics.get("updates_applied", 0) <= 0 or any(value for name, value in metrics.items() if name.startswith(("skipped_nonfinite", "rolled_back"))):
                    raise FloatingPointError("PPO finite-update gate failed")
                return metrics

        runtime.PPOTrainer = CheckedTrainer
        runner.main.__wrapped__(config)
        assert len(result["updates"]) == iterations
        checkpoints = list((folder / "logs").glob(f"G1ComponentShortStepV1/*/model_{iterations - 1}.safetensors"))
        assert len(checkpoints) == 1
        result.update(status="completed", checkpoint=str(checkpoints[0]), checkpoint_sha256=digest(checkpoints[0]))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
