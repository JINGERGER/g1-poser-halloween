import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from omegaconf import OmegaConf

from prepare import ROOT
from short_step_assets import prepare
from target_load_ablation import digest
from train_short_step import load_runner
from event_lift_task import WEIGHTS


CHECKPOINT_UPDATES = (64, 128, 256)


def main():
    parser = argparse.ArgumentParser(description="Fixed 4x unload objective budget; checkpoint evaluation occurs after training")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    bundle = prepare()
    unilab = Path.home() / "UniLab"
    config = OmegaConf.load(unilab / "conf/ppo/config_mlx.yaml")
    del config["defaults"]
    config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("short_step.yaml")))
    config.training.task_name = "G1EventLiftV1"
    config.algo.seed = config.training.seed = config.env.seed = args.seed
    config.env.bundle = str(bundle)
    config.env.reference_gain = 0.
    config.reward = {"scales": WEIGHTS}
    config.algo.num_envs = 8
    config.algo.max_iterations = 256
    config.algo.save_interval = 0
    config.algo.resume = False
    config.algo.load_run = "-1"
    folder = ROOT / "output/rl/event_lift_training" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder.mkdir(parents=True)
    config.training.log_root = str(folder / "logs")
    config.env.telemetry_path = str(folder / "episodes.json")
    OmegaConf.save(config, folder / "config.yaml", resolve=True)
    result = {"status": "running", "stage": "curve", "seed": args.seed, "planned_transitions": 49152,
              "config": str(folder / "config.yaml"), "config_sha256": digest(folder / "config.yaml"),
              "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "runner": str(unilab / "scripts/train_mlx_ppo.py"), "runner_sha256": digest(unilab / "scripts/train_mlx_ppo.py"),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "event_lift_task.py", "unload_objective_task.py", "short_step_task.py", "short_step_assets.py", "short_step_protocol.json", "short_step.yaml")},
              "checkpoint_updates": list(CHECKPOINT_UPDATES), "updates": [], "snapshots": []}
    report_path = folder / "report.json"
    started = time.monotonic()
    try:
        runner = load_runner(unilab)
        runtime = runner._require_mlx_runtime()
        original = runtime.PPOTrainer

        class CheckedTrainer(original):
            def update(self, buffer, iteration=-1):
                metrics = super().update(buffer, iteration)
                result["updates"].append({"iteration": iteration, **metrics})
                if metrics.get("updates_applied", 0) <= 0 or any(value for name, value in metrics.items() if name.startswith(("skipped_nonfinite", "rolled_back"))):
                    raise FloatingPointError("PPO finite-update gate failed")
                if iteration + 1 in CHECKPOINT_UPDATES:
                    checkpoint = folder / f"model_{iteration}.safetensors"
                    self.model.save_weights(str(checkpoint))
                    snapshot = {**result, "status": "completed", "snapshots": [], "checkpoint": str(checkpoint),
                                "checkpoint_sha256": digest(checkpoint), "completed_updates": iteration + 1,
                                "observed_transitions": (iteration + 1) * 8 * 24,
                                "limitations": "Checkpoint snapshot of one uninterrupted 256-update run, not a separate training run. Evaluation must not alter the training RNG stream."}
                    snapshot_path = folder / f"snapshot_{iteration + 1}.json"
                    snapshot_path.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
                    result["snapshots"].append({"path": str(snapshot_path), "sha256": digest(snapshot_path)})
                report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                return metrics

        runtime.PPOTrainer = CheckedTrainer
        runner.main.__wrapped__(config)
        assert len(result["updates"]) == 256 and len(result["snapshots"]) == 3
        telemetry = Path(config.env.telemetry_path)
        assert json.loads(telemetry.read_text())["transitions"] == 49152
        result.update(status="completed", telemetry=str(telemetry), telemetry_sha256=digest(telemetry))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
