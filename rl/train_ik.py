import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf

import ik_task
from ik_model import build_ik_model
from prepare import ROOT
from train_reference import guard_learner, load_runner


def configuration(assets, unilab):
    config = OmegaConf.load(unilab / "conf/ppo/config_mlx.yaml")
    del config["defaults"]
    return OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("ik_tracking.yaml")),
                           {"env": {"scene": {"model_file": str((assets / "flat.xml").resolve())}}})


def main():
    parser = argparse.ArgumentParser(description="Independent bounded IK soft-reference PPO experiment")
    parser.add_argument("assets", type=Path)
    parser.add_argument("--unilab", type=Path, default=Path.home() / "UniLab")
    parser.add_argument("--stage", choices=("smoke", "screen"), default="smoke")
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    config = configuration(args.assets, args.unilab)
    iterations = 10 if args.stage == "smoke" else 400
    config.algo.seed = args.seed
    config.algo.max_iterations = iterations
    config.algo.save_interval = iterations
    config.algo.resume = False
    config.algo.load_run = "-1"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output = ROOT / f"output/rl/ik_{args.stage}" / f"seed{args.seed}_{stamp}"
    output.mkdir(parents=True)
    config.training.log_root = str(output)
    config.training.seed = args.seed
    config.experiment = {"protocol": "ik-soft-reference-v1", "stage": args.stage,
                         "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ("ik_assets.py", "ik_task.py", "ik_model.py", "ik_tracking.yaml", "train_ik.py")},
                         "asset_sha256": {name: hashlib.sha256((args.assets / name).read_bytes()).hexdigest()
                                          for name in ("flat.xml", "manifest.json", "ik_reference.npz")},
                         "runner_sha256": hashlib.sha256((args.unilab / "scripts/train_mlx_ppo.py").read_bytes()).hexdigest()}
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    runner = load_runner(args.unilab)
    runner.build_model = build_ik_model
    records = []
    guard_learner(runner, records, output / "progress.json")
    started = time.perf_counter()
    result = {"status": "running", "planned_transitions": iterations * 32 * 24, "training_seed": args.seed}
    try:
        runner.main.__wrapped__(config)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoints = list(output.glob(f"G1IKTracking23/*/model_{iterations - 1}.safetensors"))
        if len(checkpoints) == 1:
            result["checkpoint"] = str(checkpoints[0])
        result.update(learner_iterations=records, recorded_transitions=len(records) * 32 * 24,
                      elapsed_seconds=time.perf_counter() - started)
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / "progress.json").write_text(json.dumps({"status": result["status"], "result": str(output / "result.json")}, indent=2) + "\n")
        print(output / "result.json", flush=True)


if __name__ == "__main__":
    main()
