from pathlib import Path
import argparse
import hashlib
import os
import runpy
import shutil
import sys

from omegaconf import OmegaConf

import crawl_task
import direct_task
import single_foot
import anchored_foot


def main():
    parser = argparse.ArgumentParser(description="Run the local G1 crawl task using the existing UniLab runner")
    parser.add_argument("--unilab", type=Path, default=Path.home() / "UniLab")
    parser.add_argument("--algo", choices=("mlx_ppo", "ppo"), default="mlx_ppo")
    parser.add_argument("--allow-legacy-force", action="store_true", help="Historical reproduction only: acknowledge known incorrect force rewards in V2-V4")
    parser.add_argument("--correct-force", action="store_true", help="Direct V2 only: change force data source without changing rewards or observations")
    parser.add_argument("--stage", choices=("support", "crawl", "direct-support", "direct-crawl", "balanced-crawl", "left-foot", "right-foot", "anchored-left", "anchored-right"), default="support")
    args, overrides = parser.parse_known_args()
    if args.correct_force and (args.allow_legacy_force or args.stage not in ("direct-support", "direct-crawl")):
        parser.error("--correct-force requires direct-support/direct-crawl and cannot use --allow-legacy-force")
    if any(value.split("=", 1)[0].lstrip("+~") == "env.force_source" for value in overrides):
        parser.error("Select the force path using --correct-force or --allow-legacy-force")
    root = Path(__file__).resolve().parents[1]
    source = args.unilab / "conf" / "ppo" / ("config_mlx.yaml" if args.algo == "mlx_ppo" else "config.yaml")
    config = OmegaConf.load(source)
    del config["defaults"]
    config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("flat.yaml")))
    assets = root / "output" / "rl" / "assets"
    anchored = args.stage.startswith("anchored-")
    foot_stage = args.stage in ("left-foot", "right-foot") or anchored
    balanced = args.stage == "balanced-crawl" or foot_stage
    direct = args.stage.startswith("direct-") or balanced
    if direct and not anchored and not args.allow_legacy_force and not args.correct_force:
        parser.error("V2-V4 force rewards failed the baseline audit. Training is blocked; --allow-legacy-force is only for explicitly reproducing the known flawed baseline. See docs/RL_BASELINE_AUDIT.md.")
    if direct:
        config.env.force_source = "analytic_pd" if args.correct_force else "legacy_sensor"
        assets = direct_task.prepare_direct(3 if balanced else 2)
        config.training.task_name = "G1SingleFoot23V4" if foot_stage else "G1CrawlBalanced23V3" if balanced else "G1CrawlDirect23V2"
        if anchored:
            config.training.task_name = "G1AnchoredFoot23V5"
        config.env.reference_motion = False
        config.env.phase_period = 4.0
        config.env.lift_height = 0.02
        config.reward.scales.torque_utilization = -0.05
        config.reward.scales.torque_near_limit = -2.0
        config.reward.scales.non_support_contact = -5.0
        config.env.control_config.action_scale = 0.5
        for override in overrides:
            if "load_run=" in override or "resume" in override:
                raise ValueError("V2 initial experiments start fresh; do not resume a V1 checkpoint with incompatible observations")
    filenames = ("flat.xml", "manifest.json") if direct else ("flat.xml", "manifest.json", "reference.npz")
    digest = hashlib.sha256(b"".join((assets / filename).read_bytes() for filename in filenames)).hexdigest()[:16]
    snapshot = root / "output" / "rl" / "models" / digest
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        if not (snapshot / filename).exists():
            shutil.copy2(assets / filename, snapshot / filename)
    config.env.scene = {"model_file": str(snapshot / "flat.xml")}
    config.training.log_root = str(root / "output" / "rl" / args.algo)
    if args.algo == "ppo":
        config.training.device = "mps"
    if args.stage == "crawl":
        config.env.speed_min = 0.05
        config.env.speed_max = 0.15
        config.env.reference_motion = True
        config.env.reference_scale = 1.0
        config.reward.scales.support = 4.0
        config.reward.scales.clearance = 4.0
        config.reward.scales.slip = -1.0
        config.reward.scales.pose = -0.05
    if args.stage in ("direct-crawl", "balanced-crawl"):
        config.env.speed_min = 0.02
        config.env.speed_max = 0.05
        config.reward.scales.support = 4.0
        config.reward.scales.clearance = 4.0
        config.reward.scales.slip = -1.0
        config.reward.scales.pose = -0.05
    if balanced:
        config.reward.scales.support = 0.0
        config.reward.scales.clearance = 0.0
        config.reward.scales.swing_gap = 4.0
        config.reward.scales.swing_contact = -2.0
        config.reward.scales.stance_contact = 2.0
        config.reward.scales.stance_hover = -1.0
        config.reward.scales.landing_contact = 1.0
    if foot_stage:
        config.env.target_foot = 2 if args.stage in ("left-foot", "anchored-left") else 3
        config.env.speed_min = 0.0
        config.env.speed_max = 0.0
        config.reward.scales.velocity = 1.0
        config.reward.scales.height = 0.5
        config.reward.scales.pose = -0.02
        config.reward.scales.swing_gap = 6.0
        config.reward.scales.swing_contact = -3.0
        config.reward.scales.stance_contact = 3.0
        config.reward.scales.stance_hover = -2.0
    if anchored:
        config.reward.scales.anchor_drift = -0.5
        config.reward.scales.base_drift = -1.0
    config.env.seed = "${algo.seed}"
    directory = root / "output" / "rl" / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    config_name = f"{args.algo}_{args.stage}" + ("_correct_force" if args.correct_force else "")
    OmegaConf.save(config, directory / f"{config_name}.yaml")
    script = args.unilab / "scripts" / ("train_mlx_ppo.py" if args.algo == "mlx_ppo" else "train_rsl_rl.py")
    sys.argv = [str(script), "--config-path", str(directory), "--config-name", config_name, *overrides]
    os.chdir(root)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
