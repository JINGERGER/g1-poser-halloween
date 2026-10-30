import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from component_unload_return import envelope
from prepare import ROOT
from short_step_assets import prepare
from exploration_task import ExplorationCfg as ShortStepCfg, ExplorationEnv as ShortStepEnv
from target_load_ablation import digest


def episode(bundle, label, policy, reference_gain, output):
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle), reference_gain=reference_gain))
    env._autoreset = False
    state = env.init_state()
    rewards, actions, rows = [], [], []
    positions, velocities = [env.datas[0].qpos.copy()], [env.datas[0].qvel.copy()]
    try:
        for unused in range(200):
            action = np.asarray(policy(env, state), dtype=np.float32).reshape(1, 23)
            state = env.step(action)
            actions.append(action[0].copy())
            rewards.append(float(state.reward[0]))
            rows.append(state.info["evaluation"][0])
            positions.append(env.datas[0].qpos.copy())
            velocities.append(env.datas[0].qvel.copy())
            if state.terminated[0] or state.truncated[0]:
                break
        assert state.terminated[0] and not state.truncated[0]
        np.savez_compressed(output, qpos=positions, qvel=velocities, actions=actions, rewards=rewards)
        return {"label": label, "reference_gain": reference_gain, "success": rows[-1]["success"],
                "max_stage": max(row["stage"] for row in rows), "failure": rows[-1]["failure"],
                "deadline": rows[-1]["deadline"], "elapsed_s": rows[-1]["elapsed_s"],
                "return": sum(rewards), "trace": rows, "trajectory": str(output), "trajectory_sha256": digest(output)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Complete-event evaluation; return is not success")
    parser.add_argument("--training-report", type=Path)
    args = parser.parse_args()
    trained = json.loads(args.training_report.read_text()) if args.training_report else None
    if trained:
        assert trained["status"] == "completed"
        for key in ("bundle", "checkpoint", "config"):
            assert digest(trained[key]) == trained[key + "_sha256"]
        for name, expected in trained["source_sha256"].items():
            assert digest(Path(__file__).with_name(name)) == expected
    bundle = Path(trained["bundle"]) if trained else prepare()
    payload = json.loads(bundle.read_text())
    recovery_path = ROOT / "output/rl/component_unload_return/2026-10-19_20-50-12-737895/report.json"
    recovery = json.loads(recovery_path.read_text())
    assert recovery["model_sha256"] == payload["model_sha256"]
    direction = np.array(recovery["candidate_ctrl"]) - payload["initial_ctrl"]
    cases = [("hold", lambda env, state: np.zeros(23), 0.),
             ("ik_zero_residual", lambda env, state: np.zeros(23), 1.),
             ("known_unload_return", lambda env, state: .25 * envelope(env.elapsed[0] + .02, 1.) * direction / env.protocol["residual_scale_rad"], 0.)]
    if trained:
        import mlx.core as mx
        from train_short_step import load_runner
        config = OmegaConf.load(trained["config"])
        runner = load_runner(Path(trained["runner"]).parents[1])
        model = runner.build_model(config.algo, 148, 23)
        model.load_weights(trained["checkpoint"], strict=True)
        model.eval()
        cases.append(("ppo_deterministic", lambda env, state: np.asarray(model.policy(mx.array(state.obs["obs"])))[0], 1.))
    destination = ROOT / "output/rl/exploration_evaluation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "training_report": str(args.training_report.resolve()) if args.training_report else None,
              "training_report_sha256": digest(args.training_report) if args.training_report else None,
              "recovery_report": str(recovery_path), "recovery_report_sha256": digest(recovery_path),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "exploration_task.py", "short_step_task.py", "component_unload_return.py")},
              "limitations": "RELAXED TRAINING ENVIRONMENT evaluation, not strict task acceptance. One deterministic nominal reset per case, not a statistical success rate or robustness test. Baselines differ in control reference only, not task gates or physics. Smoke policy is not expected to solve the task.", "runs": []}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for label, policy, reference_gain in cases:
            result = episode(bundle, label, policy, reference_gain, destination / f"{label}.npz")
            report["runs"].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "trace"}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
