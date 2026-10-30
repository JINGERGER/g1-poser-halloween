import argparse
from datetime import datetime
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf

from check_coordinated_exploration import verify_replay
from event_lift_task import EventLiftCfg, EventLiftEnv
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest
from train_short_step import load_runner


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", type=Path, required=True)
    args = parser.parse_args()
    trained = json.loads(args.training_report.read_text())
    assert trained["status"] == "completed"
    for key in ("config", "checkpoint", "bundle", "runner"):
        assert digest(trained[key]) == trained[key + "_sha256"]
    for name, expected in trained["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    config = OmegaConf.load(trained["config"])
    runner = load_runner(Path(trained["runner"]).parents[1])
    model = runner.build_model(config.algo, 148, 23)
    model.load_weights(trained["checkpoint"], strict=True)
    model.eval()
    folder = ROOT / "output/rl/event_lift_evaluation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder.mkdir(parents=True)
    report = {"status": "running", "bundle": trained["bundle"], "bundle_sha256": trained["bundle_sha256"],
              "training_report": str(args.training_report.resolve()), "training_report_sha256": digest(args.training_report),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "event_lift_task.py", "unload_objective_task.py", "short_step_task.py", "component_unload_ramps.py")},
              "limitations": "Policy observes event-driven goals. Independent legacy strict environment receives identical actions with clock IK gain 0; saved reward and event trace come from that strict shadow, NOT the new training reward. One nominal deterministic episode per case, not robustness evidence.", "runs": []}
    path = folder / "report.json"
    try:
        for label in ("hold", "ppo_deterministic"):
            env = EventLiftEnv(EventLiftCfg(bundle=trained["bundle"]))
            shadow = ShortStepEnv(ShortStepCfg(bundle=trained["bundle"], reference_gain=0.))
            env._autoreset = shadow._autoreset = False
            rows, actions, rewards, training_rewards = [], [], [], []
            try:
                state = env.init_state()
                shadow.init_state()
                positions, velocities = [shadow.datas[0].qpos.copy()], [shadow.datas[0].qvel.copy()]
                for unused in range(200):
                    action = np.zeros((1, 23), dtype=np.float32) if label == "hold" else np.asarray(model.policy(mx.array(state.obs["obs"])))
                    state = env.step(action)
                    strict = shadow.step(action)
                    np.testing.assert_array_equal(state.terminated, strict.terminated)
                    np.testing.assert_allclose(env.datas[0].qpos, shadow.datas[0].qpos, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(env.datas[0].qvel, shadow.datas[0].qvel, atol=1e-12, rtol=0)
                    for key in ("stage", "success", "failure", "deadline", "elapsed_s"):
                        assert state.info["evaluation"][0][key] == strict.info["evaluation"][0][key]
                    rows.append(strict.info["evaluation"][0])
                    actions.append(action[0].copy())
                    rewards.append(float(strict.reward[0]))
                    training_rewards.append(float(state.reward[0]))
                    positions.append(shadow.datas[0].qpos.copy())
                    velocities.append(shadow.datas[0].qvel.copy())
                    if strict.terminated[0]:
                        break
                assert strict.terminated[0]
                trajectory = folder / f"{label}.npz"
                np.savez_compressed(trajectory, qpos=positions, qvel=velocities, actions=actions, rewards=rewards, training_rewards=training_rewards)
                trial = {"label": label, "reference_gain": 0., "trajectory": str(trajectory), "trajectory_sha256": digest(trajectory),
                         "trace": rows, "return": sum(rewards), "training_return": sum(training_rewards), "success": rows[-1]["success"],
                         "max_stage": max(row["stage"] for row in rows), "failure": rows[-1]["failure"], "deadline": rows[-1]["deadline"],
                         "elapsed_s": rows[-1]["elapsed_s"]}
                trial["native_replay_passed"] = verify_replay(Path(trained["bundle"]), trial)
                report["runs"].append(trial)
            finally:
                env.close()
                shadow.close()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
