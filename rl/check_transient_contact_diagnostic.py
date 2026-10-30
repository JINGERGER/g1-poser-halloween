from datetime import datetime
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from omegaconf import OmegaConf

from short_step_task import ShortStepCfg
from target_load_ablation import digest
from train_short_step import load_runner
from transient_contact_diagnostic import TransientContactDiagnosticEnv, diagnostic_guard


ROOT = Path(__file__).resolve().parents[1]


def rollout(bundle, model, strict_trial, output):
    env = TransientContactDiagnosticEnv(ShortStepCfg(bundle=bundle))
    env._autoreset = False
    rows, actions, positions, velocities = [], [], [], []
    try:
        state = env.init_state()
        metric = dict(env.metrics[0])
        metric.update(other_total_n=2., other_loads_n={"left_forearm": 2.})
        assert diagnostic_guard(env.model, env.datas[0], metric, .0049, 0., .002)[0] is None
        assert diagnostic_guard(env.model, env.datas[0], metric, .0049, .08, .002)[0] == "shell_contact_over_80ms"
        assert diagnostic_guard(env.model, env.datas[0], metric, .006, 0., .002)[0] == "penetration_over_5mm"
        metric.update(loads_n=[90., 0., 35., 130.], floor_contacts=[True, False, True, True])
        assert diagnostic_guard(env.model, env.datas[0], metric, .0049, 0., .002)[0] == "remaining_support_lost"
        positions.append(env.datas[0].qpos.copy())
        velocities.append(env.datas[0].qvel.copy())
        for unused in range(200):
            action = np.asarray(model.policy(mx.array(state.obs["obs"]))).copy()
            state = env.step(action)
            actions.append(action[0])
            rows.append(state.info["evaluation"][0])
            positions.append(env.datas[0].qpos.copy())
            velocities.append(env.datas[0].qvel.copy())
            if state.terminated[0]:
                break
        assert state.terminated[0]
        with np.load(strict_trial["trajectory"], allow_pickle=False) as original:
            prefix = len(original["actions"]) - 1
            np.testing.assert_allclose(positions[:prefix + 1], original["qpos"][:prefix + 1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(actions[:prefix], original["actions"][:prefix], atol=1e-7, rtol=0)
        np.savez_compressed(output, qpos=positions, qvel=velocities, actions=actions)
        env.init_state()
        for index, action in enumerate(actions):
            state = env.step(action[None])
            np.testing.assert_allclose(env.datas[0].qpos, positions[index + 1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(env.datas[0].qvel, velocities[index + 1], atol=1e-10, rtol=0)
            assert state.info["evaluation"][0] == rows[index]
            assert bool(state.terminated[0]) == (index == len(actions) - 1)
        return {"elapsed_s": rows[-1]["elapsed_s"], "failure": rows[-1]["failure"],
                "max_stage": max(row["stage"] for row in rows), "training_rule_success": rows[-1]["success"],
                "strict_success": False, "min_left_load_n": min(row["loads_n"][0] for row in rows),
                "max_left_gap_m": max(row["gaps_m"][0] for row in rows),
                "trajectory": str(output), "trajectory_sha256": digest(output),
                "strict_prefix_matched": True, "native_replay_passed": True, "trace": rows}
    finally:
        env.close()


def main():
    source_path = ROOT / "output/rl/unload_curve/2026-10-19_21-59-26-517322/report.json"
    source = json.loads(source_path.read_text())
    output = ROOT / "output/rl/transient_contact_diagnostic" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_report": str(source_path), "source_sha256": digest(source_path),
              "training_updates": 0, "protocol": {"allowed_geom": "left_forearm", "max_load_n": 5., "max_consecutive_contact_s": .08},
              "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                              (Path(__file__).name, "transient_contact_diagnostic.py", "short_step_task.py", "component_unload_ramps.py")},
              "limitations": "Closed-loop counterfactual with frozen policies, not retraining or hardware safety calibration. No observation changes; diagnostic contact timer is not a recommended unobserved training state. Legacy strict failures remain failures.", "runs": []}
    path = output / "report.json"
    try:
        for entry in source["runs"]:
            evidence = entry["evaluations"][-1]["report"]
            assert digest(evidence["path"]) == evidence["sha256"]
            evaluation = json.loads(Path(evidence["path"]).read_text())
            snapshot = json.loads(Path(evaluation["training_report"]).read_text())
            assert digest(evaluation["training_report"]) == evaluation["training_report_sha256"]
            for key in ("config", "checkpoint", "bundle", "runner"):
                assert digest(snapshot[key]) == snapshot[key + "_sha256"]
            runner = load_runner(Path(snapshot["runner"]).parents[1])
            config = OmegaConf.load(snapshot["config"])
            model = runner.build_model(config.algo, 148, 23)
            model.load_weights(snapshot["checkpoint"], strict=True)
            model.eval()
            before = {name: np.asarray(value).copy() for name, value in tree_flatten(model.parameters())}
            strict_trial = next(run for run in evaluation["runs"] if run["label"] == "ppo_deterministic")
            assert strict_trial["failure"] == "non_support_load_over_1N"
            assert digest(strict_trial["trajectory"]) == strict_trial["trajectory_sha256"]
            result = rollout(snapshot["bundle"], model, strict_trial, output / f"seed_{entry['seed']}.npz")
            for name, value in tree_flatten(model.parameters()):
                np.testing.assert_array_equal(before[name], np.asarray(value))
            report["runs"].append({"seed": entry["seed"], "strict_evaluation": evidence, "parameters_unchanged": True, **result})
            print(json.dumps({key: value for key, value in report["runs"][-1].items() if key != "trace"}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
