from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from omegaconf import OmegaConf

from short_step_assets import prepare
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest
from train_short_step import load_runner


ROOT = Path(__file__).resolve().parents[1]


def episode(model, bundle, stochastic, output):
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle)))
    env._autoreset = False
    rows, actions, controls = [], [], []
    try:
        state = env.init_state()
        positions, velocities = [env.datas[0].qpos.copy()], [env.datas[0].qvel.copy()]
        for unused in range(200):
            observation = mx.array(state.obs["obs"])
            if stochastic:
                sampled, log_prob, value, mean, std = model.act(observation)
                assert np.isfinite(np.asarray(log_prob)).all() and np.isfinite(np.asarray(value)).all()
            else:
                mean, std, unused_log_std = model.distribution_params(observation)
                sampled = model.policy(observation)
            action, mean, std = (np.asarray(values).copy() for values in (sampled, mean, std))
            assert all(np.isfinite(values).all() for values in (action, mean, std))
            data = env.datas[0]
            before = data.ctrl.copy()
            frame = min(round(env.elapsed[0] / .02) + 1, len(env.offsets) - 1)
            raw = env.initial_ctrl + env.offsets[frame] + env.protocol["residual_scale_rad"] * np.clip(action[0], -1, 1)
            bounded = np.clip(raw, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
            limit = env.protocol["max_ctrl_delta_rad"]
            expected = before + np.clip(bounded - before, -limit, limit)
            state = env.step(action)
            np.testing.assert_allclose(data.ctrl, expected, atol=1e-12, rtol=0)
            outcome = state.info["evaluation"][0]
            deepest = min(data.contact, key=lambda contact: contact.dist, default=None)
            rows.append({**outcome, "action_mean": mean[0].tolist(), "action_std": std[0].tolist(),
                         "action_clipped_fraction": float(np.mean(abs(action) > 1)),
                         "range_clipped_fraction": float(np.mean(abs(raw - bounded) > 1e-12)),
                         "rate_limited_fraction": float(np.mean(abs(bounded - before) > limit)),
                         "requested_delta_norm_rad": float(np.linalg.norm(bounded - before)),
                         "executed_delta_norm_rad": float(np.linalg.norm(data.ctrl - before)),
                         "deepest_contact": None if deepest is None else
                         [env.model.geom(int(geom)).name for geom in (deepest.geom1, deepest.geom2)]})
            actions.append(action[0])
            controls.append(data.ctrl.copy())
            positions.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            if state.terminated[0]:
                break
        assert state.terminated[0]
        np.savez_compressed(output, actions=actions, ctrl=controls, qpos=positions, qvel=velocities)
        env.init_state()
        for index, action in enumerate(actions):
            replay = env.step(action[None])
            np.testing.assert_allclose(env.datas[0].qpos, positions[index + 1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(env.datas[0].qvel, velocities[index + 1], atol=1e-10, rtol=0)
            assert replay.info["evaluation"][0] == {key: rows[index][key] for key in replay.info["evaluation"][0]}
        return {"stochastic": stochastic, "elapsed_s": rows[-1]["elapsed_s"],
                "failure": rows[-1]["failure"], "success": rows[-1]["success"],
                "max_stage": max(row["stage"] for row in rows), "trace": rows,
                "trajectory": str(output), "trajectory_sha256": digest(output), "native_replay_passed": True}
    finally:
        env.close()


def main():
    bundle = prepare()
    training_report = ROOT / "output/rl/short_step_training/2026-10-19_21-07-58-160217/report.json"
    training = json.loads(training_report.read_text())
    assert digest(training["config"]) == training["config_sha256"]
    assert digest(training["runner"]) == training["runner_sha256"]
    assert digest(bundle) == training["bundle_sha256"]
    config = OmegaConf.load(training["config"])
    assert not config.algo.empirical_normalization and not config.training.fp16
    runner = load_runner(Path(training["runner"]).parents[1])
    destination = ROOT / "output/rl/initial_actor_diagnostics" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "updates": 0, "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "config": training["config"], "config_sha256": training["config_sha256"],
              "runner": training["runner"], "runner_sha256": training["runner_sha256"],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "short_step_task.py", "component_unload_ramps.py")},
              "limitations": "Two freshly initialized actors, eight sequential stochastic episodes and one mean-action episode each. Same model factory, seed helper, float32 observations and model.act as training; not a bitwise reproduction of an eight-env collector RNG stream. No checkpoint loading, optimizer or updates. Nominal reset only; endpoint contacts do not identify every substep guard trigger.",
              "actors": [], "runs": []}
    report_path = destination / "report.json"
    try:
        for seed in (1, 2):
            config.algo.seed = config.training.seed = config.env.seed = seed
            runner.apply_configured_training_seed(config, torch_runtime=False, cuda=False, mlx_runtime=True)
            model = runner.build_model(config.algo, 148, 23, dtype=mx.float32)
            mx.eval(model.parameters())
            before = {name: np.asarray(value).copy() for name, value in tree_flatten(model.parameters())}
            checkpoint = destination / f"initial_seed_{seed}.safetensors"
            model.save_weights(str(checkpoint))
            for index in range(9):
                result = episode(model, bundle, index < 8, destination / f"seed_{seed}_episode_{index}.npz")
                report["runs"].append({"seed": seed, "episode": index, **result})
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            for name, value in tree_flatten(model.parameters()):
                np.testing.assert_array_equal(before[name], np.asarray(value))
            report["actors"].append({"seed": seed, "checkpoint": str(checkpoint),
                                     "sha256": digest(checkpoint), "parameters_unchanged": True})
        summary = {}
        for stochastic in (True, False):
            runs = [run for run in report["runs"] if run["stochastic"] == stochastic]
            summary["sampled" if stochastic else "mean"] = {
                "episodes": len(runs), "median_duration_s": float(np.median([run["elapsed_s"] for run in runs])),
                "duration_range_s": [min(run["elapsed_s"] for run in runs), max(run["elapsed_s"] for run in runs)],
                "reached_0_8s": sum(run["elapsed_s"] >= .8 for run in runs),
                "failure_counts": dict(Counter(run["failure"] or "deadline" for run in runs)),
                "max_stage": max(run["max_stage"] for run in runs),
                "mean_rate_limited_fraction": float(np.mean([np.mean([row["rate_limited_fraction"] for row in run["trace"]]) for run in runs]))}
        report.update(status="completed", summary=summary)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
