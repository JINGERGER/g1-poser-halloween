import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry
from unilab.base.observations import flatten_obs_dict

from evaluate_support import metrics
from reference_gait import summarize_episode
from reference_model import build_reference_model
from reference_task import body_to_world, world_to_body


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(checkpoint, episodes, seed, protocol):
    config = OmegaConf.create(json.loads(checkpoint.with_name("run_config.json").read_text())["config"])
    if config.training.task_name != "G1CrawlReference23" or config.experiment.observation_protocol != "reference-inspired-75-81-v1":
        raise ValueError("Checkpoint is not the reference-inspired task")
    model_path = Path(config.env.scene.model_file)
    if sha256(model_path) != config.experiment.model_sha256:
        raise ValueError("Checkpoint model hash mismatch")
    overrides = OmegaConf.to_container(config.env, resolve=True)
    overrides.update(seed=seed, actor_noise=False, speed_min=protocol["command_mps"], speed_max=protocol["command_mps"],
                     yaw_rate_max=0.0, standing_fraction=0.0, max_episode_seconds=protocol["duration_s"],
                     reward_config=OmegaConf.to_container(config.reward, resolve=True))
    env = registry.make("G1CrawlReference23", num_envs=episodes, sim_backend="mujoco", env_cfg_override=overrides)
    try:
        policy = build_reference_model(config.algo, 156, 23)
        policy.load_weights(str(checkpoint), strict=True)
        policy.eval()
        return env, policy, config
    except BaseException:
        env.close()
        raise


def aggregate(summaries, protocol):
    values = {}
    for name in ("duration_s", "progress_m", "speed_mae_mps", "turn_mae_radps", "loaded_slip_mps",
                 "non_support_fraction", "max_non_support_load_n", "max_penetration_m", "saturation_fraction", "longest_saturation_s"):
        available = [row[name] for row in summaries if row[name] is not None]
        values[name] = {"available": len(available), "mean": float(np.mean(available)) if available else None,
                        "median": float(np.median(available)) if available else None,
                        "p95": float(np.percentile(available, 95)) if available else None}
    passed = sum(row["passed"] for row in summaries)
    return {"episodes": len(summaries), "survived": sum(row["survived"] for row in summaries), "passed": passed,
            "pass_fraction": passed / len(summaries),
            "screen_passed": len(summaries) >= protocol["episodes"] and passed / len(summaries) >= protocol["required_pass_fraction"],
            "failure_counts": {name: sum(not row["gates"][name] for row in summaries) for name in summaries[0]["gates"]},
            "total_cycles_per_limb": [sum(row["cycles"][limb]["count"] for row in summaries) for limb in range(4)],
            "distributions": values}


def main():
    parser = argparse.ArgumentParser(description="Fixed first-episode, phase-free reference controller evaluation with native physics replay")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--record-first", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("reference_eval.json")
    protocol = json.loads(protocol_path.read_text())
    count = protocol["episodes"] if args.episodes is None else args.episodes
    seed = protocol["seed"] if args.seed is None else args.seed
    if not 1 <= count <= 1000 or seed < 0:
        parser.error("Require 1–1000 episodes and a nonnegative seed")
    checkpoint = args.checkpoint.resolve()
    output = checkpoint.with_name(f"{checkpoint.stem}_reference_eval_seed{seed}_{count}episodes.json")
    env, policy, config = load_reference(checkpoint, count, seed, protocol)
    started = time.perf_counter()
    with ExitStack() as resources:
        resources.callback(env.close)
        model_path = Path(env.cfg.scene.model_file)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        state = env.init_state()
        initial = env.get_physics_state_snapshot().copy()
        initial_frames = state.info["reset_frame"].copy()
        initial_projection = state.info["reset_floor_projection"].copy()
        origins = env._backend.get_base_pos()[:, :2].copy()
        headings = body_to_world(env._backend.get_base_quat(), np.tile([0.0, 0.0, 1.0], (count, 1)))[:, :2]
        norms = np.linalg.norm(headings, axis=1)
        if np.any(norms < 0.1):
            raise ValueError("Initial crawl heading is undefined")
        headings /= norms[:, None]
        lateral = np.column_stack((-headings[:, 1], headings[:, 0]))
        active = np.ones(count, dtype=bool)
        survived = np.zeros(count, dtype=bool)
        traces = [[] for _ in range(count)]
        max_state_error = 0.0
        max_terminal_velocity_error = 0.0
        terminal_checks = 0
        video = None
        video_times = []
        renderer = None
        writer = None
        if args.record_first:
            import imageio.v2 as imageio

            video = output.with_suffix(".gif")
            model.vis.global_.offwidth = 640
            model.vis.global_.offheight = 480
            renderer = resources.enter_context(mujoco.Renderer(model, height=480, width=640))
            writer = resources.enter_context(imageio.get_writer(video, mode="I", duration=env.cfg.ctrl_dt * 5 * 1000, loop=0))
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.distance = 2.3
            camera.azimuth = 135
            camera.elevation = -22

            def capture(elapsed, final=False):
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera)
                pixels = renderer.render()
                writer.append_data(pixels)
                video_times.append(elapsed)
                if elapsed == 0 or final:
                    imageio.imwrite(output.with_name(output.stem + ("_final.png" if final else "_initial.png")), pixels)

            mujoco.mj_setState(model, data, initial[0].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            capture(0.0)
        for step in range(env.cfg.max_episode_steps):
            before = env.get_physics_state_snapshot().copy()
            commands = state.info["commands"].copy()
            actions = np.asarray(policy.policy(mx.array(flatten_obs_dict(state.obs))), dtype=np.float32)
            targets = env.action_targets(np.clip(actions, -5, 5))
            state = env.step(actions)
            after = env.get_physics_state_snapshot()
            for episode in np.flatnonzero(active):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, before[episode].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                data.ctrl[:] = targets[episode]
                for _ in range(env.cfg.sim_substeps):
                    mujoco.mj_step(model, data)
                done = bool(state.terminated[episode] or state.truncated[episode])
                if not done:
                    snapshot = np.empty(before.shape[1])
                    mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    error = float(np.max(np.abs(snapshot - after[episode])))
                    max_state_error = max(max_state_error, error)
                    if error > 2e-4:
                        raise AssertionError(f"Native replay diverged at episode {episode}, step {step}: {error}")
                else:
                    expected = np.concatenate((world_to_body(data.qpos[None, 3:7], data.qvel[None, :3])[0], data.qvel[3:6]))
                    error = float(np.max(np.abs(expected - state.final_observation["critic"][episode, :6])))
                    max_terminal_velocity_error = max(max_terminal_velocity_error, error)
                    terminal_checks += 1
                    if error > 2e-4:
                        raise AssertionError("Terminal observation differs from pre-reset native state")
                mujoco.mj_forward(model, data)
                row = metrics(model, data)
                elapsed = (step + 1) * env.cfg.ctrl_dt
                velocity = world_to_body(data.qpos[None, 3:7], data.qvel[None, :3])[0]
                angular = body_to_world(data.qpos[None, 3:7], data.qvel[None, 3:6])[0]
                delta = data.qpos[:2] - origins[episode]
                row.update(elapsed_s=elapsed, progress_m=float(delta @ headings[episode]), lateral_m=float(delta @ lateral[episode]),
                           command=commands[episode].tolist(), velocity_yz_mps=velocity[[2, 1]].tolist(),
                           speed_error_mps=float(np.linalg.norm(velocity[[2, 1]] - commands[episode, :2])),
                           turn_error_radps=float(abs(angular[2] - commands[episode, 2])),
                           penetration_m=max(0.0, max((-float(contact.dist) for contact in data.contact), default=0.0)))
                traces[episode].append(row)
                if episode == 0 and renderer is not None and ((step + 1) % 5 == 0 or done):
                    capture(elapsed, final=done)
                if done:
                    active[episode] = False
                    survived[episode] = bool(state.truncated[episode] and not state.terminated[episode])
            if (step + 1) % 250 == 0:
                print(f"step={step + 1} active_first_episodes={int(active.sum())}/{count}", flush=True)
            if not active.any():
                break
        if active.any() or terminal_checks != count:
            raise AssertionError("Not all first episodes reached a recorded terminal transition")
        summaries = []
        for episode, trace in enumerate(traces):
            summary = summarize_episode(trace, survived[episode], protocol, env.cfg.ctrl_dt)
            summary.update(episode=episode, reset_frame=int(initial_frames[episode, 0]),
                           reset_floor_projection_m=float(initial_projection[episode, 0]),
                           initial_state_sha256=hashlib.sha256(initial[episode].tobytes()).hexdigest())
            summaries.append(summary)
        report = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "model_sha256": sha256(model_path),
                  "run_config_sha256": sha256(checkpoint.with_name("run_config.json")),
                  "manifest_sha256": sha256(model_path.with_name("manifest.json")), "reset_frames_sha256": sha256(model_path.with_name("reset_frames.npy")),
                  "evaluation_protocol": protocol, "protocol_sha256": sha256(protocol_path),
                  "evaluator_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("evaluate_reference.py", "reference_gait.py", "evaluate_support.py")},
                  "seed": seed, "training_seed": int(config.algo.seed), "pd_variant": config.experiment.pd_variant,
                  "overrides": {"actor_noise": False, "commands": [protocol["command_mps"], 0, 0], "reset_noise": bool(env.cfg.reset_noise)},
                  "max_native_state_error": max_state_error, "max_terminal_velocity_error": max_terminal_velocity_error,
                  "terminal_transitions_checked": terminal_checks, "summary": aggregate(summaries, protocol),
                  "episodes": summaries, "first_episode_trace": traces[0], "first_episode_initial_state": initial[0].tolist(),
                  "video": str(video) if video else None, "video_frame_times_s": video_times,
                  "wall_seconds": time.perf_counter() - started,
                  "limitations": "20ms endpoints, not substep maxima. Fixed 0.1m/s straight command, clean actor observations, training reset distribution. Thresholds are frozen engineering screening criteria, not hardware safety limits. Passing one checkpoint is not multi-seed robustness or terrain qualification. Video is the exact first episode of this batch, never reselected."}
    report["wall_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": report["summary"], "wall_seconds": report["wall_seconds"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
