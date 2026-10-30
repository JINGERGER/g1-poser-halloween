import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mujoco
import numpy as np
from omegaconf import OmegaConf
from unilab.base import registry

import ik_task
from evaluate_support import metrics
from ik_model import build_ik_model
from reference_gait import summarize_episode


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(checkpoint, episodes, seed):
    payload = json.loads(checkpoint.with_name("run_config.json").read_text())
    config = OmegaConf.create(payload["config"])
    if config.training.task_name != "G1IKTracking23":
        raise ValueError("Not an IK tracking checkpoint")
    for name, digest in config.experiment.source_sha256.items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Training source changed: {name}")
    assets = Path(config.env.scene.model_file).parent
    for name, digest in config.experiment.asset_sha256.items():
        if sha256(assets / name) != digest:
            raise ValueError(f"Snapshot changed: {name}")
    overrides = OmegaConf.to_container(config.env, resolve=True)
    overrides["reward_config"] = OmegaConf.to_container(config.reward, resolve=True)
    overrides["seed"] = seed
    env = registry.make("G1IKTracking23", num_envs=episodes, sim_backend="mujoco", env_cfg_override=overrides)
    try:
        policy = build_ik_model(config.algo, 178, 23)
        policy.load_weights(str(checkpoint), strict=True)
        policy.eval()
        return env, policy, config
    except BaseException:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(description="Native physical assessment of first episodes, including zero-residual and held-support controls")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--controller", choices=("learned", "zero", "hold"), default="learned")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1000:
        parser.error("Require 1–1000 episodes")
    protocol_path = Path(__file__).with_name("ik_eval.json")
    protocol = json.loads(protocol_path.read_text())
    env, policy, config = load(args.checkpoint, args.episodes, protocol["seed"])
    try:
        if args.controller == "hold":
            env.targets = lambda info, actions, advance=False: np.clip(env.anchor_ctrl[info["anchor_id"][:, 0]], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
        state = env.init_state()
        initial = env.get_physics_state_snapshot().copy()
        anchor_ids = state.info["anchor_id"][:, 0].copy()
        headings = env.heading[anchor_ids].copy()
        origins = env._backend.get_base_pos()[:, :2].copy()
        lateral = np.column_stack((-headings[:, 1], headings[:, 0]))
        model = mujoco.MjModel.from_xml_path(env.cfg.scene.model_file)
        data = mujoco.MjData(model)
        traces = [[] for episode in range(args.episodes)]
        active = np.ones(args.episodes, dtype=bool)
        survived = np.zeros(args.episodes, dtype=bool)
        maximum_error = 0.0
        for step in range(env.cfg.max_episode_steps):
            before = env.get_physics_state_snapshot().copy()
            actions = np.asarray(policy.policy(mx.array(state.obs["obs"])), dtype=np.float32) if args.controller == "learned" else np.zeros((args.episodes, 23), dtype=np.float32)
            targets = env.targets(state.info, actions, advance=True)
            state = env.step(actions)
            after = env.get_physics_state_snapshot().copy()
            for episode in np.flatnonzero(active):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, before[episode].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                data.ctrl[:] = targets[episode]
                for substep in range(env.cfg.sim_substeps):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                done = state.terminated[episode] or state.truncated[episode]
                if not done:
                    snapshot = np.empty(before.shape[1])
                    mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    maximum_error = max(maximum_error, float(np.max(np.abs(snapshot - after[episode]))))
                    if maximum_error > 2e-4:
                        raise AssertionError("Native/batch mismatch")
                row = metrics(model, data)
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                angular = rotation.reshape(3, 3) @ data.qvel[3:6]
                offset = data.qpos[:2] - origins[episode]
                error = data.qvel[:2] - headings[episode] * protocol["command_mps"]
                row.update(elapsed_s=(step + 1) * env.cfg.ctrl_dt,
                           progress_m=float(offset @ headings[episode]), lateral_m=float(offset @ lateral[episode]),
                           speed_error_mps=float(np.linalg.norm(error)), turn_error_radps=abs(float(angular[2])),
                           penetration_m=max(0.0, max((-float(contact.dist) for contact in data.contact), default=0)))
                traces[episode].append(row)
                if done:
                    active[episode] = False
                    survived[episode] = bool(state.truncated[episode] and not state.terminated[episode])
            if not active.any():
                break
        episodes = []
        for episode, trace in enumerate(traces):
            summary = summarize_episode(trace, survived[episode], protocol, env.cfg.ctrl_dt)
            summary.update(episode=episode, anchor_id=int(anchor_ids[episode]), initial_state_sha256=hashlib.sha256(initial[episode].tobytes()).hexdigest())
            episodes.append(summary)
        summary = {"episodes": len(episodes), "survived": int(survived.sum()),
                   "passed": sum(row["passed"] for row in episodes),
                   "mean_duration_s": float(np.mean([row["duration_s"] for row in episodes])),
                   "mean_progress_m": float(np.mean([row["progress_m"] for row in episodes])),
                   "total_cycles_by_limb": [sum(row["cycles"][limb]["count"] for row in episodes) for limb in range(4)]}
        report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                  "config_sha256": sha256(args.checkpoint.with_name("run_config.json")), "controller": args.controller,
                  "evaluation_protocol": protocol, "protocol_sha256": sha256(protocol_path), "evaluator_sha256": sha256(__file__),
                  "summary": summary, "episodes": episodes, "first_episode_trace": traces[0],
                  "max_native_state_error": maximum_error,
                  "screen_passed": len(episodes) >= 100 and summary["passed"] / len(episodes) >= protocol["required_pass_fraction"],
                  "limitations": "Fixed first episodes, no cherry-picking, native terminal steps included. 20ms endpoint metrics, not substep maxima. Two loaded reset anchors and horizontal translations only; no robustness/hardware claim. Hold/zero controls share checkpoint configuration but do not use its learned actions."}
        output = args.checkpoint.with_name(f"{args.checkpoint.stem}_ik_{args.controller}_{args.episodes}episodes.json")
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
        print(output, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
