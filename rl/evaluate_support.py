import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mujoco
import numpy as np

from policy import load_policy
from prepare import SUPPORTS
from direct_task import DirectEnv
from support_windows import summarize_windows
from gait_audit import summarize_gait


def metrics(model, data):
    floor = model.geom("floor").id
    support_ids = [model.geom(name).id for name in SUPPORTS]
    loads = np.zeros(4)
    contacts = np.zeros(4, dtype=bool)
    slip = np.zeros(4)
    other_load = 0.0
    other_contacts = []
    for index, contact in enumerate(data.contact):
        if floor not in (contact.geom1, contact.geom2):
            continue
        geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = max(0.0, float(wrench[0]))
        if geom not in support_ids:
            other_load += normal
            other_contacts.append(model.geom(geom).name)
            continue
        limb = support_ids.index(geom)
        contacts[limb] = True
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact.pos, int(model.geom_bodyid[geom]))
        loads[limb] += normal
        slip[limb] += normal * np.linalg.norm((jacobian @ data.qvel)[:2])
    joint_ids = model.actuator_trnid[:, 0]
    force = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
    limits = model.jnt_actfrcrange[joint_ids]
    ratios = np.abs(force) / np.where(force >= 0, limits[:, 1], -limits[:, 0])
    gaps = [float(mujoco.mj_geomDistance(model, data, geom, floor, 0.5, None)) for geom in support_ids]
    return {"time_s": float(data.time), "loads_n": loads.tolist(),
            "gaps_m": gaps, "floor_contacts": contacts.tolist(),
            "other_load_n": other_load, "other_contacts": other_contacts,
            "slip_mps": (slip / np.maximum(loads, 1e-12)).tolist(),
            "max_torque_ratio": float(ratios.max()),
            "most_loaded_joint": model.joint(int(joint_ids[np.argmax(ratios)])).name,
            "base_speed_mps": float(np.linalg.norm(data.qvel[:3]))}


def main():
    parser = argparse.ArgumentParser(description="Native replay of held-out support episodes including terminal transitions")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--moving", action="store_true", help="Audit positive-command DirectEnv crawling, not zero-command support")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env, policy = load_policy(args.checkpoint, args.episodes, args.seed)
    try:
        if type(env) is not DirectEnv or env.cfg.reference_motion:
            raise ValueError("This audit requires a non-reference DirectEnv task")
        if (args.moving and env.cfg.speed_min <= 0) or (not args.moving and (env.cfg.speed_min != 0 or env.cfg.speed_max != 0)):
            raise ValueError("Use --moving only for positive-command crawling; default requires zero-command support")
        model_path = Path(env.cfg.scene.model_file)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        state = env.init_state()
        active = np.ones(args.episodes, dtype=bool)
        survived = np.zeros(args.episodes, dtype=bool)
        rows = [[] for episode in range(args.episodes)]
        initial = env.get_physics_state_snapshot()
        origins = np.empty((args.episodes, 2))
        for episode in range(args.episodes):
            mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, np.asarray(initial[episode], dtype=np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            origins[episode] = data.qpos[:2]
        max_state_error = 0.0
        for step in range(env.cfg.max_episode_steps):
            before = env.get_physics_state_snapshot().copy()
            phases = (env.limb_phase(state.info) + env.cfg.ctrl_dt / env.cfg.phase_period) % 1.0
            commands = state.info["commands"].copy()
            actions = np.asarray(policy.policy(mx.array(state.obs["obs"])))
            targets = np.clip(env.default_angles + env.cfg.control_config.action_scale * np.clip(actions, -1, 1), env.ctrl_range[:, 0], env.ctrl_range[:, 1])
            state = env.step(actions)
            after = env.get_physics_state_snapshot()
            for episode in np.flatnonzero(active):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, np.asarray(before[episode], dtype=np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                data.ctrl[:] = targets[episode]
                for substep in range(env.cfg.sim_substeps):
                    mujoco.mj_step(model, data)
                done = state.terminated[episode] or state.truncated[episode]
                if not done:
                    snapshot = np.empty(before.shape[1])
                    mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    error = float(np.max(np.abs(snapshot - after[episode])))
                    max_state_error = max(max_state_error, error)
                    if error > 2e-4:
                        raise AssertionError("Native replay diverged from batch")
                mujoco.mj_forward(model, data)
                row = metrics(model, data)
                row["elapsed_s"] = (step + 1) * env.cfg.ctrl_dt
                row["drift_m"] = float(np.linalg.norm(data.qpos[:2] - origins[episode]))
                row["phase"] = phases[episode].tolist()
                row["command_mps"] = float(commands[episode, 0])
                row["progress_m"] = float(data.qpos[0] - origins[episode, 0])
                row["forward_speed_mps"] = float(data.qvel[0])
                rows[episode].append(row)
                if done:
                    survived[episode] = bool(state.truncated[episode] and not state.terminated[episode])
                    active[episode] = False
            if not active.any():
                break
        summaries = []
        for episode, trace in enumerate(rows):
            settled = [row for row in trace if row["time_s"] >= 1.0]
            loaded = np.array([row["loads_n"] for row in settled]) > 1.0
            summaries.append({"episode": episode, "survived": bool(survived[episode]),
                "duration_s": len(trace) * env.cfg.ctrl_dt,
                "limb_loaded_fraction_after_1s": loaded.mean(axis=0).tolist() if settled else None,
                "all_four_loaded_fraction_after_1s": float(loaded.all(axis=1).mean()) if settled else None,
                "max_non_support_load_n": max(row["other_load_n"] for row in trace),
                "non_support_contact_samples": sum(bool(row["other_contacts"]) for row in trace),
                "max_torque_ratio": max(row["max_torque_ratio"] for row in trace),
                "near_limit_fraction_after_1s": float(np.mean([row["max_torque_ratio"] >= 0.95 for row in settled])) if settled else None,
                "near_limit_joint_counts_after_1s": {name: sum(row["max_torque_ratio"] >= 0.95 and row["most_loaded_joint"] == name for row in settled) for name in sorted({row["most_loaded_joint"] for row in settled if row["max_torque_ratio"] >= 0.95})},
                "mean_loaded_contact_slip_mps_after_1s": float(np.mean([np.average(row["slip_mps"], weights=np.maximum(row["loads_n"], 1e-12)) for row in settled])) if settled else None,
                "max_base_drift_m": max(row["drift_m"] for row in trace),
                "time_windows": summarize_windows(trace)})
            if args.moving:
                summaries[-1]["gait"] = summarize_gait(trace)
        report = {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
                  "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  "force_source": env.cfg.force_source, "max_native_state_error": max_state_error,
                  "stage": "crawl" if args.moving else "support",
                  "episodes": summaries, "first_episode_trace": rows[0],
                  "limitations": "20ms endpoint sampling, not substep maxima; load threshold 1N; startup excluded only from settled-load/slip summaries. Native replay includes terminal transitions before batch auto-reset. No perturbation-recovery or locomotion claim; no pass label based solely on survival."}
        label = "crawl" if args.moving else "support"
        output = args.checkpoint.with_name(f"{args.checkpoint.stem}_{label}_audit_seed{args.seed}.json")
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "first_episode_trace"}, indent=2))
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
