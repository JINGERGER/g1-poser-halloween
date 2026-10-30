import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare import ROOT, SUPPORTS
from reference_gait import paired_cycles


def pulse(elapsed):
    def smooth(fraction):
        fraction = np.clip(fraction, 0, 1)
        return float(6 * fraction ** 5 - 15 * fraction ** 4 + 10 * fraction ** 3)

    if elapsed < 0.5:
        return 0.0
    if elapsed < 1.25:
        return smooth((elapsed - 0.5) / 0.75)
    if elapsed < 1.75:
        return 1.0
    return 1 - smooth((elapsed - 1.75) / 0.75)


def leg_offset(model, anchor):
    data = mujoco.MjData(model)
    data.qpos[:] = anchor
    mujoco.mj_forward(model, data)
    site = model.site("right_foot").id
    target = data.site_xpos[site].copy() + [0, 0, 0.02]
    joints = [int(joint) for joint in model.actuator_trnid[:, 0]
              if model.joint(int(joint)).name.startswith("right_")
              and any(part in model.joint(int(joint)).name for part in ("hip", "knee", "ankle"))]
    if len(joints) != 6:
        raise ValueError("Expected six right-leg joints")
    addresses = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    jacobian = np.zeros((3, model.nv))
    lower = np.maximum(model.jnt_range[joints, 0], anchor[addresses] - 0.15)
    upper = np.minimum(model.jnt_range[joints, 1], anchor[addresses] + 0.15)
    for iteration in range(40):
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacobian, None, site)
        active = jacobian[:, dofs]
        error = target - data.site_xpos[site]
        delta = active.T @ np.linalg.solve(active @ active.T + np.eye(3) * 1e-5, error)
        data.qpos[addresses] = np.clip(data.qpos[addresses] + np.clip(delta, -0.02, 0.02), lower, upper)
    mujoco.mj_forward(model, data)
    offset = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] - anchor[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    return offset, float(np.linalg.norm(target - data.site_xpos[site]))


def run(checkpoint, branch, output):
    import mlx.core as mx

    from direct_task import DirectEnv
    from evaluate_support import metrics
    from policy import load_policy

    env, policy = load_policy(checkpoint, 1, 2001)
    try:
        if type(env) is not DirectEnv or env.cfg.force_source != "analytic_pd" or env.cfg.speed_min != 0 or env.cfg.speed_max != 0:
            raise ValueError("Require the force-corrected zero-command support policy")
        model_path = Path(env.cfg.scene.model_file)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        state = env.init_state()
        rows, native_states, applied_controls = [], [], []
        anchor = None
        offset = np.zeros(23)
        maximum_error = 0.0
        clipped_count = 0
        terminated = False
        for step in range(650):
            before = env.get_physics_state_snapshot().copy()
            if step == 500:
                mujoco.mj_setState(model, data, before[0].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(model, data)
                offset, ik_error = leg_offset(model, data.qpos.copy())
                anchor = {"physical_state": before[0].tolist(), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                          "previous_ctrl": applied_controls[-1], "previous_action": state.info["current_actions"][0].tolist(),
                          "obs": state.obs["obs"][0].tolist(), "ik_offset_rad": offset.tolist(), "ik_error_m": ik_error,
                          "paired_state_sha256": hashlib.sha256(before.tobytes() + state.obs["obs"].tobytes()).hexdigest()}
                if ik_error > 0.001:
                    raise ValueError("Bounded IK probe cannot reach its 2cm site target")
                startup = rows[-50:]
                anchor["ready"] = all(all(load > 1 for load in row["loads_n"]) and row["other_load_n"] <= 1
                                      and row["penetration_m"] <= 0.005 and row["max_torque_ratio"] < 0.95 for row in startup)
                if not anchor["ready"]:
                    break
            actions = np.asarray(policy.policy(mx.array(state.obs["obs"])), dtype=np.float32)
            if step >= 500 and branch == "freeze":
                actions = np.array([anchor["previous_action"]], dtype=np.float32)
            elif step >= 500 and branch == "right_foot_probe":
                actions = actions + (offset * pulse((step - 500) * 0.02) / env.cfg.control_config.action_scale)[None, :]
            clipped_count += int(np.count_nonzero(np.abs(actions) > 1)) if step >= 500 else 0
            actions = np.clip(actions, -1, 1).astype(np.float32)
            targets = np.clip(env.default_angles + env.cfg.control_config.action_scale * actions, env.ctrl_range[:, 0], env.ctrl_range[:, 1])
            state = env.step(actions)
            mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, before[0].astype(float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = targets[0]
            for substep in range(env.cfg.sim_substeps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            done = bool(state.terminated[0] or state.truncated[0])
            if not done:
                snapshot = np.empty(before.shape[1])
                mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                maximum_error = max(maximum_error, float(np.max(np.abs(snapshot - env.get_physics_state_snapshot()[0]))))
                if maximum_error > 2e-4:
                    raise AssertionError("Native replay mismatch")
            row = metrics(model, data)
            row.update(elapsed_s=(step + 1) * 0.02 - 10, penetration_m=max(0, max((-float(contact.dist) for contact in data.contact), default=0)),
                       base_pos=data.qpos[:3].tolist())
            rows.append(row)
            native_states.append(data.qpos.copy())
            applied_controls.append(targets[0].tolist())
            if done:
                terminated = True
                break
        probe = rows[500:]
        protocol = json.loads(Path(__file__).with_name("reference_eval.json").read_text())
        cycles = paired_cycles(probe, protocol, 0.02) if probe else None
        summary = None
        if probe:
            loads = np.array([row["loads_n"] for row in probe])
            supported = loads.sum(axis=1) > 1
            slips = np.array([row["slip_mps"] for row in probe])
            summary = {"completed_3s": len(probe) == 150 and not terminated,
                       "cycles": cycles, "loaded_fraction": (loads > 1).mean(axis=0).tolist(),
                       "max_gaps_m": np.max([row["gaps_m"] for row in probe], axis=0).tolist(),
                       "non_support_fraction": float(np.mean([row["other_load_n"] > 1 for row in probe])),
                       "max_penetration_m": max(row["penetration_m"] for row in probe),
                       "max_torque_ratio": max(row["max_torque_ratio"] for row in probe),
                       "mean_loaded_slip_mps": float(np.mean((loads[supported] * slips[supported]).sum(axis=1) / loads[supported].sum(axis=1))) if supported.any() else None,
                       "max_base_xy_drift_m": max(float(np.linalg.norm(np.array(row["base_pos"][:2]) - np.array(anchor["qpos"][:2]))) for row in probe)}
        np.savez_compressed(output, qpos=np.array(native_states), ctrl=np.array(applied_controls))
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "branch": branch, "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "anchor": anchor, "summary": summary, "terminated": terminated, "clipped_action_components": clipped_count,
                "max_native_state_error": maximum_error, "trace": rows, "trajectory": str(output)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Paired support handoff and bounded unilateral probe; no training")
    parser.parse_args()
    checkpoints = [ROOT / f"output/rl/mlx_ppo/G1CrawlDirect23V2/{stamp}_mujoco/model_299.safetensors"
                   for stamp in ("2026-10-15_11-42-03", "2026-10-15_11-48-28")]
    output = ROOT / "output/rl/support_handoff" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0, "seed": 2001,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": "One fixed first episode per training seed; 10s warmup then 3s paired intervention. Readiness uses 1s of actual load/penetration/torque, not stability proof. Probe is a bounded leg-only IK residual on the existing feedback policy, NOT whole-body balance control or a feasible expert trajectory. 2cm SITE target is not a geometry-clearance claim. Physical state is never overwritten in the batch rollout; native diagnostic replay uses snapshots. Endpoint metrics at 20ms, not substep peaks. No model/gain/force-limit changes."}
    try:
        for index, checkpoint in enumerate(checkpoints):
            paired_hash = None
            for branch in ("continue", "freeze", "right_foot_probe"):
                result = run(checkpoint, branch, output / f"seed{index + 1}_{branch}.npz")
                report["runs"].append(result)
                if result["anchor"] is not None:
                    actual = result["anchor"]["paired_state_sha256"]
                    if paired_hash is not None and actual != paired_hash:
                        raise AssertionError("Branch initial states/observations differ")
                    paired_hash = actual
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"seed": index + 1, "branch": branch, "ready": result["anchor"]["ready"] if result["anchor"] else False,
                                  "summary": result["summary"]}), flush=True)
                if result["anchor"] is None or not result["anchor"]["ready"]:
                    break
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
