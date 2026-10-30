from pathlib import Path
import argparse
import json

import mlx.core as mx
import mujoco
import numpy as np

from diagnose_lift import triangle_margin
from policy import load_policy
from prepare import SUPPORTS
from single_foot import SingleFootEnv


def main():
    parser = argparse.ArgumentParser(description="Audit the first single-foot policy rollout using a read-only physics reconstruction")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    env, policy = load_policy(args.checkpoint, 1, args.seed)
    try:
        if not isinstance(env, SingleFootEnv):
            raise ValueError("Support audit currently requires a single-foot task")
        model = mujoco.MjModel.from_xml_path(env.cfg.scene.model_file)
        data = mujoco.MjData(model)
        stance = [index for index in range(4) if index != env.cfg.target_foot]
        geom_ids = [model.geom(name).id for name in SUPPORTS]
        floor = model.geom("floor").id
        joint_ids = model.actuator_trnid[:, 0]
        force_ranges = model.jnt_actfrcrange[joint_ids]
        state = env.init_state()
        data.ctrl[:] = env.default_angles
        rows = []
        anchors = None
        start_xy = None
        ended = "step budget"
        for step in range(env.cfg.max_episode_steps):
            mujoco.mj_setState(model, data, np.asarray(env.get_physics_state_snapshot()[0], dtype=np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            points = data.geom_xpos[geom_ids, :2].copy()
            if anchors is None:
                anchors = points.copy()
                start_xy = data.qpos[:2].copy()
            normal = np.zeros(4)
            weighted_points = np.zeros((4, 2))
            wrench = np.zeros(6)
            for index, contact in enumerate(data.contact):
                if floor not in (contact.geom1, contact.geom2):
                    continue
                mujoco.mj_contactForce(model, data, index, wrench)
                for limb, geom in enumerate(geom_ids):
                    if geom in (contact.geom1, contact.geom2):
                        force = max(0.0, float(wrench[0]))
                        normal[limb] += force
                        weighted_points[limb] += force * contact.pos[:2]
            loaded = normal > 1.0
            points[loaded] = weighted_points[loaded] / normal[loaded, None]
            torque = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
            ratios = np.abs(torque) / np.where(torque >= 0, force_ranges[:, 1], -force_ranges[:, 0])
            phase = float(state.info["phase"][0, 0])
            rows.append({
                "time": float(data.time), "phase": phase,
                "peak": 0.825 < phase < 0.925,
                "support_loads_n": normal.tolist(),
                "stance_loaded": bool(loaded[stance].all()),
                "com_triangle_margin_m": triangle_margin(data.subtree_com[model.body("pelvis").id, :2], points[stance]),
                "stance_geom_center_drift_m": np.linalg.norm(data.geom_xpos[geom_ids, :2][stance] - anchors[stance], axis=1).tolist(),
                "base_drift_m": float(np.linalg.norm(data.qpos[:2] - start_xy)),
                "torque_ratios": ratios.tolist(),
                "streamed_torque_utilization": state.info.get("log", {}).get("crawl/torque_utilization"),
                "reconstructed_torque_utilization": float(np.mean(ratios ** 2)),
                "target_gap_m": float(data.sensor(f"balanced_gap_{env.cfg.target_foot}").data[0]),
            })
            actions = np.asarray(policy.policy(mx.array(state.obs["obs"])))
            data.ctrl[:] = np.clip(env.default_angles + env.cfg.control_config.action_scale * np.clip(actions[0], -1, 1), env.ctrl_range[:, 0], env.ctrl_range[:, 1])
            state = env.step(actions)
            if state.terminated[0] or state.truncated[0]:
                ended = "terminated" if state.terminated[0] else "timeout"
                break
        peak = [row for row in rows if row["peak"] and row["stance_loaded"]]
        ratios = np.array([row["torque_ratios"] for row in rows])
        report = {
            "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
            "target_foot": env.cfg.target_foot, "samples": len(rows), "ended": ended,
            "peak_loaded_samples": len(peak),
            "peak_negative_com_margin_fraction": float(np.mean([row["com_triangle_margin_m"] < 0 for row in peak])) if peak else None,
            "peak_mean_com_margin_m": float(np.mean([row["com_triangle_margin_m"] for row in peak])) if peak else None,
            "max_base_drift_m": max(row["base_drift_m"] for row in rows),
            "max_stance_geom_center_drift_m": float(np.max([row["stance_geom_center_drift_m"] for row in rows])),
            "mean_torque_utilization_difference": float(np.mean([abs(row["streamed_torque_utilization"] - row["reconstructed_torque_utilization"]) for row in rows if row["streamed_torque_utilization"] is not None])),
            "joint_near_limit_fraction": {model.joint(int(joint)).name: float(np.mean(ratios[:, index] >= 0.95)) for index, joint in enumerate(joint_ids)},
            "note": "First rollout only. Contact forces are recomputed from FULLPHYSICS snapshots and known direct position commands, not streamed force telemetry. COM margin is quasi-static, uses contact-force centroids, and is not a dynamic stability proof. Terminal autoreset snapshots are excluded. Geom-center drift also includes foot rotation, not just sliding.",
            "trace": rows,
        }
        output = args.checkpoint.with_name(f"{args.checkpoint.stem}_support_audit_seed{args.seed}.json")
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2))
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
