import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth


def balance_state(task):
    model, data = task.model, task.data
    pelvis = model.body("pelvis").id
    rotation = data.xmat[pelvis].reshape(3, 3)
    com = data.subtree_com[pelvis].copy()
    center = np.zeros(3)
    total = 0.
    force_world = np.zeros(3)
    for index, contact in enumerate(data.contact):
        if task.floor not in (contact.geom1, contact.geom2):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        weight = max(0., float(wrench[0]))
        center += weight * contact.pos
        total += weight
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        force_world += force if contact.geom1 == task.floor else -force
    return {"time_s": float(data.time), "pitch_rad": float(np.arctan2(rotation[0, 2], rotation[2, 2])),
            "pelvis_xyz_m": data.qpos[:3].tolist(), "base_velocity": data.qvel[:6].tolist(),
            "com_xyz_m": com.tolist(), "normal_weighted_contact_center_m": (center / total).tolist() if total else None,
            "ground_force_world_n": force_world.tolist()}


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    guard = PositionTorqueGuard(task.model, fraction=.8)
    output = ROOT / "output/rl/standing_pitch_sign" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"script_sha256": digest(Path(__file__)), "cases": [], "protocol": "Nominal true standing, native PD, strict gates. Hold .5s, ramp paired joint offset over .4s, hold to1.5s. No posture/state overrides."}
    for suffix in ("hold", "ankle_pitch_joint", "hip_pitch_joint", "knee_joint"):
        for offset in ((0.,) if suffix == "hold" else (-.15, .15)):
            task.reset(state)
            rows = [balance_state(task)]
            for step in range(75):
                target = task.target.copy()
                for index, name in enumerate(names):
                    if name.endswith(suffix):
                        target[index] += offset * smooth(((step + 1) * .02 - .5) / .4)
                try:
                    guard.advance(task.execution, target)
                except (UnsafeState, InfeasiblePositionTarget):
                    break
                rows.append(balance_state(task))
            trajectory = output / f"{suffix}_{offset}.npz"
            task.save_and_replay(trajectory)
            report["cases"].append({"joint": suffix, "offset_rad": offset, "failure": task.failure,
                                    "rows": rows, "trajectory": str(trajectory.relative_to(ROOT)), "sha256": digest(trajectory)})
            print(suffix, offset, task.failure, rows[-1], flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    main()
