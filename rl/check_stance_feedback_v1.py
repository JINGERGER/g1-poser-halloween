import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import fork_prefix, smooth


def feedback_target(task, target, desired_pitch, gain, gravity):
    target = target.copy()
    rotation = task.data.xmat[task.model.body("pelvis").id].reshape(3, 3)
    pitch = np.arctan2(rotation[0, 2], rotation[2, 2])
    roll = np.arctan2(-rotation[1, 2], rotation[2, 2])
    world_angular = rotation @ task.data.qvel[3:6]
    if gravity:
        target += task.data.qfrc_bias[task.model.jnt_dofadr[task.joints]] / task.model.actuator_gainprm[:, 0]
    for index, joint in enumerate(task.joints):
        name = task.model.joint(int(joint)).name
        if name.endswith("ankle_pitch_joint"):
            target[index] += gain * (pitch - desired_pitch) + .15 * gain * world_angular[1]
        elif name.endswith("ankle_roll_joint"):
            target[index] += gain * roll + .15 * gain * world_angular[0]
    return np.clip(target, task.model.actuator_ctrlrange[:, 0], task.model.actuator_ctrlrange[:, 1])


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/stance_lowering_v1/2026-10-28_14-59-00-617321/report.json"
    original = json.loads(source.read_text())
    task, branch = StandTransitionTask(), StandTransitionTask()
    guard = PositionTorqueGuard(task.model, fraction=.8)
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    assert assets == original["assets"] and digest(task.policy_path) == original["policy_sha256"]
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/stance_feedback_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "assets": assets,
              "sources": {name: digest(ROOT / "rl" / name) for name in (Path(__file__).name, "standing_capture_routes.py",
                          "stand_transition_task.py", "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "Matched stance IK references (source indices8,16), durations1.5/3s. "
              "Control gain0/no feedforward, then ankle world-roll/pitch feedback gain .3/.8/1.5 with derivative .15*gain, "
              "crossed with generalized bias-force feedforward on/off. 28 trials fixed budget. Native PD, "
              "actual joint force limits, .8 torque guard and strict original gates. No base-state override. "
              "Independent copied-prefix handoff every200ms below .6m, actual2s crawl, not causal switch.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for reference_index in (8, 16):
            reference = original["cases"][reference_index]
            angles = np.array([node["qpos"] for node in reference["nodes"]])[:, task.addresses]
            for duration in (1.5, 3.):
                for gain, gravity in [(0., False), *[(gain, gravity) for gain in (.3, .8, 1.5) for gravity in (False, True)]]:
                    task.reset(state)
                    index = len(report["cases"])
                    case = {"reference_index": reference_index, "duration": duration, "gain": gain, "gravity": gravity, "probes": []}
                    for step in range(round((duration + .5) / .02)):
                        phase = smooth((step + 1) * .02 / duration)
                        coordinate = phase * (len(angles) - 1)
                        lower = min(int(coordinate), len(angles) - 2)
                        target = angles[lower] + (coordinate - lower) * (angles[lower + 1] - angles[lower])
                        target = feedback_target(task, target, reference["pitch"] * phase, gain, gravity)
                        try:
                            guard.advance(task.execution, target)
                        except (UnsafeState, InfeasiblePositionTarget) as error:
                            case["failure"] = task.failure or [type(error).__name__]
                            break
                        if step % 10 == 9 and task.data.qpos[2] < .6:
                            fork_prefix(task, branch)
                            result = branch.finish()
                            case["probes"].append({"time_s": float(task.data.time), **result})
                            if result["success"]:
                                trajectory = output / f"case_{index}_success.npz"
                                branch.save_and_replay(trajectory)
                                case.update(success_trajectory=str(trajectory.relative_to(ROOT)), success_trajectory_sha256=digest(trajectory))
                                break
                    trajectory = output / f"case_{index}.npz"
                    task.save_and_replay(trajectory)
                    valid = np.asarray(task.samples)[:-1] if task.failure else np.asarray(task.samples)
                    case.update(success=any(probe["success"] for probe in case["probes"]),
                                min_valid_hand_gap_m=float(valid[:, 8:10].mean(axis=1).min()), max_valid_hand_load_n=float(valid[:, 4:6].max()),
                                trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
                    report["cases"].append(case)
                    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(index, reference_index, duration, gain, gravity, case["success"], case.get("failure"),
                          case["min_valid_hand_gap_m"], case["max_valid_hand_load_n"], flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
