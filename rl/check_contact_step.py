import argparse
import json
from datetime import datetime

import numpy as np
import torch

from artifact_utils import digest
from contact_step_controller import ContactStepController
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth
from whole_body_torque_bridge_v1 import TorquePositionBridge


def rotation_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])


def run(task, state, output, spread, yaw, controller_class=ContactStepController):
    task.reset(state)
    guard = PositionTorqueGuard(task.model)
    bridge = TorquePositionBridge(task.model)
    case = {"spread_m": spread, "yaw_rad": yaw, "steps": [], "events": []}
    try:
        for unused in range(25):
            guard.advance(task.execution, task.target)
        controller = controller_class(task)
        feet = task.data.geom_xpos[task.supports[2:]].copy()
        landing_heights = feet[:, 2].copy()
        rotations = task.data.geom_xmat[task.supports[2:]].reshape(2, 3, 3).copy()
        hands = task.data.geom_xpos[task.supports[:2]].copy()
        posture = task.data.qpos[task.addresses].copy()
        mode, active, mode_start, confirmed = "shift", 0, float(task.data.time), 0
        com_start = controller.initial_com.copy()
        foot_start, rotation_start = feet.copy(), rotations.copy()
        for step in range(1500):
            row = task.measure()[0]
            elapsed = task.data.time - mode_start
            com_goal = com_start.copy()
            stance = 1 - active
            sign = 1. if active == 0 else -1.
            stance_goal = feet[stance, :2].copy()
            stance_goal[1] += sign * .02
            com_goal[:2] += smooth(elapsed / 3.) * (stance_goal - com_start[:2]) if mode == "shift" else 0.
            com_goal[2] = controller.initial_com[2] - .10 * smooth(task.data.time / 3.)
            swing = None
            if mode == "shift":
                tracking = np.linalg.norm(task.data.subtree_com[controller.pelvis, :2] - com_goal[:2]) < .02
                if tracking and row[6 + stance] > controller.transfer.weight * .85:
                    feet[active, 2] = landing_heights[active] + .015 * smooth((elapsed - 3.) / 2.)
                unloaded = row[6 + active] < 5. and row[6 + stance] > controller.transfer.weight * .8
                confirmed = confirmed + 1 if unloaded else 0
                if confirmed >= 3:
                    mode, mode_start = "swing", float(task.data.time)
                    com_start = com_goal.copy()
                    foot_start, rotation_start = feet.copy(), rotations.copy()
                    foot_start[active] = task.data.geom_xpos[task.supports[2 + active]]
                    case["events"].append({"time_s": mode_start, "event": "unloaded", "foot": active, "loads_n": list(row[4:8])})
            elif mode == "swing":
                phase = smooth(elapsed / 2.)
                feet[active] = foot_start[active] + [0., sign * spread * phase, .04 * np.sin(np.pi * phase)]
                feet[active, 2] += phase * (landing_heights[active] - foot_start[active, 2])
                rotations[active] = rotation_z(sign * yaw * phase) @ rotation_start[active]
                swing = task.supports[2 + active]
                if elapsed >= 2.:
                    mode, mode_start, confirmed = "land", float(task.data.time), 0
            elif mode == "land":
                feet[active, 2] = landing_heights[active] - min(.004, elapsed * .005)
                position_error = float(np.linalg.norm(task.data.geom_xpos[task.supports[2 + active]] - feet[active]))
                rotation = task.data.geom_xmat[task.supports[2 + active]].reshape(3, 3)
                angle_error = float(np.arccos(np.clip((np.trace(rotations[active].T @ rotation) - 1.) / 2., -1., 1.)))
                landed = row[6 + active] > 20. and position_error < .01 and angle_error < .05
                confirmed = confirmed + 1 if landed else 0
                if confirmed >= 3:
                    case["events"].append({"time_s": float(task.data.time), "event": "landed", "foot": active, "loads_n": list(row[4:8]),
                                           "position_error_m": position_error, "angle_error_rad": angle_error})
                    if active == 1:
                        case["both_steps_completed"] = True
                        break
                    active, mode, mode_start = 1, "shift", float(task.data.time)
                    com_start = task.data.subtree_com[controller.pelvis].copy()
                    continue
            load_targets = np.zeros(4)
            active_fraction = .5 * (1. - smooth(elapsed / 3.)) if mode == "shift" else (.1 if mode == "land" else 0.)
            load_targets[2 + active] = controller.transfer.weight * active_fraction
            load_targets[2 + stance] = controller.transfer.weight * (1. - active_fraction)
            controller.target(com_goal, 0., hands, posture, feet, rotations, swing, load_targets)
            record = {"time_s": float(task.data.time), "mode": mode, "foot": active, "loads_n": list(row[4:8]),
                      "com_goal": com_goal.tolist(), "com_actual": task.data.subtree_com[controller.pelvis].tolist(),
                      "feet_actual": task.data.geom_xpos[task.supports[2:]].tolist(), "feet_goal": feet.tolist(), **controller.last}
            case["steps"].append(record)
            bridge.advance_torque(task.execution, controller.last["torque_nm"])
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        case["failure"] = task.failure or [str(error)]
    case.update(duration_s=float(task.data.time), physics_passed=not bool(task.failure))
    trajectory = output / f"step_{spread}_{yaw}.npz"
    task.save_and_replay(trajectory)
    case.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictive", action="store_true")
    args = parser.parse_args()
    controller_class = ContactStepController
    if args.predictive:
        from predictive_contact_controller import PredictiveContactController
        controller_class = PredictiveContactController
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/contact_step" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "controller": controller_class.__name__, "sources": {name: digest(ROOT / "rl" / name) for name in
              ("contact_step_controller.py", "check_contact_step.py", "internal_constraint_force.py", "predictive_contact_controller.py")},
              "protocol": "Contact-gated shift/unload/swing/land. Remove planned swing foot from QP only after three measured unloading samples. Small-step control then wide-turn goal. Strict physical gates; no standing-to-crawl success implied.", "cases": []}
    try:
        for spread, yaw in ((.04, .2), (.26, 1.4)):
            case = run(task, state, output, spread, yaw, controller_class)
            report["cases"].append(case)
            print(spread, yaw, case.get("failure"), case["events"], case.get("both_steps_completed", False), flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
