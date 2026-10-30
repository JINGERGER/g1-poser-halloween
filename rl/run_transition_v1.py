import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_standing_pitch_sign import balance_state
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth
from transition_v1 import DEFAULT, StandingTransitionV1


def forward_target(task, elapsed):
    if elapsed < 1.1:
        amount = smooth((elapsed - .5) / .6)
    elif elapsed < 2.6:
        amount = 1.
    else:
        amount = 1. - smooth((elapsed - 2.6) / .6)
    target = task.target.copy()
    rotation = task.data.xmat[task.model.body("pelvis").id].reshape(3, 3)
    pitch_rate = float((rotation @ task.data.qvel[3:6])[1])
    damping = np.clip(.03 * pitch_rate, -.03, .03)
    for index, joint in enumerate(task.joints):
        if task.model.joint(int(joint)).name.endswith("hip_pitch_joint"):
            target[index] += -.15 * amount + damping
    return target


def run(task, state, mode, output, name):
    task.reset(state)
    guard = PositionTorqueGuard(task.model, fraction=.8)
    controller = StandingTransitionV1(task, DEFAULT)
    balance = [balance_state(task)]
    handoff = None
    controller_failure = []
    for step in range(250 if mode == "forward" else 200):
        try:
            target = forward_target(task, (step + 1) * .02) if mode == "forward" else controller.target(task)
            guard.advance(task.execution, target)
        except (UnsafeState, InfeasiblePositionTarget) as error:
            controller_failure = task.failure or [type(error).__name__]
            break
        row = balance_state(task)
        balance.append(row)
        if mode == "full" and row["pitch_rad"] > 1.1 and task.data.qpos[2] < .4:
            handoff = task.finish()
            break
    trajectory = output / f"{name}.npz"
    task.save_and_replay(trajectory)
    if handoff is not None and not handoff["success"]:
        controller_failure = handoff["failure"] or ["crawl_acceptance_failed"]
    elif mode == "full" and handoff is None and not controller_failure:
        controller_failure = ["no_handoff_before_timeout"]
    completed = bool(len(balance) == 251 and not controller_failure)
    forward_window = [row for row in balance if 1.6 <= row["time_s"] <= 2.5]
    settled = [row for row in balance if row["time_s"] >= 4.5]
    stage_gates = {"completed_5s": completed,
                   "strict_physics": not controller_failure,
                   "forward_pitch_0_1_to_0_4": bool(forward_window and all(.1 < row["pitch_rad"] < .4 for row in forward_window)),
                   "height_above_0_65": bool(all(row["pelvis_xyz_m"][2] > .65 for row in balance)),
                   "returned_upright": bool(settled and all(abs(row["pitch_rad"]) < .08 for row in settled)),
                   "settled_linear_speed": bool(settled and all(np.linalg.norm(row["base_velocity"][:3]) < .1 for row in settled)),
                   "settled_angular_speed": bool(settled and all(np.linalg.norm(row["base_velocity"][3:]) < .15 for row in settled))}
    return {"mode": mode, "initial_state": state, "stage_gates": stage_gates,
            "forward_stage_passed": bool(mode == "forward" and all(stage_gates.values())),
            "full_transition_passed": bool(mode == "full" and handoff and handoff["success"]),
            "failure": controller_failure, "handoff": handoff, "balance": balance,
            "trajectory": str(trajectory.relative_to(ROOT)), "trajectory_sha256": digest(trajectory), "replay_passed": True}


def main():
    parser = argparse.ArgumentParser(description="Experimental simulator-only transition v1; full transition is NOT validated")
    parser.add_argument("--mode", choices=("forward", "full"), default="forward")
    parser.add_argument("--experimental", action="store_true", help="Explicit acknowledgement required for unvalidated full mode")
    parser.add_argument("--validate", action="store_true", help="Nominal plus eight fixed initial-velocity perturbations")
    args = parser.parse_args()
    if args.mode == "full" and not args.experimental:
        parser.error("Full transition is unvalidated; use --experimental only for simulator diagnosis")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    output = ROOT / "output/rl/transition_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    report = {"status": "running", "version": "0.1.0-experimental", "mode": args.mode,
              "full_transition_validated": False, "assets": assets, "policy_sha256": digest(task.policy_path),
              "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "check_standing_pitch_sign.py", "transition_v1.py", "standing_capture_routes.py",
                           "stand_transition_task.py", "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "environment": {"mujoco": mujoco.__version__, "numpy": np.__version__, "torch": torch.__version__},
              "protocol": "Forward mode: stand .5s, ramp paired hip pitch -.15rad over.6s, hold1.5s, return over.6s, settle to5s. "
              "Forward mode adds paired hip offset .03*world pitch rate, bounded+/- .03rad, for damping; "
              "arms remain in original standing posture. Full mode: default forward-event/fold controller, "
              "causal one-time handoff at pitch>1.1,height<.4; no offline oracle. Native PD, .8 target torque guard, "
              "unchanged strict physics gates, full-prefix replay. Forward-stage success is NOT stand-to-crawl success. "
              "Simulation stops on failure, not a hardware emergency-stop design.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        seeds = [-1, *range(28000, 28008)] if args.validate else [-1]
        for seed in seeds:
            state = task.make_states([max(seed, 0)])[0]
            if seed == -1:
                state["qvel"] = np.zeros(task.model.nv).tolist()
            case = run(task, state, args.mode, output, f"seed_{seed}")
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(seed, case["forward_stage_passed"], case["full_transition_passed"], case["failure"], case["stage_gates"], flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
