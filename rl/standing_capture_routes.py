import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from transition_execution import TransitionExecution


def integration_state(task):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(task.model, specification))
    mujoco.mj_getState(task.model, task.data, state, specification)
    return state


def fork_prefix(source, destination):
    if source.failure or source.phase != "transition":
        raise ValueError("Cannot certify a branch with an invalid prefix")
    destination.target = source.target.copy()
    destination.reset({"qpos": source.positions[0], "qvel": source.velocities[0]})
    destination.model.actuator_gainprm[:] = source.model.actuator_gainprm
    destination.model.actuator_biasprm[:] = source.model.actuator_biasprm
    mujoco.mj_copyData(destination.data, destination.model, source.data)
    for name in ("samples", "support_positions", "penetrations", "positions", "velocities", "controls", "phases"):
        setattr(destination, name, [np.array(value).copy() if isinstance(value, np.ndarray) else
                                   list(value) if isinstance(value, list) else value for value in getattr(source, name)])
    destination.execution = TransitionExecution(destination.model, destination.data, destination.reference,
                                                destination.constants, destination.policy, destination._observe_step)
    np.testing.assert_array_equal(integration_state(source), integration_state(destination))


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def route_target(task, family, elapsed, duration, spread, arm_lead):
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    prepared = task.target.copy()
    for index, name in enumerate(names):
        if name.endswith("shoulder_roll_joint"):
            prepared[index] = 1. if name.startswith("left") else -1.
        elif name.endswith("shoulder_pitch_joint"):
            prepared[index] = -.3
    if elapsed <= .6:
        return task.target + smooth(elapsed / .6) * (prepared - task.target)
    phase = (elapsed - .6) / duration
    goal = task.crawl_target.copy()
    middle = prepared.copy()
    for index, name in enumerate(names):
        sign = 1. if name.startswith("left") else -1.
        if name.endswith("hip_pitch_joint"):
            middle[index] = -1.2 if family == "hinge" else -.8
        elif name.endswith("knee_joint"):
            middle[index] = .9 if family == "hinge" else 1.6
        elif name.endswith("ankle_pitch_joint"):
            middle[index] = 0. if family == "hinge" else -.65
        elif name.endswith("hip_roll_joint"):
            middle[index] = sign * spread
        elif name.endswith("hip_yaw_joint"):
            middle[index] = sign * .3
    target = prepared.copy()
    for index, name in enumerate(names):
        local_phase = phase
        if family == "staggered" and name.startswith("right"):
            local_phase = (phase - .15) / .85
        if any(part in name for part in ("shoulder", "elbow", "wrist")):
            target[index] = prepared[index] + smooth((local_phase + arm_lead) / (1. + arm_lead)) * (goal[index] - prepared[index])
        elif local_phase < .5:
            target[index] = prepared[index] + smooth(local_phase * 2.) * (middle[index] - prepared[index])
        else:
            target[index] = middle[index] + smooth((local_phase - .5) * 2.) * (goal[index] - middle[index])
    return np.clip(target, task.model.actuator_ctrlrange[:, 0], task.model.actuator_ctrlrange[:, 1])


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    guard = PositionTorqueGuard(task.model, fraction=.8)
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    specifications = [{"family": family, "duration": duration, "spread": spread, "arm_lead": lead}
                      for family in ("hinge", "squat", "staggered") for duration in (.6, 1., 1.6)
                      for spread in (.3, .8) for lead in (0., .3)]
    output = ROOT / "output/rl/standing_capture_routes" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    prior_path = ROOT / "output/rl/handoff_envelope/2026-10-26_16-59-29-771057/report.json"
    prior = json.loads(prior_path.read_text())
    assert assets == prior["assets"] and digest(task.policy_path) == prior["policy_sha256"]
    assert digest(task.bundle_path) == prior["bundle_sha256"]
    report = {"status": "running", "specifications": specifications, "initial_state": state, "assets": assets,
              "prior_sha256": digest(prior_path), "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "stand_transition_task.py", "near_handoff_task.py",
                           "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "36 fixed true-standing routes, not reward optimization: hinge/squat/staggered, three durations, "
              "two hip spreads, two arm leads. Native PD, 600ms arm preparation, 80% predicted torque guard, "
              "all original strict physical gates. At 200ms intervals after preparation, and final endpoint, "
              "test actual2s frozen crawl on independent full MjData copies with complete standing prefix. "
              "No readiness/height/hand-gap prefilter. Branch cannot alter source trajectory. "
              "Exact full-trajectory replay for every branch and prefix. Offline oracle timing probes, NOT a learned "
              "causal switch or deployable policy. No reset stitching, external forces or model changes. "
              "Post-preparation duration+400ms maximum, at most13 probes per route (468 total). "
              "Nominal initial velocity only; success requires later perturbation validation.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for index, specification in enumerate(specifications):
            task.reset(state)
            case = {"index": index, **specification, "branches": []}
            total_steps = round((.6 + specification["duration"] + .4) / .02)
            for step in range(total_steps):
                try:
                    target = route_target(task, specification["family"], (step + 1) * .02,
                                          specification["duration"], specification["spread"], specification["arm_lead"])
                    guard.advance(task.execution, target)
                except (UnsafeState, InfeasiblePositionTarget) as error:
                    case["failure"] = task.failure or [type(error).__name__]
                    break
                if (step + 1 >= 30 and (step + 1) % 10 == 0) or step + 1 == total_steps:
                    before = integration_state(task)
                    fork_prefix(task, branch)
                    handoff_time = float(task.data.time)
                    result = branch.finish()
                    trajectory = output / f"route_{index}_step_{step + 1}.npz"
                    branch.save_and_replay(trajectory)
                    np.testing.assert_array_equal(integration_state(task), before)
                    case["branches"].append({"time_s": handoff_time, "height_m": float(task.data.qpos[2]),
                                             "handoff": result, "trajectory": str(trajectory.relative_to(ROOT)),
                                             "trajectory_sha256": digest(trajectory), "replay_passed": True,
                                             "peak_hand_load_n": float(np.max(np.asarray(branch.samples)[:, 4:6]))})
            trajectory = output / f"route_{index}_prefix.npz"
            task.save_and_replay(trajectory)
            case.update(success=any(entry["handoff"]["success"] for entry in case["branches"]),
                        duration_s=float(task.data.time), trajectory=str(trajectory.relative_to(ROOT)),
                        trajectory_sha256=digest(trajectory), replay_passed=True)
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(index, specification["family"], case["success"], case.get("failure"), len(case["branches"]), flush=True)
        report["summary"] = {"routes": len(report["cases"]), "successful_routes": sum(case["success"] for case in report["cases"]),
                             "probes": sum(len(case["branches"]) for case in report["cases"]),
                             "prefix_failures": dict(Counter(reason for case in report["cases"] for reason in case.get("failure", []))),
                             "handoff_failures": dict(Counter(reason for case in report["cases"] for branch_result in case["branches"]
                                                              for reason in branch_result["handoff"]["failure"]))}
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
