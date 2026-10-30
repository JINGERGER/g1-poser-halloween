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
from standing_capture_routes import fork_prefix, integration_state, route_target, smooth


def reaching_target(task, leg_target, goals):
    probe = mujoco.MjData(task.model)
    probe.qpos[:] = task.data.qpos
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    indices = np.array([index for index, name in enumerate(names)
                        if any(part in name for part in ("shoulder", "elbow", "wrist"))])
    addresses = task.addresses[indices]
    dofs = task.model.jnt_dofadr[task.joints[indices]]
    lower, upper = task.model.actuator_ctrlrange[indices].T
    for unused in range(8):
        mujoco.mj_forward(task.model, probe)
        jacobians = []
        for geom in task.supports[:2]:
            jacobian = np.zeros((3, task.model.nv))
            mujoco.mj_jac(task.model, probe, jacobian, None, probe.geom_xpos[geom], task.model.geom_bodyid[geom])
            jacobians.append(jacobian[:, dofs])
        jacobian = np.vstack(jacobians)
        error = (goals - probe.geom_xpos[task.supports[:2]]).ravel()
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .01 ** 2 * np.eye(6), error)
        probe.qpos[addresses] = np.clip(probe.qpos[addresses] + np.clip(delta, -.15, .15), lower, upper)
    target = leg_target.copy()
    target[indices] = probe.qpos[addresses]
    return target


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    guard = PositionTorqueGuard(task.model, fraction=.8)
    source = ROOT / "output/rl/standing_capture_routes/2026-10-26_18-05-47-469295/report.json"
    previous = json.loads(source.read_text())
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    assert assets == previous["assets"] and digest(task.policy_path) == previous["policy_sha256"]
    assert digest(task.bundle_path) == previous["bundle_sha256"]
    specifications = [{"family": family, "duration": duration, "reach_fraction": fraction}
                      for family in ("hinge", "squat", "staggered")
                      for duration in (.6, 1.6) for fraction in (.6, 1.)]
    output = ROOT / "output/rl/standing_cartesian_capture" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "assets": assets,
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "standing_capture_routes.py", "position_torque_guard.py",
                           "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "12 predeclared Cartesian-arm replacements for standing-route families. Same600ms arm preparation, "
              "hip spread .3, native PD, .8 predicted torque guard, strict original gates. Arms-only damped Jacobian IK "
              "at20ms,8 iterations, per-iteration delta<=.15rad, damping .01. Probe qpos is not physical robot state. "
              "World hand goals interpolate from actual post-preparation positions to pelvis x+.35,y+/-.32,z=.02. "
              "Reach finishes at .6 or1.0 of route duration. No collision-aware IK or reachability guarantee. "
              "Same200ms offline capture probes and full-prefix replay; not causal trained controller or hardware safety proof.",
              "initial_state": previous["initial_state"], "specifications": specifications, "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for index, specification in enumerate(specifications):
            task.reset(report["initial_state"])
            case = {"index": index, **specification, "branches": []}
            starting_hands = None
            total_steps = round((.6 + specification["duration"] + .4) / .02)
            for step in range(total_steps):
                try:
                    elapsed = (step + 1) * .02
                    target = route_target(task, specification["family"], elapsed, specification["duration"], .3, 0.)
                    if step >= 30:
                        if starting_hands is None:
                            starting_hands = task.data.geom_xpos[task.supports[:2]].copy()
                        goals = np.array([[task.data.qpos[0] + .35, task.data.qpos[1] + .32, .02],
                                          [task.data.qpos[0] + .35, task.data.qpos[1] - .32, .02]])
                        blend = smooth((elapsed - .6) / (specification["duration"] * specification["reach_fraction"]))
                        goals = starting_hands + blend * (goals - starting_hands)
                        before = integration_state(task)
                        target = reaching_target(task, target, goals)
                        np.testing.assert_array_equal(integration_state(task), before)
                    guard.advance(task.execution, target)
                except (UnsafeState, InfeasiblePositionTarget) as error:
                    case["failure"] = task.failure or [type(error).__name__]
                    break
                if (step + 1 >= 30 and (step + 1) % 10 == 0) or step + 1 == total_steps:
                    before = integration_state(task)
                    fork_prefix(task, branch)
                    result = branch.finish()
                    trajectory = output / f"route_{index}_step_{step + 1}.npz"
                    branch.save_and_replay(trajectory)
                    np.testing.assert_array_equal(integration_state(task), before)
                    case["branches"].append({"time_s": float(task.data.time), "handoff": result,
                                             "trajectory": str(trajectory.relative_to(ROOT)),
                                             "trajectory_sha256": digest(trajectory), "replay_passed": True})
            trajectory = output / f"route_{index}_prefix.npz"
            task.save_and_replay(trajectory)
            valid = np.asarray(task.samples)[:-1] if task.failure else np.asarray(task.samples)
            case.update(success=any(entry["handoff"]["success"] for entry in case["branches"]),
                        duration_s=float(task.data.time), min_valid_mean_hand_gap_m=float(valid[:, 8:10].mean(axis=1).min()),
                        max_valid_hand_load_n=float(valid[:, 4:6].max()), trajectory=str(trajectory.relative_to(ROOT)),
                        trajectory_sha256=digest(trajectory), replay_passed=True)
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(index, specification, case["success"], case.get("failure"), case["min_valid_mean_hand_gap_m"], flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
