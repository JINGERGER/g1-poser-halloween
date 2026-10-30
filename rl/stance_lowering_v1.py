import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import fork_prefix, smooth


def plan(task, state, height, pitch, advance):
    probe = mujoco.MjData(task.model)
    initial = np.array(state["qpos"])
    probe.qpos[:] = initial
    mujoco.mj_forward(task.model, probe)
    supports = probe.geom_xpos[task.supports].copy()
    foot_rotations = probe.geom_xmat[task.supports[2:]].copy()
    angles = initial[task.addresses].copy()
    nodes = [{"qpos": initial.tolist(), "support_error_m": 0., "self_penetration_m": 0.}]
    for phase in np.linspace(0., 1., 16)[1:]:
        base = initial[:7].copy()
        base[0] += advance * phase
        base[2] += (height - initial[2]) * phase
        base[3:7] = [np.cos(pitch * phase / 2.), 0., np.sin(pitch * phase / 2.), 0.]
        goals = supports.copy()
        final_hands = np.array([[advance + .35, .32, .02], [advance + .35, -.32, .02]])
        goals[:2] += phase * (final_hands - supports[:2])
        previous = angles.copy()

        def residual(values):
            probe.qpos[:7] = base
            probe.qpos[task.addresses] = values
            mujoco.mj_forward(task.model, probe)
            errors = (probe.geom_xpos[task.supports] - goals) * np.array([[30.], [30.], [100.], [100.]])
            orientation = (probe.geom_xmat[task.supports[2:]] - foot_rotations).ravel()
            self_depth = max([0.] + [-float(contact.dist) for contact in probe.contact
                                    if task.floor not in (contact.geom1, contact.geom2)])
            other_depth = max([0.] + [-float(contact.dist) for contact in probe.contact
                                     if task.floor in (contact.geom1, contact.geom2)
                                     and (contact.geom2 if contact.geom1 == task.floor else contact.geom1) not in task.supports])
            return np.r_[errors.ravel(), 2. * orientation, .02 * (values - previous),
                         200. * max(0., self_depth - .0005), 200. * other_depth]

        solution = least_squares(residual, angles, bounds=(task.model.jnt_range[task.joints, 0],
                                                          task.model.jnt_range[task.joints, 1]), max_nfev=80)
        angles = solution.x
        residual(angles)
        nodes.append({"qpos": probe.qpos.tolist(), "fit_success": bool(solution.success),
                      "support_error_m": float(np.linalg.norm(probe.geom_xpos[task.supports] - goals, axis=1).max()),
                      "self_penetration_m": max([0.] + [-float(contact.dist) for contact in probe.contact
                                                       if task.floor not in (contact.geom1, contact.geom2)])})
    return nodes


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    guard = PositionTorqueGuard(task.model, fraction=.8)
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/stance_lowering_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "sources": {name: digest(ROOT / "rl" / name) for name in
              (Path(__file__).name, "standing_capture_routes.py", "stand_transition_task.py", "near_handoff_task.py",
               "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "protocol": "Nine root-pose IK references: height .35/.40/.45,pitch .8/1.1/1.4,forward x .15m. "
              "Fixed world feet position/orientation goals, interpolated hand goals. Root pose exists ONLY in planner; "
              "actual robot receives joint targets through native PD and .8 torque guard. Durations1.5/3s, strict physics gates. "
              "No geometry-screen rejection; planner errors recorded rather than treated as dynamic success. "
              "Copied-prefix crawl probes every200ms belowheight.6; no production readiness changes.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for height in (.35, .4, .45):
            for pitch in (.8, 1.1, 1.4):
                nodes = plan(task, state, height, pitch, .15)
                angles = np.array([node["qpos"] for node in nodes])[:, task.addresses]
                for duration in (1.5, 3.):
                    index = len(report["cases"])
                    case = {"height": height, "pitch": pitch, "duration": duration, "nodes": nodes, "probes": []}
                    task.reset(state)
                    for step in range(round((duration + .5) / .02)):
                        coordinate = smooth((step + 1) * .02 / duration) * (len(nodes) - 1)
                        lower = min(int(coordinate), len(nodes) - 2)
                        target = angles[lower] + (coordinate - lower) * (angles[lower + 1] - angles[lower])
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
                                case["success_trajectory"] = str(trajectory.relative_to(ROOT))
                                case["success_trajectory_sha256"] = digest(trajectory)
                                break
                    trajectory = output / f"case_{index}_prefix.npz"
                    task.save_and_replay(trajectory)
                    valid = np.asarray(task.samples)[:-1] if task.failure else np.asarray(task.samples)
                    case.update(success=any(probe["success"] for probe in case["probes"]),
                                min_valid_hand_gap_m=float(valid[:, 8:10].mean(axis=1).min()),
                                max_valid_hand_load_n=float(valid[:, 4:6].max()), trajectory=str(trajectory.relative_to(ROOT)),
                                trajectory_sha256=digest(trajectory), replay_passed=True)
                    report["cases"].append(case)
                    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(index, height, pitch, duration, case["success"], case.get("failure"),
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
