import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_predictive_descent import replay_prefix
from check_single_support_reference import plan
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from predictive_position_controller import PredictivePositionController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import integration_state, smooth


def reference_gates(nodes, adaptive_height=False):
    return {
        "foot_error": all(node["foot_error_m"] < .005 for node in nodes),
        "com_reference": all((node["support_region_violation_m"] <= .0001 and
                              node["com_height_error_m"] <= (.02 if adaptive_height else .005))
                             if node.get("balance_region", False) else
                             ((node["com_height_error_m"] <= .02 and node["com_error_xy_m"] < .005)
                              if adaptive_height else node["com_error_m"] < .005) for node in nodes),
        "self_penetration": all(node["self_penetration_m"] <= .001 for node in nodes),
        "static_equilibrium": all(node["equilibrium_feasible"] for node in nodes),
    }


def segment_gates(task, segment, samples, positions, velocities):
    start, end = segment["start_index"], segment["end_index"]
    window = samples[start:end]
    actual, reference = mujoco.MjData(task.model), mujoco.MjData(task.model)
    actual.qpos[:] = positions[end]
    reference.qpos[:] = segment["nodes"][-1]["qpos"]
    mujoco.mj_forward(task.model, actual)
    mujoco.mj_forward(task.model, reference)
    feet = task.supports[2:]
    errors = np.linalg.norm(actual.geom_xpos[feet] - reference.geom_xpos[feet], axis=1)
    angles = []
    for geom in feet:
        difference = reference.geom_xmat[geom].reshape(3, 3).T @ actual.geom_xmat[geom].reshape(3, 3)
        angles.append(float(np.arccos(np.clip((np.trace(difference) - 1.) / 2., -1., 1.))))
    segment["foot_position_error_m"] = errors.tolist()
    segment["foot_angle_error_rad"] = angles
    return {
        "reference_completed": segment.get("reference_completed", False),
        "lifted_3cm": bool(len(window) and np.max(window[:, 10 + segment["side"]]) >= .03),
        "both_feet_loaded_for_1s": bool(len(window) >= 500 and np.all(window[-500:, 6:8] > 20.)),
        "foot_position": bool(np.all(errors < .01)),
        "foot_orientation": bool(max(angles) < .05),
        "settled_linear_speed": bool(np.linalg.norm(velocities[end][:3]) < .05),
        "settled_angular_speed": bool(np.linalg.norm(velocities[end][3:6]) < .1),
        "contact_confirmed": bool(segment["contact_events"]),
    }


def execute_segment(task, side, segment, nonlinear_search=False, step_distance=.04, step_yaw=.2, bounded_reference=True,
                    balance_region=False):
    segment.update(side=side, start_index=len(task.samples), start_time_s=float(task.data.time),
                   start_qpos=task.data.qpos.tolist(), contact_events=[], steps=[])
    before = integration_state(task)
    segment.update(step_distance_m=step_distance, step_yaw_rad=step_yaw, bounded_reference=bounded_reference)
    segment["balance_region"] = balance_region
    nodes = plan(task, step_distance, step_yaw, side=side, bounded_continuation=bounded_reference, balance_region=balance_region)
    np.testing.assert_array_equal(integration_state(task), before)
    segment["planner_gates"] = reference_gates(nodes)
    segment["adaptive_height"] = False
    if not all(segment["planner_gates"].values()):
        segment["fixed_height_attempt"] = {"nodes": nodes, "gates": segment["planner_gates"]}
        nodes = plan(task, step_distance, step_yaw, side=side, height_tolerance=.02, bounded_continuation=bounded_reference,
                     balance_region=balance_region)
        np.testing.assert_array_equal(integration_state(task), before)
        segment["adaptive_height"] = True
        segment["planner_gates"] = reference_gates(nodes, adaptive_height=True)
    segment["nodes"] = nodes
    segment["initial_reference_error"] = float(np.max(np.abs(np.array(nodes[0]["qpos"]) - task.data.qpos)))
    if bounded_reference:
        segment["planner_gates"]["initial_reference_continuity"] = segment["initial_reference_error"] < 1e-10
        segment["planner_gates"]["bounded_joint_increments"] = bool(np.max(np.abs(np.diff(
            np.array([node["qpos"] for node in nodes])[:, task.addresses], axis=0))) <= .120001)
    if not all(segment["planner_gates"].values()):
        raise RuntimeError("Reference screening failed: " + str(segment["planner_gates"]))
    poses = np.array([node["qpos"] for node in nodes])
    controller = PredictivePositionController(task)
    controller.nonlinear_search = nonlinear_search
    guard = PositionTorqueGuard(task.model)
    reference = mujoco.MjData(task.model)
    phase, loaded_samples, settled_samples, contact_search = 0., 0, 0, 0.
    try:
        for unused in range(1400):
            next_phase = min(1., phase + .02 / 16.)
            if .9 <= next_phase < 1.:
                required_load = 5.
                loaded_samples = loaded_samples + 1 if task.measure()[0][6 + side] > required_load else 0
                if loaded_samples < 3:
                    next_phase = max(.9, phase)
                elif not segment["contact_events"]:
                    segment["contact_events"].append({"time_s": float(task.data.time),
                                                      "loads_n": list(task.measure()[0][4:8]),
                                                      "required_load_n": required_load})
            phase = next_phase
            coordinate = phase * (len(poses) - 1)
            lower = min(int(coordinate), len(poses) - 2)
            fraction = coordinate - lower
            reference.qpos[:] = (1. - fraction) * poses[lower] + fraction * poses[lower + 1]
            reference.qpos[3:7] /= np.linalg.norm(reference.qpos[3:7])
            mujoco.mj_forward(task.model, reference)
            moving_fraction = .5 * (1. - smooth(phase / .3)) + .5 * smooth((phase - .9) / .1)
            if phase >= .9:
                moving_fraction = .1 + .4 * smooth((phase - .9) / .1)
            loads = np.zeros(4)
            loads[2 + side], loads[3 - side] = moving_fraction, 1. - moving_fraction
            loads *= controller.transfer.weight
            foot_goals = reference.geom_xpos[task.supports[2:]].copy()
            if phase >= .9:
                if task.measure()[0][6 + side] < 5.:
                    contact_search = min(.003, contact_search + .001 * .02)
                foot_goals[side, 2] -= contact_search
            controller.target(reference.subtree_com[controller.pelvis], 0., reference.geom_xpos[task.supports[:2]],
                              reference.qpos[task.addresses], foot_goals,
                              reference.geom_xmat[task.supports[2:]].reshape(2, 3, 3), load_targets=loads,
                              rotation_goal=reference.xmat[controller.pelvis].reshape(3, 3))
            record = {"time_s": float(task.data.time), "phase": phase,
                      "contact_search_offset_m": contact_search, **controller.last}
            segment["steps"].append(record)
            guard.advance(task.execution, controller.last["position_target"])
            record["prediction_velocity_error"] = float(np.max(np.abs(task.data.qvel - controller.last["predicted_qvel"])))
            settled_samples = settled_samples + 1 if phase == 1. else 0
            if settled_samples >= 200:
                break
        segment["reference_completed"] = phase == 1. and settled_samples >= 200
    finally:
        segment["failure_diagnostics"] = getattr(controller, "failure_diagnostics", None)
        segment.update(end_index=len(task.samples), end_time_s=float(task.data.time), end_qpos=task.data.qpos.tolist())
        segment["gates"] = segment_gates(task, segment, np.array(task.samples), task.positions, task.velocities)
        segment["passed"] = not bool(task.failure) and all(segment["gates"].values())
        print("segment", side, segment["passed"], segment["gates"], flush=True)
    if not segment["passed"]:
        raise RuntimeError("Step did not meet completion gates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--nonlinear-search", action="store_true")
    parser.add_argument("--prefix-report", type=Path)
    parser.add_argument("--step-distance", type=float, default=.04)
    parser.add_argument("--step-yaw", type=float, default=.2)
    parser.add_argument("--bounded-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-region", action="store_true")
    args = parser.parse_args()
    if not 0. <= args.step_distance <= .12 or not 0. <= args.step_yaw <= .5 or args.step_distance + args.step_yaw == 0.:
        parser.error("Require distance in [0, .12] m and yaw in [0, .5] rad, with at least one nonzero")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    output = ROOT / "output/rl/predictive_step_chain" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_predictive_step_chain.py", "check_single_support_reference.py", "predictive_position_controller.py",
             "predictive_contact_controller.py", "contact_step_controller.py", "position_torque_guard.py",
             "stand_transition_task.py", "near_handoff_task.py", "standing_capture_routes.py", "check_predictive_descent.py")
    report = {"status": "running", "requested_sides": args.sides, "nonlinear_search": args.nonlinear_search,
              "segments": [], "full_transition_validated": False,
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        if args.prefix_report:
            report["prefix"] = replay_prefix(task, args.prefix_report, expected_sides=None)
        else:
            guard = PositionTorqueGuard(task.model)
            for unused in range(25):
                guard.advance(task.execution, task.target)
        for side in args.sides:
            segment = {}
            report["segments"].append(segment)
            execute_segment(task, ("left", "right").index(side), segment, args.nonlinear_search,
                            args.step_distance, args.step_yaw, args.bounded_reference, args.balance_region)
        report["status"] = "completed"
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        report.update(status="stopped", failure=task.failure or [str(error)])
    finally:
        report["duration_s"] = float(task.data.time)
        report["passed"] = report["status"] == "completed" and not bool(task.failure)
        trajectory = output / "trajectory.npz"
        task.save_and_replay(trajectory)
        report.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report["status"], report.get("failure"), report["duration_s"], flush=True)
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
