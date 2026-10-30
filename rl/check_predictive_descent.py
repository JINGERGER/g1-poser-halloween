import argparse
import json
from datetime import datetime
from time import perf_counter

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from predictive_descent_reference import plan_descent
from predictive_position_controller import PredictivePositionController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import integration_state


PREFIX = ROOT / "output/rl/predictive_step_chain/2026-10-29_11-06-00-449154/report.json"


def replay_prefix(task, path, expected_sides=("right", "left")):
    path = path.resolve()
    relative_path = path.relative_to(ROOT)
    started = perf_counter()
    print("Replaying validated prefix from true standing", flush=True)
    report = json.loads(path.read_text())
    assert report["passed"] and report["replay_passed"]
    if expected_sides is not None:
        assert report["requested_sides"] == list(expected_sides)
    trajectory = ROOT / report["trajectory"]
    assert digest(trajectory) == report["trajectory_sha256"]
    with np.load(trajectory) as saved:
        positions, velocities, controls = saved["qpos"], saved["qvel"], saved["ctrl"]
        np.testing.assert_array_equal(task.data.qpos, positions[0])
        np.testing.assert_array_equal(task.data.qvel, velocities[0])
        assert np.all(saved["phase"] == 0)
        for index, control in enumerate(controls):
            assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
            task.data.ctrl[:] = control
            mujoco.mj_step(task.model, task.data)
            mujoco.mj_forward(task.model, task.data)
            task._observe_step(task.model, task.data)
            np.testing.assert_allclose(task.data.qpos, positions[index + 1], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(task.data.qvel, velocities[index + 1], atol=1e-10, rtol=0.)
    return {"report": str(relative_path), "report_sha256": digest(path),
            "trajectory": report["trajectory"], "trajectory_sha256": report["trajectory_sha256"],
            "end_index": len(task.samples), "wall_time_s": perf_counter() - started,
            "method": "integrate every archived 2ms control from true standing; no endpoint reset"}


def descent_gates(task, segment, saved):
    start, end = segment["start_index"], segment["end_index"]
    samples = np.asarray(saved["samples"])[start:end]
    data = mujoco.MjData(task.model)
    data.qpos[:] = saved["qpos"][start]
    mujoco.mj_forward(task.model, data)
    pelvis = task.model.body("pelvis").id
    initial_com = data.subtree_com[pelvis].copy()
    initial_feet = data.geom_xpos[task.supports[2:]].copy()
    initial_rotations = data.geom_xmat[task.supports[2:]].copy()
    data.qpos[:] = saved["qpos"][end]
    mujoco.mj_forward(task.model, data)
    base_drop = float(saved["qpos"][start][2] - data.qpos[2])
    com_drop = float(initial_com[2] - data.subtree_com[pelvis, 2])
    errors = np.linalg.norm(data.geom_xpos[task.supports[2:]] - initial_feet, axis=1)
    segment.update(actual_base_drop_m=base_drop, actual_com_drop_m=com_drop, foot_drift_m=errors.tolist())
    return {"reference_completed": segment.get("reference_completed", False),
            "base_drop": abs(base_drop - segment["requested_drop_m"]) < .005,
            "com_drop": abs(com_drop - segment["requested_drop_m"]) < .005,
            "horizontal_com": bool(np.linalg.norm(data.subtree_com[pelvis, :2] - initial_com[:2]) < .01),
            "feet_fixed": bool(np.all(errors < .005)),
            "foot_orientation": bool(np.max(np.abs(data.geom_xmat[task.supports[2:]] - initial_rotations)) < .01),
            "continuous_double_support": bool(len(samples) and np.all(samples[:, 6:8] > 20.)),
            "final_hold_support": bool(len(samples) >= 500 and np.all(samples[-500:, 6:8] > 20.)),
            "settled_linear_speed": bool(np.linalg.norm(saved["qvel"][end][:3]) < .05),
            "settled_angular_speed": bool(np.linalg.norm(saved["qvel"][end][3:6]) < .1)}


def descend(task, segment, drop):
    segment.update(start_index=len(task.samples), start_time_s=float(task.data.time),
                   requested_drop_m=drop, steps=[])
    before = integration_state(task)
    started = perf_counter()
    print("Planning descent from continuous state", float(task.data.time), flush=True)
    nodes = plan_descent(task, drop)
    segment["planning_wall_time_s"] = perf_counter() - started
    np.testing.assert_array_equal(integration_state(task), before)
    segment["nodes"] = nodes
    if not all(node["passed"] for node in nodes):
        raise RuntimeError("Descent reference failed geometric/static screening")
    print("Reference screening passed", segment["planning_wall_time_s"], flush=True)
    controller = PredictivePositionController(task)
    controller.nonlinear_search = True
    guard = PositionTorqueGuard(task.model)
    reference = mujoco.MjData(task.model)
    poses = np.array([node["qpos"] for node in nodes])
    loads = np.array(task.measure()[0][4:8])
    phase, held = 0., 0
    started = perf_counter()
    try:
        for unused in range(800):
            if np.all(np.asarray(task.samples[-50:])[:, 6:8] > 20.):
                phase = min(1., phase + .02 / 8.)
            coordinate = phase * (len(poses) - 1)
            lower = min(int(coordinate), len(poses) - 2)
            fraction = coordinate - lower
            reference.qpos[:] = (1. - fraction) * poses[lower] + fraction * poses[lower + 1]
            reference.qpos[3:7] /= np.linalg.norm(reference.qpos[3:7])
            mujoco.mj_forward(task.model, reference)
            controller.target(reference.subtree_com[controller.pelvis], 0., reference.geom_xpos[task.supports[:2]],
                              reference.qpos[task.addresses], reference.geom_xpos[task.supports[2:]],
                              reference.geom_xmat[task.supports[2:]].reshape(2, 3, 3), load_targets=loads,
                              rotation_goal=reference.xmat[controller.pelvis].reshape(3, 3))
            record = {"time_s": float(task.data.time), "phase": phase, **controller.last}
            segment["steps"].append(record)
            guard.advance(task.execution, controller.last["position_target"])
            record["prediction_velocity_error"] = float(np.max(np.abs(task.data.qvel - controller.last["predicted_qvel"])))
            if len(segment["steps"]) % 100 == 0:
                print("Descent control", len(segment["steps"]), "phase", phase, "wall_s", perf_counter() - started, flush=True)
            held = held + 1 if phase == 1. else 0
            if held >= 200:
                break
        segment["reference_completed"] = phase == 1. and held >= 200
    finally:
        segment["control_wall_time_s"] = perf_counter() - started
        segment.update(end_index=len(task.samples), end_time_s=float(task.data.time),
                       failure_diagnostics=getattr(controller, "failure_diagnostics", None))
        segment["gates"] = descent_gates(task, segment, {"samples": task.samples, "qpos": task.positions, "qvel": task.velocities})
        segment["passed"] = not task.failure and all(segment["gates"].values())
    if not segment["passed"]:
        raise RuntimeError("Descent did not meet completion gates")


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", type=float, default=.03)
    args = parser.parse_args()
    if not 0. < args.drop <= .1:
        parser.error("--drop must be in (0, 0.1] meters")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    output = ROOT / "output/rl/predictive_descent" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_predictive_descent.py", "predictive_descent_reference.py", "check_single_support_reference.py",
             "predictive_contact_controller.py", "predictive_position_controller.py", "contact_step_controller.py",
             "position_torque_guard.py", "stand_transition_task.py", "near_handoff_task.py", "standing_capture_routes.py")
    report = {"status": "running", "full_transition_validated": False, "descent": {},
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        report["prefix"] = replay_prefix(task, PREFIX)
        descend(task, report["descent"], args.drop)
        report["status"] = "completed"
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        report.update(status="stopped", failure=task.failure or [str(error)])
    finally:
        report["passed"] = report["status"] == "completed" and not bool(task.failure)
        report["duration_s"] = float(task.data.time)
        trajectory = output / "trajectory.npz"
        replay_started = perf_counter()
        task.save_and_replay(trajectory)
        report["final_replay_wall_time_s"] = perf_counter() - replay_started
        report["total_wall_time_s"] = perf_counter() - started
        report.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report["status"], report.get("failure"), report["descent"].get("gates"), flush=True)
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
