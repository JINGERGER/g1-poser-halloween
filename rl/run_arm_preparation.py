import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_predictive_descent import replay_prefix
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from predictive_position_controller import PredictivePositionController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


REFERENCE = ROOT / "output/rl/arm_preparation_path/2026-10-29_11-56-34-886118/report.json"


def stage_gates(task, stage):
    start = stage["start_index"]
    samples = np.array(task.samples[start:])
    initial = mujoco.MjData(task.model)
    initial.qpos[:] = task.positions[start]
    mujoco.mj_forward(task.model, initial)
    feet = task.supports[2:]
    pelvis = task.model.body("pelvis").id
    hand_errors = np.linalg.norm(task.data.geom_xpos[task.supports[:2]] - stage["hand_goals_xyz"], axis=1)
    foot_errors = np.linalg.norm(task.data.geom_xpos[feet] - initial.geom_xpos[feet], axis=1)
    stage.update(hand_error_m=hand_errors.tolist(), foot_drift_m=foot_errors.tolist(),
                 final_hand_positions_xyz=task.data.geom_xpos[task.supports[:2]].tolist())
    return {"reference_completed": stage.get("reference_completed", False),
            "hand_goals": bool(np.all(hand_errors < .01)),
            "feet_fixed": bool(np.all(foot_errors < .005)),
            "foot_orientation": bool(np.max(np.abs(task.data.geom_xmat[feet] - initial.geom_xmat[feet])) < .01),
            "continuous_double_support": bool(len(samples) and np.all(samples[:, 6:8] > 20.)),
            "body_height_held": bool(len(samples) and np.max(np.abs(samples[:, 3] - initial.qpos[2])) < .02),
            "horizontal_com": bool(np.linalg.norm(task.data.subtree_com[pelvis, :2] - initial.subtree_com[pelvis, :2]) < .02),
            "settled_linear_speed": bool(np.linalg.norm(task.data.qvel[:3]) < .05),
            "settled_angular_speed": bool(np.linalg.norm(task.data.qvel[3:6]) < .1)}


def execute(task, stage, route, gate_function=stage_gates):
    stage.update(start_index=len(task.samples), start_time_s=float(task.data.time), steps=[])
    controller = PredictivePositionController(task)
    controller.nonlinear_search = True
    guard = PositionTorqueGuard(task.model)
    reference = mujoco.MjData(task.model)
    poses = np.array([node["qpos"] for node in route["nodes"]])
    loads = np.array(task.measure()[0][4:8])
    phase, held = 0., 0
    started = perf_counter()
    try:
        for unused in range(1000):
            if np.all(np.asarray(task.samples[-50:])[:, 6:8] > 20.):
                phase = min(1., phase + .02 / 12.)
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
            stage["steps"].append(record)
            guard.advance(task.execution, controller.last["position_target"])
            record["prediction_velocity_error"] = float(np.max(np.abs(task.data.qvel - controller.last["predicted_qvel"])))
            if len(stage["steps"]) % 100 == 0:
                print(stage.get("label", "Arm preparation"), len(stage["steps"]), "phase", phase, "wall_s", perf_counter() - started, flush=True)
            held = held + 1 if phase == 1. else 0
            if held >= 200:
                break
        stage["reference_completed"] = phase == 1. and held >= 200
    finally:
        stage.update(end_index=len(task.samples), end_time_s=float(task.data.time), wall_time_s=perf_counter() - started,
                     failure_diagnostics=getattr(controller, "failure_diagnostics", None))
        stage["gates"] = gate_function(task, stage)
        stage["passed"] = not bool(task.failure) and all(stage["gates"].values())
    if not stage["passed"]:
        raise RuntimeError(f"{stage.get('label', 'Arm preparation')} did not meet completion gates")


def verify(task, path, gate_function=stage_gates):
    report = json.loads(path.read_text())
    assert report["passed"] and report["replay_passed"] and not report["full_transition_validated"]
    for name, expected in report["sources"].items():
        assert hashlib.sha256(report["source_snapshots"][name].encode()).hexdigest() == expected
    prefix = report["prefix"]
    assert digest(ROOT / prefix["report"]) == prefix["report_sha256"]
    assert digest(ROOT / prefix["trajectory"]) == prefix["trajectory_sha256"]
    assert digest(ROOT / report["reference_report"]) == report["reference_report_sha256"]
    with np.load(ROOT / report["trajectory"]) as saved, np.load(ROOT / prefix["trajectory"]) as original:
        for name in ("qpos", "qvel", "ctrl", "phase", "samples", "penetrations"):
            np.testing.assert_allclose(saved[name][:len(original[name])], original[name], atol=1e-10, rtol=0.)
    verification = replay_prefix(task, path, expected_sides=None)
    stage = report["stage"]
    assert stage["start_index"] == prefix["end_index"]
    assert stage["end_index"] == len(task.samples)
    assert len(stage["steps"]) * 10 == stage["end_index"] - stage["start_index"]
    assert sum(step["phase"] == 1. for step in stage["steps"]) >= 200
    for index, step in enumerate(stage["steps"]):
        assert abs(step["time_s"] - (stage["start_index"] + 10 * index) * .002) < 1e-9
        assert step["prediction_velocity_error"] < 1e-10
        assert step["normalized_constraint_residual"] <= 1e-4
        chosen = next(candidate for candidate in step["nonlinear_search"] if candidate["fraction"] == step["selected_fraction"])
        assert chosen["cost"] == min(candidate["cost"] for candidate in step["nonlinear_search"])
    assert all(gate_function(task, stage).values())
    print(json.dumps({"passed": True, "duration_s": float(task.data.time), "hand_error_m": stage["hand_error_m"],
                      "replay_wall_time_s": verification["wall_time_s"],
                      "full_transition_validated": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--reference-report", type=Path, default=REFERENCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    if args.verify_report:
        verify(task, args.verify_report)
        return
    reference_path = args.reference_report.resolve()
    planned = json.loads(reference_path.read_text())
    source_path = ROOT / planned["source_report"]
    assert planned["source_report_sha256"] == digest(source_path)
    route = next(route for route in planned["routes"] if route["family"] == "front_then_down")
    assert route["nodes_passed"]
    output = ROOT / "output/rl/arm_preparation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("run_arm_preparation.py", "check_predictive_descent.py", "predictive_contact_controller.py",
             "predictive_position_controller.py", "position_torque_guard.py", "stand_transition_task.py", "near_handoff_task.py")
    report = {"status": "running", "full_transition_validated": False,
              "reference_report": str(reference_path.relative_to(ROOT)), "reference_report_sha256": digest(reference_path),
              "stage": {"hand_goals_xyz": planned["hand_target_xyz"]},
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        report["prefix"] = replay_prefix(task, source_path, expected_sides=None)
        execute(task, report["stage"], route)
        report["status"] = "completed"
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        report.update(status="stopped", failure=task.failure or [str(error)])
    finally:
        report["passed"] = report["status"] == "completed" and not bool(task.failure)
        report["duration_s"] = float(task.data.time)
        trajectory = output / "trajectory.npz"
        task.save_and_replay(trajectory)
        report.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report["status"], report.get("failure"), report["stage"].get("gates"), flush=True)
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
