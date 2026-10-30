import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_predictive_descent import replay_prefix
from position_torque_guard import InfeasiblePositionTarget
from reference_policy_baseline import ROOT
from run_arm_preparation import execute, stage_gates, verify
from stand_transition_task import StandTransitionTask, UnsafeState


def lowering_gates(task, stage):
    gates = stage_gates(task, stage)
    gates.pop("body_height_held")
    gates["final_height"] = bool(abs(task.data.qpos[2] - stage["base_height_goal_m"]) < .02)
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--family", choices=("flat_close", "crouch_close", "crouch_forward"), default="crouch_close")
    parser.add_argument("--end-phase", type=float, default=.55)
    args = parser.parse_args()
    if not 0. < args.end_phase <= 1.:
        parser.error("end phase must be in (0, 1]")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    if args.verify_report:
        report = json.loads(args.verify_report.read_text())
        planned = json.loads((ROOT / report["reference_report"]).read_text())
        stage = report["stage"]
        route = next(route for route in planned["routes"] if route["family"] == stage["family"])
        expected = [node for node in route["nodes"] if node["phase"] <= stage["end_phase"] + 1e-10]
        assert all(node["passed"] and node["active_capacity"]["feasible"] for node in expected)
        np.testing.assert_array_equal(stage["nodes_qpos"][1:], [node["qpos"] for node in expected[1:]])
        scratch = mujoco.MjData(task.model)
        scratch.qpos[:] = expected[-1]["qpos"]
        mujoco.mj_forward(task.model, scratch)
        np.testing.assert_array_equal(stage["hand_goals_xyz"], scratch.geom_xpos[task.supports[:2]])
        assert stage["base_height_goal_m"] == scratch.qpos[2]
        with np.load(ROOT / report["trajectory"]) as saved:
            np.testing.assert_array_equal(stage["nodes_qpos"][0], saved["qpos"][stage["start_index"]])
        verify(task, args.verify_report, lowering_gates)
        return
    if args.reference is None:
        parser.error("Provide a reference or --verify-report")
    reference_path = args.reference.resolve()
    planned = json.loads(reference_path.read_text())
    source_path = ROOT / planned["source_report"]
    assert digest(source_path) == planned["source_report_sha256"]
    route = next(route for route in planned["routes"] if route["family"] == args.family)
    nodes = [node for node in route["nodes"] if node["phase"] <= args.end_phase + 1e-10]
    assert len(nodes) >= 2 and abs(nodes[-1]["phase"] - args.end_phase) < 1e-10
    assert all(node["passed"] and node["active_capacity"]["feasible"] for node in nodes)
    output = ROOT / "output/rl/contact_lowering" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("run_contact_lowering.py", "run_arm_preparation.py", "check_predictive_descent.py",
             "predictive_contact_controller.py", "predictive_position_controller.py", "position_torque_guard.py",
             "stand_transition_task.py", "near_handoff_task.py")
    report = {"status": "running", "full_transition_validated": False,
              "reference_report": str(reference_path.relative_to(ROOT)), "reference_report_sha256": digest(reference_path),
              "stage": {"label": "Contact lowering", "family": args.family, "end_phase": args.end_phase},
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        report["prefix"] = replay_prefix(task, source_path, expected_sides=None)
        poses = np.array([node["qpos"] for node in nodes])
        poses[0] = task.data.qpos.copy()
        assert np.max(np.abs(np.diff(poses[:, task.addresses], axis=0))) <= .200001
        scratch = mujoco.MjData(task.model)
        for lower, upper in zip(poses[:-1], poses[1:], strict=True):
            for fraction in np.linspace(0., 1., 11):
                scratch.qpos[:] = (1. - fraction) * lower + fraction * upper
                scratch.qpos[3:7] /= np.linalg.norm(scratch.qpos[3:7])
                mujoco.mj_forward(task.model, scratch)
                for contact in scratch.contact:
                    if task.floor not in (contact.geom1, contact.geom2):
                        if contact.dist < -.001:
                            raise RuntimeError("Interpolated reference self-collision")
                    else:
                        geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
                        if geom not in task.supports and contact.dist < 0.:
                            raise RuntimeError("Interpolated reference non-support ground penetration")
        report["stage"].update(hand_goals_xyz=scratch.geom_xpos[task.supports[:2]].tolist(),
                               base_height_goal_m=float(poses[-1, 2]), nodes_qpos=poses.tolist())
        execute(task, report["stage"], {"nodes": [{"qpos": pose.tolist()} for pose in poses]}, lowering_gates)
        report["status"] = "completed"
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        report.update(status="stopped", failure=task.failure or [str(error)])
    finally:
        report.update(passed=report["status"] == "completed" and not bool(task.failure), duration_s=float(task.data.time))
        trajectory = output / "trajectory.npz"
        task.save_and_replay(trajectory)
        report.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report["status"], report.get("failure"), report["stage"].get("gates"), flush=True)
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
