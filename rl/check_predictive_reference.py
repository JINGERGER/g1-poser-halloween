import argparse
import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from predictive_contact_controller import PredictiveContactController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth
from whole_body_torque_bridge_v1 import TorquePositionBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-reference", action="store_true")
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--route", choices=("single", "wide", "step"), default="single")
    args = parser.parse_args()
    if args.route != "single" and not args.online_reference:
        parser.error("step/wide routes require --online-reference")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    source = ROOT / "output/rl/single_support_reference/2026-10-28_18-22-29-396812/report.json"
    nodes = np.array([node["qpos"] for node in json.loads(source.read_text())["nodes"]])
    output = ROOT / "output/rl/predictive_reference" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "steps": [],
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("check_predictive_reference.py", "predictive_contact_controller.py", "contact_step_controller.py")},
              "protocol": "Follow independently planned single-support pose path with native-MuJoCo 20ms finite-difference predictive control. Desired references change only scratch MjData. All actual robot integration remains continuous and strictly screened."}
    task.reset(state)
    try:
        guard = PositionTorqueGuard(task.model)
        bridge = TorquePositionBridge(task.model)
        for unused in range(25):
            guard.advance(task.execution, task.target)
        if args.online_reference:
            before = task.data.qpos.copy()
            if args.route in ("single", "step"):
                from check_single_support_reference import plan
                planned = plan(task, .04, .2) if args.route == "step" else plan(task)
            else:
                from balanced_descent_path import BalancedDescentPath
                planned = BalancedDescentPath(task).solve(.4, 1.8, .26, 1.4)
            np.testing.assert_array_equal(task.data.qpos, before)
            report["online_reference_nodes"] = planned
            report["online_planner_sha256"] = digest(ROOT / "rl/check_single_support_reference.py")
            report["balanced_planner_sha256"] = digest(ROOT / "rl/balanced_descent_path.py")
            nodes = np.array([node["qpos"] for node in planned])
        report["online_reference"] = args.online_reference
        report["route"] = args.route
        if args.position:
            from predictive_position_controller import PredictivePositionController
            controller = PredictivePositionController(task)
            report["sources"]["predictive_position_controller.py"] = digest(ROOT / "rl/predictive_position_controller.py")
        else:
            controller = PredictiveContactController(task)
        report["controller"] = type(controller).__name__
        reference = mujoco.MjData(task.model)
        duration = 16. if args.route == "step" else 8.
        phase, loaded_samples, settled_samples = 0., 0, 0
        contact_search = 0.
        report["contact_events"] = []
        for step in range(round((duration + 12.) / .02)):
            next_phase = min(1., phase + .02 / duration)
            if args.route == "step" and .9 <= next_phase < 1.:
                loaded_samples = loaded_samples + 1 if task.measure()[0][6] > 5. else 0
                if loaded_samples < 3:
                    next_phase = max(.9, phase)
                elif phase <= .9:
                    report["contact_events"].append({"time_s": float(task.data.time), "event": "left_load_confirmed",
                                                     "loads_n": list(task.measure()[0][4:8])})
            phase = next_phase
            coordinate = phase * (len(nodes) - 1)
            lower = min(int(coordinate), len(nodes) - 2)
            fraction = coordinate - lower
            reference.qpos[:] = (1. - fraction) * nodes[lower] + fraction * nodes[lower + 1]
            reference.qpos[3:7] /= np.linalg.norm(reference.qpos[3:7])
            mujoco.mj_forward(task.model, reference)
            loads = np.array([0., 0., .5 * (1. - smooth(phase * 2.)), .5 * (1. + smooth(phase * 2.))]) * controller.transfer.weight
            if args.route == "wide":
                loads = None
            elif args.route == "step":
                left_fraction = .5 * (1. - smooth(phase / .3)) + .5 * smooth((phase - .9) / .1)
                if phase >= .9:
                    left_fraction = .1 + .4 * smooth((phase - .9) / .1)
                loads = np.array([0., 0., left_fraction, 1. - left_fraction]) * controller.transfer.weight
            foot_goals = reference.geom_xpos[task.supports[2:]].copy()
            if args.route == "step" and phase >= .9:
                if task.measure()[0][6] < 5.:
                    contact_search = min(.003, contact_search + .001 * .02)
                foot_goals[0, 2] -= contact_search
            controller.target(reference.subtree_com[controller.pelvis], 0., reference.geom_xpos[task.supports[:2]],
                              reference.qpos[task.addresses], foot_goals,
                              reference.geom_xmat[task.supports[2:]].reshape(2, 3, 3), load_targets=loads,
                              rotation_goal=reference.xmat[controller.pelvis].reshape(3, 3))
            record = {"time_s": float(task.data.time), "phase": phase, "contact_search_offset_m": contact_search, **controller.last}
            report["steps"].append(record)
            if args.position:
                guard.advance(task.execution, controller.last["position_target"])
            else:
                bridge.advance_torque(task.execution, controller.last["torque_nm"])
            record["prediction_velocity_error"] = float(np.max(np.abs(task.data.qvel - controller.last["predicted_qvel"])))
            settled_samples = settled_samples + 1 if phase == 1. else 0
            if settled_samples >= 200:
                break
        report["reference_completed"] = phase == 1. and settled_samples >= 200
        report["status"] = "completed"
    except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
        report.update(status="stopped", failure=task.failure or [str(error)])
        if "controller" in locals():
            report["failure_diagnostics"] = getattr(controller, "failure_diagnostics", None)
    finally:
        report["duration_s"] = float(task.data.time)
        report["full_transition_validated"] = False
        samples = np.array(task.samples)
        if args.route == "single" and len(samples) >= 500:
            window = samples[-500:]
            report["stage_gates"] = {
                "physical_substeps": not bool(task.failure),
                "reference_completed": bool(report.get("reference_completed", False)),
                "left_clearance_3cm_for_1s": bool(np.all(window[:, 10] >= .03)),
                "left_unloaded_for_1s": bool(np.all(window[:, 6] < 5.)),
                "right_carries_weight_for_1s": bool(np.all(window[:, 7] >= .8 * controller.transfer.weight)),
                "settled_linear_speed": bool(np.linalg.norm(task.data.qvel[:3]) < .05),
                "settled_angular_speed": bool(np.linalg.norm(task.data.qvel[3:6]) < .1)}
            report["single_support_passed"] = all(report["stage_gates"].values())
        report["source_snapshots"] = {name: (ROOT / "rl" / name).read_text() for name in
                                      (*report["sources"], "check_single_support_reference.py", "balanced_descent_path.py")}
        trajectory = output / "trajectory.npz"
        task.save_and_replay(trajectory)
        report.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report["status"], report.get("failure"), report["duration_s"], flush=True)
        print("single_support_passed", report.get("single_support_passed", False), flush=True)
        print(report.get("failure_diagnostics"), flush=True)
        print(output / "report.json")


if __name__ == "__main__":
    main()
