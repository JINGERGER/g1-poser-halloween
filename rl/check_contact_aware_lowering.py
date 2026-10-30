import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from contact_aware_lowering import ContactAwareLowering
from measured_support import LoadTransfer, support_contacts
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import integration_state
from whole_body_torque_bridge_v1 import TorquePositionBridge
from whole_body_transition_v1 import WholeBodyLowering


def checks(task, state):
    transfer = LoadTransfer(300.)
    for unused in range(5):
        transfer.update([0., 0., 150., 150.], [False, False, True, True], 0., 0.)
    assert transfer.mode == "feet" and transfer.phase > 0.
    for unused in range(2):
        transfer.update([20., 20., 130., 130.], [True] * 4, 0., 0.)
        assert transfer.mode == "touch"
    transfer.update([20., 20., 130., 130.], [True] * 4, 0., 0.)
    assert transfer.mode == "transfer"
    transfer.update([20., 0., 130., 150.], [True, False, True, True], 0., 0.)
    assert transfer.mode == "touch" and transfer.hand_age[1] == 0.
    before = transfer.phase
    transfer.update([0., 0., 150., 150.], [False, False, True, True], -.5, 0.)
    assert transfer.phase == before and "vertical_speed" in transfer.last["pause_reasons"]
    transfer.update([0.] * 4, [False] * 4, 0., 0.)
    assert transfer.phase == before
    task.reset(state)
    controller = ContactAwareLowering(task)
    assert not support_contacts(task)
    try:
        controller.target(controller.initial_com[2], 0., 0.)
    except RuntimeError as error:
        assert "No measured support" in str(error)
    else:
        raise AssertionError("Airborne support invented")
    guard = PositionTorqueGuard(task.model)
    for unused in range(25):
        guard.advance(task.execution, task.target)
    contacts = support_contacts(task)
    assert contacts and all(contact["geom"] in task.supports[2:] for contact in contacts)
    assert all(contact["frame"][0, 2] > .99 for contact in contacts)
    controller = ContactAwareLowering(task)
    before = integration_state(task)
    controller.target(controller.initial_com[2], 0., 0.)
    np.testing.assert_array_equal(integration_state(task), before)
    assert controller.last["predicted_support_loads_n"][:2] == [0., 0.]
    TorquePositionBridge(task.model).advance_torque(task.execution, controller.last["torque_nm"])
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "contact_qp.npz")
    original_target = task.target.copy()
    fixture = json.loads((ROOT / "output/rl/low_support_transition/2026-10-23_14-46-00-951505/report.json").read_text())
    pose = fixture["cases"][0]["nodes"][-1]["qpos"]
    task.target = np.array(pose)[task.addresses].copy()
    task.reset({"qpos": pose, "qvel": np.zeros(task.model.nv).tolist()})
    for unused in range(10):
        guard.advance(task.execution, task.target)
    controller = ContactAwareLowering(task)
    for unused in range(3):
        present = [any(contact["geom"] == geom for contact in support_contacts(task)) for geom in task.supports]
        controller.transfer.update(task.measure()[0][4:8], present, 0., 0., advance=False)
        guard.advance(task.execution, task.target)
    assert controller.transfer.mode == "transfer"
    rotation = task.data.xmat[controller.pelvis].reshape(3, 3)
    controller.target(controller.initial_com[2], float(np.arctan2(rotation[0, 2], rotation[2, 2])), 0.)
    assert set(controller.last["contact_geoms"]) == {task.model.geom(geom).name for geom in task.supports}
    assert all(load > 0. for load in controller.last["predicted_support_loads_n"][:2])
    task.target = original_target
    task.reset(state)
    print("PASS: measured-only supports, no airborne forces, contact confirmation/release, descent pause, immutable QP and replay", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.)
    args = parser.parse_args()
    if not 13. <= args.seconds <= 60.:
        parser.error("--seconds must be between 13 and 60")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    checks(task, state)
    output = ROOT / "output/rl/contact_aware_lowering" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = [Path(__file__).name, "contact_aware_lowering.py", "measured_support.py",
             "whole_body_transition_v1.py", "whole_body_torque_bridge_v1.py", "near_handoff_task.py",
             "stand_transition_task.py", "position_torque_guard.py", "transition_execution.py"]
    report = {"status": "running", "full_transition_validated": False,
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "protocol": "Matched native-PD settling .5s, hold and 12s lowering schedule. Baseline fixed-foot QP versus measured-contact QP plus load-gated phase. This is a combined architecture comparison, not an isolated ablation. Original physical gates unchanged. No crawl handoff attempted.",
              "initial_state": state, "lowering_horizon_s": args.seconds, "cases": []}
    try:
        for hold in (True, False):
            for measured in (False, True):
                task.reset(state)
                case = {"hold": hold, "measured_contact_control": measured, "steps": []}
                report["cases"].append(case)
                guard = PositionTorqueGuard(task.model)
                bridge = TorquePositionBridge(task.model)
                phase = 0.
                try:
                    for unused in range(25):
                        guard.advance(task.execution, task.target)
                    controller = ContactAwareLowering(task) if measured else WholeBodyLowering(task)
                    phase = 0.
                    for step in range(round(args.seconds / .02) if not hold else 200):
                        row = task.measure()[0]
                        if measured:
                            present = [any(contact["geom"] == geom for contact in support_contacts(task)) for geom in task.supports]
                            previous_goal = controller.initial_com[2] + phase * (.35 - controller.initial_com[2])
                            com_jac = np.zeros((3, task.model.nv))
                            mujoco.mj_jacSubtreeCom(task.model, task.data, com_jac, controller.pelvis)
                            phase = controller.transfer.update(row[4:8], present, (com_jac @ task.data.qvel)[2],
                                                               task.data.subtree_com[controller.pelvis, 2] - previous_goal,
                                                               advance=not hold)
                        else:
                            phase = 0. if hold else min(1., (step + 1) * .02 / 12.)
                        height = controller.initial_com[2] + phase * (.35 - controller.initial_com[2])
                        controller.target(height, 1.4 * phase, phase)
                        record = {"time_s": float(task.data.time), "phase": phase,
                                  "com_m": task.data.subtree_com[controller.pelvis].tolist(),
                                  "com_goal_height_m": float(height),
                                  "loads_before_n": list(row[4:8]), **controller.last}
                        case["steps"].append(record)
                        bridge.advance_torque(task.execution, controller.last["torque_nm"])
                        record["loads_after_n"] = list(task.measure()[0][4:8])
                        record["base_velocity"] = task.data.qvel[:6].tolist()
                        record["torque_tracking_max_error_nm"] = float(np.max(np.abs(task.data.qfrc_actuator[task.model.jnt_dofadr[task.joints]] - controller.last["torque_nm"])))
                except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
                    case["failure"] = task.failure or [str(error)]
                case.update(duration_s=float(task.data.time), final_height_m=float(task.data.qpos[2]),
                            physics_passed=not bool(task.failure), schedule_completed=bool(phase >= 1.),
                            controller_completed="failure" not in case,
                            **task.progress())
                trajectory = output / f"{'hold' if hold else 'lower'}_{'measured' if measured else 'fixed'}.npz"
                task.save_and_replay(trajectory)
                case.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
                print(hold, measured, case.get("failure"), case["duration_s"], case["final_height_m"], phase, flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
