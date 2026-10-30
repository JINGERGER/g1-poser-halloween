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
from standing_capture_routes import fork_prefix, smooth
from whole_body_transition_v1 import WholeBodyLowering


class TorquePositionBridge(PositionTorqueGuard):
    def advance_torque(self, execution, torque):
        if execution.phase != "transition" or execution.model is not self.model:
            raise ValueError("Mismatched transition execution")
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (self.model.nu,) or not np.isfinite(torque).all():
            raise ValueError("Expected finite generalized actuator torques")
        if np.any(np.abs(torque) > self.fraction * self.limits + .02):
            raise ValueError("Torque request exceeds declared QP force margin")
        model, data = self.model, execution.data
        for unused in range(10):
            if data.qfrc_applied.any() or data.xfrc_applied.any():
                raise ValueError("External forces prohibited")
            bias = model.actuator_biasprm[:, 0] + model.actuator_biasprm[:, 1] * data.actuator_length + model.actuator_biasprm[:, 2] * data.actuator_velocity
            requested = (torque - bias) / model.actuator_gainprm[:, 0]
            data.ctrl[:] = self.apply(data, requested)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("Nonfinite state")
            if execution.step_observer is not None:
                execution.step_observer(model, data)


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/whole_body_torque_bridge_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "whole_body_transition_v1.py", "standing_capture_routes.py", "stand_transition_task.py",
                           "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "Same four QP cases as first whole-body test, but reconvert desired torque to native position command "
              "every2ms using CURRENT actuator length/velocity. Previous test held one converted position command20ms, "
              "which is not constant torque during motion. No actuator model, force limit or physical gate changes. "
              "Conversion still subject to position/control bounds and guard; actual post-step torque can differ. "
              "Predicted point support forces remain a model, not applied external forces.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for pitch in (0., .6, 1., 1.4):
            task.reset(state)
            controller = WholeBodyLowering(task)
            bridge = TorquePositionBridge(task.model, fraction=.8)
            case = {"pitch_goal": pitch, "probes": [], "steps": []}
            for step in range(200):
                phase = 0. if pitch == 0. else smooth(((step + 1) * .02 - .5) / 2.5)
                height = controller.initial_com[2] + phase * (.35 - controller.initial_com[2])
                try:
                    controller.target(height, phase * pitch, phase)
                    case["steps"].append({"time_s": float(task.data.time), **controller.last})
                    bridge.advance_torque(task.execution, controller.last["torque_nm"])
                except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
                    case["failure"] = task.failure or [str(error)]
                    break
                if step % 10 == 9 and task.data.qpos[2] < .55:
                    fork_prefix(task, branch)
                    result = branch.finish()
                    case["probes"].append({"time_s": float(task.data.time), **result})
                    if result["success"]:
                        trajectory = output / f"pitch_{pitch}_success.npz"
                        branch.save_and_replay(trajectory)
                        case.update(success_trajectory=str(trajectory.relative_to(ROOT)), success_trajectory_sha256=digest(trajectory))
                        break
            trajectory = output / f"pitch_{pitch}.npz"
            task.save_and_replay(trajectory)
            case.update(success=any(probe["success"] for probe in case["probes"]), duration_s=float(task.data.time),
                        final_height_m=float(task.data.qpos[2]), trajectory=str(trajectory.relative_to(ROOT)),
                        trajectory_sha256=digest(trajectory), replay_passed=True)
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(pitch, case["success"], case.get("failure"), case["duration_s"], case["final_height_m"], flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
