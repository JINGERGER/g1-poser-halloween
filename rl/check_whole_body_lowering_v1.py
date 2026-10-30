import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import fork_prefix, smooth
from whole_body_transition_v1 import WholeBodyLowering


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/whole_body_lowering_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "whole_body_transition_v1.py", "standing_capture_routes.py", "stand_transition_task.py",
                           "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "protocol": "Inverse-dynamics QP: full floating-base dynamics, both feet fixed6D acceleration, "
              "eight unilateral point forces inside .6 friction pyramid, 80% torque limits and joint acceleration/position bounds. "
              "COM-height, torso-pitch, hand-position and weak posture acceleration objectives. "
              "Torques converted to native position commands, no actual external force or root-state changes. "
              "COM Jacobian derivative finite-differenced at20ms. Local QP is not collision-safe; original2ms gates retained. "
              "First standing hold control, then three pitch endpoints. Goals not achieved merely by QP feasibility.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for pitch in (0., .6, 1., 1.4):
            task.reset(state)
            controller = WholeBodyLowering(task)
            guard = PositionTorqueGuard(task.model, fraction=.8)
            case = {"pitch_goal": pitch, "steps": [], "probes": []}
            for step in range(200):
                phase = 0. if pitch == 0. else smooth(((step + 1) * .02 - .5) / 2.5)
                height = controller.initial_com[2] + phase * (.35 - controller.initial_com[2])
                try:
                    target = controller.target(height, phase * pitch, phase)
                    case["steps"].append({"time_s": float(task.data.time), **controller.last})
                    guard.advance(task.execution, target)
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
            case.update(success=any(probe["success"] for probe in case["probes"]),
                        duration_s=float(task.data.time), final_height_m=float(task.data.qpos[2]),
                        trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
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
