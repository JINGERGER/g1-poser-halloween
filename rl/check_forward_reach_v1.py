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
from standing_cartesian_capture import reaching_target
from transition_v1 import StandingTransitionV1


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/transition_v1_search/2026-10-28_11-56-32-238032/report.json"
    original = json.loads(source.read_text())
    task, branch = StandTransitionTask(), StandTransitionTask()
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    assert assets == original["assets"] and digest(task.policy_path) == original["policy_sha256"]
    guard = PositionTorqueGuard(task.model, fraction=.8)
    output = ROOT / "output/rl/forward_reach_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "assets": assets,
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "transition_v1.py", "standing_cartesian_capture.py", "standing_capture_routes.py",
                           "stand_transition_task.py", "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "18 true-standing trials: three selected forward-event candidates crossed with world hand x-offset .25/.35/.45, "
              "reach start pitch .2/.7. Hand goals z=.02,y=pelvis+/-.32; IK-only command probes, no real qpos override. "
              "World-goal blend ramps over .3s from measured hands, native PD and original strict physical gates. "
              "Actual crawl probes every20ms at pitch>1.1,height<.5 to check narrow timing windows. "
              "Not online oracle deployment or permission to relax failed gates.", "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for run in original["runs"]:
            for offset in (.25, .35, .45):
                for trigger in (.2, .7):
                    task.reset(original["initial_state"])
                    controller = StandingTransitionV1(task, run["selected"]["parameters"])
                    reach_start, hands = None, None
                    case = {"seed": run["seed"], "offset": offset, "trigger": trigger, "probes": []}
                    index = len(report["cases"])
                    for step in range(200):
                        target = controller.target(task)
                        rotation = task.data.xmat[task.model.body("pelvis").id].reshape(3, 3)
                        pitch = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
                        if reach_start is None and task.data.time >= .6 and pitch >= trigger:
                            reach_start, hands = float(task.data.time), task.data.geom_xpos[task.supports[:2]].copy()
                        if reach_start is not None:
                            goals = np.array([[task.data.qpos[0] + offset, task.data.qpos[1] + .32, .02],
                                              [task.data.qpos[0] + offset, task.data.qpos[1] - .32, .02]])
                            goals = hands + smooth((task.data.time - reach_start + .02) / .3) * (goals - hands)
                            target = reaching_target(task, target, goals)
                        try:
                            guard.advance(task.execution, target)
                        except (UnsafeState, InfeasiblePositionTarget) as error:
                            case["failure"] = task.failure or [type(error).__name__]
                            break
                        if pitch > 1.1 and task.data.qpos[2] < .5:
                            fork_prefix(task, branch)
                            result = branch.finish()
                            case["probes"].append({"time_s": float(task.data.time), **result})
                            if result["success"]:
                                trajectory = output / f"case_{index}_success.npz"
                                branch.save_and_replay(trajectory)
                                case.update(success_trajectory=str(trajectory.relative_to(ROOT)), success_trajectory_sha256=digest(trajectory))
                                break
                    trajectory = output / f"case_{index}.npz"
                    task.save_and_replay(trajectory)
                    valid = np.asarray(task.samples)[:-1] if task.failure else np.asarray(task.samples)
                    case.update(success=any(probe["success"] for probe in case["probes"]),
                                min_valid_hand_gap_m=float(valid[:, 8:10].mean(axis=1).min()), max_valid_hand_load_n=float(valid[:, 4:6].max()),
                                trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
                    report["cases"].append(case)
                    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(index, run["seed"], offset, trigger, case["success"], case.get("failure"),
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
