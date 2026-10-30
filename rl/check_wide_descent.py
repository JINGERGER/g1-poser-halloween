import json
from datetime import datetime

import numpy as np
import torch

from artifact_utils import digest
from balanced_descent_path import BalancedDescentPath
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import fork_prefix, smooth


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    output = ROOT / "output/rl/wide_descent" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "sources": {name: digest(ROOT / "rl" / name) for name in
              ("balanced_descent_path.py", "check_wide_descent.py")}, "cases": [],
              "protocol": "Wide stance geometric reference only, physical joint-target execution with native PD. No foot/root-state projection. Static-equilibrium feedforward is an ablation, not a guarantee. Independent endpoint capture is separately labelled and never counted as standing success."}
    try:
        source = ROOT / "output/rl/descent_endpoints/2026-10-28_18-06-02-242814/report.json"
        endpoints = json.loads(source.read_text())
        report["endpoint_source_sha256"] = digest(source)
        for index, candidate in enumerate(endpoints["cases"]):
            if not candidate["node"]["passed"]:
                continue
            pose = np.array(candidate["node"]["qpos"])
            branch.target = pose[branch.addresses].copy()
            branch.reset({"qpos": pose.tolist(), "qvel": np.zeros(branch.model.nv).tolist()})
            result = branch.finish()
            trajectory = output / f"endpoint_{index}.npz"
            branch.save_and_replay(trajectory)
            report["cases"].append({"kind": "independent_endpoint", "index": index, **result,
                                    "trajectory": str(trajectory.relative_to(ROOT)), "trajectory_sha256": digest(trajectory)})
            print("endpoint", index, result, flush=True)
        task.reset(state)
        planner = BalancedDescentPath(task)
        nodes = planner.solve(.4, 1.8, .26, 1.4)
        report["nodes"] = nodes
        print("path passing nodes", sum(node["passed"] for node in nodes), len(nodes), flush=True)
        angles = np.array([node["qpos"] for node in nodes])[:, task.addresses]
        torques = np.array([node["equilibrium"]["torque_nm"] if node["equilibrium"]["feasible"] else np.zeros(task.model.nu) for node in nodes])
        for duration in (4., 12.):
            for feedforward in (False, True):
                task.reset(state)
                guard = PositionTorqueGuard(task.model)
                case = {"kind": "standing_path", "duration": duration, "feedforward": feedforward, "probes": []}
                for step in range(round((duration + 1.) / .02)):
                    coordinate = smooth(((step + 1) * .02 - .5) / duration) * (len(nodes) - 1)
                    lower = min(int(coordinate), len(nodes) - 2)
                    fraction = coordinate - lower
                    target = (1. - fraction) * angles[lower] + fraction * angles[lower + 1]
                    if feedforward:
                        torque = (1. - fraction) * torques[lower] + fraction * torques[lower + 1]
                        target += torque / task.model.actuator_gainprm[:, 0]
                    try:
                        guard.advance(task.execution, target)
                    except (UnsafeState, InfeasiblePositionTarget) as error:
                        case["failure"] = task.failure or [str(error)]
                        break
                    row = task.measure()[0]
                    if step % 10 == 9 and (np.mean(row[8:10]) < .10 or task.data.qpos[2] < .4):
                        fork_prefix(task, branch)
                        result = branch.finish()
                        case["probes"].append({"time_s": float(task.data.time), **result})
                        if result["success"]:
                            trajectory = output / f"success_{duration}_{feedforward}.npz"
                            branch.save_and_replay(trajectory)
                            case["success_trajectory"] = str(trajectory.relative_to(ROOT))
                            case["success_trajectory_sha256"] = digest(trajectory)
                            break
                trajectory = output / f"path_{duration}_{feedforward}.npz"
                task.save_and_replay(trajectory)
                case.update(success=any(probe["success"] for probe in case["probes"]), duration_s=float(task.data.time),
                            trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), **task.progress())
                report["cases"].append(case)
                print(duration, feedforward, case.get("failure"), case["success"], case["min_mean_hand_gap_m"], flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
