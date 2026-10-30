import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from check_predictive_descent import replay_prefix
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from search_dynamic_transition import longest_interval
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--family", choices=("free_close", "free_forward", "free_toe"), default="free_toe")
    args = parser.parse_args()
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text())
    source_path = ROOT / reference["source_report"]
    assert digest(source_path) == reference["source_report_sha256"]
    route = next(route for route in reference["routes"] if route["family"] == args.family)
    torch.set_num_threads(1)
    output = ROOT / "output/rl/dynamic_hand_contact" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("probe_dynamic_hand_contact.py", "check_predictive_descent.py", "position_torque_guard.py",
             "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py")
    report = {"status": "running", "mode": "bounded_dynamic_contact_probes", "full_transition_validated": False,
              "reference_report": str(reference_path.relative_to(ROOT)), "reference_report_sha256": digest(reference_path),
              "family": args.family, "cases": [],
              "limitations": "Reference nodes are kinematic suggestions, not certified feasible paths. Only actual integrated states are used for physical gates and success. Every case starts from true standing and replays the complete validated prefix.",
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    for duration in (1., 2., 4.):
        task = StandTransitionTask()
        state = task.make_states([0])[0]
        state["qvel"] = np.zeros(task.model.nv).tolist()
        task.reset(state)
        case = {"command_duration_s": duration}
        report["cases"].append(case)
        case["prefix"] = replay_prefix(task, source_path, expected_sides=None)
        start = len(task.samples)
        poses = np.array([node["qpos"] for node in route["nodes"]])
        poses[0] = task.data.qpos.copy()
        commands = poses[:, task.addresses]
        offset = task.data.ctrl.copy() - commands[0]
        guard = PositionTorqueGuard(task.model)
        handoff = None
        try:
            for step in range(round((duration + 1.) / .02)):
                phase = min(1., (step + 1) * .02 / duration)
                coordinate = phase * (len(commands) - 1)
                lower = min(int(coordinate), len(commands) - 2)
                blend = smooth(coordinate - lower)
                target = (1. - blend) * commands[lower] + blend * commands[lower + 1] + (1. - smooth(phase)) * offset
                guard.advance(task.execution, target)
                if task.ready():
                    handoff = task.finish()
                    break
        except (RuntimeError, UnsafeState, InfeasiblePositionTarget) as error:
            case["failure"] = task.failure or [str(error)]
        case.update(handoff=handoff, failure=case.get("failure", task.failure), duration_s=float(task.data.time),
                    full_transition_validated=bool(handoff and handoff["success"] and not task.failure))
        window = np.asarray(task.samples[start:])
        valid = window[:-1] if task.failure else window
        case.update(both_hands_loaded_s=longest_interval(np.all(valid[:, 4:6] > 5., axis=1)) if len(valid) else 0.,
                    min_mean_hand_gap_m=float(np.min(valid[:, 8:10].mean(axis=1))) if len(valid) else None)
        trajectory = output / f"duration_{duration:g}.npz"
        task.save_and_replay(trajectory)
        case.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory), replay_passed=True)
        print("duration", duration, "failure", case["failure"], "both_hands_s", case["both_hands_loaded_s"],
              "min_gap", case["min_mean_hand_gap_m"], "full", case["full_transition_validated"], flush=True)
        report["full_transition_validated"] = case["full_transition_validated"]
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if case["full_transition_validated"]:
            break
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
