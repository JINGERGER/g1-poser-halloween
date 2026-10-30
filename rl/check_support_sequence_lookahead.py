import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_predictive_step_chain import reference_gates
from check_single_support_reference import plan
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source_path = args.report.resolve()
    source = json.loads(source_path.read_text())
    assert source["passed"] and source["replay_passed"]
    trajectory = ROOT / source["trajectory"]
    assert digest(trajectory) == source["trajectory_sha256"]
    torch.set_num_threads(1)
    task = StandTransitionTask()
    task.make_states([0])
    before = integration_state(task)
    scratch = SimpleNamespace(model=task.model, data=mujoco.MjData(task.model), joints=task.joints,
                              addresses=task.addresses, supports=task.supports, floor=task.floor)
    with np.load(trajectory) as saved:
        initial = saved["qpos"][-1].copy()
    output = ROOT / "output/rl/support_sequence_lookahead" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_support_sequence_lookahead.py", "check_single_support_reference.py", "check_predictive_step_chain.py")
    report = {"status": "running", "mode": "offline_three_step_sequence_screening", "full_transition_validated": False,
              "source_report": str(source_path.relative_to(ROOT)), "source_report_sha256": digest(source_path),
              "source_trajectory_sha256": digest(trajectory), "edges": [],
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    frontier = [((), initial)]
    for depth in range(3):
        next_frontier = []
        for prefix, pose in frontier:
            for side in (0, 1):
                scratch.data.qpos[:] = pose
                mujoco.mj_forward(task.model, scratch.data)
                attempts = []
                for tolerance in (0., .02):
                    nodes = plan(scratch, .04, .2, side=side, height_tolerance=tolerance,
                                 bounded_continuation=True, balance_region=True)
                    gates = reference_gates(nodes, adaptive_height=bool(tolerance))
                    poses = np.array([node["qpos"] for node in nodes])
                    gates["initial_continuity"] = bool(np.max(np.abs(poses[0] - pose)) < 1e-10)
                    gates["bounded_joints"] = bool(np.max(np.abs(np.diff(poses[:, task.addresses], axis=0))) <= .120001)
                    attempts.append({"height_tolerance_m": tolerance, "gates": gates, "nodes": nodes})
                    if all(gates.values()):
                        break
                sequence = prefix + (("left", "right")[side],)
                passed = all(gates.values())
                report["edges"].append({"sequence": sequence, "passed": passed, "attempts": attempts})
                if passed:
                    next_frontier.append((sequence, poses[-1]))
                np.testing.assert_array_equal(integration_state(task), before)
                print(sequence, passed, gates, flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        frontier = next_frontier
        if not frontier:
            break
    report.update(status="completed", surviving_sequences=[sequence for sequence, pose in frontier])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
