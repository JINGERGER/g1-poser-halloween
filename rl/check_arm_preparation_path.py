import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_transition_route_envelope import SOURCE, candidate
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state, smooth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, default=SOURCE)
    args = parser.parse_args()
    source_path = args.source_report.resolve()
    torch.set_num_threads(1)
    task = StandTransitionTask()
    task.make_states([0])
    before = integration_state(task)
    source = json.loads(source_path.read_text())
    assert source["passed"] and source["replay_passed"]
    trajectory = ROOT / source["trajectory"]
    assert digest(trajectory) == source["trajectory_sha256"]
    with np.load(trajectory) as saved:
        initial = saved["qpos"][-1].copy()
    scratch = mujoco.MjData(task.model)
    scratch.qpos[:] = initial
    mujoco.mj_forward(task.model, scratch)
    feet = scratch.geom_xpos[task.supports[2:]].copy()
    rotations = scratch.geom_xmat[task.supports[2:]].reshape(2, 3, 3).copy()
    hands = scratch.geom_xpos[task.supports[:2]].copy()
    com = scratch.subtree_com[task.model.body("pelvis").id].copy()
    center = feet.mean(axis=0)
    hand_target = np.array([[center[0] + .3, center[1] + .27, .6],
                            [center[0] + .3, center[1] - .27, .6]])
    output = ROOT / "output/rl/arm_preparation_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_arm_preparation_path.py", "check_transition_route_envelope.py")
    report = {"status": "running", "mode": "offline_reference_screening_only", "full_transition_validated": False,
              "source_report": str(source_path.relative_to(ROOT)), "source_report_sha256": digest(source_path),
              "hand_target_xyz": hand_target.tolist(), "routes": [],
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        for family in ("diagonal", "front_then_down"):
            seed = initial.copy()
            route = {"family": family, "nodes": []}
            report["routes"].append(route)
            for phase in np.linspace(0., 1., 31):
                horizontal = smooth(phase if family == "diagonal" else phase / .4)
                vertical = smooth(phase if family == "diagonal" else (phase - .4) / .6)
                goals = hands.copy()
                goals[:, :2] += horizontal * (hand_target[:, :2] - hands[:, :2])
                goals[:, 2] += vertical * (hand_target[:, 2] - hands[:, 2])
                node = candidate(task, initial, feet, rotations, goals, seed, "feet_center",
                                 base_height_goal=initial[2], com_goal=com, continuous=True)
                node["phase"] = float(phase)
                node["joint_step_rad"] = float(np.max(np.abs(np.array(node["qpos"])[task.addresses] - seed[task.addresses])))
                node["passed"] = bool(node["geometry_passed"] and node["feet_only"]["feasible"]
                                      and node["base_height_error_m"] < .005 and node["joint_step_rad"] <= .200001)
                route["nodes"].append(node)
                seed = np.array(node["qpos"])
                np.testing.assert_array_equal(integration_state(task), before)
                print(family, round(float(phase), 3), node["passed"], "hand_error", round(node["hand_error_m"], 4), flush=True)
            route["nodes_passed"] = all(node["passed"] for node in route["nodes"])
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
