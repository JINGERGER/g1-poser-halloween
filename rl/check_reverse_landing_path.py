import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from artifact_utils import digest
from check_transition_route_envelope import SOURCE, candidate
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state, smooth


ENVELOPE = ROOT / "output/rl/transition_route_envelope/2026-10-29_11-51-50-911235/report.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", type=Path, default=ENVELOPE)
    args = parser.parse_args()
    envelope_path = args.envelope.resolve()
    torch.set_num_threads(1)
    task = StandTransitionTask()
    task.make_states([0])
    before = integration_state(task)
    envelope = json.loads(envelope_path.read_text())
    endpoint = max((case for case in envelope["cases"] if case["layout"] == "intermediate"
                    and case["geometry_passed"] and case["feet_only"]["feasible"] and case["solver_success"]),
                   key=lambda case: case["qpos"][2])
    assert endpoint["geometry_passed"] and endpoint["feet_only"]["feasible"]
    source = json.loads(SOURCE.read_text())
    with np.load(ROOT / source["trajectory"]) as saved:
        initial = saved["qpos"][-1].copy()
    feet = np.array(endpoint["foot_goals_xyz"])
    rotations = np.array([Rotation.from_euler("Z", sign * endpoint["yaw_rad"]).as_matrix() for sign in (1., -1.)])
    low_hands = np.array(envelope["hand_goals_xyz"])
    high_hands = low_hands.copy()
    high_hands[:, 0] = feet[:, 0].mean() + .25
    high_hands[:, 2] = .65
    output = ROOT / "output/rl/reverse_landing_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_reverse_landing_path.py", "check_transition_route_envelope.py")
    report = {"status": "running", "mode": "offline_reverse_continuation_not_physical_execution",
              "full_transition_validated": False, "endpoint_report": str(envelope_path.relative_to(ROOT)),
              "endpoint_report_sha256": digest(envelope_path), "endpoint_case": endpoint, "routes": [],
              "sources": {name: digest(ROOT / "rl" / name) for name in names},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    try:
        for family in ("coordinated", "arms_first_on_descent"):
            seed = np.array(endpoint["qpos"])
            route = {"family": family, "nodes": []}
            report["routes"].append(route)
            for phase in np.linspace(0., 1., 21):
                hand_phase = phase if family == "coordinated" else np.clip((phase - .25) / .75, 0., 1.)
                hand_goals = low_hands + smooth(hand_phase) * (high_hands - low_hands)
                base_height = endpoint["qpos"][2] + smooth(phase) * (.58 - endpoint["qpos"][2])
                node = candidate(task, initial, feet, rotations, hand_goals, seed, "feet_center",
                                 base_height_goal=base_height, com_goal=endpoint["com_xyz"], continuous=True)
                node["phase"] = float(phase)
                node["passed"] = bool(node["geometry_passed"] and node["feet_only"]["feasible"]
                                      and node["base_height_error_m"] < .005)
                node["joint_step_rad"] = float(np.max(np.abs(np.array(node["qpos"])[task.addresses] - seed[task.addresses])))
                route["nodes"].append(node)
                seed = np.array(node["qpos"])
                np.testing.assert_array_equal(integration_state(task), before)
                print(family, round(float(phase), 2), node["passed"], "height_error", round(node["base_height_error_m"], 4),
                      "hand_error", round(node["hand_error_m"], 4), "joint_step", round(node["joint_step_rad"], 3), flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            route["nodes_passed"] = all(node["passed"] for node in route["nodes"])
            route["max_joint_step_rad"] = max(node["joint_step_rad"] for node in route["nodes"])
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
