import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from artifact_utils import digest
from check_transition_route_envelope import candidate, static_capacity
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state, smooth


SOURCE = ROOT / "output/rl/arm_preparation/2026-10-29_11-59-34-571823/report.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, default=SOURCE)
    parser.add_argument("--pair-collisions", action="store_true")
    parser.add_argument("--free-com", action="store_true")
    parser.add_argument("--free-foot-yaw", action="store_true")
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
    sizes = task.model.geom_size[task.supports[2:]].copy()
    pivots = np.array([foot + rotation @ (size * [1., 0., -1.])
                       for foot, rotation, size in zip(feet, rotations, sizes, strict=True)])
    output = ROOT / "output/rl/heel_roll_routes" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"mode": "offline_contact_geometry_screen", "full_transition_validated": False,
              "source_report": str(source_path.relative_to(ROOT)), "source_report_sha256": digest(source_path),
              "pair_collisions": args.pair_collisions,
              "free_com": args.free_com,
              "free_foot_yaw": args.free_foot_yaw,
              "initial_feet_xyz": feet.tolist(), "initial_hands_xyz": hands.tolist(),
              "source_trajectory_sha256": digest(trajectory), "routes": [],
              "limitations": "Local continuation, not physical execution. Vertical-force LP is a conservative screening model, not a dynamics certificate. No airborne hand forces are admitted.",
              "sources": {name: (ROOT / "rl" / name).read_text() for name in
                          ("check_heel_roll_routes.py", "check_transition_route_envelope.py")}}
    families = (("flat_close", 0., .15), ("toe_close", .8, .15), ("toe_forward", .8, .3),
                ("crouch_close", 0., .15), ("crouch_forward", 0., .3))
    if args.free_com:
        families = (("free_close", 0., .15), ("free_forward", 0., .3), ("free_toe", .8, .3))
    for name, pitch, reach in families:
        route = {"family": name, "nodes": []}
        report["routes"].append(route)
        seed = initial.copy()
        for phase in np.linspace(0., 1., 21):
            blend = smooth(phase)
            desired_rotations = np.array([rotation @ Rotation.from_euler("y", pitch * blend).as_matrix()
                                          for rotation in rotations])
            desired_feet = np.array([pivot - rotation @ (size * [1., 0., -1.])
                                     for pivot, rotation, size in zip(pivots, desired_rotations, sizes, strict=True)])
            goals = hands.copy()
            goals[:, 0] += blend * (feet[:, 0].mean() + reach - hands[:, 0])
            goals[:, 1] += blend * (feet[:, 1].mean() + np.array([.27, -.27]) - hands[:, 1])
            goals[:, 2] += blend * (.02005 - hands[:, 2])
            desired_com = com.copy()
            desired_com[:2] += (smooth(min(1., phase / .25)) if pitch else 0.) * (pivots[:, :2].mean(axis=0) - com[:2])
            node = candidate(task, initial, desired_feet, desired_rotations, goals, seed,
                             "free_com" if args.free_com else "feet_center", com_goal=desired_com, continuous=True,
                             base_height_goal=initial[2] + blend * (.25 - initial[2]) if name.startswith("crouch") or args.free_com else None,
                             pair_collisions=args.pair_collisions, free_foot_yaw=args.free_foot_yaw)
            scratch.qpos[:] = node["qpos"]
            mujoco.mj_forward(task.model, scratch)
            node.pop("feet_only")
            node.pop("after_hand_contact_hypothesis")
            node.update(phase=float(phase), active_capacity=static_capacity(task, scratch, True, active_only=True),
                        com_error_m=float(np.linalg.norm(scratch.subtree_com[task.model.body("pelvis").id, :2] - desired_com[:2])))
            node["internal_contacts"] = [{"geoms": [task.model.geom(contact.geom1).name, task.model.geom(contact.geom2).name],
                                          "depth_m": float(-contact.dist)} for contact in scratch.contact
                                         if task.floor not in (contact.geom1, contact.geom2) and contact.dist < 0.]
            node["passed"] = bool(node["geometry_passed"] and (args.free_com or node["com_error_m"] < .005))
            route["nodes"].append(node)
            seed = np.array(node["qpos"])
            np.testing.assert_array_equal(integration_state(task), before)
            print(name, round(float(phase), 2), node["passed"], "hand_error", round(node["hand_error_m"], 4),
                  "active", node["active_capacity"]["support_point_counts"], flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        route["geometry_path_passed"] = all(node["passed"] for node in route["nodes"])
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
