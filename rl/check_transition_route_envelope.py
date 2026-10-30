import argparse
import json
from datetime import datetime
from time import perf_counter

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state


SOURCE = ROOT / "output/rl/predictive_descent/2026-10-29_11-17-06-279260/report.json"


def static_capacity(task, data, include_hands, active_only=False):
    model = task.model
    columns, labels = [], []
    for support_index, geom in enumerate(task.supports):
        if support_index < 2 and not include_hands:
            continue
        if active_only and not any(contact.dist <= 0. and {contact.geom1, contact.geom2} == {geom, task.floor}
                                   for contact in data.contact):
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if support_index < 2:
            points = [data.geom_xpos[geom] - [0., 0., model.geom_size[geom, 0]]]
        else:
            size = model.geom_size[geom].copy()
            if not active_only:
                size[:2] -= .005
            points = [data.geom_xpos[geom] + rotation @ (size * [sign_x, sign_y, -1.])
                      for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))]
        for point in points:
            normal = data.geom_xmat[task.floor].reshape(3, 3)[:, 2]
            gap = float(normal @ (point - data.geom_xpos[task.floor]))
            if active_only and not -.001 <= gap <= .0001:
                continue
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, point, model.geom_bodyid[geom])
            columns.append(jacobian[2])
            labels.append(support_index)
    limits = np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1])
    matrix = np.zeros((model.nv, model.nu + len(columns)))
    matrix[model.jnt_dofadr[task.joints], np.arange(model.nu)] = 1.
    if columns:
        matrix[:, model.nu:] = np.asarray(columns).T
    result = linprog(np.zeros(matrix.shape[1]), A_eq=matrix, b_eq=data.qfrc_bias - data.qfrc_passive,
                     bounds=list(zip(-.7 * limits, .7 * limits)) + [(0., None)] * len(columns), method="highs")
    capacity = {"feasible": bool(result.success), "static_torque_limit_fraction": .7,
                "hypothetical_hand_support": include_hands and not active_only,
                "active_geometry_only": active_only, "support_point_counts": np.bincount(labels, minlength=4).tolist()}
    if result.success:
        loads = np.zeros(4)
        for label, force in zip(labels, result.x[model.nu:], strict=True):
            loads[label] += force
        capacity.update(loads_n=loads.tolist(), torque_ratio=float(np.max(np.abs(result.x[:model.nu]) / limits)))
    return capacity


def candidate(task, initial, feet, foot_rotations, hand_goals, seed, balance, base_height_goal=None, com_goal=None, continuous=False,
              pair_collisions=False, free_foot_yaw=False):
    model, data = task.model, mujoco.MjData(task.model)
    pelvis = model.body("pelvis").id
    angles = Rotation.from_quat(np.r_[seed[4:7], seed[3]]).as_rotvec()
    values = np.r_[seed[:3], angles, seed[task.addresses]]
    lower = np.r_[[-.5, -.4, .17], [-.7, -.6, -.7], model.jnt_range[task.joints, 0] + .02]
    upper = np.r_[[.5, .4, .8], [.7, 1.9, .7], model.jnt_range[task.joints, 1] - .02]
    if continuous:
        trust = np.r_[np.full(3, .05), np.full(3, .15), np.full(model.nu, .2)]
        lower, upper = np.maximum(lower, values - trust), np.minimum(upper, values + trust)
        if np.any(lower >= upper):
            raise ValueError("Continuation seed outside optimization bounds")
    values = np.clip(values, lower + 1e-8, upper - 1e-8)

    def update(candidate_values):
        data.qpos[:3] = candidate_values[:3]
        rotation = Rotation.from_rotvec(candidate_values[3:6]).as_quat()
        data.qpos[3:7] = np.r_[rotation[3], rotation[:3]]
        data.qpos[task.addresses] = candidate_values[6:]
        mujoco.mj_forward(model, data)

    def depths():
        internal, other = 0., 0.
        for contact in data.contact:
            if task.floor not in (contact.geom1, contact.geom2):
                internal = max(internal, -float(contact.dist))
            elif (contact.geom2 if contact.geom1 == task.floor else contact.geom1) not in task.supports:
                other = max(other, -float(contact.dist))
        return internal, other

    def residual(candidate_values):
        update(candidate_values)
        internal, other = depths()
        collision_cost = np.array([10000. * max(0., internal - .0005)])
        if pair_collisions:
            pairs = np.zeros((model.ngeom, model.ngeom))
            for contact in data.contact:
                if task.floor not in (contact.geom1, contact.geom2):
                    lower_geom, upper_geom = sorted((contact.geom1, contact.geom2))
                    pairs[lower_geom, upper_geom] = max(pairs[lower_geom, upper_geom], -float(contact.dist) - .0005)
            collision_cost = 10000. * pairs[np.triu_indices(model.ngeom, 1)]
        desired_com = feet[:, :2].mean(axis=0) if com_goal is None else np.asarray(com_goal)[:2]
        rotation_delta = data.geom_xmat[task.supports[2:]].reshape(2, 3, 3) - foot_rotations
        if free_foot_yaw:
            rotation_delta = rotation_delta[:, :, 2]
        return np.r_[500. * (data.geom_xpos[task.supports[2:]] - feet).ravel(),
                     100. * rotation_delta.ravel(),
                     200. * (data.geom_xpos[task.supports[:2]] - hand_goals).ravel(),
                     (150. if balance == "feet_center" else 0.) * (data.subtree_com[pelvis, :2] - desired_com),
                     0. if base_height_goal is None else 100. * (data.qpos[2] - base_height_goal),
                     .1 * (candidate_values[6:] - initial[task.addresses]),
                     .1 * (candidate_values[4] - 1.3),
                     collision_cost, 10000. * other]

    started = perf_counter()
    result = least_squares(residual, values, bounds=(lower, upper), max_nfev=160,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    update(result.x)
    internal, other = depths()
    foot_error = float(np.max(np.linalg.norm(data.geom_xpos[task.supports[2:]] - feet, axis=1)))
    hand_error = float(np.max(np.linalg.norm(data.geom_xpos[task.supports[:2]] - hand_goals, axis=1)))
    rotation_delta = data.geom_xmat[task.supports[2:]].reshape(2, 3, 3) - foot_rotations
    rotation_error = float(np.max(np.abs(rotation_delta[:, :, 2] if free_foot_yaw else rotation_delta)))
    gaps = [float(mujoco.mj_geomDistance(model, data, geom, task.floor, 2., None)) for geom in task.supports]
    margins = np.minimum(data.qpos[task.addresses] - model.jnt_range[task.joints, 0],
                         model.jnt_range[task.joints, 1] - data.qpos[task.addresses])
    near_limits = [{"joint": model.joint(int(task.joints[index])).name, "margin_rad": float(margins[index])}
                   for index in np.argsort(margins)[:4]]
    geometry = bool(foot_error < .001 and hand_error < .005 and rotation_error < .005
                    and internal <= .001 and other == 0. and all(gap >= 0. for gap in gaps[:2])
                    and all(gap >= -.005 for gap in gaps[2:]))
    return {"qpos": data.qpos.tolist(), "solver_success": bool(result.success), "evaluations": int(result.nfev),
            "wall_time_s": perf_counter() - started, "foot_error_m": foot_error, "hand_error_m": hand_error,
            "foot_rotation_error": rotation_error, "free_foot_yaw": free_foot_yaw,
            "self_penetration_m": internal, "other_penetration_m": other,
            "support_gaps_m": gaps, "near_joint_limits": near_limits, "geometry_passed": geometry,
            "com_xyz": data.subtree_com[pelvis].tolist(), "feet_only": static_capacity(task, data, False),
            "base_height_error_m": None if base_height_goal is None else float(abs(data.qpos[2] - base_height_goal)),
            "after_hand_contact_hypothesis": static_capacity(task, data, True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-layout", choices=("crawl", "near"), default="crawl")
    args = parser.parse_args()
    torch.set_num_threads(1)
    task = StandTransitionTask()
    task.make_states([0])
    before = integration_state(task)
    source = json.loads(SOURCE.read_text())
    assert source["passed"] and source["replay_passed"]
    trajectory = ROOT / source["trajectory"]
    assert digest(trajectory) == source["trajectory_sha256"]
    with np.load(trajectory) as saved:
        initial = saved["qpos"][-1].copy()
    scratch = mujoco.MjData(task.model)
    scratch.qpos[:] = initial
    mujoco.mj_forward(task.model, scratch)
    actual_feet = scratch.geom_xpos[task.supports[2:]].copy()
    actual_rotations = scratch.geom_xmat[task.supports[2:]].reshape(2, 3, 3).copy()
    center = actual_feet.mean(axis=0)
    initial_supports = scratch.geom_xpos[task.supports].copy()
    scratch.qpos[:] = task.nominal
    mujoco.mj_forward(task.model, scratch)
    crawl_supports = scratch.geom_xpos[task.supports].copy()
    crawl_center = crawl_supports[2:].mean(axis=0)
    crawl_seed = task.nominal.copy()
    crawl_seed[:2] += center[:2] - crawl_center[:2]
    hand_goals = crawl_supports[:2].copy()
    hand_goals[:, :2] += center[:2] - crawl_center[:2]
    hand_goals[:, 2] = task.model.geom_size[task.supports[:2], 0] + .005
    if args.hand_layout == "near":
        hand_goals[:, 0] = center[0] + .3
        hand_goals[:, 1] = center[1] + np.array([.27, -.27])
    layouts = (("current", float(actual_feet[0, 1] - actual_feet[1, 1]), .2),
               ("intermediate", .55, .7),
               ("crawl_width", float(crawl_supports[2, 1] - crawl_supports[3, 1]), 1.4))
    output = ROOT / "output/rl/transition_route_envelope" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "mode": "offline_geometry_and_static_screening_only", "full_transition_validated": False,
              "source_report": str(SOURCE.relative_to(ROOT)), "source_report_sha256": digest(SOURCE),
              "source_trajectory_sha256": digest(trajectory), "source_sha256": digest(__file__),
              "source_snapshot": (ROOT / "rl/check_transition_route_envelope.py").read_text(),
              "initial_supports_xyz": initial_supports.tolist(), "crawl_supports_xyz": crawl_supports.tolist(),
              "hand_goals_xyz": hand_goals.tolist(), "hand_layout": args.hand_layout, "cases": [],
              "limitations": "No state is integrated. Hand-force LP is counterfactual because hands target a 5mm air gap. Failed local solves do not prove infeasibility. Layout changes are not executed steps."}
    try:
        for name, width, yaw in layouts:
            feet = actual_feet.copy()
            feet[:, 1] = center[1] + np.array([width / 2., -width / 2.])
            rotations = np.array([Rotation.from_euler("Z", sign * yaw).as_matrix() for sign in (1., -1.)])
            if name == "current":
                rotations = actual_rotations.copy()
            for balance in ("feet_center", "free_com"):
                for seed_name, seed in (("current", initial), ("crawl", crawl_seed)):
                    print("Screening", name, balance, seed_name, flush=True)
                    result = candidate(task, initial, feet, rotations, hand_goals, seed, balance)
                    result.update(layout=name, width_m=width, yaw_rad=yaw, balance=balance, seed=seed_name,
                                  foot_goals_xyz=feet.tolist(), foot_rotation_goals=rotations.tolist())
                    report["cases"].append(result)
                    np.testing.assert_array_equal(integration_state(task), before)
                    print("geometry", result["geometry_passed"], "feet_only", result["feet_only"]["feasible"],
                          "four_support", result["after_hand_contact_hypothesis"]["feasible"],
                          "hand_error", result["hand_error_m"], "wall_s", result["wall_time_s"], flush=True)
                    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
