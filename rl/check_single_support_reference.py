import json
from datetime import datetime
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares, linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth


def quaternion(angles):
    result = np.array([1., 0., 0., 0.])
    for axis, angle in zip(np.eye(3), angles):
        rotation = np.r_[np.cos(angle / 2.), axis * np.sin(angle / 2.)]
        product = np.zeros(4)
        mujoco.mju_mulQuat(product, result, rotation)
        result = product
    return result


def support_polygon(model, supports, positions, rotations):
    corners = []
    for geom, position, rotation in zip(supports, positions, rotations, strict=True):
        size = model.geom_size[geom].copy()
        size[:2] -= .005
        corners.append([(position + rotation @ (size * [sign_x, sign_y, -1.]))[:2]
                        for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))])
    return np.asarray(corners)


def plan(task, step_distance=0., step_yaw=0., side=0, height_tolerance=0., bounded_continuation=False,
         balance_region=False):
    model, data = task.model, mujoco.MjData(task.model)
    initial = task.data.qpos.copy()
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    com = data.subtree_com[pelvis].copy()
    feet = data.geom_xpos[task.supports[2:]].copy()
    rotations = data.geom_xmat[task.supports[2:]].copy()
    angles = Rotation.from_quat(np.r_[initial[4:7], initial[3]]).as_euler("XYZ")
    values = np.r_[initial[:3], angles, initial[task.addresses]]
    lower = np.r_[[-.3, -.4, .3], [-.7, -.5, -.5], model.jnt_range[task.joints, 0] + .02]
    upper = np.r_[[.4, .4, .85], [.7, 1., .5], model.jnt_range[task.joints, 1] - .02]
    nodes = []
    stepping = step_distance != 0. or step_yaw != 0.
    for phase in np.linspace(0., 1., 61 if stepping else 41):
        goals = feet.copy()
        rotation_goals = rotations.reshape(2, 3, 3).copy()
        com_goal = com.copy()
        com_goal[:2] += smooth(phase / (.3 if stepping else .5)) * (feet[1 - side, :2] - com[:2])
        goals[side, 2] += .04 * smooth((phase - (.3 if stepping else .5)) / (.2 if stepping else .5))
        if stepping:
            sign = 1. if side == 0 else -1.
            translation = smooth((phase - .5) / .25)
            goals[side, 1] += sign * step_distance * translation
            angle = sign * step_yaw * translation
            rotation_goals[side] = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                            [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]) @ rotation_goals[side]
            goals[side, 2] -= .04 * smooth((phase - .75) / .15)
            centered = com.copy()
            centered[1] += sign * step_distance / 2.
            com_goal[:2] += smooth((phase - .9) / .1) * (centered[:2] - com_goal[:2])
        previous = values.copy()
        region = None
        if balance_region:
            corners = support_polygon(model, task.supports[2:], goals, rotation_goals)
            availability = 1. - smooth(phase / .3) + smooth((phase - .9) / .1)
            corners[side] = availability * corners[side] + (1. - availability) * corners[1 - side]
            region = ConvexHull(corners.reshape(-1, 2)).equations

        def update(candidate):
            data.qpos[:3] = candidate[:3]
            data.qpos[3:7] = quaternion(candidate[3:6])
            data.qpos[task.addresses] = candidate[6:]
            mujoco.mj_forward(model, data)

        def residual(candidate):
            update(candidate)
            depth = max([0.] + [-contact.dist for contact in data.contact if task.floor not in (contact.geom1, contact.geom2)])
            com_error = data.subtree_com[pelvis] - com_goal
            com_residual = 200. * com_error
            if height_tolerance:
                com_residual = np.r_[200. * com_error[:2], 20. * com_error[2],
                                     1000. * max(0., abs(com_error[2]) - height_tolerance)]
            if balance_region:
                com_residual = np.r_[10. * com_error[:2],
                                     (20. if height_tolerance else 200.) * com_error[2],
                                     1000. * max(0., abs(com_error[2]) - height_tolerance) if height_tolerance else 0.,
                                     2000. * np.maximum(0., region[:, :2] @ data.subtree_com[pelvis, :2] + region[:, 2] + .002)]
            return np.r_[500. * (data.geom_xpos[task.supports[2:]] - goals).ravel(),
                         100. * (data.geom_xmat[task.supports[2:]].reshape(2, 3, 3) - rotation_goals).ravel(),
                         com_residual, .1 * (candidate - previous), .2 * candidate[3:6],
                         (10000. if height_tolerance else 1000.) * max(0., depth - .0005)]

        node_lower, node_upper = lower.copy(), upper.copy()
        if bounded_continuation:
            trust = np.r_[np.full(3, .03), np.full(3, .1), np.full(model.nu, .12)]
            node_lower, node_upper = np.maximum(lower, previous - trust), np.minimum(upper, previous + trust)
        if bounded_continuation and phase == 0.:
            result = SimpleNamespace(x=values.copy(), success=True)
        else:
            result = least_squares(residual, values, bounds=(node_lower, node_upper), max_nfev=150)
        values = result.x
        update(values)
        force_jacs = []
        double_support = phase <= (.3 if stepping else .5) or (stepping and phase >= .9)
        for geom in (task.supports[2:] if double_support else [task.supports[3 - side]]):
            rotation = data.geom_xmat[geom].reshape(3, 3)
            support_size = model.geom_size[geom].copy()
            if height_tolerance or balance_region:
                support_size[:2] -= .005
            for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                point = data.geom_xpos[geom] + rotation @ (support_size * [sign_x, sign_y, -1.])
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, point, model.geom_bodyid[geom])
                force_jacs.append(jac[2])
        matrix = np.asarray(force_jacs).T
        limits = np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1])
        joint_matrix = matrix[model.jnt_dofadr[task.joints]]
        joint_bias = data.qfrc_bias[model.jnt_dofadr[task.joints]]
        solution = linprog(np.zeros(len(force_jacs)), A_eq=matrix[:6], b_eq=data.qfrc_bias[:6],
                           A_ub=np.r_[-joint_matrix, joint_matrix] if balance_region else None,
                           b_ub=np.r_[.7 * limits - joint_bias, .7 * limits + joint_bias] if balance_region else None,
                           bounds=(0., None), method="highs")
        torque = (data.qfrc_bias - matrix @ solution.x)[model.jnt_dofadr[task.joints]] if solution.success else None
        error = float(np.max(np.linalg.norm(data.geom_xpos[task.supports[2:]] - goals, axis=1)))
        com_error = float(np.linalg.norm(data.subtree_com[pelvis] - com_goal))
        nodes.append({"qpos": data.qpos.tolist(), "foot_error_m": error, "com_error_m": com_error,
                      "com_error_xy_m": float(np.linalg.norm(data.subtree_com[pelvis][:2] - com_goal[:2])),
                      "com_height_error_m": float(abs(data.subtree_com[pelvis][2] - com_goal[2])),
                      "optimizer_success": bool(result.success),
                      "support_margin_m": .005 if height_tolerance or balance_region else 0.,
                      "balance_region": balance_region,
                      "support_region_violation_m": None if region is None else float(np.max(
                          region[:, :2] @ data.subtree_com[pelvis, :2] + region[:, 2])),
                      "self_penetration_m": max([0.] + [-float(contact.dist) for contact in data.contact
                                                       if task.floor not in (contact.geom1, contact.geom2)]),
                      "equilibrium_feasible": bool(solution.success), "torque_nm": None if torque is None else torque.tolist(),
                      "phase": float(phase)})
    return nodes


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    output = ROOT / "output/rl/single_support_reference" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(__file__), "cases": []}
    try:
        nodes = plan(task)
        report["nodes"] = nodes
        print("max errors", max(node["foot_error_m"] for node in nodes), max(node["com_error_m"] for node in nodes),
              "static", sum(node["equilibrium_feasible"] for node in nodes), flush=True)
        angles = np.array([node["qpos"] for node in nodes])[:, task.addresses]
        torques = np.array([node["torque_nm"] if node["torque_nm"] is not None else np.zeros(task.model.nu) for node in nodes])
        for feedforward in (False, True):
            task.reset(state)
            guard = PositionTorqueGuard(task.model)
            case = {"feedforward": feedforward}
            for step in range(550):
                coordinate = np.clip(((step + 1) * .02 - .5) / 8., 0., 1.) * (len(nodes) - 1)
                lower = min(int(coordinate), len(nodes) - 2)
                fraction = coordinate - lower
                target = (1. - fraction) * angles[lower] + fraction * angles[lower + 1]
                if feedforward:
                    target += ((1. - fraction) * torques[lower] + fraction * torques[lower + 1]) / task.model.actuator_gainprm[:, 0]
                try:
                    guard.advance(task.execution, target)
                except (UnsafeState, InfeasiblePositionTarget) as error:
                    case["failure"] = task.failure or [str(error)]
                    break
            samples = np.array(task.samples)
            case.update(duration_s=float(task.data.time), max_left_gap_m=float(samples[:, 10].max()),
                        physics_passed=not bool(task.failure))
            trajectory = output / f"feedforward_{feedforward}.npz"
            task.save_and_replay(trajectory)
            case.update(trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory))
            report["cases"].append(case)
            print(case, flush=True)
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
