import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

from check_single_support_reference import quaternion
from standing_capture_routes import smooth


def plan_descent(task, drop):
    if not 0. < drop <= .1:
        raise ValueError("Descent drop must be in (0, 0.1] meters")
    model, data = task.model, mujoco.MjData(task.model)
    initial = task.data.qpos.copy()
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    initial_com = data.subtree_com[pelvis].copy()
    feet = data.geom_xpos[task.supports[2:]].copy()
    rotations = data.geom_xmat[task.supports[2:]].copy()
    hands = data.geom_xpos[task.supports[:2]].copy()
    angles = Rotation.from_quat(np.r_[initial[4:7], initial[3]]).as_euler("XYZ")
    values = np.r_[initial[:3], angles, initial[task.addresses]]
    lower = np.r_[initial[:2] - .3, .16, angles - .6, model.jnt_range[task.joints, 0] + .02]
    upper = np.r_[initial[:2] + .3, min(.95, initial[2] + .05), angles + .6, model.jnt_range[task.joints, 1] - .02]
    limits = .7 * np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1])
    nodes = []

    def update(candidate):
        data.qpos[:3] = candidate[:3]
        data.qpos[3:7] = quaternion(candidate[3:6])
        data.qpos[task.addresses] = candidate[6:]
        mujoco.mj_forward(model, data)

    def depths():
        internal, other = 0., 0.
        for contact in data.contact:
            if task.floor not in (contact.geom1, contact.geom2):
                internal = max(internal, -float(contact.dist))
            elif (contact.geom2 if contact.geom1 == task.floor else contact.geom1) not in task.supports:
                other = max(other, -float(contact.dist))
        return internal, other

    for phase in np.linspace(0., 1., 31):
        previous = values.copy()
        offset = np.array([0., 0., -drop * smooth(phase)])
        com_goal, hand_goals = initial_com + offset, hands + offset
        base_height = initial[2] + offset[2]

        def residual(candidate):
            update(candidate)
            internal, other = depths()
            return np.r_[500. * (data.geom_xpos[task.supports[2:]] - feet).ravel(),
                         100. * (data.geom_xmat[task.supports[2:]] - rotations).ravel(),
                         200. * (data.subtree_com[pelvis] - com_goal),
                         100. * (data.qpos[2] - base_height),
                         20. * (data.geom_xpos[task.supports[:2]] - hand_goals).ravel(),
                         .1 * (candidate - previous), .1 * (candidate[3:6] - angles),
                         10000. * max(0., internal - .0005), 10000. * other]

        result = least_squares(residual, values, bounds=(lower, upper), max_nfev=200)
        values = result.x
        update(values)
        columns = []
        for geom in task.supports[2:]:
            size = model.geom_size[geom].copy()
            size[:2] -= .005
            rotation = data.geom_xmat[geom].reshape(3, 3)
            for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                point = data.geom_xpos[geom] + rotation @ (size * [sign_x, sign_y, -1.])
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacobian, None, point, model.geom_bodyid[geom])
                columns.append(jacobian[2])
        matrix = np.zeros((model.nv, model.nu + len(columns)))
        matrix[model.jnt_dofadr[task.joints], np.arange(model.nu)] = 1.
        matrix[:, model.nu:] = np.asarray(columns).T
        equilibrium = linprog(np.zeros(matrix.shape[1]), A_eq=matrix,
                              b_eq=data.qfrc_bias - data.qfrc_passive,
                              bounds=list(zip(-limits, limits)) + [(0., None)] * len(columns), method="highs")
        internal, other = depths()
        node = {"phase": float(phase), "qpos": data.qpos.tolist(), "solver_success": bool(result.success),
                "foot_error_m": float(np.max(np.linalg.norm(data.geom_xpos[task.supports[2:]] - feet, axis=1))),
                "foot_rotation_error": float(np.max(np.abs(data.geom_xmat[task.supports[2:]] - rotations))),
                "com_error_m": float(np.linalg.norm(data.subtree_com[pelvis] - com_goal)),
                "base_height_error_m": float(abs(data.qpos[2] - base_height)),
                "self_penetration_m": internal, "other_penetration_m": other,
                "static_equilibrium": bool(equilibrium.success), "support_margin_m": .005,
                "static_torque_fraction": .7}
        node["passed"] = bool(node["solver_success"] and node["foot_error_m"] < .001
                              and node["foot_rotation_error"] < .005 and node["com_error_m"] < .003
                              and node["base_height_error_m"] < .003 and internal <= .001
                              and other == 0. and equilibrium.success)
        nodes.append(node)
    return nodes
