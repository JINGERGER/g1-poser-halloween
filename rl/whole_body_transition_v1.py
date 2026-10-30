import mujoco
import numpy as np
import osqp
from scipy import sparse


def rotation_error(desired, actual):
    delta = desired @ actual.T
    return .5 * np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])


class WholeBodyLowering:
    def __init__(self, task):
        self.task = task
        model, data = task.model, task.data
        self.feet = task.supports[2:]
        self.foot_positions = data.geom_xpos[self.feet].copy()
        self.foot_rotations = data.geom_xmat[self.feet].reshape(2, 3, 3).copy()
        self.pelvis = model.body("pelvis").id
        self.initial_com = data.subtree_com[self.pelvis].copy()
        self.initial_hands = data.geom_xpos[task.supports[:2]].copy()
        self.posture = data.qpos[task.addresses].copy()
        self.previous_com_jac = None
        self.last = None

    def target(self, com_height, pitch, hand_phase):
        task = self.task
        model, data = task.model, task.data
        velocity_count, actuator_count = model.nv, model.nu
        variable_count = velocity_count + actuator_count + 24
        force_start = velocity_count + actuator_count
        dofs = model.jnt_dofadr[task.joints]
        limits = np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1]) * .8
        quadratic = np.diag(np.r_[np.full(velocity_count, .001), np.full(actuator_count, .00001), np.full(24, .000001)])
        linear = np.zeros(variable_count)
        constraints, lower, upper = [], [], []

        def objective(jacobian, acceleration, weight):
            padded = np.zeros((len(acceleration), variable_count))
            padded[:, :velocity_count] = jacobian
            nonlocal quadratic, linear
            quadratic += weight * padded.T @ padded
            linear -= weight * padded.T @ acceleration

        def jacobian(point, body):
            position, rotation = np.zeros((3, velocity_count)), np.zeros((3, velocity_count))
            position_dot, rotation_dot = np.zeros((3, velocity_count)), np.zeros((3, velocity_count))
            mujoco.mj_jac(model, data, position, rotation, point, body)
            mujoco.mj_jacDot(model, data, position_dot, rotation_dot, point, body)
            return position, rotation, position_dot, rotation_dot

        mass = np.zeros((velocity_count, velocity_count))
        mujoco.mj_fullM(model, mass, data.qM)
        dynamics = np.zeros((velocity_count, variable_count))
        dynamics[:, :velocity_count] = mass
        dynamics[dofs, velocity_count + np.arange(actuator_count)] = -1.
        force_index = 0
        for foot_index, geom in enumerate(self.feet):
            body = model.geom_bodyid[geom]
            position, rotation, position_dot, rotation_dot = jacobian(data.geom_xpos[geom], body)
            foot_rotation = data.geom_xmat[geom].reshape(3, 3)
            contact_acceleration = np.r_[100. * (self.foot_positions[foot_index] - data.geom_xpos[geom]) - 20. * position @ data.qvel - position_dot @ data.qvel,
                                        100. * rotation_error(self.foot_rotations[foot_index], foot_rotation) - 20. * rotation @ data.qvel - rotation_dot @ data.qvel]
            contact_constraint = np.zeros((6, variable_count))
            contact_constraint[:, :velocity_count] = np.vstack([position, rotation])
            constraints.append(contact_constraint)
            lower.extend(contact_acceleration)
            upper.extend(contact_acceleration)
            for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                local = model.geom_size[geom] * [sign_x, sign_y, -1.]
                point = data.geom_xpos[geom] + foot_rotation @ local
                point_jac, unused, unused_dot, unused_rotation_dot = jacobian(point, body)
                start = force_start + force_index * 3
                dynamics[:, start:start + 3] = -point_jac.T
                friction = np.zeros((5, variable_count))
                friction[:, start:start + 3] = [[1., 0., -.6], [-1., 0., -.6], [0., 1., -.6], [0., -1., -.6], [0., 0., 1.]]
                constraints.append(friction)
                lower.extend([-np.inf] * 4 + [0.])
                upper.extend([0.] * 4 + [600.])
                force_index += 1
        constraints.append(dynamics)
        rhs = data.qfrc_passive - data.qfrc_bias
        lower.extend(rhs)
        upper.extend(rhs)
        bounds = np.zeros((actuator_count, variable_count))
        bounds[:, velocity_count:force_start] = np.eye(actuator_count)
        constraints.append(bounds)
        lower.extend(-limits)
        upper.extend(limits)
        com_jac = np.zeros((3, velocity_count))
        mujoco.mj_jacSubtreeCom(model, data, com_jac, self.pelvis)
        com_bias = np.zeros(3) if self.previous_com_jac is None else (com_jac - self.previous_com_jac) @ data.qvel / .02
        self.previous_com_jac = com_jac.copy()
        com_goal = self.initial_com.copy()
        com_goal[2] = com_height
        objective(com_jac, 80. * (com_goal - data.subtree_com[self.pelvis]) - 18. * com_jac @ data.qvel - com_bias, 10.)
        desired_rotation = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0., 1., 0.], [-np.sin(pitch), 0., np.cos(pitch)]])
        unused, torso_jac, unused_dot, torso_dot = jacobian(data.xpos[self.pelvis], self.pelvis)
        torso_acc = 80. * rotation_error(desired_rotation, data.xmat[self.pelvis].reshape(3, 3)) - 18. * torso_jac @ data.qvel - torso_dot @ data.qvel
        objective(torso_jac, torso_acc, 5.)
        joint_jac = np.eye(velocity_count)[dofs]
        objective(joint_jac, 20. * (self.posture - data.qpos[task.addresses]) - 8. * data.qvel[dofs], .03)
        for hand_index, geom in enumerate(task.supports[:2]):
            position, unused, position_dot, unused_dot = jacobian(data.geom_xpos[geom], model.geom_bodyid[geom])
            final = np.array([.4, .32 if hand_index == 0 else -.32, .02])
            goal = self.initial_hands[hand_index] + hand_phase * (final - self.initial_hands[hand_index])
            objective(position, 80. * (goal - data.geom_xpos[geom]) - 18. * position @ data.qvel - position_dot @ data.qvel, 3.)
        inequality = np.zeros((actuator_count, variable_count))
        inequality[:, dofs] = np.eye(actuator_count)
        constraints.append(inequality)
        lower.extend(np.maximum(-200., 2. * (model.jnt_range[task.joints, 0] - data.qpos[task.addresses] - .02 * data.qvel[dofs]) / .02 ** 2))
        upper.extend(np.minimum(200., 2. * (model.jnt_range[task.joints, 1] - data.qpos[task.addresses] - .02 * data.qvel[dofs]) / .02 ** 2))
        constraint_matrix = np.vstack(constraints)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(quadratic)), q=linear, A=sparse.csc_matrix(constraint_matrix),
                     l=np.array(lower), u=np.array(upper), verbose=False, eps_abs=1e-4, eps_rel=1e-5,
                     max_iter=10000, polishing=True)
        result = solver.solve(raise_error=False)
        if result.info.status != "solved" or result.x is None:
            raise RuntimeError(f"Whole-body QP: {result.info.status}")
        residual = constraint_matrix @ result.x
        violation = max(0., float(np.max(np.array(lower) - residual)), float(np.max(residual - np.array(upper))))
        if violation > .02:
            raise RuntimeError("Whole-body QP constraint residual exceeds .02")
        torque = result.x[velocity_count:force_start]
        affine_bias = model.actuator_biasprm[:, 0] + model.actuator_biasprm[:, 1] * data.actuator_length + model.actuator_biasprm[:, 2] * data.actuator_velocity
        target = (torque - affine_bias) / model.actuator_gainprm[:, 0]
        self.last = {"constraint_residual": violation, "torque_nm": torque.tolist(), "predicted_forces_n": result.x[force_start:].reshape(-1, 3).tolist()}
        return np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
