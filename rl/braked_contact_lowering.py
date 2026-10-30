import mujoco
import numpy as np
import osqp
from scipy import sparse

from measured_support import LoadTransfer, support_contacts


def rotation_error(desired, actual):
    delta = desired @ actual.T
    return .5 * np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])


class BrakedContactLowering:
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
        self.transfer = LoadTransfer(float(model.body_mass.sum() * abs(model.opt.gravity[2])))

    def target(self, com_height, pitch, hand_phase):
        task = self.task
        model, data = task.model, task.data
        velocity_count, actuator_count = model.nv, model.nu
        contacts = support_contacts(task)
        if not contacts:
            raise RuntimeError("No measured support contacts; descent QP disabled")
        force_count = 3 * len(contacts)
        variable_count = velocity_count + actuator_count + force_count
        force_start = velocity_count + actuator_count
        dofs = model.jnt_dofadr[task.joints]
        limits = np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1]) * .8
        quadratic = np.diag(np.r_[np.full(velocity_count, .001), np.full(actuator_count, .00001), np.full(force_count, .000001)])
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
        for force_index, contact in enumerate(contacts):
            geom = contact["geom"]
            point_jac, unused, point_dot, unused_dot = jacobian(contact["point"], model.geom_bodyid[geom])
            frame = contact["frame"]
            start = force_start + force_index * 3
            dynamics[:, start:start + 3] = -point_jac.T @ frame.T
            velocity = point_jac @ data.qvel
            acceleration = -30. * velocity - point_dot @ data.qvel - 100. * contact["distance"] * frame[0]
            objective(point_jac, acceleration, 2000.)
            normal_constraint = np.zeros((1, variable_count))
            normal_constraint[0, :velocity_count] = frame[0] @ point_jac
            normal_acceleration = float(frame[0] @ (-point_dot @ data.qvel - 20. * velocity))
            constraints.append(normal_constraint)
            lower.append(normal_acceleration)
            upper.append(np.inf)
            friction = np.zeros((5, variable_count))
            coefficient = contact["friction"] / np.sqrt(2.)
            friction[:, start:start + 3] = [[-coefficient, 1., 0.], [-coefficient, -1., 0.],
                                          [-coefficient, 0., 1.], [-coefficient, 0., -1.], [1., 0., 0.]]
            support_index = task.supports.index(geom)
            capacity = self.transfer.hand_capacity(support_index) if support_index < 2 else self.transfer.weight * 2.
            count = sum(other["geom"] == geom for other in contacts)
            constraints.append(friction)
            lower.extend([-np.inf] * 4 + [0.])
            upper.extend([0.] * 4 + [capacity / count])
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
            if any(contact["geom"] == geom for contact in contacts):
                continue
            position, unused, position_dot, unused_dot = jacobian(data.geom_xpos[geom], model.geom_bodyid[geom])
            final = np.array([.4, .32 if hand_index == 0 else -.32, .02])
            goal = self.initial_hands[hand_index] + hand_phase * (final - self.initial_hands[hand_index])
            objective(position, 80. * (goal - data.geom_xpos[geom]) - 18. * position @ data.qvel - position_dot @ data.qvel, 3.)
        inequality = np.zeros((actuator_count, variable_count))
        inequality[:, dofs] = np.eye(actuator_count)
        constraints.append(inequality)
        lower.extend(np.maximum(-200., 2. * (model.jnt_range[task.joints, 0] - data.qpos[task.addresses] - .02 * data.qvel[dofs]) / .02 ** 2))
        upper.extend(np.minimum(200., 2. * (model.jnt_range[task.joints, 1] - data.qpos[task.addresses] - .02 * data.qvel[dofs]) / .02 ** 2))
        margin = .02
        braking_horizon = .15
        joint_position = data.qpos[task.addresses]
        joint_velocity = data.qvel[dofs]
        velocity_lower = np.maximum(-2., (model.jnt_range[task.joints, 0] + margin - joint_position) / braking_horizon)
        velocity_upper = np.minimum(2., (model.jnt_range[task.joints, 1] - margin - joint_position) / braking_horizon)
        constraints.append(inequality.copy())
        lower.extend((velocity_lower - joint_velocity) / .02)
        upper.extend((velocity_upper - joint_velocity) / .02)
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
        local_forces = result.x[force_start:].reshape(-1, 3)
        predicted = np.zeros(4)
        for contact, force in zip(contacts, local_forces):
            predicted[task.supports.index(contact["geom"])] += force[0]
        self.last = {"constraint_residual": violation, "torque_nm": torque.tolist(),
                     "predicted_support_loads_n": predicted.tolist(),
                     "contact_geoms": [model.geom(contact["geom"]).name for contact in contacts],
                     "contact_points": [contact["point"].tolist() for contact in contacts],
                     "predicted_local_forces_n": local_forces.tolist(),
                     "supervisor": self.transfer.last}
        return np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
