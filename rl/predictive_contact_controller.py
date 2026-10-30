import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog

from contact_step_controller import ContactStepController, rotation_error
from measured_support import support_contacts
from position_torque_guard import PositionTorqueGuard


class PredictiveContactController(ContactStepController):
    def __init__(self, task):
        super().__init__(task)
        self.guard = PositionTorqueGuard(task.model)
        self.previous_torque = task.data.qfrc_actuator[task.model.jnt_dofadr[task.joints]].copy()
        self.probe = mujoco.MjData(task.model)
        self.collision_pairs = []
        self.predicted_depths = np.empty(0)
        self.nonlinear_search = False

    def prediction_target(self, torque, probe):
        model = self.task.model
        bias = model.actuator_biasprm[:, 0] + model.actuator_biasprm[:, 1] * probe.actuator_length + model.actuator_biasprm[:, 2] * probe.actuator_velocity
        return (torque - bias) / model.actuator_gainprm[:, 0]

    def predict(self, torque):
        model, data = self.task.model, self.task.data
        mujoco.mj_copyData(self.probe, model, data)
        depths = np.full(len(self.collision_pairs), -np.inf)
        for unused in range(10):
            self.probe.ctrl[:] = self.guard.apply(self.probe, self.prediction_target(torque, self.probe))
            mujoco.mj_step(model, self.probe)
            mujoco.mj_forward(model, self.probe)
            for index, pair in enumerate(self.collision_pairs):
                depths[index] = max(depths[index], -mujoco.mj_geomDistance(model, self.probe, *pair, .1, None))
        loads = np.zeros(4)
        for index, contact in enumerate(self.probe.contact):
            if self.task.floor not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == self.task.floor else contact.geom1
            if geom in self.task.supports:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, self.probe, index, wrench)
                loads[self.task.supports.index(geom)] += max(0., wrench[0])
        self.predicted_depths = depths
        return self.probe.qvel.copy(), self.probe.qpos[self.task.addresses].copy(), loads

    def target(self, com_goal, pitch, hand_goals, posture, foot_goals, foot_rotations, swing_geom=None, load_targets=None, rotation_goal=None):
        task, model, data = self.task, self.task.model, self.task.data
        limits = .8 * self.guard.limits
        baseline = np.clip(self.previous_torque, -limits, limits)
        self.collision_pairs = sorted({tuple(sorted((int(contact.geom1), int(contact.geom2)))) for contact in data.contact
                                       if task.floor not in (contact.geom1, contact.geom2)})
        velocity, position, loads = self.predict(baseline)
        self.collision_pairs = sorted(set(self.collision_pairs) | {
            tuple(sorted((int(contact.geom1), int(contact.geom2)))) for contact in self.probe.contact
            if task.floor not in (contact.geom1, contact.geom2)})
        velocity, position, loads = self.predict(baseline)
        depths = self.predicted_depths.copy()
        velocity_jac = np.empty((model.nv, model.nu))
        position_jac = np.empty((model.nu, model.nu))
        load_jac = np.empty((4, model.nu))
        depth_jac = np.empty((len(depths), model.nu))
        for index in range(model.nu):
            delta = .1 if baseline[index] + .1 < limits[index] else -.1
            perturbed = baseline.copy()
            perturbed[index] += delta
            new_velocity, new_position, new_loads = self.predict(perturbed)
            velocity_jac[:, index] = (new_velocity - velocity) / delta
            position_jac[:, index] = (new_position - position) / delta
            load_jac[:, index] = (new_loads - loads) / delta
            depth_jac[:, index] = (self.predicted_depths - depths) / delta
        quadratic = .001 * np.eye(model.nu)
        linear = np.zeros(model.nu)
        objectives = []

        def objective(jac, desired, weight):
            objectives.append((jac.copy(), desired.copy(), weight))
            sensitivity = jac @ velocity_jac / .02
            residual = jac @ (velocity - data.qvel) / .02 - desired
            nonlocal quadratic, linear
            quadratic += weight * sensitivity.T @ sensitivity
            linear += weight * sensitivity.T @ residual

        com_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, com_jac, self.pelvis)
        objective(com_jac, 80. * (com_goal - data.subtree_com[self.pelvis]) - 18. * com_jac @ data.qvel, 100.)
        desired_rotation = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0., 1., 0.], [-np.sin(pitch), 0., np.cos(pitch)]]) if rotation_goal is None else rotation_goal
        rotation_jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, None, rotation_jac, self.pelvis)
        objective(rotation_jac, 80. * rotation_error(desired_rotation, data.xmat[self.pelvis].reshape(3, 3)) - 18. * rotation_jac @ data.qvel, 5.)
        dofs = model.jnt_dofadr[task.joints]
        objective(np.eye(model.nv)[dofs], 20. * (posture - data.qpos[task.addresses]) - 8. * data.qvel[dofs], .03)
        for index, geom in enumerate(task.supports):
            position_jacobian = np.zeros((3, model.nv))
            rotation_jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, position_jacobian, rotation_jacobian, geom)
            goal = hand_goals[index] if index < 2 else foot_goals[index - 2]
            objective(position_jacobian, 80. * (goal - data.geom_xpos[geom]) - 18. * position_jacobian @ data.qvel, 3. if index < 2 else 30.)
            if index >= 2:
                angular = rotation_error(foot_rotations[index - 2], data.geom_xmat[geom].reshape(3, 3))
                objective(rotation_jacobian, 80. * angular - 18. * rotation_jacobian @ data.qvel, 10.)
        if load_targets is not None:
            quadratic += .001 * load_jac.T @ load_jac
            linear += .001 * load_jac.T @ (loads - load_targets)
        constraints = [np.eye(model.nu), position_jac, velocity_jac[dofs]]
        lower = [np.maximum(-limits - baseline, -20.), model.jnt_range[task.joints, 0] + .005 - position,
                 np.maximum(-2., (model.jnt_range[task.joints, 0] + .02 - data.qpos[task.addresses]) / .15) - velocity[dofs]]
        upper = [np.minimum(limits - baseline, 20.), model.jnt_range[task.joints, 1] - .005 - position,
                 np.minimum(2., (model.jnt_range[task.joints, 1] - .02 - data.qpos[task.addresses]) / .15) - velocity[dofs]]
        if len(depths):
            constraints.append(depth_jac)
            lower.append(np.full(len(depths), -np.inf))
            upper.append(.0006 - depths)
        for contact in data.contact:
            if task.floor in (contact.geom1, contact.geom2):
                continue
            relative = np.zeros((3, model.nv))
            for geom, sign in ((contact.geom1, -1.), (contact.geom2, 1.)):
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacobian, None, contact.pos, model.geom_bodyid[geom])
                relative += sign * jacobian
            normal = contact.frame[:3] @ relative
            minimum_speed = -(float(contact.dist) + .0004) / .15
            constraints.append((normal @ velocity_jac).reshape(1, -1))
            lower.append(np.array([minimum_speed - normal @ velocity]))
            upper.append(np.array([np.inf]))
        solver = osqp.OSQP()
        matrix = np.vstack(constraints)
        low, high = np.concatenate(lower), np.concatenate(upper)
        units = np.r_[np.full(model.nu, 10.), np.full(model.nu, .01), np.ones(model.nu),
                      np.full(len(depths), .001), np.full(len(low) - 3 * model.nu - len(depths), .01)]
        cost_scale = max(1., float(np.diag(quadratic).max()))
        solver.setup(P=sparse.csc_matrix(np.triu(quadratic / cost_scale)), q=linear / cost_scale,
                     A=sparse.csc_matrix(matrix / units[:, None]), l=low / units, u=high / units,
                     verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polishing=True)
        result = solver.solve(raise_error=False)
        solver_attempts = [{"status": result.info.status, "iterations": int(result.info.iter),
                            "primal_residual": float(result.info.prim_res), "dual_residual": float(result.info.dual_res)}]
        if result.info.status in ("solved inaccurate", "maximum iterations reached"):
            variable_scale = np.sqrt(cost_scale / np.maximum(np.diag(quadratic), 1e-12))
            conditioned = osqp.OSQP()
            conditioned.setup(P=sparse.csc_matrix(np.triu(quadratic * np.outer(variable_scale, variable_scale) / cost_scale)),
                              q=linear * variable_scale / cost_scale,
                              A=sparse.csc_matrix(matrix * variable_scale[None, :] / units[:, None]),
                              l=low / units, u=high / units, verbose=False,
                              eps_abs=1e-6, eps_rel=1e-6, max_iter=200000, polishing=True)
            if result.x is not None:
                conditioned.warm_start(x=result.x / variable_scale)
            result = conditioned.solve(raise_error=False)
            if result.x is not None:
                result.x[:] *= variable_scale
            solver_attempts.append({"status": result.info.status, "iterations": int(result.info.iter),
                                    "primal_residual": float(result.info.prim_res), "dual_residual": float(result.info.dual_res),
                                    "diagonal_variable_scaling": True})
        if result.info.status != "solved" or result.x is None:
            matrix = np.vstack(constraints)
            low, high = np.concatenate(lower), np.concatenate(upper)
            scales = np.r_[np.full(model.nu, 1.), np.full(model.nu, .001), np.full(model.nu, .01),
                           np.full(len(depths), .0001), np.full(len(low) - 3 * model.nu - len(depths), .001)]
            scales[:model.nu] = 0.
            finite_high, finite_low = np.isfinite(high), np.isfinite(low)
            diagnostic = linprog(np.r_[np.zeros(model.nu), 1.],
                                 A_ub=np.vstack([np.c_[matrix[finite_high], -scales[finite_high]],
                                                 np.c_[-matrix[finite_low], -scales[finite_low]]]),
                                 b_ub=np.r_[high[finite_high], -low[finite_low]],
                                 bounds=[(None, None)] * model.nu + [(0., None)], method="highs")
            self.failure_diagnostics = {"solver_status": result.info.status, "relaxed_diagnostic_feasible": bool(diagnostic.success),
                                        "solver_attempts": solver_attempts,
                                        "collision_pairs": self.collision_pairs,
                                        "row_groups": {"torque": [0, model.nu], "position": [model.nu, 2 * model.nu],
                                                       "velocity": [2 * model.nu, 3 * model.nu],
                                                       "depth": [3 * model.nu, 3 * model.nu + len(depths)],
                                                       "closing_speed": [3 * model.nu + len(depths), len(low)]}}
            if diagnostic.success:
                candidate = matrix @ diagnostic.x[:-1]
                violations = np.maximum(0., np.maximum(low - candidate, candidate - high))
                self.failure_diagnostics.update(minimum_normalized_slack=float(diagnostic.x[-1]),
                                                violations=[{"row": int(index), "violation": float(violations[index])}
                                                            for index in np.argsort(violations / np.maximum(scales, 1e-12))[-8:][::-1]
                                                            if violations[index] > 1e-9])
            raise RuntimeError(f"Predictive QP: {result.info.status}")
        violation = np.maximum(0., np.maximum(low - matrix @ result.x, matrix @ result.x - high)) / units
        if float(violation.max()) > 1e-4:
            raise RuntimeError("Predictive QP normalized constraint residual exceeds 1e-4")
        torque = np.clip(baseline + result.x, -limits, limits)
        search_records = []
        if self.nonlinear_search:
            best = None
            for fraction in (1., .5, .25, .125, .0625, 0.):
                change = fraction * result.x
                residual = np.maximum(0., np.maximum(low - matrix @ change, matrix @ change - high)) / units
                if float(residual.max()) > 1e-4:
                    continue
                candidate = np.clip(baseline + change, -limits, limits)
                candidate_velocity, candidate_position, candidate_loads = self.predict(candidate)
                if len(self.predicted_depths) and np.max(self.predicted_depths) > .0009:
                    continue
                if np.any(candidate_position < model.jnt_range[task.joints, 0] + .005 - 1e-6) or np.any(candidate_position > model.jnt_range[task.joints, 1] - .005 + 1e-6):
                    continue
                acceleration = (candidate_velocity - data.qvel) / .02
                cost = .001 * float(change @ change)
                for jacobian, desired, weight in objectives:
                    error = jacobian @ acceleration - desired
                    cost += weight * float(error @ error)
                if load_targets is not None:
                    error = candidate_loads - load_targets
                    cost += .001 * float(error @ error)
                search_records.append({"fraction": fraction, "cost": cost})
                if best is None or cost < best[0]:
                    best = (cost, candidate, residual, fraction)
            if best is None:
                raise RuntimeError("No feasible nonlinear line-search candidate")
            unused_cost, torque, violation, selected_fraction = best
        predicted_velocity, predicted_position, predicted_loads = self.predict(torque)
        if len(self.predicted_depths) and np.max(self.predicted_depths) > .0009:
            raise RuntimeError("Nonlinear predictive rollout violates self-contact margin")
        self.previous_torque = torque.copy()
        self.last = {"torque_nm": torque.tolist(), "predicted_support_loads_n": predicted_loads.tolist(),
                     "solver_attempts": solver_attempts,
                     "normalized_constraint_residual": float(violation.max()),
                     "predicted_qvel": predicted_velocity.tolist(), "predicted_joint_position": predicted_position.tolist(),
                     "contact_geoms": [model.geom(contact["geom"]).name for contact in support_contacts(task)]}
        if self.nonlinear_search:
            self.last.update(nonlinear_search=search_records, selected_fraction=selected_fraction)
        bias = model.actuator_biasprm[:, 0] + model.actuator_biasprm[:, 1] * data.actuator_length + model.actuator_biasprm[:, 2] * data.actuator_velocity
        return np.clip((torque - bias) / model.actuator_gainprm[:, 0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
