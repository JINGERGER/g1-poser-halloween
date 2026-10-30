import numpy as np
import mujoco


class InfeasiblePositionTarget(Exception):
    pass


def guarded_target(requested, bias, gains, limits, lower, upper, fraction):
    requested = np.asarray(requested, dtype=float)
    values = (requested, bias, gains, limits, lower, upper)
    if not 0. < fraction < .95 or any(np.shape(value) != requested.shape for value in values):
        raise ValueError("Invalid guard fraction or shapes")
    if not all(np.isfinite(value).all() for value in values) or np.any(gains <= 0.) or np.any(limits <= 0.):
        raise ValueError("Expected finite positive PD gains and force limits")
    safe_lower = np.maximum(lower, (-fraction * limits - bias) / gains)
    safe_upper = np.minimum(upper, (fraction * limits - bias) / gains)
    if np.any(safe_lower > safe_upper):
        raise InfeasiblePositionTarget("No position target satisfies both control and current torque bounds")
    return np.clip(requested, safe_lower, safe_upper)


class PositionTorqueGuard:
    def __init__(self, model, fraction=.8):
        self.model, self.fraction = model, fraction
        if not 0. < fraction < .95:
            raise ValueError("Guard fraction must be between zero and .95")
        if np.any(model.actuator_gaintype) or np.any(model.actuator_biastype != 1):
            raise ValueError("Guard requires fixed-gain affine-bias position actuators")
        joints = model.actuator_trnid[:, 0]
        if not np.allclose(model.actuator_gear[:, 0], 1.) or np.any(model.actuator_gear[:, 1:]):
            raise ValueError("Guard requires unit-gear joint transmissions")
        if not np.all(model.jnt_actfrclimited[joints]) or not np.all(model.actuator_ctrllimited):
            raise ValueError("Guard requires physical joint-force and control limits")
        if len(set(joints.tolist())) != model.nu or np.any(model.actuator_dyntype):
            raise ValueError("Guard requires one memoryless actuator per joint")
        self.limits = np.minimum(-model.jnt_actfrcrange[joints, 0], model.jnt_actfrcrange[joints, 1])

    def apply(self, data, requested):
        model = self.model
        bias = model.actuator_biasprm[:, 0] + model.actuator_biasprm[:, 1] * data.actuator_length + model.actuator_biasprm[:, 2] * data.actuator_velocity
        return guarded_target(requested, bias, model.actuator_gainprm[:, 0], self.limits,
                              model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1], self.fraction)

    def advance(self, execution, requested):
        if execution.phase != "transition" or execution.model is not self.model:
            raise ValueError("Guarded advance requires the matching transition execution")
        data = execution.data
        for unused in range(10):
            if data.qfrc_applied.any() or data.xfrc_applied.any():
                raise ValueError("Assistance forces prohibited")
            data.ctrl[:] = self.apply(data, requested)
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("Nonfinite state")
            if execution.step_observer is not None:
                execution.step_observer(self.model, data)
