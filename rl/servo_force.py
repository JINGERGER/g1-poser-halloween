import mujoco
import numpy as np


def validate_position_servos(model):
    joint_ids = model.actuator_trnid[:, 0]
    valid = (
        model.nu == 23
        and np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
        and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        and np.all(model.actuator_gear[:, 0] == 1)
        and np.all(model.actuator_gear[:, 1:] == 0)
        and np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 30))
        and np.array_equal(model.jnt_dofadr[joint_ids], np.arange(6, 29))
        and np.all(model.jnt_actfrclimited[joint_ids])
    )
    if not valid:
        raise ValueError("Force reconstruction requires 23 fixed affine, unit-gear, one-actuator-per-joint servos")


def position_servo_force(model, targets, positions, velocities):
    targets = np.where(model.actuator_ctrllimited, np.clip(targets, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]), targets)
    force = model.actuator_gainprm[:, 0] * targets + model.actuator_biasprm[:, 0]
    force += model.actuator_biasprm[:, 1] * positions + model.actuator_biasprm[:, 2] * velocities
    force = np.where(model.actuator_forcelimited, np.clip(force, model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1]), force)
    limits = model.jnt_actfrcrange[model.actuator_trnid[:, 0]]
    return np.clip(force, limits[:, 0], limits[:, 1])
