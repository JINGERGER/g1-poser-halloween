import mujoco
import numpy as np


def internal_constraint_force(model, data, floor):
    generalized = np.zeros(model.nv)
    friction = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF, data.efc_force, 0.)
    mujoco.mj_mulJacTVec(model, data, generalized, friction)
    for index, contact in enumerate(data.contact):
        if floor in (contact.geom1, contact.geom2) or contact.efc_address < 0:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        frame = contact.frame.reshape(3, 3)
        force, torque = frame.T @ local[:3], frame.T @ local[3:]
        mujoco.mj_applyFT(model, data, -force, -torque, contact.pos,
                          model.geom_bodyid[contact.geom1], generalized)
        mujoco.mj_applyFT(model, data, force, torque, contact.pos,
                          model.geom_bodyid[contact.geom2], generalized)
    return generalized
