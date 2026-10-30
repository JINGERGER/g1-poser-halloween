from types import SimpleNamespace

import mujoco
import numpy as np

from check_unitree_standing_baseline import make_observation


class LegAdapter:
    def __init__(self, model, reference):
        self.names = [reference.joint(int(joint)).name for joint in reference.actuator_trnid[:, 0]]
        if len(self.names) != 12 or len(set(self.names)) != 12:
            raise ValueError("Expected twelve unique official leg joints")
        self.joints = np.array([model.joint(name).id for name in self.names])
        actuators = []
        for joint in self.joints:
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError("Each policy joint must have exactly one actuator")
            actuators.append(int(matches[0]))
        self.actuators = np.array(actuators)
        self.qpos = model.jnt_qposadr[self.joints].copy()
        self.qvel = model.jnt_dofadr[self.joints].copy()
        if not np.all(model.jnt_type[self.joints] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only hinge joint mappings are supported")
        if not np.all(model.jnt_actfrclimited[self.joints]):
            raise ValueError("All mapped joints must retain physical actuator force limits")
        if not np.all(model.actuator_dyntype[self.actuators] == mujoco.mjtDyn.mjDYN_NONE):
            raise ValueError("Stateful actuators are unsupported")
        if not np.array_equal(model.actuator_gear[self.actuators], np.tile([1., 0., 0., 0., 0., 0.], (12, 1))):
            raise ValueError("Torque mapping requires unit transmission gear")

    def torque_interface(self, model):
        model.actuator_gainprm[self.actuators] = 0.
        model.actuator_gainprm[self.actuators, 0] = 1.
        model.actuator_biasprm[self.actuators] = 0.
        model.actuator_gaintype[self.actuators] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[self.actuators] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_ctrllimited[self.actuators] = False

    def observation(self, data, action, counter, command, config, gravity):
        view = SimpleNamespace(qpos=np.r_[data.qpos[:7], data.qpos[self.qpos]],
                               qvel=np.r_[data.qvel[:6], data.qvel[self.qvel]])
        return make_observation(view, action, counter, command, config, gravity)

    def manifest(self):
        return [{"joint": name, "actuator_id": int(actuator), "qpos_address": int(position),
                 "qvel_address": int(velocity)}
                for name, actuator, position, velocity in zip(self.names, self.actuators, self.qpos, self.qvel)]
