import mujoco
import numpy as np

from position_torque_guard import PositionTorqueGuard


class ArmClearanceGuard(PositionTorqueGuard):
    def __init__(self, model):
        super().__init__(model, fraction=.8)
        self.joints = model.actuator_trnid[:, 0]
        self.addresses = model.jnt_qposadr[self.joints]
        self.columns = model.jnt_dofadr[self.joints]
        self.pairs = []
        arms = [model.geom(f"{side}_{part}").id for side in ("left", "right")
                for part in ("forearm", "hand_collision", "elbow_collision")]
        legs = [model.geom(f"{side}_{part}").id for side in ("left", "right") for part in ("thigh", "shin")]
        self.pairs = [(arm, leg) for arm in arms for leg in legs]
        self.inverse_weights = np.array([1. if any(part in model.joint(int(joint)).name for part in ("shoulder", "elbow", "wrist"))
                                         else .25 for joint in self.joints])
        self.projections = 0

    def constraints(self, data):
        result = []
        for first, second in self.pairs:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(self.model, data, first, second, .25, segment)
            if distance >= .25 or abs(distance) < 1e-8:
                continue
            normal = (segment[3:] - segment[:3]) / distance
            first_jacobian, second_jacobian = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, first_jacobian, None, segment[:3], int(self.model.geom_bodyid[first]))
            mujoco.mj_jac(self.model, data, second_jacobian, None, segment[3:], int(self.model.geom_bodyid[second]))
            gradient = normal @ (second_jacobian - first_jacobian)[:, self.columns]
            result.append((gradient, .02 - float(distance)))
        return result

    def apply(self, data, requested):
        angles = data.qpos[self.addresses]
        delta = np.asarray(requested) - angles
        constraints = self.constraints(data)
        for unused in range(6):
            for gradient, bound in constraints:
                missing = bound - float(gradient @ delta)
                norm = float(np.sum(gradient ** 2 * self.inverse_weights))
                if missing > 0. and norm > 1e-12:
                    delta += self.inverse_weights * gradient * (missing / norm)
            delta = np.clip(delta, self.model.actuator_ctrlrange[:, 0] - angles, self.model.actuator_ctrlrange[:, 1] - angles)
        projected = angles + delta
        self.projections += int(np.max(np.abs(projected - requested)) > 1e-8)
        return super().apply(data, projected)
