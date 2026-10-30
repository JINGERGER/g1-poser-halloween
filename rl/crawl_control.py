import mujoco
import numpy as np
import torch


class CrawlController:
    def __init__(self, model, constants, policy, command=.5, ankle_min_target=-.2, foot_clearance_boost=.08):
        self.model, self.constants, self.policy = model, constants, policy
        self.command, self.ankle_min_target, self.foot_clearance_boost = command, ankle_min_target, foot_clearance_boost
        self.previous = np.zeros(23, dtype=np.float32)
        self.order = constants["PYTORCH_JOINT_ORDER"]
        self.defaults = constants["TRAINING_DEFAULT_ANGLES"]
        self.joint_ids = {model.joint(index).name: index for index in range(1, model.njnt)}
        self.floor = next(index for index in range(model.ngeom) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE)
        self.support_groups = [[], [], [], []]
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            name = model.geom(geom).name
            if name in ("left_hand_collision", "right_hand_collision"):
                self.support_groups[0 if name.startswith("left") else 1].append(geom)
            body = model.body(int(model.geom_bodyid[geom])).name
            if body in ("left_ankle_roll_link", "right_ankle_roll_link"):
                self.support_groups[2 if body.startswith("left") else 3].append(geom)
        assert all(self.support_groups)

    def update(self, data):
        model, constants = self.model, self.constants
        joint_ids, order, defaults = self.joint_ids, self.order, self.defaults
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ [0., 0., -1.]
        relative, speeds = np.zeros(23, dtype=np.float32), np.zeros(23, dtype=np.float32)
        for index, name in enumerate(order):
            if name in joint_ids:
                joint = joint_ids[name]
                relative[index] = np.float32(data.qpos[model.jnt_qposadr[joint]]) - np.float32(defaults[name])
                speeds[index] = data.qvel[model.jnt_dofadr[joint]]
        observation = np.r_[gravity, [self.command, 0., 0.], relative, speeds, self.previous].astype(np.float32)
        assert observation.shape == (75,) and np.isfinite(observation).all()
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(observation)).numpy().copy()
        assert action.shape == (23,) and np.isfinite(action).all()
        self.previous = action
        correction = {}
        if self.foot_clearance_boost:
            for side, opposite_hand in (("left", self.support_groups[1][0]), ("right", self.support_groups[0][0])):
                gap = float(mujoco.mj_geomDistance(model, data, opposite_hand, self.floor, 2., None))
                phase = np.clip((gap - .002) / .015, 0., 1.)
                phase = phase * phase * (3 - 2 * phase)
                leg_names = [f"{side}_{suffix}_joint" for suffix in ("hip_pitch", "hip_roll", "hip_yaw", "knee")]
                columns = [model.jnt_dofadr[joint_ids[name]] for name in leg_names]
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jacGeom(model, data, jacobian, None, self.support_groups[2 if side == "left" else 3][0])
                leg_jacobian = jacobian[:, columns]
                delta = leg_jacobian.T @ np.linalg.solve(
                    leg_jacobian @ leg_jacobian.T + .001 ** 2 * np.eye(3),
                    np.array([0., 0., self.foot_clearance_boost * phase]))
                correction.update(zip(leg_names, np.clip(delta, -.2, .2)))
        for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
            name = model.joint(int(joint)).name
            target = np.float32(defaults[name]) + np.float32(.5) * action[order.index(name)]
            target += correction.get(name, 0.)
            limits = constants["JOINT_LIMITS"].get(name, model.jnt_range[joint])
            data.ctrl[actuator] = np.clip(target, *limits)
            if self.ankle_min_target is not None and name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
                data.ctrl[actuator] = max(data.ctrl[actuator], self.ankle_min_target)
            if model.actuator_ctrllimited[actuator]:
                data.ctrl[actuator] = np.clip(data.ctrl[actuator], *model.actuator_ctrlrange[actuator])
        return action
