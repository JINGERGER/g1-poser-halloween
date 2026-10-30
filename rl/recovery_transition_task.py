import mujoco
import numpy as np

from position_torque_guard import PositionTorqueGuard, InfeasiblePositionTarget
from stand_transition_task import StandTransitionTask, UnsafeState


def paired_action(raw_action, names):
    raw_action = np.asarray(raw_action)
    if raw_action.shape != (23,) or not np.isfinite(raw_action).all():
        raise ValueError("Expected 23 finite latent actions")
    transformed = raw_action.copy()
    for index, name in enumerate(names):
        if not name.startswith("left_"):
            continue
        right = names.index("right_" + name[5:])
        sign = -1. if any(axis in name for axis in ("roll", "yaw")) else 1.
        common, difference = raw_action[index], .2 * raw_action[right]
        transformed[index] = (common + difference) / np.sqrt(2.)
        transformed[right] = sign * (common - difference) / np.sqrt(2.)
    return np.tanh(transformed)


class RecoveryTransitionTask(StandTransitionTask):
    horizon = 150

    def __init__(self):
        super().__init__()
        self.training_mode = False
        self.names = [self.model.joint(int(joint)).name for joint in self.joints]
        self.guard = PositionTorqueGuard(self.model, fraction=.8)

    def reset(self, state):
        self.filtered = np.zeros(23)
        self.violations = set()
        self.violation_steps = 0
        self.first_violation_time = None
        self.load_time = 0.
        super().reset(state)
        return self.observation()

    def observation(self):
        observation = super().observation()
        stage = 2 if self.load_time > 0. else (1 if observation[0] > .2 else 0)
        return np.r_[observation, self.filtered, np.eye(3)[stage], min(self.load_time, .1) / .1,
                     float(bool(self.violations))].astype(np.float32)

    def _observe_step(self, model, data):
        try:
            super()._observe_step(model, data)
        except UnsafeState:
            self.violations.update(self.failure)
            self.violation_steps += 1
            if self.first_violation_time is None:
                self.first_violation_time = float(data.time)
            catastrophic = (max(self.penetrations[-1]) > .01 or data.qpos[2] < .08 or data.qpos[2] > 1.1)
            if not self.training_mode or self.phase == "crawl" or catastrophic:
                raise
        self.load_time = self.load_time + .002 if np.all(np.array(self.samples[-1][4:6]) > 5.) else 0.

    def step(self, raw_action):
        action = paired_action(raw_action, self.names)
        self.filtered = .65 * self.filtered + .35 * action
        self.previous = action
        reward = 0.
        result = None
        violations_before = self.violation_steps
        try:
            for unused in range(2):
                observation = super().observation()
                alignment = float(np.clip(observation[0], 0., 1.))
                center = self.target + alignment * (self.crawl_target - self.target)
                ranges = self.model.actuator_ctrlrange
                offsets = np.where(self.filtered >= 0., ranges[:, 1] - center, center - ranges[:, 0])
                requested = np.clip(center + self.filtered * offsets, ranges[:, 0], ranges[:, 1])
                for unused_substep in range(10):
                    if self.data.qfrc_applied.any() or self.data.xfrc_applied.any():
                        raise ValueError("Assistance forces prohibited")
                    try:
                        self.data.ctrl[:] = self.guard.apply(self.data, requested)
                    except InfeasiblePositionTarget:
                        if not self.training_mode:
                            self.failure = ["infeasible_guard_target"]
                            raise UnsafeState("infeasible guard target")
                        self.violations.add("infeasible_guard_target")
                        self.data.ctrl[:] = requested
                        reward -= .02
                    mujoco.mj_step(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                        raise ValueError("Nonfinite physical state")
                    self._observe_step(self.model, self.data)
                self.integrated = self.data.ctrl.copy()
                row, unused_gates, penetration = self.measure()
                alignment = float(np.clip(super().observation()[0], 0., 1.))
                speed = float(np.linalg.norm(self.data.qvel[:6]))
                posture = np.exp(-((row[3] - .25) / .2) ** 2)
                reach = np.exp(-max(0., np.mean(row[8:10])) / .12)
                support = float(self.load_time >= .02) * np.exp(-speed / 2.)
                reward += .02 * (alignment + .5 * posture + .5 * reach + support)
                reward -= .02 * (2. * min(row[12] / 312., 2.) + 20. * max(penetration) + max(0., row[13] - .9))
                if self.ready():
                    prefix = set(self.violations)
                    self.failure = []
                    tail = self.finish()
                    recovery = bool(tail["success"] and prefix)
                    result = {**tail, "success": bool(tail["success"] and not prefix), "recovery_only": recovery,
                              "failure": sorted(prefix | set(self.failure))}
                    reward += 50. if result["success"] else (5. if recovery else -5.)
                    break
        except UnsafeState:
            reward -= 5.
            result = {"success": False, "recovery_only": False, "failure": sorted(self.violations | set(self.failure))}
        self.tick += 1
        reward -= .001 * float(np.mean(self.filtered ** 2)) + .01 * (self.violation_steps - violations_before)
        if result is None and self.tick >= self.horizon:
            result = {"success": False, "recovery_only": False, "failure": sorted(self.violations | {"timeout_without_readiness"})}
            reward -= 5.
        if result is not None:
            result.update(self.progress(), first_violation_time_s=self.first_violation_time,
                          strict_prefix_valid=not self.violations, training_mode=self.training_mode)
        return self.observation(), float(reward), result is not None, result or {}
