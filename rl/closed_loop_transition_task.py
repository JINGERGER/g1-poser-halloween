import numpy as np

from crawl_control import CrawlController
from position_torque_guard import PositionTorqueGuard, InfeasiblePositionTarget
from stand_transition_task import StandTransitionTask, UnsafeState, progress_reward


class ClosedLoopTransitionTask(StandTransitionTask):
    horizon = 100

    def __init__(self, prior=True):
        super().__init__()
        self.use_prior = prior
        mapping = {self.reference.joint(int(joint)).name: index for index, joint in enumerate(self.reference.actuator_trnid[:, 0])}
        for actuator, joint in enumerate(self.joints):
            name = self.model.joint(int(joint)).name
            if name in mapping:
                source = mapping[name]
                self.native_gain[actuator, 0] = self.reference.actuator_gainprm[source, 0] * 1.4
                self.native_bias[actuator, 1:3] = self.reference.actuator_biasprm[source, 1:3] * 1.4
        self.guard = PositionTorqueGuard(self.model, fraction=.5)

    def reset(self, state):
        self.residual = np.zeros(23)
        self.raw_previous = np.zeros(23)
        self.event = 0
        self.load_duration = 0.
        super().reset(state)
        self.controller = CrawlController(self.model, self.constants, self.policy, command=0.,
                                          ankle_min_target=None, foot_clearance_boost=0.)
        return self.observation()

    def observation(self):
        observation = super().observation()
        event = np.eye(3)[self.event]
        return np.r_[observation, self.residual, self.raw_previous, event, self.load_duration / .1].astype(np.float32)

    def step(self, raw_action):
        raw_action = np.asarray(raw_action)
        if raw_action.shape != (23,) or not np.isfinite(raw_action).all():
            raise ValueError("Expected 23 finite raw actions")
        self.residual = .5 * self.residual + .5 * np.tanh(raw_action)
        self.previous = np.tanh(raw_action)
        terminal = False
        result = None
        contact_reward = 0.
        try:
            for unused in range(3):
                if self.use_prior:
                    self.controller.update(self.data)
                    self.raw_previous = self.controller.previous.copy()
                    baseline = self.data.ctrl.copy()
                else:
                    baseline = self.target.copy()
                requested = np.clip(baseline + 1.5 * self.residual, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
                self.guard.advance(self.execution, requested)
                self.integrated = self.data.ctrl.copy()
                row, unused_gates, unused_pen = self.measure()
                if row[4] > 5. or row[5] > 5.:
                    self.event = max(self.event, 1)
                window = np.array(self.samples[-51:])
                self.load_duration = min(.1, self.load_duration + .02) if np.all(window[-10:, 4:6] > 5.) else 0.
                if self.load_duration >= .1:
                    self.event = 2
                speed = float(np.linalg.norm(self.data.qvel[:6]))
                contact_reward += .02 * np.exp(-max(0., np.mean(row[8:10])) / .2) * np.exp(-speed / 2.)
                contact_reward += .05 * float(np.all(window[-10:, 4:6] > 5.)) * np.exp(-speed / 2.)
                if self.ready():
                    result = self.finish()
                    terminal = True
                    break
        except (UnsafeState, InfeasiblePositionTarget) as error:
            if isinstance(error, InfeasiblePositionTarget):
                self.failure = ["infeasible_guard_target"]
            terminal = True
            result = {"success": False, "failure": self.failure.copy()}
        self.tick += 1
        if not terminal and self.tick >= self.horizon:
            terminal = True
            result = {"success": False, "failure": ["timeout_without_readiness"]}
        row, unused_gates, unused_pen = self.measure()
        potential = 4. * (self.start_height - row[3]) + 2. * (self.start_hand_gap - np.mean(row[8:10]))
        outcome = (50. if result["success"] else -5.) if terminal else 0.
        reward = progress_reward(self.potential, potential, terminal, outcome) + contact_reward - .001 * float(np.mean(self.residual ** 2))
        self.potential = float(potential)
        if terminal:
            result.update(self.progress(), event=self.event)
        return self.observation(), float(reward), terminal, result or {}
