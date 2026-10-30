import mujoco
import numpy as np

from check_standing_endpoint import standing_pose
from near_handoff_task import NearHandoffTask, UnsafeState


def progress_reward(previous, current, terminal=False, outcome=0.):
    return float(outcome + 10. * ((0. if terminal else .99 * current) - previous))


class StandTransitionTask(NearHandoffTask):
    horizon = 300

    def __init__(self):
        super().__init__()
        self.crawl_target = self.target.copy()
        self.target = standing_pose(self.model, .6).qpos[self.addresses].copy()
        self.reference_enabled = True

    def measure(self):
        row, gates, penetration = super().measure()
        if self.phase != "crawl":
            gates["height"] = .16 <= self.data.qpos[2] <= .95
        return row, gates, penetration

    def reset(self, state):
        self.integrated = self.target.copy()
        observation = super().reset(state)
        self.start_height = float(self.data.qpos[2])
        self.start_hand_gap = float(np.mean(self.initial_row[8:10]))
        self.potential = 0.
        return observation

    def observation(self):
        observation = super().observation()
        observation[-2] = self.tick / self.horizon
        row, unused, unused_penetration = self.measure()
        observation[55:59] = np.clip(row[8:12], -.01, 1.)
        return np.r_[observation, self.integrated - self.target].astype(np.float32)

    def make_states(self, seeds, stage="standing"):
        if stage != "standing":
            raise ValueError("Standing task cannot silently use near-crawl resets")
        pose = standing_pose(self.model, .6).qpos
        states = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            velocity = np.zeros(self.model.nv)
            velocity[:2] = rng.uniform(-.02, .02, 2)
            state = {"seed": int(seed), "qpos": pose.tolist(), "qvel": velocity.tolist(), "stage": stage}
            self.reset(state)
            states.append(state)
        return states

    def ready(self):
        if len(self.samples) < 51:
            return False
        window = np.array(self.samples[-51:])
        return bool(np.all(window[:, 4:8] > 5.) and np.all(window[:, 3] <= .4)
                    and np.linalg.norm(self.data.qvel[:6]) < .5
                    and np.sqrt(np.mean((self.data.qpos[self.addresses] - self.crawl_target) ** 2)) < .35)

    def progress(self):
        samples = np.array(self.samples)
        if not len(samples):
            return {}
        hands = samples[:, 4:6] > 5.
        consecutive, longest = 0, 0
        for loaded in np.all(hands, axis=1):
            consecutive = consecutive + 1 if loaded else 0
            longest = max(longest, consecutive)
        return {"max_height_reduction_m": float(self.start_height - samples[:, 3].min()),
                "min_mean_hand_gap_m": float(samples[:, 8:10].mean(axis=1).min()),
                "any_hand_loaded": bool(hands.any()), "both_hands_loaded_s": max(0, longest - 1) * .002,
                "transition_time_s": min(float(self.data.time), 6.)}

    def step(self, raw_action):
        raw_action = np.asarray(raw_action)
        if raw_action.shape != (23,) or not np.isfinite(raw_action).all():
            raise ValueError("Expected 23 finite raw actions")
        action = np.tanh(raw_action)
        self.integrated = np.clip(self.integrated + .02 * action,
                                  self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        target = self.integrated.copy()
        if self.reference_enabled:
            phase = np.clip((self.tick - 50) / 50., 0., 1.)
            phase = phase * phase * (3. - 2. * phase)
            for actuator, joint in enumerate(self.joints):
                name = self.model.joint(int(joint)).name
                if name.endswith("knee_joint"):
                    target[actuator] += .2 * phase
                elif name.endswith(("hip_pitch_joint", "ankle_pitch_joint")):
                    target[actuator] -= .1 * phase
        target = np.clip(target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        self.previous = action
        try:
            self.execution.transition_step(target)
        except UnsafeState:
            return self.observation(), progress_reward(self.potential, 0., True, -5.), True, {"success": False, "failure": self.failure,
                                                  "phase": "transition", **self.progress()}
        self.tick += 1
        row, unused, unused_penetration = self.measure()
        potential = 4. * (self.start_height - row[3]) + 2. * (self.start_hand_gap - np.mean(row[8:10]))
        previous_potential = self.potential
        cost = .001 * float(np.mean(action ** 2))
        reward = progress_reward(previous_potential, potential) - cost
        self.potential = float(potential)
        if self.ready():
            progress = self.progress()
            result = self.finish()
            result.update(progress, phase="handoff")
            terminal_reward = progress_reward(previous_potential, potential, True, 50. if result["success"] else -5.) - cost
            return self.observation(), terminal_reward, True, result
        if self.tick >= self.horizon:
            return self.observation(), progress_reward(previous_potential, potential, True, -2.) - cost, True, {"success": False, "failure": ["timeout_without_readiness"],
                                                                "phase": "transition", **self.progress()}
        return self.observation(), float(reward), False, {}

    def run_baseline(self):
        self.reference_enabled = False
        try:
            for unused in range(self.horizon):
                observation, reward, done, result = self.step(np.zeros(23))
                if done:
                    break
            return {**result, "baseline": "standing_hold_no_lowering"}
        finally:
            self.reference_enabled = True
