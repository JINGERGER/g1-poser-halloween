import numpy as np

from standing_capture_routes import smooth


PARAMETERS = ("hip_prepare", "ankle_prepare", "pitch_trigger", "fold_duration", "arm_lead", "roll_delay",
              "shoulder_roll_prepare", "shoulder_pitch_prepare", "hip_pitch_offset", "knee_offset",
              "ankle_offset", "hip_roll_offset", "hip_yaw_offset", "shoulder_pitch_offset",
              "shoulder_roll_offset", "shoulder_yaw_offset", "elbow_offset")
LOWER = np.array([-.45, -.3, .15, .3, -.3, 0., .5, -.5, -.4, -.5, -.3, -.5, -.5, -.5, -.4, -.5, -.4])
UPPER = np.array([0., .02, .65, 1.3, .5, .6, 1.4, .5, .4, .5, .3, .3, .5, .5, .4, .5, .4])
DEFAULT = np.array([-.15, -.1, .3, .7, .15, .25, 1., -.3, 0., 0., 0., 0., 0., 0., 0., 0., 0.])


class StandingTransitionV1:
    def __init__(self, task, parameters):
        self.parameters = np.asarray(parameters, dtype=float)
        if self.parameters.shape != DEFAULT.shape or not np.isfinite(self.parameters).all():
            raise ValueError("Invalid transition parameters")
        if np.any(self.parameters < LOWER) or np.any(self.parameters > UPPER):
            raise ValueError("Transition parameters outside declared bounds")
        self.names = [task.model.joint(int(joint)).name for joint in task.joints]
        self.initial = task.target.copy()
        self.prepared = self.initial.copy()
        self.goal = task.crawl_target.copy()
        values = dict(zip(PARAMETERS, self.parameters, strict=True))
        offsets = {"hip_pitch": "hip_pitch_offset", "knee": "knee_offset", "ankle_pitch": "ankle_offset",
                   "hip_roll": "hip_roll_offset", "hip_yaw": "hip_yaw_offset", "shoulder_pitch": "shoulder_pitch_offset",
                   "shoulder_roll": "shoulder_roll_offset", "shoulder_yaw": "shoulder_yaw_offset", "elbow": "elbow_offset"}
        for index, name in enumerate(self.names):
            sign = 1. if name.startswith("left") else -1.
            if name.endswith("hip_pitch_joint"):
                self.prepared[index] += values["hip_prepare"]
            elif name.endswith("ankle_pitch_joint"):
                self.prepared[index] += values["ankle_prepare"]
            elif name.endswith("shoulder_roll_joint"):
                self.prepared[index] = sign * values["shoulder_roll_prepare"]
            elif name.endswith("shoulder_pitch_joint"):
                self.prepared[index] = values["shoulder_pitch_prepare"]
            for suffix, key in offsets.items():
                if name.endswith(suffix + "_joint"):
                    multiplier = sign if "roll" in suffix or "yaw" in suffix else 1.
                    self.goal[index] += multiplier * values[key]
        self.goal = np.clip(self.goal, task.model.actuator_ctrlrange[:, 0], task.model.actuator_ctrlrange[:, 1])
        self.fold_start = None
        self.fold_initial = None

    def target(self, task):
        elapsed = float(task.data.time)
        rotation = task.data.xmat[task.model.body("pelvis").id].reshape(3, 3)
        pitch = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        if self.fold_start is None and elapsed >= .6 and pitch >= self.parameters[2]:
            self.fold_start = elapsed
            self.fold_initial = task.data.ctrl.copy()
        if self.fold_start is None:
            return self.initial + smooth((elapsed - .2) / .6) * (self.prepared - self.initial)
        progress = (elapsed - self.fold_start + .02) / self.parameters[3]
        target = self.fold_initial.copy()
        for index, name in enumerate(self.names):
            phase = progress
            if any(part in name for part in ("shoulder", "elbow", "wrist")):
                phase *= 1. + self.parameters[4]
            elif name.endswith("hip_roll_joint") or name.endswith("hip_yaw_joint"):
                phase = (phase - self.parameters[5]) / (1. - self.parameters[5])
            target[index] += smooth(phase) * (self.goal[index] - self.fold_initial[index])
        return np.clip(target, task.model.actuator_ctrlrange[:, 0], task.model.actuator_ctrlrange[:, 1])
