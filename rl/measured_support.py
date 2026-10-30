import mujoco
import numpy as np


def support_contacts(task):
    contacts = []
    for contact_index, contact in enumerate(task.data.contact):
        if task.floor not in (contact.geom1, contact.geom2) or contact.efc_address < 0 or contact.dist > 0.:
            continue
        geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
        if geom not in task.supports:
            continue
        frame = contact.frame.reshape(3, 3).copy()
        if contact.geom2 == task.floor:
            frame = -frame
        wrench = np.zeros(6)
        mujoco.mj_contactForce(task.model, task.data, contact_index, wrench)
        contacts.append({"geom": int(geom), "point": contact.pos.copy(), "frame": frame,
                         "distance": float(contact.dist), "load": float(max(0., wrench[0])),
                         "friction": float(min(.6, *contact.friction[:2]))})
    return contacts


class LoadTransfer:
    def __init__(self, weight):
        self.weight = float(weight)
        self.confirmed = np.zeros(4, dtype=int)
        self.hand_age = np.zeros(2)
        self.mode = "feet"
        self.phase = 0.
        self.last = {}

    def update(self, loads, present, vertical_speed, tracking_error, dt=.02, advance=True):
        loads = np.asarray(loads)
        present = np.asarray(present, dtype=bool)
        loaded = present & (loads > 5.)
        self.confirmed = np.where(loaded, self.confirmed + 1, 0)
        self.hand_age = np.where(loaded[:2], self.hand_age + dt, 0.)
        hands_ready = bool(np.all(self.confirmed[:2] >= 3))
        feet_ready = bool(np.all(self.confirmed[2:] >= 3))
        self.mode = "transfer" if hands_ready else ("touch" if present[:2].any() else "feet")
        reasons = []
        if not (feet_ready or hands_ready):
            reasons.append("unconfirmed_support")
        if loads.sum() < .7 * self.weight:
            reasons.append("insufficient_measured_load")
        if abs(vertical_speed) > .10:
            reasons.append("vertical_speed")
        if abs(tracking_error) > .04:
            reasons.append("com_tracking")
        if advance and not reasons:
            self.phase = min(1., self.phase + dt / 12.)
        self.last = {"mode": self.mode, "phase": self.phase, "pause_reasons": reasons,
                     "measured_loads_n": loads.tolist(), "present": present.tolist(),
                     "hands_confirmed": hands_ready, "feet_confirmed": feet_ready}
        return self.phase

    def hand_capacity(self, index):
        return self.weight * (.1 + .9 * min(1., self.hand_age[index] / .3))
