import mujoco
import numpy as np

from crawl_control import CrawlController


def physical_state(model, data):
    specification = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, specification))
    mujoco.mj_getState(model, data, state, specification)
    return state


class TransitionExecution:
    def __init__(self, model, data, reference_model, constants, policy, step_observer=None):
        if model.nu != 23 or model.na != 0 or model.opt.timestep != .002:
            raise ValueError("Expected the frozen 23-actuator, 2ms position-servo model")
        if not np.all(model.actuator_ctrllimited) or not np.all(model.actuator_biasprm[:, 1] < 0.):
            raise ValueError("Torque-motor interface cannot receive position targets")
        if not np.all(model.geom_solref[:, 0] == .005):
            raise ValueError("Contact profile must already match the frozen crawl configuration")
        self.model, self.data = model, data
        self.reference_model, self.constants, self.policy = reference_model, constants, policy
        self.phase = "transition"
        self.controller = None
        self.pending_control = False
        self.step_observer = step_observer

    def _advance(self):
        for unused in range(10):
            if self.data.xfrc_applied.any() or self.data.qfrc_applied.any():
                raise ValueError("Transition execution does not permit assistance forces")
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise ValueError("Nonfinite physics state")
            if self.step_observer is not None:
                self.step_observer(self.model, self.data)

    def transition_step(self, target):
        if self.phase != "transition":
            raise RuntimeError("Cannot send transition targets after crawl handoff")
        target = np.asarray(target, dtype=float)
        if target.shape != (self.model.nu,) or not np.isfinite(target).all():
            raise ValueError("Expected finite targets in actuator order")
        if np.any(target < self.model.actuator_ctrlrange[:, 0]) or np.any(target > self.model.actuator_ctrlrange[:, 1]):
            raise ValueError("Transition target exceeds actuator position range")
        self.data.ctrl[:] = target
        self._advance()

    def _instantaneous_torque(self):
        probe = mujoco.MjData(self.model)
        mujoco.mj_copyData(probe, self.model, self.data)
        mujoco.mj_forward(self.model, probe)
        return probe.qfrc_actuator.copy()

    def handoff(self):
        if self.phase != "transition":
            raise RuntimeError("Crawl history cannot be reset by repeated handoff")
        model, data = self.model, self.data
        before = physical_state(model, data)
        warmstart = data.qacc_warmstart.copy()
        control_before = data.ctrl.copy()
        torque_before = self._instantaneous_torque()
        gain_before, bias_before = model.actuator_gainprm.copy(), model.actuator_biasprm.copy()
        reference = self.reference_model
        mapping = {reference.joint(int(joint)).name: index for index, joint in enumerate(reference.actuator_trnid[:, 0])}
        for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
            name = model.joint(int(joint)).name
            if name in mapping:
                source = mapping[name]
                model.actuator_gainprm[actuator, 0] = reference.actuator_gainprm[source, 0] * 1.4
                model.actuator_biasprm[actuator, 1:3] = reference.actuator_biasprm[source, 1:3] * 1.4
        self.controller = CrawlController(model, self.constants, self.policy)
        self.controller.update(data)
        torque_after = self._instantaneous_torque()
        np.testing.assert_array_equal(physical_state(model, data), before)
        np.testing.assert_array_equal(data.qacc_warmstart, warmstart)
        self.phase, self.pending_control = "crawl", True
        return {"time_s": float(data.time), "physical_state_unchanged": True, "warmstart_unchanged": True,
                "previous_action_initialization": "zero; first crawl action evaluated exactly once at handoff",
                "gain_before": gain_before.tolist(), "bias_before": bias_before.tolist(),
                "gain_after": model.actuator_gainprm.tolist(), "bias_after": model.actuator_biasprm.tolist(),
                "control_before": control_before.tolist(), "control_after": data.ctrl.tolist(),
                "max_target_jump_rad": float(np.max(np.abs(data.ctrl - control_before))),
                "max_instantaneous_generalized_torque_jump_nm": float(np.max(np.abs(torque_after - torque_before))),
                "limitations": "Abrupt explicit gain/target switch, not certified safe. No readiness or contact gate; caller must validate entry."}

    def crawl_step(self):
        if self.phase != "crawl":
            raise RuntimeError("Explicit handoff required")
        if not self.pending_control:
            self.controller.update(self.data)
        self.pending_control = False
        self._advance()
