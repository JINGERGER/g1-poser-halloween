import mujoco
import numpy as np

from position_torque_guard import PositionTorqueGuard
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth
from verify_crawl_candidate import reconstruct


class CoordinatedBlendTask(StandTransitionTask):
    blend_duration = .2

    def finish(self):
        if (not np.isfinite(self.blend_duration) or self.blend_duration <= 0.
                or abs(self.blend_duration / .02 - round(self.blend_duration / .02)) > 1e-9):
            raise ValueError("Blend duration must be a positive multiple of 20ms")
        torque_before = self.data.qfrc_actuator[self.model.jnt_dofadr[self.joints]].copy()
        self.phase = "crawl"
        self.handoff = self.execution.handoff()
        self.crawl_gain, self.crawl_bias = self.model.actuator_gainprm.copy(), self.model.actuator_biasprm.copy()
        bias = self.model.actuator_biasprm[:, 0] + self.model.actuator_biasprm[:, 1] * self.data.actuator_length + self.model.actuator_biasprm[:, 2] * self.data.actuator_velocity
        anchor = (torque_before - bias) / self.model.actuator_gainprm[:, 0]
        guard = PositionTorqueGuard(self.model)
        bridge_start = len(self.samples)
        clipped = 0
        try:
            for step in range(round(self.blend_duration / .02)):
                if self.execution.pending_control:
                    self.execution.pending_control = False
                else:
                    self.execution.controller.update(self.data)
                fraction = smooth((step + 1) * .02 / self.blend_duration)
                requested = (1. - fraction) * anchor + fraction * self.data.ctrl.copy()
                for unused in range(10):
                    if self.data.qfrc_applied.any() or self.data.xfrc_applied.any():
                        raise ValueError("Assistance prohibited")
                    control = guard.apply(self.data, requested)
                    clipped += int(np.any(np.abs(control - requested) > 1e-10))
                    self.data.ctrl[:] = control
                    mujoco.mj_step(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    self._observe_step(self.model, self.data)
        except UnsafeState:
            pass
        bridge_steps = len(self.samples) - bridge_start
        start = self.data.qpos[:3].copy()
        sample_start = len(self.samples)
        if not self.failure:
            try:
                for unused in range(100):
                    self.execution.crawl_step()
            except UnsafeState:
                pass
        rows = np.asarray(self.samples[sample_start:])
        positions = np.asarray(self.support_positions[sample_start:])
        events, cycles, consecutive = reconstruct(rows, positions) if len(rows) else ([], [], False)
        forward = float(self.data.qpos[0] - start[0])
        gates = {"physical_substeps": not self.failure, "bridge_completed": bridge_steps == round(self.blend_duration / .002),
                 "completed_2s": len(rows) == 1000, "consecutive_cycles": bool(consecutive), "forward_6cm": forward >= .06}
        return {"success": bool(all(gates.values())), "failure": self.failure, "gates": gates,
                "cycle_count": len(cycles), "forward_m": forward, "crawl_steps": len(rows),
                "bridge_steps": bridge_steps, "bridge_clipped_steps": clipped, "pure_crawl_start_index": sample_start,
                "bridge_duration_s": self.blend_duration,
                "limitations": "Frozen gains switch once; torque-preserving anchor and 80% torque guard apply only during bridge. Final two seconds use original unguarded frozen policy with continuous history."}
