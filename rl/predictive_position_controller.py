from predictive_contact_controller import PredictiveContactController


class PredictivePositionController(PredictiveContactController):
    def prediction_target(self, torque, probe):
        return super().prediction_target(torque, self.task.data)

    def target(self, *args, **kwargs):
        task = self.task
        self.previous_torque = task.data.qfrc_actuator[task.model.jnt_dofadr[task.joints]].copy()
        requested = super().target(*args, **kwargs)
        self.last["position_target"] = requested.tolist()
        return requested
