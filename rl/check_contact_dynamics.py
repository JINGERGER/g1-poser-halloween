import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from internal_constraint_force import internal_constraint_force
from position_torque_guard import PositionTorqueGuard
from predictive_contact_controller import PredictiveContactController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state
from whole_body_torque_bridge_v1 import TorquePositionBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--nonlinear-search", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    guard = PositionTorqueGuard(task.model)
    for unused in range(25):
        guard.advance(task.execution, task.target)
    before = integration_state(task)
    internal = internal_constraint_force(task.model, task.data, task.floor)
    ground = np.zeros(task.model.nv)
    for index, contact in enumerate(task.data.contact):
        if task.floor not in (contact.geom1, contact.geom2):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(task.model, task.data, index, wrench)
        frame = contact.frame.reshape(3, 3)
        for geom, sign in ((contact.geom1, -1.), (contact.geom2, 1.)):
            mujoco.mj_applyFT(task.model, task.data, sign * frame.T @ wrench[:3], sign * frame.T @ wrench[3:],
                              contact.pos, task.model.geom_bodyid[geom], ground)
    assert not np.any(task.data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    np.testing.assert_allclose(internal + ground, task.data.qfrc_constraint, atol=1e-8, rtol=0.)
    np.testing.assert_allclose(internal[:6], 0., atol=1e-8, rtol=0.)
    np.testing.assert_array_equal(integration_state(task), before)
    if args.position:
        from predictive_position_controller import PredictivePositionController
        controller = PredictivePositionController(task)
    else:
        controller = PredictiveContactController(task)
    controller.nonlinear_search = args.nonlinear_search
    bridge = TorquePositionBridge(task.model)
    com = task.data.subtree_com[controller.pelvis].copy()
    hands = task.data.geom_xpos[task.supports[:2]].copy()
    feet = task.data.geom_xpos[task.supports[2:]].copy()
    rotations = task.data.geom_xmat[task.supports[2:]].reshape(2, 3, 3).copy()
    orientation = task.data.xmat[controller.pelvis].reshape(3, 3).copy()
    posture = task.data.qpos[task.addresses].copy()
    max_prediction_error = 0.
    for unused in range(100):
        before = integration_state(task)
        controller.target(com, 0., hands, posture, feet, rotations, rotation_goal=orientation)
        np.testing.assert_array_equal(integration_state(task), before)
        if args.nonlinear_search:
            candidates = controller.last["nonlinear_search"]
            chosen = next(candidate for candidate in candidates if candidate["fraction"] == controller.last["selected_fraction"])
            assert chosen["cost"] == min(candidate["cost"] for candidate in candidates)
        if args.position:
            guard.advance(task.execution, controller.last["position_target"])
        else:
            bridge.advance_torque(task.execution, controller.last["torque_nm"])
        error = float(np.max(np.abs(task.data.qvel - controller.last["predicted_qvel"])))
        max_prediction_error = max(max_prediction_error, error)
        assert error < 1e-10
        assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "hold.npz")
    print(json.dumps({"passed": True, "physical_hold_s": 2., "prediction_velocity_error": max_prediction_error,
                      "controller": type(controller).__name__, "nonlinear_search": args.nonlinear_search,
                      "checks": ["complete constraint-force decomposition", "internal force cancels at floating base",
                                 "planner state immutability", "exact 20ms prediction", "no external forces", "strict physics", "exact replay"]}))


if __name__ == "__main__":
    main()
