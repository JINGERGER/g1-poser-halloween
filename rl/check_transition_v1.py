import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from reference_policy_baseline import ROOT
from run_transition_v1 import forward_target
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state
from transition_v1 import DEFAULT, StandingTransitionV1
from whole_body_torque_bridge_v1 import TorquePositionBridge
from whole_body_transition_v1 import WholeBodyLowering


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    before = integration_state(task)
    for elapsed in (0., .5, 1.1, 2., 2.6, 3.2, 5.):
        target = forward_target(task, elapsed)
        assert target.shape == (23,) and np.isfinite(target).all()
        assert np.all(target >= task.model.actuator_ctrlrange[:, 0])
        assert np.all(target <= task.model.actuator_ctrlrange[:, 1])
    np.testing.assert_array_equal(forward_target(task, 0.), task.target)
    np.testing.assert_array_equal(forward_target(task, 5.), task.target)
    np.testing.assert_array_equal(integration_state(task), before)
    controller = StandingTransitionV1(task, DEFAULT)
    np.testing.assert_array_equal(controller.target(task), task.target)
    for parameters in (np.zeros(3), np.full(DEFAULT.shape, np.nan), DEFAULT + 100.):
        try:
            StandingTransitionV1(task, parameters)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid controller parameters accepted")
    whole_body = WholeBodyLowering(task)
    whole_body.target(whole_body.initial_com[2], 0., 0.)
    np.testing.assert_array_equal(integration_state(task), before)
    assert whole_body.last["constraint_residual"] <= .02
    TorquePositionBridge(task.model).advance_torque(task.execution, whole_body.last["torque_nm"])
    assert not task.failure and not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
    with tempfile.TemporaryDirectory() as directory:
        task.save_and_replay(Path(directory) / "qp_bridge.npz")
    report_path = ROOT / "output/rl/transition_v1/2026-10-28_15-32-07-891249/report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed" and len(report["cases"]) == 9
    assert not report["full_transition_validated"]
    for name, expected in report["sources"].items():
        assert digest(ROOT / "rl" / name) == expected
    for case in report["cases"]:
        assert case["forward_stage_passed"] and all(case["stage_gates"].values())
        assert not case["full_transition_passed"] and case["handoff"] is None
        trajectory = ROOT / case["trajectory"]
        assert digest(trajectory) == case["trajectory_sha256"]
        saved = np.load(trajectory)
        assert len(saved["samples"]) == 2500 and np.all(saved["phase"] == 0)
        np.testing.assert_array_equal(saved["qpos"][0], case["initial_state"]["qpos"])
    print("PASS: target bounds/state immutability, invalid parameters, QP constraints/bridge replay, nine forward-stage reports; no full-transition success claimed")


if __name__ == "__main__":
    main()
