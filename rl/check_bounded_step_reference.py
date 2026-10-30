import json

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_heel_roll_routes import SOURCE
from check_predictive_step_chain import reference_gates
from check_single_support_reference import plan
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state


def main():
    node = {"foot_error_m": 0., "com_height_error_m": .01, "com_error_xy_m": .006,
            "com_error_m": .012, "self_penetration_m": 0., "equilibrium_feasible": True}
    assert not reference_gates([node], adaptive_height=True)["com_reference"]
    node["com_error_xy_m"] = .004
    assert reference_gates([node], adaptive_height=True)["com_reference"]
    node.update(balance_region=True, support_region_violation_m=.001, com_error_xy_m=0.)
    assert not reference_gates([node], adaptive_height=True)["com_reference"]
    node.update(support_region_violation_m=-.002, com_error_xy_m=.03)
    assert reference_gates([node], adaptive_height=True)["com_reference"]
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    report = json.loads(SOURCE.read_text())
    trajectory = ROOT / report["trajectory"]
    assert digest(trajectory) == report["trajectory_sha256"]
    with np.load(trajectory) as saved:
        state["qpos"] = saved["qpos"][-1].tolist()
        state["qvel"] = saved["qvel"][-1].tolist()
    task.data.qpos[:] = state["qpos"]
    task.data.qvel[:] = state["qvel"]
    mujoco.mj_forward(task.model, task.data)
    before = integration_state(task)
    nodes = plan(task, .04, .2, side=1, bounded_continuation=True)
    poses = np.array([node["qpos"] for node in nodes])
    np.testing.assert_allclose(poses[0], state["qpos"], atol=1e-10, rtol=0.)
    assert np.max(np.abs(np.diff(poses[:, task.addresses], axis=0))) <= .120001
    np.testing.assert_array_equal(integration_state(task), before)
    region_nodes = plan(task, .04, .2, side=1, bounded_continuation=True, balance_region=True, height_tolerance=.02)
    np.testing.assert_allclose(region_nodes[0]["qpos"], state["qpos"], atol=1e-10, rtol=0.)
    limits = np.minimum(-task.model.jnt_actfrcrange[task.joints, 0], task.model.jnt_actfrcrange[task.joints, 1])
    for node in region_nodes:
        if node["equilibrium_feasible"]:
            assert np.max(np.abs(node["torque_nm"]) / limits) <= .700001
    np.testing.assert_array_equal(integration_state(task), before)
    print("PASS: horizontal COM gate, exact initial reference, bounded joint increments, planner state immutability; offline test only")


if __name__ == "__main__":
    main()
