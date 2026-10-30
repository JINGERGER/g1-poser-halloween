import tempfile
from pathlib import Path

import numpy as np
import torch

from position_torque_guard import PositionTorqueGuard
from stand_transition_task import StandTransitionTask
from standing_capture_routes import fork_prefix, integration_state, route_target
from standing_cartesian_capture import reaching_target


def main():
    torch.set_num_threads(1)
    source, branch = StandTransitionTask(), StandTransitionTask()
    state = source.make_states([0])[0]
    state["qvel"] = np.zeros(source.model.nv).tolist()
    source.reset(state)
    guard = PositionTorqueGuard(source.model, fraction=.8)
    for step in range(30):
        target = route_target(source, "hinge", (step + 1) * .02, 1., .3, 0.)
        guard.advance(source.execution, target)
    before = integration_state(source)
    gains, biases = source.model.actuator_gainprm.copy(), source.model.actuator_biasprm.copy()
    count = len(source.samples)
    fork_prefix(source, branch)
    assert source.model is not branch.model and source.data is not branch.data
    assert not np.shares_memory(source.positions[-1], branch.positions[-1])
    branch.finish()
    with tempfile.TemporaryDirectory() as folder:
        branch.save_and_replay(Path(folder) / "branch.npz")
    np.testing.assert_array_equal(integration_state(source), before)
    np.testing.assert_array_equal(source.model.actuator_gainprm, gains)
    np.testing.assert_array_equal(source.model.actuator_biasprm, biases)
    assert len(source.samples) == count
    goals = source.data.geom_xpos[source.supports[:2]].copy()
    requested = source.data.qpos[source.addresses].copy()
    target = reaching_target(source, requested, goals)
    np.testing.assert_allclose(target, requested, atol=1e-8)
    goals[:, 0] += .01
    target = reaching_target(source, requested, goals)
    np.testing.assert_array_equal(target[:13], requested[:13])
    np.testing.assert_array_equal(integration_state(source), before)
    assert np.isfinite(target).all()
    assert np.all(target >= source.model.actuator_ctrlrange[:, 0])
    assert np.all(target <= source.model.actuator_ctrlrange[:, 1])
    for family in ("hinge", "squat", "staggered"):
        np.testing.assert_array_equal(route_target(source, family, 0., 1., .3, 0.), source.target)
        np.testing.assert_allclose(route_target(source, family, 3., 1., .3, 0.), source.crawl_target, atol=1e-14, rtol=0)
    print("PASS: exact fork, independent state/model/history, full-prefix replay, IK immutability and bounds, route endpoints")


if __name__ == "__main__":
    main()
