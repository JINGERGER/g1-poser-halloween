import mujoco
import numpy as np
import torch

from arm_clearance_guard import ArmClearanceGuard
from recovery_transition_task import RecoveryTransitionTask


def main():
    torch.set_num_threads(1)
    task = RecoveryTransitionTask()
    state = task.make_states([0])[0]
    task.reset(state)
    guard = ArmClearanceGuard(task.model)
    before_position, before_velocity = task.data.qpos.copy(), task.data.qvel.copy()
    projected = guard.apply(task.data, task.target)
    assert projected.shape == (23,) and np.isfinite(projected).all()
    np.testing.assert_array_equal(task.data.qpos, before_position)
    np.testing.assert_array_equal(task.data.qvel, before_velocity)
    constraints = guard.constraints(task.data)
    assert constraints
    for first, second in guard.pairs:
        distance = mujoco.mj_geomDistance(task.model, task.data, first, second, .25, None)
        if abs(distance) > 1e-8 and distance < .25:
            break
    gradient, unused_bound = constraints[0]
    actuator = int(np.argmax(np.abs(gradient)))
    address = guard.addresses[actuator]
    probe = mujoco.MjData(task.model)
    distances = []
    for sign in (-1., 1.):
        probe.qpos[:] = before_position
        probe.qpos[address] += sign * 1e-6
        mujoco.mj_forward(task.model, probe)
        distances.append(mujoco.mj_geomDistance(task.model, probe, first, second, .25, None))
    np.testing.assert_allclose((distances[1] - distances[0]) / 2e-6, gradient[actuator], atol=.002, rtol=.02)
    print("Arm clearance: finite bounded output, no state override, distance Jacobian finite-difference check passed")


if __name__ == "__main__":
    main()
