import json

import mujoco
import numpy as np

from short_step_assets import prepare
from short_step_task import StepEvents
from exploration_task import ExplorationCfg as ShortStepCfg, ExplorationEnv as ShortStepEnv, training_guard
from component_unload_ramps import guard


def check_events(protocol):
    initial = np.array([0., 0, .02])
    row = {"loads_n": [80., 80, 80, 80], "gaps_m": [0., 0, 0, 0],
           "floor_contacts": [True] * 4, "loaded_slip_mps": 0.}
    event = StepEvents()
    for unused in range(200):
        event.update(row, initial + [.03, 0, 0], initial, 0, .002, protocol)
    assert event.stage == 0
    row["loads_n"][0] = 0
    event.update(row, initial, initial, 0, .002, protocol)
    assert event.stage == 1
    row["gaps_m"][0] = .02
    row["floor_contacts"][0] = False
    for unused in range(20):
        event.update(row, initial + [.03, 0, .02], initial, 0, .002, protocol)
    assert event.stage == 1
    for unused in range(20):
        event.update(row, initial + [0, 0, .02], initial, 0, .002, protocol)
    assert event.stage == 2
    event.update(row, initial + [.03, 0, .02], initial, 0, .002, protocol)
    assert event.stage == 3
    row["loads_n"][0], row["gaps_m"][0], row["floor_contacts"][0] = 40, 0, True
    for unused in range(200):
        event.update(row, initial + [.03, 0, 0], initial, 1., .002, protocol)
    assert event.stage == 3
    for unused in range(150):
        event.update(row, initial + [.03, 0, 0], initial, 0., .002, protocol)
    assert event.stage == 4
    dragged = StepEvents()
    row["loads_n"][0], row["gaps_m"][0], row["floor_contacts"][0] = 0, .02, False
    for unused in range(100):
        dragged.update(row, initial + [.03, 0, .02], initial, 0., .002, protocol)
    assert dragged.stage == 2
    row["loads_n"][0], row["gaps_m"][0] = 40, 0
    dragged.update(row, initial + [.06, 0, 0], initial, 0., .002, protocol)
    row["loads_n"][0], row["gaps_m"][0] = 0, .02
    dragged.update(row, initial + [.06, 0, .02], initial, 0., .002, protocol)
    assert dragged.stage == 2


def main():
    bundle = prepare()
    assert prepare() == bundle
    payload = json.loads(bundle.read_text())
    check_events(payload["protocol"])
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle), reference_gain=0.), 2)
    state = env.init_state()
    metric = dict(env.metrics[0])
    metric["loads_n"] = [90., 0., 35., 130.]
    metric["floor_contacts"] = [True, False, True, True]
    assert guard(env.model, env.datas[0], metric, .0049) == "remaining_support_lost"
    assert training_guard(env.model, env.datas[0], metric, .0049) is None
    assert training_guard(env.model, env.datas[0], metric, .006) is None
    assert training_guard(env.model, env.datas[0], metric, .021) == "diagnostic_penetration_over_20mm"
    metric["other_total_n"] = 31.
    assert training_guard(env.model, env.datas[0], metric, .0049) == "body_support_over_30N"
    assert state.obs["obs"].shape == (2, 148)
    np.testing.assert_array_equal(state.obs["obs"][0], state.obs["obs"][1])
    native = mujoco.MjData(env.model)
    mujoco.mj_setState(env.model, native, env.initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    native.ctrl[:] = env.initial_ctrl
    mujoco.mj_forward(env.model, native)
    for unused in range(10):
        mujoco.mj_step(env.model, native)
        mujoco.mj_forward(env.model, native)
    state = env.step(np.zeros((2, 23)))
    np.testing.assert_allclose(env.datas[0].qpos, native.qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(env.datas[0].qvel, native.qvel, atol=1e-12, rtol=0)
    for unused in range(199):
        state = env.step(np.zeros((2, 23)))
    assert state.terminated.all() and not state.truncated.any()
    assert all(row["deadline"] and not row["success"] for row in state.info["evaluation"])
    assert state.final_observation is not None
    assert np.all(env.elapsed == 0)
    assert not np.array_equal(state.final_observation["obs"], state.obs["obs"])
    env.datas[0].qpos[2] = .05
    state = env.step(np.zeros((2, 23)))
    assert state.terminated[0] and not state.terminated[1]
    assert state.reward[0] < -9 and state.info["evaluation"][0]["failure"]
    assert env.elapsed[0] == 0 and env.elapsed[1] > 0
    assert state.final_observation is not None
    for invalid in (np.zeros((2, 22)), np.full((2, 23), np.nan)):
        try:
            env.step(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action accepted")
    env.close()
    print("PASS: immutable assets, ordered events/anti-shortcuts, native integration, finite-horizon termination, final observations, partial autoreset, unsafe-state penalty and invalid actions")
    print(bundle)
    print("IK sampled maximum residual m:", max(payload["ik_error_m"]))


if __name__ == "__main__":
    main()
