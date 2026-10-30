import numpy as np

from event_lift_task import EventLiftCfg, EventLiftEnv, dense_terms, target_position, WEIGHTS
from short_step_assets import prepare
from short_step_task import ShortStepCfg, ShortStepEnv


def main():
    initial = np.array([0., 0., .015])
    lift = target_position(initial, .02, 0)
    advance = target_position(initial, .02, 2)
    land = target_position(initial, .02, 3)
    np.testing.assert_allclose(lift, [0, 0, .035])
    np.testing.assert_allclose(advance, [.03, 0, .035])
    np.testing.assert_allclose(land, [.03, 0, .018])
    old = {"unload": .5, "action_rate": 0., "slip": 0.}
    near = dense_terms(.015, lift, lift, .02, 0, old)
    held = dense_terms(-.005, initial, lift, .02, 0, old)
    dragged = dense_terms(.015, lift + [.03, 0, 0], lift, .02, 0, old)
    assert near["height"] > held["height"] and near["horizontal"] > dragged["horizontal"]
    assert dense_terms(-.002, land, land, .02, 3, old)["height"] > dense_terms(.015, advance, land, .02, 3, old)["height"]
    bundle = str(prepare())
    course = EventLiftEnv(EventLiftCfg(bundle=bundle, reward_config={"scales": WEIGHTS}), 2)
    shadow = ShortStepEnv(ShortStepCfg(bundle=bundle, reference_gain=0.), 2)
    try:
        course.init_state()
        shadow.init_state()
        random = np.random.default_rng(9)
        for step in range(220):
            actions = np.zeros((2, 23)) if step < 200 else random.normal(0., .15, (2, 23))
            actual = course.step(actions)
            strict = shadow.step(actions)
            np.testing.assert_array_equal(actual.terminated, strict.terminated)
            np.testing.assert_array_equal(actual.truncated, strict.truncated)
            for index, row in enumerate(actual.info["evaluation"]):
                for key in ("stage", "success", "failure", "deadline", "elapsed_s", "loads_n", "gaps_m"):
                    assert row[key] == strict.info["evaluation"][index][key]
                np.testing.assert_array_equal(course.datas[index].qpos, shadow.datas[index].qpos)
                np.testing.assert_array_equal(course.datas[index].qvel, shadow.datas[index].qvel)
                np.testing.assert_array_equal(actual.obs["obs"][index, 114:137], 0.)
                expected = target_position(course.initial_hand, .02, course.events[index].stage) - course.datas[index].geom_xpos[course.hand]
                np.testing.assert_allclose(actual.obs["obs"][index, 137:140], expected, atol=1e-8, rtol=0)
                assert np.isfinite(actual.reward[index])
            if np.any(actual.terminated):
                assert actual.final_observation is not None
                np.testing.assert_array_equal(actual.final_observation["obs"][:, :114], strict.final_observation["obs"][:, :114])
        assert len(course.completed) > 0
    finally:
        course.close()
        shadow.close()
    print("PASS: target stages, reward direction, no clock-IK observation leakage, 440 transitions identical native physics/strict events/termination, partial autoreset/final observations")


if __name__ == "__main__":
    main()
