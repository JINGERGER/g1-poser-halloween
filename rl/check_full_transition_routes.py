import numpy as np

from compare_full_transition_routes import joint_target, schedule


def main():
    for route in ("coordinated", "lean_first", "lower_first"):
        np.testing.assert_array_equal(schedule(0., route), np.zeros(3))
        np.testing.assert_array_equal(schedule(1., route), np.ones(3))
        phases = np.array([schedule(phase, route) for phase in np.linspace(0., 1., 101)])
        assert np.all(np.diff(phases, axis=0) >= 0.)
        assert np.all((phases >= 0.) & (phases <= 1.))
    assert schedule(.5, "lean_first")[0] > schedule(.5, "lean_first")[1]
    assert schedule(.5, "lower_first")[0] < schedule(.5, "lower_first")[1]
    nodes = [{"qpos": [9., phase, -phase]} for phase in np.linspace(0., 1., 21)]
    addresses = np.array([1, 2])
    for phase in np.linspace(-.1, 1.1, 101):
        bounded = np.clip(phase, 0., 1.)
        np.testing.assert_allclose(joint_target(nodes, addresses, phase), [bounded, -bounded], atol=1e-12)
    try:
        schedule(.5, "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown route accepted")
    print("Full-route phase endpoints, monotonicity, route ordering and joint-only interpolation passed")


if __name__ == "__main__":
    main()
