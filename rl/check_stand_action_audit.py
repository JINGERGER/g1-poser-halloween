import numpy as np

from audit_stand_transition_actions import reference_target


def main():
    names = ["left_knee_joint", "right_knee_joint", "left_elbow_joint", "right_elbow_joint"]
    start = np.array([.6, .6, .3, .3])
    goal = np.array([2., 2., -.3, -.3])
    for route in ("together", "legs_lead", "arms_lead"):
        for duration in (.5, 1., 2.5, 5.):
            np.testing.assert_array_equal(reference_target(start, goal, names, 0., duration, route), start)
            np.testing.assert_allclose(reference_target(start, goal, names, duration, duration, route), goal)
            np.testing.assert_allclose(reference_target(start, goal, names, duration + .5, duration, route), goal)
            for elapsed in np.linspace(0., duration, 101):
                target = reference_target(start, goal, names, elapsed, duration, route)
                assert np.all(target >= np.minimum(start, goal) - 1e-12)
                assert np.all(target <= np.maximum(start, goal) + 1e-12)
                assert target[0] == target[1] and target[2] == target[3]
            before = reference_target(start, goal, names, duration * .5 - 1e-7, duration, route)
            after = reference_target(start, goal, names, duration * .5 + 1e-7, duration, route)
            np.testing.assert_allclose(before, after, atol=1e-5)
    for duration in (0., -1., float("nan"), float("inf")):
        try:
            reference_target(start, goal, names, 0., duration, "together")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid duration accepted")
    print("Reference endpoint, continuity, bounds and bilateral timing checks passed")


if __name__ == "__main__":
    main()
