import numpy as np

from check_low_support_transition import target_at


def main():
    nodes = [{"qpos": [0., phase, phase * 2.]} for phase in np.linspace(0., 1., 11)]
    addresses = np.array([1, 2])
    np.testing.assert_array_equal(target_at(nodes, addresses, 0.), [1., 2.])
    np.testing.assert_array_equal(target_at(nodes, addresses, .8), [0., 0.])
    np.testing.assert_array_equal(target_at(nodes, addresses, 2.), [0., 0.])
    previous = np.array([1., 2.])
    for elapsed in np.linspace(0., .8, 101):
        target = target_at(nodes, addresses, elapsed)
        assert np.all(target <= previous + 1e-12) and np.all(target >= 0.)
        previous = target
    for duration in (0., -1., float("nan"), float("inf")):
        try:
            target_at(nodes, addresses, 0., duration)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid duration accepted")
    print("Low-support reference endpoints, ordering, bounds and invalid durations passed")


if __name__ == "__main__":
    main()
