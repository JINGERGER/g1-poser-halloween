import numpy as np

from position_torque_guard import InfeasiblePositionTarget, guarded_target


def main():
    gains, limits = np.array([100., 40.]), np.array([80., 20.])
    bias, lower, upper = np.array([-20., 30.]), np.array([-2., -2.]), np.array([2., 2.])
    for requested in ([0., 0.], [100., -100.], [-100., 100.]):
        target = guarded_target(requested, bias, gains, limits, lower, upper, .8)
        assert np.all(np.abs(gains * target + bias) <= .8 * limits + 1e-10)
        assert np.all((target >= lower) & (target <= upper))
    requested = np.array([.2, -.75])
    np.testing.assert_array_equal(guarded_target(requested, bias, gains, limits, lower, upper, .8), requested)
    try:
        guarded_target(requested, bias + 1e6, gains, limits, lower, upper, .8)
    except InfeasiblePositionTarget:
        pass
    else:
        raise AssertionError("Infeasible target accepted")
    print("Torque guard: predicted PD bounds, control bounds, feasible invariance and infeasibility passed")


if __name__ == "__main__":
    main()
