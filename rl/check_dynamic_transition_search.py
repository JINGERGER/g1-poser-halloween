import numpy as np

from search_dynamic_transition import command_at, ranking, summarize_prefix


def main():
    start, goal = np.zeros(23), np.ones(23)
    parameters = np.r_[np.full(23, .25), np.full(23, .5), np.full(23, .75), 2.]
    np.testing.assert_array_equal(command_at(parameters, start, goal, -goal, goal, 0.), start)
    np.testing.assert_array_equal(command_at(parameters, start, goal, -goal, goal, 3.), goal)
    for elapsed in np.linspace(0., 3., 101):
        target = command_at(parameters, start, goal, -goal, goal, elapsed)
        assert np.all(target >= -1.) and np.all(target <= 1.)
    for bad in (np.zeros(69), np.full(70, np.nan)):
        try:
            command_at(bad, start, goal, -goal, goal, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid parameters accepted")
    samples = np.zeros((51, 15))
    samples[:, 4:6] = 6.
    samples[:, 8:10] = .02
    assert summarize_prefix(samples, False)["both_hands_s"] == .1
    assert summarize_prefix(samples, True)["both_hands_s"] == .098
    assert summarize_prefix(samples[:1], True)["both_hands_s"] == 0.
    samples[:-1, 4:6] = 0.
    assert summarize_prefix(samples, True)["either_hand_s"] == 0.
    failed = {"success": False, "both_hands_s": 1., "either_hand_s": 1., "gap_progress": 1.}
    success = {"success": True, "both_hands_s": 0., "either_hand_s": 0., "gap_progress": 0.}
    assert ranking(success) > ranking(failed)
    supported = {"success": False, "both_hands_s": .1, "either_hand_s": .1, "gap_progress": 0.}
    close = {"success": False, "both_hands_s": 0., "either_hand_s": 0., "gap_progress": 1.}
    assert ranking(supported) > ranking(close)
    print("Dynamic search: targets, invalid input, failed-sample exclusion and success/support priority passed")


if __name__ == "__main__":
    main()
