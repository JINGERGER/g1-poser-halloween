import numpy as np

from termination_returns import compare_returns


def main():
    rewards = np.full((4, 3), -1.0)
    failures = np.zeros((4, 3), dtype=bool)
    failures[1:, 0] = True
    failures[-1, 1] = True
    result = compare_returns(rewards, failures, 0.9, 0.02)
    first, last, never = result["episodes"]
    assert np.isclose(first["stopped_discounted_return"], -1.9)
    assert np.isclose(first["full_discounted_return"], -3.439)
    assert np.isclose(first["stopping_benefit"], 1.539)
    assert first["first_failure_s"] == 0.04
    assert last["stopping_benefit"] == 0
    assert never["stopping_benefit"] is None and never["first_failure_s"] is None
    positive = compare_returns(-rewards, failures, 0.9, 0.02)
    assert positive["episodes"][0]["stopping_benefit"] < 0
    assert positive["summary"]["fraction_failed_episodes_with_stopping_benefit"] == 0
    assert compare_returns(rewards, np.zeros_like(failures), 1, 0.02)["summary"]["mean_stopping_benefit"] is None
    for bad_rewards, bad_failures, gamma, dt in ((np.full((2, 1), np.nan), np.zeros((2, 1)), .99, .02),
                                                 (rewards, failures[:2], .99, .02), (rewards, failures, 0, .02)):
        try:
            compare_returns(bad_rewards, bad_failures, gamma, dt)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid comparison inputs accepted")
    print("PASS: terminal reward included, discounted tail, no-failure/last-step cases, positive tail and invalid inputs")


if __name__ == "__main__":
    main()
