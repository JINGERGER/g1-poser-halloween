import numpy as np

from check_standing_endpoint import evaluate
from unitree_balance_probes import assess, inputs


def main():
    names = [f"{side}_{suffix}_joint" for side in ("left", "right")
             for suffix in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    for kind, axis in (("push_forward", 0), ("push_lateral", 1)):
        forces = np.array([inputs(kind, step, names)[0] for step in range(2500)])
        expected = np.zeros(3)
        expected[axis] = 3.
        np.testing.assert_array_equal(forces.sum(axis=0) * .002, expected)
        assert not forces[999].any() and not forces[1100].any()
        assert forces[1000, axis] == 15. and forces[1099, axis] == 15.
    assert not inputs("lower", 1000, names)[1].any()
    np.testing.assert_allclose(inputs("lower", 1500, names)[1], [-.1, 0., 0., .2, -.1, 0.] * 2)
    np.testing.assert_array_equal(inputs("lower", 1500, names)[1], inputs("lower", 2499, names)[1])
    samples = np.zeros((2501, 13))
    samples[:, 0] = np.arange(2501) * .002
    samples[:, 1] = .75
    samples[:, 5:7] = 150.
    positions = np.zeros((2501, 3))
    gates = evaluate(samples, True, True)
    assert assess("push_forward", samples, samples, gates, positions, positions)["probe_passed"]
    lower = samples.copy()
    lower[1500:, 1] -= .02
    assert assess("lower", lower, samples, evaluate(lower, True, True), positions, positions)["probe_passed"]
    assert not assess("lower", samples, samples, gates, positions, positions)["probe_passed"]
    transient = samples.copy()
    transient[1000:1500, 1] -= .02
    assert not assess("lower", transient, samples, gates, positions, positions)["probe_passed"]
    moving = samples.copy()
    moving[2000:, 3] = .06
    assert not assess("push_forward", moving, samples, evaluate(moving, True, True), positions, positions)["probe_passed"]
    bad = lower.copy()
    bad[1000, 9] = .002
    assert not assess("lower", bad, samples, evaluate(bad, True, True), positions, positions)["probe_passed"]
    assert not assess("push_forward", samples[:-1], samples, evaluate(samples[:-1], False, True), positions[:-1], positions)["probe_passed"]
    print("Probe schedule, exact impulse, lowering and recovery positive/negative tests passed")


if __name__ == "__main__":
    main()
