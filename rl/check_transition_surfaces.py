import numpy as np

from audit_transition_surfaces import crossing


def main():
    first = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    second = np.array([[[.5, .5, -1.], [.5, .5, 1.], [1., .5, 1.]]])
    assert crossing(first, second) is not None
    assert crossing(second, first) is not None
    assert crossing(first, second + [0., 0., 3.]) is None
    assert crossing(first, first) is None
    rotation, unused = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    assert crossing(first @ rotation + 42., second @ rotation + 42.) is not None
    assert crossing(first * .001, second * .001) is not None
    print("PASS: crossings, reversal, separation, coplanar non-certification, transforms, millimeter scale")


if __name__ == "__main__":
    main()
