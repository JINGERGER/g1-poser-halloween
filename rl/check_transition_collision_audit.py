import itertools

import numpy as np

from audit_transition_collision import separating_plane


def main():
    cube = np.array(list(itertools.product((-1., 1.), repeat=3)))
    result = separating_plane(cube, cube + [3., 0., 0.])
    assert result["separated"]
    np.testing.assert_allclose(result["gap_lower_bound_m"], 1., atol=1e-8)
    assert not separating_plane(cube, cube + [.5, 0., 0.])["separated"]
    assert not separating_plane(cube, cube + [2., 0., 0.])["separated"]
    rotation, unused = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    rotated = separating_plane(cube @ rotation + 42., (cube + [3., 0., 0.]) @ rotation + 42.)
    assert rotated["separated"]
    np.testing.assert_allclose(rotated["gap_lower_bound_m"], 1., atol=1e-8)
    print("PASS: separated, overlapping, touching, rotated and translated hulls")


if __name__ == "__main__":
    main()
