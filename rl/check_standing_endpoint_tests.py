import numpy as np
import mujoco

from check_standing_endpoint import evaluate, standing_pose
from reference_policy_baseline import ROOT


def main():
    samples = np.zeros((11, 13))
    samples[:, 0] = np.linspace(0., 5., 11)
    samples[:, 1] = .75
    samples[:, 5:7] = 150.
    assert all(evaluate(samples, True, True).values())
    failures = [(1, .4), (2, .16), (3, .04), (4, .06), (5, 0.), (6, 0.),
                (7, 2.), (8, .006), (9, .002), (10, .96), (11, .002), (12, .011)]
    for column, value in failures:
        bad = samples.copy()
        bad[-1, column] = value
        assert not all(evaluate(bad, True, True).values()), (column, value)
    assert not all(evaluate(samples[:1], False, True).values())
    assert not all(evaluate(samples, True, False).values())
    bad = samples.copy()
    bad[0, 9] = .002
    assert not evaluate(bad, True, True)["self_penetration_1mm"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"))
    data = standing_pose(model, .6)
    assert data.qpos[2] > .55 and not data.qvel.any()
    np.testing.assert_array_equal(data.qpos[3:7], [1., 0., 0., 0.])
    for name in ("left_foot", "right_foot"):
        geom = model.geom(name).id
        bottom = data.geom_xpos[geom, 2] - np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom]
        assert abs(bottom) < 1e-8
    print("Standing gate positive/negative cases and geometric initialization passed")


if __name__ == "__main__":
    main()
