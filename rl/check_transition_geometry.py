import unittest

import numpy as np

from transition_geometry import nearest_inset_target, polygon_margin


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_margin_sign_and_distance(self):
        self.assertAlmostEqual(polygon_margin(self.square, [.5, .5]), .5)
        self.assertAlmostEqual(polygon_margin(self.square, [-.1, .5]), -.1)
        self.assertAlmostEqual(polygon_margin(self.square, [0, .5]), 0)

    def test_degenerate_patch(self):
        for points in ([], [[0, 0]], [[0, 0], [1, 0], [2, 0]]):
            self.assertIsNone(polygon_margin(points, [0, 0]))
            self.assertFalse(nearest_inset_target(points, [0, 0])["passed"])

    def test_nearest_target(self):
        result = nearest_inset_target(self.square, [-.1, .5], .05)
        self.assertTrue(result["passed"])
        np.testing.assert_allclose(result["target_xy_m"], [.05, .5], atol=1e-9)
        self.assertAlmostEqual(result["shift_norm_m"], .15)

    def test_interior_stays_and_impossible_inset_fails(self):
        result = nearest_inset_target(self.square, [.5, .5])
        self.assertTrue(result["passed"])
        self.assertEqual(result["shift_norm_m"], 0)
        self.assertFalse(nearest_inset_target(self.square, [.5, .5], .6)["passed"])

    def test_coordinate_invariance_and_duplicates(self):
        rotation = np.array([[0, -1], [1, 0]])
        translation = np.array([3, -2])
        points = np.vstack((self.square, self.square)) @ rotation + translation
        position = np.array([-.1, .5]) @ rotation + translation
        result = nearest_inset_target(points, position, .05)
        self.assertTrue(result["passed"])
        np.testing.assert_allclose(result["target_xy_m"], np.array([.05, .5]) @ rotation + translation, atol=1e-9)
        self.assertAlmostEqual(polygon_margin(points, position), -.1)


if __name__ == "__main__":
    unittest.main()
