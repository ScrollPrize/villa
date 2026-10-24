import unittest

import numpy as np

from scrollcase import alignment


class FitCylinderAxisTests(unittest.TestCase):
    def setUp(self):
        # Flattened box: 80 x 40 x 60 mm. The smallest enclosing cylinder runs along x.
        rng = np.random.default_rng(0)
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        self.points = np.r_[corners, rng.uniform(-1, 1, (5000, 3))] * [40, 20, 30]

    def test_given_axis_is_used(self):
        rotation, translation, radius, height = alignment.fit_cylinder(
            self.points, axis=np.array([0, 0, 2.0])
        )
        np.testing.assert_allclose(rotation[2], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(height, np.ptp(self.points[:, 2]), places=6)
        # Smallest circle about z encloses the 80 x 40 cross-section.
        self.assertAlmostEqual(radius, np.hypot(40, 20), places=3)
        transformed = self.points @ rotation.T + translation
        self.assertAlmostEqual(transformed[:, 2].min(), 0, places=6)
        self.assertLessEqual(np.hypot(*transformed[:, :2].T).max(), radius + 1e-6)

    def test_default_searches_for_smallest_cylinder(self):
        rotation, _, radius, _ = alignment.fit_cylinder(self.points, tol=1e-6)
        self.assertGreater(abs(rotation[2][0]), 0.99)
        self.assertLess(radius, np.hypot(20, 30) + 0.5)


if __name__ == "__main__":
    unittest.main()
