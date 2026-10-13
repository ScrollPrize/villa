import unittest

import numpy as np

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase import mesh

mn = mesh.mn


def flat_box(angle_rad, tilt=0.0):
    """Box 60 x 20 x 40 (wide along its own x), turned by angle_rad about Z.

    tilt shears the box along Z so its 3D principal axes leave the XY plane.
    """
    corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    rng = np.random.default_rng(0)
    points = np.r_[corners, rng.uniform(-1, 1, (2002, 3))] * [30, 10, 20]
    points[:, 0] += tilt * points[:, 2]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    points = points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    return points


class RotateAbout2ndPrincipalTests(unittest.TestCase):
    def wide_angle_after(self, angle_rad, rotation_rad=0.0, tilt=0.0):
        points = flat_box(angle_rad, tilt)
        # Only the vertices matter, but meshlib drops vertices no face uses.
        faces = np.arange(len(points)).reshape(-1, 3)
        m = mn.meshFromFacesVerts(faces, points)
        rotated = mn.getNumpyVerts(mesh.rotate_about_2nd_principal(m, rotation_rad))
        np.testing.assert_allclose(rotated[:, 2], points[:, 2], atol=1e-4)
        w, v = np.linalg.eigh(np.cov(rotated[:, :2], rowvar=False))
        wide = v[:, np.argmax(w)]
        return np.degrees(np.arctan2(wide[1], wide[0])) % 180

    def assertAngle(self, actual, expected):
        diff = (actual - expected + 90) % 180 - 90
        self.assertLess(abs(diff), 0.1, f"{actual} vs {expected}")

    def test_wide_direction_goes_to_x_for_either_sign(self):
        for deg in (-70, -25, 0, 25, 70, 135):
            self.assertAngle(self.wide_angle_after(np.radians(deg)), 0)

    def test_tilted_principal_axes(self):
        self.assertAngle(self.wide_angle_after(np.radians(-40), tilt=1.2), 0)

    def test_rotation_offset(self):
        self.assertAngle(self.wide_angle_after(np.radians(-25), np.pi / 2), 90)


if __name__ == "__main__":
    unittest.main()
