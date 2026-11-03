import unittest

import numpy as np

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase import case, mesh
from build123d import Location, Vector

from test_mesh_rotation import flat_box

mn = mesh.mn


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0


class NarrowestInYTests(unittest.TestCase):
    def test_flat_box_turns_long_side_along_x(self):
        for deg in (-70, -25, 0, 25, 70, 135):
            points = flat_box(np.radians(deg))
            m = mn.meshFromFacesVerts(np.arange(len(points)).reshape(-1, 3), points)
            angle, aspect = mesh.narrowest_in_y(m)
            self.assertGreater(angle, -np.pi / 2)
            self.assertLessEqual(angle, np.pi / 2)
            self.assertAlmostEqual(aspect, 3, places=3)
            c, s = np.cos(angle), np.sin(angle)
            self.assertAlmostEqual(np.ptp(s * points[:, 0] + c * points[:, 1]), 20, places=3)
            self.assertAlmostEqual(np.ptp(c * points[:, 0] - s * points[:, 1]), 60, places=3)


class TurnedSplitTests(unittest.TestCase):
    def test_turned_halves_part_along_the_divider_normal(self):
        for rotation, cap_side in ((25, 0), (-70, -90)):
            with self.subTest(rotation=rotation):
                config = case.ScrollCaseConfig(scroll_height_mm=30, scroll_radius_mm=20, split_rotation_deg=rotation)
                self.assertEqual(config.cap_split_deg, cap_side)
                left, right = case.build_case(config)
                self.assertLess(volume(left & right), 1e-3)
                normal = Vector(-np.sin(np.radians(rotation)), np.cos(np.radians(rotation)), 0)
                for distance in (0.5, 5, 40):
                    self.assertLess(volume(left & right.moved(Location(normal * distance))), 1e-3)
                self.assertGreater(volume(left & right.moved(Location(normal * -0.5))), 1)
                # The caps stay square, their bolt protrusions on the sides nearest the divider.
                size = (left + right).bounding_box().size
                square, tabs = 2 * config.square_loft_radius, 2 * config.square_loft_radius + 2 * config.square_height_mm
                across, along = (size.X, size.Y) if cap_side else (size.Y, size.X)
                self.assertAlmostEqual(across, square, places=3)
                self.assertAlmostEqual(along, tabs, places=3)

    def test_split_too_far_from_the_caps_sides_is_refused(self):
        config = case.ScrollCaseConfig(scroll_height_mm=30, scroll_radius_mm=45, split_rotation_deg=45)
        self.assertFalse(config.split_rotation_fits)
        with self.assertRaises(ValueError):
            case.build_case(config)
        self.assertTrue(case.ScrollCaseConfig(scroll_height_mm=30, scroll_radius_mm=20, split_rotation_deg=45).split_rotation_fits)


if __name__ == "__main__":
    unittest.main()
