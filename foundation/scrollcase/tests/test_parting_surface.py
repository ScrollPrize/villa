import unittest

import numpy as np

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase import curved_divider_wall, mesh

mm = mesh.mm

LINING_OUTER_RADIUS = 30
WALL = 2
PITCH = 0.5


def box(lo, hi):
    return mm.makeCube(mm.Vector3f(*(h - l for l, h in zip(lo, hi))), mm.Vector3f(*lo))


def parting(scroll):
    return mesh._parting_surface(scroll, 2, LINING_OUTER_RADIUS, WALL, 2.5, 50, PITCH)


class HeightfieldSolidTests(unittest.TestCase):
    def test_closed_with_the_volume_between_its_heights(self):
        xs, zs = np.linspace(-5, 5, 11), np.linspace(0, 20, 21)
        bottom = np.sin(xs)[:, None] + 0 * zs[None, :]
        solid = mesh._heightfield_solid(xs, zs, bottom, bottom + 2)
        self.assertTrue(solid.topology.isClosed())
        self.assertAlmostEqual(solid.volume(), 10 * 20 * 2, delta=1e-3)


class PartingSurfaceTests(unittest.TestCase):
    def test_divider_where_the_scroll_crosses_it(self):
        # A slab across the divider's middle, which it contains in every column.
        xs, zs, height = parting(box((-2, -6, 0), (2, 6, 20)))
        divider = curved_divider_wall.divider_y(xs, LINING_OUTER_RADIUS, WALL)
        self.assertLess(np.abs(height - divider[:, None]).max(), 1e-6)

    def test_passes_through_a_scroll_off_the_divider(self):
        # Wholly above the divider: the divider rises to y = 6 at x = -15.
        scroll = box((-20, 10, 0), (20, 14, 20))
        xs, zs, height = parting(scroll)
        # Away from its ends, where the surface is free of the slope limit.
        over = (np.abs(xs) < 19)[:, None] & ((zs > 8) & (zs < 12))[None, :]
        # Through the lined scroll (y from 8 to 16).
        self.assertGreaterEqual(height[over].min(), 8 - PITCH)
        self.assertLessEqual(height[over].max(), 16 + PITCH)
        # Back on the divider at the region's ends along the axis.
        divider = curved_divider_wall.divider_y(xs, LINING_OUTER_RADIUS, WALL)
        self.assertLess(np.abs(height[:, 0] - divider).max(), 1e-6)
        self.assertLess(np.abs(height[:, -1] - divider).max(), 1e-6)
        # No steeper than 50 degrees towards them.
        steep = np.degrees(np.arctan(np.abs(np.gradient(height, zs, axis=1))))
        self.assertLess(steep[:, :12].max(), 50 + 1)


if __name__ == "__main__":
    unittest.main()
