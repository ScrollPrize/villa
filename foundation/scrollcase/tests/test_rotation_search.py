import unittest

import numpy as np

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase import mesh

mm = mesh.mm

LINING_OUTER_RADIUS = 30
PITCH = 0.5


def slab(shift=(0, 0, 0)):
    """Flat 40 x 4 x 20 slab, wide along X and centred on the axis: the divider's S
    leaves room either side of it unless it is turned to lie along the S's middle."""
    return mm.makeCube(mm.Vector3f(40, 4, 20), mm.Vector3f(-20 + shift[0], -2 + shift[1], shift[2]))


def fill(m, degrees):
    return mesh.split_fill_and_depth(m, np.radians(degrees), LINING_OUTER_RADIUS, 2, 2, PITCH)[0]


class SplitFillTests(unittest.TestCase):
    def test_fill_does_not_depend_on_where_the_voxels_fall(self):
        for degrees in (0, 20, 45):
            fills = [fill(slab(shift), degrees) for shift in ((0, 0, 0), (0.25, 0.25, 0.25), (0.1, 0.3, 0))]
            self.assertLess(np.ptp(fills), 0.1 * np.mean(fills))

    def test_slab_is_turned_along_the_divider(self):
        angle = mesh.rotation_minimising_fill(slab(), LINING_OUTER_RADIUS, 2, 2, 5, 0.01)
        self.assertNotEqual(angle, 0)
        self.assertLess(fill(slab(), np.degrees(angle)), 0.9 * fill(slab(), 0))

    def test_cylinder_keeps_its_split(self):
        cylinder = mm.makeCylinderAdvanced(20, 20, 0, 2 * np.pi, 40, 128)
        self.assertEqual(mesh.rotation_minimising_fill(cylinder, 24, 2, 2, 5, 0.01), 0)


if __name__ == "__main__":
    unittest.main()
