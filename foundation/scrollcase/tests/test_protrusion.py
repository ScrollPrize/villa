import unittest

import build123d as bd

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase import curved_divider_wall, mesh

mm = mesh.mm

LINING_OUTER_RADIUS = 30
WALL = 2
VOXEL = 0.4


def box(lo, hi):
    return mm.makeCube(mm.Vector3f(*(h - l for l, h in zip(lo, hi))), mm.Vector3f(*lo))


def divider():
    solid = (
        curved_divider_wall.build_divider_solid(LINING_OUTER_RADIUS, 112.5 / 2, WALL)
        .part.move(bd.Location((0, 0, -100)))
        .solid()
    )
    return mesh.brep_to_mesh(solid)


def protrusion(offset_mesh, below):
    return mesh._protrusion(
        offset_mesh,
        divider(),
        below=below,
        lining_outer_radius=LINING_OUTER_RADIUS,
        wall_thickness_mm=WALL,
        min_gap_mm=1,
        clearance_mm=1,
        voxel_size=VOXEL,
    )


class ProtrusionTests(unittest.TestCase):
    def test_scroll_wholly_above_divider_gets_mesa_from_below(self):
        # The divider rises to y = 6 at x = -15 and dips to y = -6 at x = 15.
        scroll = box((-20, 10, 0), (20, 14, 20))
        mesa, cut = protrusion(scroll, below=True)
        self.assertIsNotNone(mesa)

        b = mesa.computeBoundingBox()
        # Reaches from the negative half's divider wall up to the scroll's underside...
        self.assertLess(b.min.y, -6 - WALL + 0.5)
        self.assertLess(b.max.y, 10 + VOXEL)
        self.assertGreater(b.max.y, 10 - 2 * VOXEL)
        # ...kept the sliding clearance inside the scroll's footprint.
        self.assertGreaterEqual(b.min.x, -20 + 1)
        self.assertLessEqual(b.max.x, 20 - 1)
        self.assertGreaterEqual(b.min.z, 0 + 1)
        self.assertLessEqual(b.max.z, 20 - 1)
        overlap = mm.voxelBooleanIntersect(mesa, scroll, VOXEL)
        self.assertLess(overlap.volume(), VOXEL**3)

        # The positive half is cleared from under the scroll, wider than the mesa.
        c = cut.computeBoundingBox()
        self.assertLess(c.min.x, b.min.x - 1)
        self.assertGreater(c.max.x, b.max.x + 1)

        # Nothing descends from above: the scroll's top is on the positive side.
        self.assertEqual(protrusion(scroll, below=False), (None, None))

    def test_scroll_crossing_divider_is_unchanged(self):
        scroll = box((-4, -3, 0), (4, 3, 20))
        self.assertEqual(protrusion(scroll, below=True), (None, None))
        self.assertEqual(protrusion(scroll, below=False), (None, None))


if __name__ == "__main__":
    unittest.main()
