"""
Synthetic checks for the photogrammetry review tools.

Run with:  python -m unittest discover -s tests   (from foundation/photogrammetry-tools)
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mesh_qa  # noqa: E402
import scale_review  # noqa: E402
import sfm_review  # noqa: E402

CUBE_V = np.array([[x, y, z] for x in (0, 2) for y in (0, 3) for z in (0, 5)], float)
CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def write_ply(path, V, F, texcoords=False):
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(V)}",
            "property float32 x", "property float32 y", "property float32 z",
            f"element face {len(F)}", "property list uint8 uint32 vertex_indices"]
    if texcoords:
        head.append("property list uint8 float32 texcoord")
    dt = [("n", "u1"), ("i", "<u4", (3,))] + ([("m", "u1"), ("t", "<f4", (6,))] if texcoords else [])
    faces = np.zeros(len(F), dt)
    faces["n"], faces["i"] = 3, F
    if texcoords:
        faces["m"], faces["t"] = 6, np.linspace(0, 1, 6 * len(F)).reshape(-1, 6)
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["end_header"]) + "\n").encode())
        f.write(V.astype("<f4").tobytes())
        f.write(faces.tobytes())


class MeshQATest(unittest.TestCase):
    def test_closed_box(self):
        with tempfile.TemporaryDirectory() as d:
            write_ply(os.path.join(d, "box.ply"), CUBE_V, CUBE_F)
            V, F, T, _ = mesh_qa.read_ply(os.path.join(d, "box.ply"))
        r = mesh_qa.mesh_report(V, F)
        self.assertIsNone(T)
        self.assertEqual((r["boundary_edges"], r["nonmanifold_edges"], r["components"]), (0, 0, 1))
        self.assertAlmostEqual(abs(r["volume"]), 30.0, places=5)
        np.testing.assert_allclose(sorted(r["extents"]), [2, 3, 5], atol=1e-6)

    def test_open_and_split_mesh_with_texcoords(self):
        # drop one face (two boundary loops' worth of edges: 3) and add a separate box
        V = np.r_[CUBE_V, CUBE_V + 10]
        F = np.r_[CUBE_F[1:], CUBE_F + 8]
        with tempfile.TemporaryDirectory() as d:
            write_ply(os.path.join(d, "open.ply"), V, F, texcoords=True)
            V2, F2, T, _ = mesh_qa.read_ply(os.path.join(d, "open.ply"))
        np.testing.assert_array_equal(F2, F)
        self.assertEqual(T.shape, (len(F), 3, 2))
        r = mesh_qa.mesh_report(V2, F2)
        self.assertEqual((r["boundary_edges"], r["components"]), (3, 2))


class ScaleReviewTest(unittest.TestCase):
    def test_marker_ring(self):
        # 12 squares of side 31.5 on a ring of radius 99.5 in a tilted plane, corners consecutive per marker
        corners = []
        for k in range(12):
            a = 2 * np.pi * k / 12
            centre, t = 99.5 * np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            r = np.array([np.cos(a), np.sin(a)])
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append(centre + 15.75 * (su * r + sv * t))
        P = np.c_[np.array(corners), np.zeros(len(corners))]
        tilt = np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
        g = scale_review.corner_geometry(P @ tilt.T + [5, -3, 200], complete=True)
        np.testing.assert_allclose(g["sides"], 31.5, atol=1e-9)
        self.assertLess(g["off_max"], 1e-9)
        self.assertAlmostEqual(g["ring"], 99.5, places=6)


class SfmReviewTest(unittest.TestCase):
    def test_circle_fit(self):
        a = np.radians(np.arange(0, 360, 7.5))
        P = np.c_[2 * np.cos(a), 2 * np.sin(a), np.full(len(a), 0.5)]
        f = sfm_review.fit_circle(P)
        self.assertAlmostEqual(f["r"], 2.0, places=9)
        self.assertLess(f["radial"], 1e-9)
        self.assertAlmostEqual(f["max_gap"], 7.5, places=6)
        self.assertAlmostEqual(abs(f["normal"][2]), 1.0, places=9)

    def test_orientation_from_block_name(self):
        self.assertEqual(sfm_review.orientation("orientation2_camera_midlow"), "orientation2")


if __name__ == "__main__":
    unittest.main()
