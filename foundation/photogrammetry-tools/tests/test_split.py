"""
Synthetic checks for the split-orientation tools (split_align.py, marker_scale.py).

Run with:  python -m unittest discover -s tests   (from foundation/photogrammetry-tools)
"""

import argparse
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import marker_scale  # noqa: E402
import split_align  # noqa: E402


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return split_align.rodrigues(axis, angle)


def fragment():
    """Vertices and normals of a closed, bent, lopsided slab about 60 x 40 x 14 mm (no mirror symmetry)."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=6)
    V = m.vertices * [30, 20, 7]
    V[:, 1] *= 1 + 0.01 * V[:, 0]                      # wider at +x
    V[:, 2] += 0.004 * V[:, 0] ** 2 + 0.002 * V[:, 0] * V[:, 1]  # bent and twisted
    m = trimesh.Trimesh(V, m.faces, process=False)
    return np.asarray(m.vertices), np.asarray(m.vertex_normals)


def camera(C, target, f=3000.0, k=(0.0, 0.0, 0.0)):
    """OpenMVG-style camera looking from C at target: (intrinsic dict, world-to-camera rotation, centre)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    K = {"width": 4000, "height": 3000, "focal_length": f, "principal_point": [2000.0, 1500.0], "disto_k3": list(k)}
    return K, R, np.asarray(C, float)


class AlignTest(unittest.TestCase):
    def test_recovers_flipped_half_from_the_rim(self):
        P, N = fragment()
        nz = N[:, 2]
        A, NA = P[nz > -0.3], N[nz > -0.3]   # top face and rim
        B, NB = P[nz < 0.3], N[nz < 0.3]     # bottom face and rim, seen upside down
        R = rotation([0.3, 1, 0.1], np.pi * 0.95)
        t = np.array([12.0, -40.0, 300.0])
        Bs, NBs = B @ R.T + t, NB @ R.T      # B in its own frame
        best = None
        for name, T0 in split_align.hypotheses(A, Bs):
            T, st = split_align.icp(Bs, NBs, A, NA, T0)
            if st is not None and (best is None or st["pairs"] / max(st["rms"], 1e-3) > best[0]):
                best = (st["pairs"] / max(st["rms"], 1e-3), T, st)
        _, T, st = best
        err = np.abs(split_align.apply(T, Bs) - B).max()
        self.assertLess(err, 0.01)           # mm, over a 60 mm object
        self.assertLess(st["rms"], 1e-3)

    def test_rim_scale_measures_relative_size(self):
        P, N = fragment()
        nz = N[:, 2]
        A, NA = P[nz > -0.3], N[nz > -0.3]
        B, NB = P[nz < 0.3], N[nz < 0.3]
        c = B.mean(0)
        k = split_align.rim_scale(c + 1.01 * (B - c), NB, A, NA, np.eye(4))
        self.assertAlmostEqual(k, 1 / 1.01, delta=5e-4)


class MergeTest(unittest.TestCase):
    def scene(self, cams, X, names, scale):
        """OpenMVG JSON scene (one intrinsic per view) with the cameras and landmarks shrunk by 1/scale."""
        views, intr, ext = [], [], []
        for k, ((K, R, C), fn) in enumerate(zip(cams, names)):
            views.append({"key": k, "value": {"polymorphic_id": 1073741824, "ptr_wrapper": {"id": 2147483649 + k, "data": {
                "local_path": "", "filename": fn, "width": 4000, "height": 3000,
                "id_view": k, "id_intrinsic": k, "id_pose": k}}}})
            data = {"width": 4000, "height": 3000, "focal_length": K["focal_length"],
                    "principal_point": K["principal_point"], "disto_k3": K["disto_k3"]}
            val = {"polymorphic_id": 2147483649 if k == 0 else 1, "ptr_wrapper": {"id": 2147483700 + k, "data": data}}
            if k == 0:
                val["polymorphic_name"] = "pinhole_radial_k3"
            intr.append({"key": k, "value": val})
            ext.append({"key": k, "value": {"rotation": R.tolist(), "center": (C / scale).tolist()}})
        struct = []
        for j, x in enumerate(X):
            obs = [{"key": k, "value": {"id_feat": j, "x": split_align.project(x[None], cam)[0][0].tolist()}}
                   for k, cam in enumerate(cams)]
            struct.append({"key": j, "value": {"X": (x / scale).tolist(), "observations": obs}})
        return {"sfm_data_version": "0.3", "root_path": "/workspace/photos", "views": views, "intrinsics": intr,
                "extrinsics": ext, "structure": struct, "control_points": []}

    def test_merged_scene_keeps_both_halves_consistent(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 10, (20, 3))
        camsA = [camera(np.array([300.0 * np.cos(a), 300.0 * np.sin(a), 100.0]), [0, 0, 0], k=(-0.05, 0.01, 0))
                 for a in (0.0, 0.4)]
        R = rotation([1, 0.2, 0], np.pi)    # B's frame: object flipped and moved
        t = np.array([5.0, 7.0, -20.0])
        to_b = lambda Y: (Y - t) @ R        # inverse of x_A = R x_B + t
        XB = to_b(X)
        camsB = []
        for a in (2.0, 2.5):
            K, Rc, C = camera(np.array([300.0 * np.cos(a), 300.0 * np.sin(a), -100.0]), [0, 0, 0])
            camsB.append((K, Rc @ R, to_b(C[None])[0]))  # same physical camera, expressed in B's frame
        with tempfile.TemporaryDirectory() as d:
            sa, sb = 2.0, 4.0                # marker scale factors of each scene
            sceneA = self.scene(camsA, X, ["a0.jpg", "a1.jpg"], sa)
            # A also lists, unposed, a photograph that B poses (as a capture folder's export does): left out
            sceneA["views"].append({"key": 2, "value": {"polymorphic_id": 1073741824, "ptr_wrapper": {
                "id": 2147483651, "data": {"local_path": "", "filename": "b0.jpg", "width": 4000, "height": 3000,
                                           "id_view": 2, "id_intrinsic": 0, "id_pose": 99}}}})
            json.dump(sceneA, open(os.path.join(d, "a.json"), "w"))
            json.dump(self.scene(camsB, XB, ["b0.jpg", "b1.jpg"], sb), open(os.path.join(d, "b.json"), "w"))
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            split_align.save_T(os.path.join(d, "alignment.json"), T,
                               {"scale_a": sa, "scale_b": sb, "axes": np.eye(3).tolist()})
            out = os.path.join(d, "merged.json")
            split_align.merge(argparse.Namespace(out_dir=d, scene_a=os.path.join(d, "a.json"),
                                                 scene_b=os.path.join(d, "b.json"), out_scene=out, thin_shift=0.0))
            m, intr, ext, views = split_align.load_views(out)
            with open(os.path.join(d, "merge.json")) as f:
                offset = json.load(f)["view_offset_b"]
        self.assertEqual(len(views), 4)
        self.assertEqual(len({v["id_pose"] for v in views.values()}), 4)
        self.assertEqual(sum("polymorphic_name" in i["value"] for i in m["intrinsics"]), 1)
        ids = [v["value"]["ptr_wrapper"]["id"] for v in m["views"]] + [i["value"]["ptr_wrapper"]["id"] for i in m["intrinsics"]]
        self.assertEqual(ids, list(range(2147483649, 2147483649 + 8)))
        err = split_align.reprojection_by_half(m, intr, ext, views, offset)
        self.assertLess(err["A"].max(), 1e-6)
        self.assertLess(err["B"].max(), 1e-6)
        # B's landmarks land on A's, in mm
        XA = np.array([s["value"]["X"] for s in m["structure"][:20]])
        XBm = np.array([s["value"]["X"] for s in m["structure"][20:]])
        np.testing.assert_allclose(XA, X, atol=1e-9)
        np.testing.assert_allclose(XBm, X, atol=1e-9)


class FaceOffsetTest(unittest.TestCase):
    def test_signed_offsets_per_face(self):
        g = np.linspace(-10, 10, 41)
        x, y = np.meshgrid(g, g)
        top = np.c_[x.ravel(), y.ravel(), np.full(x.size, 1.0)]
        V = np.r_[top, top * [1, 1, -1]]
        VN = np.r_[np.tile([0, 0, 1.0], (x.size, 1)), np.tile([0, 0, -1.0], (x.size, 1))]
        P = np.r_[top + [0, 0, 0.1], top * [1, 1, -1] - [0, 0, 0.1]]  # both faces 0.1 mm outside: thicker
        off = split_align.face_offsets(P, VN, V, VN, np.array([0, 0, 1.0]))
        self.assertAlmostEqual(np.median(off["+thin face"]), 0.1, places=9)
        self.assertAlmostEqual(np.median(off["-thin face"]), 0.1, places=9)
        self.assertEqual(len(off["rim"]), 0)


class MarkerScaleTest(unittest.TestCase):
    def test_triangulation_rejects_an_outlier_and_handles_distortion(self):
        X = np.array([10.0, -5.0, 2.0])
        cams, obs = {}, []
        for k, a in enumerate(np.linspace(0, 1.2, 8)):
            K, R, C = camera(np.array([300 * np.cos(a), 300 * np.sin(a), 150.0]), [0, 0, 0], k=(-0.08, 0.02, 0.0))
            fn = f"v{k}.jpg"
            cams[fn] = (K, R, C)
            px = marker_scale.project(X, cams[fn])[0]
            obs.append((fn, px + (40.0 if k == 3 else 0.0)))  # one misdetection
        x, ninl, rms = marker_scale.triangulate(obs, cams, 1.5, np.random.default_rng(0))
        self.assertEqual(ninl, 7)
        np.testing.assert_allclose(x, X, atol=1e-6)
        self.assertLess(rms, 1e-6)

    def test_distortion_refit_recovers_scale_despite_a_wrong_scene_model(self):
        # 12 markers of side 31.5 on a ring of radius 100 mm, seen by 24 cameras through k1 -0.12, k2 +0.06; the
        # scene's own model is far off at the image edges, as SfM leaves it on small objects
        true_k, wild_k = [-0.12, 0.06, 0.0], [0.3, -20.0, 300.0]
        corners = {}
        for m in range(12):
            a = 2 * np.pi * m / 12
            c, r, t = 100 * np.array([np.cos(a), np.sin(a), 0]), np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
            for k, (su, sv) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                corners[m, k] = c + 15.75 * (su * r + sv * t)
        cams, obs = {}, {}
        for v, a in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False)):
            K, R, C = camera(np.array([320 * np.cos(a), 320 * np.sin(a), 160.0]), [0, 0, 0], f=2600.0, k=true_k)
            fn = f"v{v}.jpg"
            for key, X in corners.items():
                px, z = marker_scale.project(X, (K, R, C))
                if z > 0 and 0 <= px[0] < 4000 and 0 <= px[1] < 3000:
                    obs.setdefault(key, []).append((fn, px))
            cams[fn] = (dict(K, disto_k3=wild_k), R, C)
        wrong = marker_scale.measure(obs, cams, 1.5, 31.5)
        # the scene's model: only the sightings near the image centre agree, and the scale is off
        self.assertLessEqual(wrong["inliers_median"], 0.6 * wrong["sightings_median"])
        self.assertGreater(abs(wrong.get("scale", 0) - 1.0), 2e-3)
        k = marker_scale.fit_distortion(obs, cams)
        np.testing.assert_allclose(k, true_k[:2], atol=2e-3)
        fixed = {fn: (dict(K, disto_k3=[k[0], k[1], 0.0]), R, C) for fn, (K, R, C) in cams.items()}
        res = marker_scale.measure(obs, fixed, 1.5, 31.5)
        self.assertEqual(res["markers"], 12)
        self.assertEqual(res["inliers_median"], res["sightings_median"])
        self.assertAlmostEqual(res["scale"], 1.0, delta=1e-3)
        self.assertAlmostEqual(res["ring"], 100.0, delta=0.1)

    def test_choose_takes_the_median_of_blocks_with_a_consistent_ring(self):
        blocks = [{"list": "low", "markers": 12, "scale": 1.000, "ring": 100.1},
                  {"list": "mid", "markers": 12, "scale": 1.004, "ring": 100.3},
                  {"list": "high", "markers": 12, "scale": 1.060, "ring": 104.0},   # outlying ring: dropped
                  {"list": "few", "markers": 4, "scale": 0.900, "ring": 100.2}]     # too few markers
        pick = marker_scale.choose(blocks)
        self.assertEqual(pick["kept"], ["low", "mid"])
        self.assertAlmostEqual(pick["factor"], 1.002)
        self.assertIsNone(marker_scale.choose(blocks[3:]))

    def test_undistort_inverts_distort(self):
        xn = np.array([[0.1, -0.2], [0.3, 0.25]])
        k = [-0.08, 0.02, 0.001]
        np.testing.assert_allclose(marker_scale.undistort(marker_scale.distort(xn, k), k), xn, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
