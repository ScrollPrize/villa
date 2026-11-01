"""relative_windings_box on a synthetic spiral: a binary surface prediction, a grad_mag that doubles from x = 150
on, and a Lasagna manifest naming it. The command must keep every exported winding on the true wraps, cut only
where the grad_mag is doubled, write point collections the spiral fit loads, and refuse before any work without
a prediction, a grad_mag source or an umbilicus."""

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import relative_windings_box as rwb  # noqa: E402
from relative_windings_box import _chains, _split, _zarr  # noqa: E402

CENTRE, R0, PITCH, PLANT_X = (120.0, 120.0), 20.0, 14.0, 150.0


def spiral_volume(shape=(48, 240, 240), half=1.6):
    """Binary prediction (z, y, x) of an Archimedean spiral sheet r = r0 + pitch * (theta / 2 pi + n)."""
    Z, Y, X = shape
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    dx, dy = xx - CENTRE[0], yy - CENTRE[1]
    r = np.hypot(dx, dy)
    th = np.mod(np.arctan2(dy, dx), 2 * math.pi)
    u = (r - R0) / PITCH - th / (2 * math.pi)
    n = np.rint(u)
    sl = (np.abs(u - n) * PITCH <= half) & (r > R0) & (n >= 0)
    vol = np.zeros(shape, np.uint8)
    vol[:] = (sl * 255).astype(np.uint8)[None]
    return vol


def true_wrap(p):
    dx, dy = p[0] - CENTRE[0], p[1] - CENTRE[1]
    th = math.atan2(dy, dx) % (2 * math.pi)
    return round((math.hypot(dx, dy) - R0) / PITCH - th / (2 * math.pi))


def _blocks(vol, chunk):
    for i in range(-(-vol.shape[0] // chunk[0])):
        for j in range(-(-vol.shape[1] // chunk[1])):
            for m in range(-(-vol.shape[2] // chunk[2])):
                blk = np.zeros(chunk, vol.dtype)
                sub = vol[i * chunk[0]:(i + 1) * chunk[0], j * chunk[1]:(j + 1) * chunk[1],
                          m * chunk[2]:(m + 1) * chunk[2]]
                blk[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                if blk.any():
                    yield (i, j, m), blk


def write_zarr(root, vol, chunk):
    """A one-level OME-Zarr v2 group, uncompressed; all-zero chunks left out."""
    d = Path(root) / "0"
    d.mkdir(parents=True)
    (d / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": list(vol.shape), "chunks": list(chunk), "dtype": vol.dtype.str,
        "compressor": None, "fill_value": 0, "filters": None, "order": "C", "dimension_separator": "/"}))
    for (i, j, m), blk in _blocks(vol, chunk):
        q = d / str(i) / str(j)
        q.mkdir(parents=True, exist_ok=True)
        (q / str(m)).write_bytes(blk.tobytes())
    (Path(root) / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (Path(root) / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "axes": [
        {"name": a} for a in "zyx"], "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]}))


def write_zarr_v3(root, vol, chunk):
    """The same volume as a bare zarr v3 array, gzip compressed."""
    import gzip
    root = Path(root)
    root.mkdir(parents=True)
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": list(vol.shape), "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                   {"name": "gzip", "configuration": {"level": 1}}]}))
    for (i, j, m), blk in _blocks(vol, chunk):
        q = root / "c" / str(i) / str(j)
        q.mkdir(parents=True, exist_ok=True)
        (q / str(m)).write_bytes(gzip.compress(blk.tobytes()))


def write_inputs(d):
    write_zarr(d / "pred.zarr", spiral_volume(), (16, 64, 64))
    g = np.full((24, 120, 120), 70, np.uint8)                 # level 1 of a 48 x 240 x 240 base
    g[:, :, ((np.arange(120) + 0.5) * 2 - 0.5) >= PLANT_X] = 140
    write_zarr(d / "las" / "gm.zarr", g, (16, 32, 32))
    (d / "las" / "run.lasagna.json").write_text(json.dumps({
        "groups": {"grad_mag": {"zarr": "gm.zarr", "scaledown": 1, "channels": ["grad_mag"]},
                   "normals": {"zarr": "gm.zarr", "scaledown": 1, "channels": ["nx", "ny"]}},
        "base_shape_zyx": [48, 240, 240], "grad_mag_factor": 1.0, "grad_mag_encode_scale": 1000.0}))
    (d / "umbilicus.json").write_text(json.dumps([[0.0, 120.0, 120.0], [47.0, 120.0, 120.0]]))


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = rwb.main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestRelativeWindingsBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.d = d = Path(cls._tmp.name)
        write_inputs(d)
        cls.res = {}
        for rule in ("v1", "v2", "none"):
            rc, out, err = run(["--prediction", str(d / "pred.zarr"), "--grad-mag", str(d / "las/run.lasagna.json"),
                                "--point", "120,120,24", "--box", "140", "--seeds", "30",
                                "--umbilicus", str(d / "umbilicus.json"), "--split-rule", rule,
                                "--out", str(d / rule), "--json"])
            cls.res[rule] = (rc, out, err)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_box_writes_point_collections_on_the_true_wraps(self):
        d = self.d
        rc, out, err = self.res["v2"]
        self.assertEqual(rc, 0, err[-1500:])
        res = json.loads(out)                                 # only JSON on stdout
        self.assertTrue(res["ok"])
        self.assertGreater(res["collections"], 5)
        self.assertGreater(res["split_steps"], 0)
        ch = json.loads((d / "v2/chains.json").read_text())
        self.assertTrue(ch["rule_digest"].startswith("0e53e0ae5016"))
        self.assertEqual(ch["axis_source"], "file")
        self.assertEqual(ch["box"], [[50.0, 190.0], [50.0, 190.0], [-46.0, 94.0]])
        ex = json.loads((d / "v2/export.json").read_text())
        self.assertEqual(ex["split_rule"]["params_digest"], "cd273aed7673")
        # every exported winding difference is the true one
        pw = json.loads((d / "v2/relative_windings.json").read_text())
        self.assertEqual(pw["vc_pointcollections_json_version"], "1")
        pairs = right = 0
        for c in pw["collections"].values():
            self.assertFalse(c["metadata"]["winding_is_absolute"])
            pts = list(c["points"].values())
            for a_, b_ in zip(pts, pts[1:]):
                pairs += 1
                right += (b_["wind_a"] - a_["wind_a"]) == (true_wrap(b_["p"]) - true_wrap(a_["p"]))
        self.assertGreater(pairs, 20)
        self.assertGreaterEqual(right / pairs, 0.95)
        self.assertIn("chains", json.loads((d / "v2/split.json").read_text()))
        # every cut step reaches the doubled grad_mag
        fl = json.loads((d / "v2/flags.json").read_text())
        split_pts = [p["p"] for c in fl["collections"].values() if "split" in c["name"]
                     for p in c["points"].values()]
        self.assertTrue(split_pts)
        self.assertGreater(min(p[0] for p in split_pts), PLANT_X - 2 * PITCH)

    def test_spiral_fit_loader_reads_it(self):
        try:
            import point_collection
        except ImportError as exc:  # torch and the rest of the spiral-fitting environment
            self.skipTest(f"point_collection does not import here ({exc})")
        res = json.loads(self.res["v2"][1])
        with contextlib.redirect_stdout(io.StringIO()):
            cols = point_collection.load_point_collection(str(self.d / "v2/relative_windings.json"))
        self.assertEqual(len(cols), res["collections"])
        self.assertEqual(sum(len(c["points"]) for c in cols.values()), res["points"])
        for c in cols.values():
            self.assertIs(c["metadata"]["winding_is_absolute"], False)
            self.assertTrue(all(math.isfinite(p["winding_annotation"]) for p in c["points"].values()))

    def test_split_rule_v2_cuts_at_one(self):
        d = self.d
        for rule in ("v1", "v2"):
            self.assertEqual(self.res[rule][0], 0, self.res[rule][2][-1500:])
        # the same chains; v2 cuts a subset of v1's steps
        self.assertEqual(json.loads((d / "v1/chains.json").read_text())["chains"],
                         json.loads((d / "v2/chains.json").read_text())["chains"])
        self.assertLessEqual(json.loads(self.res["v2"][1])["split_steps"],
                             json.loads(self.res["v1"][1])["split_steps"])
        self.assertEqual({k for k in _split.PARAMS if _split.PARAMS[k] != _split.PARAMS_V2[k]},
                         {"version", "abs_own"})

    def test_split_rule_none_keeps_every_step(self):
        d = self.d
        rc, out, err = self.res["none"]
        self.assertEqual(rc, 0, err[-1500:])
        res = json.loads(out)
        self.assertTrue(res["ok"])
        self.assertEqual(res["split_steps"], 0)
        sd = json.loads((d / "none/split.json").read_text())
        sr = sd["split_rule"]
        self.assertIsNone(sr["params"]["abs_own"])
        self.assertIsNone(sr["params"]["q_own"])
        self.assertEqual(sr["pieces"], sr["chains_in"])
        # the steps kept with the cut off are still marked, where the v2 cut would act
        cols = lambda doc, tag: sorted(p["p"] for c in json.loads((d / doc).read_text())["collections"].values()
                                       if tag in c["name"] for p in c["points"].values())  # noqa: E731
        self.assertEqual(cols("none/flags.json", "check-split-step"), cols("v2/flags.json", "split-step"))
        self.assertTrue(cols("none/flags.json", "check-split-step"))
        # the gm_x stats point at where the v2 threshold sits
        gx = res["gm_x"]
        self.assertGreater(gx["n"], 20)
        self.assertGreater(gx["above_v2"]["count"], 0)
        self.assertLess(gx["median"], 1.0)
        # every exported winding difference is still the true one, with the cut's pairs back in
        pw = json.loads((d / "none/relative_windings.json").read_text())

        def claims(name):
            doc = json.loads((d / name).read_text())
            pairs = right = 0
            for c in doc["collections"].values():
                pts_ = list(c["points"].values())
                for a_, b_ in zip(pts_, pts_[1:]):
                    pairs += 1
                    right += (b_["wind_a"] - a_["wind_a"]) == (true_wrap(b_["p"]) - true_wrap(a_["p"]))
            return pairs, right

        none_pairs, none_right = claims("none/relative_windings.json")
        v2_pairs, v2_right = claims("v2/relative_windings.json")
        self.assertGreater(none_pairs, 20)
        self.assertGreaterEqual(none_right / none_pairs, 0.95)
        self.assertGreater(none_pairs, v2_pairs)
        self.assertGreaterEqual(none_right, v2_right)

    def test_rule_digests(self):
        self.assertTrue(_chains.rule_digest(_chains.RULES[1]).startswith("0e53e0ae5016"))
        self.assertEqual(_split.digest(_split.SPLIT_RULES["v1"]), "c6d2eecbebc2")
        self.assertEqual(_split.digest(_split.SPLIT_RULES["v2"]), "cd273aed7673")
        self.assertEqual(_split.digest(_split.SPLIT_RULES["none"]), "6e3aff82495f")

    def test_zarr_v3_reads_like_v2(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            vol = spiral_volume((20, 100, 100))
            write_zarr(t / "v2.zarr", vol, (8, 32, 32))
            write_zarr_v3(t / "v3.zarr", vol, (8, 32, 32))
            pts = np.random.default_rng(1).uniform([0, 0, 0], [99, 99, 19], (2000, 3))
            a = _chains.Pred.from_volume(_zarr.Volume(str(t / "v2.zarr"))).on(pts)
            b = _chains.Pred.from_volume(_zarr.Volume(str(t / "v3.zarr"))).on(pts)
            self.assertTrue(a.any())
            np.testing.assert_array_equal(a, b)
            zyx = np.rint(pts[:, ::-1]).astype(int)
            np.testing.assert_array_equal(a, vol[zyx[:, 0], zyx[:, 1], zyx[:, 2]] > 127)

    def test_refuses_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_inputs(d)
            pred, gm, umb = str(d / "pred.zarr"), str(d / "las/run.lasagna.json"), str(d / "umbilicus.json")
            cases = [(["--grad-mag", gm, "--umbilicus", umb], "no surface prediction"),
                     (["--prediction", pred, "--umbilicus", umb], "no grad_mag source"),
                     (["--prediction", pred, "--grad-mag", gm], "no umbilicus given"),
                     (["--prediction", pred, "--grad-mag", gm, "--umbilicus", str(d / "nope.json")], "no such file"),
                     (["--prediction", str(d / "missing.zarr"), "--grad-mag", gm, "--umbilicus", umb],
                      "cannot open the surface"),
                     (["--prediction", pred, "--grad-mag", str(d / "missing.lasagna.json"), "--umbilicus", umb],
                      "no lasagna manifest")]
            for extra, why in cases:
                with self.subTest(why=why):
                    out = d / f"run-{len(why)}"
                    rc, _, err = run([*extra, "--point", "120,120,24", "--out", str(out)])
                    self.assertEqual(rc, 2)
                    self.assertIn(why, err)
                    self.assertFalse((out / "chains.json").exists())
            for point, box, seeds in (("1,2", "512", "200"), ("1,2,3", "0", "200"), ("1,2,3", "512", "x")):
                rc, _, _ = run(["--prediction", pred, "--grad-mag", gm, "--umbilicus", umb, "--point", point,
                                "--box", box, "--seeds", seeds, "--out", str(d / "bad")])
                self.assertEqual(rc, 2)

    def test_python_m_runs_it(self):
        r = subprocess.run([sys.executable, "-m", "relative_windings_box", "--help"], cwd=SPIRAL_DIR,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--umbilicus", r.stdout)


if __name__ == "__main__":
    unittest.main()
