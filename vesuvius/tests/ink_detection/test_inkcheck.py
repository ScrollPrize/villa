"""Synthetic tests: run with `python -m pytest tests` or `python -m unittest discover tests`."""
import unittest

import numpy as np
from scipy import ndimage as ndi

from vesuvius.ink_detection.eval.inkcheck import CheckConfig, run_check
from vesuvius.ink_detection.eval.inkcheck.register import masked_ncc_map, pearson, resample_to_grid, shift_image


def strokes(shape, n, rng, width=6):
    """Random 'letters': short thick line segments on a dark background."""
    img = np.zeros(shape, np.float32)
    H, W = shape
    for _ in range(n):
        y, x = rng.uniform(0, H), rng.uniform(0, W)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(40, 160)
        t = np.linspace(-L / 2, L / 2, int(L))
        yy = np.clip((y + t * np.sin(ang)).astype(int), 0, H - 1)
        xx = np.clip((x + t * np.cos(ang)).astype(int), 0, W - 1)
        img[yy, xx] = 1
    img = ndi.binary_dilation(img > 0, iterations=width // 2).astype(np.float32)
    return 30 + 200 * ndi.gaussian_filter(img, 1.0)


def cfg(**kw):
    base = dict(pred_um=8.0, ref_um=4.0, ref_origin=(200.0, 300.0), n_shift_nulls=49, n_metric_nulls=24,
                min_shift_mm=2.0, search_mm=0.3, workers=1, alpha_r=0.05)
    base.update(kw)
    return CheckConfig(**base)


QUIET = dict(log=lambda s: None)


class TestRegister(unittest.TestCase):
    def test_masked_ncc_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(60, 70)).astype(np.float32)
        b = np.roll(a, (3, -2), axis=(0, 1)) + 0.3 * rng.normal(size=a.shape).astype(np.float32)
        ma = np.ones(a.shape, bool)
        ma[:5] = False
        mb = np.ones(a.shape, bool)
        mb[:, -4:] = False
        r, _ = masked_ncc_map(a, ma, b, mb, 5, workers=1)
        for dy, dx in [(0, 0), (3, -2), (-4, 5), (2, 2)]:
            bs, ms = shift_image(b, mb, dy, dx)
            ref = pearson(a, bs, ma & ms)
            self.assertAlmostEqual(float(r[5 + dy, 5 + dx]), ref, places=4)
        iy, ix = np.unravel_index(np.nanargmax(r), r.shape)
        self.assertEqual((iy - 5, ix - 5), (3, -2))

    def test_resample_identity_and_scale(self):
        a = np.arange(64, dtype=np.float32).reshape(8, 8)
        out, ok = resample_to_grid(a, 1.0, 1.0, (8, 8))
        np.testing.assert_allclose(out, a, atol=1e-4)
        self.assertTrue(ok.all())
        out2, _ = resample_to_grid(a, 1.0, 2.0, (4, 4))
        np.testing.assert_allclose(out2[0, 0], a[:2, :2].mean(), atol=1e-4)


class TestVerdicts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.ref = strokes((2400, 2400), 500, rng)  # 9.6 mm at 4 um
        # prediction: 8 um grid, covering ref[200:2200, 300:2300] -> 1000x1000
        crop = cls.ref[200:2200, 300:2300]
        lo = crop.reshape(1000, 2, 1000, 2).mean((1, 3))
        cls.lo = lo
        cls.rng = rng

    def test_true_prediction_passes(self):
        pred = 0.6 * self.lo + 25 * self.rng.normal(size=self.lo.shape) + 10
        res = run_check(pred.astype(np.float32), self.ref, cfg(), **QUIET)
        self.assertEqual(res.verdict, "PASS", res.summary)
        self.assertLess(max(abs(v) for v in res.numbers["shift_um"]), 20)

    def test_misregistered_origin_is_recovered(self):
        pred = 0.6 * self.lo + 25 * self.rng.normal(size=self.lo.shape) + 10
        # tell the tool the wrong origin by 30 ref px = 120 um in y
        res = run_check(pred.astype(np.float32), self.ref, cfg(ref_origin=(170.0, 300.0)), **QUIET)
        self.assertEqual(res.verdict, "PASS", res.summary)
        self.assertAlmostEqual(res.numbers["shift_um"][0], 120, delta=20)

    def test_noise_fails(self):
        pred = 100 + 30 * self.rng.normal(size=self.lo.shape)
        res = run_check(pred.astype(np.float32), self.ref, cfg(), **QUIET)
        self.assertEqual(res.verdict, "FAIL", res.summary)

    def test_trend_only_is_not_pass(self):
        """A prediction sharing only the slow brightness trend of the reference
        has a high raw r but must not PASS (this is the trap the tool exists for)."""
        yy, xx = np.mgrid[0:2400, 0:2400].astype(np.float32)
        trend = 60 * np.sin(yy / 500) * np.cos(xx / 700)
        ref = self.ref + trend
        lo_trend = trend[200:2200, 300:2300].reshape(1000, 2, 1000, 2).mean((1, 3))
        pred = lo_trend + 5 * self.rng.normal(size=lo_trend.shape) + 100
        res = run_check(pred.astype(np.float32), ref, cfg(), **QUIET)
        self.assertGreater(res.numbers["r_raw"], 0.3)
        self.assertNotEqual(res.verdict, "PASS", res.summary)


if __name__ == "__main__":
    unittest.main()
