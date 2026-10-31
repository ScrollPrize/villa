"""Tests for Parting (dinovol_2/eval/parting/) on synthetic data: no scan, no network, no model download."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dinovol_2.eval.parting import classifier as M
from dinovol_2.eval.parting import ct as CT
from dinovol_2.eval.parting import features as FE
from dinovol_2.eval.parting import geometry as G
from dinovol_2.eval.parting.cli import main

try:
    import sklearn  # noqa: F401
    import joblib  # noqa: F401
except ImportError:          # the classifier's dependencies are optional for dinovol itself
    sklearn = None


class AucTests(unittest.TestCase):
    def test_auc_counts_ties_half_and_matches_the_sorting_form(self):
        rng = np.random.default_rng(0)
        same = rng.integers(0, 5, 40).astype(float)
        cross = rng.integers(0, 5, 30).astype(float)
        brute = np.mean([(s > c) + 0.5 * (s == c) for s in same for c in cross])
        self.assertAlmostEqual(M.auc(same, cross), brute, places=12)
        self.assertAlmostEqual(M.fast_auc(same, cross), brute, places=12)

    def test_auc_of_separated_scores(self):
        self.assertEqual(M.auc([2, 3], [0, 1]), 1.0)
        self.assertEqual(M.auc([0, 1], [2, 3]), 0.0)


def synthetic_run(n=6, seed=0, stratum="touching"):
    rng = np.random.default_rng(seed)
    F = {"tok": rng.normal(size=(n, 3, 288)).astype(np.float32),
         "ct_n": rng.normal(size=(n, 3, 17)).astype(np.float32),
         "ct_seg": rng.normal(size=(n, 2, 16)).astype(np.float32),
         "p_tok": rng.uniform(-1, 1, size=(n, 2, 2, 16)).astype(np.float32)}
    rows = [{"id": i, "pair": f"L{i % 2}-L{i % 2 + 1}", "stratum": stratum,
             "scores": {"dino_base": [float(rng.uniform()), float(rng.uniform())],
                        "ct_valley": [float(rng.uniform()), float(rng.uniform())]}} for i in range(n)]
    return rows, F


class RowsTests(unittest.TestCase):
    def test_design_has_the_669_column_layout(self):
        rows, F = synthetic_run()
        X, y, grp, strat, ctrl = M.design(rows, F)
        self.assertEqual(X.shape, (12, M.COLUMNS))
        self.assertEqual(list(y[:4]), [1, 0, 1, 0])
        self.assertEqual(ctrl[1], rows[0]["scores"]["dino_base"][1])

    def test_rows_do_not_depend_on_the_sign_of_a_normal(self):
        rows, F = synthetic_run()
        flipped = dict(F, ct_n=F["ct_n"][..., ::-1].copy())     # every normal flipped: each profile runs backward
        np.testing.assert_allclose(M.design(rows, F)[0], M.design(rows, flipped)[0], atol=1e-6)


class ResampleTests(unittest.TestCase):
    def test_box_weights_sum_to_one(self):
        for ratio in (1.0, 2.0, 7.91 / 3.24):
            i0, w = CT.box_weights(37, 128, ratio)
            np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-9)
            self.assertTrue((w >= 0).all())

    def test_resampled_shape_of_a_3_24_um_array_at_7_91_um(self):
        self.assertEqual(CT.resampled_shape((6656, 1440, 6312), 7.91 / 3.24), [2727, 590, 2586])

    def test_a_constant_volume_stays_constant(self):
        class Flat:
            shape = (1000, 1000, 1000)
            url = "synthetic"

            def read(self, lo, size):
                return np.full(tuple(size), 77, np.uint8)

        rw = CT.ResampledWindow.__new__(CT.ResampledWindow)
        rw.src_um, rw.dst_um, rw.ratio, rw.src, rw.url = 3.24, 7.91, 7.91 / 3.24, Flat(), "synthetic"
        out = rw.read((5, 6, 7), 16)
        self.assertEqual(out.shape, (16, 16, 16))
        self.assertTrue((out == 77).all())

    def test_parse_resample_refuses_a_finer_target(self):
        self.assertEqual(CT.parse_resample("3.24:7.91"), (3.24, 7.91))
        self.assertIsNone(CT.parse_resample(None))
        with self.assertRaises(SystemExit):
            CT.parse_resample("7.91:3.24")


class GateTests(unittest.TestCase):
    def scored(self, sites=120, pairs=5, gain=True, seed=1):
        rng = np.random.default_rng(seed)
        n = 2 * sites
        y = np.tile([1, 0], sites)
        grp = np.repeat([f"P{i % pairs}" for i in range(sites)], 2)
        ctrl = rng.normal(size=n) + 0.6 * y
        s = ctrl + (1.5 * y if gain else 0.0)
        strat = np.array(["touching"] * n)
        apart = np.array(["apart"] * 40)
        ya = np.tile([1, 0], 20)
        ga = np.repeat([f"P{i % pairs}" for i in range(20)], 2)
        ca = 5.0 * ya + rng.normal(size=40) * 0.1
        return (np.concatenate([s, ca]), np.concatenate([y, ya]), np.concatenate([grp, ga]),
                np.concatenate([strat, apart]), np.concatenate([ctrl, ca]))

    def test_a_clear_gain_passes(self):
        s, y, grp, strat, ctrl = self.scored()
        doc = M.readout(None, s, y, grp, strat, ctrl, boot=500, permutations=200)
        self.assertEqual(doc["verdict"], "PASS", doc)

    def test_no_gain_over_the_control_fails(self):
        s, y, grp, strat, ctrl = self.scored(gain=False)
        doc = M.readout(None, s, y, grp, strat, ctrl, boot=500, permutations=200)
        self.assertEqual(doc["verdict"], "FAIL", doc)

    def test_too_few_sites_is_void(self):
        s, y, grp, strat, ctrl = self.scored(sites=40)
        doc = M.readout(None, s, y, grp, strat, ctrl, boot=200, permutations=100)
        self.assertEqual(doc["verdict"], "VOID")

    def test_the_permutation_gate_sits_near_one_half(self):
        rng = np.random.default_rng(3)
        ok, mean, half = M.perm_gate(rng.normal(size=300), np.tile([1, 0], 150), k=300)
        self.assertTrue(ok)
        self.assertLess(abs(mean - 0.5), 0.02)


class ModelFileTests(unittest.TestCase):
    def test_an_unknown_model_file_is_refused_before_it_is_unpickled(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "other.joblib"
            p.write_bytes(b"not a model file")
            with self.assertRaises(SystemExit) as cm:
                M.load_model(p)
            self.assertIn("--trust-model", str(cm.exception))

    @unittest.skipIf(sklearn is None, "needs scikit-learn and joblib")
    def test_fit_then_score_a_run_from_the_command_line(self):
        rows, F = synthetic_run(n=40, seed=4)
        F["tok"][:, 1] = F["tok"][:, 0] + 0.1 * F["tok"][:, 1]     # a2 near a in token space, b not: learnable
        sites = [dict(r, gap_a=5.0, gap_b=5.0) for r in rows]
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            FE.save_run(run, sites, {r["id"]: {k: v[n] for k, v in F.items()} for n, r in enumerate(rows)},
                        {r["id"]: r["scores"] for r in rows}, {"kind": "test"})
            model = Path(td) / "mine.joblib"
            self.assertEqual(main(["fit", "--run", str(run), "--out", str(model)]), 0)
            out = Path(td) / "readout.json"
            with self.assertRaises(SystemExit):           # a model made here is not a known file
                main(["score", "--run", str(run), "--model", str(model), "--out", str(out)])
            self.assertEqual(main(["score", "--run", str(run), "--model", str(model), "--trust-model",
                                   "--boot", "50", "--permutations", "20", "--out", str(out)]), 0)
            doc = json.loads(out.read_text())
            self.assertEqual(doc["table"]["touching"]["sites"], 40)
            self.assertGreater(doc["primary"]["parting"], 0.9)          # scored on its own training rows
            self.assertIsNone(doc["verdict"])


def flat_layers(gap):
    z, x = np.meshgrid(np.arange(100, 400, 4.0), np.arange(100, 400, 4.0), indexing="ij")
    lower = np.stack([x, np.full_like(x, 200.0), z], -1)
    upper = np.stack([x, np.full_like(x, 200.0 + gap), z], -1)
    return {"L0": lower, "L1": upper}


class SitesTests(unittest.TestCase):
    def test_two_layers_five_voxels_apart_give_touching_sites(self):
        rule = G.Rule(seed=7, dense=2, caps={"touching": 5, "apart": 5, "fused": 5, "mixed": 5})
        sites = G.make_sites(flat_layers(5.0), (500, 500, 500), rule)
        self.assertTrue(sites)
        self.assertEqual({s["stratum"] for s in sites}, {"touching"})
        for s in sites:
            self.assertAlmostEqual(s["gap_a"], 5.0, places=1)
            self.assertAlmostEqual(s["dist"], s["dist_same"], delta=1.0)    # matched distance

    def test_the_site_hash_ignores_key_order(self):
        a = [{"id": 0, "pair": "L0-L1", "a": [1, 2, 3]}]
        b = [{"a": [1, 2, 3], "pair": "L0-L1", "id": 0}]
        self.assertEqual(G.sites_sha256(a), G.sites_sha256(b))

    def test_crossings_are_reported_and_an_unordered_stack_is_refused(self):
        g = flat_layers(5.0)
        report, cross = G.stack_layers(g, min_matched=10)
        self.assertEqual(len(cross), 0)
        with self.assertRaises(ValueError):
            G.stack_layers({"L0": g["L1"], "L1": g["L0"]}, min_matched=10)

    def test_the_sites_command_reads_npy_grids(self):
        g = flat_layers(5.0)
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for name, grid in g.items():
                paths += ["--layer", str(Path(td) / f"{name}.npy")]
                np.save(paths[-1], grid)
            out = Path(td) / "sites.json"
            self.assertEqual(main(["sites", *paths, "--shape", "500", "500", "500", "--dense", "2",
                                   "--caps", "5", "5", "5", "5", "--out", str(out)]), 0)
            doc = json.loads(out.read_text())
            self.assertEqual(doc["sites_sha256"], G.sites_sha256(doc["sites"]))
            self.assertEqual({s["stratum"] for s in doc["sites"]}, {"touching"})

    def test_a_vcps_pointset_round_trips(self):
        grid = np.arange(2 * 3 * 3, dtype="<f8").reshape(2, 3, 3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pointset.vcps"
            p.write_bytes(b"width: 3\nheight: 2\ndim: 3\nordered: true\ntype: double\nversion: 1\n<>\n"
                          + grid.tobytes())
            np.testing.assert_array_equal(G.read_vcps(p), grid)


class FeatureTests(unittest.TestCase):
    def test_site_features_have_the_stored_shapes(self):
        rng = np.random.default_rng(0)
        site = {"id": 0, "pair": "L0-L1", "stratum": "touching", "centre": [64, 64, 64],
                "a": [60.0, 64.0, 64.0], "a2": [70.0, 64.0, 64.0], "b": [64.0, 70.0, 64.0],
                "n_a": [0, 1, 0], "n_a2": [0, 1, 0], "n_b": [0, 1, 0]}
        ct = rng.integers(0, 255, (128, 128, 128)).astype(np.uint8)
        tokens = np.ones((288, 16, 16, 16), np.float32)
        f, sc = FE.site_features(site, ct, tokens)
        for k, shape in FE.GROUPS.items():
            self.assertEqual(f[k].shape, shape, k)
        self.assertAlmostEqual(sc["dino_base"][0], 1.0, places=6)       # constant tokens: cosine 1
        with tempfile.TemporaryDirectory() as td:
            FE.save_run(td, [site], {0: f}, {0: sc}, {"kind": "test"})
            feats, scores = FE.load_done(td)
            self.assertEqual(sorted(feats), [0])
            self.assertEqual(json.loads((Path(td) / "run.json").read_text())["kind"], "test")

    def test_trilinear_hits_token_centers(self):
        grid = np.random.default_rng(1).normal(size=(4, 16, 16, 16))
        center = np.array([3, 7, 11]) * FE.STRIDE + (FE.STRIDE - 1) / 2
        np.testing.assert_allclose(FE.trilinear(grid, center), grid[:, 3, 7, 11], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
