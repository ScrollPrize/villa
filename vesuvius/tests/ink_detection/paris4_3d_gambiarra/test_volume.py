"""Inference pipeline: exact percentiles, normalization, unit planning, and the GPU engine on a synthetic local zarr
against a NumPy accumulation reference (grid/Gaussian/normalization helpers are shared; exact stand-in network)."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vesuvius.ink_detection.paris4_3d_gambiarra import ct
from vesuvius.ink_detection.paris4_3d_gambiarra import volume as V


def parent_with_fraction(rng):
    """A uint8 256^3 block whose p1 and p99 interpolate between two different values."""
    lo_idx, hi_idx, _ = V.percentile_table()
    x = np.full(V.N_PARENT, 51, np.uint8)
    x[:lo_idx[0] + 1] = 10      # order statistic lo_idx[0] = 10, the next one 51: p1 = 10 + 41 * 0.15
    x[hi_idx[1]:] = 200         # order statistic hi_idx[1] = 200, the one before 51: p99 = 200 - 149 * 0.15
    return rng.permutation(x).reshape((256,) * 3)


class Percentiles(unittest.TestCase):
    def test_equal_to_numpy(self):
        rng = np.random.default_rng(0)
        blocks = [rng.integers(0, 256, (256,) * 3, dtype=np.uint8),
                  np.clip(rng.normal(100, 3, (256,) * 3), 0, 255).astype(np.uint8),
                  np.full((256,) * 3, 7, np.uint8),
                  parent_with_fraction(rng)]
        pct = V.Percentiles('cpu')
        for b in blocks:
            counts = torch.bincount(torch.from_numpy(b).view(-1).long(), minlength=256)[None]
            lo, hi = pct(counts)
            ref = np.percentile(b.astype(np.float32), [1, 99])
            self.assertEqual((float(lo[0]), float(hi[0])), (ref[0], ref[1]))
            # training limits (the original loaders called np.percentile once per quantile, which returns float32)
            # are the float32 roundings of these inference limits (np.percentile with q = [1, 99], float64)
            self.assertEqual(ct.percentiles(b), tuple(float(np.float32(v)) for v in ref))
        self.assertAlmostEqual(float(ref[0]), 16.15, places=6)  # the interpolated case is exercised: here the two differ
        self.assertAlmostEqual(float(ref[1]), 177.65, places=6)
        self.assertNotEqual(ct.percentiles(blocks[-1]), tuple(ref))


class Normalization(unittest.TestCase):
    def test_against_references(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(0, 256, (4, 128, 128, 128), dtype=np.uint8)
        for lo, hi in ((31., 156.), (40.15, 199.85), (7., 7.)):
            x = V.normalize_batch(torch.from_numpy(raw), torch.full((4,), lo, dtype=torch.float64),
                                  torch.full((4,), hi, dtype=torch.float64)).numpy()
            np.testing.assert_array_equal(x[2], V.normalize_reference(raw[2], lo, hi))
            if lo == int(lo):  # float32-exact limits: equal to the training-side normalization
                np.testing.assert_array_equal(x[2], ct.normalize(raw[2], lo, hi))


class Planning(unittest.TestCase):
    def test_grid(self):
        a = V.grid_axes(16347)
        self.assertEqual((a[0], a[-1] + 64), (0, 16347))
        self.assertTrue(np.all(np.diff(a) <= 32))
        grid = V.crop_grid((75784, 32693, 32693))
        for g, s in zip(grid, (75784, 32693, 32693)):
            self.assertTrue(np.all(2 * g - 64 >= 0) and np.all(2 * g + 192 <= s))

    def test_skip_and_chunks(self):
        shape0 = (768, 768, 768)
        present = np.zeros((6, 6, 6), bool)
        present[1, 1, 1] = True                     # one present level-0 chunk: level-0 [128, 256)^3
        grid = V.crop_grid(shape0)
        plan = V.unit_plan((0, 0, 0), 384, shape0, grid, lambda c0, c1: present[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]], 128)
        for o in plan['origins']:                   # every kept crop's level-0 128^3 touches the present chunk
            self.assertTrue(all(2 * v < 256 and 2 * v + 128 > 128 for v in o))
        self.assertEqual(plan['chunks'], [(0, 0, 0)])  # output chunk 0 holds output voxels of level-0 chunk 1
        empty = V.unit_plan((2, 2, 2), 128, shape0, grid, lambda c0, c1: present[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]], 128)
        self.assertIn('empty', empty)


def write_level0(root, vol, present):
    root.mkdir(parents=True)
    (root / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=list(vol.shape), chunks=[128] * 3, dtype='|u1', order='C',
                                                  fill_value=0, filters=None, compressor=None, dimension_separator='/')))
    for key in itertools.product(*[range(n) for n in present.shape]):
        if present[key]:
            d = root / str(key[0]) / str(key[1])
            d.mkdir(parents=True, exist_ok=True)
            block = np.zeros((128,) * 3, np.uint8)
            part = vol[tuple(slice(k * 128, (k + 1) * 128) for k in key)]
            block[tuple(slice(0, s) for s in part.shape)] = part
            (d / str(key[2])).write_bytes(block.tobytes())


def read_codes(root, shape1, chunk):
    out = np.zeros(shape1, np.uint8)
    for f in root.glob('*/*/*'):
        if f.name.startswith('.'):
            continue
        key = tuple(int(v) for v in f.relative_to(root).parts)
        block = np.frombuffer(f.read_bytes(), np.uint8).reshape((chunk,) * 3)
        lo = np.array(key) * chunk
        hi = np.minimum(lo + chunk, shape1)
        out[tuple(slice(a, b) for a, b in zip(lo, hi))] = block[tuple(slice(0, b - a) for a, b in zip(lo, hi))]
    return out


def stand_in(x):
    """[B, 1, 128^3] -> [B, 1, 64^3]; exact arithmetic on CPU and GPU (subsampling, x8, -4)."""
    return x[:, :, ::2, ::2, ::2] * 8 - 4


def cpu_codes(vol, present):
    """NumPy blend/mask/coding reference; grid, Gaussian and normalization helpers are shared with the engine."""
    shape1 = V.output_shape(vol.shape)
    grid = V.crop_grid(vol.shape)
    g = V.gaussian()
    S, W = np.zeros(shape1), np.zeros(shape1)
    for a in itertools.product(*grid):
        a = np.array(a)
        lo, hi = 2 * a // 128, (2 * a + 127) // 128 + 1
        if not present[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
            continue
        parent = vol[tuple(slice(2 * v - 64, 2 * v + 192) for v in a)]
        p1, p99 = np.percentile(parent.astype(np.float32), [1, 99])
        x = V.normalize_reference(parent[64:192, 64:192, 64:192], p1, p99)
        f = stand_in(torch.from_numpy(x)[None, None]).numpy()[0, 0]
        region = tuple(slice(v, v + 64) for v in a)
        S[region] += (g * f).astype(np.float64)
        W[region] += g.astype(np.float64)
    logit = np.zeros(shape1, np.float32)
    np.divide(S, W, out=S, where=W > 0)
    logit[:] = S
    code = np.clip(np.floor(logit.astype(np.float64) * 10) + 128, 1, 255).astype(np.uint8)
    cov = [np.zeros(n, bool) for n in shape1]
    for c, g_ in zip(cov, grid):
        c[g_[0]:g_[-1] + 64] = True
    idx = np.ix_(*[np.arange(n) // 64 for n in shape1])
    mask = present[idx] & cov[0][:, None, None] & cov[1][None, :, None] & cov[2][None, None, :]
    assert np.all(W[mask] > 0)
    return np.where(mask, code, 0)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class EngineOnSyntheticVolume(unittest.TestCase):
    def test_units_match_cpu(self):
        rng = np.random.default_rng(2)
        shape0 = (384, 384, 384)                    # crop origins 32, 64, 96 per axis: 27 crops
        z, y, x = np.meshgrid(*[np.arange(s) for s in shape0], indexing='ij')
        vol = np.clip(60 + 0.3 * z + 0.2 * y - 0.1 * x + rng.normal(0, 20, shape0), 0, 255).astype(np.uint8)
        present = np.ones((3, 3, 3), bool)
        present[0, 0, 0] = present[2, 1, 2] = present[0, 2, 2] = False
        filled = vol.copy()
        for key in zip(*np.nonzero(~present)):
            filled[tuple(slice(k * 128, (k + 1) * 128) for k in key)] = 0
        expected = cpu_codes(filled, present)
        self.assertGreater(len(np.unique(expected)), 20)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_level0(tmp / 'l0', vol, present)
            store = ct.ChunkStore(tmp / 'l0', threads=4)
            self.assertTrue(np.array_equal(store.presence(), present))
            self.assertTrue(np.array_equal(store.probe((0, 0, 0), (3, 3, 3)), present))
            for unit, chunk in ((128, 64), (192, 64)):
                out = tmp / f'out-{unit}'
                meta = V.zarray(V.output_shape(shape0), chunk, None)
                writer = V.ZarrWriter(out, meta, threads=2, fsync=False)
                engine = V.Engine(stand_in, store, writer, batch=16, check=V.cpu_check)
                for idx in itertools.product(*[range(-(-s // unit)) for s in V.output_shape(shape0)]):
                    plan = V.unit_plan(idx, unit, shape0, engine.grid, lambda c0, c1: present[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]], chunk)
                    if 'empty' not in plan:
                        engine.run_unit(plan)
                got = read_codes(out, V.output_shape(shape0), chunk)
                np.testing.assert_array_equal(got, expected, err_msg=f'unit {unit}')
            store.close()


class EngineOnCPU(unittest.TestCase):
    def test_sparse_ring_and_edge_units(self):
        """Exercise the actual engine on CPU: sparse rows, ring wrap, padding, masks and two unit sizes."""
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        shape0 = (769, 256, 256)
        vol = np.broadcast_to(np.arange(256, dtype=np.uint8), shape0).copy()
        present = np.ones((7, 2, 2), bool)
        present[2:4] = False
        vol[256:512] = 0
        expected = cpu_codes(vol, present)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_level0(root / 'l0', vol, present)
            store = ct.ChunkStore(root / 'l0', threads=2)
            try:
                for unit in (128, 512):
                    out = root / str(unit)
                    writer = V.ZarrWriter(out, V.zarray(V.output_shape(shape0), 64, None), threads=2, fsync=False)
                    engine = V.Engine(stand_in, store, writer, batch=2, device='cpu', check=V.cpu_check)
                    for iz in range(-(-V.output_shape(shape0)[0] // unit)):
                        plan = V.unit_plan((iz, 0, 0), unit, shape0, engine.grid,
                                           lambda a, b: present[a[0]:b[0], a[1]:b[1], a[2]:b[2]], 64)
                        if 'empty' not in plan:
                            engine.run_unit(plan)
                    np.testing.assert_array_equal(read_codes(out, V.output_shape(shape0), 64), expected)
            finally:
                store.close()

    def test_no_crop_coverage(self):
        store = type('Store', (), {'shape': (127, 256, 256)})()
        engine = V.Engine(stand_in, store, None, batch=1, device='cpu')
        self.assertEqual(engine.coverage[0], (0, 0))
        plan = V.unit_plan((0, 0, 0), 128, store.shape, engine.grid,
                           lambda *_: self.fail('no presence needed outside crop coverage'), 64)
        self.assertEqual(plan['empty'], 'outside crop coverage')


if __name__ == '__main__':
    unittest.main()
