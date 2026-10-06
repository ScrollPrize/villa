"""Fused CPU depth projection must match the original per-layer projection."""

from __future__ import annotations

import builtins
import importlib.util
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from vesuvius.tifxyz_label_transfer import self_render_tifxyz as renderer
from vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence import ZarrLevel


HAS_NUMBA = importlib.util.find_spec("numba") is not None


def reference_render_values(xyz, normals, valid, offsets, sampler, chunks):
    """The original _render_values body: one scipy trilinear pass per offset."""

    output = np.zeros(valid.shape, dtype=np.float32)
    selected = valid.ravel()
    if not np.any(selected):
        return output
    block, origin = sampler.load_block(chunks)
    points = xyz.reshape(-1, 3)[selected]
    directions = normals.reshape(-1, 3)[selected]
    maximum = np.zeros(points.shape[0], dtype=np.float32)
    for offset in offsets:
        values = sampler.sample_block(points + directions * float(offset), block, origin)
        maximum = np.maximum(maximum, values)
    output.ravel()[selected] = maximum
    return output


class BlockSampler:
    """In-memory CT block using the unchanged RawChunkSampler.sample_block."""

    sample_block = renderer.RawChunkSampler.sample_block

    def __init__(self, block, origin=(0, 0, 0), scale=(1.0, 1.0, 1.0), shape=None):
        self.block = block
        self.origin = np.asarray(origin, dtype=np.int64)
        self.info = SimpleNamespace(
            shape=shape or block.shape,
            scale_zyx=scale,
            metadata={"dtype": block.dtype.str},
        )
        self.loads = 0

    def load_block(self, chunks):
        self.loads += 1
        return self.block, self.origin


def assert_close(actual, expected):
    # Observed real-scroll renders are bit-identical; allow one float32 ULP
    # because the summation order of the eight trilinear terms may differ.
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)


@unittest.skipUnless(HAS_NUMBA, "numba is not installed")
class FastDepthProjectionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261004)

    def test_matches_original_across_dtypes_scales_and_offsets(self):
        # (block shape, block origin, scale_zyx, full volume shape); points fall
        # outside the volume and the block so both clamps are exercised.
        setups = (
            ((7, 9, 11), (0, 0, 0), (1.0, 1.0, 1.0), (7, 9, 11)),
            ((3, 5, 7), (4, 7, 9), (0.5, 1.7, 2.3), (19, 23, 31)),
            ((1, 2, 3), (0, 0, 0), (1.0, 1.0, 1.0), (1, 2, 3)),
        )
        for ct, point, direction, setup in itertools.product(
            (np.uint8, np.uint16, np.int16),
            (np.float32, np.float64),
            (np.float32, np.float64),
            setups,
        ):
            with self.subTest(ct=ct, point=point, direction=direction, setup=setup):
                shape, origin, scale, full_shape = setup
                block = (self.rng.integers(0, 255, shape) - (100 if ct is np.int16 else 0)).astype(ct)
                xyz = self.rng.uniform(-10, 50, (5, 7, 3)).astype(point)
                normals = self.rng.normal(size=xyz.shape).astype(direction)
                valid = self.rng.random((5, 7)) > 0.2
                xyz[~valid] = np.nan
                normals[~valid] = np.nan
                for offsets in ([], [0.0], [-10.1, -2.0, 0.0, 2.0, 10.1]):
                    sampler = BlockSampler(block, origin, scale, full_shape)
                    arguments = (xyz, normals, valid, offsets, sampler, {(0, 0, 0)})
                    actual = renderer._render_values(*arguments)
                    assert_close(actual, reference_render_values(*arguments))
                    np.testing.assert_array_equal(actual[~valid], 0)

    def test_maximum_starts_at_zero_and_empty_mask_skips_loading(self):
        xyz = np.ones((3, 5, 3), dtype=np.float64)
        normals = np.zeros_like(xyz)
        valid = np.ones((3, 5), dtype=bool)
        sampler = BlockSampler(np.full((4, 4, 4), -50, dtype=np.int16))
        output = renderer._render_values(xyz, normals, valid, [-1, 0, 1], sampler, {(0, 0, 0)})
        np.testing.assert_array_equal(output, np.zeros(valid.shape, dtype=np.float32))
        sampler.loads = 0
        valid[:] = False
        output = renderer._render_values(xyz, normals, valid, [0], sampler, set())
        self.assertEqual(sampler.loads, 0)
        np.testing.assert_array_equal(output, np.zeros(valid.shape, dtype=np.float32))

    def test_dispatch_and_fallbacks(self):
        from vesuvius.tifxyz_label_transfer import _fast_depth_projection as fast

        xyz = self.rng.uniform(0, 5, (3, 4, 3))
        normals = self.rng.normal(size=xyz.shape)
        valid = np.ones((3, 4), dtype=bool)
        sampler = BlockSampler(self.rng.integers(0, 255, (6, 6, 6), dtype=np.uint8))
        arguments = (xyz, normals, valid, [-2.0, 0.0, 2.0], sampler, {(0, 0, 0)})
        expected = reference_render_values(*arguments)

        # Integer CT takes the fused path.
        with mock.patch.object(fast, "render_values", wraps=fast.render_values) as used:
            assert_close(renderer._render_values(*arguments), expected)
            used.assert_called_once()

        # Without numba the original path runs; other import errors propagate.
        real_import = builtins.__import__
        for missing in ("numba", "unrelated_dependency"):
            def failing_import(name, *args, missing=missing, **kwargs):
                if name.endswith("_fast_depth_projection"):
                    raise ModuleNotFoundError("simulated", name=missing)
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=failing_import):
                if missing == "numba":
                    np.testing.assert_array_equal(renderer._render_values(*arguments), expected)
                else:
                    with self.assertRaises(ModuleNotFoundError):
                        renderer._render_values(*arguments)

        # Floating-point CT keeps the original path.
        float_sampler = BlockSampler(sampler.block.astype(np.float32) - 10.5)
        float_arguments = (xyz, normals, valid, [-2.0, 0.0, 2.0], float_sampler, {(0, 0, 0)})
        with mock.patch.object(fast, "render_values", side_effect=AssertionError("unexpected dispatch")):
            np.testing.assert_array_equal(
                renderer._render_values(*float_arguments), reference_render_values(*float_arguments)
            )

    def test_raw_chunk_sampler_sparse_fill_and_cache_sizes(self):
        metadata = {
            "shape": [8, 8, 8],
            "chunks": [4, 4, 4],
            "dtype": "|u1",
            "dimension_separator": "/",
            "fill_value": 17,
        }
        info = ZarrLevel("offline", "0", {}, metadata, metadata, (1.0, 1.0, 1.0))
        sparse = {(1, 1, 1)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in sorted(set(itertools.product(range(2), repeat=3)) - sparse):
                path = root / renderer._relative_chunk_key(info, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.rng.integers(0, 255, (4, 4, 4), dtype=np.uint8).tobytes())
            xyz = self.rng.uniform(-2, 10, (9, 11, 3))
            normals = self.rng.normal(size=xyz.shape)
            valid = self.rng.random((9, 11)) > 0.1
            offsets = [-2.0, 0.0, 2.0]
            plan = renderer.required_chunks(xyz, normals, valid, offsets, info)
            baseline = renderer.RawChunkSampler(info, root, sparse)
            expected = reference_render_values(xyz, normals, valid, offsets, baseline, plan)
            for capacity in (0, 1, 96):
                with self.subTest(max_cached_chunks=capacity):
                    sampler = renderer.RawChunkSampler(info, root, sparse, max_cached_chunks=capacity)
                    for _ in range(2):
                        assert_close(renderer._render_values(xyz, normals, valid, offsets, sampler, plan), expected)


if __name__ == "__main__":
    unittest.main()
