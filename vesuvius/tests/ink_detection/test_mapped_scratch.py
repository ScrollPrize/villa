"""Mapped accumulation storage: copies, cleanup, and tiled output."""

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesuvius.ink_detection.inference import infer


CODE = vars(infer)


class ScratchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gpu01_audit_")
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.cleanup = ExitStack()
        self.addCleanup(self.cleanup.close)

    def store(self, name="p", shape=(17, 19)):
        return CODE["open_temp_accumulation_array"](
            self.root / (name + ".zarr"), shape=shape, chunks=(8, 8),
            backend="memmap", cleanup=self.cleanup,
        )

    def test_zero_initialization_size_dtype_and_no_read_alias(self):
        store = self.store()
        self.assertEqual(store.path.stat().st_size, 17 * 19 * 4)
        self.assertEqual(store[:].dtype, np.float32)
        np.testing.assert_array_equal(store[:], np.zeros(store.shape, np.float32))
        store[2:7, 3:11] = np.float32(0.25)
        read = store[2:7, 3:11]
        read[:] = 0.9
        np.testing.assert_array_equal(store[2:7, 3:11], np.full((5, 8), 0.25, np.float32))

    def test_closed_access_rejected_and_close_idempotent(self):
        store = self.store()
        retained_copy = store[:]
        mapping = store._array._mmap
        store.close()
        store.close()
        self.assertTrue(mapping.closed)
        with self.assertRaisesRegex(ValueError, "closed"):
            store[:]
        with self.assertRaisesRegex(ValueError, "closed"):
            store[:] = 1
        np.testing.assert_array_equal(retained_copy, np.zeros(store.shape, np.float32))
        store.path.unlink()  # Open mappings would prevent this on Windows.

    def test_shape_validation(self):
        for shape in [(), (2,), (2, 0), (-1, 2), (2, 2, 2)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.store(shape=shape)

    def test_exception_releases_handles_before_directory_removal(self):
        nested = self.root / "scratch"
        nested.mkdir()
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with ExitStack() as cleanup:
                cleanup.callback(shutil.rmtree, nested)
                store = CODE["open_temp_accumulation_array"](
                    nested / "p.zarr", shape=(7, 9), chunks=(4, 4),
                    backend="memmap", cleanup=cleanup,
                )
                mapping = store._array._mmap
                raise RuntimeError("consumer failed")
        self.assertTrue(mapping.closed)
        self.assertFalse(nested.exists())

    def test_failed_initialization_closes_mapping(self):
        mapping = SimpleNamespace(closed=False)

        def close():
            mapping.closed = True

        mapping.close = close

        class FailedInitialization:
            _mmap = mapping

            def __setitem__(self, key, value):
                raise OSError("initialization failed")

        with patch.object(np, "memmap", return_value=FailedInitialization()):
            with self.assertRaisesRegex(OSError, "initialization failed"):
                self.store()
        self.assertTrue(mapping.closed)

    def test_flush_failure_still_closes_mapping(self):
        store = self.store()
        mapping = store._array._mmap
        with patch.object(np.memmap, "flush", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                store.close()
        self.assertTrue(mapping.closed)
        store.close()

    def test_second_store_failure_closes_first(self):
        with self.assertRaises(OSError):
            with ExitStack() as cleanup:
                store = CODE["open_temp_accumulation_array"](
                    self.root / "p.zarr", shape=(7, 9), chunks=(4, 4),
                    backend="memmap", cleanup=cleanup,
                )
                mapping = store._array._mmap
                with patch.object(np, "memmap", side_effect=OSError("create failed")):
                    CODE["open_temp_accumulation_array"](
                        self.root / "w.zarr", shape=(7, 9), chunks=(4, 4),
                        backend="memmap", cleanup=cleanup,
                    )
        self.assertTrue(mapping.closed)

    def test_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown scratch backend"):
            CODE["open_temp_accumulation_array"](
                self.root / "p.zarr", shape=(7, 9), chunks=(4, 4),
                backend="bad", cleanup=self.cleanup,
            )

    def test_zarr_dispatch_closes_store_standin(self):
        calls = []
        result = SimpleNamespace(store=SimpleNamespace(close=lambda: calls.append("close")))
        with patch.dict(CODE, open_temp_zarr_array=lambda *args, **kwargs: result):
            with ExitStack() as cleanup:
                actual = CODE["open_temp_accumulation_array"](
                    self.root / "p.zarr", shape=(7, 9), chunks=(4, 4),
                    backend="zarr", cleanup=cleanup,
                )
                self.assertIs(actual, result)
        self.assertEqual(calls, ["close"])

    def test_overlapping_edges_skipped_block_and_repeat_encoding(self):
        shape, chunk_shape = (11, 13), (4, 5)
        # One scheduled block is skipped, as happens for raw-empty input.
        blocks = [(0, 0, 7, 8), (3, 4, 8, 9), (0, 9, 5, 4)]
        counts = Counter()
        for y, x, h, w in blocks:
            counts.update(CODE["iter_overlapping_chunks"](y, x, h, w, chunk_shape))
        probability, weight = self.store("p", shape), self.store("w", shape)
        accumulator = CODE["ChunkAccumulator"](
            shape=shape, chunk_shape=chunk_shape, prob_sum_store=probability,
            weight_sum_store=weight, contribution_counts=counts,
        )
        expected_probability = np.zeros(shape, np.float32)
        expected_weight = np.zeros(shape, np.float32)
        for index, (y, x, h, w) in enumerate(blocks[:2]):
            tile = np.full((h, w), 0.25 * (index + 1), np.float32)
            weights = np.full((h, w), 0.5, np.float32)
            weights[:2, :2] = 0
            accumulator.add_tile(y0=y, x0=x, tile=tile, tile_weights=weights)
            expected_probability[y:y+h, x:x+w] += tile * weights
            expected_weight[y:y+h, x:x+w] += weights
        accumulator.flush_remaining()
        np.testing.assert_array_equal(probability[:], expected_probability)
        np.testing.assert_array_equal(weight[:], expected_weight)
        expected = expected_probability.copy()
        np.divide(expected, expected_weight, out=expected, where=expected_weight > 1e-6)
        expected = (np.clip(expected, 0, 1) * 255).astype(np.uint8)
        for _ in range(2):
            tiles = iter(CODE["iter_probability_tiles"](probability, weight, chunk_shape))
            for y in range(0, shape[0], chunk_shape[0]):
                for x in range(0, shape[1], chunk_shape[1]):
                    np.testing.assert_array_equal(next(tiles), expected[y:y+4, x:x+5])
            np.testing.assert_array_equal(probability[:], expected_probability)
            np.testing.assert_array_equal(weight[:], expected_weight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
