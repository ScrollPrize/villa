"""Check temporary array ownership and unchanged label-transfer results."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import weakref

import numpy as np

from vesuvius.tifxyz_label_transfer import core


SEARCH_ATTRIBUTES = ("tree", "grid_index", "valid_flat")


def plane(dtype=np.float32):
    rows, cols = np.meshgrid(np.arange(9, dtype=dtype), np.arange(9, dtype=dtype), indexing="ij")
    return core.Surface(x=cols, y=rows, z=np.full(rows.shape, 10, dtype=dtype))


class KeepSearchMapper(core.SurfaceMapper):
    """Numerical control: retain the same search state through rasterization."""

    def __delattr__(self, name):
        if name not in SEARCH_ATTRIBUTES:
            super().__delattr__(name)


def tracked_mapper(case, *, cached=False, overlap=True):
    state = dict(active=0, started=0, completed=0, mapped=False, deleted=[], peak_active=0)
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    class TrackedMapper(core.SurfaceMapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state["mapper"] = self
            arrays = [self.valid_flat]
            if self.tree is not None:
                arrays.extend((self.tree.data, self.tree.indices))
            if self.grid_index is not None:
                arrays.extend((
                    self.grid_index.sorted_points,
                    self.grid_index.order,
                    self.grid_index.sorted_cell_ids,
                ))
            state["search_refs"] = [weakref.ref(array) for array in arrays]

        def locate(self, *args, **kwargs):
            with lock:
                ordinal = state["started"]
                state["started"] += 1
                state["active"] += 1
                state["peak_active"] = max(state["peak_active"], state["active"])
            try:
                case.assertTrue(all(hasattr(self, name) for name in SEARCH_ATTRIBUTES))
                if overlap and ordinal < 2:
                    barrier.wait(timeout=10)
                return super().locate(*args, **kwargs)
            finally:
                with lock:
                    state["active"] -= 1
                    state["completed"] += 1

        def build_target_uv_map(self, *args, **kwargs):
            case.assertFalse(cached, "A matching UV cache must skip mapping")
            result = super().build_target_uv_map(*args, **kwargs)
            state["mapped"] = True
            state["distance_ref"] = weakref.ref(result[2])
            return result

        def __delattr__(self, name):
            if name in SEARCH_ATTRIBUTES:
                case.assertEqual(state["active"], 0)
                case.assertEqual(state["started"], state["completed"])
                case.assertEqual(state["mapped"], not cached)
                state["deleted"].append(name)
            super().__delattr__(name)

    return TrackedMapper, state


class MapperLifetimeTests(unittest.TestCase):
    def test_affine_input_is_released_before_search_index_construction(self):
        # Check both point dtypes: float64 input must also be independent of
        # the affine result, even though astype(copy=False) can reuse its input.
        for kind in ("kdtree", "grid"):
            for dtype in (np.float32, np.float64):
                with self.subTest(kind=kind, dtype=dtype):
                    source = plane(dtype)
                    affine = np.eye(4)
                    affine[0, 0] = 1.25
                    affine[:3, 3] = (0.25, 0.5, 2.0)
                    apply = core.apply_affine
                    state = {}

                    def watch_affine(points, matrix):
                        result = apply(points, matrix)
                        if "input_ref" not in state:
                            state["input_ref"] = weakref.ref(points)
                            self.assertFalse(np.shares_memory(points, result))
                        return result

                    name = "cKDTree" if kind == "kdtree" else "GridVertexIndex"
                    index = getattr(core, name)

                    def watch_index(*args, **kwargs):
                        self.assertIsNone(state["input_ref"]())
                        state["index_checked"] = True
                        return index(*args, **kwargs)

                    # Plain replacements avoid Mock.call_args retaining arrays.
                    with (
                        mock.patch.object(core, "apply_affine", watch_affine),
                        mock.patch.object(core, name, watch_index),
                    ):
                        mapper = core.SurfaceMapper(
                            source, affine, vertex_index=kind, index_max_distance=0.25
                        )
                    self.assertTrue(state["index_checked"])
                    expected = apply(
                        np.column_stack((source.x.ravel(), source.y.ravel(), source.z.ravel())),
                        affine,
                    )
                    for i, field in enumerate((mapper.tx, mapper.ty, mapper.tz)):
                        np.testing.assert_array_equal(field.ravel(), expected[:, i])

    def test_public_mapper_remains_reusable_after_parallel_mapping_and_callbacks(self):
        source = plane()
        target = plane()
        query = np.array([[1.25, 2.5, 10.0], [6.5, 5.25, 10.0]])
        for kind in ("kdtree", "grid"):
            for workers in (1, 2):
                with self.subTest(kind=kind, workers=workers):
                    mapper = core.SurfaceMapper(source, vertex_index=kind, index_max_distance=0.25)
                    before = mapper.locate(query, 0.25, query_workers=1)
                    callbacks = []

                    def progress(complete, total):
                        callbacks.append((complete, total))
                        self.assertTrue(all(hasattr(mapper, name) for name in SEARCH_ATTRIBUTES))
                        during = mapper.locate(query, 0.25, query_workers=1)
                        for a, b in zip(before, during):
                            np.testing.assert_array_equal(a, b)

                    first = mapper.build_target_uv_map(
                        target, 0.25, query_batch_size=9, workers=workers, progress=progress
                    )
                    second = mapper.build_target_uv_map(
                        target, 0.25, query_batch_size=9, workers=workers
                    )
                    self.assertEqual(callbacks, [(i, 9) for i in range(1, 10)])
                    for a, b in zip(first, second):
                        np.testing.assert_array_equal(a, b)
                    for a, b in zip(before, mapper.locate(query, 0.25, query_workers=1)):
                        np.testing.assert_array_equal(a, b)

    def _check_transfer(self, rasterizer):
        source, target = plane(), plane()
        target.z[3:6, 3:6] += 2.0  # Rejected geometry needs a real seam fill.
        label = np.arange(81, dtype=np.uint16).reshape(9, 9) + 1
        second_label = np.flip(label, axis=1).copy()
        validity = np.full((9, 9), 255, dtype=np.uint8)
        validity[6:, :] = 128  # Previously interpolated provenance must survive.
        validity[:, 0] = 0
        for kind in ("kdtree", "grid"):
            kwargs = dict(output_shape=(18, 18), max_distance=0.25, vertex_index=kind,
                          workers=2, query_batch_size=9, tile_size=6, fill_seams=True,
                          source_validity=validity, rasterizer=rasterizer,
                          additional_source_labels=[second_label])
            reference_extra = np.zeros((18, 18), dtype=np.uint16)
            with mock.patch.object(core, "SurfaceMapper", KeepSearchMapper):
                expected, expected_valid, _, expected_stats = core.transfer_array(
                    source, target, label, additional_outputs=[reference_extra], **kwargs)
            self.assertGreater(expected_stats.seam_filled_pixels, 0)
            self.assertGreater(expected_stats.inherited_filled_pixels, 0)
            with tempfile.TemporaryDirectory() as temporary:
                uv_cache = Path(temporary) / "uv.npz"
                for cached in (False, True):
                    with self.subTest(rasterizer=rasterizer, kind=kind, cached=cached):
                        mapper_class, state = tracked_mapper(self, cached=cached)
                        output = np.zeros_like(expected)
                        extra = np.zeros_like(reference_extra)
                        output_valid = np.zeros_like(expected_valid)
                        tile_order = []
                        mapping_callbacks = []
                        original_fill = core._fill_uv_field

                        def assert_released():
                            mapper = state["mapper"]
                            self.assertFalse(any(
                                hasattr(mapper, name) for name in SEARCH_ATTRIBUTES
                            ))
                            self.assertEqual(state["active"], 0)
                            self.assertEqual(state["deleted"], list(SEARCH_ATTRIBUTES))
                            self.assertTrue(all(ref() is None for ref in state["search_refs"]))
                            if not cached:
                                self.assertIsNone(state["distance_ref"]())
                            for field in (mapper.tx, mapper.ty, mapper.tz):
                                self.assertEqual(field.shape, source.shape)
                            self.assertIs(mapper.source.valid, source.valid)

                        def fill(*args, **kw):
                            assert_released()
                            state["fill_checked"] = True
                            return original_fill(*args, **kw)

                        def progress(complete, total):
                            if not state["deleted"]:
                                self.assertTrue(all(
                                    hasattr(state["mapper"], name) for name in SEARCH_ATTRIBUTES
                                ))
                                mapping_callbacks.append((complete, total))

                        def callback(bounds, labels, valid):
                            assert_released()
                            tile_order.append(bounds)
                            r0, r1, c0, c1 = bounds
                            output[r0:r1, c0:c1] = labels[0]
                            extra[r0:r1, c0:c1] = labels[1]
                            output_valid[r0:r1, c0:c1] = valid

                        with (
                            mock.patch.object(core, "SurfaceMapper", mapper_class),
                            mock.patch.object(core, "_fill_uv_field", fill),
                        ):
                            result = core.transfer_array(
                                source, target, label, uv_cache=uv_cache,
                                materialize_output=False, tile_callback=callback,
                                progress=progress, **kwargs,
                            )
                        self.assertEqual(result[:3], (None, None, None))
                        self.assertTrue(state["fill_checked"])
                        self.assertEqual(tile_order, list(core.iter_tiles((18, 18), 6)))
                        self.assertEqual(state["started"], 0 if cached else 9)
                        self.assertEqual(len(mapping_callbacks), 1 if cached else 9)
                        if not cached:
                            self.assertGreaterEqual(state["peak_active"], 2)
                        np.testing.assert_array_equal(output, expected)
                        np.testing.assert_array_equal(extra, reference_extra)
                        np.testing.assert_array_equal(output_valid, expected_valid)
                        self.assertEqual(result[3].as_dict(), expected_stats.as_dict())

    def test_python_transfer_releases_private_state_before_seams_and_tile_callbacks(self):
        self._check_transfer("python")

    def test_native_transfer_releases_private_state_before_seams_and_tile_callbacks(self):
        from vesuvius.tifxyz_label_transfer.native import (
            load_native_library, native_unavailable_reason,
        )

        if load_native_library() is None:
            reason = native_unavailable_reason()
            # An absent optional build may be skipped; a broken build must fail.
            if reason and reason.startswith("native rasterizer was not built at "):
                self.skipTest(reason)
        # Use the native backend explicitly; a Python fallback would not test it.
        self._check_transfer("native")

    def test_mapping_callback_exception_waits_for_workers_without_early_release(self):
        for kind in ("kdtree", "grid"):
            with self.subTest(kind=kind):
                mapper_class, state = tracked_mapper(self)

                def fail_progress(complete, total):
                    self.assertTrue(all(
                        hasattr(state["mapper"], name) for name in SEARCH_ATTRIBUTES
                    ))
                    raise RuntimeError("mapping callback failure")

                with mock.patch.object(core, "SurfaceMapper", mapper_class):
                    with self.assertRaisesRegex(RuntimeError, "mapping callback failure"):
                        core.transfer_array(plane(), plane(), np.ones((9, 9), dtype=np.uint8),
                            max_distance=0.25, vertex_index=kind, workers=2, query_batch_size=9,
                            progress=fail_progress, rasterizer="python")
                self.assertEqual(state["active"], 0)
                self.assertEqual(state["started"], state["completed"])
                self.assertGreaterEqual(state["peak_active"], 2)
                self.assertEqual(state["deleted"], [])
                self.assertFalse(state["mapped"])


if __name__ == "__main__":
    unittest.main()
