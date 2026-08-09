from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tifxyz_label_transfer.core import (
    GridVertexIndex,
    Surface,
    SurfaceMapper,
    choose_affine_direction,
    infer_output_shape,
    load_affine,
    transfer_array,
)


def plane(
    height: int,
    width: int,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    scale_yx=(1.0, 1.0),
) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return Surface(
        x=cols + x_offset,
        y=rows + y_offset,
        z=np.full((height, width), 10.0, dtype=np.float32),
        scale_yx=scale_yx,
    )


def wavy_surface(height: int, width: int, seed: int = 7) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    rng = np.random.default_rng(seed)
    surface = Surface(
        x=cols + 0.2 * np.sin(rows / 3.0),
        y=rows + 0.2 * np.cos(cols / 4.0),
        z=10.0 + 1.5 * np.sin(cols / 5.0) + 0.05 * rng.normal(size=rows.shape),
    )
    return surface


class GridVertexIndexTests(unittest.TestCase):
    def test_matches_kdtree_neighbours_within_guarantee(self) -> None:
        from scipy.spatial import cKDTree

        surface = wavy_surface(40, 50)
        assert surface.valid is not None
        points = np.column_stack(
            (surface.x.ravel(), surface.y.ravel(), surface.z.ravel())
        )
        rng = np.random.default_rng(11)
        queries = points[rng.choice(points.shape[0], 500, replace=False)]
        queries = queries + rng.uniform(-0.4, 0.4, size=queries.shape)

        index = GridVertexIndex(points, cell_size=2.75)
        tree = cKDTree(points)
        grid_distances, grid_indices = index.query(queries, k=8)
        tree_distances, tree_indices = tree.query(queries, k=8)

        # Same nearest-neighbour sets whenever the 8th neighbour lies within
        # the covered radius (always true here: spacing 1, cell size 2.75).
        np.testing.assert_allclose(
            np.sort(grid_distances, axis=1), tree_distances, atol=1e-12
        )
        for row in range(queries.shape[0]):
            self.assertEqual(
                set(grid_indices[row].tolist()),
                set(tree_indices[row].tolist()),
            )

    def test_far_and_nonfinite_queries_return_missing(self) -> None:
        points = np.column_stack(
            (
                np.tile(np.arange(5.0), 5),
                np.repeat(np.arange(5.0), 5),
                np.zeros(25),
            )
        )
        index = GridVertexIndex(points, cell_size=1.5)

        queries = np.array(
            [
                [1000.0, 1000.0, 1000.0],
                [-1e30, 0.0, 0.0],
                [np.nan, 1.0, 1.0],
                [2.0, 2.0, 0.0],
            ]
        )
        distances, indices = index.query(queries, k=3)

        self.assertTrue(np.all(np.isinf(distances[:3])))
        self.assertTrue(np.all(indices[:3] == points.shape[0]))
        self.assertEqual(distances[3, 0], 0.0)
        self.assertEqual(indices[3, 0], 12)

    def test_query_pads_when_candidates_fewer_than_k(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
        index = GridVertexIndex(points, cell_size=2.0)

        distances, indices = index.query(np.array([[0.1, 0.0, 0.0]]), k=4)

        self.assertAlmostEqual(distances[0, 0], 0.1)
        self.assertEqual(indices[0, 0], 0)
        self.assertTrue(np.all(np.isinf(distances[0, 1:])))
        self.assertTrue(np.all(indices[0, 1:] == 2))


class VertexIndexEquivalenceTests(unittest.TestCase):
    def test_grid_and_kdtree_transfers_are_identical(self) -> None:
        source = wavy_surface(30, 36)
        target = wavy_surface(30, 36, seed=7)
        label = (
            np.arange(30, dtype=np.uint16)[:, None] * 64
            + np.arange(36, dtype=np.uint16)[None, :]
        )

        results = {}
        for kind in ("grid", "kdtree"):
            output, valid, _, stats = transfer_array(
                source,
                target,
                label,
                output_shape=(60, 72),
                nearest_vertices=8,
                tile_size=32,
                vertex_index=kind,
            )
            results[kind] = (output, valid, stats.mapped_pixels)

        np.testing.assert_array_equal(results["grid"][0], results["kdtree"][0])
        np.testing.assert_array_equal(results["grid"][1], results["kdtree"][1])
        self.assertEqual(results["grid"][2], results["kdtree"][2])
        self.assertGreater(results["grid"][2], 0)

    def test_grid_matches_kdtree_on_stretched_surface(self) -> None:
        # A band of columns stretched ~12x above the median edge length:
        # a median-derived bucket size would miss candidate vertices there.
        def stretched(seed: int) -> Surface:
            surface = wavy_surface(30, 36, seed=seed)
            x = np.asarray(surface.x, dtype=np.float64).copy()
            x[:, 24:] += (
                11.0 * (np.arange(36 - 24, dtype=np.float64) + 1.0)
            )[None, :]
            return Surface(
                x=x, y=surface.y, z=surface.z, valid=surface.valid
            )

        source = stretched(3)
        target = stretched(4)
        label = (
            np.arange(30, dtype=np.uint16)[:, None] * 64
            + np.arange(36, dtype=np.uint16)[None, :]
        )

        results = {}
        for kind in ("grid", "kdtree"):
            output, valid, _, stats = transfer_array(
                source,
                target,
                label,
                output_shape=(60, 72),
                nearest_vertices=8,
                tile_size=32,
                vertex_index=kind,
            )
            results[kind] = (output, valid, stats.mapped_pixels)

        np.testing.assert_array_equal(results["grid"][0], results["kdtree"][0])
        np.testing.assert_array_equal(results["grid"][1], results["kdtree"][1])
        self.assertEqual(results["grid"][2], results["kdtree"][2])
        self.assertGreater(results["grid"][2], 0)

    def test_locate_rejects_max_distance_beyond_index_guarantee(self) -> None:
        source = wavy_surface(10, 12)
        mapper = SurfaceMapper(
            source, vertex_index="grid", index_max_distance=0.5
        )

        with self.assertRaisesRegex(ValueError, "guarantee"):
            mapper.locate(np.zeros((1, 3)), max_distance=5.0)
        # Infinite max_distance means "report what the index can find".
        mapper.locate(np.zeros((1, 3)), max_distance=np.inf)

    def test_rejects_unknown_vertex_index(self) -> None:
        source = wavy_surface(6, 6)
        with self.assertRaisesRegex(ValueError, "vertex_index"):
            SurfaceMapper(source, vertex_index="octree")


class TransferTests(unittest.TestCase):
    def test_identity_preserves_categorical_label(self) -> None:
        source = plane(5, 6)
        label = np.arange(30, dtype=np.uint16).reshape(5, 6)

        output, valid, distance, stats = transfer_array(
            source,
            source,
            label,
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=3,
        )

        np.testing.assert_array_equal(output, label)
        np.testing.assert_array_equal(valid, np.full(label.shape, 255, np.uint8))
        self.assertIsNone(distance)
        self.assertEqual(stats.mapped_pixels, label.size)
        self.assertAlmostEqual(stats.distance_max, 0.0)

    def test_different_canvas_size_uses_3d_correspondence(self) -> None:
        source = plane(6, 7)
        target = plane(3, 4, x_offset=2.0, y_offset=1.0)
        label = (
            np.arange(6, dtype=np.uint16)[:, None] * 100
            + np.arange(7, dtype=np.uint16)[None, :]
        )

        output, valid, _, _ = transfer_array(
            source,
            target,
            label,
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=2,
        )

        expected = label[np.ix_([1, 2, 3], [2, 3, 4, 5])]
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(valid, np.full(target.shape, 255, np.uint8))

    def test_affine_maps_source_into_target_volume(self) -> None:
        source = plane(5, 6)
        target = Surface(
            x=source.x * 2.0 + 30.0,
            y=source.y * 2.0 - 12.0,
            z=source.z * 2.0 + 4.0,
        )
        matrix = np.array(
            [
                [2.0, 0.0, 0.0, 30.0],
                [0.0, 2.0, 0.0, -12.0],
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        label = np.arange(30, dtype=np.uint8).reshape(5, 6)

        output, valid, _, _ = transfer_array(
            source,
            target,
            label,
            affine=matrix,
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=3,
        )

        np.testing.assert_array_equal(output, label)
        np.testing.assert_array_equal(valid, np.full(label.shape, 255, np.uint8))

    def test_rejects_interpolated_uv_jump_between_nearby_folds(self) -> None:
        rows = np.broadcast_to(
            np.arange(2, dtype=np.float32)[:, None], (2, 12)
        ).copy()
        folded_x = np.concatenate(
            (
                np.arange(6, dtype=np.float32),
                np.arange(5, -1, -1, dtype=np.float32),
            )
        )
        folded_z = np.concatenate(
            (
                np.full(6, 10.0, dtype=np.float32),
                np.full(6, 10.6, dtype=np.float32),
            )
        )
        source = Surface(
            x=np.broadcast_to(folded_x, (2, 12)).copy(),
            y=rows,
            z=np.broadcast_to(folded_z, (2, 12)).copy(),
        )
        target = Surface(
            x=np.ones((2, 2), dtype=np.float32),
            y=np.broadcast_to(
                np.arange(2, dtype=np.float32)[:, None], (2, 2)
            ).copy(),
            z=np.broadcast_to(
                np.asarray([10.0, 10.6], dtype=np.float32), (2, 2)
            ).copy(),
        )
        label = np.broadcast_to(
            np.arange(12, dtype=np.uint8), source.shape
        ).copy()

        output, valid, _, stats = transfer_array(
            source,
            target,
            label,
            output_shape=(2, 18),
            max_distance=0.2,
            nearest_vertices=4,
            tile_size=9,
        )

        # Target vertices touch source columns 1 and 10, but interpolating
        # between those UVs crosses the distant hairpin tip. The fabricated
        # middle band must not be treated as a valid surface match.
        self.assertFalse(
            np.any((valid > 0) & (output > 1) & (output < 10))
        )
        self.assertTrue(np.any(valid == 0))
        self.assertLess(stats.mapped_pixels, output.size)

    def test_label_canvas_offset_shifts_sampling_and_rejects_outside(
        self,
    ) -> None:
        source = plane(6, 8)
        label = np.arange(48, dtype=np.uint8).reshape(6, 8)

        output, valid, _, _ = transfer_array(
            source,
            source,
            label,
            label_offset_yx=(1.0, -2.0),
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=4,
        )

        # Label pixel (i, j) depicts canvas (i + 1, j - 2), so canvas
        # position (r, c) reads label[r - 1, c + 2]; positions whose
        # corrected index leaves the raster become invalid, not clamped.
        np.testing.assert_array_equal(valid[0, :], np.zeros(8, np.uint8))
        np.testing.assert_array_equal(
            valid[:, 6:], np.zeros((6, 2), np.uint8)
        )
        np.testing.assert_array_equal(
            valid[1:, :6], np.full((5, 6), 255, np.uint8)
        )
        np.testing.assert_array_equal(output[1:, :6], label[:5, 2:])

    def test_source_validity_is_composed_into_output_validity(self) -> None:
        source = plane(4, 5)
        label = np.arange(20, dtype=np.uint8).reshape(source.shape)
        source_validity = np.full(source.shape, 255, dtype=np.uint8)
        source_validity[1, 2] = 0

        output, valid, _, stats = transfer_array(
            source,
            source,
            label,
            source_validity=source_validity,
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=3,
        )

        self.assertEqual(output[1, 2], 0)
        self.assertEqual(valid[1, 2], 0)
        self.assertEqual(stats.mapped_pixels, label.size - 1)
        np.testing.assert_array_equal(
            valid[source_validity > 0],
            np.full(label.size - 1, 255, dtype=np.uint8),
        )

    def test_auto_affine_direction_selects_forward(self) -> None:
        source = plane(7, 8, x_offset=4.0, y_offset=3.0)
        matrix = np.array(
            [
                [0.25, 0.0, 0.0, 100.0],
                [0.0, 0.25, 0.0, 80.0],
                [0.0, 0.0, 0.25, 50.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target = Surface(
            x=source.x * 0.25 + 100.0,
            y=source.y * 0.25 + 80.0,
            z=source.z * 0.25 + 50.0,
        )

        choice = choose_affine_direction(
            source, target, matrix, direction="auto", sample_limit=1000
        )

        self.assertEqual(choice.direction, "forward")
        np.testing.assert_allclose(choice.matrix, matrix)
        self.assertIsNotNone(choice.forward_median)
        self.assertAlmostEqual(float(choice.forward_median), 0.0)

    def test_output_shape_is_inferred_from_source_label_sampling(self) -> None:
        source = plane(5, 6, scale_yx=(0.25, 0.5))
        target = plane(10, 9, scale_yx=(0.5, 0.25))

        shape = infer_output_shape(source, (20, 12), target)

        self.assertEqual(shape, (20, 36))

    def test_output_shape_snaps_cropped_raster_to_native_target_canvas(
        self,
    ) -> None:
        # Paris4 w00 ratios: the annotation raster (32249x51380) disagrees
        # with the source TIFXYZ canvas (32281x51345) because renders crop
        # independently, not because the render scale differs. The output
        # must land on the native target canvas (31960x51960), not a
        # 31928x51995 pseudo-canvas scaled by the raster disagreement.
        source = plane(10, 10, scale_yx=(10 / 32281, 10 / 51345))
        target = plane(10, 10, scale_yx=(10 / 31960, 10 / 51960))

        shape = infer_output_shape(source, (32249, 51380), target)

        self.assertEqual(shape, (31960, 51960))

    def test_output_shape_keeps_genuinely_scaled_labels(self) -> None:
        # A quarter-resolution render must not snap to the full canvas.
        source = plane(10, 10, scale_yx=(10 / 32000, 10 / 51000))
        target = plane(10, 10, scale_yx=(10 / 32000, 10 / 52000))

        shape = infer_output_shape(source, (8000, 12750), target)

        self.assertEqual(shape, (8000, 13000))

    def test_output_shape_uses_cpp_half_up_rounding(self) -> None:
        source = plane(2, 2)
        target = plane(3, 3, scale_yx=(2.0, 2.0))

        shape = infer_output_shape(source, (2, 2), target)

        self.assertEqual(shape, (2, 2))

    def test_loads_three_by_four_affine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transform.json"
            path.write_text(
                json.dumps(
                    {
                        "transformation_matrix": [
                            [1, 0, 0, 2],
                            [0, 1, 0, 3],
                            [0, 0, 1, 4],
                        ]
                    }
                ),
                encoding="utf-8",
            )

            matrix = load_affine(path)

        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1])



class SeamFillTests(unittest.TestCase):
    def test_fill_seams_continues_field_with_distinct_validity(self) -> None:
        source = plane(24, 24)
        target = plane(24, 24)
        # A band of target vertices lifted beyond max_distance: the strict
        # transfer rejects them; --fill-seams must continue the UV field
        # across the band and mark those pixels with validity 128.
        target.z[8:12, :] += 5.0
        rng = np.random.default_rng(6)
        label = rng.integers(1, 255, (24, 24), dtype=np.uint8)

        strict_output, strict_valid, _, _ = transfer_array(source, target, label)
        output, valid, _, stats = transfer_array(
            source, target, label, fill_seams=True
        )

        rejected = strict_valid == 0
        self.assertGreater(int(rejected.sum()), 0)
        # Strictly accepted pixels are untouched.
        np.testing.assert_array_equal(valid[~rejected], strict_valid[~rejected])
        np.testing.assert_array_equal(output[~rejected], strict_output[~rejected])
        # Rejected pixels are filled and flagged as interpolated, not measured.
        self.assertTrue(np.all(valid[rejected] == 128))
        self.assertEqual(stats.seam_filled_pixels, int(rejected.sum()))
        self.assertIn("seam_filled_pixels", stats.as_dict())
        # On a plane the continued field reproduces the true linear map
        # except within a few columns of the raster border, where the
        # edge-padded relaxation bends the field slightly. Interior filled
        # pixels must match exactly; overall the fill stays close.
        interior = np.zeros_like(rejected)
        interior[:, 4:-4] = True
        np.testing.assert_array_equal(
            output[rejected & interior], label[rejected & interior]
        )
        match_fraction = float((output[rejected] == label[rejected]).mean())
        self.assertGreaterEqual(match_fraction, 0.85)

    def test_seam_filled_validity_survives_a_second_stage(self) -> None:
        source = plane(24, 24)
        target = plane(24, 24)
        target.z[8:12, :] += 5.0
        rng = np.random.default_rng(6)
        label = rng.integers(1, 255, (24, 24), dtype=np.uint8)

        stage_one_output, stage_one_valid, _, _ = transfer_array(
            source, target, label, fill_seams=True
        )
        self.assertGreater(int((stage_one_valid == 128).sum()), 0)

        # Stage two maps the stage-one raster onto an identical surface: a
        # measured mapping everywhere. Pixels whose stage-one value was
        # interpolated (128) must stay 128 in the final validity.
        final_target = plane(24, 24)
        final_target.z[8:12, :] += 5.0
        _, final_valid, _, stats = transfer_array(
            target,
            final_target,
            stage_one_output,
            source_validity=stage_one_valid,
        )

        measured = final_valid == 255
        inherited = final_valid == 128
        self.assertGreater(int(measured.sum()), 0)
        self.assertGreater(int(inherited.sum()), 0)
        self.assertEqual(stats.inherited_filled_pixels, int(inherited.sum()))
        self.assertIn("inherited_filled_pixels", stats.as_dict())
        # The identity mapping preserves pixel positions, so the inherited
        # mask must be exactly the stage-one interpolated mask.
        np.testing.assert_array_equal(inherited, stage_one_valid == 128)

    def test_fill_seams_off_leaves_holes(self) -> None:
        source = plane(24, 24)
        target = plane(24, 24)
        target.z[8:12, :] += 5.0
        label = np.full((24, 24), 9, dtype=np.uint8)

        _, valid, _, stats = transfer_array(source, target, label)

        self.assertGreater(int((valid == 0).sum()), 0)
        self.assertEqual(stats.seam_filled_pixels, 0)


class ParallelAndCacheTests(unittest.TestCase):
    @staticmethod
    def _case():
        source = plane(24, 24)
        target = plane(24, 24)
        target.z[8:12, :] += 5.0
        rng = np.random.default_rng(7)
        label = rng.integers(1, 255, (24, 24), dtype=np.uint8)
        return source, target, label

    def test_workers_do_not_change_outputs_or_stats(self) -> None:
        source, target, label = self._case()
        kwargs = dict(
            fill_seams=True, tile_size=8, query_batch_size=64
        )
        serial_out, serial_valid, _, serial_stats = transfer_array(
            source, target, label, workers=1, **kwargs
        )
        thread_out, thread_valid, _, thread_stats = transfer_array(
            source, target, label, workers=4, **kwargs
        )
        np.testing.assert_array_equal(thread_out, serial_out)
        np.testing.assert_array_equal(thread_valid, serial_valid)
        self.assertEqual(thread_stats.as_dict(), serial_stats.as_dict())

    def test_additional_labels_share_geometry_without_changing_outputs(self) -> None:
        source, target, label = self._case()
        second = np.asarray(255 - label, dtype=np.uint8)
        expected_first, expected_valid, _, expected_stats = transfer_array(
            source, target, label, fill_seams=True, workers=1
        )
        expected_second, second_valid, _, second_stats = transfer_array(
            source, target, second, fill_seams=True, workers=1
        )
        first_output = np.zeros_like(expected_first)
        second_output = np.zeros_like(expected_second)

        actual_first, actual_valid, _, actual_stats = transfer_array(
            source,
            target,
            label,
            output=first_output,
            additional_source_labels=[second],
            additional_outputs=[second_output],
            fill_seams=True,
            workers=1,
        )

        np.testing.assert_array_equal(actual_first, expected_first)
        np.testing.assert_array_equal(second_output, expected_second)
        np.testing.assert_array_equal(actual_valid, expected_valid)
        np.testing.assert_array_equal(second_valid, expected_valid)
        self.assertEqual(actual_stats.as_dict(), expected_stats.as_dict())
        self.assertEqual(second_stats.as_dict(), expected_stats.as_dict())

    def test_streamed_tiles_equal_materialized_outputs(self) -> None:
        source, target, label = self._case()
        second = np.asarray(255 - label, dtype=np.uint8)
        expected_first, expected_valid, _, expected_stats = transfer_array(
            source,
            target,
            label,
            additional_source_labels=[second],
            additional_outputs=[np.zeros_like(label)],
            fill_seams=True,
            tile_size=7,
            workers=3,
        )
        expected_second = np.zeros_like(label)
        transfer_array(
            source,
            target,
            second,
            output=expected_second,
            fill_seams=True,
            tile_size=7,
            workers=1,
        )
        streamed_first = np.zeros_like(label)
        streamed_second = np.zeros_like(label)
        streamed_valid = np.zeros_like(label)

        def receive(bounds, labels, validity):
            y0, y1, x0, x1 = bounds
            streamed_first[y0:y1, x0:x1] = labels[0]
            streamed_second[y0:y1, x0:x1] = labels[1]
            streamed_valid[y0:y1, x0:x1] = validity

        actual_first, actual_valid, _, actual_stats = transfer_array(
            source,
            target,
            label,
            additional_source_labels=[second],
            fill_seams=True,
            tile_size=7,
            workers=3,
            materialize_output=False,
            tile_callback=receive,
        )

        self.assertIsNone(actual_first)
        self.assertIsNone(actual_valid)
        np.testing.assert_array_equal(streamed_first, expected_first)
        np.testing.assert_array_equal(streamed_second, expected_second)
        np.testing.assert_array_equal(streamed_valid, expected_valid)
        self.assertEqual(actual_stats.as_dict(), expected_stats.as_dict())

    def test_uv_cache_reuse_and_invalidation(self) -> None:
        import tempfile
        from pathlib import Path

        source, target, label = self._case()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "uv.npz"
            first_out, first_valid, _, _ = transfer_array(
                source, target, label, uv_cache=cache
            )
            self.assertTrue(cache.exists())
            with np.load(cache, allow_pickle=False) as data:
                cached_meta = str(data["meta"])

            cached_out, cached_valid, _, _ = transfer_array(
                source, target, label, uv_cache=cache
            )
            np.testing.assert_array_equal(cached_out, first_out)
            np.testing.assert_array_equal(cached_valid, first_valid)
            with np.load(cache, allow_pickle=False) as data:
                self.assertEqual(str(data["meta"]), cached_meta)

            # A different matching configuration must not reuse the cache.
            transfer_array(
                source, target, label, uv_cache=cache, max_distance=0.4
            )
            with np.load(cache, allow_pickle=False) as data:
                self.assertNotEqual(str(data["meta"]), cached_meta)
                self.assertIn('"max_distance": 0.4', str(data["meta"]))


if __name__ == "__main__":
    unittest.main()
