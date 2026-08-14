from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesuvius.tifxyz_label_transfer.core import Surface, SurfaceMapper
from vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence import ZarrLevel
from vesuvius.tifxyz_label_transfer.self_render_tifxyz import (
    RawChunkSampler,
    _annotation_offsets,
    _canvas_shape_check,
    _measurement_is_translation,
    _resolution_from_raw_path,
    _source_canvas_measurements,
    _source_reference_render,
    _to_uint8,
    _source_raw_volume_remote,
    build_parser,
    mapped_source_tile_geometry,
    matched_offsets,
    required_chunks,
    surface_tile_geometry,
)


def plane_surface(height: int, width: int) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return Surface(
        x=cols,
        y=rows,
        z=np.full((height, width), 2.0, dtype=np.float32),
    )


def raw_info() -> ZarrLevel:
    return ZarrLevel(
        remote="remote:raw.zarr",
        level="0",
        attrs={},
        metadata={
            "shape": [4, 4, 4],
            "chunks": [4, 4, 4],
            "dtype": "|u1",
            "compressor": None,
            "filters": None,
            "order": "C",
            "dimension_separator": "/",
        },
        level_zero_metadata={"shape": [4, 4, 4]},
        scale_zyx=(1.0, 1.0, 1.0),
    )


class _FakeSampler:
    def __init__(self, dtype: str) -> None:
        self.info = type("Info", (), {"metadata": {"dtype": dtype}})()


class ToUint8Tests(unittest.TestCase):
    def test_uint16_values_scale_instead_of_wrapping(self) -> None:
        sampler = _FakeSampler("<u2")
        values = np.asarray([0.0, 257.0, 65535.0])
        np.testing.assert_array_equal(
            _to_uint8(values, sampler), np.asarray([0, 1, 255], dtype=np.uint8)
        )

    def test_uint8_values_pass_through_with_clipping(self) -> None:
        sampler = _FakeSampler("|u1")
        values = np.asarray([-3.0, 0.4, 254.6, 300.0])
        np.testing.assert_array_equal(
            _to_uint8(values, sampler), np.asarray([0, 0, 255, 255], dtype=np.uint8)
        )


class CanvasShapeCheckTests(unittest.TestCase):
    def test_pure_downsample_has_no_anisotropy(self) -> None:
        check = _canvas_shape_check(
            np.asarray([32000.0, 48000.0]), np.asarray([8000.0, 12000.0])
        )
        self.assertEqual(check["assumed_scale_yx"], [4.0, 4.0])
        self.assertEqual(check["scale_anisotropy"], 0.0)

    def test_crop_disagreement_shows_as_anisotropy(self) -> None:
        # paris4 stage 1: raster 32249x51380 vs TIFXYZ canvas 32281x51345
        check = _canvas_shape_check(
            np.asarray([32281.0, 51345.0]), np.asarray([32249.0, 51380.0])
        )
        self.assertGreater(check["scale_anisotropy"], 1e-3)
        self.assertEqual(
            check["tifxyz_full_resolution_shape_yx"], [32281, 51345]
        )
        self.assertEqual(
            check["annotation_render_shape_yx"], [32249, 51380]
        )


class SourceCanvasMeasurementTests(unittest.TestCase):
    def test_shipped_max_can_approve_when_center_has_no_usable_tiles(
        self,
    ) -> None:
        measurement = {
            "shift_yx": [1.0, -2.0],
            "shift_field": {
                "inlier_tiles": 4,
                "max_corner_drift_px": 0.25,
            },
        }
        args = SimpleNamespace(
            tile_size=8,
            min_coverage=0.6,
            max_tiles=4,
            max_corner_drift_px=1.5,
            maximum_source_disagreement_full_px=3.0,
        )
        with patch(
            "vesuvius.tifxyz_label_transfer.self_render_tifxyz.measure_render_shift",
            side_effect=[ValueError("no usable tiles"), measurement],
        ):
            result = _source_canvas_measurements(
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.asarray([16, 16]),
                args,
                allow_annotation_only=True,
            )

        self.assertTrue(result["approved"])
        self.assertEqual(result["approval_basis"], "annotation-maximum-only")
        self.assertEqual(result["annotation_offset_full"], [4.0, -8.0])
        self.assertIsNone(result["center_offset_full"])
        self.assertIn("no usable tiles", result["center_error"])

    def test_center_failure_cannot_approve_without_independent_max(self) -> None:
        measurement = {
            "shift_yx": [1.0, -2.0],
            "shift_field": {
                "inlier_tiles": 4,
                "max_corner_drift_px": 0.25,
            },
        }
        args = SimpleNamespace(
            tile_size=8,
            min_coverage=0.6,
            max_tiles=4,
            max_corner_drift_px=1.5,
            maximum_source_disagreement_full_px=3.0,
        )
        with patch(
            "vesuvius.tifxyz_label_transfer.self_render_tifxyz.measure_render_shift",
            side_effect=[ValueError("no usable tiles"), measurement],
        ):
            result = _source_canvas_measurements(
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.asarray([16, 16]),
                args,
            )

        self.assertFalse(result["approved"])
        self.assertEqual(result["approval_basis"], "not-approved")


class SurfaceGeometryTests(unittest.TestCase):
    def test_source_only_mode_is_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--case-dir",
                "/tmp/case",
                "--source-only",
                "--source-surface-zarr",
                "public:surface.zarr",
            ]
        )
        self.assertTrue(args.source_only)
        self.assertEqual(args.source_surface_zarr, "public:surface.zarr")

    def test_source_center_is_used_when_annotation_max_is_absent(self) -> None:
        center = Path("renders/offset-evidence/source-center.tif")
        path, kind = _source_reference_render(
            {"annotation_render": None}, center
        )
        self.assertEqual(path, center)
        self.assertEqual(kind, "surface-volume-center")

    def test_source_raw_volume_comes_from_surface_provenance(self) -> None:
        info = raw_info()
        info = ZarrLevel(
            **{
                **info.__dict__,
                "attrs": {
                    "source_zarr": (
                        "/volpkgs/volumes/s3_volumes/esrf/20250717/"
                        "scan-2.403um.zarr/"
                    )
                },
            }
        )
        remote, provenance = _source_raw_volume_remote(
            info, "remote:scrollprize-volumes", None
        )
        self.assertEqual(
            remote,
            "remote:scrollprize-volumes/esrf/20250717/scan-2.403um.zarr",
        )
        self.assertIn("source surface Zarr", provenance)

    def test_explicit_raw_volume_override_wins(self) -> None:
        info = raw_info()
        info = ZarrLevel(
            **{
                **info.__dict__,
                "attrs": {"source_zarr": "/s3_volumes/esrf/scan-2um.zarr/"},
            }
        )
        remote, provenance = _source_raw_volume_remote(
            info, "any-remote:volumes", "mine:private/scan-2um.zarr/"
        )
        self.assertEqual(remote, "mine:private/scan-2um.zarr")
        self.assertEqual(provenance, "command line")

    def test_provenance_remap_without_raw_root_is_actionable(self) -> None:
        info = raw_info()
        info = ZarrLevel(
            **{
                **info.__dict__,
                "attrs": {"source_zarr": "/s3_volumes/esrf/scan-2um.zarr/"},
            }
        )
        with self.assertRaises(ValueError) as caught:
            _source_raw_volume_remote(info, None, None)
        self.assertIn("--source-raw-rclone-root", str(caught.exception))
        self.assertIn("--source-raw-volume", str(caught.exception))

    def test_annotation_offsets_include_slice_step(self) -> None:
        info = raw_info()
        info = ZarrLevel(
            **{
                **info.__dict__,
                "attrs": {"slice_step": 2.0},
                "level_zero_metadata": {"shape": [5, 4, 4]},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "render_max_1_3.tif"
            annotation.touch()
            self.assertEqual(
                _annotation_offsets(annotation, info), [-2.0, 0.0, 2.0]
            )

    def test_resolution_is_inferred_from_raw_provenance(self) -> None:
        self.assertEqual(
            _resolution_from_raw_path("remote:bucket/scan-2.403um.zarr"),
            2.403,
        )

    def test_translation_requires_three_inliers_and_bounded_drift(self) -> None:
        measurement = {
            "shift_field": {"inlier_tiles": 3, "max_corner_drift_px": 1.0}
        }
        self.assertTrue(_measurement_is_translation(measurement, 1.5))
        measurement["shift_field"]["inlier_tiles"] = 2
        self.assertFalse(_measurement_is_translation(measurement, 1.5))
        measurement["shift_field"]["inlier_tiles"] = 3
        measurement["shift_field"]["max_corner_drift_px"] = 2.0
        self.assertFalse(_measurement_is_translation(measurement, 1.5))

    def test_plane_geometry_uses_xyz_and_left_hand_normal(self) -> None:
        surface = plane_surface(6, 8)
        xyz, normals, valid = surface_tile_geometry(
            surface, (6, 8), (1, 5, 1, 7)
        )

        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(xyz[..., 2], 2.0)
        np.testing.assert_allclose(
            normals,
            np.broadcast_to([0.0, 0.0, 1.0], normals.shape),
            atol=1e-6,
        )

    def test_required_chunks_include_trilinear_neighbours(self) -> None:
        xyz = np.asarray([[[1.5, 1.5, 1.5]]])
        normals = np.asarray([[[0.0, 0.0, 1.0]]])
        chunks = required_chunks(
            xyz,
            normals,
            np.asarray([[True]]),
            [0.0],
            raw_info(),
        )
        self.assertEqual(chunks, {(0, 0, 0)})

    def test_smaller_source_surface_maps_only_its_target_overlap(self) -> None:
        source = plane_surface(4, 4)
        source.x += 2.0
        source.y += 2.0
        target = plane_surface(8, 8)
        mapper = SurfaceMapper(source, nearest_vertices=4)
        rows, cols, _, valid = mapper.build_target_uv_map(
            target, max_distance=0.01
        )

        geometry = mapped_source_tile_geometry(
            source,
            target,
            rows,
            cols,
            valid,
            (8, 8),
            (2, 6, 2, 6),
            np.eye(4),
            0.01,
        )

        self.assertGreater(int(geometry[-1].sum()), 0)
        self.assertLess(int(valid.sum()), int(target.valid.sum()))

    def test_matched_offsets_preserve_physical_thickness(self) -> None:
        offsets = matched_offsets([-2, -1, 0, 1, 2], 2.0, 4.0)
        self.assertAlmostEqual(min(offsets) * 4.0, -4.0)
        self.assertAlmostEqual(max(offsets) * 4.0, 4.0)

    def test_matched_offsets_do_not_duplicate_a_center_only_sample(self) -> None:
        self.assertEqual(matched_offsets([0.0], 2.4, 2.4), [0.0])


class RawSamplerTests(unittest.TestCase):
    def test_trilinear_sampling_matches_eight_corner_average(self) -> None:
        volume = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "0" / "0" / "0"
            chunk_path.parent.mkdir(parents=True)
            chunk_path.write_bytes(volume.tobytes())
            sampler = RawChunkSampler(raw_info(), Path(directory))
            actual = sampler.sample(
                np.asarray([[1.5, 1.5, 1.5]], dtype=np.float64)
            )

        expected = volume[1:3, 1:3, 1:3].mean()
        np.testing.assert_allclose(actual, [expected], atol=1e-6)

    def test_block_sampling_matches_scalar_sampler(self) -> None:
        volume = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "0" / "0" / "0"
            chunk_path.parent.mkdir(parents=True)
            chunk_path.write_bytes(volume.tobytes())
            sampler = RawChunkSampler(raw_info(), Path(directory))
            block, origin = sampler.load_block({(0, 0, 0)})
            points = np.asarray([[1.25, 1.5, 1.75]], dtype=np.float64)

            scalar = sampler.sample(points)
            blocked = sampler.sample_block(points, block, origin)

        np.testing.assert_allclose(blocked, scalar, atol=1e-6)

    def test_block_sampling_clamps_to_logical_shape_not_chunk_padding(self) -> None:
        info = raw_info()
        info = ZarrLevel(
            **{
                **info.__dict__,
                "metadata": {**info.metadata, "shape": [3, 3, 3]},
                "level_zero_metadata": {"shape": [3, 3, 3]},
            }
        )
        volume = np.full((4, 4, 4), 255, dtype=np.uint8)
        volume[:3, :3, :3] = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "0" / "0" / "0"
            chunk_path.parent.mkdir(parents=True)
            chunk_path.write_bytes(volume.tobytes())
            sampler = RawChunkSampler(info, Path(directory))
            block, origin = sampler.load_block({(0, 0, 0)})
            actual = sampler.sample_block(
                np.asarray([[4.0, 4.0, 4.0]]), block, origin
            )

        np.testing.assert_allclose(actual, [volume[2, 2, 2]])


if __name__ == "__main__":
    unittest.main()
