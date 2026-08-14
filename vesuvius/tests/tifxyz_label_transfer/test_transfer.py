from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

from vesuvius.tifxyz_label_transfer.core import MappingStats, Surface
from vesuvius.tifxyz_label_transfer import transfer


def plane(*, x_offset: float = 0.0) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(2, dtype=np.float32),
        np.arange(2, dtype=np.float32),
        indexing="ij",
    )
    return Surface(
        x=cols + x_offset,
        y=rows,
        z=np.full((2, 2), 10.0, dtype=np.float32),
    )


class PipelineTests(unittest.TestCase):
    def test_stage_two_consumes_stage_one_validity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intermediate = root / "updated.tif"
            output = root / "final.tif"
            args = argparse.Namespace(
                intermediate_output=str(intermediate),
                output=str(output),
                dry_run=False,
                overwrite=False,
                old_tifxyz="old.tifxyz",
                updated_tifxyz="updated.tifxyz",
                target_tifxyz="target.tifxyz",
                label="labels.tif",
                additional_labels=[
                    (
                        "validation.zarr",
                        str(root / "validation-updated.tif"),
                        str(root / "validation-final.tif"),
                    )
                ],
                affine="registration.json",
                label_canvas_offset=(30.0, -48.0),
                same_volume_affine="frame-shift.json",
                same_volume_affine_direction="forward",
                same_volume_max_distance=None,
                cross_volume_max_distance=None,
                updated_reference=None,
                target_reference=None,
                intermediate_shape=None,
                output_shape=None,
                affine_direction="auto",
                affine_sample_points=100,
                preflight_sample_points=32,
                minimum_mapping_coverage=0.05,
                nearest_vertices=8,
                tile_size=32,
                query_batch_size=1024,
                fill_value=0,
                ignore_tifxyz_mask=False,
            )
            intermediate_valid = root / "updated.valid.tif"
            stage_one = {"valid_output": str(intermediate_valid)}
            stage_two = {"mapping": {}}

            with mock.patch.object(
                transfer,
                "run_single",
                side_effect=[stage_one, stage_two],
            ) as run_single:
                result = transfer.run_pipeline(args)

        self.assertEqual(result["stage_two"], stage_two)
        stage_one_args = run_single.call_args_list[0].args[0]
        self.assertEqual(stage_one_args.affine, "frame-shift.json")
        self.assertEqual(stage_one_args.affine_direction, "forward")
        self.assertEqual(stage_one_args.label_canvas_offset, (30.0, -48.0))
        self.assertEqual(stage_one_args.preflight_sample_points, 32)
        self.assertEqual(stage_one_args.minimum_mapping_coverage, 0.05)
        self.assertEqual(
            stage_one_args.additional_labels,
            [("validation.zarr", str(root / "validation-updated.tif"))],
        )
        stage_two_args = run_single.call_args_list[1].args[0]
        self.assertEqual(stage_two_args.affine, "registration.json")
        self.assertIsNone(stage_two_args.label_canvas_offset)
        self.assertEqual(
            stage_two_args.source_validity,
            str(intermediate_valid),
        )
        self.assertEqual(
            stage_two_args.additional_labels,
            [
                (
                    str(root / "validation-updated.tif"),
                    str(root / "validation-final.tif"),
                )
            ],
        )

    def test_final_collision_is_detected_before_stage_one_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "final.tif"
            output.touch()
            args = argparse.Namespace(
                intermediate_output=str(root / "updated.tif"),
                output=str(output),
                dry_run=False,
                overwrite=False,
            )

            with mock.patch.object(transfer, "run_single") as run_single:
                with self.assertRaises(FileExistsError):
                    transfer.run_pipeline(args)

        run_single.assert_not_called()


class CoverageGuardTests(unittest.TestCase):
    def args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            source_tifxyz="source.tifxyz",
            target_tifxyz="target.tifxyz",
            label="labels.tif",
            source_validity=None,
            label_canvas_offset=None,
            output=str(root / "output.tif"),
            affine=None,
            affine_direction="auto",
            affine_sample_points=100,
            preflight_sample_points=16,
            minimum_mapping_coverage=0.01,
            max_distance=0.1,
            nearest_vertices=2,
            tile_size=2,
            query_batch_size=16,
            fill_value=0,
            output_shape=None,
            target_reference=None,
            valid_output=None,
            distance_output=None,
            report_output=None,
            ignore_tifxyz_mask=False,
            dry_run=False,
            overwrite=False,
        )

    def test_preflight_rejects_frame_mismatch_before_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            source = plane()
            target = plane(x_offset=100.0)
            with (
                mock.patch.object(
                    transfer, "load_surface", side_effect=[source, target]
                ),
                mock.patch.object(
                    transfer,
                    "read_image",
                    return_value=np.ones((2, 2), dtype=np.uint8),
                ),
                mock.patch.object(transfer, "transfer_array") as transfer_array,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    r"different volume frames.*--stage-one-affine",
                ):
                    transfer.run_single(args, stage_name="old-to-updated")

        transfer_array.assert_not_called()
        self.assertFalse((root / "output.tif").exists())

    def test_preflight_accepts_supplied_frame_affine(self) -> None:
        source = plane()
        target = plane(x_offset=100.0)
        matrix = np.eye(4)
        matrix[0, 3] = 100.0

        report = transfer._mapping_preflight(
            source,
            target,
            matrix,
            0.1,
            nearest_vertices=2,
            sample_limit=16,
        )

        self.assertEqual(report["sampled_target_valid_vertices"], 4)
        self.assertEqual(report["mapped_within_distance"], 4)
        self.assertEqual(report["mapping_coverage"], 1.0)

    def test_run_single_streams_exact_tiff_outputs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            args.tile_size = 16
            args.minimum_mapping_coverage = 0.0
            args.rasterizer = "python"
            source = plane()
            label = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
            with (
                mock.patch.object(
                    transfer,
                    "load_surface",
                    side_effect=[source, source],
                ),
                mock.patch.object(
                    transfer,
                    "read_image",
                    return_value=label,
                ),
            ):
                report = transfer.run_single(args)

            output = root / "output.tif"
            valid = root / "output.valid.tif"
            np.testing.assert_array_equal(tifffile.imread(output), label)
            np.testing.assert_array_equal(
                tifffile.imread(valid), np.full((2, 2), 255, dtype=np.uint8)
            )
            self.assertEqual(
                report["output_storage_mode"], "streamed-compressed-tiles"
            )
            self.assertEqual(report["rasterizer"], "python")
            self.assertEqual(report["temporary_full_raster_bytes"], 0)
            self.assertLess(
                report["estimated_buffered_output_bytes"],
                report["logical_output_bytes"] * 1024,
            )

    def test_final_guard_runs_before_any_output_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            source = plane()
            empty_stats = MappingStats(
                target_pixels=4,
                target_surface_valid=4,
            )
            with (
                mock.patch.object(
                    transfer,
                    "load_surface",
                    side_effect=[source, source],
                ),
                mock.patch.object(
                    transfer,
                    "read_image",
                    return_value=np.ones((2, 2), dtype=np.uint8),
                ),
                mock.patch.object(
                    transfer,
                    "_mapping_preflight",
                    return_value={
                        "sampled_target_valid_vertices": 4,
                        "mapped_within_distance": 4,
                        "mapping_coverage": 1.0,
                    },
                ),
                mock.patch.object(
                    transfer,
                    "transfer_array",
                    return_value=(None, None, None, empty_stats),
                ),
                mock.patch.object(transfer, "write_image") as write_image,
            ):
                with self.assertRaisesRegex(
                    ValueError, "final mapping coverage"
                ):
                    transfer.run_single(args, stage_name="old-to-updated")

        write_image.assert_not_called()
        self.assertFalse((root / "output.tif").exists())


if __name__ == "__main__":
    unittest.main()
