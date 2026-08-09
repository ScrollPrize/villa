from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile
import zarr

from tifxyz_label_transfer.io import (
    load_surface,
    read_image,
    read_image_shape,
    StreamingTiffOutputs,
)


class SurfaceIoTests(unittest.TestCase):
    def test_streaming_tiffs_write_exact_edge_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.tif"
            second_path = root / "second.tif"
            permission_probe = root / "permission-probe"
            permission_probe.touch()
            expected_mode = permission_probe.stat().st_mode & 0o777
            expected_first = (
                np.arange(21 * 23, dtype=np.uint16)
                .reshape(21, 23)
                .astype(np.uint8)
            )
            expected_second = (
                np.arange(21 * 23, dtype=np.uint16).reshape(21, 23) + 1000
            )
            with StreamingTiffOutputs(
                [
                    (first_path, np.dtype(np.uint8), 9),
                    (second_path, np.dtype(np.uint16), 77),
                ],
                expected_first.shape,
                tile_size=16,
            ) as outputs:
                for y0 in range(0, 21, 16):
                    for x0 in range(0, 23, 16):
                        y1, x1 = min(21, y0 + 16), min(23, x0 + 16)
                        outputs.write_tile(
                            (y0, y1, x0, x1),
                            [
                                expected_first[y0:y1, x0:x1],
                                expected_second[y0:y1, x0:x1],
                            ],
                        )

            np.testing.assert_array_equal(
                tifffile.imread(first_path), expected_first
            )
            np.testing.assert_array_equal(
                tifffile.imread(second_path), expected_second
            )
            self.assertEqual(first_path.stat().st_mode & 0o777, expected_mode)
            self.assertFalse(list(root.glob(".*.stream-*.tif")))

    def test_streaming_tiffs_remove_partials_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.tif"
            with self.assertRaisesRegex(ValueError, "streamed tile"):
                with StreamingTiffOutputs(
                    [(output, np.dtype(np.uint8), 0)],
                    (21, 23),
                    tile_size=16,
                ) as outputs:
                    outputs.write_tile(
                        (0, 16, 0, 16),
                        [np.zeros((15, 16), dtype=np.uint8)],
                    )
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".*.stream-*.tif")))

    def test_streaming_tiffs_remove_partials_on_encoder_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.tif"
            with (
                mock.patch.object(
                    tifffile,
                    "imwrite",
                    side_effect=OSError("injected encoder failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "encoder failed"),
            ):
                with StreamingTiffOutputs(
                    [(output, np.dtype(np.uint8), 0)],
                    (21, 23),
                    tile_size=16,
                ):
                    pass
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".*.stream-*.tif")))

    def test_reads_center_slice_from_ome_zarr_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inklabels_v2.zarr"
            group = zarr.open_group(str(path), mode="w")
            array = group.create_dataset(
                "0", shape=(5, 3, 4), chunks=(5, 2, 2), dtype="u1"
            )
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            array[2] = expected

            actual = read_image(path)
            shape = read_image_shape(path)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(shape, (3, 4))

    def test_high_resolution_mask_matches_quad_surface_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "surface.tifxyz"
            path.mkdir()
            rows, cols = np.meshgrid(
                np.arange(2, dtype=np.float32),
                np.arange(3, dtype=np.float32),
                indexing="ij",
            )
            tifffile.imwrite(path / "x.tif", cols)
            tifffile.imwrite(path / "y.tif", rows)
            tifffile.imwrite(
                path / "z.tif", np.full((2, 3), 10.0, dtype=np.float32)
            )
            (path / "meta.json").write_text(
                json.dumps({"scale": [0.5, 0.25]}),
                encoding="utf-8",
            )
            mask = np.full((4, 6), 255, dtype=np.uint8)
            mask[2, 4] = 0
            tifffile.imwrite(path / "mask.tif", mask)

            surface = load_surface(path)

        self.assertEqual(surface.scale_yx, (0.25, 0.5))
        expected = np.ones((2, 3), dtype=bool)
        expected[1, 2] = False
        np.testing.assert_array_equal(surface.valid, expected)


if __name__ == "__main__":
    unittest.main()
