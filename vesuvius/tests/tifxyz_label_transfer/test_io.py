from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

from vesuvius.tifxyz_label_transfer.io import (
    load_surface,
    read_image,
    read_image_shape,
    StreamingTiffOutputs,
    TemporaryRaster,
)
from tests.zarr_utils import create_v2_group_array


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
            array = create_v2_group_array(
                path,
                "0",
                shape=(5, 3, 4),
                chunks=(5, 2, 2),
                dtype="u1",
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

    def test_readers_and_temporary_rasters_leave_no_file_mapped(self) -> None:
        """Nothing this module hands back may keep its file mapped.

        A mapping that outlives the call pins the file: on POSIX it merely
        survives unlink, but on Windows the file cannot be removed at all,
        which is what made every temporary directory in this suite fail to
        clean up. Both platforms can see it in the returned arrays.
        """

        def maps_a_file(array: object) -> bool:
            while array is not None:
                if isinstance(array, np.memmap):
                    return True
                array = getattr(array, "base", None)
            return False

        temporary = Path(tempfile.mkdtemp())
        surface_path = temporary / "surface.tifxyz"
        surface_path.mkdir()
        grid = np.zeros((2, 3), dtype=np.float32)
        for axis in ("x", "y", "z"):
            tifffile.imwrite(surface_path / f"{axis}.tif", grid, metadata=None)
        (surface_path / "meta.json").write_text(
            json.dumps({"scale": [1.0, 1.0]}), encoding="utf-8"
        )
        label_path = temporary / "label.tif"
        tifffile.imwrite(
            label_path, np.zeros((2, 3), dtype=np.uint8), metadata=None
        )

        surface = load_surface(surface_path, use_mask=False)
        label = read_image(label_path)

        for array in (surface.x, surface.y, surface.z, label):
            self.assertFalse(maps_a_file(array))

        # A caller holding one of the raster's own views must not stop its
        # owner from releasing the backing file.
        raster = TemporaryRaster(
            temporary, (2, 3), np.dtype(np.uint8), 0, ".probe-"
        )
        raster_path = raster.path
        escaped_view = raster.array[:1]
        raster.close()
        del escaped_view
        self.assertFalse(raster_path.exists())

        # Removable immediately, with the loaded arrays still referenced.
        for item in (label_path, *surface_path.iterdir()):
            item.unlink()
        surface_path.rmdir()
        temporary.rmdir()
        self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
