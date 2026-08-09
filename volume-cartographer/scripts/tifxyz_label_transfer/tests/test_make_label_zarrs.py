from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tifxyz_label_transfer.make_label_zarrs import (  # noqa: E402
    CHUNK_SHAPE,
    LABEL_SLICE,
    VOLUME_DEPTH,
    convert_label_tiff,
    write_label_zarr,
)


def _checker_label(height: int = 300, width: int = 260) -> np.ndarray:
    rows = np.arange(height)[:, None] // 16
    cols = np.arange(width)[None, :] // 16
    return np.where((rows + cols) % 2 == 0, 255, 0).astype(np.uint8)


class MakeLabelZarrsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def test_pyramid_matches_koine_conventions(self) -> None:
        label = _checker_label()
        output = self.tmp_path / "label.zarr"
        write_label_zarr(label, output, levels=4)

        group = zarr.open_group(str(output), mode="r")
        multiscales = group.attrs["multiscales"]
        self.assertEqual(multiscales[0]["version"], "0.4")
        self.assertEqual(
            [axis["name"] for axis in multiscales[0]["axes"]], ["z", "y", "x"]
        )
        self.assertEqual(
            [d["path"] for d in multiscales[0]["datasets"]],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            multiscales[0]["datasets"][2]["coordinateTransformations"][0][
                "scale"
            ],
            [1.0, 4.0, 4.0],
        )

        expected = label
        for level in range(4):
            dataset = group[str(level)]
            self.assertEqual(
                dataset.shape,
                (VOLUME_DEPTH, expected.shape[0], expected.shape[1]),
            )
            self.assertEqual(dataset.chunks, CHUNK_SHAPE)
            self.assertEqual(dataset.dtype, label.dtype)
            self.assertEqual(
                dataset.attrs["_ARRAY_DIMENSIONS"], ["z", "y", "x"]
            )
            self.assertEqual(dataset.compressor.cname, "zstd")
            self.assertEqual(dataset.compressor.clevel, 3)
            data = np.asarray(dataset)
            np.testing.assert_array_equal(data[LABEL_SLICE], expected)
            other = np.delete(data, LABEL_SLICE, axis=0)
            self.assertFalse(other.any())
            expected = expected[::2, ::2]

        # Odd dimensions follow the koine (n + 1) // 2 recurrence.
        self.assertEqual(group["1"].shape[1], (label.shape[0] + 1) // 2)

    def test_dimension_separator_is_slash(self) -> None:
        output = self.tmp_path / "label.zarr"
        write_label_zarr(_checker_label(64, 64), output, levels=1)
        self.assertTrue(
            (output / "0" / str(LABEL_SLICE // CHUNK_SHAPE[0])).exists()
            or (output / "0" / "0").is_dir()
        )

    def test_convert_label_tiff_provenance_and_default_output(self) -> None:
        label = _checker_label(128, 96)
        input_path = self.tmp_path / "supervision-2.399um.tif"
        tifffile.imwrite(input_path, label)
        report_path = self.tmp_path / "supervision-2.399um.report.json"
        report_path.write_text('{"output_shape": [128, 96]}')

        output = convert_label_tiff(input_path, levels=2)
        self.assertEqual(output, self.tmp_path / "supervision-2.399um.zarr")

        group = zarr.open_group(str(output), mode="r")
        self.assertEqual(group.attrs["canvas_size"], [128, 96])
        self.assertEqual(group.attrs["source_image"], str(input_path))
        self.assertEqual(group.attrs["transfer_report"], str(report_path))
        np.testing.assert_array_equal(
            np.asarray(group["0"])[LABEL_SLICE], label
        )

        with self.assertRaises(FileExistsError):
            convert_label_tiff(input_path, levels=2)
        convert_label_tiff(input_path, levels=2, overwrite=True)

    def test_rejects_non_2d_input(self) -> None:
        input_path = self.tmp_path / "bad_inklabels.tif"
        tifffile.imwrite(input_path, np.zeros((2, 8, 8, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "2D"):
            convert_label_tiff(input_path)


if __name__ == "__main__":
    unittest.main()
