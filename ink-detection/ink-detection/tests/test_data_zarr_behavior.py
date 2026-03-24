from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.zarr_io import ZarrSegmentVolume


def _write_zarr_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=tuple(array.shape), dtype=array.dtype)
    store[:] = array


class StandaloneZarrReaderTests(unittest.TestCase):
    def test_zarr_segment_volume_uses_full_layer_range_when_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            volume = np.stack([np.full((6, 6), layer_value, dtype=np.uint8) for layer_value in range(6)], axis=0)
            _write_zarr_array(segment_dir / "segA.zarr", volume)
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.full((6, 6), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((6, 6), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            reader = ZarrSegmentVolume(
                layout,
                "segA",
                layer_range=None,
                in_channels=4,
            )

            patch = reader.read_patch(1, 3, 2, 4)

            self.assertEqual(reader.layer_range, (0, 6))
            self.assertEqual(tuple(patch.shape), (2, 2, 4))
            self.assertEqual(patch[0, 0].tolist(), [1, 2, 3, 4])

    def test_zarr_segment_volume_reads_center_cropped_and_reversed_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            volume = np.stack([np.full((6, 6), layer_value, dtype=np.uint8) for layer_value in range(6)], axis=0)
            _write_zarr_array(segment_dir / "segA.zarr", volume)
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.full((6, 6), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((6, 6), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            reader = ZarrSegmentVolume(
                layout,
                "segA",
                layer_range=(0, 6),
                reverse_layers=True,
                in_channels=4,
            )

            patch = reader.read_patch(1, 3, 2, 4)

            self.assertEqual(reader.shape, (256, 256, 4))
            self.assertTrue(reader._reverse_layer_read)
            self.assertEqual(tuple(patch.shape), (2, 2, 4))
            self.assertEqual(patch[0, 0].tolist(), [4, 3, 2, 1])

    def test_zarr_segment_volume_reads_fresh_zarr_data_on_each_patch_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            volume = np.stack([np.full((6, 6), layer_value, dtype=np.uint8) for layer_value in (10, 20, 30, 40)], axis=0)
            updated_volume = np.stack([np.full((6, 6), layer_value, dtype=np.uint8) for layer_value in (50, 60, 70, 80)], axis=0)
            _write_zarr_array(segment_dir / "segA.zarr", volume)

            layout = NestedZarrLayout(root)
            reader = ZarrSegmentVolume(
                layout,
                "segA",
                layer_range=(0, 4),
                in_channels=4,
            )

            first_patch = reader.read_patch(1, 3, 2, 4)
            zarr.open(str(segment_dir / "segA.zarr"), mode="r+")[:] = updated_volume

            second_patch = reader.read_patch(1, 3, 2, 4)

            self.assertEqual(first_patch[0, 0].tolist(), [10, 20, 30, 40])
            self.assertEqual(second_patch[0, 0].tolist(), [50, 60, 70, 80])

if __name__ == "__main__":
    unittest.main()
