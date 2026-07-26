import unittest
import os
import sys
import tempfile
from pathlib import Path
import types

import numpy as np
import zarr


common_stub = types.ModuleType("common")
common_stub.load_unet = None
common_stub.unet_infer_tiled = None
sys.modules.setdefault("common", common_stub)

train_stub = types.ModuleType("train_unet_3d")
train_stub.build_model = None
sys.modules.setdefault("train_unet_3d", train_stub)

from preprocess_cos_omezarr import (
	_RollingZBand,
	_atomic_zarr_write,
	_canonical_local_tile_positions,
	_canonical_tile_positions_for_output_region,
	_cleanup_predict3d_temp_files,
	_create_omezarr,
	_grad_mag_factor_from_input_sd,
	_omezarr_chunk_group_complete,
)


class PreprocessCosOmezarrTests(unittest.TestCase):
	def test_grad_mag_factor_uses_input_scale_not_output_level(self):
		self.assertEqual(_grad_mag_factor_from_input_sd(1), 1.0)
		self.assertEqual(_grad_mag_factor_from_input_sd(4), 0.25)

	def test_output_chunk_group_requires_all_channel_chunks(self):
		with tempfile.TemporaryDirectory() as td:
			paths = []
			for name in ("gm", "nx", "ny"):
				path = str(Path(td) / f"{name}.ome.zarr")
				_create_omezarr(path, (32, 32, 32), 0, 1, 16, name)
				paths.append(path)

			block = np.ones((16, 16, 16), dtype=np.uint8)
			_atomic_zarr_write(paths[0], 0, 0, 0, 0, 16, 16, 16, block, 16)
			self.assertFalse(_omezarr_chunk_group_complete(tuple(paths), 0, 0, 0, 0, 16))
			_atomic_zarr_write(paths[1], 0, 0, 0, 0, 16, 16, 16, block, 16)
			self.assertFalse(_omezarr_chunk_group_complete(tuple(paths), 0, 0, 0, 0, 16))
			_atomic_zarr_write(paths[2], 0, 0, 0, 0, 16, 16, 16, block, 16)
			self.assertTrue(_omezarr_chunk_group_complete(tuple(paths), 0, 0, 0, 0, 16))

	def test_atomic_zarr_write_cleans_unique_temp_dir(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 1, 16, "cos")
			block = np.full((16, 16, 16), 7, dtype=np.uint8)
			_atomic_zarr_write(path, 0, 0, 0, 0, 16, 16, 16, block, 16)
			arr = zarr.open(str(Path(path) / "0"), mode="r")
			self.assertEqual(int(np.asarray(arr[0, 0, 0])), 7)
			self.assertEqual([p.name for p in Path(td).iterdir() if p.name.startswith(".tmp.")], [])

	def test_predict3d_temp_cleanup_is_output_directory_wide(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			(root / ".tmp.foo_cos.ome.zarr.0.1").mkdir()
			(root / ".tmp.foo_grad_mag.ome.zarr.0.1").mkdir()
			(root / ".predict3d_foo_acc_fine.tmp").write_text("")
			(root / ".tmp.bar_cos.ome.zarr.0.1").mkdir()
			live_current = root / f".predict3d_pid{os.getpid()}_acc_fine.tmp"
			live_current.write_text("")
			removed = _cleanup_predict3d_temp_files(root, "foo_")
			self.assertEqual(removed, 4)
			self.assertFalse((root / ".tmp.foo_cos.ome.zarr.0.1").exists())
			self.assertFalse((root / ".tmp.bar_cos.ome.zarr.0.1").exists())
			self.assertTrue(live_current.exists())
			removed = _cleanup_predict3d_temp_files(root, "foo_", remove_current_process=True)
			self.assertEqual(removed, 1)
			self.assertFalse(live_current.exists())

	def test_rolling_z_band_discards_without_cross_channel_release(self):
		with tempfile.TemporaryDirectory() as td:
			band = _RollingZBand(
				name="test", channel_count=2, y_size=2, x_size=2,
				tmp_dir=td, prefix="unit_",
			)
			band.add(0, 0, 4, 0, 2, 0, 2, np.ones((4, 2, 2), dtype=np.float32))
			band.add(1, 0, 4, 0, 2, 0, 2, np.full((4, 2, 2), 5, dtype=np.float32))
			band.discard_before(2)
			np.testing.assert_array_equal(band.view(0, 2, 4), np.ones((2, 2, 2), dtype=np.float32))
			np.testing.assert_array_equal(band.view(1, 2, 4), np.full((2, 2, 2), 5, dtype=np.float32))
			band.cleanup()
			self.assertEqual([p for p in Path(td).iterdir() if p.name.startswith(".predict3d_")], [])

	def test_canonical_tile_positions_do_not_shift_with_crop_origin(self):
		kwargs = {
			"volume_size": 512,
			"tile_size": 128,
			"stride": 96,
			"border": 16,
			"scaledown_multiple": 4,
		}
		crop_a = _canonical_local_tile_positions(crop_start=0, crop_padded_size=192, **kwargs)
		crop_b = _canonical_local_tile_positions(crop_start=64, crop_padded_size=192, **kwargs)
		global_a = {p + 0 for p in crop_a}
		global_b = {p + 64 for p in crop_b}
		shared = global_a & global_b
		self.assertGreater(len(shared), 0)
		self.assertTrue(all((p - 0) in crop_a for p in shared))
		self.assertTrue(all((p - 64) in crop_b for p in shared))

	def test_output_region_tile_support_is_global(self):
		kwargs = {
			"volume_size": 512,
			"scaledown": 4,
			"tile_size": 128,
			"stride": 96,
			"border": 16,
			"scaledown_multiple": 4,
		}
		a = _canonical_tile_positions_for_output_region(
			output_start=16, output_end=48, **kwargs,
		)
		b = _canonical_tile_positions_for_output_region(
			output_start=16, output_end=48, **kwargs,
		)
		self.assertEqual(a, b)
		self.assertIn(96, a)


if __name__ == "__main__":
	unittest.main()
