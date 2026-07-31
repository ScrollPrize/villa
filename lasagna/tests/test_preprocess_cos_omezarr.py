import json
import io
import unittest
import os
import sys
import tempfile
from pathlib import Path
import types
from unittest import mock

import numpy as np
import torch
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
	_infer_tiled_3d,
	_omezarr_chunk_exists,
	_omezarr_chunk_group_complete,
	_predict3d_overall_eta,
	run_preprocess_3d,
)
from omezarr_pyramid import (
	_make_downsample_work,
	_write_level_block,
	build_normal_omezarr_pyramid,
	build_scalar_omezarr_pyramid,
	downsample_normal_pair_chunk_worker,
	downsample_scalar_chunk_worker,
)


class _StopAfterManifest(Exception):
	pass


def _write_zarr_array(path: Path, shape: tuple[int, int, int], value: int = 1) -> Path:
	arr = zarr.open(str(path), mode="w", shape=shape, chunks=(4, 4, 4), dtype="uint8")
	arr[:] = np.full(shape, value, dtype=np.uint8)
	return path


def _write_predict3d_manifest(path: Path, groups: dict) -> None:
	path.write_text(
		json.dumps(
			{
				"version": 2,
				"source_to_base": 2.5,
				"base_shape_zyx": [8, 8, 8],
				"grad_mag_encode_scale": 1000.0,
				"grad_mag_factor": 9.0,
				"umbilicus_json": "umbilicus.json",
				"init_shell_dir": "init_shells",
				"crops": [[1, 2, 3, 4, 5, 6]],
				"groups": groups,
			}
		)
		+ "\n",
		encoding="utf-8",
	)


class _ConstantPredict3dModel:
	def eval(self):
		return self

	def __call__(self, tile_t: torch.Tensor) -> torch.Tensor:
		return torch.zeros(
			(tile_t.shape[0], 8, tile_t.shape[2], tile_t.shape[3], tile_t.shape[4]),
			dtype=torch.float32,
			device=tile_t.device,
		)


class _OnesPredict3dModel:
	def eval(self):
		return self

	def __call__(self, tile_t: torch.Tensor) -> torch.Tensor:
		return torch.ones(
			(tile_t.shape[0], 8, tile_t.shape[2], tile_t.shape[3], tile_t.shape[4]),
			dtype=torch.float32,
			device=tile_t.device,
		)


class PreprocessCosOmezarrTests(unittest.TestCase):
	def _run_predict3d_until_model_build(
		self,
		*,
		input_path: Path,
		output_path: Path,
		pred_dt_path: Path | None = None,
		crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None,
	) -> None:
		gpu_pause_stub = types.ModuleType("gpu_pause")
		gpu_pause_stub.gpu_pause_context = lambda: None
		with mock.patch.dict(sys.modules, {"gpu_pause": gpu_pause_stub}):
			with mock.patch.object(train_stub, "build_model", side_effect=_StopAfterManifest):
				with self.assertRaises(_StopAfterManifest):
					run_preprocess_3d(
						input_path=str(input_path),
						output_path=str(output_path),
						unet3d_checkpoint=str(output_path.parent / "missing_model.pt"),
						device="cpu",
						crop_xyzwhd=crop_xyzwhd,
						tile_size=8,
						overlap=0,
						border=0,
						cos_scaledown=1,
						scaledown=2,
						source_to_base=8.0,
						pred_dt_path=str(pred_dt_path) if pred_dt_path is not None else None,
						base_ref=str(input_path),
						n_levels=3,
						ome_chunk=4,
					)

	def test_predict3d_overall_eta_uses_processed_counts_not_skipped_done(self):
		progress = {
			"tiles_total": 100,
			"tiles_done": 90,
			"tiles_processed": 10,
			"tile_time_sum": 20.0,
			"edt_total_est": 100,
			"edt_done": 50,
			"edt_processed": 10,
			"edt_time_sum": 30.0,
		}

		self.assertEqual(_predict3d_overall_eta(progress), " | overall eta 02:50")

	def test_predict3d_tile_eta_uses_processed_tiles_not_runtime_skips(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = _write_zarr_array(root / "input.zarr", (8, 8, 8))
			arr = zarr.open(str(input_path), mode="r")
			progress = {
				"finalized_base_z": 0,
				"finalized_cos_base_z": 0,
				"finalized_other_base_z": 0,
				"finalized_base_z_total": 8,
			}
			skip_state = {"calls": 0}

			def _is_tile_done(*_args):
				skip_state["calls"] += 1
				return skip_state["calls"] <= 3

			times = iter([0, 10, 14, 20, 24, 30, 34, 40, 44, 50, 54, 60])
			stdout = io.StringIO()
			with mock.patch("preprocess_cos_omezarr.time.time", side_effect=lambda: next(times)):
				with mock.patch("sys.stdout", stdout):
					_infer_tiled_3d(
						_ConstantPredict3dModel(),
						arr,
						crop_slices=(0, 8, 0, 8, 0, 8),
						device=torch.device("cpu"),
						tile_size=4,
						overlap=0,
						border=0,
						cos_scaledown=1,
						other_scaledown=1,
						tmp_dir=str(root),
						output_sigmoid=False,
						on_z_complete=lambda *_args: None,
						progress=progress,
						is_tile_done=_is_tile_done,
					)

			out = stdout.getvalue()
			self.assertIn("4/8 tiles", out)
			self.assertIn("eta 00:16 avg=4000ms/tile", out)
			self.assertIn("final_z=0/8", out)

	def test_predict3d_status_reports_finalized_z_after_band_flush(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = _write_zarr_array(root / "input.zarr", (8, 4, 4))
			output_path = root / "vol.lasagna.json"
			gpu_pause_stub = types.ModuleType("gpu_pause")
			gpu_pause_stub.gpu_pause_context = lambda: None
			stdout = io.StringIO()

			with mock.patch.dict(sys.modules, {"gpu_pause": gpu_pause_stub}):
				with mock.patch.object(
					train_stub,
					"build_model",
					return_value=(_ConstantPredict3dModel(), None, None, False),
				):
					with mock.patch("preprocess_cos_omezarr._build_omezarr_pyramid"):
						with mock.patch("preprocess_cos_omezarr.build_normal_omezarr_pyramid"):
							with mock.patch("sys.stdout", stdout):
								run_preprocess_3d(
									input_path=str(input_path),
									output_path=str(output_path),
									unet3d_checkpoint=str(root / "missing_model.pt"),
									device="cpu",
									crop_xyzwhd=None,
									tile_size=4,
									overlap=0,
									border=0,
									cos_scaledown=1,
									scaledown=1,
									source_to_base=1.0,
									base_ref=str(input_path),
									n_levels=2,
									ome_chunk=4,
								)

			out = stdout.getvalue()
			self.assertIn(
				"rolling accumulators: fine channels=1 zyx=(8,4,4) sd=1; "
				"coarse channels=7 zyx=(8,4,4) sd=1",
				out,
			)
			self.assertIn("final_z=4/8", out)
			self.assertIn("final_z=8/8", out)
			self.assertNotIn("\n[predict3d] final_z=", out)

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

	def test_atomic_zarr_write_invalidates_before_replacing_chunk(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 2, 16, "cos")
			block = np.full((16, 16, 16), 7, dtype=np.uint8)
			events: list[str] = []
			real_replace = os.replace
			live_level = str(Path(path) / "0")

			def _replace(src, dst):
				if str(dst).startswith(live_level):
					events.append("replace")
				return real_replace(src, dst)

			def _invalidate(*_args, **_kwargs):
				events.append("invalidate")

			with mock.patch("preprocess_cos_omezarr.os.replace", side_effect=_replace):
				with mock.patch("preprocess_cos_omezarr._invalidate_pyramid_chunks", side_effect=_invalidate):
					_atomic_zarr_write(path, 0, 0, 0, 0, 16, 16, 16, block, 16, n_levels=2)

			self.assertEqual(events[:2], ["invalidate", "replace"])

	def test_pyramid_full_source_scan_schedules_missing_chunk_outside_crop(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 3, 4, "cos")
			arr = zarr.open(str(Path(path) / "0"), mode="r+")
			arr[8:12, 0:4, 0:4] = np.full((4, 4, 4), 9, dtype=np.uint8)

			crop_work, _ = _make_downsample_work(
				omezarr_path=path,
				src_level=0,
				dst_level=1,
				chunk=4,
				crop_zyx=(0, 0, 0, 4, 4, 4),
				skip_existing=True,
				require_source_chunks=True,
			)
			full_work, _ = _make_downsample_work(
				omezarr_path=path,
				src_level=0,
				dst_level=1,
				chunk=4,
				crop_zyx=None,
				skip_existing=True,
				require_source_chunks=True,
			)

			self.assertEqual(crop_work, [])
			self.assertTrue(any((z0, y0, x0) == (8, 0, 0) for *_prefix, z0, _z1, y0, _y1, x0, _x1, _zero in full_work))

	def test_pyramid_level_write_is_atomic_and_cleans_temp_dir(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 3, 4, "cos")
			block = np.full((4, 4, 4), 11, dtype=np.uint8)

			_write_level_block(
				omezarr_path=path,
				level=1,
				z0=0,
				y0=0,
				x0=0,
				data=block,
				n_levels=3,
			)

			arr1 = zarr.open(str(Path(path) / "1"), mode="r")
			self.assertEqual(int(np.asarray(arr1[0, 0, 0])), 11)
			self.assertEqual([p.name for p in Path(td).iterdir() if p.name.startswith(".tmp.")], [])

	def test_pyramid_level_atomic_write_invalidates_coarser_chunk(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 3, 4, "cos")
			arr2 = zarr.open(str(Path(path) / "2"), mode="r+")
			arr2[0:4, 0:4, 0:4] = np.full((4, 4, 4), 5, dtype=np.uint8)
			self.assertTrue(Path(path, "2", "0", "0", "0").is_file())

			_write_level_block(
				omezarr_path=path,
				level=1,
				z0=0,
				y0=0,
				x0=0,
				data=np.full((4, 4, 4), 9, dtype=np.uint8),
				n_levels=3,
			)

			self.assertFalse(Path(path, "2", "0", "0", "0").exists())

	def test_pyramid_full_source_stream_writes_missing_chunk_outside_crop(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 3, 4, "cos")
			arr0 = zarr.open(str(Path(path) / "0"), mode="r+")
			arr0[8:12, 0:4, 0:4] = np.full((4, 4, 4), 8, dtype=np.uint8)
			stdout = io.StringIO()

			with mock.patch("sys.stdout", stdout):
				build_scalar_omezarr_pyramid(
					path,
					0,
					2,
					4,
					workers=1,
					crop_zyx=(0, 0, 0, 4, 4, 4),
					label="cos",
					scan_existing_source_chunks=True,
				)

			arr1 = zarr.open(str(Path(path) / "1"), mode="r")
			self.assertEqual(int(np.asarray(arr1[4, 0, 0])), 8)
			out = stdout.getvalue()
			self.assertIn("[pyramid cos L1]", out)
			self.assertIn("write=", out)
			self.assertIn("skip_empty=", out)

	def test_scalar_pyramid_worker_reports_skip_and_write_statuses(self):
		with tempfile.TemporaryDirectory() as td:
			path = str(Path(td) / "cos.ome.zarr")
			_create_omezarr(path, (16, 16, 16), 0, 2, 4, "cos")
			args = (
				path, 0, 1, 0, 8, 0, 8, 0, 8,
				False, True, True, (4, 4, 4), (4, 4, 4),
			)
			self.assertEqual(downsample_scalar_chunk_worker(args), "skipped_empty_source")

			arr0 = zarr.open(str(Path(path) / "0"), mode="r+")
			arr0[0:4, 0:4, 0:4] = np.full((4, 4, 4), 12, dtype=np.uint8)
			self.assertEqual(downsample_scalar_chunk_worker(args), "written")
			arr1 = zarr.open(str(Path(path) / "1"), mode="r")
			self.assertEqual(int(np.asarray(arr1[0, 0, 0])), 12)
			self.assertEqual(downsample_scalar_chunk_worker(args), "skipped_existing")

	def test_normal_pyramid_worker_requires_both_source_channels(self):
		with tempfile.TemporaryDirectory() as td:
			nx_path = str(Path(td) / "nx.ome.zarr")
			ny_path = str(Path(td) / "ny.ome.zarr")
			_create_omezarr(nx_path, (16, 16, 16), 0, 2, 4, "nx")
			_create_omezarr(ny_path, (16, 16, 16), 0, 2, 4, "ny")
			args = (
				nx_path, ny_path, 0, 1, 0, 8, 0, 8, 0, 8,
				True, True, (4, 4, 4), (4, 4, 4), (4, 4, 4),
			)
			nx0 = zarr.open(str(Path(nx_path) / "0"), mode="r+")
			nx0[0:4, 0:4, 0:4] = np.full((4, 4, 4), 128, dtype=np.uint8)
			self.assertEqual(downsample_normal_pair_chunk_worker(args), "skipped_empty_source")

			ny0 = zarr.open(str(Path(ny_path) / "0"), mode="r+")
			ny0[0:4, 0:4, 0:4] = np.full((4, 4, 4), 128, dtype=np.uint8)
			self.assertEqual(downsample_normal_pair_chunk_worker(args), "written")
			self.assertEqual(downsample_normal_pair_chunk_worker(args), "skipped_existing")

	def test_normal_pyramid_full_source_stream_writes_outside_crop(self):
		with tempfile.TemporaryDirectory() as td:
			nx_path = str(Path(td) / "nx.ome.zarr")
			ny_path = str(Path(td) / "ny.ome.zarr")
			_create_omezarr(nx_path, (16, 16, 16), 0, 2, 4, "nx")
			_create_omezarr(ny_path, (16, 16, 16), 0, 2, 4, "ny")
			for path in (nx_path, ny_path):
				arr0 = zarr.open(str(Path(path) / "0"), mode="r+")
				arr0[8:12, 0:4, 0:4] = np.full((4, 4, 4), 128, dtype=np.uint8)

			with mock.patch("sys.stdout", io.StringIO()):
				build_normal_omezarr_pyramid(
					nx_path,
					ny_path,
					0,
					2,
					4,
					workers=1,
					crop_zyx=(0, 0, 0, 4, 4, 4),
					scan_existing_source_chunks=True,
				)

			nx1 = zarr.open(str(Path(nx_path) / "1"), mode="r")
			ny1 = zarr.open(str(Path(ny_path) / "1"), mode="r")
			self.assertTrue(Path(nx_path, "1", "1", "0", "0").is_file())
			self.assertTrue(Path(ny_path, "1", "1", "0", "0").is_file())
			self.assertEqual(int(np.asarray(nx1[4, 0, 0])), 128)
			self.assertEqual(int(np.asarray(ny1[4, 0, 0])), 128)

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
				name="test", channel_count=2, z_size=4, y_size=2, x_size=2,
				tmp_dir=td, prefix="unit_",
			)
			band.add(0, 0, 4, 0, 2, 0, 2, np.ones((4, 2, 2), dtype=np.float32))
			band.add(1, 0, 4, 0, 2, 0, 2, np.full((4, 2, 2), 5, dtype=np.float32))
			band.discard_before(2)
			np.testing.assert_array_equal(band.view(0, 2, 4), np.ones((2, 2, 2), dtype=np.float32))
			np.testing.assert_array_equal(band.view(1, 2, 4), np.full((2, 2, 2), 5, dtype=np.float32))
			band.cleanup()
			self.assertEqual([p for p in Path(td).iterdir() if p.name.startswith(".predict3d_")], [])

	def test_rolling_z_band_sparse_ranges_read_as_zero(self):
		with tempfile.TemporaryDirectory() as td:
			band = _RollingZBand(
				name="test", channel_count=1, z_size=6, y_size=2, x_size=2,
				tmp_dir=td, prefix="unit_",
			)
			np.testing.assert_array_equal(
				band.view(0, 0, 6),
				np.zeros((6, 2, 2), dtype=np.float32),
			)

			band.add(0, 2, 4, 0, 2, 0, 2, np.full((2, 2, 2), 7, dtype=np.float32))
			np.testing.assert_array_equal(
				band.view(0, 0, 2),
				np.zeros((2, 2, 2), dtype=np.float32),
			)
			np.testing.assert_array_equal(
				band.view(0, 2, 4),
				np.full((2, 2, 2), 7, dtype=np.float32),
			)

			band.add(0, 5, 6, 0, 2, 0, 2, np.full((1, 2, 2), 3, dtype=np.float32))
			np.testing.assert_array_equal(
				band.view(0, 4, 5),
				np.zeros((1, 2, 2), dtype=np.float32),
			)
			np.testing.assert_array_equal(
				band.view(0, 5, 6),
				np.full((1, 2, 2), 3, dtype=np.float32),
			)
			band.cleanup()

	def test_rolling_z_band_discard_releases_and_rejects_revisit(self):
		with tempfile.TemporaryDirectory() as td:
			band = _RollingZBand(
				name="test", channel_count=1, z_size=6, y_size=2, x_size=2,
				tmp_dir=td, prefix="unit_",
			)
			band.add(0, 4, 6, 0, 2, 0, 2, np.full((2, 2, 2), 9, dtype=np.float32))
			with mock.patch("preprocess_cos_omezarr._release_memmap_pages") as release:
				band.discard_before(4)
			release.assert_called_once()
			self.assertEqual((release.call_args.args[1], release.call_args.args[2]), (0, 4))
			self.assertEqual(band.origin_z, 4)
			np.testing.assert_array_equal(
				band.view(0, 4, 6),
				np.full((2, 2, 2), 9, dtype=np.float32),
			)
			with self.assertRaises(ValueError):
				band.view(0, 3, 4)
			with self.assertRaises(ValueError):
				band.add(0, 3, 4, 0, 2, 0, 2, np.ones((1, 2, 2), dtype=np.float32))
			band.cleanup()

	def test_predict3d_sparse_skipped_z_prefix_flushes_later_suffix(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = root / "input.zarr"
			arr = zarr.open(str(input_path), mode="w", shape=(8, 4, 4), chunks=(4, 4, 4), dtype="uint8")
			arr[4:8, :, :] = np.full((4, 4, 4), 1, dtype=np.uint8)
			output_path = root / "vol.lasagna.json"
			gpu_pause_stub = types.ModuleType("gpu_pause")
			gpu_pause_stub.gpu_pause_context = lambda: None
			stdout = io.StringIO()

			with mock.patch.dict(sys.modules, {"gpu_pause": gpu_pause_stub}):
				with mock.patch.object(
					train_stub,
					"build_model",
					return_value=(_OnesPredict3dModel(), None, None, False),
				):
					with mock.patch("preprocess_cos_omezarr._build_omezarr_pyramid"):
						with mock.patch("preprocess_cos_omezarr.build_normal_omezarr_pyramid"):
							with mock.patch("sys.stdout", stdout):
								run_preprocess_3d(
									input_path=str(input_path),
									output_path=str(output_path),
									unet3d_checkpoint=str(root / "missing_model.pt"),
									device="cpu",
									crop_xyzwhd=None,
									tile_size=4,
									overlap=0,
									border=0,
									cos_scaledown=1,
									scaledown=1,
									source_to_base=1.0,
									base_ref=str(input_path),
									n_levels=2,
									ome_chunk=4,
								)

			cos_path = str(root / "vol_cos.ome.zarr")
			gm_path = str(root / "vol_grad_mag.ome.zarr")
			nx_path = str(root / "vol_nx.ome.zarr")
			ny_path = str(root / "vol_ny.ome.zarr")
			self.assertFalse(_omezarr_chunk_exists(cos_path, 0, 0, 0, 0, 4))
			self.assertTrue(_omezarr_chunk_exists(cos_path, 0, 4, 0, 0, 4))
			for path in (gm_path, nx_path, ny_path):
				self.assertFalse(_omezarr_chunk_exists(path, 0, 0, 0, 0, 4))
				self.assertTrue(_omezarr_chunk_exists(path, 0, 4, 0, 0, 4))
			out = stdout.getvalue()
			self.assertIn("final_z=4/8", out)
			self.assertIn("final_z=8/8", out)

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

	def test_predict3d_early_manifest_removes_stale_pred_dt_and_preserves_metadata(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = _write_zarr_array(root / "input.zarr", (8, 8, 8))
			output_path = root / "vol.lasagna.json"
			pred_dt_output = root / "vol_pred_dt.ome.zarr"
			pred_dt_output.mkdir()
			(pred_dt_output / "sentinel").write_text("keep\n", encoding="utf-8")
			_write_predict3d_manifest(
				output_path,
				{
					"cos": {"zarr": "old_cos.ome.zarr/0", "scaledown": 0, "channels": ["cos"]},
					"pred_dt": {"zarr": "vol_pred_dt.ome.zarr/0", "scaledown": 0, "channels": ["pred_dt"]},
					"obsolete": {"zarr": "obsolete.ome.zarr/0", "scaledown": 0, "channels": ["obsolete"]},
				},
			)

			self._run_predict3d_until_model_build(
				input_path=input_path,
				output_path=output_path,
				crop_xyzwhd=(0, 0, 0, 8, 8, 8),
			)

			raw = json.loads(output_path.read_text(encoding="utf-8"))
			self.assertEqual(list(raw["groups"]), ["cos", "grad_mag", "nx", "ny"])
			self.assertEqual(raw["umbilicus_json"], "umbilicus.json")
			self.assertEqual(raw["init_shell_dir"], "init_shells")
			self.assertEqual(raw["source_to_base"], 2.5)
			self.assertIn([1, 2, 3, 4, 5, 6], raw["crops"])
			self.assertIn([0, 0, 0, 8, 8, 8], raw["crops"])
			self.assertTrue((pred_dt_output / "sentinel").exists())
			backups = sorted(root.glob("vol_old.*.lasagna.json"))
			self.assertEqual(len(backups), 1)
			old_raw = json.loads(backups[0].read_text(encoding="utf-8"))
			self.assertIn("pred_dt", old_raw["groups"])

	def test_predict3d_later_pred_dt_run_readds_manifest_group_and_reuses_output(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = _write_zarr_array(root / "input.zarr", (8, 8, 8))
			pred_dt_source = _write_zarr_array(root / "pred_source.zarr", (8, 8, 8))
			output_path = root / "vol.lasagna.json"
			_write_predict3d_manifest(
				output_path,
				{
					"cos": {"zarr": "vol_cos.ome.zarr/0", "scaledown": 0, "channels": ["cos"]},
					"grad_mag": {"zarr": "vol_grad_mag.ome.zarr/1", "scaledown": 1, "channels": ["grad_mag"]},
					"nx": {"zarr": "vol_nx.ome.zarr/1", "scaledown": 1, "channels": ["nx"]},
					"ny": {"zarr": "vol_ny.ome.zarr/1", "scaledown": 1, "channels": ["ny"]},
				},
			)
			_create_omezarr(str(root / "vol_pred_dt.ome.zarr"), (8, 8, 8), 0, 3, 4, "pred_dt")
			dt_arr = zarr.open(str(root / "vol_pred_dt.ome.zarr" / "0"), mode="r+")
			dt_arr[0:4, 0:4, 0:4] = np.full((4, 4, 4), 13, dtype=np.uint8)

			self._run_predict3d_until_model_build(
				input_path=input_path,
				output_path=output_path,
				pred_dt_path=pred_dt_source,
			)

			raw = json.loads(output_path.read_text(encoding="utf-8"))
			self.assertIn("pred_dt", raw["groups"])
			self.assertEqual(raw["groups"]["pred_dt"]["zarr"], "vol_pred_dt.ome.zarr/0")
			self.assertEqual(int(np.asarray(dt_arr[0, 0, 0])), 13)


if __name__ == "__main__":
	unittest.main()
