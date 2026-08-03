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

import tiled_predict3d as shared_predict3d


common_stub = types.ModuleType("common")
common_stub.load_unet = None
common_stub.unet_infer_tiled = None
sys.modules.setdefault("common", common_stub)

train_stub = types.ModuleType("train_unet_3d")
train_stub.build_model = None
sys.modules.setdefault("train_unet_3d", train_stub)

from preprocess_cos_omezarr import (
	LasagnaCosPredict3DAdapter,
	ModelAdapter,
	OmeZarrOutputAdapter,
	OutputAdapter,
	OutputChannelSpec,
	OutputProductSpec,
	PYRAMID_POLICY_CUSTOM,
	PYRAMID_POLICY_NONE,
	PYRAMID_POLICY_SCALAR,
	_CircularZBand,
	_plan_circular_z_depth,
	_atomic_zarr_write,
	_canonical_local_tile_positions,
	_canonical_tile_positions_for_output_region,
	_cleanup_predict3d_temp_files,
	_create_omezarr,
	_grad_mag_factor_from_input_sd,
	run_tiled_inference_3d,
	_omezarr_chunk_exists,
	_omezarr_chunk_group_complete,
	_predict3d_overall_eta,
	cli_main,
	run_preprocess_3d,
)
import preprocess_cos_omezarr as preprocess_wrapper
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


class _FakeModelAdapter:
	@property
	def output_products(self):
		return (
			OutputProductSpec(
				name="fiber_option0",
				level=2,
				scaledown=4,
				channels=("dir0_z", "dir1_z", "dir0_y", "dir1_y", "dir0_x", "dir1_x", "presence"),
				chunk_size=32,
			),
		)

	def load_model(self, *, device: torch.device):
		return object()

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		return tile

	def product_tensors_from_output(self, raw_output):
		return {self.output_products[0].name: raw_output}

	def finalize_product_slab(self, product, raw_slab):
		return {channel.name: raw_slab[index] for index, channel in enumerate(product.channels)}


class _FakeOutputAdapter:
	def product_chunk_complete(self, product: OutputProductSpec, *, chunk_origin_zyx):
		return False

	def write_product_chunk(self, product: OutputProductSpec, *, chunk_origin_zyx, data):
		pass

	def update_metadata(self, products):
		pass


class _SpyPredict3dAdapter:
	def __init__(self):
		self.calls = 0

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		self.calls += 1
		return model(tile)


class _IdentityProductAdapter:
	def __init__(self, product: OutputProductSpec):
		self._products = (product,)
		self.calls = 0

	@property
	def output_products(self):
		return self._products

	def load_model(self, *, device: torch.device):
		return object()

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		self.calls += 1
		_ = model
		return tile

	def product_tensors_from_output(self, raw_output):
		return {self._products[0].name: raw_output}


class _RaisingProductAdapter(_IdentityProductAdapter):
	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		self.calls += 1
		_ = model, tile, device
		raise AssertionError("resume should skip completed product chunks")


class PreprocessCosOmezarrTests(unittest.TestCase):
	def test_predict3d_shared_helpers_are_reexported_for_compatibility(self):
		self.assertIs(preprocess_wrapper.OutputProductSpec, shared_predict3d.OutputProductSpec)
		self.assertIs(preprocess_wrapper.OutputChannelSpec, shared_predict3d.OutputChannelSpec)
		self.assertIs(preprocess_wrapper.OmeZarrOutputAdapter, shared_predict3d.OmeZarrOutputAdapter)
		self.assertIs(preprocess_wrapper.ModelAdapter, shared_predict3d.ModelAdapter)
		self.assertIs(preprocess_wrapper.OutputAdapter, shared_predict3d.OutputAdapter)
		self.assertIs(preprocess_wrapper.run_tiled_inference_3d, shared_predict3d.run_tiled_inference_3d)
		self.assertFalse(hasattr(shared_predict3d, "_infer_tiled_3d"))
		self.assertIs(
			preprocess_wrapper._canonical_tile_positions_for_output_region,
			shared_predict3d._canonical_tile_positions_for_output_region,
		)
		self.assertIs(
			preprocess_wrapper._cleanup_predict3d_temp_files,
			shared_predict3d._cleanup_predict3d_temp_files,
		)

	def test_output_product_spec_normalizes_and_validates_bundle(self):
		spec = OutputProductSpec(
			name="fiber_option0",
			level=2,
			scaledown=4,
			channels=(
				"dir0_z",
				OutputChannelSpec("dir1_z", relative_path="option0/dir1_z.ome.zarr"),
				"dir0_y",
				"dir1_y",
				"dir0_x",
				"dir1_x",
				"presence",
			),
			chunk_size=32,
			dtype="uint8",
			value_range=(0, 255),
			pyramid_policy=PYRAMID_POLICY_NONE,
		)

		self.assertEqual(spec.channel_count, 7)
		self.assertEqual(
			spec.channel_names,
			("dir0_z", "dir1_z", "dir0_y", "dir1_y", "dir0_x", "dir1_x", "presence"),
		)
		self.assertEqual(spec.dtype, np.dtype("uint8"))
		self.assertEqual(spec.value_range, (0.0, 255.0))

		with self.assertRaises(ValueError):
			OutputProductSpec(
				name="bad",
				level=0,
				scaledown=1,
				channels=("presence", "presence"),
				chunk_size=32,
			)
		with self.assertRaises(ValueError):
			OutputProductSpec(
				name="bad",
				level=0,
				scaledown=0,
				channels=("presence",),
				chunk_size=32,
			)

	def test_product_manifest_does_not_write_trace_scale_aliases(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			output_path = root / "fiber.lasagna.json"
			product = OutputProductSpec(
				name="fiber_option_000",
				level=4,
				scaledown=16,
				inference_scaledown=4,
				channels=(
					OutputChannelSpec("presence", relative_path=str(root / "presence.ome.zarr")),
					OutputChannelSpec("nx", relative_path=str(root / "nx.ome.zarr")),
					OutputChannelSpec("ny", relative_path=str(root / "ny.ome.zarr")),
				),
				chunk_size=64,
			)

			shared_predict3d.write_lasagna_product_manifest(
				output_path=output_path,
				products=(product,),
				base_shape_zyx=(64, 64, 64),
			)

			raw = json.loads(output_path.read_text(encoding="utf-8"))
			self.assertNotIn("trace_to_base_scale", raw)
			self.assertNotIn("prediction_to_base_scale", raw)
			self.assertNotIn("prediction_spacing_in_trace_voxels", raw)
			self.assertNotIn("inference_scaledown_factor", raw)
			self.assertEqual(raw["groups"]["presence"]["scaledown"], 4)
			self.assertNotIn("inference_scaledown", raw["groups"]["presence"])
			self.assertNotIn("inference_scaledown", raw["groups"]["nx"])
			self.assertNotIn("inference_scaledown", raw["groups"]["ny"])

			from lasagna_volume import LasagnaVolume

			loaded = LasagnaVolume.load(output_path)
			self.assertEqual(loaded.groups["presence"].scaledown, 4)

	def test_predict3d_adapter_protocols_are_runtime_checkable(self):
		self.assertIsInstance(_FakeModelAdapter(), ModelAdapter)
		self.assertIsInstance(_FakeOutputAdapter(), OutputAdapter)

	def test_lasagna_predict3d_adapter_schema_keeps_current_products(self):
		adapter = LasagnaCosPredict3DAdapter(
			checkpoint="model.pt",
			tile_size=64,
			device_name="cpu",
			cos_product=OutputProductSpec(
				name=LasagnaCosPredict3DAdapter.COS_PRODUCT,
				level=1,
				scaledown=2,
				channels=(OutputChannelSpec("cos", relative_path="cos.ome.zarr"),),
				chunk_size=32,
				pyramid_policy=PYRAMID_POLICY_SCALAR,
			),
			normal_product=OutputProductSpec(
				name=LasagnaCosPredict3DAdapter.NORMAL_PRODUCT,
				level=2,
				scaledown=4,
				channels=(
					OutputChannelSpec("grad_mag", relative_path="grad_mag.ome.zarr"),
					OutputChannelSpec("nx", relative_path="nx.ome.zarr"),
					OutputChannelSpec("ny", relative_path="ny.ome.zarr"),
				),
				chunk_size=32,
				pyramid_policy=PYRAMID_POLICY_CUSTOM,
			),
			pred_dt_product=OutputProductSpec(
				name=LasagnaCosPredict3DAdapter.PRED_DT_PRODUCT,
				level=1,
				scaledown=2,
				channels=(OutputChannelSpec("pred_dt", relative_path="pred_dt.ome.zarr"),),
				chunk_size=32,
				pyramid_policy=PYRAMID_POLICY_SCALAR,
			),
		)

		self.assertIsInstance(adapter, ModelAdapter)
		self.assertEqual(
			[product.name for product in adapter.model_output_products],
			[
				LasagnaCosPredict3DAdapter.COS_PRODUCT,
				LasagnaCosPredict3DAdapter.NORMAL_PRODUCT,
			],
		)
		self.assertEqual(
			adapter.normal_product.channel_names,
			("grad_mag", "nx", "ny"),
		)
		self.assertEqual(
			[product.name for product in adapter.derived_output_products],
			[LasagnaCosPredict3DAdapter.PRED_DT_PRODUCT],
		)

	def test_lasagna_output_adapter_requires_complete_normal_bundle(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			paths = {
				name: str(root / f"{name}.ome.zarr")
				for name in ("grad_mag", "nx", "ny")
			}
			for path in paths.values():
				_create_omezarr(path, (16, 16, 16), 0, 1, 16, "test")

			product = OutputProductSpec(
				name=LasagnaCosPredict3DAdapter.NORMAL_PRODUCT,
				level=0,
				scaledown=1,
				channels=(
					OutputChannelSpec("grad_mag", relative_path=paths["grad_mag"]),
					OutputChannelSpec("nx", relative_path=paths["nx"]),
					OutputChannelSpec("ny", relative_path=paths["ny"]),
				),
				chunk_size=16,
			)
			adapter = OmeZarrOutputAdapter(products=(product,), n_levels=1)
			block = np.ones((16, 16, 16), dtype=np.uint8)

			self.assertIsInstance(adapter, OutputAdapter)
			self.assertFalse(
				adapter.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))
			)
			adapter.write_product_chunk(
				product,
				chunk_origin_zyx=(0, 0, 0),
				data={"grad_mag": block},
			)
			self.assertTrue(
				adapter.channel_chunk_exists(
					product, "grad_mag", chunk_origin_zyx=(0, 0, 0)
				)
			)
			self.assertFalse(
				adapter.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))
			)
			adapter.write_product_chunk(
				product,
				chunk_origin_zyx=(0, 0, 0),
				data={"nx": block, "ny": block},
			)
			self.assertTrue(
				adapter.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))
			)

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

	def test_predict3d_cli_dispatch_preserves_legacy_args(self):
		args = [
			"predict3d",
			"--input", "in.zarr",
			"--output", "out.lasagna.json",
			"--unet-checkpoint", "model.pt",
			"--tile-size", "64",
			"--overlap", "8",
			"--border", "4",
			"--cos-scaledown", "2",
			"--scaledown", "4",
			"--source-to-base", "3.5",
			"--crop", "1", "2", "3", "4", "5", "6",
			"--pred-dt", "pred.zarr",
			"--device", "cpu",
			"--chunk-z", "16",
			"--chunk-yx", "24",
			"--edt-chunk-depth", "40",
			"--edt-chunk-yx", "48",
			"--calibrate-norm",
			"--base-ref", "base.zarr",
			"--base-scale", "1",
			"--levels", "3",
			"--ome-chunk", "32",
			"--no-download",
		]
		with mock.patch("builtins.open", side_effect=PermissionError):
			with mock.patch("preprocess_cos_omezarr._auto_download") as auto_download:
				with mock.patch("preprocess_cos_omezarr.run_preprocess_3d") as run:
					self.assertEqual(cli_main(args), 0)

		auto_download.assert_not_called()
		run.assert_called_once()
		kwargs = run.call_args.kwargs
		self.assertEqual(kwargs["input_path"], "in.zarr")
		self.assertEqual(kwargs["output_path"], "out.lasagna.json")
		self.assertEqual(kwargs["unet3d_checkpoint"], "model.pt")
		self.assertEqual(kwargs["device"], "cpu")
		self.assertEqual(kwargs["crop_xyzwhd"], (1, 2, 3, 4, 5, 6))
		self.assertEqual(kwargs["tile_size"], 64)
		self.assertEqual(kwargs["overlap"], 8)
		self.assertEqual(kwargs["border"], 4)
		self.assertEqual(kwargs["cos_scaledown"], 2)
		self.assertEqual(kwargs["scaledown"], 4)
		self.assertEqual(kwargs["source_to_base"], 3.5)
		self.assertEqual(kwargs["pred_dt_path"], "pred.zarr")
		self.assertEqual(kwargs["chunk_z"], 16)
		self.assertEqual(kwargs["chunk_yx"], 24)
		self.assertEqual(kwargs["edt_chunk_depth"], 40)
		self.assertEqual(kwargs["edt_chunk_yx"], 48)
		self.assertTrue(kwargs["calibrate_norm"])
		self.assertEqual(kwargs["base_ref"], "base.zarr")
		self.assertEqual(kwargs["base_scale"], 1)
		self.assertEqual(kwargs["n_levels"], 3)
		self.assertEqual(kwargs["ome_chunk"], 64)

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
					with mock.patch("preprocess_cos_omezarr.build_product_omezarr_pyramids"):
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

			with mock.patch(f"{_atomic_zarr_write.__module__}.os.replace", side_effect=_replace):
				with mock.patch(f"{_atomic_zarr_write.__module__}._invalidate_pyramid_chunks", side_effect=_invalidate):
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
			band = _CircularZBand(
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
			band = _CircularZBand(
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

	def test_circular_z_band_wraps_and_rejects_unflushed_overwrite(self):
		with tempfile.TemporaryDirectory() as td:
			band = _CircularZBand(
				name="test", channel_count=1, z_size=12, y_size=2, x_size=2,
				tmp_dir=td, prefix="unit_", ring_depth=4,
			)
			band.add(0, 0, 4, 0, 2, 0, 2, np.full((4, 2, 2), 9, dtype=np.float32))
			with self.assertRaises(ValueError):
				band.add(0, 4, 5, 0, 2, 0, 2, np.ones((1, 2, 2), dtype=np.float32))
			band.clear(0, 2, 0, 2, 0, 2)
			band.discard_before(2)
			band.add(0, 4, 6, 0, 2, 0, 2, np.full((2, 2, 2), 7, dtype=np.float32))
			np.testing.assert_array_equal(
				band.read(0, 4, 6, 0, 2, 0, 2),
				np.full((2, 2, 2), 7, dtype=np.float32),
			)
			with self.assertRaises(ValueError):
				band.read(0, 0, 1, 0, 2, 0, 2)
			paths = list(Path(td).glob(".predict3d_*.tmp"))
			self.assertEqual(len(paths), 1)
			self.assertEqual(paths[0].stat().st_size, 4 * 2 * 2 * 4)
			band.cleanup()

	def test_circular_depth_is_independent_of_full_output_z(self):
		kwargs = dict(
			z_positions=(0, 192, 384, 576), tile_size=256, scaledown=1,
			chunk_size=32, output_begin=32, output_end=800,
		)
		self.assertEqual(
			_plan_circular_z_depth(z_size=832, **kwargs),
			_plan_circular_z_depth(z_size=8_000_032, **kwargs),
		)

	def test_shared_runner_infers_multiscale_products_once_per_tile(self):
		fine = OutputProductSpec(
			name="fine", level=0, scaledown=1, inference_scaledown=1,
			channels=("fine",), chunk_size=2,
		)
		coarse = OutputProductSpec(
			name="coarse", level=1, scaledown=2, inference_scaledown=2,
			channels=("coarse",), chunk_size=2,
		)

		class Adapter:
			calls = 0

			def run_tile_inference(self, model, tile, *, device):
				self.calls += 1
				return tile

			def product_tensors_from_output(self, output):
				return {"fine": output, "coarse": output}

			def finalize_product_slab(self, product, raw):
				return {product.channels[0].name: raw[0]}

		class Output:
			def __init__(self):
				self.complete = set()

			def product_chunk_complete(self, product, *, chunk_origin_zyx):
				return (product.name, chunk_origin_zyx) in self.complete

			def write_product_chunk(self, product, *, chunk_origin_zyx, data):
				self.complete.add((product.name, chunk_origin_zyx))
				self.assert_data = data

		adapter = Adapter()
		output = Output()
		with tempfile.TemporaryDirectory() as td:
			with mock.patch(f"{run_tiled_inference_3d.__module__}._input_has_chunks", return_value=True):
				run_tiled_inference_3d(
					object(), np.ones((8, 8, 8), dtype=np.uint8),
					crop_slices=(0, 8, 0, 8, 0, 8), device=torch.device("cpu"),
					model_adapter=adapter, output_adapter=output,
					products=(fine, coarse),
					output_regions_zyx={"fine": (0, 0, 0, 8, 8, 8), "coarse": (0, 0, 0, 4, 4, 4)},
					full_output_shapes_zyx={"fine": (8, 8, 8), "coarse": (4, 4, 4)},
					tile_size=4, overlap=0, border=0, tmp_dir=td,
				)
		self.assertEqual(adapter.calls, 8)
		self.assertEqual(len([key for key in output.complete if key[0] == "fine"]), 64)
		self.assertEqual(len([key for key in output.complete if key[0] == "coarse"]), 8)

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
					with mock.patch("preprocess_cos_omezarr.build_product_omezarr_pyramids"):
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

	def test_shared_runner_does_not_clear_or_write_unsupported_xy_chunks(self):
		product = OutputProductSpec(
			name="identity", level=0, scaledown=1, inference_scaledown=1,
			channels=("value",), chunk_size=4,
		)
		adapter = _IdentityProductAdapter(product)

		class Output:
			def __init__(self):
				self.written = []

			def product_chunk_complete(self, product, *, chunk_origin_zyx):
				return False

			def write_product_chunk(self, product, *, chunk_origin_zyx, data):
				self.written.append(chunk_origin_zyx)

		output = Output()
		clear_calls = []
		original_clear = _CircularZBand.clear

		def spy_clear(band, *args):
			clear_calls.append((band.name, args))
			return original_clear(band, *args)

		with tempfile.TemporaryDirectory() as td:
			input_path = Path(td) / "input.zarr"
			arr = zarr.open(str(input_path), mode="w", shape=(4, 8, 8), chunks=(4, 4, 4), dtype="uint8")
			arr[0:4, 0:4, 0:4] = 1
			with mock.patch.object(_CircularZBand, "clear", new=spy_clear):
				run_tiled_inference_3d(
					object(), zarr.open(str(input_path), mode="r"),
					crop_slices=(0, 4, 0, 8, 0, 8), device=torch.device("cpu"),
					model_adapter=adapter, output_adapter=output, products=(product,),
					output_regions_zyx={"identity": (0, 0, 0, 4, 8, 8)},
					full_output_shapes_zyx={"identity": (4, 8, 8)},
					input_zarr_path=str(input_path), tile_size=4, overlap=0, border=0,
					tmp_dir=td,
				)
		self.assertEqual(output.written, [(0, 0, 0)])
		self.assertEqual([name for name, _ in clear_calls], ["acc_identity", "weight_sd1"])

	def test_shared_runner_supports_global_crop_beyond_local_ring_at_sd4(self):
		product = OutputProductSpec(
			name="identity", level=2, scaledown=4, inference_scaledown=4,
			channels=("value",), chunk_size=2,
		)
		adapter = _IdentityProductAdapter(product)

		class Output:
			def __init__(self):
				self.written = []

			def product_chunk_complete(self, product, *, chunk_origin_zyx):
				return False

			def write_product_chunk(self, product, *, chunk_origin_zyx, data):
				self.written.append(chunk_origin_zyx)

		with tempfile.TemporaryDirectory() as td:
			input_path = Path(td) / "input.zarr"
			arr = zarr.open(
				str(input_path), mode="w", shape=(16, 16, 16),
				chunks=(8, 8, 8), dtype="uint8",
			)
			arr[8:16, 8:16, 8:16] = 1
			output = Output()
			run_tiled_inference_3d(
				object(), zarr.open(str(input_path), mode="r"),
				crop_slices=(8, 16, 8, 16, 8, 16), device=torch.device("cpu"),
				model_adapter=adapter, output_adapter=output, products=(product,),
				output_regions_zyx={"identity": (2, 2, 2, 4, 4, 4)},
				full_output_shapes_zyx={"identity": (4, 4, 4)},
				input_zarr_path=str(input_path), tile_size=8, overlap=0, border=0,
				tmp_dir=td,
			)

		self.assertGreater(adapter.calls, 0)
		self.assertEqual(output.written, [(2, 2, 2)])

	def test_shared_runner_skips_absent_global_crop_source_at_sd4(self):
		product = OutputProductSpec(
			name="identity", level=2, scaledown=4, inference_scaledown=4,
			channels=("value",), chunk_size=2,
		)
		adapter = _IdentityProductAdapter(product)

		class Output:
			def __init__(self):
				self.written = []

			def product_chunk_complete(self, product, *, chunk_origin_zyx):
				return False

			def write_product_chunk(self, product, *, chunk_origin_zyx, data):
				self.written.append(chunk_origin_zyx)

		with tempfile.TemporaryDirectory() as td:
			input_path = Path(td) / "input.zarr"
			zarr.open(
				str(input_path), mode="w", shape=(16, 16, 16),
				chunks=(8, 8, 8), dtype="uint8",
			)
			output = Output()
			run_tiled_inference_3d(
				object(), zarr.open(str(input_path), mode="r"),
				crop_slices=(8, 16, 8, 16, 8, 16), device=torch.device("cpu"),
				model_adapter=adapter, output_adapter=output, products=(product,),
				output_regions_zyx={"identity": (2, 2, 2, 4, 4, 4)},
				full_output_shapes_zyx={"identity": (4, 4, 4)},
				input_zarr_path=str(input_path), tile_size=8, overlap=0, border=0,
				tmp_dir=td,
			)

		self.assertEqual(adapter.calls, 0)
		self.assertEqual(output.written, [])

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

	def test_shared_product_runner_is_crop_composable_and_resumable(self):
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			input_path = root / "input.zarr"
			arr = zarr.open(
				str(input_path),
				mode="w",
				shape=(16, 16, 16),
				chunks=(4, 4, 4),
				dtype=np.uint8,
			)
			zz, yy, xx = np.indices((16, 16, 16), dtype=np.int32)
			arr[:] = ((zz * 17 + yy * 5 + xx * 3) % 251).astype(np.uint8)

			def _run_product_crop(
				output_root: Path,
				*,
				crop_slices: tuple[int, int, int, int, int, int],
				output_region_zyx: tuple[int, int, int, int, int, int],
				adapter_cls=_IdentityProductAdapter,
			):
				channel_path = output_root / "identity.ome.zarr"
				product = OutputProductSpec(
					name="identity",
					level=0,
					scaledown=1,
					channels=(
						OutputChannelSpec("value", relative_path=str(channel_path)),
					),
					chunk_size=4,
				)
				if not channel_path.exists():
					_create_omezarr(str(channel_path), (16, 16, 16), 0, 1, 4, "value")
				model_adapter = adapter_cls(product)
				output_adapter = OmeZarrOutputAdapter(products=(product,), n_levels=1)
				run_tiled_inference_3d(
					model_adapter.load_model(device=torch.device("cpu")),
					zarr.open(str(input_path), mode="r"),
					crop_slices=crop_slices,
					device=torch.device("cpu"),
					model_adapter=model_adapter,
					output_adapter=output_adapter,
					products=model_adapter.output_products,
					output_regions_zyx={product.name: output_region_zyx},
					full_output_shapes_zyx={product.name: (16, 16, 16)},
					input_zarr_path=str(input_path),
					tile_size=8,
					overlap=4,
					border=0,
					tmp_dir=str(root),
					temp_prefix=f"{output_root.name}_",
				)
				return model_adapter, zarr.open(str(channel_path / "0"), mode="r")

			full_adapter, full_arr = _run_product_crop(
				root / "out_full",
				crop_slices=(0, 16, 0, 16, 0, 16),
				output_region_zyx=(0, 0, 0, 16, 16, 16),
			)
			crop_adapter, crop_arr = _run_product_crop(
				root / "out_crop",
				crop_slices=(4, 12, 4, 12, 4, 12),
				output_region_zyx=(4, 4, 4, 12, 12, 12),
			)

			self.assertGreater(full_adapter.calls, 0)
			self.assertGreater(crop_adapter.calls, 0)
			np.testing.assert_array_equal(
				np.asarray(full_arr[4:8, 4:8, 4:8]),
				np.asarray(crop_arr[4:8, 4:8, 4:8]),
			)

			resume_adapter, _ = _run_product_crop(
				root / "out_full",
				crop_slices=(0, 16, 0, 16, 0, 16),
				output_region_zyx=(0, 0, 0, 16, 16, 16),
				adapter_cls=_RaisingProductAdapter,
			)
			self.assertEqual(resume_adapter.calls, 0)

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
