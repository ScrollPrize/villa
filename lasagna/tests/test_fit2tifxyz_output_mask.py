from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit2tifxyz
import model
import opt_loss_corr


def _plane_mesh(h: int, w: int, *, z_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	rows, cols = np.indices((h, w), dtype=np.float32)
	return cols, rows, np.full((h, w), float(z_value), dtype=np.float32)


def _square_payload(*, radius: int = 0) -> dict:
	return {
		"version": 2,
		"source": "corr_points",
		"corr_collection_ids": [7],
		"corr_contours": [{"collection_id": 7, "point_ids": [0, 1, 2, 3]}],
		"dilation_radius": int(radius),
	}


def _corr_point_result(
	row_index: int,
	h: float,
	w: float,
	*,
	valid: bool = True,
	collection_id: int = 7,
	layer: int = 0,
) -> dict:
	return {
		"row_index": int(row_index),
		"point_id": int(row_index),
		"collection_id": int(collection_id),
		"valid": bool(valid),
		"winding_err": 0.0 if valid else None,
		"model_locations": [
			{
				"anchor": "avg_low",
				"d": int(layer),
				"h": float(h),
				"w": float(w),
				"weight": 1.0,
				"residual": 0.0,
			}
		],
	}


def _square_corr_results(*, include_invalid: bool = False) -> dict:
	points = [
		_corr_point_result(0, 1.5, 1.5),
		_corr_point_result(1, 4.5, 1.5),
		_corr_point_result(2, 4.5, 4.5),
		_corr_point_result(3, 1.5, 4.5),
	]
	if include_invalid:
		points.insert(2, _corr_point_result(99, 6.0, 6.0, valid=False))
	return {"points_list": points, "collection_avgs": {"7": 0.0}}


def _model_params(*, mesh_step: int = 1, depth_windings: tuple[int, ...] = (0,)) -> dict:
	return {
		"mesh_step": mesh_step,
		"winding_step": 1,
		"subsample_mesh": 1,
		"subsample_winding": 1,
		"scaledown": 1.0,
		"z_step_eff": 1,
		"volume_extent": None,
		"pyramid_d": False,
		"depth_windings": [int(v) for v in depth_windings],
	}


def _flow_gate_payload(local: np.ndarray, normalized: np.ndarray) -> dict:
	return {
		"version": 1,
		"stage_name": "snap",
		"mesh_shape_dhw": list(local.shape),
		"flow_gate_local_contrast": torch.from_numpy(local.astype(np.float32)),
		"flow_gate_component_normalized": torch.from_numpy(normalized.astype(np.float32)),
		"source_config": {
			"local_boost": 0.5,
			"backtrack_distance": 50.0,
		},
	}


class Fit2TifxyzOutputMaskHelpersTest(unittest.TestCase):
	def test_corr_point_roi_mask_seeds_fractional_four_corners_and_maxpools(self) -> None:
		import fit

		corr_results = {
			"points_list": [
				_corr_point_result(0, 2.2, 3.7),
			],
		}

		mask, debug = fit._corr_point_roi_mask_from_results(
			corr_results,
			shape=(8, 8),
			radius=1,
			device=torch.device("cpu"),
		)

		expected_seed = np.zeros((8, 8), dtype=bool)
		expected_seed[2, 3] = True
		expected_seed[2, 4] = True
		expected_seed[3, 3] = True
		expected_seed[3, 4] = True
		expected = np.zeros((8, 8), dtype=bool)
		expected[1:5, 2:6] = True
		self.assertEqual(debug["seed_vertex_count"], int(expected_seed.sum()))
		self.assertEqual(mask.detach().cpu().numpy().tolist(), expected.tolist())

	def test_corr_point_roi_mask_requires_usable_locations(self) -> None:
		import fit

		with self.assertRaisesRegex(ValueError, "no usable final corr projections"):
			fit._corr_point_roi_mask_from_results(
				{"points_list": [_corr_point_result(0, 1.0, 1.0, valid=False)]},
				shape=(4, 4),
				radius=1,
				device=torch.device("cpu"),
			)

	def test_corr_point_polygon_fills_expected_vertices(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		mask = fit2tifxyz._approval_output_mask_for_layer(
			_square_payload(radius=0), x, y, z,
			layer_index=0, corr_results=_square_corr_results(), fit_config=None,
		)

		expected = np.zeros((7, 7), dtype=bool)
		expected[2:5, 2:5] = True
		self.assertEqual(mask.tolist(), expected.tolist())

	def test_invalid_corr_points_are_skipped(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		mask = fit2tifxyz._approval_output_mask_for_layer(
			_square_payload(radius=0), x, y, z,
			layer_index=0, corr_results=_square_corr_results(include_invalid=True), fit_config=None,
		)

		expected = np.zeros((7, 7), dtype=bool)
		expected[2:5, 2:5] = True
		self.assertEqual(mask.tolist(), expected.tolist())

	def test_corr_contour_point_ids_define_polygon_order(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		corr_results = {
			"points_list": [
				_corr_point_result(0, 1.5, 1.5),
				_corr_point_result(1, 4.5, 4.5),
				_corr_point_result(2, 4.5, 1.5),
				_corr_point_result(3, 1.5, 4.5),
			],
		}
		payload = _square_payload(radius=0)
		payload["corr_contours"] = [{"collection_id": 7, "point_ids": [0, 2, 1, 3]}]

		mask = fit2tifxyz._approval_output_mask_for_layer(
			payload, x, y, z,
			layer_index=0, corr_results=corr_results, fit_config=None,
		)

		expected = np.zeros((7, 7), dtype=bool)
		expected[2:5, 2:5] = True
		self.assertEqual(mask.tolist(), expected.tolist())

	def test_too_few_usable_corr_points_masks_layer_out(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		corr_results = {
			"points_list": [
				_corr_point_result(0, 1.5, 1.5),
				_corr_point_result(1, 4.5, 1.5),
			],
		}

		mask = fit2tifxyz._approval_output_mask_for_layer(
			_square_payload(radius=0), x, y, z,
			layer_index=0, corr_results=corr_results, fit_config=None,
		)

		self.assertFalse(bool(mask.any()))

	def test_concave_polygon_uses_even_odd_fill(self) -> None:
		contour = [
			(1.5, 1.5),
			(1.5, 4.5),
			(2.5, 4.5),
			(2.5, 2.5),
			(4.5, 2.5),
			(4.5, 1.5),
		]
		mask = fit2tifxyz._rasterize_contours([contour], (7, 7))

		self.assertTrue(mask[2, 2])
		self.assertTrue(mask[2, 4])
		self.assertTrue(mask[4, 2])
		self.assertFalse(mask[4, 4])

	def test_dilation_radius_three_expands_by_three_vertex_steps(self) -> None:
		mask = np.zeros((11, 11), dtype=bool)
		mask[5, 5] = True

		got = fit2tifxyz._dilate_chebyshev(mask, 3)

		expected = np.zeros((11, 11), dtype=bool)
		expected[2:9, 2:9] = True
		self.assertEqual(int(got.sum()), 49)
		self.assertEqual(got.tolist(), expected.tolist())

	def test_sentinel_vertices_are_excluded_from_area_and_bbox(self) -> None:
		x = np.array(
			[
				[-1.0, 11.0, 12.0],
				[10.0, 11.0, 12.0],
				[10.0, 11.0, 12.0],
			],
			dtype=np.float32,
		)
		y = np.array(
			[
				[-1.0, 20.0, 20.0],
				[21.0, 21.0, 21.0],
				[22.0, 22.0, 22.0],
			],
			dtype=np.float32,
		)
		z = np.array(
			[
				[-1.0, 30.0, 30.0],
				[30.0, 30.0, 30.0],
				[30.0, 30.0, 30.0],
			],
			dtype=np.float32,
		)

		area = fit2tifxyz._get_area(x, y, z, 1.0, None)
		self.assertEqual(area["area_vx2"], 3.0)

		with tempfile.TemporaryDirectory() as td:
			out_dir = Path(td) / "sentinel.tifxyz"
			fit2tifxyz._write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, scale=1.0, area=area)
			meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

		self.assertEqual(meta["bbox"], [[10.0, 20.0, 30.0], [12.0, 22.0, 30.0]])

	def test_all_invalid_export_writes_invalid_bbox(self) -> None:
		x = np.full((2, 2), -1.0, dtype=np.float32)
		y = np.full((2, 2), -1.0, dtype=np.float32)
		z = np.full((2, 2), -1.0, dtype=np.float32)

		with tempfile.TemporaryDirectory() as td:
			out_dir = Path(td) / "empty.tifxyz"
			area = fit2tifxyz._get_area(x, y, z, 1.0, None)
			fit2tifxyz._write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, scale=1.0, area=area)
			meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

			self.assertEqual(area["area_vx2"], 0.0)
			self.assertEqual(meta["bbox"], [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

	def test_write_tifxyz_preserves_job_spec_and_uses_uncompressed_tiffs(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		job_spec = {
			"model": {"type": "lasagna_model", "name": "sheet/model.pt", "hash": "md5:" + "1" * 32},
			"linked_surfaces": [
				{"type": "tifxyz_segment", "name": "ref.tifxyz", "hash": "md5:" + "2" * 32}
			],
			"config": {},
		}
		with tempfile.TemporaryDirectory() as td:
			out_dir = Path(td) / "job.tifxyz"
			fit2tifxyz._write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, scale=1.0, job_spec=job_spec)
			meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
			with tifffile.TiffFile(out_dir / "x.tif") as tif:
				compression = tif.pages[0].compression.name

		self.assertEqual(meta["lasagna_job"], job_spec)
		self.assertEqual(compression, "NONE")


class CorrWindingResultsOutputMaskTest(unittest.TestCase):
	def test_winding_for_layer_uses_depth_windings(self) -> None:
		params = {"depth_windings": [-2, -1, 0, 1]}

		self.assertEqual(fit2tifxyz._winding_for_layer(0, params), -2.0)
		self.assertEqual(fit2tifxyz._winding_for_layer(2, params), 0.0)
		self.assertEqual(fit2tifxyz._winding_for_layer(3, params), 1.0)

	def test_checkpoint_load_preserves_depth_windings(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": {
				"mesh_step": 1,
				"winding_step": 1,
				"subsample_mesh": 1,
				"subsample_winding": 1,
				"scaledown": 1.0,
				"z_step_eff": 1,
				"volume_extent": None,
				"pyramid_d": False,
				"depth_windings": [-2],
			},
		}

		mdl = model.Model3D.from_checkpoint(state, device=torch.device("cpu"))

		self.assertEqual(mdl.params.depth_windings, (-2,))

	def test_checkpoint_load_requires_depth_windings(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": {
				"mesh_step": 1,
				"winding_step": 1,
				"subsample_mesh": 1,
				"subsample_winding": 1,
				"scaledown": 1.0,
				"z_step_eff": 1,
				"volume_extent": None,
				"pyramid_d": False,
			},
		}

		with self.assertRaisesRegex(ValueError, "depth_windings"):
			model.Model3D.from_checkpoint(state, device=torch.device("cpu"))

	def test_winding_results_store_model_surface_locations(self) -> None:
		result = opt_loss_corr._build_winding_results(
			winding_obs=torch.tensor([0.0], dtype=torch.float32),
			target=torch.tensor([0.0], dtype=torch.float32),
			err=torch.tensor([0.125], dtype=torch.float32),
			pt_ids=torch.tensor([9], dtype=torch.int64),
			col=torch.tensor([7], dtype=torch.int64),
			pts=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
			winda=torch.tensor([0.0], dtype=torch.float32),
			valid=torch.tensor([True]),
			is_absolute=torch.tensor([True]),
			point_normal=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
			target_normal=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
			normal_alignment=torch.tensor([1.0], dtype=torch.float32),
			model_loc_d=torch.tensor([[0, -1]], dtype=torch.int64),
			model_loc_h=torch.tensor([[2.25, float("nan")]], dtype=torch.float32),
			model_loc_w=torch.tensor([[3.5, float("nan")]], dtype=torch.float32),
			model_loc_weight=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
			model_loc_residual=torch.tensor([[0.125, float("nan")]], dtype=torch.float32),
			model_loc_valid=torch.tensor([[True, False]]),
		)

		entry = result["points_list"][0]
		self.assertEqual(entry["collection_id"], 7)
		self.assertEqual(entry["point_id"], 9)
		self.assertEqual(entry["model_locations"], [
			{
				"anchor": "avg_low",
				"d": 0,
				"h": 2.25,
				"w": 3.5,
				"weight": 1.0,
				"residual": 0.125,
			}
		])


class Fit2TifxyzOutputMaskSmokeTest(unittest.TestCase):
	def test_checkpoint_export_writes_atlas_control_results_and_meta_summary(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": _model_params(),
			"_atlas_control_points_results_": {
				"format": "lasagna_atlas_control_points_results",
				"version": 1,
				"summary": {
					"total_count": 1,
					"control_count": 1,
					"valid_count": 1,
					"max_distance": 2.0,
					"rms_distance": 2.0,
				},
				"records": [{
					"fiber_id": "fiber_a",
					"object_id": "fiber_a",
					"source_index": 3,
					"control_index": 0,
					"target_xyz": [1.0, 2.0, 3.0],
					"mesh_xyz": [1.0, 2.0, 1.0],
					"model_h": 1.0,
					"model_w": 2.0,
					"distance": 2.0,
					"signed_delta": -2.0,
					"valid": True,
				}],
				"fibers": [],
			},
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main(["--input", str(ckpt), "--output", str(out)])

			tifxyz = out / "winding_0000.tifxyz"
			meta = json.loads((tifxyz / "meta.json").read_text(encoding="utf-8"))
			results = json.loads((tifxyz / "atlas_control_points_results.json").read_text(encoding="utf-8"))

		self.assertEqual(meta["atlas_control_points_results"]["path"], "atlas_control_points_results.json")
		self.assertEqual(meta["atlas_control_points_results"]["total_count"], 1)
		self.assertEqual(meta["atlas_control_points_results"]["valid_count"], 1)
		self.assertEqual(results["records"][0]["fiber_id"], "fiber_a")
		self.assertEqual(results["records"][0]["mesh_xyz"], [1.0, 2.0, 1.0])
		self.assertEqual(results["fibers"][0]["control_points"][0]["control_index"], 0)

	def test_checkpoint_export_does_not_scale_atlas_control_results(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		params = _model_params()
		params["lasagna_base_shape_zyx"] = [4, 4, 4]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": params,
			"_atlas_control_points_results_": {
				"format": "lasagna_atlas_control_points_results",
				"version": 1,
				"records": [{
					"fiber_id": "fiber_a",
					"object_id": "fiber_a",
					"source_index": 3,
					"target_xyz": [1.0, 2.0, 3.0],
					"mesh_xyz": [4.0, 5.0, 6.0],
					"model_h": 1.0,
					"model_w": 2.0,
					"distance": 7.0,
					"signed_delta": -8.0,
					"valid": True,
				}],
			},
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main([
				"--input", str(ckpt),
				"--output", str(out),
				"--target-volume-shape-zyx", "8", "8", "8",
			])

			results = json.loads(
				(out / "winding_0000.tifxyz" / "atlas_control_points_results.json")
					.read_text(encoding="utf-8")
			)

		record = results["records"][0]
		self.assertEqual(record["target_xyz"], [1.0, 2.0, 3.0])
		self.assertEqual(record["mesh_xyz"], [4.0, 5.0, 6.0])
		self.assertEqual(record["distance"], 7.0)
		self.assertEqual(record["signed_delta"], -8.0)

	def test_checkpoint_export_masks_xyz_d_and_area(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": {
				"mesh_step": 1,
				"winding_step": 1,
				"subsample_mesh": 1,
				"subsample_winding": 1,
				"scaledown": 1.0,
				"z_step_eff": 1,
				"volume_extent": None,
				"pyramid_d": False,
				"depth_windings": [0],
			},
			"_approval_inpaint_output_mask_": _square_payload(radius=0),
			"_corr_points_results_": _square_corr_results(),
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main(["--input", str(ckpt), "--output", str(out)])

			tifxyz = out / "winding_0000.tifxyz"
			x_out = tifffile.imread(str(tifxyz / "x.tif"))
			d_out = tifffile.imread(str(tifxyz / "d.tif"))
			meta = json.loads((tifxyz / "meta.json").read_text(encoding="utf-8"))

		expected = np.zeros((7, 7), dtype=bool)
		expected[2:5, 2:5] = True
		self.assertTrue(np.all(x_out[expected] != -1.0))
		self.assertTrue(np.all(x_out[~expected] == -1.0))
		self.assertTrue(np.all(d_out[expected] == 0.0))
		self.assertTrue(np.all(d_out[~expected] == -1.0))
		self.assertEqual(meta["area_vx2"], 4.0)

	def test_checkpoint_export_writes_flow_gate_channels_with_output_mask(self) -> None:
		x, y, z = _plane_mesh(7, 7)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		local = np.linspace(-0.5, 1.5, 49, dtype=np.float32).reshape(1, 7, 7)
		normalized = np.full((1, 7, 7), 0.75, dtype=np.float32)
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": _model_params(),
			"_fit_config_": {"args": {"tifxyz-flow-gate-channels": True}},
			"_approval_inpaint_output_mask_": _square_payload(radius=0),
			"_corr_points_results_": _square_corr_results(),
			"_flow_gate_channels_": _flow_gate_payload(local, normalized),
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main(["--input", str(ckpt), "--output", str(out)])

			tifxyz = out / "winding_0000.tifxyz"
			local_out = tifffile.imread(str(tifxyz / "flow_gate_local_contrast.tif"))
			normalized_out = tifffile.imread(str(tifxyz / "flow_gate_component_normalized.tif"))
			meta = json.loads((tifxyz / "meta.json").read_text(encoding="utf-8"))

		expected_mask = np.zeros((7, 7), dtype=bool)
		expected_mask[2:5, 2:5] = True
		self.assertEqual(local_out.dtype, np.float32)
		self.assertTrue(float(local_out.min()) >= 0.0)
		self.assertTrue(float(local_out.max()) <= 1.0)
		self.assertTrue(np.all(local_out[~expected_mask] == 0.0))
		self.assertTrue(np.all(normalized_out[~expected_mask] == 0.0))
		np.testing.assert_allclose(normalized_out[expected_mask], 0.75)
		self.assertEqual(
			[name["name"] for name in meta["extra_channels"]],
			["flow_gate_local_contrast", "flow_gate_component_normalized"],
		)
		self.assertEqual(meta["extra_channels"][0]["source_config"]["local_boost"], 0.5)

	def test_single_segment_export_concatenates_flow_gate_channels_with_border_zeroes(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack(
			[
				np.stack([x, x + 10.0], axis=0),
				np.stack([y, y], axis=0),
				np.stack([z, z + 1.0], axis=0),
			],
			axis=0,
		)
		local = np.stack([
			np.full((3, 3), 0.25, dtype=np.float32),
			np.full((3, 3), 0.5, dtype=np.float32),
		], axis=0)
		normalized = np.stack([
			np.full((3, 3), 0.75, dtype=np.float32),
			np.full((3, 3), 1.0, dtype=np.float32),
		], axis=0)
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": _model_params(depth_windings=(0, 1)),
			"_fit_config_": {"args": {"tifxyz-flow-gate-channels": True}},
			"_flow_gate_channels_": _flow_gate_payload(local, normalized),
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main(["--input", str(ckpt), "--output", str(out), "--single-segment"])

			tifxyz = out / "winding_.tifxyz"
			local_out = tifffile.imread(str(tifxyz / "flow_gate_local_contrast.tif"))

		self.assertEqual(local_out.shape, (3, 8))
		np.testing.assert_allclose(local_out[:, :3], 0.25)
		np.testing.assert_allclose(local_out[:, 3:5], 0.0)
		np.testing.assert_allclose(local_out[:, 5:8], 0.5)

	def test_flow_gate_channels_on_requires_payload(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": _model_params(),
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			with self.assertRaisesRegex(ValueError, "no _flow_gate_channels_ payload"):
				fit2tifxyz.main([
					"--input", str(ckpt),
					"--output", str(out),
					"--flow-gate-channels", "on",
				])

	def test_checkpoint_export_observes_cancel_callback(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": {
				"mesh_step": 1,
				"winding_step": 1,
				"subsample_mesh": 1,
				"subsample_winding": 1,
				"scaledown": 1.0,
				"z_step_eff": 1,
				"volume_extent": None,
				"pyramid_d": False,
				"depth_windings": [0],
			},
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			calls = 0

			def cancel() -> None:
				nonlocal calls
				calls += 1
				raise KeyboardInterrupt("cancelled")

			with self.assertRaises(KeyboardInterrupt):
				fit2tifxyz.main(["--input", str(ckpt), "--output", str(out)], cancel_fn=cancel)
			self.assertGreaterEqual(calls, 1)

	def test_checkpoint_export_scales_to_target_volume_shape(self) -> None:
		x, y, z = _plane_mesh(3, 3)
		mesh_flat = np.stack([x, y, z], axis=0)[:, None, :, :]
		state = {
			"mesh_flat": torch.from_numpy(mesh_flat.astype(np.float32)),
			"_model_params_": {
				"mesh_step": 2,
				"winding_step": 1,
				"subsample_mesh": 1,
				"subsample_winding": 1,
				"scaledown": 1.0,
				"z_step_eff": 1,
				"volume_extent": None,
				"pyramid_d": False,
				"lasagna_base_shape_zyx": [100, 100, 100],
				"depth_windings": [0],
			},
		}

		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			ckpt = root / "model.pt"
			out = root / "out"
			torch.save(state, ckpt)

			fit2tifxyz.main([
				"--input", str(ckpt),
				"--output", str(out),
				"--target-volume-shape-zyx", "50", "50", "50",
			])

			tifxyz = out / "winding_0000.tifxyz"
			x_out = tifffile.imread(str(tifxyz / "x.tif"))
			meta = json.loads((tifxyz / "meta.json").read_text(encoding="utf-8"))

		np.testing.assert_allclose(x_out, x * 0.5)
		self.assertEqual(meta["base_shape_zyx"], [50, 50, 50])
		self.assertEqual(meta["lasagna_base_shape_zyx"], [100, 100, 100])
		self.assertEqual(meta["scale"], [1.0, 1.0])
		self.assertEqual(meta["bbox"], [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
		self.assertEqual(meta["area_vx2"], 4.0)


if __name__ == "__main__":
	unittest.main()
