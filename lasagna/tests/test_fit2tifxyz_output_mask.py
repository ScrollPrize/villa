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


class Fit2TifxyzOutputMaskHelpersTest(unittest.TestCase):
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


if __name__ == "__main__":
	unittest.main()
