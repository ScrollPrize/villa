from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import approval_inpaint


def _write_tifxyz(
	path: Path,
	approval: np.ndarray,
	*,
	d_value: float | None = 2.0,
	xy_step: float = 10.0,
) -> None:
	path.mkdir(parents=True, exist_ok=True)
	h, w = approval.shape[:2]
	rows, cols = np.indices((h, w), dtype=np.float32)
	x = cols * float(xy_step)
	y = rows * float(xy_step)
	z = np.full((h, w), 100.0, dtype=np.float32)
	tifffile.imwrite(str(path / "x.tif"), x)
	tifffile.imwrite(str(path / "y.tif"), y)
	tifffile.imwrite(str(path / "z.tif"), z)
	if d_value is not None:
		d = np.full((h, w), d_value, dtype=np.float32)
		tifffile.imwrite(str(path / "d.tif"), d)
	tifffile.imwrite(str(path / "approval.tif"), approval)
	(path / "meta.json").write_text(
		json.dumps({"format": "tifxyz", "scale": [0.1, 0.1]}),
		encoding="utf-8",
	)


def _generated_positions(result: approval_inpaint.ApprovalInpaintResult) -> list[list[float]]:
	collections = result.corr_points["collections"]
	generated = [col for col in collections.values() if col.get("name") == "approval_inpaint"]
	if len(generated) != 1:
		raise AssertionError(f"expected one generated approval_inpaint collection, got {len(generated)}")
	return [point["p"] for point in generated[0]["points"].values()]


def _assert_points_inside_centered_extent(
	test: unittest.TestCase,
	result: approval_inpaint.ApprovalInpaintResult,
) -> None:
	half_w = 0.5 * float(result.model_w) + 1.0e-6
	half_h = 0.5 * float(result.model_h) + 1.0e-6
	for p in _generated_positions(result):
		test.assertLessEqual(abs(float(p[0]) - float(result.seed[0])), half_w)
		test.assertLessEqual(abs(float(p[1]) - float(result.seed[1])), half_h)


class ApprovalInpaintTest(unittest.TestCase):
	def test_load_approval_mask_uses_any_nonzero_rgb_channel(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			path = Path(td) / "approval.tif"
			rgb = np.zeros((2, 3, 3), dtype=np.uint8)
			rgb[0, 1, 1] = 10
			rgb[1, 2, 2] = 20
			tifffile.imwrite(str(path), rgb)

			mask = approval_inpaint.load_approval_mask(path, expected_shape=(2, 3))

		self.assertEqual(mask.tolist(), [[False, True, False], [False, False, True]])

	def test_build_generates_corr_points_seed_and_padded_extents(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0
		existing = {
			"collections": {
				"2": {
					"metadata": {"winding_is_absolute": True},
					"points": {"0": {"p": [0, 0, 100], "wind_a": 0}},
				}
			}
		}

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			result = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(30.0, 30.0, 100.0),
				mesh_step=10.0,
				existing_corr_points=existing,
			)

		self.assertEqual(result.seed, (30.0, 30.0, 100.0))
		self.assertEqual(result.model_w, 60)
		self.assertEqual(result.model_h, 60)
		self.assertEqual(result.index_bounds, (1, 5, 1, 5))
		self.assertGreaterEqual(result.point_count, 8)
		self.assertIn("2", result.corr_points["collections"])
		self.assertIn("3", result.corr_points["collections"])
		generated = result.corr_points["collections"]["3"]
		self.assertTrue(generated["metadata"]["winding_is_absolute"])
		self.assertIsNone(result.output_mask)
		for point in generated["points"].values():
			self.assertEqual(point["wind_a"], 2.0)
		_assert_points_inside_centered_extent(self, result)

	def test_output_mask_uses_generated_corr_collection(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			result = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(30.0, 30.0, 100.0),
				mesh_step=10.0,
				output_mask=True,
				output_mask_dilate=3,
			)

		self.assertIsNotNone(result.output_mask)
		mask = result.output_mask or {}
		self.assertEqual(mask["version"], 2)
		self.assertEqual(mask["source"], "corr_points")
		self.assertEqual(mask["corr_collection_ids"], [0])
		self.assertEqual(mask["dilation_radius"], 3)
		self.assertEqual(len(mask["corr_contours"]), 1)
		self.assertGreaterEqual(len(mask["corr_contours"][0]["point_ids"]), 4)
		self.assertIn("0", result.corr_points["collections"])
		self.assertEqual(result.corr_points["collections"]["0"]["name"], "approval_inpaint")

	def test_skeleton_points_are_emitted_in_loop_order(self) -> None:
		skeleton = np.zeros((9, 9), dtype=bool)
		skeleton[2, 3:6] = True
		skeleton[3:6, 6] = True
		skeleton[6, 3:6] = True
		skeleton[3:6, 2] = True

		contours = approval_inpaint.sample_skeleton_contours(skeleton, spacing_px=1.0)

		self.assertEqual(len(contours), 1)
		contour = contours[0]
		self.assertEqual(len(contour), int(skeleton.sum()))
		for a, b in zip(contour, contour[1:] + contour[:1]):
			self.assertLessEqual(max(abs(a[0] - b[0]), abs(a[1] - b[1])), 1)

	def test_output_mask_recenters_with_off_center_seed(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			result_a = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(20.0, 20.0, 100.0),
				mesh_step=10.0,
				output_mask=True,
			)
			result_b = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(40.0, 40.0, 100.0),
				mesh_step=10.0,
				output_mask=True,
			)

		self.assertEqual(result_a.seed, (30.0, 30.0, 100.0))
		mask_a = result_a.output_mask or {}
		mask_b = result_b.output_mask or {}
		self.assertEqual(mask_a["corr_collection_ids"], [0])
		self.assertEqual(mask_a["corr_collection_ids"], mask_b["corr_collection_ids"])

	def test_build_recenters_off_center_seed_from_approval_bounds(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			result_a = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(20.0, 20.0, 100.0),
				mesh_step=10.0,
			)
			result_b = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(40.0, 40.0, 100.0),
				mesh_step=10.0,
			)

		self.assertEqual(result_a.seed, (30.0, 30.0, 100.0))
		self.assertEqual(result_a.model_w, 60)
		self.assertEqual(result_a.model_h, 60)
		self.assertEqual(result_a.seed, result_b.seed)
		self.assertEqual(result_a.model_w, result_b.model_w)
		self.assertEqual(result_a.model_h, result_b.model_h)

	def test_build_sizes_from_actual_xyz_span_not_only_index_step(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval, xy_step=100.0)

			result = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(300.0, 300.0, 100.0),
				mesh_step=10.0,
				padding_frac=0.0,
			)

		self.assertEqual(result.seed, (300.0, 300.0, 100.0))
		self.assertEqual(result.model_w, 400)
		self.assertEqual(result.model_h, 400)

	def test_build_applies_padding_then_snaps_to_mesh_step(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			result_16 = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(30.0, 30.0, 100.0),
				mesh_step=16.0,
				corr_spacing=10.0,
			)
			result_25 = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(30.0, 30.0, 100.0),
				mesh_step=25.0,
				corr_spacing=10.0,
			)

		self.assertEqual(result_16.model_w, 64)
		self.assertEqual(result_16.model_h, 64)
		self.assertEqual(result_25.model_w, 75)
		self.assertEqual(result_25.model_h, 75)
		_assert_points_inside_centered_extent(self, result_16)
		_assert_points_inside_centered_extent(self, result_25)

	def test_missing_d_tif_defaults_generated_corr_points_to_zero(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval, d_value=None)

			result = approval_inpaint.build_approval_inpaint(
				tifxyz_path=tifxyz,
				seed=(30.0, 30.0, 100.0),
				mesh_step=10.0,
			)

		generated = next(iter(result.corr_points["collections"].values()))
		for point in generated["points"].values():
			self.assertEqual(point["wind_a"], 0.0)

	def test_missing_approval_tif_fails_in_lasagna(self) -> None:
		approval = np.zeros((7, 7), dtype=np.uint8)
		approval[1:6, 1:6] = 255
		approval[2:5, 2:5] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)
			(tifxyz / "approval.tif").unlink()

			with self.assertRaisesRegex(ValueError, "missing required file\\(s\\): approval.tif"):
				approval_inpaint.build_approval_inpaint(
					tifxyz_path=tifxyz,
					seed=(30.0, 30.0, 100.0),
					mesh_step=10.0,
				)

	def test_seed_must_project_to_unapproved_region(self) -> None:
		approval = np.zeros((5, 5), dtype=np.uint8)
		approval[1:4, 1:4] = 255
		approval[2, 2] = 0

		with tempfile.TemporaryDirectory() as td:
			tifxyz = Path(td) / "source.tifxyz"
			_write_tifxyz(tifxyz, approval)

			with self.assertRaisesRegex(ValueError, "does not project"):
				approval_inpaint.build_approval_inpaint(
					tifxyz_path=tifxyz,
					seed=(10.0, 10.0, 100.0),
					mesh_step=10.0,
				)


if __name__ == "__main__":
	unittest.main()
