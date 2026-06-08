from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import atlas


def _sample_model_xyz(xyz: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	return atlas._bilinear_sample_grid(xyz[0], h, w)


def _write_tifxyz(
	path: Path,
	*,
	rows: int = 3,
	cols: int = 4,
	row_step: float = 1.0,
	col_step: float = 1.0,
) -> None:
	path.mkdir(parents=True, exist_ok=True)
	x = np.zeros((rows, cols), dtype=np.float32)
	y = np.zeros((rows, cols), dtype=np.float32)
	z = np.zeros((rows, cols), dtype=np.float32)
	for r in range(rows):
		for c in range(cols):
			src_c = 0 if c == cols - 1 else c
			x[r, c] = float(src_c) * col_step
			y[r, c] = float(r) * row_step
	tifffile.imwrite(path / "x.tif", x)
	tifffile.imwrite(path / "y.tif", y)
	tifffile.imwrite(path / "z.tif", z)
	(path / "meta.json").write_text(json.dumps({"scale": [1.0, 1.0, 1.0]}) + "\n", encoding="utf-8")


class AtlasParserTest(unittest.TestCase):
	def test_atlas_init_crops_from_anchor_extents_with_margin(self) -> None:
		with self.subTest("synthetic atlas"):
			import tempfile
			with tempfile.TemporaryDirectory() as td:
				root = Path(td)
				base = root / "base_mesh.tifxyz"
				_write_tifxyz(base, row_step=8.0, col_step=4.0)
				fiber = root / "fiber.json"
				fiber.write_text(json.dumps({
					"type": "vc3d_fiber",
					"version": 1,
					"line_points": [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]],
					"control_points": [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
				}) + "\n", encoding="utf-8")
				mapping = root / "mapping.json"
				mapping.write_text(json.dumps({
					"type": "vc3d_atlas_fiber_mapping",
					"version": 2,
					"fiber_path": "fibers/fiber.json",
					"winding_offset": 99,
					"line_anchors": [{
						"source_index": 0,
						"world": [1.0, 1.0, 0.0],
						"atlas": [2.0, 1.0],
						"distance": 0.0,
					}, {
						"source_index": 1,
						"world": [2.0, 1.0, 0.0],
						"atlas": [8.0, 1.0],
						"distance": 0.0,
					}, {
						"source_index": 2,
						"world": [3.0, 1.0, 0.0],
						"atlas": [100.0, 1.0],
						"distance": 0.0,
					}],
					"control_anchors": [{
						"source_index": 0,
						"world": [1.0, 1.0, 0.0],
						"atlas": [2.0, 1.0],
						"distance": 0.0,
					}, {
						"source_index": 1,
						"world": [2.0, 1.0, 0.0],
						"atlas": [8.0, 1.0],
						"distance": 0.0,
					}],
				}) + "\n", encoding="utf-8")
				atlas_obj = {
					"type": "lasagna_atlas",
					"version": 1,
					"name": "a",
					"base": {"path": str(base), "ref": {"type": "atlas-base", "name": "a/base", "hash": "md5:" + "0" * 32}},
					"metadata": {"zero_winding_column": 0},
					"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
					"maps": [{
						"object_type": "line",
						"object_id": "fibers/fiber.json",
						"map_path": str(mapping),
						"winding_offset": 0,
					}],
				}

				init = atlas.build_atlas_init(
					atlas_obj,
					device=torch.device("cpu"),
					mesh_step=1,
					winding_step=1,
				)

				self.assertEqual(init.metadata["period_columns"], 3)
				self.assertEqual(init.metadata["leftmost_winding"], 0)
				self.assertEqual(init.metadata["rightmost_winding"], 2)
				self.assertEqual(init.metadata["init_margin_vx"], 4000)
				self.assertEqual(init.metadata["init_margin_rows"], 500)
				self.assertEqual(init.metadata["init_margin_columns"], 750)
				self.assertLess(init.metadata["init_margin_rows"], init.metadata["init_margin_vx"])
				self.assertLess(init.metadata["init_margin_columns"], init.metadata["init_margin_vx"])
				self.assertEqual(init.metadata["crop_row_start"], 0)
				self.assertEqual(init.metadata["crop_row_end"], 3)
				self.assertEqual(init.metadata["crop_column_start"], -748)
				self.assertEqual(init.metadata["crop_column_end"], 758)
				self.assertEqual(init.metadata["atlas_u_offset"], -748.0)
				self.assertEqual(init.metadata["control_point_sample_count"], 2)
				self.assertEqual(init.metadata["other_line_point_sample_count"], 0)
				self.assertEqual(init.metadata["requested_mesh_step"], 1)
				self.assertEqual(init.metadata["resampled_source_rows"], 3)
				self.assertEqual(init.metadata["resampled_source_columns"], 1506)
				self.assertEqual(tuple(init.model._grid_xyz().shape), (
					1,
					init.metadata["resampled_rows"],
					init.metadata["resampled_columns"],
					3,
				))
				self.assertEqual(int(init.model.params.mesh_step), 1)
				self.assertTrue(torch.allclose(init.atlas_lines.target_xyz[0], torch.tensor([10.0, 20.0, 30.0])))
				self.assertEqual(init.atlas_lines.source_indices, (0, 1))
				self.assertEqual(init.atlas_lines.is_control_point.tolist(), [True, True])
				row_scale = float(init.metadata["resampled_row_index_scale"])
				col_scale = float(init.metadata["resampled_column_index_scale"])
				self.assertAlmostEqual(float(init.atlas_lines.model_h[0]), 1.0 * row_scale, delta=1.0e-6)
				self.assertAlmostEqual(float(init.atlas_lines.model_w[0]), 750.0 * col_scale, delta=1.0e-4)
				self.assertAlmostEqual(float(init.atlas_lines.model_h[1]), 1.0 * row_scale, delta=1.0e-6)
				self.assertAlmostEqual(float(init.atlas_lines.model_w[1]), 756.0 * col_scale, delta=1.0e-4)

	def test_atlas_init_keeps_in_span_line_anchors_and_ignores_tails(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=4, cols=5, row_step=10.0, col_step=5.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[float(i), float(i + 1), float(i + 2)] for i in range(5)],
				"control_points": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [
					{"source_index": 0, "world": [0.0, 1.0, 2.0], "atlas": [-50.0, 1.0], "distance": 0.0},
					{"source_index": 2, "world": [2.0, 3.0, 4.0], "atlas": [2.0, 1.5], "distance": 0.0},
					{"source_index": 3, "world": [3.0, 4.0, 5.0], "atlas": [3.0, 2.0], "distance": 0.0},
					{"source_index": 4, "world": [4.0, 5.0, 6.0], "atlas": [80.0, 2.0], "distance": 0.0},
				],
				"control_anchors": [
					{"source_index": 0, "world": [1.0, 2.0, 3.0], "atlas": [1.0, 1.0], "distance": 0.0},
					{"source_index": 1, "world": [3.0, 4.0, 5.0], "atlas": [3.0, 2.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 1,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=1,
				winding_step=1,
			)

			self.assertEqual(init.metadata["period_columns"], 4)
			self.assertEqual(init.metadata["crop_column_start"], -529)
			self.assertEqual(init.metadata["crop_column_end"], 541)
			self.assertEqual(init.atlas_lines.source_indices, (0, 1, 2))
			self.assertEqual(init.atlas_lines.is_control_point.tolist(), [True, True, False])
			self.assertEqual(init.metadata["control_point_sample_count"], 2)
			self.assertEqual(init.metadata["other_line_point_sample_count"], 1)
			self.assertEqual(tuple(init.model._grid_xyz().shape), (
				1,
				init.metadata["resampled_rows"],
				init.metadata["resampled_columns"],
				3,
			))

	def test_atlas_init_mesh_step_coarsens_base_and_remaps_anchor_coords(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=5, cols=5, row_step=10.0, col_step=10.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
				"control_points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [],
				"control_anchors": [
					{"source_index": 0, "world": [1.0, 2.0, 3.0], "atlas": [1.0, 1.0], "distance": 0.0},
					{"source_index": 1, "world": [4.0, 5.0, 6.0], "atlas": [3.0, 3.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 0,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=20,
				winding_step=1,
			)

			self.assertEqual(int(init.model.params.mesh_step), 20)
			self.assertLess(init.metadata["resampled_rows"], init.metadata["resampled_source_rows"])
			self.assertLess(init.metadata["resampled_columns"], init.metadata["resampled_source_columns"])
			row_scale = float(init.metadata["resampled_row_index_scale"])
			col_scale = float(init.metadata["resampled_column_index_scale"])
			expected_h0 = (1.0 - float(init.metadata["crop_row_start"])) * row_scale
			expected_w0 = (1.0 - float(init.metadata["atlas_u_offset"])) * col_scale
			self.assertAlmostEqual(float(init.atlas_lines.model_h[0]), expected_h0, delta=1.0e-6)
			self.assertAlmostEqual(float(init.atlas_lines.model_w[0]), expected_w0, delta=1.0e-5)

	def test_atlas_init_resampled_anchor_hits_target_with_nonunit_base_step(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=9, cols=9, row_step=1000.0, col_step=1000.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[2000.0, 2000.0, 0.0], [4000.0, 4000.0, 0.0]],
				"control_points": [[2000.0, 2000.0, 0.0], [4000.0, 4000.0, 0.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [],
				"control_anchors": [
					{"source_index": 0, "world": [2000.0, 2000.0, 0.0], "atlas": [2.0, 2.0], "distance": 0.0},
					{"source_index": 1, "world": [4000.0, 4000.0, 0.0], "atlas": [4.0, 4.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 0,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=2000,
				winding_step=1,
			)

			model_xyz = init.model._grid_xyz().detach()
			h = init.atlas_lines.model_h
			w = init.atlas_lines.model_w
			hit = _sample_model_xyz(model_xyz, h, w)
			self.assertTrue(torch.allclose(hit, init.atlas_lines.target_xyz, atol=1.0e-4))

	def test_atlas_init_control_anchor_target_uses_control_points_not_line_index(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=9, cols=9, row_step=1000.0, col_step=1000.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[9999.0, 9999.0, 9999.0], [4000.0, 4000.0, 0.0]],
				"control_points": [[2000.0, 2000.0, 0.0], [4000.0, 4000.0, 0.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [],
				"control_anchors": [
					{"source_index": 0, "world": [1111.0, 1111.0, 1111.0], "atlas": [2.0, 2.0], "distance": 0.0},
					{"source_index": 1, "world": [4000.0, 4000.0, 0.0], "atlas": [4.0, 4.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 0,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=2000,
				winding_step=1,
			)

			self.assertTrue(torch.allclose(
				init.atlas_lines.target_xyz[0],
				torch.tensor([2000.0, 2000.0, 0.0]),
				atol=1.0e-6,
			))

	def test_atlas_init_control_anchor_span_uses_nearest_line_indices(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=7, cols=8, row_step=10.0, col_step=5.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[float(i), 0.0, 0.0] for i in range(8)],
				"control_points": [[2.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [
					{"source_index": 1, "world": [91.0, 0.0, 0.0], "atlas": [-20.0, 1.0], "distance": 0.0},
					{"source_index": 2, "world": [92.0, 0.0, 0.0], "atlas": [2.0, 1.5], "distance": 0.0},
					{"source_index": 3, "world": [93.0, 0.0, 0.0], "atlas": [3.0, 2.0], "distance": 0.0},
					{"source_index": 4, "world": [94.0, 0.0, 0.0], "atlas": [4.0, 2.5], "distance": 0.0},
					{"source_index": 5, "world": [95.0, 0.0, 0.0], "atlas": [5.0, 3.0], "distance": 0.0},
					{"source_index": 6, "world": [96.0, 0.0, 0.0], "atlas": [30.0, 3.5], "distance": 0.0},
				],
				"control_anchors": [
					{"source_index": 0, "world": [222.0, 0.0, 0.0], "atlas": [2.0, 1.0], "distance": 0.0},
					{"source_index": 1, "world": [555.0, 0.0, 0.0], "atlas": [5.0, 3.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 0,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=1,
				winding_step=1,
			)

			self.assertEqual(init.atlas_lines.source_indices, (0, 1, 3, 4))
			self.assertEqual(init.atlas_lines.is_control_point.tolist(), [True, True, False, False])
			self.assertTrue(torch.allclose(
				init.atlas_lines.target_xyz,
				torch.tensor([
					[2.1, 0.0, 0.0],
					[4.9, 0.0, 0.0],
					[3.0, 0.0, 0.0],
					[4.0, 0.0, 0.0],
				]),
				atol=1.0e-6,
			))

	def test_atlas_init_line_anchor_target_uses_line_points_before_world(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			base = root / "base_mesh.tifxyz"
			_write_tifxyz(base, rows=5, cols=5, row_step=10.0, col_step=10.0)
			fiber = root / "fiber.json"
			fiber.write_text(json.dumps({
				"type": "vc3d_fiber",
				"version": 1,
				"line_points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
				"control_points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
			}) + "\n", encoding="utf-8")
			mapping = root / "mapping.json"
			mapping.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"fiber_path": "fibers/fiber.json",
				"line_anchors": [
					{"source_index": 1, "world": [99.0, 99.0, 99.0], "atlas": [1.0, 1.5], "distance": 0.0},
				],
				"control_anchors": [
					{"source_index": 0, "world": [0.0, 0.0, 0.0], "atlas": [0.0, 1.0], "distance": 0.0},
					{"source_index": 1, "world": [2.0, 0.0, 0.0], "atlas": [2.0, 2.0], "distance": 0.0},
				],
			}) + "\n", encoding="utf-8")
			atlas_obj = {
				"type": "lasagna_atlas",
				"version": 1,
				"name": "a",
				"base": {"path": str(base)},
				"metadata": {"zero_winding_column": 0},
				"objects": {"line": [{"id": "fibers/fiber.json", "path": str(fiber)}]},
				"maps": [{
					"object_type": "line",
					"object_id": "fibers/fiber.json",
					"map_path": str(mapping),
					"winding_offset": 0,
				}],
			}

			init = atlas.build_atlas_init(
				atlas_obj,
				device=torch.device("cpu"),
				mesh_step=1,
				winding_step=1,
			)

			self.assertEqual(init.atlas_lines.source_indices, (0, 1, 1))
			self.assertEqual(init.atlas_lines.is_control_point.tolist(), [True, True, False])
			self.assertTrue(torch.allclose(
				init.atlas_lines.target_xyz[2],
				torch.tensor([1.0, 0.0, 0.0]),
				atol=1.0e-6,
			))

	def test_wrapped_base_crop_allows_negative_start_and_multiple_wraps(self) -> None:
		xyz = torch.tensor([[
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[2.0, 0.0, 0.0],
			[0.0, 0.0, 0.0],
		]], dtype=torch.float32)
		valid = torch.ones((1, 4), dtype=torch.bool)

		crop_xyz, crop_valid = atlas._crop_wrapped_base_shell(
			xyz,
			valid,
			row_start=0,
			row_end=1,
			column_start=-5,
			column_end=4,
		)

		self.assertEqual(tuple(crop_xyz.shape), (1, 9, 3))
		self.assertTrue(bool(crop_valid.all()))
		self.assertTrue(torch.allclose(crop_xyz[0, :, 0], torch.tensor([1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0])))

	def test_mapping_loader_accepts_vc3d_mapping_schema(self) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as td:
			path = Path(td) / "mapping.json"
			path.write_text(json.dumps({
				"type": "vc3d_atlas_fiber_mapping",
				"version": 2,
				"line_anchors": [{"source_index": 0, "atlas": [1.0, 2.0], "world": [0.0, 0.0, 0.0]}],
			}), encoding="utf-8")
			obj = atlas.load_vc3d_atlas_fiber_mapping(path)
			self.assertEqual(obj["line_anchors"][0]["source_index"], 0)


if __name__ == "__main__":
	unittest.main()
