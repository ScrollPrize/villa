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


def _write_tifxyz(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)
	rows, cols = 3, 4
	x = np.zeros((rows, cols), dtype=np.float32)
	y = np.zeros((rows, cols), dtype=np.float32)
	z = np.zeros((rows, cols), dtype=np.float32)
	for r in range(rows):
		for c in range(cols):
			src_c = 0 if c == cols - 1 else c
			x[r, c] = float(src_c)
			y[r, c] = float(r)
	tifffile.imwrite(path / "x.tif", x)
	tifffile.imwrite(path / "y.tif", y)
	tifffile.imwrite(path / "z.tif", z)
	(path / "meta.json").write_text(json.dumps({"scale": [1.0, 1.0, 1.0]}) + "\n", encoding="utf-8")


class AtlasParserTest(unittest.TestCase):
	def test_atlas_init_unwraps_period_and_places_actual_u(self) -> None:
		with self.subTest("synthetic atlas"):
			import tempfile
			with tempfile.TemporaryDirectory() as td:
				root = Path(td)
				base = root / "base_mesh.tifxyz"
				_write_tifxyz(base)
				fiber = root / "fiber.json"
				fiber.write_text(json.dumps({
					"type": "vc3d_fiber",
					"version": 1,
					"line_points": [[10.0, 20.0, 30.0]],
					"control_points": [[10.0, 20.0, 30.0]],
				}) + "\n", encoding="utf-8")
				mapping = root / "mapping.json"
				mapping.write_text(json.dumps({
					"type": "vc3d_atlas_fiber_mapping",
					"version": 2,
					"fiber_path": "fibers/fiber.json",
					"winding_offset": 0,
					"line_anchors": [{
						"source_index": 0,
						"world": [1.0, 1.0, 0.0],
						"atlas": [1.0, 1.0],
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
						"winding_offset": 2,
					}],
				}

				init = atlas.build_atlas_init(
					atlas_obj,
					device=torch.device("cpu"),
					mesh_step=1,
					winding_step=1,
					subsample_mesh=1,
					subsample_winding=1,
					depth=1,
				)

				self.assertEqual(init.metadata["period_columns"], 3)
				self.assertEqual(init.metadata["leftmost_winding"], 2)
				self.assertEqual(init.metadata["rightmost_winding"], 2)
				self.assertEqual(tuple(init.model._grid_xyz().shape), (1, 3, 3, 3))
				self.assertTrue(torch.allclose(init.atlas_lines.target_xyz[0], torch.tensor([10.0, 20.0, 30.0])))
				self.assertAlmostEqual(float(init.atlas_lines.model_h[0]), 1.0, delta=1.0e-6)
				self.assertAlmostEqual(float(init.atlas_lines.model_w[0]), 1.0, delta=1.0e-6)

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
