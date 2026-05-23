from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from snap_surf import map_fixture_cli
from snap_surf_test_utils import _plane_xyz, _result, opt_loss_snap_surf


class SnapSurfMapFixtureTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()

	def _write_tiny_fixture(self, fixture_dir: str) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 1,
					"iters": 1,
					"seed_opt_iters": 1,
					"grow_opt_iters": 1,
					"first_global_opt_iters": 1,
					"last_global_opt_iters": 0,
					"subdiv": 1,
					"seed_radius": 1,
					"fixture_export_dir": fixture_dir,
					"fixture_export_objs": False,
				}
			},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)
		opt_loss_snap_surf.set_debug_step(0, label="fixture")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

	def test_map_fixture_export_writes_reference_and_geometry(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			self._write_tiny_fixture(tmp)

			for rel in (
				"fixture.json",
				"map/model_x.tif",
				"map/model_y.tif",
				"map/active_quad.tif",
				"map/blocked_quad.tif",
				"ext_surface/x.tif",
				"ext_surface/y.tif",
				"ext_surface/z.tif",
				"ext_surface/valid.tif",
				"ext_surface/quad_valid.tif",
				"model_stack/x.tif",
				"model_stack/y.tif",
				"model_stack/z.tif",
				"model_stack/valid.tif",
				"ext_normals/x.tif",
				"model_normals/x.tif",
			):
				self.assertTrue(os.path.exists(os.path.join(tmp, rel)), rel)

			meta = json.loads(Path(tmp, "fixture.json").read_text(encoding="utf-8"))
			self.assertEqual(meta["schema_version"], 1)
			self.assertEqual(meta["model_depth"], 0)
			self.assertEqual(meta["surface_index"], 0)
			self.assertEqual(meta["seed_xyz"], [1.5, 1.5, 0.0])
			self.assertEqual(meta["map_init_config"]["fixture_export_dir"], tmp)
			self.assertGreater(meta["map_counts"]["active_quads"], 0)

			fixture = opt_loss_snap_surf.load_map_fixture(tmp)
			self.assertEqual(tuple(fixture.reference_uv.shape), (4, 4, 2))
			self.assertEqual(tuple(fixture.model_xyz.shape), (1, 4, 4, 3))
			self.assertEqual(tuple(fixture.ext_xyz.shape), (4, 4, 3))

	def test_map_fixture_cli_compare_reference_is_near_zero(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "rerun")
			self._write_tiny_fixture(fixture_dir)

			rc = map_fixture_cli.main(["compare-reference", fixture_dir, "--out", out_dir, "--device", "cpu"])

			self.assertEqual(rc, 0)
			metrics = json.loads(Path(out_dir, "metrics.json").read_text(encoding="utf-8"))
			self.assertEqual(metrics["active_quad_diff"], 0)
			self.assertEqual(metrics["blocked_quad_diff"], 0)
			self.assertLessEqual(metrics["model_x_max_abs_delta"], 1.0e-6)
			self.assertLessEqual(metrics["model_y_max_abs_delta"], 1.0e-6)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "map", "model_x.tif")))


if __name__ == "__main__":
	unittest.main()
