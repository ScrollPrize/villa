from __future__ import annotations

import os
import tempfile
import unittest

import torch

from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz, _result, opt_loss_snap_surf


class SnapSurfDebugObjTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(1.0, 1.0, 0.0),
			active=True,
		)

	def test_debug_obj_outputs_write_files_in_iteration_dir(self) -> None:
		model_xyz = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		with tempfile.TemporaryDirectory() as tmp:
			opt_loss_snap_surf.configure_snap_surf(
				cfg={
					"init_distance": 10.0,
					"debug_obj_dir": tmp,
					"debug_obj_interval": 1,
				},
				seed_xyz=(1.0, 1.0, 0.0),
				active=True,
			)
			opt_loss_snap_surf.set_debug_step(0, label="stageX_initial")

			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

			out = os.path.join(tmp, "stageX_initial_step000000")
			self.assertTrue(os.path.isdir(out))
			self.assertTrue(os.path.exists(os.path.join(out, "ext_surface.obj")))
			self.assertTrue(os.path.exists(os.path.join(out, "model_surface.obj")))
			self.assertTrue(os.path.exists(os.path.join(out, "corr_model_to_ext.obj")))
			self.assertTrue(os.path.exists(os.path.join(out, "corr_ext_to_model.obj")))

	def test_map_init_debug_obj_outputs_write_files(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		with tempfile.TemporaryDirectory() as tmp:
			opt_loss_snap_surf.configure_snap_surf(
				cfg={
					"debug_obj_dir": tmp,
					"debug_obj_interval": 1,
					"map_init": {"enabled": True, "subdiv": 1, "iters": 1, "grow_opt_iters": 1},
				},
				seed_xyz=(1.5, 1.5, 0.0),
				active=True,
			)
			opt_loss_snap_surf.set_debug_step(0, label="map_init")
			out = os.path.join(tmp, "map_init_step000000")
			os.makedirs(out, exist_ok=True)
			stale_corr = os.path.join(out, "corr_ext_to_model.obj")
			with open(stale_corr, "w", encoding="utf-8") as f:
				f.write("stale previous grow\n")

			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

			for name in ("ext_surface.obj", "model_surface.obj", "map_mapped_surface.obj", "map_ext_to_model.obj", "map_active_mask.obj"):
				path = os.path.join(out, name)
				self.assertTrue(os.path.exists(path), name)
				self.assertGreater(os.path.getsize(path), 0, name)
			with open(stale_corr, "r", encoding="utf-8") as f:
				self.assertIn("map_init_no_corr_ext_to_model", f.read())

	def test_map_init_debug_obj_outputs_scale_snapshots(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		with tempfile.TemporaryDirectory() as tmp:
			opt_loss_snap_surf.configure_snap_surf(
				cfg={
					"debug_obj_dir": tmp,
					"debug_obj_interval": 1,
					"map_init": {
						"enabled": True,
						"subdiv": 1,
						"iters": 3,
						"seed_opt_iters": 0,
						"grow_opt_iters": 1,
						"seed_radius": 0,
						"scale_levels": 3,
					},
				},
				seed_xyz=(2.0, 2.0, 0.0),
				active=True,
			)
			opt_loss_snap_surf.set_debug_step(0, label="map_init_scales")

			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

			scale_root = os.path.join(tmp, "map_init_scales_step000000", "map_init_scales")
			self.assertTrue(os.path.isdir(scale_root))
			snapshots = os.listdir(scale_root)
			for token in ("scale_l02", "scale_l01", "scale_l00"):
				self.assertIn(token, snapshots)
				for obj_name in ("map_mapped_surface.obj", "map_ext_to_model.obj", "map_active_mask.obj"):
					path = os.path.join(scale_root, token, obj_name)
					self.assertTrue(os.path.exists(path), path)
					self.assertGreater(os.path.getsize(path), 0, path)

if __name__ == "__main__":
	unittest.main()
