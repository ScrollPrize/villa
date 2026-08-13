from __future__ import annotations

import os
import tempfile
import unittest

import torch

import fit_data
import model
import optimizer


class OptimizerExpandStageTest(unittest.TestCase):
	def test_snap_surf_map_fixture_export_paths_resolve_under_out_dir(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			resolved = optimizer._resolve_snap_surf_map_fixture_export_args(
				{
					"fixture_export_dir": "flat_fixture",
					"export_fixture": {
						"dir": "nested_fixture",
						"once": True,
						"objs": False,
					},
					"fixture_export": {
						"path": os.path.join(tmp, "absolute_fixture"),
					},
				},
				out_dir=tmp,
			)

			self.assertEqual(resolved["fixture_export_dir"], os.path.join(tmp, "flat_fixture"))
			self.assertEqual(resolved["export_fixture"]["dir"], os.path.join(tmp, "nested_fixture"))
			self.assertEqual(resolved["fixture_export"]["path"], os.path.join(tmp, "absolute_fixture"))

	def test_expand_z_wrapper_parses_nested_stages_and_counts_steps(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 0.0,
				"snap_surf_map": 0.1,
			},
			"stages": [
				{
					"name": "init_center",
					"steps": 7,
					"lr": 1.0,
					"params": ["mesh_ms"],
				},
				{
					"name": "expand-z",
					"grow": {"d_pos": 1},
					"stages": [
						{
							"name": "expand_up",
							"steps": 11,
							"lr": 0.5,
							"params": ["mesh_ms"],
						}
					],
				},
			],
		})

		self.assertEqual(len(stages), 2)
		self.assertEqual(stages[1].name, "expand-z")
		self.assertIsNone(stages[1].global_opt)
		self.assertEqual(stages[1].grow, {"d_pos": 1})
		self.assertEqual(len(stages[1].children), 1)
		self.assertEqual(stages[1].children[0].name, "expand_up")
		self.assertEqual(optimizer.total_steps_for_stages(stages), 18)

	def test_expand_z_requires_nested_stage_list(self) -> None:
		with self.assertRaisesRegex(ValueError, "requires non-empty nested 'stages'"):
			optimizer.load_stages_cfg({
				"stages": [
					{"name": "expand-z"}
				],
			})

	def test_expand_z_noop_child_appends_copied_reference_layer(self) -> None:
		device = torch.device("cpu")
		yy = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3)
		xx = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3)
		xyz = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
		valid = torch.ones(3, 3, dtype=torch.bool)
		mdl = model.Model3D.from_tifxyz_crop(
			xyz,
			valid,
			device=device,
			mesh_step=1,
			winding_step=1,
			subsample_mesh=1,
			subsample_winding=1,
			depth=1,
		)
		data = fit_data.FitData3D(
			cos=None,
			grad_mag=torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
			nx=None,
			ny=None,
			pred_dt=None,
			corr_points=None,
			winding_volume=None,
			origin_fullres=(0.0, 0.0, 0.0),
			spacing=(1.0, 1.0, 1.0),
			cuda_gridsample=False,
		)
		stages = optimizer.load_stages_cfg({
			"stages": [
				{
					"name": "expand-z",
					"stages": [
						{
							"name": "noop",
							"steps": 0,
							"lr": 1.0,
							"params": ["mesh_ms"],
						}
					],
				}
			],
		})

		optimizer.optimize(
			model=mdl,
			data=data,
			stages=stages,
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			init_grow={"order": ("up",), "step": 1, "target_depth": 2},
		)

		self.assertEqual(mdl.depth, 2)
		self.assertEqual(mdl.params.depth_windings, (0, 1))
		grid = mdl._grid_xyz().detach()
		self.assertTrue(torch.allclose(grid[0], grid[1], atol=1.0e-6))

	def test_expand_z_appends_boundary_self_maps_into_self_map_state(self) -> None:
		device = torch.device("cpu")
		yy = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3)
		xx = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3)
		xyz = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
		valid = torch.ones(3, 3, dtype=torch.bool)
		mdl = model.Model3D.from_tifxyz_crop(
			xyz,
			valid,
			device=device,
			mesh_step=1,
			winding_step=1,
			subsample_mesh=1,
			subsample_winding=1,
			depth=1,
		)
		data = fit_data.FitData3D(
			cos=None,
			grad_mag=torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
			nx=None,
			ny=None,
			pred_dt=None,
			corr_points=None,
			winding_volume=None,
			origin_fullres=(0.0, 0.0, 0.0),
			spacing=(1.0, 1.0, 1.0),
			cuda_gridsample=False,
		)
		stages = optimizer.load_stages_cfg({
			"base": {
				"step": 1.0,
				"normal": 0.0,
				"snap_surf_map": 0.1,
				"map_dist": 0.0001,
				"map_vec_normal": 1.0,
				"map_surface_normal": 1.0,
			},
			"stages": [
				{
					"name": "expand-z",
					"stages": [
						{
							"name": "expand_up",
							"steps": 1,
							"lr": 0.0,
							"params": ["mesh_ms"],
							"w_fac": {
								"snap_surf_map": 0.0,
							},
							"args": {
								"snap_surf_map": {
									"map_opt": {
										"steps": 0,
										"lr": 0.01,
										"params": ["map_surf_ms"],
									}
								}
							},
						}
					],
				}
			],
		})

		optimizer.optimize(
			model=mdl,
			data=data,
			stages=stages,
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			init_grow={"order": ("up",), "step": 1, "target_depth": 3},
			self_map_init="multi_wrap_d",
		)

		self.assertEqual(mdl.depth, 3)
		state = getattr(mdl, "_snap_surf_map_state_for_save", None)
		self.assertIsInstance(state, dict)
		self.assertNotIn("global_map", state)
		self.assertIn("out", state["self_maps"])
		self.assertIn("in", state["self_maps"])
		self.assertEqual(state["self_maps"]["out"]["map_uv_ms"][0].shape[0], 2)
		self.assertEqual(state["self_maps"]["in"]["map_uv_ms"][0].shape[0], 2)
		self.assertEqual(state["self_maps"]["out"]["direction"], "out")
		self.assertEqual(state["self_maps"]["in"]["direction"], "in")
		reopt_stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 0.0,
				"snap_surf_map": 0.1,
			},
			"stages": [
				{
					"name": "reopt",
					"steps": 1,
					"lr": 0.0,
					"params": ["mesh_ms"],
				}
			],
		})
		optimizer.optimize(
			model=mdl,
			data=data,
			stages=reopt_stages,
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			self_map_init="multi_wrap_d",
			snap_surf_map_state=state,
			require_snap_surf_map_state=True,
		)

	def test_self_d_snap_surf_maps_are_published_and_required_for_reopt(self) -> None:
		device = torch.device("cpu")
		yy = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3)
		xx = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3)
		xyz = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
		valid = torch.ones(3, 3, dtype=torch.bool)
		mdl = model.Model3D.from_tifxyz_crop(
			xyz,
			valid,
			device=device,
			mesh_step=1,
			winding_step=1,
			subsample_mesh=1,
			subsample_winding=1,
			depth=2,
		)
		data = fit_data.FitData3D(
			cos=None,
			grad_mag=torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
			nx=None,
			ny=None,
			pred_dt=None,
			corr_points=None,
			winding_volume=None,
			origin_fullres=(0.0, 0.0, 0.0),
			spacing=(1.0, 1.0, 1.0),
			cuda_gridsample=False,
		)
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 0.0,
				"snap_surf_map": 0.1,
				"map_dist": 0.0001,
				"map_vec_normal": 1.0,
				"map_surface_normal": 1.0,
				"map_smooth": 0.05,
				"map_bend": 0.01,
				"map_jac": 1.0,
				"map_metric_smooth": 0.05,
				"map_area_smooth": 0.02,
				"map_dense_prior": 0.001,
			},
			"stages": [
				{
					"name": "snap",
					"steps": 1,
					"lr": 0.0,
					"params": ["mesh_ms"],
					"args": {
						"snap_surf_map": {
							"map_opt": {
								"steps": 0,
								"lr": 0.01,
								"params": ["map_surf_ms"],
							}
						}
					},
				}
			],
		})

		optimizer.optimize(
			model=mdl,
			data=data,
			stages=stages,
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			self_map_init="multi_wrap_d",
		)

		state = getattr(mdl, "_snap_surf_map_state_for_save", None)
		self.assertIsInstance(state, dict)
		self.assertIn("out", state["self_maps"])
		self.assertIn("in", state["self_maps"])
		self.assertTrue(state["self_maps"]["out"]["map_uv_ms"])

		with self.assertRaisesRegex(ValueError, "requires checkpoint snap-surf self maps"):
			optimizer.optimize(
				model=mdl,
				data=data,
				stages=[],
				snapshot_interval=0,
				snapshot_fn=lambda **_kw: None,
				self_map_init="multi_wrap_d",
				require_snap_surf_map_state=True,
			)

		optimizer.optimize(
			model=mdl,
			data=data,
			stages=[],
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			self_map_init="multi_wrap_d",
			snap_surf_map_state=state,
			require_snap_surf_map_state=True,
		)


if __name__ == "__main__":
	unittest.main()
