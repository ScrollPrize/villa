from __future__ import annotations

import os
import tempfile
import unittest

import torch

from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz, _result, opt_loss_snap_surf


class SnapSurfMapGrowthTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(1.0, 1.0, 0.0),
			active=True,
		)

	def test_map_init_returns_zero_loss_and_zero_model_gradient(self) -> None:
		model_xyz = (_plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 1, "iters": 2, "grow_opt_iters": 1}},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		loss.backward()

		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertIsNotNone(model_xyz.grad)
		self.assertAlmostEqual(float(model_xyz.grad.abs().sum().detach()), 0.0, places=6)
		self.assertGreater(opt_loss_snap_surf.last_stats()["snaps_map_active"], 0.0)

	def test_interleaved_map_surface_loss_pulls_model_along_normal(self) -> None:
		model_xyz = (_plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 1,
					"seed_opt_iters": 1,
					"grow_opt_iters": 0,
					"first_global_opt_iters": 0,
					"last_global_opt_iters": 0,
					"subdiv": 1,
				}
			},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		loss.backward()

		stats = opt_loss_snap_surf.last_stats()
		self.assertGreater(float(loss.detach()), 0.0)
		self.assertGreater(stats["snaps_map_surf_n"], 0.0)
		self.assertIsNotNone(model_xyz.grad)
		self.assertGreater(float(model_xyz.grad[..., 2].sum().detach()), 0.0)
		self.assertAlmostEqual(float(model_xyz.grad[..., :2].abs().sum().detach()), 0.0, places=6)

	def test_interleaved_map_growth_runs_on_update_interval_with_forced_global(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 0,
					"seed_opt_iters": 0,
					"candidate_opt_iters": 1,
					"fringe_opt_iters": 1,
					"grow_opt_iters": 0,
					"update_interval": 2,
					"update_global_opt_iters": 1,
					"tracking_opt_iters": 1,
					"first_global_opt_iters": 0,
					"last_global_opt_iters": 0,
					"subdiv": 1,
					"seed_radius": 0,
					"edge_init_radius": 2,
					"iters": 20,
				}
			},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.set_debug_step(0, label="initial")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init
		initial_active = mi.active_count()
		self.assertEqual(mi.global_opt_blocks, 0)

		opt_loss_snap_surf.set_debug_step(1, label="snap")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		self.assertEqual(mi.active_count(), initial_active)
		self.assertEqual(mi.global_opt_blocks, 1)

		opt_loss_snap_surf.set_debug_step(2, label="snap")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		self.assertGreater(mi.active_count(), initial_active)
		self.assertEqual(mi.global_opt_blocks, 2)
		self.assertEqual(mi.surface_last_update_step, 2)

	def test_interleaved_map_runs_last_global_on_final_stage_step(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 0,
					"seed_opt_iters": 0,
					"update_interval": 99,
					"first_global_opt_iters": 0,
					"last_global_opt_iters": 1,
					"subdiv": 1,
					"seed_radius": 0,
					"iters": 20,
				}
			},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
			stage_steps=2,
		)

		opt_loss_snap_surf.set_debug_step(0, label="initial")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init
		self.assertEqual(mi.global_opt_blocks, 0)

		opt_loss_snap_surf.set_debug_step(2, label="snap")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		self.assertEqual(mi.global_opt_blocks, 1)
		self.assertTrue(mi.surface_last_global_done)

	def test_interleaved_initial_map_uses_budget_after_scale_transition(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 4,
					"iters": 4,
					"seed_opt_iters": 0,
					"grow_opt_iters": 1,
					"first_global_opt_iters": 0,
					"last_global_opt_iters": 0,
					"subdiv": 1,
					"seed_radius": 0,
					"scale_levels": 3,
				}
			},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.set_debug_step(0, label="initial")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

		mi = opt_loss_snap_surf._states[0].map_init
		self.assertTrue(mi.surface_initial_done)
		self.assertEqual(mi.total_iters, 4)
		self.assertEqual(mi.scale_level, 0)
		self.assertGreater(mi.active_count(), 1)

	def test_interleaved_initial_map_stops_after_no_new_best_then_refines(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"surface_loss": True,
					"initial_iters": 20,
					"iters": 20,
					"seed_opt_iters": 0,
					"grow_opt_iters": 1,
					"first_global_opt_iters": 3,
					"last_global_opt_iters": 0,
					"subdiv": 1,
					"seed_radius": 10,
					"no_progress_iters": 2,
				}
			},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.set_debug_step(0, label="initial")
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

		mi = opt_loss_snap_surf._states[0].map_init
		self.assertTrue(mi.surface_initial_done)
		self.assertTrue(mi.surface_first_global_done)
		self.assertEqual(mi.total_iters, 5)
		self.assertEqual(mi.global_opt_blocks, 3)

	def test_map_init_planar_aligned_surfaces_produce_identity_map(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 1, "iters": 2, "grow_opt_iters": 1, "seed_radius": 1}},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init
		self.assertTrue(mi.done)
		self.assertGreater(mi.active_count(), 0)
		active_vertex = opt_loss_snap_surf._map_init_active_vertex_mask(mi.active_quad, tuple(mi.uv.shape[:2]))
		delta = (mi.uv[active_vertex] - mi.ext_coords[active_vertex]).abs().max()
		self.assertLess(float(delta.detach()), 0.2)

	def test_map_init_state_is_lr_sized_with_quad_activity(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 3, "iters": 1, "grow_opt_iters": 1, "seed_radius": 0}},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init

		self.assertEqual(tuple(mi.uv.shape), (4, 4, 2))
		self.assertEqual(tuple(mi.active_quad.shape), (3, 3))
		self.assertNotEqual(tuple(mi.active_quad.shape), ((4 - 1) * 3, (4 - 1) * 3))

	def test_map_init_seed_activation_defers_sample_geometry_limits(self) -> None:
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		model_xyz[..., 0] += 100.0
		state = opt_loss_snap_surf._SurfaceState()
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				enabled=True,
				subdiv=1,
				iters=0,
				seed_radius=0,
				max_sample_distance=500.0,
				max_sample_angle_deg=45.0,
				sample_angle_step_fraction=0.0,
			),
		)

		ok, _model_dist, _ext_dist, init_count = opt_loss_snap_surf._map_init_seed_state(
			state,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			ext_xyz=ext_xyz,
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_normals=_normals_2d(2, 2),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			cfg=cfg,
			seed_xyz=(0.0, 0.0, 0.0),
		)

		self.assertTrue(ok)
		self.assertEqual(init_count, 1)
		pruned_sample, pruned_fold, pruned_sparse = opt_loss_snap_surf._map_init_prune_bad_active_quads(
			state,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			cfg=cfg,
		)
		self.assertEqual(pruned_sample, 1)
		self.assertEqual(pruned_fold, 0)
		self.assertEqual(pruned_sparse, 0)
		self.assertEqual(state.map_init.active_count(), 0)

	def test_map_init_seed_quad_uv_uses_seed_anchor_and_physical_scale(self) -> None:
		model_xyz = _plane_xyz(h=9, w=9, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		ext_xyz[..., 0] *= 2.0
		ext_xyz[..., 1] *= 2.0
		points = torch.tensor([
			[0.0, 0.0, 0.0],
			[8.0, 8.0, 0.0],
		])

		model_point, model_uv, dist = opt_loss_snap_surf._closest_point_uv_on_model_quad(
			point=torch.tensor([4.0, 4.0, 0.0]),
			model_xyz=model_xyz,
			model_quad=(0, 3, 3),
		)
		uv, ok, reason = opt_loss_snap_surf._map_init_seed_quad_uv_for_points(
			points,
			ext_xyz=ext_xyz,
			model_xyz=model_xyz,
			ext_quad=(1, 1),
			model_quad=(0, 3, 3),
			transform=opt_loss_snap_surf._dihedral_transforms()[0],
			ext_anchor=torch.tensor([4.0, 4.0, 0.0]),
			model_anchor_uv=model_uv,
		)

		self.assertIsNone(reason)
		self.assertTrue(bool(ok.all()))
		self.assertTrue(torch.allclose(uv[0], torch.tensor([0.0, 0.0]), atol=1.0e-5))
		self.assertTrue(torch.allclose(uv[1], torch.tensor([8.0, 8.0]), atol=1.0e-5))
		self.assertTrue(torch.allclose(model_point, torch.tensor([4.0, 4.0, 1.0]), atol=1.0e-5))
		self.assertAlmostEqual(dist, 1.0, places=5)

	def test_map_init_seed_uv_uses_seed_quad_transform_on_mismatched_grid_steps(self) -> None:
		model_xyz = _plane_xyz(h=9, w=9, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		ext_xyz[..., 0] *= 2.0
		ext_xyz[..., 1] *= 2.0
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"subdiv": 1,
					"iters": 0,
					"seed_radius": 0,
					"scale_levels": 3,
					"min_scale_level": 2,
				}
			},
			seed_xyz=(4.0, 4.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init

		self.assertEqual(mi.scale_level, 2)
		self.assertEqual(tuple(mi.uv.shape), (2, 2, 2))
		self.assertTrue(torch.allclose(mi.uv, mi.ext_coords * 2.0, atol=1.0e-5))

	def test_map_init_seed_opt_runs_before_growth_and_uses_budget(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"subdiv": 1,
					"iters": 2,
					"seed_opt_iters": 2,
					"grow_opt_iters": 1,
					"seed_radius": 0,
					"scale_levels": 1,
				}
			},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init

		self.assertEqual(mi.total_iters, 2)
		self.assertEqual(mi.opt_blocks, 1)
		self.assertEqual(int(mi.active_quad.sum()), 1)

	def test_map_init_dense_optimizer_runs_on_full_field(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 1, "iters": 1, "grow_opt_iters": 1, "seed_radius": 0, "dense_opt": True, "scale_levels": 2}},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		mi = opt_loss_snap_surf._states[0].map_init
		stats = opt_loss_snap_surf.last_stats()

		self.assertTrue(mi.done)
		self.assertTrue(torch.isfinite(mi.uv).all())
		self.assertGreater(stats["snaps_map_reg"], stats["snaps_map_active"])

	def test_map_init_grow_ignores_inpaint_guess_when_not_dense(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		H, W = 5, 5
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		active_quad = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active_quad[1:3, 1:3] = True
		active_vertex = opt_loss_snap_surf._map_init_active_vertex_mask(active_quad, (H, W))
		state.map_init.active_quad = active_quad
		state.map_init.uv = torch.where(active_vertex.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
		state.map_init.uv_guess = torch.zeros_like(uv)
		state.map_init.ext_valid = torch.ones(H, W, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		state.map_init.ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		state.map_init.ext_normals = _normals_2d(H, W)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1

		added = opt_loss_snap_surf._map_init_grow_once(
			state,
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(edge_init_radius=2),
			),
		)

		self.assertGreater(added, 0)
		self.assertEqual(int(state.map_init.active_quad.sum()), int(active_quad.sum()) + added)
		active_vertex = opt_loss_snap_surf._map_init_active_vertex_mask(state.map_init.active_quad, (H, W))
		self.assertTrue(torch.isfinite(state.map_init.uv[active_vertex]).all())

	def test_map_init_grow_runs_candidate_and_fringe_prefits(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		H, W = 5, 5
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		active_quad = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active_quad[1:3, 1:3] = True
		active_vertex = opt_loss_snap_surf._map_init_active_vertex_mask(active_quad, (H, W))
		state.map_init.active_quad = active_quad
		state.map_init.blocked_quad = torch.zeros_like(active_quad)
		state.map_init.uv = torch.where(active_vertex.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
		state.map_init.ext_valid = torch.ones(H, W, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		state.map_init.ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		state.map_init.ext_normals = _normals_2d(H, W)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1
		model_xyz = _plane_xyz(h=H, w=W, z=0.0).unsqueeze(0)

		added = opt_loss_snap_surf._map_init_grow_once(
			state,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=1,
					iters=20,
					candidate_opt_iters=2,
					candidate_lr=0.01,
					fringe_opt_iters=3,
					fringe_lr=0.01,
					grow_opt_iters=0,
					edge_init_radius=2,
				),
			),
		)

		self.assertGreater(added, 0)
		self.assertEqual(state.map_init.total_iters, 5)
		self.assertEqual(state.map_init.opt_blocks, 2)
		self.assertIn("loss", state.map_init.last_growth_terms)
		self.assertEqual(float(state.map_init.last_growth_terms["uv_bad"].detach()), 0.0)
		self.assertEqual(float(state.map_init.last_growth_terms["model_bad"].detach()), 0.0)

	def test_map_init_grow_rejects_candidate_quad_when_any_oversample_is_invalid(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		H, W = 3, 2
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		active_quad = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active_quad[0, 0] = True
		active_vertex = opt_loss_snap_surf._map_init_active_vertex_mask(active_quad, (H, W))
		state.map_init.active_quad = active_quad
		state.map_init.uv = torch.where(active_vertex.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
		state.map_init.ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		state.map_init.ext_normals = _normals_2d(H, W)
		state.map_init.ext_valid = torch.ones(H, W, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1
		model_valid = torch.ones(1, H, W, dtype=torch.bool)
		model_valid[0, 2, 0] = False

		added = opt_loss_snap_surf._map_init_grow_once(
			state,
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2, edge_init_radius=2),
			),
		)

		self.assertEqual(added, 0)
		self.assertFalse(bool(state.map_init.active_quad[1, 0]))
		self.assertFalse(torch.isfinite(state.map_init.uv[2]).any())

	def test_map_init_coarse_grow_accepts_candidate_with_one_reachable_model_sample(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		uv = torch.full((2, 3, 2), float("nan"))
		uv[0, 0] = torch.tensor([0.0, 0.0])
		uv[0, 1] = torch.tensor([0.0, 2.0])
		uv[1, 0] = torch.tensor([2.0, 0.0])
		uv[1, 1] = torch.tensor([2.0, 2.0])
		active_quad = torch.zeros(1, 2, dtype=torch.bool)
		active_quad[0, 0] = True
		state.map_init.scale_level = 1
		state.map_init.scale_strides = [1, 2]
		state.map_init.active_quad = active_quad
		state.map_init.blocked_quad = torch.zeros_like(active_quad)
		state.map_init.uv = uv
		state.map_init.ext_pos = _plane_xyz(h=3, w=5, z=0.0)
		state.map_init.ext_normals = _normals_2d(3, 5)
		state.map_init.ext_valid = torch.ones(3, 5, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(2, 4, dtype=torch.bool)
		state.map_init.ext_coords = opt_loss_snap_surf._map_init_dyadic_level_coords(state.map_init.ext_pos, 1)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1
		model_valid = torch.zeros(1, 5, 5, dtype=torch.bool)
		model_valid[0, 0:2, 2:4] = True

		added = opt_loss_snap_surf._map_init_grow_once(
			state,
			model_xyz=_plane_xyz(h=5, w=5, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 5, 5),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=2,
					edge_init_radius=2,
				),
			),
		)

		self.assertEqual(added, 1)
		self.assertTrue(bool(state.map_init.active_quad[0, 1]))
		self.assertTrue(torch.isfinite(state.map_init.uv[:, 2]).all())

	def test_map_init_rejects_bad_new_quads_after_fringe(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		H, W = 3, 2
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		active_before = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active_before[0, 0] = True
		active_quad = active_before.clone()
		active_quad[1, 0] = True
		state.map_init.ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		state.map_init.ext_normals = _normals_2d(H, W)
		state.map_init.ext_valid = torch.ones(H, W, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1
		model_valid = torch.ones(1, H, W, dtype=torch.bool)
		model_valid[0, 2, 0] = False

		active_new, uv_new, blocked_new, sample_bad, _fold_bad, _sparse = opt_loss_snap_surf._map_init_reject_bad_new_quads(
			state,
			active_before=active_before,
			active_quad=active_quad,
			uv=uv,
			blocked_quad=torch.zeros_like(active_quad),
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
			),
		)

		self.assertEqual(sample_bad, 1)
		self.assertTrue(bool(active_new[0, 0]))
		self.assertFalse(bool(active_new[1, 0]))
		self.assertTrue(bool(blocked_new[1, 0]))
		self.assertFalse(torch.isfinite(uv_new[2]).any())

	def test_map_init_refresh_skips_inpaint_when_not_dense(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		state.map_init.active_quad = torch.ones(2, 2, dtype=torch.bool)
		state.map_init.uv = torch.zeros(3, 3, 2)
		state.map_init.uv_guess = torch.ones(3, 3, 2)

		opt_loss_snap_surf._map_init_refresh_uv_guess(
			state,
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					dense_opt=False,
					scale_levels=8,
				),
			),
		)

		self.assertIsNone(state.map_init.uv_guess)
		self.assertEqual(state.map_init.scale_levels_used, 1)

	def test_map_init_inverted_external_normals_choose_negative_sign(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		ext_normals = -_normals_2d(4, 4)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 1, "iters": 1, "grow_opt_iters": 1}},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz, ext_normals=ext_normals))

		self.assertEqual(opt_loss_snap_surf._states[0].map_init.normal_sign, -1)
		self.assertEqual(opt_loss_snap_surf.last_stats()["snaps_map_nsign"], -1.0)

	def test_map_init_normal_sign_is_held_after_initialization(self) -> None:
		model_xyz = _plane_xyz(h=4, w=4, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=4, w=4, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"map_init": {"enabled": True, "subdiv": 1, "iters": 1, "grow_opt_iters": 1}},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz, ext_normals=-_normals_2d(4, 4)))
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz, ext_normals=_normals_2d(4, 4)))

		self.assertEqual(opt_loss_snap_surf._states[0].map_init.normal_sign, -1)
		self.assertEqual(opt_loss_snap_surf.last_stats()["snaps_map_nsign"], -1.0)

	def test_map_init_repair_detects_folded_jacobian(self) -> None:
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		uv_flip = torch.tensor([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]])
		model_xyz = _plane_xyz(h=2, w=2, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		loss, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv_flip,
			active_quad=active_quad,
			ext_pos=ext_xyz,
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			normal_sign=1,
			orientation_sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(w_dist=0.0, w_vec_normal=0.0, w_surface_normal=0.0),
			),
		)

		self.assertTrue(torch.isfinite(loss))
		self.assertGreater(float(terms["jac_bad"].detach()), 0.0)
		self.assertLess(float(terms["jac_min"].detach()), 0.0)
		self.assertTrue(opt_loss_snap_surf._map_init_needs_repair(terms))

	def test_map_init_repair_ignores_positive_jacobian_margin_warnings(self) -> None:
		terms = {
			"uv_bad": torch.tensor(0.0),
			"model_bad": torch.tensor(0.0),
			"jac_bad": torch.tensor(12.0),
			"jac_min": torch.tensor(0.03),
		}

		self.assertFalse(opt_loss_snap_surf._map_init_needs_repair(terms))

	def test_map_init_global_opt_interval_skips_clean_rim_steps(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(global_opt_interval=3),
		)
		terms = {
			"loss": torch.tensor(1.0),
			"uv_bad": torch.tensor(0.0),
			"model_bad": torch.tensor(0.0),
			"step_bad_quad": torch.tensor(0.0),
			"jac_bad": torch.tensor(12.0),
			"jac_min": torch.tensor(0.03),
			"jac_inv_bad_quad": torch.tensor(0.0),
		}

		run, reason = opt_loss_snap_surf._map_init_should_run_global_opt(
			state,
			cfg,
			added=4,
			pruned_sample=0,
			pruned_fold=0,
			pruned_sparse=0,
			terms=terms,
		)
		self.assertFalse(run)
		self.assertEqual(reason, "rim_ok")

		state.map_init.rim_blocks_since_global_opt = 2
		run, reason = opt_loss_snap_surf._map_init_should_run_global_opt(
			state,
			cfg,
			added=4,
			pruned_sample=0,
			pruned_fold=0,
			pruned_sparse=0,
			terms=terms,
		)
		self.assertTrue(run)
		self.assertEqual(reason, "interval")

	def test_map_init_global_opt_runs_on_rim_problems(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(global_opt_interval=10),
		)
		terms = {
			"loss": torch.tensor(1.0),
			"uv_bad": torch.tensor(0.0),
			"model_bad": torch.tensor(0.0),
			"step_bad_quad": torch.tensor(0.0),
			"jac_bad": torch.tensor(1.0),
			"jac_min": torch.tensor(-0.01),
			"jac_inv_bad_quad": torch.tensor(0.0),
		}

		run, reason = opt_loss_snap_surf._map_init_should_run_global_opt(
			state,
			cfg,
			added=1,
			pruned_sample=0,
			pruned_fold=0,
			pruned_sparse=0,
			terms=terms,
		)
		self.assertTrue(run)
		self.assertEqual(reason, "rim_problem")

		terms["jac_bad"] = torch.tensor(0.0)
		terms["jac_min"] = torch.tensor(0.1)
		terms["step_bad_quad"] = torch.tensor(1.0)
		run, reason = opt_loss_snap_surf._map_init_should_run_global_opt(
			state,
			cfg,
			added=1,
			pruned_sample=0,
			pruned_fold=0,
			pruned_sparse=0,
			terms=terms,
		)
		self.assertTrue(run)
		self.assertEqual(reason, "rim_problem")

	def test_map_init_repair_detects_invalid_model_uv(self) -> None:
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		uv_bad = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [10.0, 1.0]]])
		model_xyz = _plane_xyz(h=2, w=2, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv_bad,
			active_quad=active_quad,
			ext_pos=ext_xyz,
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			normal_sign=1,
			orientation_sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)

		self.assertGreater(float(terms["model_bad"].detach()), 0.0)
		self.assertFalse(opt_loss_snap_surf._map_init_needs_repair(terms))

	def test_map_init_prunes_invalid_active_quad_instead_of_repairing(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		state.map_init.active_quad = torch.ones(1, 1, dtype=torch.bool)
		state.map_init.blocked_quad = torch.zeros(1, 1, dtype=torch.bool)
		state.map_init.uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [10.0, 1.0]]])
		state.map_init.ext_pos = _plane_xyz(h=2, w=2, z=0.0)
		state.map_init.ext_normals = _normals_2d(2, 2)
		state.map_init.ext_valid = torch.ones(2, 2, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(1, 1, dtype=torch.bool)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1

		sample_bad, folded_bad, sparse_bad = opt_loss_snap_surf._map_init_prune_bad_active_quads(
			state,
			model_xyz=_plane_xyz(h=2, w=2, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)

		self.assertEqual(sample_bad, 1)
		self.assertEqual(folded_bad, 0)
		self.assertEqual(sparse_bad, 0)
		self.assertFalse(bool(state.map_init.active_quad[0, 0]))
		self.assertTrue(bool(state.map_init.blocked_quad[0, 0]))

	def test_map_init_prunes_folded_active_quad_instead_of_stopping_growth(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		state.map_init.active_quad = torch.ones(1, 1, dtype=torch.bool)
		state.map_init.blocked_quad = torch.zeros(1, 1, dtype=torch.bool)
		state.map_init.uv = torch.tensor([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]])
		state.map_init.ext_pos = _plane_xyz(h=2, w=2, z=0.0)
		state.map_init.ext_normals = _normals_2d(2, 2)
		state.map_init.ext_valid = torch.ones(2, 2, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(1, 1, dtype=torch.bool)
		state.map_init.model_depth = 0
		state.map_init.orientation_sign = 1

		sample_bad, folded_bad, sparse_bad = opt_loss_snap_surf._map_init_prune_bad_active_quads(
			state,
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)

		self.assertEqual(sample_bad, 0)
		self.assertEqual(folded_bad, 1)
		self.assertEqual(sparse_bad, 0)
		self.assertFalse(bool(state.map_init.active_quad[0, 0]))
		self.assertTrue(bool(state.map_init.blocked_quad[0, 0]))

	def test_map_init_sparse_cleanup_recursively_removes_under_supported_quads(self) -> None:
		active = torch.zeros(3, 3, dtype=torch.bool)
		active[1, :] = True
		sparse = opt_loss_snap_surf._map_init_sparse_quad_mask(active, min_neighbors=3)

		self.assertTrue(torch.equal(sparse, active))

	def test_map_init_sparse_cleanup_keeps_solid_frontier_at_threshold_three(self) -> None:
		active = torch.ones(3, 3, dtype=torch.bool)
		sparse = opt_loss_snap_surf._map_init_sparse_quad_mask(active, min_neighbors=3)

		self.assertFalse(bool(sparse.any()))

	def test_map_init_blocked_quads_are_revisited_on_progress_interval(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		H, W = 3, 3
		active = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active[0, 0] = True
		state.map_init.active_quad = active
		state.map_init.blocked_quad = torch.zeros_like(active)
		state.map_init.blocked_quad[0, 1] = True
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		state.map_init.uv = torch.stack([hh, ww], dim=-1)
		state.map_init.ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		state.map_init.ext_normals = _normals_2d(H, W)
		state.map_init.ext_valid = torch.ones(H, W, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		state.map_init.model_depth = 0
		state.map_init.total_iters = 100

		opt_loss_snap_surf._map_init_grow_once(
			state,
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(progress_interval=100),
			),
		)

		self.assertFalse(bool(state.map_init.blocked_quad[0, 1]))
		self.assertEqual(state.map_init.blocked_last_revisit_iter, 100)

	def test_map_init_repair_max_blocks_zero_is_unlimited(self) -> None:
		limited = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(repair_max_blocks=3),
		)
		unlimited = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(repair_max_blocks=0),
		)

		self.assertFalse(opt_loss_snap_surf._map_init_repair_block_allowed(limited, 3))
		self.assertTrue(opt_loss_snap_surf._map_init_repair_block_allowed(unlimited, 300))

if __name__ == "__main__":
	unittest.main()
