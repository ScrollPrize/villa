from __future__ import annotations

import os
import tempfile
import unittest

import torch

from snap_surf.map_pyramid import _map_init_quad_offsets
from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz, _result, opt_loss_snap_surf


class SnapSurfMapPyramidTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()

	def test_map_init_config_parse_and_validation(self) -> None:
		self.assertEqual(opt_loss_snap_surf.SnapSurfMapInitConfig().global_opt_interval, 10)
		self.assertAlmostEqual(opt_loss_snap_surf.SnapSurfMapInitConfig().sample_angle_step_fraction, 0.0)
		self.assertAlmostEqual(opt_loss_snap_surf._parse_map_init_config({}).sample_angle_step_fraction, 0.0)
		cfg = opt_loss_snap_surf._parse_map_init_config({
			"enabled": True,
			"surface_loss": True,
			"initial_iters": 11,
			"update_interval": 13,
			"update_global_opt_iters": 17,
			"tracking_opt_iters": 29,
			"first_global_opt_iters": 19,
			"last_global_opt_iters": 23,
			"subdiv": 2,
			"iters": 3,
			"seed_opt_iters": 7,
			"candidate_opt_iters": 4,
			"candidate_lr": 0.07,
			"fringe_opt_iters": 5,
			"fringe_lr": 0.03,
			"global_opt_interval": 6,
			"progress_mode": "both",
			"no_progress_iters": 31,
			"scale_levels": 4,
			"min_scale_level": 2,
			"dense_opt": True,
			"dense_reg_radius": 5,
			"w_dense_prior": 0.25,
			"repair_max_blocks": 2,
			"repair_lr_mult": 0.5,
			"repair_w_jac_mult": 8.0,
			"edge_init_radius": 3,
			"progress_interval": 7,
			"w_metric_smooth": 0.12,
			"w_area_smooth": 0.03,
			"z_lift_enabled": False,
			"z_lift_refine_enabled": True,
			"z_lift_norm_xy_min": 0.2,
			"map_turn": 12.0,
			"z_lift_huber_delta": 0.5,
			"max_sample_distance": 500.0,
			"max_sample_angle_deg": 45.0,
			"sample_angle_step_fraction": 0.2,
			"max_step_neighbor_ratio": 10.0,
			"compile_objective": True,
			"compile_objective_mode": "reduce-overhead",
			"fixture_export_dir": "fixture_out",
			"fixture_export_once": False,
			"fixture_export_objs": False,
		})

		self.assertTrue(cfg.enabled)
		self.assertTrue(cfg.surface_loss)
		self.assertEqual(cfg.initial_iters, 11)
		self.assertEqual(cfg.update_interval, 13)
		self.assertEqual(cfg.update_global_opt_iters, 17)
		self.assertEqual(cfg.tracking_opt_iters, 29)
		self.assertEqual(cfg.first_global_opt_iters, 19)
		self.assertEqual(cfg.last_global_opt_iters, 23)
		self.assertEqual(cfg.subdiv, 2)
		self.assertEqual(cfg.iters, 3)
		self.assertEqual(cfg.seed_opt_iters, 7)
		self.assertEqual(cfg.candidate_opt_iters, 4)
		self.assertAlmostEqual(cfg.candidate_lr, 0.07)
		self.assertEqual(cfg.fringe_opt_iters, 5)
		self.assertAlmostEqual(cfg.fringe_lr, 0.03)
		self.assertEqual(cfg.global_opt_interval, 6)
		self.assertEqual(cfg.progress_mode, "both")
		self.assertEqual(cfg.no_progress_iters, 31)
		self.assertEqual(cfg.scale_levels, 4)
		self.assertEqual(cfg.min_scale_level, 2)
		self.assertTrue(cfg.dense_opt)
		self.assertEqual(cfg.dense_reg_radius, 5)
		self.assertAlmostEqual(cfg.w_dense_prior, 0.25)
		self.assertEqual(cfg.repair_max_blocks, 2)
		self.assertAlmostEqual(cfg.repair_lr_mult, 0.5)
		self.assertAlmostEqual(cfg.repair_w_jac_mult, 8.0)
		self.assertEqual(cfg.edge_init_radius, 3)
		self.assertEqual(cfg.progress_interval, 100)
		self.assertAlmostEqual(cfg.w_metric_smooth, 0.12)
		self.assertAlmostEqual(cfg.w_area_smooth, 0.03)
		self.assertFalse(cfg.z_lift_enabled)
		self.assertTrue(cfg.z_lift_refine_enabled)
		self.assertAlmostEqual(cfg.z_lift_norm_xy_min, 0.2)
		self.assertAlmostEqual(cfg.w_z_lift, 12.0)
		self.assertAlmostEqual(cfg.z_lift_huber_delta, 0.5)
		self.assertAlmostEqual(cfg.max_sample_distance, 500.0)
		self.assertAlmostEqual(cfg.max_sample_angle_deg, 45.0)
		self.assertAlmostEqual(cfg.sample_angle_step_fraction, 0.2)
		self.assertAlmostEqual(cfg.max_step_neighbor_ratio, 10.0)
		self.assertTrue(cfg.compile_objective)
		self.assertEqual(cfg.compile_objective_mode, "reduce-overhead")
		self.assertEqual(cfg.fixture_export_dir, "fixture_out")
		self.assertFalse(cfg.fixture_export_once)
		self.assertFalse(cfg.fixture_export_objs)
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"unknown": 1})
		for key in ("w_metric_smooth", "w_area_smooth", "map_turn", "z_lift_norm_xy_min", "max_sample_distance", "sample_angle_step_fraction", "max_step_neighbor_ratio"):
			with self.assertRaises(ValueError):
				opt_loss_snap_surf._parse_map_init_config({key: -0.1})
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"z_lift_huber_delta": 0.0})
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"max_sample_angle_deg": 181.0})
		for key in ("candidate_lr", "fringe_lr"):
			with self.assertRaises(ValueError):
				opt_loss_snap_surf._parse_map_init_config({key: 0.0})
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"progress_mode": "loud"})
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"compile_objective_mode": "experimental"})
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"scale_levels": 2, "scale_factor": 3})
		cfg = opt_loss_snap_surf._parse_map_init_config({"minscale": 1})
		self.assertEqual(cfg.min_scale_level, 1)
		with self.assertRaises(ValueError):
			opt_loss_snap_surf._parse_map_init_config({"minscale": 1, "min_scale_level": 1})

	def test_map_init_dyadic_level_helpers_are_exact(self) -> None:
		strides = opt_loss_snap_surf._map_init_dyadic_strides(
			9,
			5,
			requested_levels=4,
			scale_factor=2,
		)

		self.assertEqual(strides, [1, 2, 4])
		self.assertEqual(opt_loss_snap_surf._map_init_dyadic_level_shape(9, 5, 2), (3, 2))
		coords = opt_loss_snap_surf._map_init_dyadic_level_coords(torch.zeros(9, 5, 3), 2)
		self.assertTrue(torch.equal(coords[1, 1], torch.tensor([4.0, 4.0])))

	def test_map_init_quad_offsets_are_unshifted(self) -> None:
		one = _map_init_quad_offsets(subdiv=1, device=torch.device("cpu"), dtype=torch.float32)
		self.assertTrue(torch.equal(one, torch.tensor([[0.0, 0.0]])))

		two = _map_init_quad_offsets(subdiv=2, device=torch.device("cpu"), dtype=torch.float32)
		expected = torch.tensor([
			[0.0, 0.0],
			[0.0, 0.5],
			[0.5, 0.0],
			[0.5, 0.5],
		])
		self.assertTrue(torch.equal(two, expected))

	def test_map_init_single_neighbor_prediction_uses_source_to_uv_transform(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=2, target_rank=2)
		valid = torch.zeros(1, 4, 4, dtype=torch.bool)
		valid[0, 1, 1] = True
		map_b = torch.full((1, 4, 4, 2), float("nan"))
		map_b[0, 1, 1] = torch.tensor([10.0, 20.0])
		candidate = torch.tensor([[0, 2, 1]], dtype=torch.long)
		transform = torch.tensor([
			[5.0, 0.0],
			[0.0, 7.0],
		])

		pred, count, _nearest = opt_loss_snap_surf._direct_predict_candidates_batched(
			state,
			valid_b=valid,
			map_b=map_b,
			candidate_bidx=candidate,
			radius=1,
			single_neighbor_transform=transform,
		)

		self.assertEqual(int(count[0]), 1)
		self.assertTrue(torch.allclose(pred[0], torch.tensor([15.0, 20.0]), atol=1.0e-5))

	def test_map_init_underconstrained_prediction_uses_source_to_uv_transform(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=2, target_rank=2)
		valid = torch.zeros(1, 4, 4, dtype=torch.bool)
		valid[0, 1, 1] = True
		valid[0, 1, 2] = True
		map_b = torch.full((1, 4, 4, 2), float("nan"))
		map_b[0, 1, 1] = torch.tensor([10.0, 20.0])
		map_b[0, 1, 2] = torch.tensor([10.0, 27.0])
		candidate = torch.tensor([[0, 2, 2]], dtype=torch.long)
		transform = torch.tensor([
			[5.0, 0.0],
			[0.0, 7.0],
		])

		pred, count, _nearest = opt_loss_snap_surf._direct_predict_candidates_batched(
			state,
			valid_b=valid,
			map_b=map_b,
			candidate_bidx=candidate,
			radius=1,
			single_neighbor_transform=transform,
		)

		self.assertEqual(int(count[0]), 2)
		self.assertTrue(torch.allclose(pred[0], torch.tensor([15.0, 27.0]), atol=1.0e-5))

	def test_map_init_source_to_uv_transform_requires_rank_two(self) -> None:
		uv = torch.full((3, 3, 2), float("nan"))
		active = torch.zeros(3, 3, dtype=torch.bool)
		active[0, 0] = True
		active[1, 1] = True
		uv[0, 0] = torch.tensor([10.0, 20.0])
		uv[1, 1] = torch.tensor([12.0, 23.0])

		self.assertIsNone(opt_loss_snap_surf._map_init_source_to_uv_transform(uv, active))

		active[0, 1] = True
		uv[0, 1] = torch.tensor([10.0, 23.0])
		got = opt_loss_snap_surf._map_init_source_to_uv_transform(uv, active)

		self.assertIsNotNone(got)
		assert got is not None
		self.assertTrue(torch.allclose(got, torch.tensor([[2.0, 0.0], [0.0, 3.0]]), atol=1.0e-5))

	def test_map_init_active_transition_repeats_quads_to_finer_blocks(self) -> None:
		active = torch.tensor([[True, False], [False, True]])

		got = opt_loss_snap_surf._map_init_repeat_quads_to_finer(active)

		expected = torch.tensor(
			[
				[True, True, False, False],
				[True, True, False, False],
				[False, False, True, True],
				[False, False, True, True],
			]
		)
		self.assertTrue(torch.equal(got, expected))

	def test_map_init_full_finer_blocks_promote_to_coarser(self) -> None:
		active = torch.tensor(
			[
				[True, True, True, False],
				[True, True, True, True],
				[False, True, True, True],
				[True, True, True, True],
			]
		)

		got = opt_loss_snap_surf._map_init_full_blocks_to_coarser(active)

		expected = torch.tensor([[True, False], [False, True]])
		self.assertTrue(torch.equal(got, expected))

	def test_map_init_promote_coarser_zeroes_residual_for_new_quads(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		state.map_init.scale_strides = [1, 2]
		state.map_init.scale_level = 0
		state.map_init.uv_pyramid = opt_loss_snap_surf._map_init_make_zero_uv_pyramid(
			ext_xyz=ext_xyz,
			strides=[1, 2],
			dtype=torch.float32,
		)
		state.map_init.uv_pyramid[1].fill_(7.0)
		state.map_init.active_quad = torch.ones(4, 4, dtype=torch.bool)
		state.map_init.blocked_quad = torch.zeros(4, 4, dtype=torch.bool)
		state.map_init.scale_active_quads = [
			state.map_init.active_quad.clone(),
			torch.tensor([[True, False], [False, False]], dtype=torch.bool),
		]
		state.map_init.scale_blocked_quads = [
			torch.zeros(4, 4, dtype=torch.bool),
			torch.zeros(2, 2, dtype=torch.bool),
		]

		opt_loss_snap_surf._map_init_promote_full_active_to_coarser(state.map_init, from_level=0, to_level=1)

		residual = state.map_init.uv_pyramid[1][0]
		keep = torch.zeros(3, 3, dtype=torch.bool)
		keep[:2, :2] = True
		self.assertTrue(torch.equal(state.map_init.scale_active_quads[1], torch.ones(2, 2, dtype=torch.bool)))
		self.assertTrue(torch.all(residual[:, keep] == 7.0))
		self.assertTrue(torch.all(residual[:, ~keep] == 0.0))

	def test_map_init_scale_transition_keeps_finer_residual_zero(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		state.map_init.ext_pos = ext_xyz
		state.map_init.ext_normals = _normals_2d(5, 5)
		state.map_init.ext_valid = torch.ones(5, 5, dtype=torch.bool)
		state.map_init.ext_quad_valid = torch.ones(4, 4, dtype=torch.bool)
		state.map_init.scale_strides = [1, 2]
		state.map_init.scale_level = 1
		state.map_init.uv_pyramid = opt_loss_snap_surf._map_init_make_zero_uv_pyramid(
			ext_xyz=ext_xyz,
			strides=[1, 2],
			dtype=torch.float32,
		)
		coarse = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij"), dim=-1).to(torch.float32)
		state.map_init.uv_pyramid[1] = coarse.permute(2, 0, 1).unsqueeze(0).contiguous()
		state.map_init.active_quad = torch.ones(2, 2, dtype=torch.bool)
		state.map_init.blocked_quad = torch.zeros(2, 2, dtype=torch.bool)
		opt_loss_snap_surf._map_init_set_current_level_external_coords(state.map_init)
		opt_loss_snap_surf._map_init_refresh_current_uv_from_pyramid(
			state.map_init,
			opt_loss_snap_surf.SnapSurfConfig(map_init=opt_loss_snap_surf.SnapSurfMapInitConfig()),
		)

		ok = opt_loss_snap_surf._map_init_transition_to_finer(
			state,
			opt_loss_snap_surf.SnapSurfConfig(map_init=opt_loss_snap_surf.SnapSurfMapInitConfig()),
		)

		self.assertTrue(ok)
		self.assertEqual(state.map_init.scale_level, 0)
		self.assertTrue(torch.equal(state.map_init.uv_pyramid[0], torch.zeros_like(state.map_init.uv_pyramid[0])))
		self.assertTrue(torch.isfinite(state.map_init.uv).all())
		self.assertTrue(torch.allclose(state.map_init.uv[2, 2], torch.tensor([1.0, 1.0])))

	def test_map_init_transition_zeroes_finer_residual_for_new_quads(self) -> None:
		state = opt_loss_snap_surf._SurfaceState()
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		state.map_init.scale_strides = [1, 2]
		state.map_init.scale_level = 1
		state.map_init.uv_pyramid = opt_loss_snap_surf._map_init_make_zero_uv_pyramid(
			ext_xyz=ext_xyz,
			strides=[1, 2],
			dtype=torch.float32,
		)
		state.map_init.uv_pyramid[0].fill_(5.0)
		state.map_init.active_quad = torch.ones(2, 2, dtype=torch.bool)
		state.map_init.blocked_quad = torch.zeros(2, 2, dtype=torch.bool)
		state.map_init.uv = torch.zeros(3, 3, 2)
		state.map_init.scale_active_quads = [
			torch.tensor(
				[
					[True, False, False, False],
					[False, False, False, False],
					[False, False, False, False],
					[False, False, False, False],
				],
				dtype=torch.bool,
			),
			state.map_init.active_quad.clone(),
		]
		state.map_init.scale_blocked_quads = [
			torch.zeros(4, 4, dtype=torch.bool),
			torch.zeros(2, 2, dtype=torch.bool),
		]
		cfg = opt_loss_snap_surf.SnapSurfConfig()

		self.assertTrue(opt_loss_snap_surf._map_init_transition_to_finer(state, cfg))

		residual = state.map_init.uv_pyramid[0][0]
		keep = torch.zeros(5, 5, dtype=torch.bool)
		keep[:2, :2] = True
		self.assertTrue(torch.all(residual[:, keep] == 5.0))
		self.assertTrue(torch.all(residual[:, ~keep] == 0.0))
		self.assertTrue(torch.equal(state.map_init.active_quad, torch.ones(4, 4, dtype=torch.bool)))

	def test_map_init_scalespace_inpaint_preserves_active_uv(self) -> None:
		uv = torch.full((5, 5, 2), float("nan"))
		active = torch.zeros(5, 5, dtype=torch.bool)
		active[2, 2] = True
		uv[2, 2] = torch.tensor([2.0, 3.0])

		got = opt_loss_snap_surf._map_init_scalespace_inpaint_uv(
			uv,
			active,
			cfg=opt_loss_snap_surf.SnapSurfMapInitConfig(scale_levels=3),
			model_h=10,
			model_w=10,
		)

		self.assertTrue(torch.isfinite(got).all())
		self.assertTrue(torch.equal(got[2, 2], uv[2, 2]))

if __name__ == "__main__":
	unittest.main()
