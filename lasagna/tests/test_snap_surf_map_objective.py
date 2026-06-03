from __future__ import annotations

import os
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz, _result, opt_loss_snap_surf
import snap_surf.map_global as map_global
import snap_surf.map_objective as map_objective


class SnapSurfMapObjectiveTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()

	def test_packed_pos_norm_sampling_matches_separate_sampling(self) -> None:
		model_xyz = torch.arange(1 * 4 * 5 * 3, dtype=torch.float32).reshape(1, 4, 5, 3) / 10.0
		model_normals = torch.flip(model_xyz, dims=(-1,)) + 1.0
		model_valid = torch.ones(1, 4, 5, dtype=torch.bool)
		model_valid[0, 1, 2] = False
		coords = torch.tensor([
			[[0.0, 0.25, 0.25], [0.0, 1.5, 2.25]],
			[[0.0, 2.25, 3.5], [0.0, 0.75, 1.25]],
		], dtype=torch.float32)

		pos_safe = opt_loss_snap_surf._map_init_valid_field_values(model_xyz, model_valid)
		norm_safe = opt_loss_snap_surf._map_init_valid_field_values(model_normals, model_valid)
		packed = opt_loss_snap_surf._map_init_packed_pos_norm_values(model_xyz, model_normals, model_valid)
		packed_sample = opt_loss_snap_surf._sample_surface_grid(packed, coords)
		pos_sample, norm_sample = opt_loss_snap_surf._map_init_split_packed_pos_norm(packed_sample)

		torch.testing.assert_close(pos_sample, opt_loss_snap_surf._sample_surface_grid(pos_safe, coords))
		torch.testing.assert_close(norm_sample, opt_loss_snap_surf._sample_surface_grid(norm_safe, coords))

	def test_packed_pos_norm_cache_invalidates_by_prefix_and_tensor_identity(self) -> None:
		model_xyz = _plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)
		model_normals = _normals_3d(1, 3, 3)
		model_valid = torch.ones(1, 3, 3, dtype=torch.bool)
		cache: dict[tuple[object, ...], object] = {}

		packed_epoch0 = opt_loss_snap_surf._map_init_cached_packed_pos_norm_values(
			kind="model",
			pos=model_xyz,
			normals=model_normals,
			valid=model_valid,
			cache=cache,
			prefix=("mesh_epoch", 0),
		)
		packed_epoch0_again = opt_loss_snap_surf._map_init_cached_packed_pos_norm_values(
			kind="model",
			pos=model_xyz,
			normals=model_normals,
			valid=model_valid,
			cache=cache,
			prefix=("mesh_epoch", 0),
		)
		packed_epoch1 = opt_loss_snap_surf._map_init_cached_packed_pos_norm_values(
			kind="model",
			pos=model_xyz,
			normals=model_normals,
			valid=model_valid,
			cache=cache,
			prefix=("mesh_epoch", 1),
		)
		packed_clone = opt_loss_snap_surf._map_init_cached_packed_pos_norm_values(
			kind="model",
			pos=model_xyz.clone(),
			normals=model_normals,
			valid=model_valid,
			cache=cache,
			prefix=("mesh_epoch", 0),
		)

		self.assertIs(packed_epoch0, packed_epoch0_again)
		self.assertIsNot(packed_epoch0, packed_epoch1)
		self.assertIsNot(packed_epoch0, packed_clone)

	def test_shared_sample_plan_matches_model_and_theta_sampling(self) -> None:
		model_xyz = torch.arange(1 * 4 * 5 * 3, dtype=torch.float32).reshape(1, 4, 5, 3) / 10.0
		model_normals = torch.flip(model_xyz, dims=(-1,)) + 1.0
		model_valid = torch.ones(1, 4, 5, dtype=torch.bool)
		model_valid[0, 2, 3] = False
		theta = torch.arange(1 * 4 * 5, dtype=torch.float32).reshape(1, 4, 5) / 7.0
		coords = torch.tensor([
			[[[0.0, 0.25, 0.25], [0.0, 1.5, 2.25]]],
			[[[0.0, 2.25, 3.5], [0.0, 0.75, 1.25]]],
		], dtype=torch.float32)
		packed = opt_loss_snap_surf._map_init_packed_pos_norm_values(model_xyz, model_normals, model_valid)

		plan = opt_loss_snap_surf._map_init_surface_sample_plan(coords, tuple(model_valid.shape))
		shared_packed = opt_loss_snap_surf._map_init_sample_surface_grid_plan(packed, plan)
		shared_valid = opt_loss_snap_surf._map_init_sample_valid_plan(model_valid, plan)
		theta_sample, theta_valid = opt_loss_snap_surf._map_init_sample_scalar_plan(theta, model_valid, plan)
		old_packed, old_valid = opt_loss_snap_surf._map_init_sample_surface_grid_with_valid(packed, model_valid, coords)
		old_theta, old_theta_valid = opt_loss_snap_surf._map_init_sample_scalar_quad_field(theta, model_valid, coords, tuple(model_valid.shape))

		torch.testing.assert_close(shared_packed, old_packed)
		self.assertTrue(torch.equal(shared_valid, old_valid))
		torch.testing.assert_close(theta_sample, old_theta)
		self.assertTrue(torch.equal(theta_valid, old_theta_valid))

	def test_map_init_objective_reuses_cached_external_samples(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32), indexing="ij"), dim=-1)
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=2,
				w_dist=1.0,
				w_vec_normal=1.0,
				w_surface_normal=1.0,
				w_smooth=0.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
			),
		)
		cache: dict[tuple[object, ...], object] = {}
		ext_pos = _plane_xyz(h=3, w=3, z=0.0)
		ext_normals = _normals_2d(3, 3)
		ext_valid = torch.ones(3, 3, dtype=torch.bool)
		ext_quad_valid = torch.ones(2, 2, dtype=torch.bool)
		model_xyz = _plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)
		model_valid = torch.ones(1, 3, 3, dtype=torch.bool)
		model_normals = _normals_3d(1, 3, 3)

		def _run(profile_blocks: dict[str, list[float]]) -> None:
			opt_loss_snap_surf._map_init_objective(
				uv_full=uv,
				active_quad=torch.ones(2, 2, dtype=torch.bool),
				ext_pos=ext_pos,
				ext_normals=ext_normals,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				model_depth=0,
				sign=1,
				cfg=cfg,
				need_stats=False,
				profile_blocks=profile_blocks,
				runtime_cache=cache,
				cache_key_prefix=("stage", 0),
				external_static_cache_key=("level0",),
			)

		first_profile: dict[str, list[float]] = {}
		second_profile: dict[str, list[float]] = {}
		_run(first_profile)
		_run(second_profile)

		self.assertIn("sample_ext_packed_bilerp", first_profile)
		self.assertIn("sample_ext_cached", second_profile)
		self.assertNotIn("sample_ext_packed_bilerp", second_profile)

	def test_map_init_objective_gates_sample_model_metric_vertices(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32), indexing="ij"), dim=-1)
		base_kwargs = dict(
			uv_full=uv,
			active_quad=torch.ones(2, 2, dtype=torch.bool),
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			need_stats=False,
		)
		no_step_profile: dict[str, list[float]] = {}
		opt_loss_snap_surf._map_init_objective(
			**base_kwargs,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=2,
					w_dist=1.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_z_lift=0.0,
					w_smooth=0.0,
					w_bend=0.0,
					w_jac=0.0,
					w_metric_smooth=0.0,
					w_area_smooth=0.0,
					max_sample_angle_deg=180.0,
				),
			),
			profile_blocks=no_step_profile,
		)
		step_profile: dict[str, list[float]] = {}
		opt_loss_snap_surf._map_init_objective(
			**base_kwargs,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=2,
					w_dist=1.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_z_lift=0.0,
					w_smooth=0.0,
					w_bend=0.0,
					w_jac=0.0,
					w_metric_smooth=0.0,
					w_area_smooth=0.0,
					max_sample_angle_deg=45.0,
					sample_angle_step_fraction=0.1,
				),
			),
			profile_blocks=step_profile,
		)

		self.assertIn("sample_model_context", no_step_profile)
		self.assertNotIn("sample_model_metric_steps", no_step_profile)
		self.assertIn("sample_model_metric_steps", step_profile)
		self.assertEqual(len(step_profile["sample_model_metric_steps"]), 1)

	def test_map_init_objective_samples_model_metric_in_regularization_only_when_active(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij"), dim=-1)
		profile: dict[str, list[float]] = {}
		opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(3, 3, dtype=torch.bool),
			ext_pos=_plane_xyz(h=4, w=4, z=0.0),
			ext_normals=_normals_2d(4, 4),
			ext_valid=torch.ones(4, 4, dtype=torch.bool),
			ext_quad_valid=torch.ones(3, 3, dtype=torch.bool),
			model_xyz=_plane_xyz(h=4, w=4, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 4, 4, dtype=torch.bool),
			model_normals=_normals_3d(1, 4, 4),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					w_dist=0.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_z_lift=0.0,
					w_smooth=1.0,
					w_bend=0.0,
					w_jac=0.0,
					w_metric_smooth=0.0,
					w_area_smooth=0.0,
					max_sample_angle_deg=180.0,
				),
			),
			need_stats=False,
			profile_blocks=profile,
		)

		self.assertNotIn("sample_model_metric_steps", profile)
		self.assertIn("reg_model_metric_sample", profile)

	def test_map_init_objective_reuses_cached_reg_physical_ref(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij"), dim=-1)
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				w_dist=0.0,
				w_vec_normal=0.0,
				w_surface_normal=0.0,
				w_z_lift=0.0,
				w_smooth=1.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
			),
		)
		cache: dict[tuple[object, ...], object] = {}
		kwargs = dict(
			uv_full=uv,
			active_quad=torch.ones(3, 3, dtype=torch.bool),
			ext_pos=_plane_xyz(h=4, w=4, z=0.0),
			ext_normals=_normals_2d(4, 4),
			ext_valid=torch.ones(4, 4, dtype=torch.bool),
			ext_quad_valid=torch.ones(3, 3, dtype=torch.bool),
			model_xyz=_plane_xyz(h=4, w=4, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 4, 4, dtype=torch.bool),
			model_normals=_normals_3d(1, 4, 4),
			model_depth=0,
			sign=1,
			cfg=cfg,
			need_stats=False,
			runtime_cache=cache,
			cache_key_prefix=("stage", 0),
			external_static_cache_key=("level", 0),
		)
		with mock.patch.object(
			map_objective,
			"_map_init_reference_edge_square",
			wraps=map_objective._map_init_reference_edge_square,
		) as ref_mock:
			opt_loss_snap_surf._map_init_objective(**kwargs)
			opt_loss_snap_surf._map_init_objective(**kwargs)

		self.assertEqual(ref_mock.call_count, 1)

	def test_z_lift_field_construction_unwraps_multiple_turns(self) -> None:
		angles = torch.linspace(0.0, 4.0 * math.pi, 9)
		normals = torch.zeros(1, 9, 3)
		normals[..., 0] = torch.cos(angles).view(1, 9)
		normals[..., 1] = torch.sin(angles).view(1, 9)
		base_valid = torch.ones(1, 9, dtype=torch.bool)

		theta, valid, stats = opt_loss_snap_surf._map_init_lifted_z_vertex_heading_field(
			normals,
			base_valid,
			(0, 0),
			norm_xy_min=0.01,
		)

		self.assertTrue(torch.equal(valid, base_valid))
		self.assertEqual(stats["invalid"], 0.0)
		self.assertEqual(stats["unreachable"], 0.0)
		self.assertAlmostEqual(float(theta[0, 0].item()), 0.0, places=5)
		self.assertAlmostEqual(float(theta[0, 4].item()), 2.0 * math.pi, places=5)
		self.assertAlmostEqual(float(theta[0, -1].item()), 4.0 * math.pi, places=5)

	def test_z_lift_field_construction_unwraps_pi_boundary(self) -> None:
		angles = torch.tensor([math.radians(170.0), math.radians(-170.0), math.radians(-150.0)])
		normals = torch.zeros(1, 3, 3)
		normals[..., 0] = torch.cos(angles).view(1, 3)
		normals[..., 1] = torch.sin(angles).view(1, 3)
		base_valid = torch.ones(1, 3, dtype=torch.bool)

		theta, valid, stats = opt_loss_snap_surf._map_init_lifted_z_vertex_heading_field(
			normals,
			base_valid,
			(0, 0),
			norm_xy_min=0.01,
		)

		self.assertTrue(torch.equal(valid, base_valid))
		self.assertEqual(stats["unreachable"], 0.0)
		self.assertAlmostEqual(float(theta[0, 0].item()), 0.0, places=5)
		self.assertAlmostEqual(float(theta[0, 1].item()), math.radians(20.0), places=5)

	def test_z_lift_field_construction_does_not_cross_depth_planes(self) -> None:
		angles = torch.linspace(0.0, 2.0 * math.pi, 5)
		normals = torch.zeros(2, 1, 5, 3)
		normals[..., 0] = torch.cos(angles).view(1, 1, 5)
		normals[..., 1] = torch.sin(angles).view(1, 1, 5)
		base_valid = torch.ones(2, 1, 5, dtype=torch.bool)

		theta, valid, stats = opt_loss_snap_surf._map_init_lifted_z_vertex_heading_field(
			normals,
			base_valid,
			(0, 0, 0),
			norm_xy_min=0.01,
		)

		self.assertTrue(valid[0].all())
		self.assertFalse(valid[1].any())
		self.assertEqual(stats["invalid"], 0.0)
		self.assertEqual(stats["unreachable"], 5.0)
		self.assertAlmostEqual(float(theta[0, 0, 0].item()), 0.0, places=5)
		self.assertAlmostEqual(float(theta[0, 0, -1].item()), 2.0 * math.pi, places=5)

	def test_z_lift_field_construction_blocks_invalid_and_unreachable_vertices(self) -> None:
		normals = torch.zeros(1, 5, 3)
		normals[..., 0] = 1.0
		normals[:, 2, 0] = 0.0
		normals[:, 2, 1] = 0.0
		normals[:, 2, 2] = 1.0
		base_valid = torch.ones(1, 5, dtype=torch.bool)

		_theta, valid, stats = opt_loss_snap_surf._map_init_lifted_z_vertex_heading_field(
			normals,
			base_valid,
			(0, 0),
			norm_xy_min=0.75,
		)

		self.assertTrue(bool(valid[0, 0]))
		self.assertTrue(bool(valid[0, 1]))
		self.assertFalse(bool(valid[0, 2]))
		self.assertFalse(bool(valid[0, 3]))
		self.assertFalse(bool(valid[0, 4]))
		self.assertEqual(stats["invalid"], 1.0)
		self.assertEqual(stats["unreachable"], 2.0)

	def test_z_lift_legacy_quad_seed_metadata_falls_back_to_nearest_vertex(self) -> None:
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		ext_normals = torch.zeros(2, 2, 3)
		ext_normals[..., 0] = 1.0
		model_normals = torch.zeros(1, 2, 2, 3)
		model_normals[..., 0] = 1.0
		fixture = map_global.MapFixture(
			root=Path("."),
			metadata={
				"seed_xyz": [1.0, 0.0, 0.0],
				"seed_ext_sample_hw": [0, 0],
				"seed_model_quad": [0, 0, 0],
				"model_depth": 0,
				"sign": 1,
			},
			ext_xyz=ext_xyz,
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			ext_normals=ext_normals,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=model_normals,
			reference_uv=torch.zeros(2, 2, 2),
			reference_active_quad=torch.ones(1, 1, dtype=torch.bool),
			reference_blocked_quad=torch.zeros(1, 1, dtype=torch.bool),
		)

		self.assertEqual(map_global._fixture_seed_ext_vertex(fixture), (0, 1))
		self.assertEqual(map_global._fixture_seed_model_vertex(fixture), (0, 0, 1))
		z_lift = map_global._map_init_z_lift_for_fixture(
			fixture,
			opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					w_z_lift=1.0,
					z_lift_norm_xy_min=0.01,
				),
			),
			sign=1,
		)

		self.assertIsNotNone(z_lift)
		assert z_lift is not None
		self.assertEqual(tuple(z_lift["ext_theta_lifted"].shape), (2, 2))
		self.assertEqual(tuple(z_lift["model_theta_lifted"].shape), (1, 2, 2))

	def test_map_init_z_lift_turn_distinguishes_winding_branch_without_wrapping(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		ext_normals = torch.zeros(2, 2, 3)
		ext_normals[..., 0] = 1.0
		model_normals = torch.zeros(1, 2, 2, 3)
		model_normals[..., 0] = 1.0
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=1,
				w_dist=0.0,
				w_vec_normal=0.0,
				w_surface_normal=0.0,
				w_z_lift=1.0,
			),
		)

		low_loss, low_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=ext_normals,
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=model_normals,
			model_depth=0,
			sign=1,
			cfg=cfg,
			ext_z_lift_theta=torch.zeros(2, 2),
			ext_z_lift_valid=torch.ones(2, 2, dtype=torch.bool),
			model_z_lift_theta=torch.zeros(1, 2, 2),
			model_z_lift_valid=torch.ones(1, 2, 2, dtype=torch.bool),
		)
		high_loss, high_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=ext_normals,
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=model_normals,
			model_depth=0,
			sign=1,
			cfg=cfg,
			ext_z_lift_theta=torch.zeros(2, 2),
			ext_z_lift_valid=torch.ones(2, 2, dtype=torch.bool),
			model_z_lift_theta=torch.full((1, 2, 2), 2.0 * math.pi),
			model_z_lift_valid=torch.ones(1, 2, 2, dtype=torch.bool),
		)

		self.assertAlmostEqual(float(low_terms["turn"].detach()), 0.0)
		self.assertGreater(float(high_terms["turn"].detach()), 4.0)
		self.assertGreater(float(high_loss.detach()), float(low_loss.detach()) + 4.0)
		self.assertEqual(float(high_terms["turn_smp"].detach()), 1.0)

	def test_map_init_z_lift_samples_dense_model_theta(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		ext_normals = torch.zeros(2, 2, 3)
		ext_normals[..., 0] = 1.0
		model_normals = torch.zeros(1, 3, 3, 3)
		model_normals[..., 0] = 1.0
		kwargs = dict(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=ext_normals,
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=model_normals,
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=1,
					w_dist=0.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_z_lift=1.0,
				),
			),
			ext_z_lift_theta=torch.zeros(2, 2),
			ext_z_lift_valid=torch.ones(2, 2, dtype=torch.bool),
			model_z_lift_valid=torch.ones(1, 3, 3, dtype=torch.bool),
		)

		profile: dict[str, list[float]] = {}
		_, low = opt_loss_snap_surf._map_init_objective(
			model_z_lift_theta=torch.zeros(1, 3, 3),
			profile_blocks=profile,
			**kwargs,
		)
		_, high = opt_loss_snap_surf._map_init_objective(
			model_z_lift_theta=torch.full((1, 3, 3), 2.0 * math.pi),
			**kwargs,
		)

		self.assertAlmostEqual(float(low["turn"].detach()), 0.0)
		self.assertGreater(float(high["turn"].detach()), 4.0)
		self.assertIn("sample_model_context", profile)
		self.assertNotIn("sample_turn_field", profile)
		self.assertNotIn("sample_turn_ext_field", profile)

	def test_map_init_z_lift_samples_full_external_theta_at_coarse_level(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		ext_coords = torch.tensor([[[0.0, 0.0], [0.0, 4.0]], [[4.0, 0.0], [4.0, 4.0]]])
		ext_theta = torch.zeros(5, 5)
		ext_theta[0, 0] = 2.0 * math.pi
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=1,
				w_dist=0.0,
				w_vec_normal=0.0,
				w_surface_normal=0.0,
				w_z_lift=1.0,
				w_smooth=0.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
			),
		)

		_loss, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=5, w=5, z=0.0),
			ext_normals=_normals_2d(5, 5),
			ext_valid=torch.ones(5, 5, dtype=torch.bool),
			ext_quad_valid=torch.ones(4, 4, dtype=torch.bool),
			ext_coords=ext_coords,
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			sign=1,
			cfg=cfg,
			ext_z_lift_theta=ext_theta,
			ext_z_lift_valid=torch.ones(5, 5, dtype=torch.bool),
			model_z_lift_theta=torch.zeros(1, 2, 2),
			model_z_lift_valid=torch.ones(1, 2, 2, dtype=torch.bool),
		)

		self.assertGreater(float(terms["turn"].detach()), 4.0)
		self.assertEqual(float(terms["turn_smp"].detach()), 1.0)

	def test_map_init_z_lift_invalid_theta_does_not_poison_uv_grad(self) -> None:
		uv = torch.tensor(
			[[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]],
			requires_grad=True,
		)
		loss, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(1, 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=3,
					w_dist=0.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_z_lift=1.0,
					w_smooth=0.0,
					w_bend=0.0,
					w_jac=0.0,
					w_metric_smooth=0.0,
					w_area_smooth=0.0,
				),
			),
			ext_z_lift_theta=torch.zeros(2, 2),
			ext_z_lift_valid=torch.ones(2, 2, dtype=torch.bool),
			model_z_lift_theta=torch.full((1, 2, 2), float("nan")),
			model_z_lift_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
			allow_partial_model_samples=True,
		)

		self.assertTrue(bool(torch.isfinite(loss.detach())))
		self.assertEqual(float(terms["turn_smp"].detach()), 0.0)
		loss.backward()
		self.assertIsNotNone(uv.grad)
		self.assertTrue(bool(torch.isfinite(uv.grad).all()))

	def test_map_init_one_active_quad_produces_subdiv_squared_samples(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=2, w=2, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=3),
			),
		)

		self.assertEqual(float(terms["active"].detach()), 1.0)
		self.assertEqual(float(terms["samples"].detach()), 9.0)

	def test_map_init_dense_objective_respects_sparse_active_quad_mask(self) -> None:
		hw = torch.stack(torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32), indexing="ij"), dim=-1)
		active_quad = torch.zeros(2, 2, dtype=torch.bool)
		active_quad[0, 0] = True

		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=hw,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
			),
		)

		self.assertEqual(float(terms["active"].detach()), 1.0)
		self.assertEqual(float(terms["sample_total"].detach()), 4.0)
		self.assertEqual(float(terms["samples"].detach()), 4.0)

	def test_map_init_coarse_quad_samples_full_external_span(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		ext_coords = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_coords=ext_coords,
			model_xyz=_plane_xyz(h=2, w=2, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=3),
			),
		)

		self.assertEqual(float(terms["sample_total"].detach()), 9.0)
		self.assertEqual(float(terms["samples"].detach()), 9.0)

	def test_map_init_invalid_high_res_external_sample_rejects_coarse_quad(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		ext_coords = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		ext_valid = torch.ones(3, 3, dtype=torch.bool)
		ext_valid[1, 1] = False

		ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=ext_valid,
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_coords=ext_coords,
			model_xyz=_plane_xyz(h=2, w=2, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=1),
			),
		)

		self.assertFalse(bool(ok[0]))

	def test_map_init_coarse_candidate_accepts_one_model_reachable_sample(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		model_valid = torch.zeros(1, 3, 3, dtype=torch.bool)
		model_valid[0, 0:2, 0:2] = True
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
		)

		strict_ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			cfg=cfg,
		)
		coarse_ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			cfg=cfg,
			allow_partial_model_samples=True,
		)

		self.assertFalse(bool(strict_ok[0]))
		self.assertTrue(bool(coarse_ok[0]))

	def test_map_init_coarse_partial_model_keeps_external_validity_strict(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		ext_coords = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		ext_valid = torch.ones(3, 3, dtype=torch.bool)
		ext_valid[1, 1] = False
		model_valid = torch.zeros(1, 3, 3, dtype=torch.bool)
		model_valid[0, 0:2, 0:2] = True

		ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=ext_valid,
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_coords=ext_coords,
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
			),
			allow_partial_model_samples=True,
		)

		self.assertFalse(bool(ok[0]))

	def test_map_init_sample_distance_limit_rejects_candidate_quad(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		model_xyz = _plane_xyz(h=2, w=2, z=600.0).unsqueeze(0)

		ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=1, max_sample_distance=500.0),
			),
		)

		self.assertFalse(bool(ok[0]))

	def test_map_init_sample_angle_limit_rejects_sideways_connection(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		model_xyz[..., 0] += 100.0

		ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=1,
					max_sample_distance=500.0,
					max_sample_angle_deg=45.0,
				),
			),
		)

		self.assertFalse(bool(ok[0]))

	def test_map_init_sample_angle_limit_accepts_front_and_back_normal_connections(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=1,
				max_sample_distance=500.0,
				max_sample_angle_deg=45.0,
			),
		)

		def accepted(model_z: float) -> bool:
			ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
				uv_full=uv,
				quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
				ext_pos=_plane_xyz(h=2, w=2, z=0.0),
				ext_normals=_normals_2d(2, 2),
				ext_valid=torch.ones(2, 2, dtype=torch.bool),
				ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
				model_xyz=_plane_xyz(h=2, w=2, z=model_z).unsqueeze(0),
				model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
				model_normals=_normals_3d(1, 2, 2),
				model_depth=0,
				sign=1,
				cfg=cfg,
			)
			return bool(ok[0])

		self.assertTrue(accepted(1.0))
		self.assertTrue(accepted(-1.0))

	def test_map_init_sample_angle_limit_scales_with_quad_step(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		ext_xyz[..., :2] *= 1000.0
		model_xyz = ext_xyz.unsqueeze(0).clone()
		model_xyz[..., 0] += 100.0
		model_xyz[..., 2] += 1.0

		ok = opt_loss_snap_surf._map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=torch.tensor([[0, 0]], dtype=torch.long),
			ext_pos=ext_xyz,
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 2),
			model_depth=0,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=1,
					max_sample_distance=500.0,
					max_sample_angle_deg=45.0,
					sample_angle_step_fraction=0.1,
				),
			),
		)

		self.assertTrue(bool(ok[0]))

	def test_map_init_step_neighbor_ratio_marks_long_step_quad(self) -> None:
		H, W = 3, 4
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		model_xyz = _plane_xyz(h=H, w=W, z=0.0).unsqueeze(0)
		model_xyz[0, 1, 2, 1] = 100.0
		active = torch.ones(H - 1, W - 1, dtype=torch.bool)

		bad = opt_loss_snap_surf._map_init_step_neighbor_bad_quad_mask(
			uv,
			active,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_depth=0,
			max_ratio=10.0,
		)

		self.assertTrue(bool(bad.any()))
		self.assertTrue(bool(bad[0, 1]))

	def test_map_init_objective_reports_sample_loss_and_bad_fraction(self) -> None:
		H, W = 2, 2
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		model_valid = torch.ones(1, H, W, dtype=torch.bool)
		model_valid[0, 1, 1] = False

		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(1, 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=H, w=W, z=0.0),
			ext_normals=_normals_2d(H, W),
			ext_valid=torch.ones(H, W, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, H, W),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
			),
		)

		self.assertEqual(float(terms["sample_total"].detach()), 4.0)
		self.assertGreater(float(terms["sample_bad"].detach()), 0.0)
		self.assertGreater(float(terms["sample_bad_frac"].detach()), 0.0)
		self.assertTrue(torch.isfinite(terms["sample_loss"]))

	def test_map_init_objective_coarse_partial_model_uses_reachable_samples(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 2.0]]])
		model_valid = torch.zeros(1, 3, 3, dtype=torch.bool)
		model_valid[0, 0:2, 0:2] = True
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
		)

		_, strict_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(1, 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=cfg,
		)
		_, coarse_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(1, 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=2, w=2, z=0.0),
			ext_normals=_normals_2d(2, 2),
			ext_valid=torch.ones(2, 2, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=0.0).unsqueeze(0),
			model_valid=model_valid,
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=cfg,
			allow_partial_model_samples=True,
		)

		self.assertEqual(float(strict_terms["quad_success"].detach()), 0.0)
		self.assertEqual(float(coarse_terms["quad_success"].detach()), 1.0)
		self.assertEqual(float(coarse_terms["samples"].detach()), 1.0)
		self.assertEqual(float(coarse_terms["sample_valid"].detach()), 1.0)
		self.assertEqual(float(coarse_terms["sample_bad"].detach()), 3.0)
		self.assertEqual(float(coarse_terms["model_bad"].detach()), 0.0)
		self.assertTrue(torch.isfinite(coarse_terms["loss"]))

	def test_map_init_objective_quad_success_includes_jac_failures(self) -> None:
		H, W = 2, 2
		uv = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])

		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(1, 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=H, w=W, z=0.0),
			ext_normals=_normals_2d(H, W),
			ext_valid=torch.ones(H, W, dtype=torch.bool),
			ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=H, w=W, z=0.0).unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2),
			),
		)

		self.assertEqual(float(terms["sample_valid"].detach()), float(terms["sample_total"].detach()))
		self.assertGreater(float(terms["jac_bad_quad"].detach()), 0.0)
		self.assertEqual(float(terms["quad_success"].detach()), 0.0)
		self.assertEqual(float(terms["quad_success_frac"].detach()), 0.0)

	def test_map_init_angle_distance_multiplier_at_ninety_degrees(self) -> None:
		cfg = opt_loss_snap_surf.SnapSurfMapInitConfig(angle_dist_mult=9.0)
		got = opt_loss_snap_surf._map_init_distance_multiplier(
			torch.tensor([1.0]),
			torch.tensor([0.0]),
			cfg,
		)

		self.assertAlmostEqual(float(got[0]), 10.0, places=5)

	def test_map_init_connection_terms_are_front_back_invariant(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=2,
				w_dist=1.0,
				w_vec_normal=1.0,
				w_surface_normal=1.0,
				angle_dist_mult=3.0,
				w_smooth=0.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
				w_dense_prior=0.0,
			),
		)

		def terms_for(model_z: float) -> dict[str, torch.Tensor]:
			_, terms = opt_loss_snap_surf._map_init_objective(
				uv_full=uv,
				active_quad=torch.ones(1, 1, dtype=torch.bool),
				ext_pos=_plane_xyz(h=2, w=2, z=0.0),
				ext_normals=_normals_2d(2, 2),
				ext_valid=torch.ones(2, 2, dtype=torch.bool),
				ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
				model_xyz=_plane_xyz(h=2, w=2, z=model_z).unsqueeze(0),
				model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
				model_normals=_normals_3d(1, 2, 2),
				model_depth=0,
				sign=1,
				cfg=cfg,
			)
			return terms

		front = terms_for(1.0)
		back = terms_for(-1.0)
		self.assertAlmostEqual(float(front["vec"].detach()), float(back["vec"].detach()), places=6)
		self.assertAlmostEqual(float(front["dist"].detach()), float(back["dist"].detach()), places=6)
		self.assertAlmostEqual(float(front["norm"].detach()), float(back["norm"].detach()), places=6)
		self.assertAlmostEqual(float(front["vec"].detach()), 0.0, places=6)

	def test_map_init_surface_normal_loss_uses_model_alignment_sign(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		model_normals = -_normals_3d(1, 2, 2)

		def norm_loss(sign: int) -> float:
			_, terms = opt_loss_snap_surf._map_init_objective(
				uv_full=uv,
				active_quad=torch.ones(1, 1, dtype=torch.bool),
				ext_pos=_plane_xyz(h=2, w=2, z=0.0),
				ext_normals=_normals_2d(2, 2),
				ext_valid=torch.ones(2, 2, dtype=torch.bool),
				ext_quad_valid=torch.ones(1, 1, dtype=torch.bool),
				model_xyz=_plane_xyz(h=2, w=2, z=1.0).unsqueeze(0),
				model_valid=torch.ones(1, 2, 2, dtype=torch.bool),
				model_normals=model_normals,
				model_depth=0,
				sign=sign,
				cfg=opt_loss_snap_surf.SnapSurfConfig(
					map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
						subdiv=2,
						w_dist=0.0,
						w_vec_normal=0.0,
						w_surface_normal=1.0,
						w_smooth=0.0,
						w_bend=0.0,
						w_jac=0.0,
						w_metric_smooth=0.0,
						w_area_smooth=0.0,
						w_dense_prior=0.0,
					),
				),
			)
			return float(terms["norm"].detach())

		self.assertGreater(norm_loss(1), 1.9)
		self.assertLess(norm_loss(-1), 1.0e-6)

	def test_map_init_jacobian_penalty_catches_flipped_cells(self) -> None:
		active = torch.ones(1, 1, dtype=torch.bool)
		uv_ok = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		uv_flip = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [-1.0, 1.0]]])

		ok_pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv_ok,
			active,
			jac_margin=0.05,
		)
		flip_pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv_flip,
			active,
			jac_margin=0.05,
		)

		self.assertAlmostEqual(float(ok_pen.detach()), 0.0, places=6)
		self.assertGreater(float(flip_pen.detach()), 0.0)

	def test_map_init_inverse_regularization_penalizes_compression(self) -> None:
		active = torch.ones(1, 1, dtype=torch.bool)
		uv_ok = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		uv_compressed = uv_ok * 0.1

		ok_terms = opt_loss_snap_surf._map_init_inverse_regularization_terms(
			uv_ok,
			active,
			jac_margin=0.05,
		)
		compressed_terms = opt_loss_snap_surf._map_init_inverse_regularization_terms(
			uv_compressed,
			active,
			jac_margin=0.05,
		)

		self.assertAlmostEqual(float(ok_terms["smooth"].detach()), 1.0, places=6)
		self.assertGreater(float(compressed_terms["smooth"].detach()), 50.0)

	def test_map_init_inverse_jacobian_penalizes_large_expansion(self) -> None:
		active = torch.ones(1, 1, dtype=torch.bool)
		uv_expanded = torch.tensor([[[0.0, 0.0], [0.0, 30.0]], [[30.0, 0.0], [30.0, 30.0]]])

		forward_pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv_expanded,
			active,
			jac_margin=0.05,
		)
		reverse_terms = opt_loss_snap_surf._map_init_inverse_regularization_terms(
			uv_expanded,
			active,
			jac_margin=0.05,
		)

		self.assertAlmostEqual(float(forward_pen.detach()), 0.0, places=6)
		self.assertGreater(float(reverse_terms["jac"].detach()), 0.0)
		self.assertGreater(float(reverse_terms["jac_bad"].detach()), 0.0)

	def test_map_init_local_evenness_constant_scale_is_zero(self) -> None:
		H, W = 3, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh * 2.0, ww * 2.0], dim=-1)
		terms = opt_loss_snap_surf._map_init_local_evenness_terms(
			uv,
			_plane_xyz(h=H, w=W, z=0.0),
			torch.ones(H - 1, W - 1, dtype=torch.bool),
		)

		self.assertAlmostEqual(float(terms["metric_smooth"].detach()), 0.0, places=6)
		self.assertAlmostEqual(float(terms["area_smooth"].detach()), 0.0, places=6)

	def test_map_init_local_evenness_detects_stretched_edge(self) -> None:
		uv = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [2.0, 1.0]]])
		terms = opt_loss_snap_surf._map_init_local_evenness_terms(
			uv,
			_plane_xyz(h=2, w=2, z=0.0),
			torch.ones(1, 1, dtype=torch.bool),
		)

		self.assertGreater(float(terms["metric_smooth"].detach()), 0.0)
		self.assertAlmostEqual(float(terms["area_smooth"].detach()), 0.0, places=6)

	def test_map_init_local_evenness_metric_only_skips_area_cross_products(self) -> None:
		H, W = 3, 3
		uv = torch.stack(torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"), dim=-1)
		uv[1, 1] = uv[1, 1] + torch.tensor([0.25, 0.0])
		with mock.patch.object(torch, "cross", side_effect=AssertionError("area path should be inactive")):
			terms = opt_loss_snap_surf._map_init_local_evenness_terms(
				uv,
				_plane_xyz(h=H, w=W, z=0.0),
				torch.ones(H - 1, W - 1, dtype=torch.bool),
				need_metric=True,
				need_area=False,
			)

		self.assertGreater(float(terms["metric_smooth"].detach()), 0.0)
		self.assertAlmostEqual(float(terms["area_smooth"].detach()), 0.0, places=6)

	def test_map_init_local_evenness_area_only_skips_metric_lengths(self) -> None:
		H, W = 2, 3
		uv = torch.stack(torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"), dim=-1)
		ext = uv.detach().clone()
		uv[:, 2, 1] = 4.0
		with mock.patch.object(torch.Tensor, "norm", side_effect=AssertionError("metric length path should be inactive")):
			terms = opt_loss_snap_surf._map_init_local_evenness_terms(
				uv,
				ext,
				torch.ones(H - 1, W - 1, dtype=torch.bool),
				need_metric=False,
				need_area=True,
			)

		self.assertAlmostEqual(float(terms["metric_smooth"].detach()), 0.0, places=6)
		self.assertGreater(float(terms["area_smooth"].detach()), 0.0)

	def test_map_init_evenness_external_context_cache_matches_direct(self) -> None:
		H, W = 3, 4
		ext = _plane_xyz(h=H, w=W, z=0.0)
		ext[2, 3] = float("nan")
		active = torch.ones(H - 1, W - 1, dtype=torch.bool)
		active[1, 2] = False
		cache: dict[tuple[object, ...], object] = {}
		key = opt_loss_snap_surf._map_init_evenness_external_context_cache_key(
			ext_pos=ext,
			active_quad=active,
			prefix=("test",),
			external_static_cache_key=("level", 0),
		)

		direct = opt_loss_snap_surf._map_init_build_evenness_external_context(ext, active, need_metric=True, need_area=True)
		cached = opt_loss_snap_surf._map_init_cached_evenness_external_context(
			ext_pos=ext,
			active_quad=active,
			need_metric=True,
			need_area=True,
			cache=cache,
			key=key,
		)

		for name in (
			"finite_ext", "static_quad", "ext_len_h_valid", "ext_len_w_valid",
			"metric_h_pair_down", "metric_h_pair_right", "metric_w_pair_down", "metric_w_pair_right",
			"ext_area_valid", "area_pair_down", "area_pair_right",
		):
			self.assertTrue(torch.equal(getattr(cached, name), getattr(direct, name)), name)
		torch.testing.assert_close(cached.ext_len_h, direct.ext_len_h)
		torch.testing.assert_close(cached.ext_len_w, direct.ext_len_w)
		torch.testing.assert_close(cached.ext_area, direct.ext_area)

	def test_map_init_evenness_external_context_cache_upgrades_for_area(self) -> None:
		H, W = 3, 3
		ext = _plane_xyz(h=H, w=W, z=0.0)
		active = torch.ones(H - 1, W - 1, dtype=torch.bool)
		cache: dict[tuple[object, ...], object] = {}
		key = opt_loss_snap_surf._map_init_evenness_external_context_cache_key(ext_pos=ext, active_quad=active)

		metric_ctx = opt_loss_snap_surf._map_init_cached_evenness_external_context(
			ext_pos=ext,
			active_quad=active,
			need_metric=True,
			need_area=False,
			cache=cache,
			key=key,
		)
		area_ctx = opt_loss_snap_surf._map_init_cached_evenness_external_context(
			ext_pos=ext,
			active_quad=active,
			need_metric=False,
			need_area=True,
			cache=cache,
			key=key,
		)

		self.assertIsNone(metric_ctx.ext_area)
		self.assertIsNotNone(area_ctx.ext_area)
		self.assertIsNotNone(area_ctx.ext_len_h)

	def test_map_init_signed_reciprocal_scale_keeps_expansion_sign(self) -> None:
		eps_t = torch.tensor(0.0)
		scale = opt_loss_snap_surf._map_init_signed_reciprocal_scale(
			torch.tensor([2.0, 0.5]),
			torch.tensor([1.0, 1.0]),
			eps_t,
		)

		torch.testing.assert_close(scale, torch.tensor([0.75, -0.75]))

	def test_map_init_local_evenness_uses_model_surface_lengths(self) -> None:
		H, W = 4, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		ext_xyz = _plane_xyz(h=H, w=W, z=0.0)
		model_xyz = _plane_xyz(h=H, w=W, z=1.0)
		model_xyz[..., 1] = torch.tensor([0.0, 1.0, 5.0, 6.0]).view(H, 1).expand(H, W)
		active = torch.ones(H - 1, W - 1, dtype=torch.bool)

		uv_terms = opt_loss_snap_surf._map_init_local_evenness_terms(uv, ext_xyz, active)
		model_terms = opt_loss_snap_surf._map_init_local_evenness_terms(
			uv,
			ext_xyz,
			active,
			model_xyz=model_xyz.unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_depth=0,
		)

		self.assertAlmostEqual(float(uv_terms["metric_smooth"].detach()), 0.0, places=6)
		self.assertGreater(float(model_terms["metric_smooth"].detach()), 0.0)
		self.assertGreater(float(model_terms["area_smooth"].detach()), 0.0)

	def test_map_init_forward_smoothness_includes_uv_and_model_surface(self) -> None:
		H, W = 4, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		model_xyz = _plane_xyz(h=H, w=W, z=1.0)
		model_xyz[..., 1] = torch.tensor([0.0, 1.0, 5.0, 6.0]).view(H, 1).expand(H, W)

		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(H - 1, W - 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=H, w=W, z=0.0),
			ext_normals=_normals_2d(H, W),
			ext_valid=torch.ones(H, W, dtype=torch.bool),
			ext_quad_valid=torch.ones(H - 1, W - 1, dtype=torch.bool),
			model_xyz=model_xyz.unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					w_dist=0.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_jac=0.0,
					w_metric_smooth=0.0,
					w_area_smooth=0.0,
					w_dense_prior=0.0,
				),
			),
		)

		expected_smooth = terms["smooth_uv_fwd"] + terms["smooth_model_fwd"] + terms["smooth_rev"]
		expected_bend = terms["bend_uv_fwd"] + terms["bend_model_fwd"] + terms["bend_rev"]
		self.assertGreater(float(terms["smooth_model_fwd"].detach()), float(terms["smooth_uv_fwd"].detach()))
		self.assertAlmostEqual(float(terms["bend_uv_fwd"].detach()), 0.0, places=6)
		self.assertGreater(float(terms["bend_model_fwd"].detach()), 0.0)
		self.assertAlmostEqual(float(terms["smooth"].detach()), float(expected_smooth.detach()), places=6)
		self.assertAlmostEqual(float(terms["bend"].detach()), float(expected_bend.detach()), places=6)

	def test_map_init_model_surface_smoothness_is_physical_scale_normalized(self) -> None:
		H, W = 4, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		model_xyz = _plane_xyz(h=H, w=W, z=1.0)
		model_xyz[..., 1] = torch.tensor([0.0, 1.0, 5.0, 6.0]).view(H, 1).expand(H, W)

		def eval_terms(ext_pos: torch.Tensor) -> dict[str, torch.Tensor]:
			_, terms = opt_loss_snap_surf._map_init_objective(
				uv_full=uv,
				active_quad=torch.ones(H - 1, W - 1, dtype=torch.bool),
				ext_pos=ext_pos,
				ext_normals=_normals_2d(H, W),
				ext_valid=torch.ones(H, W, dtype=torch.bool),
				ext_quad_valid=torch.ones(H - 1, W - 1, dtype=torch.bool),
				model_xyz=model_xyz.unsqueeze(0),
				model_valid=torch.ones(1, H, W, dtype=torch.bool),
				model_normals=_normals_3d(1, H, W),
				model_depth=0,
				sign=1,
				cfg=opt_loss_snap_surf.SnapSurfConfig(
					map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
						w_dist=0.0,
						w_vec_normal=0.0,
						w_surface_normal=0.0,
						w_jac=0.0,
						w_metric_smooth=0.0,
						w_area_smooth=0.0,
						w_dense_prior=0.0,
					),
				),
			)
			return terms

		ext_unit = _plane_xyz(h=H, w=W, z=0.0)
		ext_scaled = ext_unit.clone()
		ext_scaled[..., :2] *= 10.0
		unit_terms = eval_terms(ext_unit)
		scaled_terms = eval_terms(ext_scaled)

		self.assertLess(float(scaled_terms["smooth_model_fwd"].detach()), float(unit_terms["smooth_model_fwd"].detach()) / 50.0)
		self.assertLess(float(scaled_terms["bend_model_fwd"].detach()), float(unit_terms["bend_model_fwd"].detach()) / 50.0)

	def test_map_init_local_evenness_detects_area_jump(self) -> None:
		H, W = 2, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		uv[:, 2, 1] = 4.0
		terms = opt_loss_snap_surf._map_init_local_evenness_terms(
			uv,
			_plane_xyz(h=H, w=W, z=0.0),
			torch.ones(H - 1, W - 1, dtype=torch.bool),
		)

		self.assertGreater(float(terms["area_smooth"].detach()), 0.0)

	def test_map_init_local_evenness_ignores_inactive_nans(self) -> None:
		uv = torch.full((3, 3, 2), float("nan"))
		uv[:2, :2] = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		active = torch.zeros(2, 2, dtype=torch.bool)
		active[0, 0] = True
		terms = opt_loss_snap_surf._map_init_local_evenness_terms(
			uv,
			_plane_xyz(h=3, w=3, z=0.0),
			active,
		)

		self.assertTrue(torch.isfinite(terms["metric_smooth"]))
		self.assertTrue(torch.isfinite(terms["area_smooth"]))
		self.assertAlmostEqual(float(terms["metric_smooth"].detach()), 0.0, places=6)
		self.assertAlmostEqual(float(terms["area_smooth"].detach()), 0.0, places=6)

	def test_map_init_objective_includes_local_evenness_weights(self) -> None:
		H, W = 2, 3
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		uv[:, 2, 1] = 4.0
		loss, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(H - 1, W - 1, dtype=torch.bool),
			ext_pos=_plane_xyz(h=H, w=W, z=0.0),
			ext_normals=_normals_2d(H, W),
			ext_valid=torch.ones(H, W, dtype=torch.bool),
			ext_quad_valid=torch.ones(H - 1, W - 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=2, w=5, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 2, 5, dtype=torch.bool),
			model_normals=_normals_3d(1, 2, 5),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					w_dist=0.0,
					w_vec_normal=0.0,
					w_surface_normal=0.0,
					w_smooth=0.0,
					w_bend=0.0,
					w_jac=0.0,
					w_metric_smooth=2.0,
					w_area_smooth=3.0,
					w_dense_prior=0.0,
				),
			),
		)
		expected = 2.0 * terms["metric_smooth"] + 3.0 * terms["area_smooth"]

		self.assertGreater(float(terms["metric_smooth"].detach()), 0.0)
		self.assertGreater(float(terms["area_smooth"].detach()), 0.0)
		self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=6)

	def test_map_init_local_jacobian_pass_rejects_overexpanded_lr_quad(self) -> None:
		active_quad = torch.ones(1, 1, dtype=torch.bool)
		uv_expanded = torch.tensor([[[0.0, 0.0], [0.0, 30.0]], [[30.0, 0.0], [30.0, 30.0]]])

		self.assertFalse(opt_loss_snap_surf._map_init_local_jacobian_pass(
			uv_expanded,
			active_quad,
			h=0,
			w=0,
			jac_margin=0.05,
		))

	def test_map_init_dense_objective_regularizes_inactive_field(self) -> None:
		active_quad = torch.zeros(2, 2, dtype=torch.bool)
		active_quad[0, 0] = True
		hh = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3)
		ww = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3)
		uv_flip = torch.stack([hh, ww], dim=-1)
		uv_flip[2, 1] = torch.tensor([0.0, 1.0])
		uv_flip[2, 2] = torch.tensor([0.0, 2.0])
		model_xyz = _plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		_, sparse_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv_flip,
			active_quad=active_quad,
			ext_pos=ext_xyz,
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)
		_, dense_terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv_flip,
			active_quad=active_quad,
			ext_pos=ext_xyz,
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(dense_opt=True),
			),
		)

		self.assertEqual(float(sparse_terms["jac_bad"].detach()), 0.0)
		self.assertGreater(float(dense_terms["jac_bad"].detach()), 0.0)
		self.assertGreater(float(dense_terms["reg"].detach()), float(sparse_terms["reg"].detach()))

	def test_map_init_dense_objective_regularizes_only_active_band(self) -> None:
		H, W = 7, 7
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		uv = torch.stack([hh, ww], dim=-1)
		active_quad = torch.zeros(H - 1, W - 1, dtype=torch.bool)
		active_quad[3, 3] = True
		ext_valid = torch.ones(H, W, dtype=torch.bool)
		ext_valid[2, 2] = False

		_, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=active_quad,
			ext_pos=_plane_xyz(h=H, w=W, z=0.0),
			ext_normals=_normals_2d(H, W),
			ext_valid=ext_valid,
			ext_quad_valid=torch.ones(H - 1, W - 1, dtype=torch.bool),
			model_xyz=_plane_xyz(h=H, w=W, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			model_normals=_normals_3d(1, H, W),
			model_depth=0,
			sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					dense_opt=True,
					dense_reg_radius=2,
				),
			),
		)

		self.assertEqual(float(terms["reg"].detach()), 35.0)

	def test_map_init_jacobian_penalty_empty_cells_ignores_inactive_nans(self) -> None:
		active_quad = torch.zeros(2, 2, dtype=torch.bool)
		uv = torch.full((3, 3, 2), float("nan"))
		uv[1, 1] = torch.tensor([1.0, 1.0])

		pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv,
			active_quad,
			jac_margin=0.05,
		)

		self.assertTrue(torch.isfinite(pen))
		self.assertAlmostEqual(float(pen.detach()), 0.0, places=6)

	def test_map_init_compile_objective_matches_eager_training_terms_and_grad(self) -> None:
		H, W = 4, 4
		hh = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		ww = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		base_uv = torch.stack([hh, ww], dim=-1)
		uv_init = base_uv.clone()
		uv_init[1, 1] = uv_init[1, 1] + torch.tensor([0.15, -0.05])
		uv_init[2, 2] = uv_init[2, 2] + torch.tensor([-0.1, 0.2])
		active_quad = torch.ones(H - 1, W - 1, dtype=torch.bool)
		ext_pos = _plane_xyz(h=H, w=W, z=0.0)
		ext_normals = _normals_2d(H, W)
		ext_valid = torch.ones(H, W, dtype=torch.bool)
		ext_quad_valid = torch.ones(H - 1, W - 1, dtype=torch.bool)
		model_xyz = _plane_xyz(h=H, w=W, z=1.0).unsqueeze(0)
		model_valid = torch.ones(1, H, W, dtype=torch.bool)
		model_normals = _normals_3d(1, H, W)

		def _cfg(*, compile_objective: bool) -> opt_loss_snap_surf.SnapSurfConfig:
			return opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					subdiv=2,
					dense_opt=True,
					dense_reg_radius=1,
					w_dist=0.7,
					w_vec_normal=0.2,
					w_surface_normal=0.3,
					z_lift_enabled=False,
					w_z_lift=0.0,
					w_smooth=0.05,
					w_bend=0.02,
					w_jac=0.6,
					w_metric_smooth=0.4,
					w_area_smooth=0.0,
					w_dense_prior=0.1,
					max_sample_angle_deg=180.0,
					compile_objective=compile_objective,
				),
			)

		def _run(cfg: opt_loss_snap_surf.SnapSurfConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
			uv = uv_init.detach().clone().requires_grad_(True)
			loss, terms = opt_loss_snap_surf._map_init_objective(
				uv_full=uv,
				active_quad=active_quad,
				ext_pos=ext_pos,
				ext_normals=ext_normals,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				model_depth=0,
				sign=1,
				cfg=cfg,
				uv_prior=base_uv,
				need_stats=False,
				runtime_cache={},
			)
			grad = torch.autograd.grad(loss, uv)[0]
			return loss.detach(), terms, grad.detach()

		eager_loss, eager_terms, eager_grad = _run(_cfg(compile_objective=False))
		map_objective._MAP_INIT_COMPILED_FN_CACHE.clear()
		with mock.patch.object(map_objective.torch, "compile", side_effect=lambda fn, **_kwargs: fn, create=True):
			compiled_loss, compiled_terms, compiled_grad = _run(_cfg(compile_objective=True))

		torch.testing.assert_close(compiled_loss, eager_loss, rtol=1.0e-6, atol=1.0e-6)
		for key in ("loss", "dist", "vec", "norm", "turn", "smooth", "bend", "jac", "metric_smooth", "area_smooth", "prior"):
			torch.testing.assert_close(compiled_terms[key], eager_terms[key], rtol=1.0e-6, atol=1.0e-6)
		torch.testing.assert_close(compiled_grad, eager_grad, rtol=1.0e-6, atol=1.0e-6)
		compiled_names = {str(key[0]) for key in map_objective._MAP_INIT_COMPILED_FN_CACHE}
		self.assertTrue({
			"sample_reduce",
			"forward_smooth_bend",
			"jacobian_penalty",
			"inverse_regularization",
			"metric_evenness",
			"prior_loss",
		}.issubset(compiled_names))
		self.assertNotIn("area_evenness", compiled_names)

	def test_map_init_objective_skips_zero_weight_training_branches(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32), indexing="ij"), dim=-1)
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=2,
				dense_opt=True,
				w_dist=0.0,
				w_vec_normal=0.0,
				w_surface_normal=0.0,
				w_z_lift=0.0,
				w_smooth=0.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
				w_dense_prior=0.0,
				compile_objective=True,
			),
		)
		map_objective._MAP_INIT_COMPILED_FN_CACHE.clear()
		with (
			mock.patch.object(map_objective.torch, "compile", side_effect=AssertionError("zero-weight branches should not compile"), create=True) as compile_mock,
			mock.patch.object(opt_loss_snap_surf, "_map_init_dense_quad_sample_tensors", wraps=opt_loss_snap_surf._map_init_dense_quad_sample_tensors) as sample_mock,
			mock.patch.object(opt_loss_snap_surf, "_map_init_model_metric_positions", wraps=opt_loss_snap_surf._map_init_model_metric_positions) as metric_mock,
			mock.patch.object(opt_loss_snap_surf, "_map_init_inverse_regularization_terms", wraps=opt_loss_snap_surf._map_init_inverse_regularization_terms) as inv_mock,
			mock.patch.object(opt_loss_snap_surf, "_map_init_local_evenness_terms", wraps=opt_loss_snap_surf._map_init_local_evenness_terms) as even_mock,
		):
			loss, terms = opt_loss_snap_surf._map_init_objective(
				uv_full=uv.requires_grad_(True),
				active_quad=torch.ones(2, 2, dtype=torch.bool),
				ext_pos=_plane_xyz(h=3, w=3, z=0.0),
				ext_normals=_normals_2d(3, 3),
				ext_valid=torch.ones(3, 3, dtype=torch.bool),
				ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
				model_xyz=_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0),
				model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
				model_normals=_normals_3d(1, 3, 3),
				model_depth=0,
				sign=1,
				cfg=cfg,
				uv_prior=uv.detach().clone(),
				need_stats=False,
			)

		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		for key in ("dist", "vec", "norm", "turn", "smooth", "bend", "jac", "metric_smooth", "area_smooth", "prior"):
			self.assertIn(key, terms)
			self.assertAlmostEqual(float(terms[key].detach()), 0.0, places=6)
		self.assertEqual(sample_mock.call_count, 0)
		self.assertEqual(metric_mock.call_count, 0)
		self.assertEqual(inv_mock.call_count, 0)
		self.assertEqual(even_mock.call_count, 0)
		self.assertEqual(compile_mock.call_count, 0)

	def test_map_init_objective_zero_weight_diagnostics_still_compute_terms(self) -> None:
		uv = torch.stack(torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32), indexing="ij"), dim=-1)
		uv[1, 1] = uv[1, 1] + torch.tensor([0.25, -0.15])
		cfg = opt_loss_snap_surf.SnapSurfConfig(
			map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
				subdiv=2,
				w_dist=0.0,
				w_vec_normal=0.0,
				w_surface_normal=0.0,
				w_smooth=0.0,
				w_bend=0.0,
				w_jac=0.0,
				w_metric_smooth=0.0,
				w_area_smooth=0.0,
			),
		)

		_loss, terms = opt_loss_snap_surf._map_init_objective(
			uv_full=uv,
			active_quad=torch.ones(2, 2, dtype=torch.bool),
			ext_pos=_plane_xyz(h=3, w=3, z=0.0),
			ext_normals=_normals_2d(3, 3),
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_quad_valid=torch.ones(2, 2, dtype=torch.bool),
			model_xyz=_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0),
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			model_normals=_normals_3d(1, 3, 3),
			model_depth=0,
			sign=1,
			cfg=cfg,
			need_stats=True,
		)

		self.assertGreater(float(terms["dist"].detach()), 0.0)
		self.assertGreater(float(terms["smooth"].detach()), 0.0)
		self.assertGreater(float(terms["metric_smooth"].detach()), 0.0)

if __name__ == "__main__":
	unittest.main()
