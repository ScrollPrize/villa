from __future__ import annotations

import os
import contextlib
import io
import tempfile
import unittest

import torch

from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz, _result, opt_loss_snap_surf


class SnapSurfLegacyTest(unittest.TestCase):
	def setUp(self) -> None:
		opt_loss_snap_surf.reset_state()
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(1.0, 1.0, 0.0),
			active=True,
		)

	def test_mapper_can_grow_from_one_inlier_in_direct_search_mode(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(3, 3), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 1, 1] = torch.tensor([1.0, 1.0])
		state.valid[0, 1, 1] = True
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		target = _plane_xyz(h=3, w=3, z=0.0)
		valid_source = torch.ones(1, 3, 3, dtype=torch.bool)
		valid_target = torch.ones(3, 3, dtype=torch.bool)

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=valid_source,
			target_xyz=target,
			target_valid=valid_target,
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=10.0, grid_error=2.0),
		)

		self.assertGreater(state.count(), 1)
		self.assertTrue(bool(state.valid[0, 1, 2]))

	def test_two_inlier_similarity_predicts_neighbor_step(self) -> None:
		source = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
		target = torch.tensor([[10.0, 20.0], [10.0, 21.0]])
		query = torch.tensor([1.0, 1.0])

		got = opt_loss_snap_surf._predict_target_coord(source, target, query, orientation_sign=1)

		self.assertTrue(torch.allclose(got, torch.tensor([11.0, 21.0]), atol=1.0e-6))

	def test_affine_three_inlier_mapping_predicts_grid_coords(self) -> None:
		source = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
		target = torch.tensor([[5.0, 7.0], [5.0, 8.0], [6.0, 7.0]])
		query = torch.tensor([1.0, 1.0])

		got = opt_loss_snap_surf._predict_target_coord(source, target, query, orientation_sign=1)

		self.assertTrue(torch.allclose(got, torch.tensor([6.0, 8.0]), atol=1.0e-6))

	def test_seed_initialization_creates_four_oriented_correspondences(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		res = _result(model_xyz, ext_xyz)

		opt_loss_snap_surf.snap_surf_loss(res=res)
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 4)
		self.assertEqual(state.ext_to_model.count(), 0)
		self.assertTrue(torch.allclose(state.model_to_ext.map[0, 0, 0], torch.tensor([0.0, 0.0])))
		self.assertAlmostEqual(opt_loss_snap_surf.last_stats()["snaps_sdist"], 0.0, places=6)
		self.assertAlmostEqual(opt_loss_snap_surf.last_stats()["snaps_sext"], 0.0, places=6)

	def test_first_loss_call_logs_configured_offset_and_current_interpretation(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		buf = io.StringIO()

		with contextlib.redirect_stdout(buf):
			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz, offset=1.5))
			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz, offset=1.5))

		out = buf.getvalue()
		self.assertEqual(out.count("external surface offsets at first loss call"), 1)
		self.assertIn("configured=[1.5]", out)
		self.assertIn("used_by_snap_surf=not_applied", out)

	def test_seed_initialization_detects_flipped_quad_orientation(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		model_xyz[..., 0] = 1.0 - model_xyz[..., 0]
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		res = _result(model_xyz, ext_xyz)

		opt_loss_snap_surf.snap_surf_loss(res=res)
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 4)
		self.assertEqual(state.ext_to_model.count(), 0)
		self.assertTrue(torch.allclose(state.model_to_ext.map[0, 0, 0], torch.tensor([0.0, 1.0])))

	def test_seed_orientation_is_scale_invariant(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		model_xyz[..., 0] = 1.0 - model_xyz[..., 0]
		ext_xyz = 20.0 * _plane_xyz(h=2, w=2, z=0.0)
		res = _result(model_xyz, ext_xyz)

		opt_loss_snap_surf.snap_surf_loss(res=res)
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 4)
		self.assertEqual(state.ext_to_model.count(), 0)

	def test_seed_region_rays_replace_growth_affine_convention(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		model_xyz[..., 0] = 1.0 - model_xyz[..., 0]
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 4)
		self.assertEqual(state.ext_to_model.count(), 0)
		self.assertTrue(torch.isfinite(state.model_to_ext.map[state.model_to_ext.valid]).all())

	def test_direct_growth_accepts_one_ring_candidates_without_orientation_gate(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 5, 5), target_shape=(5, 5), device=torch.device("cpu"), dtype=torch.float32)
		for h, w in ((1, 1), (2, 1), (1, 2), (2, 2)):
			state.valid[0, h, w] = True
			state.map[0, h, w] = torch.tensor([float(h), float(w)])

		stats = opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=_plane_xyz(h=5, w=5, z=0.0).unsqueeze(0),
			source_valid=torch.ones(1, 5, 5, dtype=torch.bool),
			target_xyz=_plane_xyz(h=5, w=5, z=0.0),
			target_valid=torch.ones(5, 5, dtype=torch.bool),
			normal_xyz=_normals_2d(5, 5),
			normal_from_source=False,
			cfg=opt_loss_snap_surf.SnapSurfConfig(affine_radius=2, search_ring=1),
		)

		self.assertGreater(stats["new"], 0)
		self.assertEqual(stats["ori"], stats["new"])

	def test_seed_attachment_uses_closest_surface_not_quad_center(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		ext_xyz = 50.0 * _plane_xyz(h=2, w=2, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 1.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(0.0, 0.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()

		self.assertAlmostEqual(stats["snaps_sext"], 0.0, places=6)
		self.assertAlmostEqual(stats["snaps_sdist"], 0.0, places=6)
		self.assertEqual(opt_loss_snap_surf._states[0].model_to_ext.count(), 4)

	def test_normal_distance_does_not_drop_ray_correspondences(self) -> None:
		model_xyz = _plane_xyz(h=2, w=2, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=2, w=2, z=0.0)
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))

		far_model = (model_xyz + torch.tensor([0.0, 0.0, 100.0])).clone()
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 1.0, "point_distance": 5.0, "grid_error": 0.25},
			seed_xyz=(0.5, 0.5, 0.0),
			active=True,
		)
		opt_loss_snap_surf.snap_surf_loss(res=_result(far_model, ext_xyz))
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 4)
		self.assertEqual(state.ext_to_model.count(), 0)

	def test_snap_loss_floods_from_seed_quad(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(1.5, 1.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]
		valid = state.model_to_ext.valid[0].nonzero(as_tuple=False)

		self.assertGreater(len(valid), 12)

	def test_seed_radius_does_not_limit_seed_grown_inliers(self) -> None:
		model_xyz = _plane_xyz(h=8, w=8, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=8, w=8, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "seed_radius": 1},
			seed_xyz=(3.5, 3.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]

		self.assertEqual(state.model_to_ext.count(), 64)
		self.assertEqual(state.ext_to_model.count(), 0)

	def test_existing_ray_correspondences_use_local_update_before_bruteforce(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "search_ring": 0},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()

		self.assertGreater(stats["snaps_local"], 0.0)
		self.assertEqual(stats["snaps_brute"], 0.0)
		self.assertGreater(stats["snaps_pairs_m"], 0.0)

	def test_invalid_finite_ray_correspondence_is_refined_without_bruteforce(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "brute_interval": 10},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]
		state.model_to_ext.valid[0, 2, 2] = False
		state.model_to_ext.map[0, 2, 2] = torch.tensor([2.0, 2.0])

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()

		self.assertEqual(stats["snaps_brute_on"], 0.0)
		self.assertEqual(stats["snaps_brute"], 0.0)
		self.assertEqual(state.model_to_ext.count(), 25)
		self.assertTrue(bool(state.model_to_ext.valid[0, 2, 2]))

	def test_bruteforce_runs_only_on_interval(self) -> None:
		model_xyz = _plane_xyz(h=5, w=5, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=5, w=5, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "search_ring": 0, "brute_interval": 10},
			seed_xyz=(2.0, 2.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]
		state.model_to_ext.valid[0, 2, 2] = False
		state.model_to_ext.map[0, 2, 2] = torch.tensor([float("nan"), float("nan")])

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()
		self.assertEqual(stats["snaps_brute_on"], 0.0)
		self.assertEqual(stats["snaps_brute"], 0.0)
		self.assertEqual(state.model_to_ext.count(), 24)

		for _ in range(8):
			opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()
		self.assertEqual(stats["snaps_brute_on"], 1.0)
		self.assertGreater(stats["snaps_brute"], 0.0)
		self.assertEqual(state.model_to_ext.count(), 25)

	def test_bruteforce_is_limited_to_seed_front_initially(self) -> None:
		model_xyz = _plane_xyz(h=31, w=31, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=31, w=31, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "search_ring": 0, "brute_boundary_radius": 1},
			seed_xyz=(15.0, 15.0, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()

		self.assertEqual(stats["snaps_brute_on"], 1.0)
		self.assertEqual(stats["snaps_front"], 16.0)
		self.assertEqual(stats["snaps_brute"], 16.0)
		self.assertLess(opt_loss_snap_surf._states[0].model_to_ext.count(), 31 * 31)

	def test_seeded_mapping_filter_rejects_local_global_jump(self) -> None:
		raw_map = torch.full((1, 4, 5, 2), float("nan"))
		raw_valid = torch.zeros(1, 4, 5, dtype=torch.bool)
		for h, w in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3)):
			raw_valid[0, h, w] = True
			raw_map[0, h, w] = torch.tensor([float(h), float(w)])
		raw_valid[0, 2, 4] = True
		raw_map[0, 2, 4] = torch.tensor([40.0, 40.0])

		inlier, stats = opt_loss_snap_surf._seeded_mapping_inlier_filter(
			raw_valid=raw_valid,
			raw_map=raw_map,
			seed_quad=(0, 1, 1),
			max_distance=8.0,
		)

		self.assertTrue(bool(inlier[0, 2, 3]))
		self.assertFalse(bool(inlier[0, 2, 4]))
		self.assertEqual(stats, {})

	def test_seeded_mapping_filter_rejects_normal_distance_jump(self) -> None:
		raw_map = torch.full((1, 1, 3, 2), float("nan"))
		raw_valid = torch.ones(1, 1, 3, dtype=torch.bool)
		normal_dist = torch.full((1, 1, 3), float("nan"))
		for w in range(3):
			raw_map[0, 0, w] = torch.tensor([0.0, float(w)])
		normal_dist[0, 0, 0] = 10.0
		normal_dist[0, 0, 1] = 14.0
		normal_dist[0, 0, 2] = 30.0
		initial = torch.zeros_like(raw_valid)
		initial[0, 0, 0] = True

		inlier, stats = opt_loss_snap_surf._seeded_mapping_inlier_filter(
			raw_valid=raw_valid,
			raw_map=raw_map,
			initial_inlier=initial,
			max_distance=8.0,
			normal_dist=normal_dist,
			max_normal_ratio=1.5,
		)

		self.assertTrue(bool(inlier[0, 0, 1]))
		self.assertFalse(bool(inlier[0, 0, 2]))
		self.assertEqual(stats, {})

	def test_seeded_mapping_filter_clamps_small_normal_distances(self) -> None:
		raw_map = torch.full((1, 1, 2, 2), float("nan"))
		raw_valid = torch.ones(1, 1, 2, dtype=torch.bool)
		raw_map[0, 0, 0] = torch.tensor([0.0, 0.0])
		raw_map[0, 0, 1] = torch.tensor([0.0, 1.0])
		normal_dist = torch.tensor([[[1.0, 9.0]]])
		initial = torch.zeros_like(raw_valid)
		initial[0, 0, 0] = True

		inlier, stats = opt_loss_snap_surf._seeded_mapping_inlier_filter(
			raw_valid=raw_valid,
			raw_map=raw_map,
			initial_inlier=initial,
			max_distance=8.0,
			normal_dist=normal_dist,
			max_normal_ratio=1.5,
			normal_distance_floor=10.0,
		)

		self.assertTrue(bool(inlier[0, 0, 1]))
		self.assertEqual(stats, {})

	def test_ray_intersection_rejects_off_normal_line_candidate(self) -> None:
		source_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
		source_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
		ext_xyz = torch.tensor(
			[
				[[-2.6301122, 0.9572288, 1.4183259], [-1.1567254, -2.6544838, -2.2087803]],
				[[0.2669331, -8.669413, 5.194581], [4.1991696, -4.5634775, -0.6482223]],
			],
			dtype=torch.float32,
		)
		ext_valid = torch.ones(2, 2, dtype=torch.bool)
		ext_quad_valid = torch.ones(1, 1, dtype=torch.bool)

		coords, accepted, stats = opt_loss_snap_surf._intersect_model_points_with_ext_surface(
			source_pos=source_pos,
			source_normals=source_normals,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			cfg=opt_loss_snap_surf.SnapSurfConfig(ray_residual=0.5),
		)

		self.assertEqual(stats["target_hit"], 1)
		self.assertEqual(stats["accepted"], 0)
		self.assertFalse(bool(accepted[0]))
		self.assertFalse(bool(torch.isfinite(coords[0]).all()))

	def test_debug_step_burst_grows_until_stalled(self) -> None:
		model_xyz = _plane_xyz(h=6, w=6, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=6, w=6, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "point_distance": 10.0, "grid_error": 0.25},
			seed_xyz=(2.5, 2.5, 0.0),
			active=True,
		)
		opt_loss_snap_surf.set_debug_step(100)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		stats = opt_loss_snap_surf.last_stats()

		self.assertGreater(stats["snaps_m2e"], 0.0)
		self.assertGreater(opt_loss_snap_surf._states[0].model_to_ext.count(), 12)

	def test_growth_accepts_continuous_target_quad_coordinate(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(4, 4), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 0, 0] = torch.tensor([0.5, 0.5])
		state.map[0, 0, 1] = torch.tensor([0.5, 1.5])
		state.valid[0, 0, 0] = True
		state.valid[0, 0, 1] = True
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		source[0, 0, 2] = torch.tensor([2.5, 0.5, 0.0])
		target = _plane_xyz(h=4, w=4, z=0.0)

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			target_xyz=target,
			target_valid=torch.ones(4, 4, dtype=torch.bool),
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=0.01, grid_error=0.01, search_ring=0),
		)

		self.assertTrue(bool(state.valid[0, 0, 2]))
		self.assertTrue(torch.allclose(state.map[0, 0, 2], torch.tensor([0.5, 2.5]), atol=1.0e-3))

	def test_invalid_prediction_falls_back_to_neighbor_correspondence(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 4), target_shape=(5, 5), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 1, 1] = torch.tensor([1.0, 1.0])
		state.map[0, 1, 2] = torch.tensor([1.0, 3.0])
		state.valid[0, 1, 1] = True
		state.valid[0, 1, 2] = True

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=_plane_xyz(h=3, w=4, z=0.0).unsqueeze(0),
			source_valid=torch.ones(1, 3, 4, dtype=torch.bool),
			target_xyz=_plane_xyz(h=5, w=5, z=0.0),
			target_valid=torch.ones(5, 5, dtype=torch.bool),
			normal_xyz=_normals_3d(1, 3, 4),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(affine_radius=1, search_ring=0),
		)

		self.assertTrue(bool(state.valid[0, 1, 3]))

	def test_growth_bruteforces_target_quads_outside_prediction_window(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(8, 8), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 1, 1] = torch.tensor([1.0, 1.0])
		state.valid[0, 1, 1] = True
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		source[0, 1, 2] = torch.tensor([5.4, 5.4, 0.0])

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			target_xyz=_plane_xyz(h=8, w=8, z=0.0),
			target_valid=torch.ones(8, 8, dtype=torch.bool),
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(affine_radius=1, search_ring=0),
		)

		self.assertTrue(bool(state.valid[0, 1, 2]))
		self.assertTrue(torch.allclose(state.map[0, 1, 2], torch.tensor([5.4, 5.4]), atol=1.0e-3))

	def test_no_orientation_gate_rejects_wrong_handed_growth(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(3, 3), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 0, 0] = torch.tensor([0.0, 0.0])
		state.map[0, 0, 1] = torch.tensor([0.0, 1.0])
		state.map[0, 1, 0] = torch.tensor([1.0, 0.0])
		state.valid[0, 0, 0] = True
		state.valid[0, 0, 1] = True
		state.valid[0, 1, 0] = True
		state.orientation_sign = -1
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		target = _plane_xyz(h=3, w=3, z=0.0)

		stats = opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			target_xyz=target,
			target_valid=torch.ones(3, 3, dtype=torch.bool),
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=0.01, grid_error=0.01, search_ring=0),
		)

		self.assertGreaterEqual(stats["grid"], 1)
		self.assertEqual(stats["ori"], stats["grid"])
		self.assertTrue(bool(state.valid[0, 1, 1]))

	def test_orientation_gate_does_not_reject_two_support_growth(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(3, 3), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 0, 1] = torch.tensor([0.0, 1.0])
		state.map[0, 1, 0] = torch.tensor([1.0, 0.0])
		state.valid[0, 0, 1] = True
		state.valid[0, 1, 0] = True
		state.orientation_sign = -1
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		target = _plane_xyz(h=3, w=3, z=0.0)

		stats = opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			target_xyz=target,
			target_valid=torch.ones(3, 3, dtype=torch.bool),
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=0.01, grid_error=0.01, search_ring=0),
		)

		self.assertGreaterEqual(stats["grid"], 1)
		self.assertEqual(stats["ori"], stats["grid"])
		self.assertTrue(bool(state.valid[0, 1, 1]))

	def test_low_distance_affine_inconsistent_candidate_is_accepted_without_gates(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(3, 3), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 0, 0] = torch.tensor([0.0, 0.0])
		state.map[0, 0, 1] = torch.tensor([0.0, 1.0])
		state.valid[0, 0, 0] = True
		state.valid[0, 0, 1] = True
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		target = _plane_xyz(h=3, w=3, z=0.0)
		target[0, 2] = torch.tensor([100.0, 100.0, 100.0])
		source[0, 0, 2] = target[1, 1]
		valid_source = torch.ones(1, 3, 3, dtype=torch.bool)
		valid_target = torch.ones(3, 3, dtype=torch.bool)

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=valid_source,
			target_xyz=target,
			target_valid=valid_target,
			normal_xyz=_normals_3d(1, 3, 3),
			normal_from_source=True,
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=1.0, grid_error=0.25, search_ring=2),
		)

		self.assertTrue(bool(state.valid[0, 0, 2]))

	def test_model_to_ext_direction_produces_model_gradients(self) -> None:
		model_xyz = (_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		res = _result(model_xyz, ext_xyz)

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=res)
		loss.backward()

		self.assertGreater(float(model_xyz.grad.abs().sum()), 0.0)
		stats = opt_loss_snap_surf.last_stats()
		self.assertGreater(stats["snaps_m2e"], 0.0)
		self.assertLessEqual(stats["snaps_m2e"], 1.0)

	def test_nonfinite_external_normals_do_not_poison_loss(self) -> None:
		model_xyz = (_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		ext_normals = _normals_2d(3, 3)
		ext_normals[0, 0] = torch.tensor([float("nan"), float("nan"), float("nan")])

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(
			res=_result(model_xyz, ext_xyz, ext_normals=ext_normals),
		)
		loss.backward()

		self.assertTrue(bool(torch.isfinite(loss).detach()))
		self.assertTrue(bool(torch.isfinite(model_xyz.grad).all().detach()))

	def test_valid_state_never_retains_nonfinite_correspondence(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 2, 2), target_shape=(2, 2), device=torch.device("cpu"), dtype=torch.float32)
		valid_b = torch.ones(1, 2, 2, dtype=torch.bool)
		map_b = torch.zeros(1, 2, 2, 2)
		map_b[0, 1, 1] = torch.tensor([float("nan"), 1.0])

		opt_loss_snap_surf._write_batched_state(state, valid_b, map_b)

		self.assertFalse(bool(state.valid[0, 1, 1]))
		self.assertTrue(bool(torch.isfinite(state.map[state.valid]).all()))

	def test_planar_smoke_distance_decreases_over_optimizer_steps(self) -> None:
		param = torch.nn.Parameter(_plane_xyz(h=3, w=3, z=4.0).unsqueeze(0))
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		optim = torch.optim.SGD([param], lr=0.4)
		values: list[float] = []

		for _ in range(6):
			optim.zero_grad()
			loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=_result(param, ext_xyz))
			values.append(float(loss.detach()))
			loss.backward()
			optim.step()

		self.assertLess(values[-1], values[0])
		self.assertLess(float(param[..., 2].abs().mean().detach()), 4.0)

	def test_snap_descent_direction_points_toward_proxy_plane(self) -> None:
		model_xyz = (_plane_xyz(h=3, w=3, z=2.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		loss.backward()
		stats = opt_loss_snap_surf.last_stats()

		self.assertGreater(stats["snaps_tow"], 0.0)
		self.assertGreater(float(model_xyz.grad[..., 2].mean().detach()), 0.0)

if __name__ == "__main__":
	unittest.main()
