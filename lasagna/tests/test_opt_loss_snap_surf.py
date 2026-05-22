from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import model as fit_model
import opt_loss_snap_surf


def _plane_xyz(*, h: int, w: int, z: float, offset_h: float = 0.0, offset_w: float = 0.0) -> torch.Tensor:
	hh = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w) + float(offset_h)
	ww = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w) + float(offset_w)
	zz = torch.full((h, w), float(z), dtype=torch.float32)
	return torch.stack([ww, hh, zz], dim=-1)


def _normals_2d(h: int, w: int) -> torch.Tensor:
	n = torch.zeros(h, w, 3, dtype=torch.float32)
	n[..., 2] = 1.0
	return n


def _normals_3d(d: int, h: int, w: int) -> torch.Tensor:
	n = torch.zeros(d, h, w, 3, dtype=torch.float32)
	n[..., 2] = 1.0
	return n


def _result(
	xyz_lr: torch.Tensor,
	ext_xyz: torch.Tensor,
	*,
	normals: torch.Tensor | None = None,
	ext_normals: torch.Tensor | None = None,
) -> fit_model.FitResult3D:
	d, h, w, _ = xyz_lr.shape
	eh, ew, _ = ext_xyz.shape
	if normals is None:
		normals = _normals_3d(d, h, w)
	if ext_normals is None:
		ext_normals = _normals_2d(eh, ew)
	ext_valid = torch.isfinite(ext_xyz).all(dim=-1)
	ext_quad_valid = (
		ext_valid[:-1, :-1] &
		ext_valid[1:, :-1] &
		ext_valid[:-1, 1:] &
		ext_valid[1:, 1:]
	)
	return fit_model.FitResult3D(
		xyz_lr=xyz_lr,
		xyz_hr=None,
		data=SimpleNamespace(),
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.ones(d, 1, h, w),
		bias_lr=torch.zeros(d, 1, h, w),
		mask_hr=None,
		mask_lr=None,
		normals=normals,
		xy_conn=None,
		mask_conn=None,
		sign_conn=None,
		params=fit_model.ModelParams3D(
			mesh_step=1,
			winding_step=1,
			subsample_mesh=1,
			subsample_winding=1,
			scaledown=1.0,
			z_step_eff=1,
			volume_extent=None,
			pyramid_d=False,
		),
		gt_normal_lr=None,
		ext_conn=None,
		ext_surfaces=[(ext_xyz.detach(), ext_valid, ext_normals.detach(), ext_quad_valid)],
	)


class SnapSurfMapperTest(unittest.TestCase):
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

	def test_invalid_correspondence_coordinates_do_not_poison_loss(self) -> None:
		model_xyz = (_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]
		state.model_to_ext.map[0, 1, 1] = torch.tensor([float("nan"), float("nan")])
		state.model_to_ext.valid[0, 1, 1] = True
		state.ext_to_model.map[1, 1] = torch.tensor([float("nan"), float("nan"), float("nan")])
		state.ext_to_model.valid[1, 1] = True

		l_m2e, _, _, _, _ = opt_loss_snap_surf._direction_loss_model_to_ext(
			state.model_to_ext,
			model_xyz=model_xyz,
			ext_xyz=ext_xyz,
			ext_valid=torch.ones(3, 3, dtype=torch.bool),
			ext_normals=_normals_2d(3, 3),
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)
		l_e2m, _, _ = opt_loss_snap_surf._direction_loss_ext_to_model(
			state.ext_to_model,
			model_xyz=model_xyz,
			model_valid=torch.ones(1, 3, 3, dtype=torch.bool),
			ext_normals=_normals_2d(3, 3),
			ext_xyz=ext_xyz,
			cfg=opt_loss_snap_surf.SnapSurfConfig(),
		)
		loss = l_m2e + l_e2m
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

	def test_map_init_config_parse_and_validation(self) -> None:
		opt_loss_snap_surf.configure_snap_surf(
			cfg={
				"map_init": {
					"enabled": True,
					"subdiv": 2,
					"iters": 3,
					"scale_levels": 4,
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
				}
			},
			seed_xyz=(1.0, 1.0, 0.0),
			active=True,
		)

		self.assertTrue(opt_loss_snap_surf._cfg.map_init.enabled)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.subdiv, 2)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.iters, 3)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.scale_levels, 4)
		self.assertTrue(opt_loss_snap_surf._cfg.map_init.dense_opt)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.dense_reg_radius, 5)
		self.assertAlmostEqual(opt_loss_snap_surf._cfg.map_init.w_dense_prior, 0.25)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.repair_max_blocks, 2)
		self.assertAlmostEqual(opt_loss_snap_surf._cfg.map_init.repair_lr_mult, 0.5)
		self.assertAlmostEqual(opt_loss_snap_surf._cfg.map_init.repair_w_jac_mult, 8.0)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.edge_init_radius, 3)
		self.assertEqual(opt_loss_snap_surf._cfg.map_init.progress_interval, 100)
		self.assertAlmostEqual(opt_loss_snap_surf._cfg.map_init.w_metric_smooth, 0.12)
		self.assertAlmostEqual(opt_loss_snap_surf._cfg.map_init.w_area_smooth, 0.03)
		with self.assertRaises(ValueError):
			opt_loss_snap_surf.configure_snap_surf(
				cfg={"map_init": {"unknown": 1}},
				seed_xyz=(1.0, 1.0, 0.0),
				active=True,
			)
		for key in ("w_metric_smooth", "w_area_smooth"):
			with self.assertRaises(ValueError):
				opt_loss_snap_surf.configure_snap_surf(
					cfg={"map_init": {key: -0.1}},
					seed_xyz=(1.0, 1.0, 0.0),
					active=True,
				)

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
			normal_sign=1,
			orientation_sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=3),
			),
		)

		self.assertEqual(float(terms["active"].detach()), 1.0)
		self.assertEqual(float(terms["samples"].detach()), 9.0)

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
			model_valid=model_valid,
			model_normals=_normals_3d(1, H, W),
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(subdiv=2, edge_init_radius=2),
			),
		)

		self.assertEqual(added, 0)
		self.assertFalse(bool(state.map_init.active_quad[1, 0]))
		self.assertFalse(torch.isfinite(state.map_init.uv[2]).any())

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

	def test_map_init_angle_distance_multiplier_at_ninety_degrees(self) -> None:
		cfg = opt_loss_snap_surf.SnapSurfMapInitConfig(angle_dist_mult=9.0)
		got = opt_loss_snap_surf._map_init_distance_multiplier(
			torch.tensor([1.0]),
			torch.tensor([0.0]),
			cfg,
		)

		self.assertAlmostEqual(float(got[0]), 10.0, places=5)

	def test_map_init_jacobian_penalty_catches_flipped_cells(self) -> None:
		active = torch.ones(1, 1, dtype=torch.bool)
		uv_ok = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
		uv_flip = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [-1.0, 1.0]]])

		ok_pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv_ok,
			active,
			orientation_sign=1,
			jac_margin=0.05,
		)
		flip_pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv_flip,
			active,
			orientation_sign=1,
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
			orientation_sign=1,
			jac_margin=0.05,
		)
		compressed_terms = opt_loss_snap_surf._map_init_inverse_regularization_terms(
			uv_compressed,
			active,
			orientation_sign=1,
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
			orientation_sign=1,
			jac_margin=0.05,
		)
		reverse_terms = opt_loss_snap_surf._map_init_inverse_regularization_terms(
			uv_expanded,
			active,
			orientation_sign=1,
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
			normal_sign=1,
			orientation_sign=1,
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
			orientation_sign=1,
			jac_margin=0.05,
		))

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
			normal_sign=1,
			orientation_sign=1,
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
			normal_sign=1,
			orientation_sign=1,
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
			normal_sign=1,
			orientation_sign=1,
			cfg=opt_loss_snap_surf.SnapSurfConfig(
				map_init=opt_loss_snap_surf.SnapSurfMapInitConfig(
					dense_opt=True,
					dense_reg_radius=2,
				),
			),
		)

		self.assertEqual(float(terms["reg"].detach()), 35.0)

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
		active = torch.ones(3, 3, dtype=torch.bool)
		sparse = opt_loss_snap_surf._map_init_sparse_quad_mask(active, min_neighbors=4)

		self.assertTrue(torch.equal(sparse, active))

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

	def test_map_init_jacobian_penalty_empty_cells_ignores_inactive_nans(self) -> None:
		active_quad = torch.zeros(2, 2, dtype=torch.bool)
		uv = torch.full((3, 3, 2), float("nan"))
		uv[1, 1] = torch.tensor([1.0, 1.0])

		pen = opt_loss_snap_surf._map_init_jacobian_penalty(
			uv,
			active_quad,
			orientation_sign=1,
			jac_margin=0.05,
		)

		self.assertTrue(torch.isfinite(pen))
		self.assertAlmostEqual(float(pen.detach()), 0.0, places=6)

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


if __name__ == "__main__":
	unittest.main()
