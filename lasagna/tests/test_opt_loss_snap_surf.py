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

	def test_seed_radius_limits_flood_to_seed_neighborhood(self) -> None:
		model_xyz = _plane_xyz(h=8, w=8, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=8, w=8, z=0.0)
		opt_loss_snap_surf.configure_snap_surf(
			cfg={"init_distance": 10.0, "seed_radius": 1},
			seed_xyz=(3.5, 3.5, 0.0),
			active=True,
		)

		opt_loss_snap_surf.snap_surf_loss(res=_result(model_xyz, ext_xyz))
		state = opt_loss_snap_surf._states[0]

		self.assertLessEqual(state.model_to_ext.count(), 16)
		self.assertLessEqual(state.ext_to_model.count(), 16)

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

		self.assertGreater(stats["snaps_new"], 0)
		self.assertEqual(stats["snaps_grid"], stats["snaps_ori"])
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
		self.assertEqual(stats["snaps_e2m"], 0.0)
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


if __name__ == "__main__":
	unittest.main()
