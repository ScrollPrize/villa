from __future__ import annotations

import os
import sys
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

	def test_mapper_refuses_to_grow_from_one_inlier(self) -> None:
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
			target_normals=_normals_2d(3, 3),
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=10.0, grid_error=2.0),
		)

		self.assertEqual(state.count(), 1)
		self.assertFalse(bool(state.valid[0, 1, 2]))

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
		self.assertEqual(state.ext_to_model.count(), 4)
		self.assertTrue(torch.allclose(state.model_to_ext.map[0, 0, 0], torch.tensor([0.0, 0.0])))
		self.assertTrue(torch.allclose(state.ext_to_model.map[0, 0], torch.tensor([0.0, 0.0, 0.0])))
		self.assertAlmostEqual(opt_loss_snap_surf.last_stats()["snaps_sdist"], 0.0, places=6)
		self.assertAlmostEqual(opt_loss_snap_surf.last_stats()["snaps_sext"], 0.0, places=6)

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

	def test_outlier_correspondences_drop_on_next_loss_evaluation(self) -> None:
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

		self.assertEqual(state.model_to_ext.count(), 0)
		self.assertEqual(state.ext_to_model.count(), 0)

	def test_growth_advances_at_most_one_neighbor_layer_per_call(self) -> None:
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

		self.assertTrue(all(0 <= int(h) <= 3 and 0 <= int(w) <= 3 for h, w in valid.tolist()))
		self.assertFalse(bool(state.model_to_ext.valid[0, 0, 0]))
		self.assertFalse(bool(state.model_to_ext.valid[0, 3, 3]))

	def test_low_distance_affine_inconsistent_candidate_is_rejected(self) -> None:
		state = opt_loss_snap_surf._DirectionState(source_rank=3, target_rank=2)
		state.ensure(source_shape=(1, 3, 3), target_shape=(3, 3), device=torch.device("cpu"), dtype=torch.float32)
		state.map[0, 0, 0] = torch.tensor([0.0, 0.0])
		state.map[0, 0, 1] = torch.tensor([0.0, 1.0])
		state.valid[0, 0, 0] = True
		state.valid[0, 0, 1] = True
		source = _plane_xyz(h=3, w=3, z=0.0).unsqueeze(0)
		target = _plane_xyz(h=3, w=3, z=0.0)
		target[0, 2] = torch.tensor([100.0, 100.0, 0.0])
		source[0, 0, 2] = target[1, 1]
		valid_source = torch.ones(1, 3, 3, dtype=torch.bool)
		valid_target = torch.ones(3, 3, dtype=torch.bool)

		opt_loss_snap_surf._grow_direction(
			state,
			source_xyz=source,
			source_valid=valid_source,
			target_xyz=target,
			target_valid=valid_target,
			target_normals=_normals_2d(3, 3),
			cfg=opt_loss_snap_surf.SnapSurfConfig(point_distance=1.0, grid_error=0.25, search_ring=2),
		)

		self.assertFalse(bool(state.valid[0, 0, 2]))

	def test_both_directions_produce_model_gradients(self) -> None:
		model_xyz = (_plane_xyz(h=3, w=3, z=1.0).unsqueeze(0)).requires_grad_(True)
		ext_xyz = _plane_xyz(h=3, w=3, z=0.0)
		res = _result(model_xyz, ext_xyz)

		loss, _, _ = opt_loss_snap_surf.snap_surf_loss(res=res)
		loss.backward()

		self.assertGreater(float(model_xyz.grad.abs().sum()), 0.0)
		stats = opt_loss_snap_surf.last_stats()
		self.assertGreater(stats["snaps_m2e"], 0.0)
		self.assertGreater(stats["snaps_e2m"], 0.0)
		self.assertLessEqual(stats["snaps_m2e"], 1.0)
		self.assertLessEqual(stats["snaps_e2m"], 1.0)

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


if __name__ == "__main__":
	unittest.main()
