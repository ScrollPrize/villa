from __future__ import annotations

import math
import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import model as fit_model
import opt_loss_step


def _step_result(xyz_lr: torch.Tensor, *, mesh_step: int = 1) -> fit_model.FitResult3D:
	d, h, w, _ = xyz_lr.shape
	return fit_model.FitResult3D(
		xyz_lr=xyz_lr,
		xyz_hr=None,
		data=None,
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.ones(d, 1, h, w, device=xyz_lr.device, dtype=xyz_lr.dtype),
		bias_lr=torch.zeros(d, 1, h, w, device=xyz_lr.device, dtype=xyz_lr.dtype),
		mask_hr=None,
		mask_lr=None,
		normals=None,
		xy_conn=None,
		mask_conn=None,
		sign_conn=None,
		params=fit_model.ModelParams3D(
			mesh_step=mesh_step,
			winding_step=mesh_step,
			subsample_mesh=1,
			subsample_winding=1,
			scaledown=1.0,
			z_step_eff=1,
			volume_extent=None,
			pyramid_d=False,
		),
	)


def _planar_grid(h: int, w: int, *, step: float = 1.0) -> torch.Tensor:
	y = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w) * step
	x = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w) * step
	z = torch.zeros_like(x)
	return torch.stack([x, y, z], dim=-1).unsqueeze(0).contiguous()


class StepLossTest(unittest.TestCase):
	def test_target_length_edges_produce_near_zero_loss(self) -> None:
		xyz = _planar_grid(4, 5, step=2.0)

		lm, mask = opt_loss_step.step_loss_maps(res=_step_result(xyz, mesh_step=2))
		loss, maps, masks = opt_loss_step.step_loss(res=_step_result(xyz, mesh_step=2))

		self.assertEqual(tuple(lm.shape), (1, 1, 3, 4))
		self.assertEqual(tuple(mask.shape), tuple(lm.shape))
		self.assertEqual(tuple(maps[0].shape), tuple(lm.shape))
		self.assertEqual(tuple(masks[0].shape), tuple(mask.shape))
		self.assertLess(float(lm.max()), 1.0e-10)
		self.assertLess(float(loss), 1.0e-10)

	def test_step_regularizers_target_scale_uniform_grid_are_near_zero(self) -> None:
		xyz = _planar_grid(4, 5, step=2.0)

		terms = opt_loss_step.step_regularizer_loss(res=_step_result(xyz, mesh_step=2))

		self.assertIn("smooth_step", terms)
		self.assertIn("avg_step", terms)
		self.assertLess(float(terms["smooth_step"][0]), 1.0e-10)
		self.assertLess(float(terms["avg_step"][0]), 1.0e-10)

	def test_wrong_scale_uniform_grid_has_avg_step_gradient_only(self) -> None:
		base = _planar_grid(4, 5, step=1.0)
		scale = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
		xyz = base * scale

		terms = opt_loss_step.step_regularizer_loss(res=_step_result(xyz, mesh_step=1))
		smooth_loss = terms["smooth_step"][0]
		avg_loss = terms["avg_step"][0]
		avg_loss.backward()

		self.assertLess(float(smooth_loss.detach()), 1.0e-10)
		self.assertGreater(float(avg_loss.detach()), 0.1)
		self.assertIsNotNone(scale.grad)
		self.assertGreater(float(scale.grad), 0.1)

	def test_nonuniform_grid_has_nonzero_smooth_step(self) -> None:
		xyz = _planar_grid(5, 5, step=1.0)
		xyz[0, 2, 2, 0] += 0.5

		terms = opt_loss_step.step_regularizer_loss(res=_step_result(xyz, mesh_step=1))

		self.assertGreater(float(terms["smooth_step"][0]), 1.0e-4)

	def test_smooth_step_targets_are_differentiable(self) -> None:
		xyz = _planar_grid(5, 5, step=1.0).requires_grad_()
		edge_data = opt_loss_step._step_regularizer_edge_data(xyz)

		smooth_targets, _avg_targets, _global_avg_step = opt_loss_step._step_regularizer_targets(
			edge_data,
			mesh_step=1.0,
		)
		target = smooth_targets["h"][0, 2, 1, 0]
		target.backward()

		self.assertTrue(smooth_targets["h"].requires_grad)
		self.assertIsNotNone(xyz.grad)
		self.assertGreater(float(xyz.grad.abs().sum()), 0.0)

	def test_avg_step_targets_are_detached(self) -> None:
		xyz = _planar_grid(5, 5, step=1.0).requires_grad_()
		edge_data = opt_loss_step._step_regularizer_edge_data(xyz)

		_smooth_targets, avg_targets, global_avg_step = opt_loss_step._step_regularizer_targets(
			edge_data,
			mesh_step=1.0,
		)

		self.assertTrue(global_avg_step.requires_grad)
		for target in avg_targets.values():
			self.assertFalse(target.requires_grad)

	def test_step_loss_analysis_reports_uncropped_grid_stats(self) -> None:
		xyz = _planar_grid(4, 5, step=2.0)[0]

		stats = opt_loss_step.step_loss_analysis(xyz, mesh_step=2.0)

		self.assertLess(stats["loss"], 1.0e-10)
		self.assertAlmostEqual(stats["target"], 2.0, delta=1.0e-6)
		self.assertAlmostEqual(stats["step_min"], 2.0, delta=1.0e-6)
		self.assertAlmostEqual(stats["step_avg"], 2.0, delta=1.0e-6)
		self.assertAlmostEqual(stats["step_med"], 2.0, delta=1.0e-6)
		self.assertAlmostEqual(stats["step_max"], 2.0, delta=1.0e-6)

	def test_short_h_and_w_edges_gradients_follow_local_edge_direction(self) -> None:
		diff_h = torch.tensor(
			[[[[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]]],
			dtype=torch.float32,
			requires_grad=True,
		)
		dir_h, valid_h = opt_loss_step._h_edge_directions(diff_h)
		pen_h = opt_loss_step._directional_step_penalty(diff_h[:, :, :-1, :], 1.0, dir_h, valid_h)
		pen_h[0, 0, 1, 0].backward()
		h_grad = diff_h.grad[0, 0, 1]
		self.assertAlmostEqual(float(h_grad[0]), 0.0, delta=1.0e-7)
		self.assertLess(float(h_grad[1]), -0.9)
		self.assertAlmostEqual(float(h_grad[2]), 0.0, delta=1.0e-7)

		diff_w = torch.tensor(
			[[[[1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]],
			dtype=torch.float32,
			requires_grad=True,
		)
		dir_w, valid_w = opt_loss_step._w_edge_directions(diff_w)
		pen_w = opt_loss_step._directional_step_penalty(diff_w[:, :-1, :, :], 1.0, dir_w, valid_w)
		pen_w[0, 1, 0, 0].backward()
		w_grad = diff_w.grad[0, 1, 0]
		self.assertLess(float(w_grad[0]), -0.9)
		self.assertAlmostEqual(float(w_grad[1]), 0.0, delta=1.0e-7)
		self.assertAlmostEqual(float(w_grad[2]), 0.0, delta=1.0e-7)

	def test_short_edge_normal_component_does_not_create_normal_expansion_gradient(self) -> None:
		diff = torch.tensor([[[[0.5, 0.0, 0.4]]]], dtype=torch.float32, requires_grad=True)
		direction = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
		target = torch.ones(1, 1, 1, 1, dtype=torch.float32)
		valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)

		pen = opt_loss_step._directional_step_penalty(diff, target, direction, valid)
		pen.sum().backward()

		grad = diff.grad[0, 0, 0]
		self.assertLess(float(grad[0]), -0.9)
		self.assertAlmostEqual(float(grad[2]), 0.0, delta=1.0e-7)

	def test_long_edge_uses_full_3d_contraction(self) -> None:
		diff = torch.tensor([[[[1.0, 0.0, 1.0]]]], dtype=torch.float32, requires_grad=True)
		direction = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
		target = torch.ones(1, 1, 1, 1, dtype=torch.float32)
		valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)

		pen = opt_loss_step._directional_step_penalty(diff, target, direction, valid)
		pen.sum().backward()

		grad = diff.grad[0, 0, 0]
		self.assertAlmostEqual(float(grad[0]), 2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(2.0), delta=1.0e-6)
		self.assertGreater(float(grad[2]), 0.5)

	def test_degenerate_local_direction_falls_back_and_remains_finite(self) -> None:
		xyz = torch.zeros(1, 3, 3, 3, dtype=torch.float32, requires_grad=True)

		loss, _maps, _masks = opt_loss_step.step_loss(res=_step_result(xyz, mesh_step=1))
		loss.backward()

		self.assertTrue(torch.isfinite(loss).item())
		self.assertTrue(torch.isfinite(xyz.grad).all().item())


if __name__ == "__main__":
	unittest.main()
