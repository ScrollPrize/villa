from __future__ import annotations

import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import opt_loss_station
import model as fit_model


class _StationData:
	def grid_sample_fullres(self, query, *, channels, diff=False):
		shape = query.shape[:-1]
		nx = torch.zeros(shape, device=query.device, dtype=query.dtype)
		ny = torch.zeros(shape, device=query.device, dtype=query.dtype)
		return SimpleNamespace(nx=nx, ny=ny)

	def has_channel(self, name: str) -> bool:
		return False


def _station_result(xyz_lr: torch.Tensor, *, mesh_step: int = 10) -> fit_model.FitResult3D:
	d, h, w, _ = xyz_lr.shape
	mask = torch.ones(d, 1, h, w, dtype=torch.float32)
	return fit_model.FitResult3D(
		xyz_lr=xyz_lr,
		xyz_hr=None,
		data=_StationData(),
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.ones(d, 1, h, w),
		bias_lr=torch.zeros(d, 1, h, w),
		mask_hr=None,
		mask_lr=mask,
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


class StationLossHuberTest(unittest.TestCase):
	def test_huberized_mse_matches_mse_inside_delta_and_linear_outside(self) -> None:
		residual = torch.tensor([0.5, -1.0, 3.0], dtype=torch.float32)

		got = opt_loss_station._huberized_mse(residual, delta=2.0)

		expected = torch.tensor((0.25 + 1.0 + (2.0 * 2.0 * 3.0 - 2.0 * 2.0)) / 3.0)
		self.assertAlmostEqual(float(got), float(expected), delta=1.0e-6)

	def test_initial_station_diagnostic_prints_once_with_raw_distance(self) -> None:
		x = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3) * 10.0
		y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3) * 10.0
		z = torch.full((3, 3), 10.0, dtype=torch.float32)
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		seed = torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32)

		buf = StringIO()
		try:
			with redirect_stdout(buf):
				opt_loss_station.set_seed(seed, _StationData(), Hm=3, Wm=3, D=1)
				opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
				opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
		finally:
			opt_loss_station.reset()

		out = buf.getvalue()
		self.assertIn("[station] initial:", out)
		self.assertIn("anchor=(10.000,10.000,10.000)", out)
		self.assertIn("seed_dist=10.000vx", out)
		self.assertIn("seed_dist_norm=1.000", out)
		self.assertEqual(out.count("[station] initial:"), 1)

	def test_normal_loss_is_scalar_huber_after_mesh_step_normalization(self) -> None:
		x = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3) * 10.0
		y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3) * 10.0
		z = torch.full((3, 3), 20.0, dtype=torch.float32)
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		seed = torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32)

		try:
			opt_loss_station.set_seed(seed, _StationData(), Hm=3, Wm=3, D=1)
			losses = opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
		finally:
			opt_loss_station.reset()

		# normal offset is 20vx, normalized to 2 mesh steps, Huber(delta=1) = 3,
		# then averaged with the smooth local station weights over the 3x3 mesh.
		weights = opt_loss_station._station_normal_weights(
			D=1, Hm=3, Wm=3, h_center=1.0, w_center=1.0,
			device=xyz.device, dtype=xyz.dtype)
		self.assertAlmostEqual(
			float(losses["station_n"][0].detach()),
			float(3.0 * weights.mean()),
			delta=1.0e-6,
		)
		self.assertLess(float(losses["station_t"][0].detach()), 1.0e-6)

	def test_normal_loss_uses_closest_point_model_normal_for_push(self) -> None:
		x = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3) * 10.0
		y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3) * 10.0
		z = 20.0 + 0.5 * x
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		model_normal = torch.tensor([0.5, 0.0, -1.0], dtype=torch.float32)
		model_normal = model_normal / model_normal.norm()
		seed = torch.tensor([10.0, 10.0, 25.0], dtype=torch.float32) - 20.0 * model_normal

		buf = StringIO()
		try:
			with redirect_stdout(buf):
				opt_loss_station.set_seed(seed, _StationData(), Hm=3, Wm=3, D=1)
				losses = opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
				losses["station_n"][0].backward()
		finally:
			opt_loss_station.reset()

		out = buf.getvalue()
		self.assertIn("normal_source=model_closest", out)
		self.assertIn("anchor=(10.000,10.000,25.000)", out)
		self.assertIn("normal_offset=20.000vx", out)
		# The closest point is on a tilted model surface. The 20vx model-normal
		# offset normalizes to 2 mesh steps: Huber(delta=1) = 3, then averaged
		# with the smooth local station weights over the 3x3 mesh.
		weights = opt_loss_station._station_normal_weights(
			D=1, Hm=3, Wm=3, h_center=1.0, w_center=1.0,
			device=xyz.device, dtype=xyz.dtype)
		self.assertAlmostEqual(
			float(losses["station_n"][0].detach()),
			float(3.0 * weights.mean()),
			delta=1.0e-6,
		)
		self.assertLess(float(losses["station_t"][0].detach()), 1.0e-6)
		center_grad = xyz.grad[0, 1, 1]
		self.assertGreater(abs(float(center_grad[0])), 1.0e-4)
		self.assertGreater(abs(float(center_grad[2])), 1.0e-4)

	def test_normal_loss_uses_base_point_normal_globally(self) -> None:
		H = W = 5
		x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) * 10.0
		y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) * 10.0
		z = 20.0 + 0.04 * (x - 20.0).square()
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		seed = torch.tensor([20.0, 20.0, 0.0], dtype=torch.float32)

		buf = StringIO()
		try:
			with redirect_stdout(buf):
				opt_loss_station.set_seed(seed, _StationData(), Hm=H, Wm=W, D=1)
				losses = opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
				losses["station_n"][0].backward()
		finally:
			opt_loss_station.reset()

		out = buf.getvalue()
		self.assertIn("normal_source=model_closest", out)
		self.assertIn("anchor=(20.000,20.000,20.000)", out)
		# The right edge's local model normal points toward +X, while the
		# selected anchor normal points toward -X on this curved surface. The
		# station_n push/pull uses the base point normal globally, so the right
		# edge gets the anchor-normal X sign.
		right_edge_grad = xyz.grad[0, 2, 4]
		self.assertGreater(float(right_edge_grad[0]), 1.0e-4)
		self.assertGreater(float(right_edge_grad[2]), 1.0e-4)

	def test_normal_loss_uses_wide_gaussian_weight_falloff(self) -> None:
		H = W = 21
		x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) * 10.0
		y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) * 10.0
		z = torch.full((H, W), 20.0, dtype=torch.float32)
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		seed = torch.tensor([100.0, 100.0, 0.0], dtype=torch.float32)

		try:
			opt_loss_station.set_seed(seed, _StationData(), Hm=H, Wm=W, D=1)
			losses = opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
			losses["station_n"][0].backward()
		finally:
			opt_loss_station.reset()

		center_grad = abs(float(xyz.grad[0, 10, 10, 2]))
		mid_grad = abs(float(xyz.grad[0, 10, 14, 2]))
		far_grad = abs(float(xyz.grad[0, 0, 0, 2]))
		self.assertGreater(center_grad, 1.0e-6)
		self.assertGreater(mid_grad / center_grad, 0.95)
		self.assertLess(far_grad / center_grad, mid_grad / center_grad)

	def test_tangent_loss_is_vector_huber_after_mesh_step_normalization(self) -> None:
		x = torch.arange(3, dtype=torch.float32).view(1, 3).expand(3, 3) * 10.0
		y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 3) * 10.0
		z = torch.zeros((3, 3), dtype=torch.float32)
		xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).requires_grad_(True)
		seed = torch.tensor([20.0, 10.0, 0.0], dtype=torch.float32)

		try:
			opt_loss_station.set_seed(seed, _StationData(), Hm=3, Wm=3, D=1)
			losses = opt_loss_station.station_loss(res=_station_result(xyz, mesh_step=10))
		finally:
			opt_loss_station.reset()

		self.assertLess(float(losses["station_n"][0].detach()), 1.0e-6)
		# tangent correction is one 10vx mesh step, so Huber(delta=1) is 1.
		self.assertAlmostEqual(float(losses["station_t"][0].detach()), 1.0, delta=1.0e-6)


if __name__ == "__main__":
	unittest.main()
