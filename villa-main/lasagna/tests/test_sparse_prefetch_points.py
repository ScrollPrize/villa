from __future__ import annotations

import os
from types import SimpleNamespace
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import model as fit_model
import opt_loss_winding_density


class _CaptureData:
	def __init__(self) -> None:
		self.samples: list[torch.Tensor] = []

	def grid_sample_fullres(self, xyz: torch.Tensor, *, channels: set[str] | None = None):
		self.samples.append(xyz.detach().clone())
		shape = (1, 1, *xyz.shape[:-1])
		return SimpleNamespace(grad_mag=torch.ones(shape, device=xyz.device, dtype=xyz.dtype))


def _params(*, subsample_mesh: int = 2) -> fit_model.ModelParams3D:
	return fit_model.ModelParams3D(
		mesh_step=1,
		winding_step=1,
		subsample_mesh=subsample_mesh,
		subsample_winding=1,
		scaledown=1.0,
		z_step_eff=1,
		volume_extent=None,
		pyramid_d=False,
	)


class SparsePrefetchPointTests(unittest.TestCase):
	def test_mesh_conn_prefetch_points_match_xyz_conn_samples(self) -> None:
		device = torch.device("cpu")
		mdl = fit_model.Model3D(
			device=device,
			depth=2,
			mesh_h=3,
			mesh_w=3,
			mesh_step=1,
			winding_step=1,
			init_mode="arc",
			pyramid_d=False,
		)
		y = torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(2, 3, 3)
		x = torch.arange(3, dtype=torch.float32).view(1, 1, 3).expand(2, 3, 3)
		z = torch.arange(2, dtype=torch.float32).view(2, 1, 1).expand(2, 3, 3)
		xyz_lr = torch.stack([x, y, z], dim=-1).contiguous()

		prefetch_points = mdl.mesh_conn_prefetch_points(xyz_lr)
		capture = _CaptureData()
		mdl._xyz_conn(xyz_lr, capture)

		self.assertEqual(len(prefetch_points), 3)
		self.assertEqual(len(capture.samples), 3)
		for prefetched, sampled in zip(prefetch_points, capture.samples):
			torch.testing.assert_close(prefetched, sampled)

	def test_winding_density_prefetch_points_match_loss_strips(self) -> None:
		D, Hm, Wm = 2, 2, 2
		He, We = 3, 4
		xy_conn = torch.zeros(D, Hm, Wm, 3, 3)
		xy_conn[:, :, :, 0, 0] = 0.0
		xy_conn[:, :, :, 0, 1] = 3.0
		xy_conn[:, :, :, 0, 2] = 6.0
		res = fit_model.FitResult3D(
			xyz_lr=torch.zeros(D, Hm, Wm, 3),
			xyz_hr=torch.zeros(D, He, We, 3),
			data=None,
			data_s=None,
			data_lr=None,
			target_plain=None,
			target_mod=None,
			amp_lr=torch.ones(D, 1, Hm, Wm),
			bias_lr=torch.zeros(D, 1, Hm, Wm),
			mask_hr=None,
			mask_lr=None,
			normals=None,
			xy_conn=xy_conn,
			mask_conn=torch.ones(D, 1, Hm, Wm, 3),
			sign_conn=torch.ones(D, 1, Hm, Wm, 2),
			params=_params(subsample_mesh=2),
		)

		batches = list(opt_loss_winding_density.winding_density_prefetch_grad_mag_batches_for_result(res=res))

		self.assertEqual(len(batches), 2)
		self.assertEqual(tuple(batches[0].shape), (D, He, We * 3, 3))
		self.assertEqual(tuple(batches[1].shape), (D, He, We * 3, 3))
		expected_prev_x = torch.tensor([0.0, 1.5, 3.0] * We)
		expected_next_x = torch.tensor([3.0, 4.5, 6.0] * We)
		torch.testing.assert_close(batches[0][0, 0, :, 0], expected_prev_x)
		torch.testing.assert_close(batches[1][0, 0, :, 0], expected_next_x)


if __name__ == "__main__":
	unittest.main()
