from __future__ import annotations

import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_data
import model as fit_model
import opt_loss_atlas_line


def _fit_data(lines: fit_data.AtlasLines3D) -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=None,
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		atlas_lines=lines,
		_vol_size=(1, 1, 1),
	)


def _result(xyz: torch.Tensor, lines: fit_data.AtlasLines3D) -> fit_model.FitResult3D:
	D, H, W, _ = xyz.shape
	return fit_model.FitResult3D(
		xyz_lr=xyz,
		xyz_hr=None,
		data=_fit_data(lines),
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.ones(D, 1, H, W, dtype=xyz.dtype),
		bias_lr=torch.zeros(D, 1, H, W, dtype=xyz.dtype),
		mask_hr=None,
		mask_lr=None,
		normals=torch.zeros_like(xyz),
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
	)


def _plane_grid() -> torch.Tensor:
	H, W = 3, 4
	y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
	x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
	z = torch.zeros_like(x)
	return torch.stack([x, y, z], dim=-1).unsqueeze(0).contiguous()


class AtlasLineLossTest(unittest.TestCase):
	def test_atlas_line_updates_at_most_one_neighboring_quad_and_clamps(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[2.2, 1.2, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.2], dtype=torch.float32),
			model_w=torch.tensor([1.2], dtype=torch.float32),
			object_ids=("fiber",),
			source_indices=(0,),
		)

		loss, maps, masks = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))
		loss.backward()

		self.assertAlmostEqual(float(loss.detach()), 1.0, delta=1.0e-5)
		self.assertEqual(tuple(maps[0].shape), (1, 1, 1, 1))
		self.assertEqual(float(masks[0].sum()), 1.0)
		self.assertEqual(int(opt_loss_atlas_line._cols[0, 0]), 2)
		self.assertAlmostEqual(float(opt_loss_atlas_line._frac_w[0, 0]), 0.2, delta=1.0e-5)
		self.assertTrue(torch.isfinite(xyz.grad).all().item())

	def test_atlas_line_skips_invalid_samples(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[float("nan"), 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		loss, _maps, masks = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))

		self.assertEqual(float(loss.detach()), 0.0)
		self.assertEqual(float(masks[0].sum()), 0.0)


if __name__ == "__main__":
	unittest.main()
