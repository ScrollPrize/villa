from __future__ import annotations

import os
import sys
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
