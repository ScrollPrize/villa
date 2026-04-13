from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model
from grid_sample_3d_u8_diff import grid_sample_3d_u8_diff


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def _pool_w3(*, lm: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Max-pool loss map along winding (W) axis with kernel 3, preserving shape."""
	lm_p = F.max_pool2d(lm, kernel_size=(1, 3), stride=1, padding=(0, 1))
	mask_p = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
	return lm_p, mask_p


def _sample_cos_diff(res: fit_model.FitResult3D) -> torch.Tensor:
	"""Differentiably sample cos channel at xyz_hr. Returns (D, 1, He, We)."""
	dev = res.xyz_hr.device
	offset = torch.tensor(res.data.origin_fullres, dtype=torch.float32, device=dev)
	inv_scale = torch.tensor([1.0 / s for s in res.data.spacing], dtype=torch.float32, device=dev)
	vol = res.data.cos.squeeze(0)  # (1, Z, Y, X) uint8
	sampled_raw = grid_sample_3d_u8_diff(vol, res.xyz_hr, offset, inv_scale)  # (1, D, He, We)
	return (sampled_raw / 255.0).permute(1, 0, 2, 3)  # (D, 1, He, We)


def data_loss_map(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for MSE between sampled cosine and target_mod."""
	pred = _sample_cos_diff(res)
	tgt = res.target_mod
	d = pred - tgt
	lm = d * d
	mask = res.mask_hr
	lm, mask = _pool_w3(lm=lm, mask=mask)
	return lm, mask


def data_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = data_loss_map(res=res)
	return _masked_mean(lm, mask), (lm,), (mask,)


def data_plain_loss_map(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for MSE between sampled cosine and target_plain."""
	pred = _sample_cos_diff(res)
	tgt = res.target_plain
	d = pred - tgt
	lm = d * d
	mask = res.mask_hr
	lm, mask = _pool_w3(lm=lm, mask=mask)
	return lm, mask


def data_plain_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = data_plain_loss_map(res=res)
	return _masked_mean(lm, mask), (lm,), (mask,)
