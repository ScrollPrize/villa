from __future__ import annotations

import torch

import model as fit_model


def data_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for MSE between sampled cosine and target_mod."""
	pred = res.data_s.cos
	tgt = res.target_mod
	d = pred - tgt
	lm = d * d
	mask = res.mask_hr
	return lm * mask, mask


def data_grad_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for gradient matching between sampled cosine and target_mod.

	L = 0.5*(||dx pred - dx tgt||^2 + ||dy pred - dy tgt||^2)

	- pred: res.data_s.cos (sampled at xy_hr)
	- tgt:  res.target_plain (at xy_hr)
	"""
	pred = res.data_s.cos
	tgt = res.target_mod

	gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
	gx_tgt = tgt[:, :, :, 1:] - tgt[:, :, :, :-1]
	diff_gx = gx_pred - gx_tgt

	gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
	gy_tgt = tgt[:, :, 1:, :] - tgt[:, :, :-1, :]
	diff_gy = gy_pred - gy_tgt

	mask_hr = res.mask_hr
	wx = torch.minimum(mask_hr[:, :, :, 1:], mask_hr[:, :, :, :-1])
	wy = torch.minimum(mask_hr[:, :, 1:, :], mask_hr[:, :, :-1, :])

	lx = diff_gx * diff_gx
	ly = diff_gy * diff_gy

	wsum_x = wx.sum()
	wsum_y = wy.sum()
	loss_x = (lx * wx).sum() / wsum_x if float(wsum_x.detach().cpu()) > 0.0 else lx.mean()
	loss_y = (ly * wy).sum() / wsum_y if float(wsum_y.detach().cpu()) > 0.0 else ly.mean()
	base = 0.5 * (loss_x + loss_y)
	lm = base.view(int(pred.shape[0]), 1, 1, 1)
	mask = torch.ones_like(lm)
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def data_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = data_loss_map(res=res)
	return _masked_mean(lm, mask)


def data_grad_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = data_grad_loss_map(res=res)
	return _masked_mean(lm, mask)
