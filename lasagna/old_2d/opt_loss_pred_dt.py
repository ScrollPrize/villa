from __future__ import annotations

import torch

import model as fit_model


def pred_dt_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Loss map penalizing coarse mesh points far from predicted surfaces.

	Samples `pred_dt` (Euclidean distance to nearest predicted surface, in
	model-pixel voxels, 0–255) at coarse mesh positions `xy_lr`.

	Returns (lm, mask) both (N, 1, Hm, Wm).
	"""
	pdt = res.data.pred_dt
	if pdt is None:
		raise RuntimeError("pred_dt loss requested but FitData.pred_dt is None (channel missing from preprocessed zarr?)")
	sampled = res.data.grid_sample_px(xy_px=res.xy_lr)
	# pdt is (N, 1, Hm, Wm) — raw distance values
	mask = res.mask_lr
	return sampled.pred_dt, mask


def pred_dt_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = pred_dt_loss_map(res=res)
	wsum = mask.sum()
	if float(wsum) > 0.0:
		loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	return loss, (lm,), (mask,)
