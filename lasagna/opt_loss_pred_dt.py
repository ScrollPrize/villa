from __future__ import annotations

import torch

import model as fit_model


def pred_dt_loss_map(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Loss map penalizing coarse mesh points far from predicted surfaces.

	Samples `pred_dt` (Euclidean distance to nearest predicted surface, in
	model-pixel voxels, 0-255) at coarse mesh positions `xyz_lr`.

	Returns (lm, mask) both (D, 1, Hm, Wm).
	"""
	pdt = res.data.pred_dt
	if pdt is None:
		raise RuntimeError("pred_dt loss requested but FitData3D.pred_dt is None (channel missing from preprocessed zarr?)")
	sampled = res.data.grid_sample_fullres(res.xyz_lr)
	# sampled.pred_dt is (1, 1, D, Hm, Wm) -> (D, 1, Hm, Wm)
	lm = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)
	mask = res.mask_lr
	return lm, mask


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = pred_dt_loss_map(res=res)
	wsum = mask.sum()
	if float(wsum) > 0.0:
		loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	return loss, (lm,), (mask,)
