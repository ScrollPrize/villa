from __future__ import annotations

import torch

import model as fit_model
from grid_sample_3d_u8_diff import grid_sample_3d_u8_diff


def pred_dt_loss_map(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Loss map penalizing coarse mesh points far from predicted surfaces.

	Samples `pred_dt` (Euclidean distance to nearest predicted surface, in
	model-pixel voxels, 0-255) at coarse mesh positions `xyz_lr`.
	Uses differentiable grid_sample so gradients flow back to mesh positions.

	Returns (lm, mask) both (D, 1, Hm, Wm).
	"""
	pdt = res.data.pred_dt
	if pdt is None:
		raise RuntimeError("pred_dt loss requested but FitData3D.pred_dt is None (channel missing from preprocessed zarr?)")
	dev = res.xyz_lr.device
	offset = torch.tensor(res.data.origin_fullres, dtype=torch.float32, device=dev)
	inv_scale = torch.tensor([1.0 / s for s in res.data.spacing], dtype=torch.float32, device=dev)
	vol = pdt.squeeze(0)  # (1, Z, Y, X) uint8
	sampled_raw = grid_sample_3d_u8_diff(vol, res.xyz_lr, offset, inv_scale)  # (1, D, Hm, Wm) float32
	# decode pred_dt: sqrt of raw interpolated uint8 value
	lm = sampled_raw.clamp(min=0.01).sqrt().permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)
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
