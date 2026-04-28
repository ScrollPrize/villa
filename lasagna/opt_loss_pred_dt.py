from __future__ import annotations

import torch

import model as fit_model
from opt_loss_dir import _vertex_normals

_INNER_FACTOR = 0.25  # penalty reduction for points inside the predicted surface


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: two clamped L1 terms pushing mesh into the prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)      — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	# Project gradients onto surface normal only (prevents tangential crimping)
	n = _vertex_normals(res.xyz_lr.detach())
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	xyz = xyz_normal + xyz_tangential.detach()

	# Sample pred_dt using common sampling with per-channel spacing and diff gradients
	sampled = res.data.grid_sample_fullres(xyz, diff=True)
	sampled_raw = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	lm_out = (127.0 - sampled_raw).clamp(min=0)      # outside: 1–47, inside: 0 (no grad)
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = lm_out + _INNER_FACTOR * lm_in

	mask = res.mask_lr
	wsum = mask.sum()
	if float(wsum) > 0.0:
		loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	return loss, (lm,), (mask,)
