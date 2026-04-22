from __future__ import annotations

import torch

import model as fit_model
from grid_sample_3d_u8_diff import grid_sample_3d_u8_diff
from opt_loss_dir import _vertex_normals

_INNER_FACTOR = 0.25  # penalty reduction for points inside the predicted surface


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: two clamped L1 terms pushing mesh into the prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)      — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	pdt = res.data.pred_dt
	if pdt is None:
		raise RuntimeError("pred_dt loss requested but FitData3D.pred_dt is None")
	dev = res.xyz_lr.device
	offset = torch.tensor(res.data.origin_fullres, dtype=torch.float32, device=dev)
	inv_scale = torch.tensor([1.0 / s for s in res.data.spacing], dtype=torch.float32, device=dev)
	vol = pdt.squeeze(0)  # (1, Z, Y, X) uint8

	# Project gradients onto surface normal only (prevents tangential crimping)
	n = _vertex_normals(res.xyz_lr.detach())
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	xyz = xyz_normal + xyz_tangential.detach()

	sampled_raw = grid_sample_3d_u8_diff(vol, xyz, offset, inv_scale)  # (1, D, Hm, Wm)

	lm_out = (127.0 - sampled_raw).clamp(min=0)      # outside: 1–47, inside: 0 (no grad)
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = (lm_out + _INNER_FACTOR * lm_in).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	mask = res.mask_lr
	wsum = mask.sum()
	if float(wsum) > 0.0:
		loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	return loss, (lm,), (mask,)
