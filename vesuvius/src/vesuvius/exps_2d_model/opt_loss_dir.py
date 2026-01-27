from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def direction_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	# Vertical-direction match (existing UNet encoding).
	diff0_v = res.dir0_pred - res.data_s.dir0
	diff1_v = res.dir1_pred - res.data_s.dir1
	lm_v = 0.5 * (diff0_v * diff0_v + diff1_v * diff1_v)

	# Horizontal-direction match using line-offset connections.
	# The connection-derived direction should match the UNet direction encoding directly.
	# Compare in base-mesh resolution to avoid mixing base-mesh geometry with HR-sampled direction fields.
	unet_dir0_lr = F.interpolate(res.data_s.dir0, size=res.xy_lr.shape[-2:], mode="bilinear", align_corners=True)
	unet_dir1_lr = F.interpolate(res.data_s.dir1, size=res.xy_lr.shape[-2:], mode="bilinear", align_corners=True)
	mask_lr = F.interpolate(res.mask, size=res.xy_lr.shape[-2:], mode="nearest")
	xy = res.xy_conn
	left = xy[:, 0]
	right = xy[:, 2]
	dx = right[:, 0:1] - left[:, 0:1]
	dy = right[:, 1:2] - left[:, 1:2]
	eps = 1e-8
	r2 = dx * dx + dy * dy + eps
	cos2 = (dx * dx - dy * dy) / r2
	sin2 = (2.0 * dx * dy) / r2
	inv_sqrt2 = 1.0 / (2.0 ** 0.5)
	dir0_h = 0.5 + 0.5 * cos2
	dir1_h = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
	diff0_h = dir0_h - unet_dir0_lr
	diff1_h = dir1_h - unet_dir1_lr
	lm_h = 0.5 * (diff0_h * diff0_h + diff1_h * diff1_h)

	lm_v_lr = F.interpolate(lm_v, size=res.xy_lr.shape[-2:], mode="bilinear", align_corners=True)
	lm = 0.5 * (lm_v_lr + lm_h)
	return lm * mask_lr, mask_lr


def direction_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm, _mask = direction_loss_map(res=res)
	wsum = res.mask.sum()
	if float(wsum.detach().cpu()) > 0.0:
		return lm.sum() / wsum
	return lm.mean()
