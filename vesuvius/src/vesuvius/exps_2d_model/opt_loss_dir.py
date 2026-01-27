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
	# Target is a right angle to the base UNet direction.
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
	if dir0_h.shape[-2:] != res.data_s.dir0.shape[-2:]:
		dir0_h = F.interpolate(dir0_h, size=res.data_s.dir0.shape[-2:], mode="bilinear", align_corners=True)
		dir1_h = F.interpolate(dir1_h, size=res.data_s.dir0.shape[-2:], mode="bilinear", align_corners=True)

	diff0_h = dir0_h - (1.0 - res.data_s.dir0)
	diff1_h = dir1_h - (1.0 - res.data_s.dir1)
	lm_h = 0.5 * (diff0_h * diff0_h + diff1_h * diff1_h)

	lm = 0.5 * (lm_v + lm_h)
	return lm * res.mask, res.mask


def direction_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm, _mask = direction_loss_map(res=res)
	wsum = res.mask.sum()
	if float(wsum.detach().cpu()) > 0.0:
		return lm.sum() / wsum
	return lm.mean()
