from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def _dir_pred_v(*, xy_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	# Vertical mesh edge vector (v-edge), then rotate by +90° to match the
	# connection-supervised direction convention.
	x = xy_lr[..., 0].unsqueeze(1)
	y = xy_lr[..., 1].unsqueeze(1)
	dvx = x.new_zeros(x.shape)
	dvy = y.new_zeros(y.shape)
	dvx[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
	dvy[:, :, :-1, :] = y[:, :, 1:, :] - y[:, :, :-1, :]
	if x.shape[2] >= 2:
		dvx[:, :, -1, :] = dvx[:, :, -2, :]
		dvy[:, :, -1, :] = dvy[:, :, -2, :]
	gx = dvy
	gy = -dvx
	eps = 1e-8
	r2 = gx * gx + gy * gy + eps
	cos2 = (gx * gx - gy * gy) / r2
	sin2 = (2.0 * gx * gy) / r2
	inv_sqrt2 = 1.0 / (2.0 ** 0.5)
	dir0 = 0.5 + 0.5 * cos2
	dir1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
	return dir0, dir1


def direction_loss_maps(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Return (lm_v_lr, lm_conn_lr, mask_lr) at base-mesh resolution."""
	unet_dir0_lr = F.interpolate(res.data_s.dir0, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	unet_dir1_lr = F.interpolate(res.data_s.dir1, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	mask_lr = torch.minimum(res.mask_lr, res.mask_conn[..., 1])

	dir0_v_lr, dir1_v_lr = _dir_pred_v(xy_lr=res.xy_lr)
	diff0_v = dir0_v_lr - unet_dir0_lr
	diff1_v = dir1_v_lr - unet_dir1_lr
	lm_v_lr = 0.5 * (diff0_v * diff0_v + diff1_v * diff1_v)

	xy = res.xy_conn
	left = xy[..., 0, :]
	right = xy[..., 2, :]
	dx = (right[..., 0] - left[..., 0]).unsqueeze(1)
	dy = (right[..., 1] - left[..., 1]).unsqueeze(1)
	eps = 1e-8
	r2 = dx * dx + dy * dy + eps
	cos2 = (dx * dx - dy * dy) / r2
	sin2 = (2.0 * dx * dy) / r2
	inv_sqrt2 = 1.0 / (2.0 ** 0.5)
	dir0_h = 0.5 + 0.5 * cos2
	dir1_h = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
	diff0_h = dir0_h - unet_dir0_lr
	diff1_h = dir1_h - unet_dir1_lr
	lm_conn_lr = 0.5 * (diff0_h * diff0_h + diff1_h * diff1_h)

	return lm_v_lr * mask_lr, lm_conn_lr * mask_lr, mask_lr


def direction_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	lm_v_lr, lm_conn_lr, mask_lr = direction_loss_maps(res=res)
	lm = 0.5 * (lm_v_lr + lm_conn_lr)
	return lm, mask_lr


def direction_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm, mask_lr = direction_loss_map(res=res)
	wsum = mask_lr.sum()
	if float(wsum.detach().cpu()) > 0.0:
		return lm.sum() / wsum
	return lm.mean()
