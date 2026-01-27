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


def direction_loss_maps(
	*,
	res: fit_model.FitResult,
) -> tuple[
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
]:
	"""Return (lm_v_lr, lm_conn_l_lr, lm_conn_r_lr, mask_v_lr, mask_conn_l_lr, mask_conn_r_lr) at base-mesh resolution."""
	# print("src:",res.data_s.dir0.shape, "tgt",res.xy_lr.shape[1:3])
	unet_dir0_lr = F.interpolate(res.data_s.dir0, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	unet_dir1_lr = F.interpolate(res.data_s.dir1, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	mask_lr = res.mask_lr

	# Separate masks:
	# - vertical uses v-edge (y -> y+1), so mask is min of the two samples
	# - conn uses left/right endpoints, so mask is min of those endpoints
	mask_v_lr = mask_lr.clone()
	if mask_v_lr.shape[2] >= 2:
		mask_v_lr[:, :, :-1, :] = torch.minimum(mask_lr[:, :, :-1, :], mask_lr[:, :, 1:, :])
		mask_v_lr[:, :, -1, :] = mask_v_lr[:, :, -2, :]
	mask_conn_l_lr = torch.minimum(res.mask_conn[..., 0], res.mask_conn[..., 1])
	mask_conn_r_lr = torch.minimum(res.mask_conn[..., 1], res.mask_conn[..., 2])

	dir0_v_lr, dir1_v_lr = _dir_pred_v(xy_lr=res.xy_lr)
	diff0_v = dir0_v_lr - unet_dir0_lr
	diff1_v = dir1_v_lr - unet_dir1_lr
	lm_v_lr = 0.5 * (diff0_v * diff0_v + diff1_v * diff1_v)

	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	right = xy[..., 2, :]

	def _dir_lm(*, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
		dx = (x1[..., 0] - x0[..., 0]).unsqueeze(1)
		dy = (x1[..., 1] - x0[..., 1]).unsqueeze(1)
		eps = 1e-8
		r2 = dx * dx + dy * dy + eps
		cos2 = (dx * dx - dy * dy) / r2
		sin2 = (2.0 * dx * dy) / r2
		inv_sqrt2 = 1.0 / (2.0 ** 0.5)
		dir0 = 0.5 + 0.5 * cos2
		dir1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
		diff0 = dir0 - unet_dir0_lr
		diff1 = dir1 - unet_dir1_lr
		return 0.5 * (diff0 * diff0 + diff1 * diff1)

	lm_conn_l_lr = _dir_lm(x0=left, x1=mid)
	lm_conn_r_lr = _dir_lm(x0=mid, x1=right)

	return (
		lm_v_lr * mask_v_lr,
		lm_conn_l_lr * mask_conn_l_lr,
		lm_conn_r_lr * mask_conn_r_lr,
		mask_v_lr,
		mask_conn_l_lr,
		mask_conn_r_lr,
	)
def direction_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm_v_lr, lm_conn_l_lr, lm_conn_r_lr, mask_v_lr, mask_conn_l_lr, mask_conn_r_lr = direction_loss_maps(res=res)
	wsum_v = mask_v_lr.sum()
	wsum_l = mask_conn_l_lr.sum()
	wsum_r = mask_conn_r_lr.sum()

	lv = lm_v_lr.sum() / wsum_v if float(wsum_v.detach().cpu()) > 0.0 else lm_v_lr.mean()
	ll = lm_conn_l_lr.sum() / wsum_l if float(wsum_l.detach().cpu()) > 0.0 else lm_conn_l_lr.mean()
	lr = lm_conn_r_lr.sum() / wsum_r if float(wsum_r.detach().cpu()) > 0.0 else lm_conn_r_lr.mean()
	return (lv + ll + lr) / 3.0
