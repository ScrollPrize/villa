from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def _encode_dir(gx: torch.Tensor, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Encode a 2D direction vector (gx, gy) into the (dir0, dir1) representation."""
	eps = 1e-8
	r2 = gx * gx + gy * gy + eps
	cos2 = (gx * gx - gy * gy) / r2
	sin2 = 2.0 * gx * gy / r2
	inv_sqrt2 = 1.0 / (2.0 ** 0.5)
	d0 = 0.5 + 0.5 * cos2
	d1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
	return d0, d1


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
	return _encode_dir(gx, gy)


def dir_v_loss_maps(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm_v_lr, mask_v_lr) at base-mesh resolution."""
	unet_dir0_lr = F.interpolate(res.data_s.dir0, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	unet_dir1_lr = F.interpolate(res.data_s.dir1, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	mask_lr = res.mask_lr
	mask_v_lr = mask_lr.clone()
	if mask_v_lr.shape[2] >= 2:
		mask_v_lr[:, :, :-1, :] = torch.minimum(mask_lr[:, :, :-1, :], mask_lr[:, :, 1:, :])
		mask_v_lr[:, :, -1, :] = mask_v_lr[:, :, -2, :]

	dir0_v_lr, dir1_v_lr = _dir_pred_v(xy_lr=res.xy_lr)
	diff0_v = dir0_v_lr - unet_dir0_lr
	diff1_v = dir1_v_lr - unet_dir1_lr
	lm_v_lr = 0.5 * (diff0_v * diff0_v + diff1_v * diff1_v)
	return lm_v_lr * mask_v_lr, mask_v_lr


def dir_conn_loss_maps(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Return (lm_conn_l_lr, lm_conn_r_lr, mask_conn_l_lr, mask_conn_r_lr) at base-mesh resolution."""
	unet_dir0_lr = F.interpolate(res.data_s.dir0, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	unet_dir1_lr = F.interpolate(res.data_s.dir1, size=res.xy_lr.shape[1:3], mode="bilinear", align_corners=True)
	mask_conn_l_lr = torch.minimum(res.mask_conn[..., 0], res.mask_conn[..., 1])
	mask_conn_r_lr = torch.minimum(res.mask_conn[..., 1], res.mask_conn[..., 2])
	if mask_conn_l_lr.shape[3] >= 1:
		mask_conn_l_lr[:, :, :, 0] = 0.0
	if mask_conn_r_lr.shape[3] >= 1:
		mask_conn_r_lr[:, :, :, -1] = 0.0

	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	right = xy[..., 2, :]

	def _dir_lm(*, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
		dx = (x1[..., 0] - x0[..., 0]).unsqueeze(1)
		dy = (x1[..., 1] - x0[..., 1]).unsqueeze(1)
		dir0, dir1 = _encode_dir(dx, dy)
		diff0 = dir0 - unet_dir0_lr
		diff1 = dir1 - unet_dir1_lr
		return 0.5 * (diff0 * diff0 + diff1 * diff1)

	lm_conn_l_lr = _dir_lm(x0=left, x1=mid) * mask_conn_l_lr
	lm_conn_r_lr = _dir_lm(x0=mid, x1=right) * mask_conn_r_lr
	return lm_conn_l_lr, lm_conn_r_lr, mask_conn_l_lr, mask_conn_r_lr


def z_normal_loss_maps(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for z-normal direction loss.

	Compares z-segment direction (between consecutive z meshes) to direction
	channels from y-axis (ZX plane) and x-axis (ZY plane) processing.

	For the ZX plane: project z-segment to (dx, dz), compare to dir0_y/dir1_y.
	For the ZY plane: project z-segment to (dy, dz), compare to dir0_x/dir1_x.

	Loss is evaluated at both z endpoints of each segment and averaged.
	"""
	xy = res.xy_lr  # (N, Hm, Wm, 2)
	N = int(xy.shape[0])
	if N < 2:
		z = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return z, z

	data = res.data
	has_y = data.dir0_y is not None and data.dir1_y is not None
	has_x = data.dir0_x is not None and data.dir1_x is not None
	if not has_y and not has_x:
		z = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return z, z

	# Z-segment vectors in model pixel units
	dxy = xy[1:] - xy[:-1]  # (N-1, Hm, Wm, 2)
	dx = dxy[..., 0]  # (N-1, Hm, Wm)
	dy = dxy[..., 1]  # (N-1, Hm, Wm)

	# dz in LR model units (same coordinate space as dx/dy from xy_lr)
	dz = float(res.params.z_step_vx)

	lr_size = xy.shape[1:3]  # (Hm, Wm)

	# Mask: valid at both z_i and z_{i+1}
	m = res.mask_lr
	mask = torch.minimum(m[:-1], m[1:])  # (N-1, 1, Hm, Wm)

	lm = torch.zeros_like(mask)
	n_terms = 0

	if has_y:
		# ZX plane: z-segment tangent is (dx, dz), rotate +90° to get normal
		# matching _dir_pred_v convention: (gx, gy) = (dz, -dx)
		pred_d0, pred_d1 = _encode_dir(torch.full_like(dx, dz), -dx)
		pred_d0 = pred_d0.unsqueeze(1)  # (N-1, 1, Hm, Wm)
		pred_d1 = pred_d1.unsqueeze(1)

		# Sample dir_y at base-mesh resolution (downsample from HR)
		d0y_lr = F.interpolate(res.data_s.dir0_y, size=lr_size, mode="bilinear", align_corners=True)
		d1y_lr = F.interpolate(res.data_s.dir1_y, size=lr_size, mode="bilinear", align_corners=True)

		# Loss at both endpoints of the z-segment, averaged
		loss_zi = 0.5 * ((pred_d0 - d0y_lr[:-1]) ** 2 + (pred_d1 - d1y_lr[:-1]) ** 2)
		loss_zip1 = 0.5 * ((pred_d0 - d0y_lr[1:]) ** 2 + (pred_d1 - d1y_lr[1:]) ** 2)
		lm = lm + 0.5 * (loss_zi + loss_zip1)
		n_terms += 1

	if has_x:
		# ZY plane: z-segment tangent is (dy, dz), rotate +90° to get normal
		# matching _dir_pred_v convention: (gx, gy) = (dz, -dy)
		pred_d0, pred_d1 = _encode_dir(torch.full_like(dy, dz), -dy)
		pred_d0 = pred_d0.unsqueeze(1)
		pred_d1 = pred_d1.unsqueeze(1)

		d0x_lr = F.interpolate(res.data_s.dir0_x, size=lr_size, mode="bilinear", align_corners=True)
		d1x_lr = F.interpolate(res.data_s.dir1_x, size=lr_size, mode="bilinear", align_corners=True)

		loss_zi = 0.5 * ((pred_d0 - d0x_lr[:-1]) ** 2 + (pred_d1 - d1x_lr[:-1]) ** 2)
		loss_zip1 = 0.5 * ((pred_d0 - d0x_lr[1:]) ** 2 + (pred_d1 - d1x_lr[1:]) ** 2)
		lm = lm + 0.5 * (loss_zi + loss_zip1)
		n_terms += 1

	if n_terms > 1:
		lm = lm / float(n_terms)

	return lm * mask, mask


def dir_v_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Vertical direction loss vs (dir0, dir1) encodings."""
	lm_v_lr, mask_v_lr = dir_v_loss_maps(res=res)
	wsum_v = mask_v_lr.sum()
	loss = lm_v_lr.sum() / wsum_v if float(wsum_v.detach().cpu()) > 0.0 else lm_v_lr.mean()
	return loss, (lm_v_lr,), (mask_v_lr,)


def dir_conn_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Connection direction loss vs (dir0, dir1) encodings."""
	lm_l, lm_r, mask_l, mask_r = dir_conn_loss_maps(res=res)
	wsum_l = mask_l.sum()
	wsum_r = mask_r.sum()
	ll = lm_l.sum() / wsum_l if float(wsum_l.detach().cpu()) > 0.0 else lm_l.mean()
	lr = lm_r.sum() / wsum_r if float(wsum_r.detach().cpu()) > 0.0 else lm_r.mean()
	return 0.5 * (ll + lr), (lm_l, lm_r), (mask_l, mask_r)


def z_normal_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Z-normal direction loss: z-segment alignment with ZX/ZY direction channels."""
	lm, mask = z_normal_loss_maps(res=res)
	wsum = mask.sum()
	loss = lm.sum() / wsum if float(wsum.detach().cpu()) > 0.0 else lm.mean()
	return loss, (lm,), (mask,)
