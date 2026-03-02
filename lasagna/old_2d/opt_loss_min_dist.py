from __future__ import annotations

import torch

import model as fit_model


def min_dist_loss_map(*, res: fit_model.FitResult, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
	"""Penalize winding neighbors that are too close or on the wrong side.

	Uses xy_conn (N, Hm, Wm, 3, 2) where dim 3 = [left_conn, self, right_conn].
	Computes signed distance from self to left/right connection points along
	the orthogonal direction d = (-vy, vx).  Left should be on the -d side
	(negative signed dist), right on the +d side (positive signed dist).

	Any signed distance that violates the expected sign or is within
	`threshold` model pixels triggers a quadratic penalty.

	Returns (lm, mask) both (N, 1, Hm, Wm).
	"""
	xyc = res.xy_conn  # (N, Hm, Wm, 3, 2)
	n, hm, wm, _3, _2 = (int(v) for v in xyc.shape)

	self_xy = xyc[:, :, :, 1, :]   # (N, Hm, Wm, 2)
	left_xy = xyc[:, :, :, 0, :]   # (N, Hm, Wm, 2)
	right_xy = xyc[:, :, :, 2, :]  # (N, Hm, Wm, 2)

	# V-direction from mesh vertical neighbors (central difference, forward/backward at edges)
	sx = self_xy[..., 0]  # (N, Hm, Wm)
	sy = self_xy[..., 1]
	vx = torch.zeros_like(sx)
	vy = torch.zeros_like(sy)
	if hm >= 3:
		vx[:, 1:-1, :] = sx[:, 2:, :] - sx[:, :-2, :]
		vx[:, 0, :] = sx[:, 1, :] - sx[:, 0, :]
		vx[:, -1, :] = sx[:, -1, :] - sx[:, -2, :]
		vy[:, 1:-1, :] = sy[:, 2:, :] - sy[:, :-2, :]
		vy[:, 0, :] = sy[:, 1, :] - sy[:, 0, :]
		vy[:, -1, :] = sy[:, -1, :] - sy[:, -2, :]
	elif hm >= 2:
		vx[:, 0, :] = sx[:, 1, :] - sx[:, 0, :]
		vx[:, 1, :] = vx[:, 0, :]
		vy[:, 0, :] = sy[:, 1, :] - sy[:, 0, :]
		vy[:, 1, :] = vy[:, 0, :]

	# Orthogonal direction: rotate 90° CCW → d = (-vy, vx)
	dx = -vy
	dy = vx
	d_len = torch.sqrt(dx * dx + dy * dy + 1e-12)
	dx = dx / d_len
	dy = dy / d_len

	# Signed distance: project (conn - self) onto d
	# Left: expected negative (on the -d side)
	# Right: expected positive (on the +d side)
	vec_l_x = left_xy[..., 0] - sx
	vec_l_y = left_xy[..., 1] - sy
	signed_dist_l = vec_l_x * dx + vec_l_y * dy  # should be < 0

	vec_r_x = right_xy[..., 0] - sx
	vec_r_y = right_xy[..., 1] - sy
	signed_dist_r = vec_r_x * dx + vec_r_y * dy  # should be > 0

	# Penalty: for left, penalize when signed_dist_l > -threshold (too close or wrong side)
	# For right, penalize when signed_dist_r < threshold (too close or wrong side)
	# margin_l = -signed_dist_l - threshold  (should be >= 0 when OK)
	# margin_r =  signed_dist_r - threshold  (should be >= 0 when OK)
	margin_l = -signed_dist_l - threshold
	margin_r = signed_dist_r - threshold

	penalty_l = torch.where(margin_l < 0.0, margin_l * margin_l, torch.zeros_like(margin_l))
	penalty_r = torch.where(margin_r < 0.0, margin_r * margin_r, torch.zeros_like(margin_r))

	# Mask out edge windings (col 0 has no real left neighbor, col -1 no real right)
	mask = torch.ones((n, 1, hm, wm), device=xyc.device, dtype=xyc.dtype)
	mask_l = torch.ones_like(penalty_l)
	mask_r = torch.ones_like(penalty_r)
	if wm > 1:
		mask_l[:, :, 0] = 0.0   # col 0: left conn is mirrored, not real
		mask_r[:, :, -1] = 0.0  # col -1: right conn is mirrored, not real

	lm = 0.5 * (penalty_l * mask_l + penalty_r * mask_r)
	lm = lm.unsqueeze(1)  # (N, 1, Hm, Wm)

	return lm, mask


def min_dist_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = min_dist_loss_map(res=res)
	wsum = mask.sum()
	if float(wsum) > 0.0:
		loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	return loss, (lm,), (mask,)
