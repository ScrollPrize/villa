from __future__ import annotations

import torch

import model as fit_model


def smooth_x_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for horizontal straightness on the base mesh.

	Uses 3-point connectivity from `res.xy_conn`: (left, mid, right).
	Penalizes `v_left + v_right` where vectors are anchored at `mid`.
	"""
	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	right = xy[..., 2, :]

	v_left = mid - left
	v_right = right - mid
	v_sum = v_left + v_right
	lm = (v_sum * v_sum).sum(dim=-1).unsqueeze(1)

	m = res.mask_conn
	m_left = m[..., 0]
	m_mid = m[..., 1]
	m_right = m[..., 2]
	mask = torch.minimum(torch.minimum(m_left, m_mid), m_right)
	return lm, mask


def smooth_y_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for y-smoothness of the vertical step vectors on `xy_lr`.

	Penalizes changes in the y-step vector: if `v_j = p[j+1]-p[j]`, we penalize
	`v_{j+1}-v_j`.
	"""
	xy = res.xy_lr
	if int(xy.shape[1]) < 3:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0

	step = xy[:, 1:, :, :] - xy[:, :-1, :, :]
	d2 = step[:, 1:, :, :] - step[:, :-1, :, :]
	lm = (d2 * d2).sum(dim=-1).unsqueeze(1)

	m = res.mask_lr
	mask = torch.minimum(torch.minimum(m[:, :, :-2, :], m[:, :, 1:-1, :]), m[:, :, 2:, :])
	return lm, mask


def meshoff_smooth_y_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for y-smoothness of horizontal connections.

	Regularizes how the connection vectors (left-mid & mid-right) change along y.
	This captures the effect of `conn_offset_ms` (previously `mesh_offset_ms`) in the model.
	"""
	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	right = xy[..., 2, :]

	v_l = left - mid
	v_r = right - mid
	if int(v_l.shape[1]) < 2:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0

	dy_l = v_l[:, 1:, :, :] - v_l[:, :-1, :, :]
	dy_r = v_r[:, 1:, :, :] - v_r[:, :-1, :, :]
	lm = 0.5 * ((dy_l * dy_l).sum(dim=-1) + (dy_r * dy_r).sum(dim=-1))
	lm = lm.unsqueeze(1)

	mc = res.mask_conn
	mask_l = torch.minimum(mc[..., 0], mc[..., 1])
	mask_r = torch.minimum(mc[..., 1], mc[..., 2])
	mask_l = torch.minimum(mask_l[:, :, 1:, :], mask_l[:, :, :-1, :])
	mask_r = torch.minimum(mask_r[:, :, 1:, :], mask_r[:, :, :-1, :])
	mask = torch.minimum(mask_l, mask_r)
	return lm, mask


def conn_y_smooth_l_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for y-smoothness of the left connection vector."""
	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	v_l = left - mid
	if int(v_l.shape[1]) < 2:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0
	ddy = v_l[:, 1:, :, :] - v_l[:, :-1, :, :]
	lm = (ddy * ddy).sum(dim=-1).unsqueeze(1)

	mc = res.mask_conn
	mask = torch.minimum(mc[..., 0], mc[..., 1])
	mask = torch.minimum(mask[:, :, 1:, :], mask[:, :, :-1, :])
	return lm, mask


def conn_y_smooth_r_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for y-smoothness of the right connection vector."""
	xy = res.xy_conn
	right = xy[..., 2, :]
	mid = xy[..., 1, :]
	v_r = right - mid
	if int(v_r.shape[1]) < 2:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0
	ddy = v_r[:, 1:, :, :] - v_r[:, :-1, :, :]
	lm = (ddy * ddy).sum(dim=-1).unsqueeze(1)

	mc = res.mask_conn
	mask = torch.minimum(mc[..., 1], mc[..., 2])
	mask = torch.minimum(mask[:, :, 1:, :], mask[:, :, :-1, :])
	return lm, mask


def angle_symmetry_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) penalizing deviation from a right angle (|sin|=1) with signed convention.

	Uses the same winding sign convention as the grad-mag direction sign:
	- mesh_dir is inferred from the vertical mesh direction (v-1 -> v+1)
	- horizontal vectors use (mid-left) and (right-mid) (both point along winding)
	- we flip vv to align with mesh_dir (dot >= 0)
	- we penalize so cross(v_dir, h_dir)/(|v||h|) == +1 (right angle with correct sign)
	"""
	xy = res.xy_conn
	left = xy[..., 0, :]
	mid = xy[..., 1, :]
	right = xy[..., 2, :]

	hv_l = mid - left
	hv_r = right - mid

	xy_lr = res.xy_lr
	if int(xy_lr.shape[1]) < 2:
		base = torch.zeros((), device=xy_lr.device, dtype=xy_lr.dtype)
		mask0 = torch.zeros((), device=xy_lr.device, dtype=xy_lr.dtype)
		return base, mask0

	# mesh_dir at vertex rows (same as grad-mag sign convention), then use
	# md_seg (row -> row+1) as the vertical direction for the angle penalty.
	md = xy_lr.new_zeros(xy_lr.shape)
	if int(xy_lr.shape[1]) >= 3:
		md[:, 1:-1, :, :] = xy_lr[:, 2:, :, :] - xy_lr[:, :-2, :, :]
		md[:, 0, :, :] = xy_lr[:, 1, :, :] - xy_lr[:, 0, :, :]
		md[:, -1, :, :] = xy_lr[:, -1, :, :] - xy_lr[:, -2, :, :]
	else:
		md[:, 0, :, :] = xy_lr[:, 1, :, :] - xy_lr[:, 0, :, :]
		md[:, 1, :, :] = md[:, 0, :, :]
	md_seg = md[:, :-1, :, :]

	# Determine winding sign per row/col (same logic as opt_loss_gradmag).
	v_l = left - mid
	v_r = right - mid
	mdx = md[..., 0]
	mdy = md[..., 1]
	cross_r = mdx * v_r[..., 1] - mdy * v_r[..., 0]
	cross_l = mdx * v_l[..., 1] - mdy * v_l[..., 0]
	ok = (cross_r > 0.0) & (cross_l < 0.0)
	target_sign = torch.where(ok, torch.ones_like(cross_r), -torch.ones_like(cross_r))
	target_sign = target_sign[:, :-1, :]

	vv = md_seg[:, :, :, None, :]
	hv_l = hv_l[:, :-1, :, None, :]
	hv_r = hv_r[:, :-1, :, None, :]
	hv = torch.cat([hv_l, hv_r], dim=3)

	eps = 1e-8
	cross = vv[..., 0] * hv[..., 1] - vv[..., 1] * hv[..., 0]
	hn = torch.sqrt((hv * hv).sum(dim=-1) + eps)
	vn = torch.sqrt((vv * vv).sum(dim=-1) + eps)
	sin = cross / (hn * vn + eps)
	target_sign2 = target_sign.unsqueeze(-1).expand_as(sin)
	d = sin - target_sign2
	lm = (d * d).mean(dim=3).unsqueeze(1)

	m_lr = torch.minimum(res.mask_lr[:, :, 1:, :], res.mask_lr[:, :, :-1, :])
	mc = res.mask_conn
	m_l = torch.minimum(mc[:, :, :-1, :, 0], mc[:, :, :-1, :, 1])
	m_r = torch.minimum(mc[:, :, :-1, :, 1], mc[:, :, :-1, :, 2])
	# Crop out left/right mesh edges: no true neighbor conn exists there.
	if m_l.shape[3] >= 1:
		m_l[:, :, :, 0] = 0.0
	if m_r.shape[3] >= 1:
		m_r[:, :, :, -1] = 0.0
	mask_h = torch.minimum(m_l, m_r)
	mask = torch.minimum(m_lr, mask_h)
	return lm, mask


def y_straight_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for second-difference straightness along y in pixel xy."""
	xy = res.xy_lr
	if int(xy.shape[1]) < 3:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0

	step = xy[:, 1:, :, :] - xy[:, :-1, :, :]
	d2 = step[:, 1:, :, :] - step[:, :-1, :, :]
	lm = (d2 * d2).sum(dim=-1).unsqueeze(1)

	m = res.mask_lr
	mask = torch.minimum(torch.minimum(m[:, :, :-2, :], m[:, :, 1:-1, :]), m[:, :, 2:, :])
	return lm, mask


def z_straight_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for second-difference straightness along z (depth) in pixel xy."""
	xy = res.xy_lr
	if int(xy.shape[0]) < 3:
		base = torch.zeros((), device=xy.device, dtype=xy.dtype)
		mask0 = torch.zeros((), device=xy.device, dtype=xy.dtype)
		return base, mask0

	z_step_px = float(res.params.z_step_vx)
	z_step_px = max(1e-6, z_step_px)

	prev = xy[:-2, :, :, :]
	mid = xy[1:-1, :, :, :]
	next = xy[2:, :, :, :]
	d = (mid - 0.5 * (prev + next)) / z_step_px
	lm = (d * d).sum(dim=-1).unsqueeze(1)

	m = res.mask_lr
	mask = torch.minimum(torch.minimum(m[:-2, :, :, :], m[1:-1, :, :, :]), m[2:, :, :, :])
	return lm, mask


def mean_pos_loss_map(*, res: fit_model.FitResult, target_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) penalizing mean(xy_lr) deviating from `target_xy`.

	`target_xy` must be (2,) or (N,2) in pixel xy.
	"""
	xy = res.xy_lr
	mean_xy = xy.mean(dim=(1, 2))
	if target_xy.ndim == 1:
		tgt = target_xy.view(1, 2).expand(int(mean_xy.shape[0]), 2)
	else:
		tgt = target_xy
	d = mean_xy - tgt
	lm = (d * d).sum(dim=-1).view(-1, 1, 1, 1)
	mask = torch.ones_like(lm)
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def _full_mean(lm: torch.Tensor) -> torch.Tensor:
	return lm.mean()


def smooth_x_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = smooth_x_loss_map(res=res)
	return _full_mean(lm)


def smooth_y_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = smooth_y_loss_map(res=res)
	return _full_mean(lm)


def meshoff_smooth_y_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = meshoff_smooth_y_loss_map(res=res)
	return _full_mean(lm)


def conn_y_smooth_l_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = conn_y_smooth_l_loss_map(res=res)
	return _full_mean(lm)


def conn_y_smooth_r_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = conn_y_smooth_r_loss_map(res=res)
	return _full_mean(lm)


def angle_symmetry_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = angle_symmetry_loss_map(res=res)
	return _full_mean(lm)


def y_straight_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = y_straight_loss_map(res=res)
	return _full_mean(lm)


def z_straight_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = z_straight_loss_map(res=res)
	return _full_mean(lm)


def angle_symmetry_loss_uv(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Same as `angle_symmetry_loss`, but averaged over both horizontal directions."""
	lm, mask = angle_symmetry_loss_map(res=res)
	return _full_mean(lm)


def mean_pos_loss(*, res: fit_model.FitResult, target_xy: torch.Tensor) -> torch.Tensor:
	lm, mask = mean_pos_loss_map(res=res, target_xy=target_xy)
	return _full_mean(lm)
