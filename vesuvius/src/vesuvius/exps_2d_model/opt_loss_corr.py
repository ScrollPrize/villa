from __future__ import annotations

import math

import torch


def corr_winding_loss(
	*,
	res,
	pts_c,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Collection-coupled winding loss from fixed enclosing segment indices.

	Uses both z_lo (floor) and z_hi (ceil) slices and interpolates observed
	winding by the fractional z position (bilinear in z).

	Returns loss maps of shape (N_z, K) so per-z reporting correctly
	distributes each point's contribution to its z-slice(s).
	"""
	dev = res.xy_conn.device
	dt = res.xy_conn.dtype
	n_z = int(res.xy_conn.shape[0])
	if pts_c is None:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((n_z, 1), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	pts = pts_c.points_xyz_winda.to(device=dev, dtype=dt)
	col = pts_c.collection_idx.to(device=dev, dtype=torch.int64)
	k = int(pts.shape[0])
	if k <= 0:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((n_z, 1), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	center = res.xy_conn[:, :, :, 1, :]
	leftp = res.xy_conn[:, :, :, 0, :]
	rightp = res.xy_conn[:, :, :, 2, :]
	n, hm, wm = int(center.shape[0]), int(center.shape[1]), int(center.shape[2])

	def _xf(*, side_pts: torch.Tensor, idx: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
		xf = torch.full((k,), float("nan"), device=dev, dtype=dt)
		zi = idx[:, 0]
		ri = idx[:, 1]
		ci = idx[:, 2]
		valid = ok & (zi >= 0) & (zi < n) & (ri >= 0) & (ri + 1 < hm) & (ci >= 0) & (ci < wm)
		if not valid.any():
			return xf
		zi_s = zi.clamp(0, max(n - 1, 0))
		ri_s = ri.clamp(0, max(hm - 2, 0))
		ci_s = ci.clamp(0, max(wm - 1, 0))
		q = pts[:, 0:2]
		p0 = center[zi_s, ri_s, ci_s]
		p1 = center[zi_s, ri_s + 1, ci_s]
		c0 = side_pts[zi_s, ri_s, ci_s]
		c1 = side_pts[zi_s, ri_s + 1, ci_s]
		s = p1 - p0
		s2 = (s * s).sum(dim=-1).clamp_min(1e-12)
		a = (((q - p0) * s).sum(dim=-1) / s2).clamp(0.0, 1.0)
		cp = p0 + s * a.unsqueeze(-1)
		v0 = c0 - p0
		v1 = c1 - p1
		v = v0 * (1.0 - a).unsqueeze(-1) + v1 * a.unsqueeze(-1)
		sn = s / s2.sqrt().unsqueeze(-1)
		nrm = torch.stack([-sn[:, 1], sn[:, 0]], dim=-1)
		num = ((q - cp) * nrm).sum(dim=-1)
		den = (v * nrm).sum(dim=-1)
		den = torch.where(den.abs() < 1e-12, den.sign() * 1e-12 + (den == 0).to(dtype=dt) * 1e-12, den)
		xf[valid] = (num / den)[valid]
		return xf

	def _obs_winding(*, idx_l, ok_l, idx_r, ok_r):
		"""Compute observed winding and validity mask for one z-slice."""
		idx_l = idx_l.to(device=dev, dtype=torch.int64)
		idx_r = idx_r.to(device=dev, dtype=torch.int64)
		ok_l = ok_l.to(device=dev, dtype=torch.bool)
		ok_r = ok_r.to(device=dev, dtype=torch.bool)
		xf_l = _xf(side_pts=leftp, idx=idx_l, ok=ok_l)
		xf_r = _xf(side_pts=rightp, idx=idx_r, ok=ok_r)
		w_left = idx_l[:, 2].to(dtype=dt) - xf_l
		w_right = idx_r[:, 2].to(dtype=dt) + xf_r
		valid = ok_l & ok_r & torch.isfinite(w_left) & torch.isfinite(w_right)
		obs = 0.5 * (w_left + w_right)
		return obs, valid

	# Observed winding at lo z-slice (floor)
	obs_lo, valid_lo = _obs_winding(
		idx_l=pts_c.idx_left, ok_l=pts_c.valid_left,
		idx_r=pts_c.idx_right, ok_r=pts_c.valid_right)

	# Observed winding at hi z-slice (ceil)
	obs_hi, valid_hi = _obs_winding(
		idx_l=pts_c.idx_left_hi, ok_l=pts_c.valid_left_hi,
		idx_r=pts_c.idx_right_hi, ok_r=pts_c.valid_right_hi)

	# Z-interpolation weights
	z_frac = pts_c.z_frac.to(device=dev, dtype=dt)
	w_lo = 1.0 - z_frac
	w_hi = z_frac

	# Combine: both valid → lerp, only one valid → use that one
	both = valid_lo & valid_hi
	only_lo = valid_lo & ~valid_hi
	only_hi = ~valid_lo & valid_hi
	valid = valid_lo | valid_hi

	obs = torch.full((k,), float("nan"), device=dev, dtype=dt)
	obs[both] = w_lo[both] * obs_lo[both] + w_hi[both] * obs_hi[both]
	obs[only_lo] = obs_lo[only_lo]
	obs[only_hi] = obs_hi[only_hi]

	# Collection-coupled error
	err = torch.full((k,), float("nan"), device=dev, dtype=dt)
	uc = torch.unique(col)
	for cid in uc.tolist():
		m = (col == int(cid)) & valid
		if bool(m.any()):
			obs_m = obs[m]
			wa_m = pts[m, 3]
			avg_pos = (obs_m - wa_m).mean()
			err_pos = obs_m - (avg_pos + wa_m)
			mse_pos = (err_pos * err_pos).mean()
			avg_neg = (obs_m + wa_m).mean()
			err_neg = obs_m - (avg_neg - wa_m)
			mse_neg = (err_neg * err_neg).mean()
			use_neg = bool((mse_neg < mse_pos).item())
			err[m] = err_neg if use_neg else err_pos

	point_valid = torch.isfinite(err)
	if not bool(point_valid.any()):
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((n_z, k), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	# Scalar loss: mean of valid squared errors
	err_sq = torch.where(point_valid, err * err, torch.zeros_like(err))
	loss = err_sq[point_valid].mean()

	# Per-z loss map: distribute each point's error to its z-slice(s)
	lm = torch.zeros((n_z, k), device=dev, dtype=dt)
	mask = torch.zeros((n_z, k), device=dev, dtype=dt)
	z_lo_idx = pts_c.idx_left[:, 0].to(device=dev, dtype=torch.int64).clamp(0, n_z - 1)
	z_hi_idx = pts_c.idx_left_hi[:, 0].to(device=dev, dtype=torch.int64).clamp(0, n_z - 1)
	col_idx = torch.arange(k, device=dev, dtype=torch.int64)
	same_z = (z_lo_idx == z_hi_idx) | (w_hi == 0.0)
	# lo-z entries for all valid points
	pv = point_valid
	lm[z_lo_idx[pv], col_idx[pv]] = err_sq[pv]
	mask[z_lo_idx[pv], col_idx[pv]] = torch.where(
		same_z[pv], torch.ones_like(w_lo[pv]), w_lo[pv])
	# hi-z entries for valid points with different z
	diff_valid = pv & ~same_z
	if diff_valid.any():
		lm[z_hi_idx[diff_valid], col_idx[diff_valid]] = err_sq[diff_valid]
		mask[z_hi_idx[diff_valid], col_idx[diff_valid]] = w_hi[diff_valid]

	return loss, (lm,), (mask,)
