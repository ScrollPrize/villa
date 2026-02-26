from __future__ import annotations

import math

import torch


def corr_winding_loss(
	*,
	res,
	pts_c,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Collection-coupled winding loss from fixed enclosing segment indices.

	Uses z-interpolated mesh lookups (z_lo from idx[:, 0], z_hi, z_frac)
	to compute observed winding on a single interpolated mesh.

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

	z_frac = pts_c.z_frac.to(device=dev, dtype=dt)
	z_hi_idx = pts_c.z_hi.to(device=dev, dtype=torch.int64).clamp(0, n - 1)

	def _xf(*, side_pts: torch.Tensor, idx: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
		xf = torch.full((k,), float("nan"), device=dev, dtype=dt)
		zi = idx[:, 0]
		ri = idx[:, 1]
		ci = idx[:, 2]
		valid = ok & (zi >= 0) & (zi < n) & (ri >= 0) & (ri + 1 < hm) & (ci >= 0) & (ci < wm)
		if not valid.any():
			return xf
		zi_s = zi.clamp(0, max(n - 1, 0))
		z_hi_s = z_hi_idx
		ri_s = ri.clamp(0, max(hm - 2, 0))
		ci_s = ci.clamp(0, max(wm - 1, 0))
		fk = z_frac.unsqueeze(-1)  # (K, 1)
		q = pts[:, 0:2]
		p0 = center[zi_s, ri_s, ci_s] * (1.0 - fk) + center[z_hi_s, ri_s, ci_s] * fk
		p1 = center[zi_s, ri_s + 1, ci_s] * (1.0 - fk) + center[z_hi_s, ri_s + 1, ci_s] * fk
		c0 = side_pts[zi_s, ri_s, ci_s] * (1.0 - fk) + side_pts[z_hi_s, ri_s, ci_s] * fk
		c1 = side_pts[zi_s, ri_s + 1, ci_s] * (1.0 - fk) + side_pts[z_hi_s, ri_s + 1, ci_s] * fk
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

	# Single observed winding on interpolated mesh
	idx_l = pts_c.idx_left.to(device=dev, dtype=torch.int64)
	idx_r = pts_c.idx_right.to(device=dev, dtype=torch.int64)
	ok_l = pts_c.valid_left.to(device=dev, dtype=torch.bool)
	ok_r = pts_c.valid_right.to(device=dev, dtype=torch.bool)
	xf_l = _xf(side_pts=leftp, idx=idx_l, ok=ok_l)
	xf_r = _xf(side_pts=rightp, idx=idx_r, ok=ok_r)
	w_left = idx_l[:, 2].to(dtype=dt) - xf_l
	w_right = idx_r[:, 2].to(dtype=dt) + xf_r
	valid = ok_l & ok_r & torch.isfinite(w_left) & torch.isfinite(w_right)
	obs = 0.5 * (w_left + w_right)

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
	z_lo_map = pts_c.idx_left[:, 0].to(device=dev, dtype=torch.int64).clamp(0, n_z - 1)
	z_hi_map = pts_c.z_hi.to(device=dev, dtype=torch.int64).clamp(0, n_z - 1)
	col_idx = torch.arange(k, device=dev, dtype=torch.int64)
	w_lo = 1.0 - z_frac
	w_hi = z_frac
	same_z = (z_lo_map == z_hi_map) | (w_hi == 0.0)
	# lo-z entries for all valid points
	pv = point_valid
	lm[z_lo_map[pv], col_idx[pv]] = err_sq[pv]
	mask[z_lo_map[pv], col_idx[pv]] = torch.where(
		same_z[pv], torch.ones_like(w_lo[pv]), w_lo[pv])
	# hi-z entries for valid points with different z
	diff_valid = pv & ~same_z
	if diff_valid.any():
		lm[z_hi_map[diff_valid], col_idx[diff_valid]] = err_sq[diff_valid]
		mask[z_hi_map[diff_valid], col_idx[diff_valid]] = w_hi[diff_valid]

	return loss, (lm,), (mask,)
