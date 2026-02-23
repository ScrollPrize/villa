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
		for i in range(k):
			if not bool(ok[i]):
				continue
			zi = int(idx[i, 0])
			r = int(idx[i, 1])
			c = int(idx[i, 2])
			if zi < 0 or zi >= n or r < 0 or (r + 1) >= hm or c < 0 or c >= wm:
				continue
			q = pts[i, 0:2]
			p0 = center[zi, r, c]
			p1 = center[zi, r + 1, c]
			c0 = side_pts[zi, r, c]
			c1 = side_pts[zi, r + 1, c]
			s = p1 - p0
			s2 = (s * s).sum().clamp_min(1e-12)
			a = (((q - p0) * s).sum() / s2).clamp(0.0, 1.0)
			cp = p0 + s * a
			v0 = c0 - p0
			v1 = c1 - p1
			v = v0 * (1.0 - a) + v1 * a
			sn = s / torch.sqrt(s2)
			nrm = torch.stack((-sn[1], sn[0]))
			num = ((q - cp) * nrm).sum()
			den = (v * nrm).sum()
			den = torch.where(den.abs() < 1e-12, den.sign() * 1e-12 + (den == 0).to(dtype=dt) * 1e-12, den)
			xf[i] = num / den
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
	for i in range(k):
		if not bool(point_valid[i]):
			continue
		e2 = err_sq[i]
		z_lo_i = int(pts_c.idx_left[i, 0].item())
		z_hi_i = int(pts_c.idx_left_hi[i, 0].item())
		z_lo_i = max(0, min(n_z - 1, z_lo_i))
		z_hi_i = max(0, min(n_z - 1, z_hi_i))
		wl = float(w_lo[i].item())
		wh = float(w_hi[i].item())
		if z_lo_i == z_hi_i or wh == 0.0:
			lm[z_lo_i, i] = e2
			mask[z_lo_i, i] = 1.0
		else:
			lm[z_lo_i, i] = e2
			mask[z_lo_i, i] = wl
			lm[z_hi_i, i] = e2
			mask[z_hi_i, i] = wh

	return loss, (lm,), (mask,)
