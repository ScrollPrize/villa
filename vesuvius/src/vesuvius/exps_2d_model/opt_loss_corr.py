from __future__ import annotations

import torch


def corr_winding_loss(
	*,
	res,
	pts_c,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Collection-coupled winding loss from fixed enclosing segment indices.

	- Uses only points with both left/right valid.
	- Observed winding is computed from left/right side x-frac.
	- Error is obs - (collection_avg + wind_a), where avg is differentiable.
	"""
	dev = res.xy_conn.device
	dt = res.xy_conn.dtype
	if pts_c is None:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1, 1), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	points_xyz_winda = pts_c.points_xyz_winda
	collection_idx = pts_c.collection_idx
	idx_left = pts_c.idx_left
	valid_left = pts_c.valid_left
	idx_right = pts_c.idx_right
	valid_right = pts_c.valid_right

	pts = points_xyz_winda.to(device=dev, dtype=dt)
	col = collection_idx.to(device=dev, dtype=torch.int64)
	idx_l = idx_left.to(device=dev, dtype=torch.int64)
	idx_r = idx_right.to(device=dev, dtype=torch.int64)
	ok_l = valid_left.to(device=dev, dtype=torch.bool)
	ok_r = valid_right.to(device=dev, dtype=torch.bool)
	k = int(pts.shape[0])
	if k <= 0:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1, 1), device=dev, dtype=dt)
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
			z = int(idx[i, 0])
			r = int(idx[i, 1])
			c = int(idx[i, 2])
			if z < 0 or z >= n or r < 0 or (r + 1) >= hm or c < 0 or c >= wm:
				continue
			q = pts[i, 0:2]
			p0 = center[z, r, c]
			p1 = center[z, r + 1, c]
			c0 = side_pts[z, r, c]
			c1 = side_pts[z, r + 1, c]
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

	xf_l = _xf(side_pts=leftp, idx=idx_l, ok=ok_l)
	xf_r = _xf(side_pts=rightp, idx=idx_r, ok=ok_r)
	w_left = idx_l[:, 2].to(dtype=dt) - xf_l
	w_right = idx_r[:, 2].to(dtype=dt) + xf_r
	valid = ok_l & ok_r & torch.isfinite(w_left) & torch.isfinite(w_right)
	obs = 0.5 * (w_left + w_right)

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

	mask = torch.isfinite(err)
	if not bool(mask.any()):
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1, k), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)
	lm = torch.where(mask, err * err, torch.zeros_like(err))
	loss = lm[mask].mean()
	return loss, (lm.view(1, k),), (mask.to(dtype=dt).view(1, k),)
