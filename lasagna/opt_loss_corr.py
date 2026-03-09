from __future__ import annotations

import math

import torch

import model as fit_model

_dbg_call_count = 0
_last_results: dict | None = None
_snap_mode: bool = True

# Snap-mode anchor state (persists across calls)
_snap_anchors_h: torch.Tensor | None = None
_snap_anchors_w: torch.Tensor | None = None
_snap_anchors_u: torch.Tensor | None = None
_snap_anchors_v: torch.Tensor | None = None
_snap_anchors_dir: torch.Tensor | None = None  # 0=prev, 1=next
_snap_initialized: bool = False


def set_snap_mode(enabled: bool) -> None:
	global _snap_mode
	_snap_mode = enabled


def corr_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Dispatch to snap or legacy corr loss."""
	if _snap_mode:
		return _corr_snap_loss(res=res)
	return _corr_legacy_loss(res=res)


def _corr_legacy_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Legacy 3D correction point loss: point-to-quad nearest surface, collection-coupled winding error."""
	global _dbg_call_count, _last_results
	_dbg_call_count += 1
	dbg = (_dbg_call_count == 1)

	dev = res.xyz_lr.device
	dt = res.xyz_lr.dtype

	pts_c = res.data.corr_points
	if pts_c is None or pts_c.points_xyz_winda.shape[0] == 0:
		if _dbg_call_count <= 2:
			print("[corr] no correction points")
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	pts = pts_c.points_xyz_winda.to(device=dev, dtype=dt)   # (K, 4)
	col = pts_c.collection_idx.to(device=dev, dtype=torch.int64)  # (K,)
	pt_ids = pts_c.point_ids.to(device=dev, dtype=torch.int64)  # (K,)
	K = int(pts.shape[0])
	P = pts[:, :3]      # (K, 3)
	winda = pts[:, 3]   # (K,)

	xyz_lr = res.xyz_lr          # (D, Hm, Wm, 3) — has grad
	normals = res.normals        # (D, Hm, Wm, 3) — detached
	D, Hm, Wm, _ = xyz_lr.shape

	if dbg:
		print(f"[corr] step={_dbg_call_count}: K={K} points, mesh=({D},{Hm},{Wm}), "
			  f"collections={torch.unique(col).tolist()}")

	# --- Step 2: unit normals (detached) ---
	n_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)  # (D, Hm, Wm, 3)

	# --- Step 3: per-depth-layer nearest quad search ---
	Qh = Hm - 1
	Qw = Wm - 1
	NQ_layer = Qh * Qw
	if Qh <= 0 or Qw <= 0 or D < 2:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	xyz_det = xyz_lr.detach()
	v00 = xyz_det[:, :-1, :-1]  # (D, Qh, Qw, 3)
	v10 = xyz_det[:, 1:, :-1]
	v01 = xyz_det[:, :-1, 1:]
	v11 = xyz_det[:, 1:, 1:]
	v00_f = v00.reshape(D, NQ_layer, 3)
	v10_f = v10.reshape(D, NQ_layer, 3)
	v01_f = v01.reshape(D, NQ_layer, 3)
	v11_f = v11.reshape(D, NQ_layer, 3)

	P_exp = P.unsqueeze(1)  # (K, 1, 3)
	kidx = torch.arange(K, device=dev)

	nearest_qi = torch.zeros(K, D, dtype=torch.long, device=dev)
	u_per_d = torch.zeros(K, D, device=dev, dtype=dt)
	v_per_d = torch.zeros(K, D, device=dev, dtype=dt)

	for d in range(D):
		v00_d = v00_f[d].unsqueeze(0)  # (1, NQ_layer, 3)
		v10_d = v10_f[d].unsqueeze(0)
		v01_d = v01_f[d].unsqueeze(0)
		v11_d = v11_f[d].unsqueeze(0)

		e1 = v10_d - v00_d
		e2 = v01_d - v00_d
		g = P_exp - v00_d

		e1e1 = (e1 * e1).sum(-1)
		e1e2 = (e1 * e2).sum(-1)
		e2e2 = (e2 * e2).sum(-1)
		ge1 = (g * e1).sum(-1)
		ge2 = (g * e2).sum(-1)

		det = e1e1 * e2e2 - e1e2 * e1e2
		det_safe = det + (det.abs() < 1e-20).float() * 1e-20
		u_d = ((ge1 * e2e2 - ge2 * e1e2) / det_safe).clamp(0.0, 1.0)
		v_d = ((ge2 * e1e1 - ge1 * e1e2) / det_safe).clamp(0.0, 1.0)

		u1 = u_d.unsqueeze(-1)
		v1 = v_d.unsqueeze(-1)
		Q_closest = (v00_d * (1 - u1) * (1 - v1) + v10_d * u1 * (1 - v1) +
					 v01_d * (1 - u1) * v1 + v11_d * u1 * v1)

		diff = P_exp - Q_closest
		dist_sq = (diff * diff).sum(-1)  # (K, NQ_layer)

		best = dist_sq.argmin(dim=1)  # (K,)
		nearest_qi[:, d] = best
		u_per_d[:, d] = u_d[kidx, best]
		v_per_d[:, d] = v_d[kidx, best]

	# --- Step 4: per-layer signed distances, edge checks, bracket pair ---
	def _edge_check(va, vb, na, nb):
		edge = vb - va
		wn_a = torch.cross(na, edge, dim=-1)
		wn_b = torch.cross(nb, edge, dim=-1)
		side_a = ((P - va) * wn_a).sum(-1)
		side_b = ((P - vb) * wn_b).sum(-1)
		return (side_a >= 0) | (side_b >= 0)

	signed_dist_per_d = torch.zeros(K, D, device=dev, dtype=dt)
	edge_valid_per_d = torch.zeros(K, D, dtype=torch.bool, device=dev)

	for d in range(D):
		qi = nearest_qi[:, d]
		h_idx = qi // Qw
		w_idx = qi % Qw

		c00 = v00_f[d][qi]; c10 = v10_f[d][qi]
		c01 = v01_f[d][qi]; c11 = v11_f[d][qi]

		ud = u_per_d[:, d].unsqueeze(-1)
		vd = v_per_d[:, d].unsqueeze(-1)
		Q_pos = (c00 * (1 - ud) * (1 - vd) + c10 * ud * (1 - vd) +
				 c01 * (1 - ud) * vd + c11 * ud * vd)

		nn00 = n_unit[d, h_idx, w_idx]
		nn10 = n_unit[d, h_idx + 1, w_idx]
		nn01 = n_unit[d, h_idx, w_idx + 1]
		nn11 = n_unit[d, h_idx + 1, w_idx + 1]

		n_interp = (nn00 * (1 - ud) * (1 - vd) + nn10 * ud * (1 - vd) +
					nn01 * (1 - ud) * vd + nn11 * ud * vd)
		n_interp = n_interp / (n_interp.norm(dim=-1, keepdim=True) + 1e-12)

		signed_dist_per_d[:, d] = ((P - Q_pos) * n_interp).sum(-1)

		ok_e0 = _edge_check(c00, c10, nn00, nn10)
		ok_e1 = _edge_check(c10, c11, nn10, nn11)
		ok_e2 = _edge_check(c11, c01, nn11, nn01)
		ok_e3 = _edge_check(c01, c00, nn01, nn00)
		edge_valid_per_d[:, d] = ok_e0 & ok_e1 & ok_e2 & ok_e3

	bracket_d = torch.zeros(K, device=dev, dtype=torch.long)
	bracket_valid = torch.zeros(K, dtype=torch.bool, device=dev)
	bracket_score = torch.full((K,), float('inf'), device=dev, dtype=dt)

	for d in range(D - 1):
		sd_lo = signed_dist_per_d[:, d]
		sd_hi = signed_dist_per_d[:, d + 1]
		between = (sd_lo * sd_hi) < 0
		both_valid = edge_valid_per_d[:, d] & edge_valid_per_d[:, d + 1]
		pair_ok = between & both_valid
		score = sd_lo.abs() + sd_hi.abs()
		better = pair_ok & (score < bracket_score)
		bracket_d[better] = d
		bracket_valid[better] = True
		bracket_score[better] = score[better]

	valid = bracket_valid

	# --- Step 5: winding number observation (with grad) ---
	bd = bracket_d
	bd1 = bd + 1

	qi_lo = nearest_qi[kidx, bd]
	qi_hi = nearest_qi[kidx, bd1]
	h_lo = qi_lo // Qw; w_lo = qi_lo % Qw
	h_hi = qi_hi // Qw; w_hi = qi_hi % Qw

	v00_g_lo = xyz_lr[bd, h_lo, w_lo]
	v10_g_lo = xyz_lr[bd, h_lo + 1, w_lo]
	v01_g_lo = xyz_lr[bd, h_lo, w_lo + 1]
	v11_g_lo = xyz_lr[bd, h_lo + 1, w_lo + 1]

	v00_g_hi = xyz_lr[bd1, h_hi, w_hi]
	v10_g_hi = xyz_lr[bd1, h_hi + 1, w_hi]
	v01_g_hi = xyz_lr[bd1, h_hi, w_hi + 1]
	v11_g_hi = xyz_lr[bd1, h_hi + 1, w_hi + 1]

	u_lo = u_per_d[kidx, bd].detach().unsqueeze(-1)
	v_lo = v_per_d[kidx, bd].detach().unsqueeze(-1)
	u_hi = u_per_d[kidx, bd1].detach().unsqueeze(-1)
	v_hi = v_per_d[kidx, bd1].detach().unsqueeze(-1)

	Q_lo = (v00_g_lo * (1 - u_lo) * (1 - v_lo) + v10_g_lo * u_lo * (1 - v_lo) +
			v01_g_lo * (1 - u_lo) * v_lo + v11_g_lo * u_lo * v_lo)
	Q_hi = (v00_g_hi * (1 - u_hi) * (1 - v_hi) + v10_g_hi * u_hi * (1 - v_hi) +
			v01_g_hi * (1 - u_hi) * v_hi + v11_g_hi * u_hi * v_hi)

	n_lo_00 = n_unit[bd, h_lo, w_lo]
	n_lo_10 = n_unit[bd, h_lo + 1, w_lo]
	n_lo_01 = n_unit[bd, h_lo, w_lo + 1]
	n_lo_11 = n_unit[bd, h_lo + 1, w_lo + 1]
	n_lo_i = (n_lo_00 * (1 - u_lo) * (1 - v_lo) + n_lo_10 * u_lo * (1 - v_lo) +
			  n_lo_01 * (1 - u_lo) * v_lo + n_lo_11 * u_lo * v_lo)
	n_lo_i = n_lo_i / (n_lo_i.norm(dim=-1, keepdim=True) + 1e-12)

	n_hi_00 = n_unit[bd1, h_hi, w_hi]
	n_hi_10 = n_unit[bd1, h_hi + 1, w_hi]
	n_hi_01 = n_unit[bd1, h_hi, w_hi + 1]
	n_hi_11 = n_unit[bd1, h_hi + 1, w_hi + 1]
	n_hi_i = (n_hi_00 * (1 - u_hi) * (1 - v_hi) + n_hi_10 * u_hi * (1 - v_hi) +
			  n_hi_01 * (1 - u_hi) * v_hi + n_hi_11 * u_hi * v_hi)
	n_hi_i = n_hi_i / (n_hi_i.norm(dim=-1, keepdim=True) + 1e-12)

	dist_d = ((P - Q_lo) * n_lo_i).sum(-1)
	dist_d1 = ((P - Q_hi) * n_hi_i).sum(-1)

	denom = dist_d - dist_d1
	denom_safe = denom + (denom.abs() < 1e-12).float() * 1e-12
	frac = (dist_d / denom_safe).clamp(0.0, 1.0)

	obs = bd.to(dtype=dt) + frac

	# --- Step 6: collection-coupled +/- winda error ---
	err = torch.full((K,), float("nan"), device=dev, dtype=dt)
	avg_per_point = torch.full((K,), float("nan"), device=dev, dtype=dt)
	uc = torch.unique(col)
	for cid in uc.tolist():
		m = (col == int(cid)) & valid
		if not bool(m.any()):
			continue
		obs_m = obs[m]
		wa_m = winda[m]
		avg_pos = (obs_m - wa_m).mean()
		err_pos = obs_m - (avg_pos + wa_m)
		mse_pos = (err_pos * err_pos).mean()
		avg_neg = (obs_m + wa_m).mean()
		err_neg = obs_m - (avg_neg - wa_m)
		mse_neg = (err_neg * err_neg).mean()
		use_neg = bool((mse_neg < mse_pos).item())
		err[m] = err_neg if use_neg else err_pos
		avg_per_point[col == int(cid)] = float(avg_neg.item()) if use_neg else float(avg_pos.item())

	point_valid = torch.isfinite(err)

	if not bool(point_valid.any()):
		if dbg:
			print("[corr] no valid points after collection coupling")
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	# --- Step 7: loss ---
	err_sq = torch.where(point_valid, err * err, torch.zeros_like(err))
	loss = err_sq[point_valid].mean()

	if dbg:
		print(f"[corr] loss={loss.item():.6f}, valid={point_valid.sum().item()}/{K}, "
			  f"rms_err={loss.item()**0.5:.4f}")

	# --- Build results dict for feedback (JSON-serializable) ---
	_last_results = _build_results(
		obs=obs, avg=avg_per_point, err=err,
		pt_ids=pt_ids, col=col, pts=pts, valid=point_valid,
	)

	mask = point_valid.to(dtype=dt)
	return loss, (err_sq,), (mask,)


def get_last_results() -> dict | None:
	"""Return the last corr_points_results dict (for saving to checkpoint)."""
	return _last_results


def print_summary():
	"""Print a one-line corr summary (called at end of opt)."""
	if _last_results is None:
		return
	pts = _last_results.get("points", {})
	n_valid = sum(1 for p in pts.values() if p.get("valid"))
	errs = [p["winding_err"] for p in pts.values() if p.get("winding_err") is not None]
	rms = (sum(e * e for e in errs) / len(errs)) ** 0.5 if errs else float("nan")
	print(f"[corr] final: valid={n_valid}/{len(pts)}, rms_err={rms:.4f}")


def _build_results(
	*, obs: torch.Tensor, avg: torch.Tensor, err: torch.Tensor,
	pt_ids: torch.Tensor, col: torch.Tensor, pts: torch.Tensor,
	valid: torch.Tensor,
) -> dict:
	"""Build JSON-serializable dict of per-point winding results."""
	result: dict = {"points": {}, "collection_avgs": {}}
	K = int(obs.shape[0])
	for i in range(K):
		pid = int(pt_ids[i].item())
		cid = int(col[i].item())
		o = float(obs[i].item())
		e = float(err[i].item()) if bool(valid[i]) else None
		a = float(avg[i].item())
		entry: dict = {
			"collection_id": cid,
			"p": [round(float(pts[i, 0].item()), 2),
				  round(float(pts[i, 1].item()), 2),
				  round(float(pts[i, 2].item()), 2)],
		}
		entry["winding_obs"] = round(o, 6) if math.isfinite(o) else None
		entry["winding_err"] = round(e, 6) if e is not None and math.isfinite(e) else None
		entry["valid"] = bool(valid[i])
		result["points"][str(pid)] = entry
	# Per-collection averages
	uc = torch.unique(col)
	for cid_t in uc.tolist():
		cid_int = int(cid_t)
		mask = (col == cid_int) & valid
		if mask.any():
			v = float(avg[mask][0].item())
			if math.isfinite(v):
				result["collection_avgs"][str(cid_int)] = round(v, 6)
	return result


# ---------------------------------------------------------------------------
# Snap-to-surface corr loss
# ---------------------------------------------------------------------------

def _bilinear_project(P: torch.Tensor, v00: torch.Tensor, v10: torch.Tensor,
					  v01: torch.Tensor, v11: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Project points P onto bilinear quads defined by corners.

	P: (K, NQ, 3) or (K, 3) broadcastable
	v00/v10/v01/v11: (K, NQ, 3) or (1, NQ, 3) broadcastable
	Returns (u, v) each (K, NQ) clamped to [0, 1].
	"""
	e1 = v10 - v00
	e2 = v01 - v00
	g = P - v00
	e1e1 = (e1 * e1).sum(-1)
	e1e2 = (e1 * e2).sum(-1)
	e2e2 = (e2 * e2).sum(-1)
	ge1 = (g * e1).sum(-1)
	ge2 = (g * e2).sum(-1)
	det = e1e1 * e2e2 - e1e2 * e1e2
	det_safe = det + (det.abs() < 1e-20).float() * 1e-20
	u = ((ge1 * e2e2 - ge2 * e1e2) / det_safe).clamp(0.0, 1.0)
	v = ((ge2 * e1e1 - ge1 * e1e2) / det_safe).clamp(0.0, 1.0)
	return u, v


def _bilinear_interp(v00: torch.Tensor, v10: torch.Tensor,
					 v01: torch.Tensor, v11: torch.Tensor,
					 u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	"""Bilinear interpolation: (1-u)(1-v)*v00 + u(1-v)*v10 + (1-u)*v*v01 + u*v*v11.

	u, v: (...) or (..., 1) for broadcasting with 3D vectors.
	"""
	if u.dim() < v00.dim():
		u = u.unsqueeze(-1)
		v = v.unsqueeze(-1)
	return v00 * (1 - u) * (1 - v) + v10 * u * (1 - v) + v01 * (1 - u) * v + v11 * u * v


def _snap_brute_force(P: torch.Tensor, xyz_det: torch.Tensor, wind_d: torch.Tensor,
					  conn_prev: torch.Tensor, conn_next: torch.Tensor,
					  mask_conn: torch.Tensor, Qh: int, Qw: int,
					  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Brute-force nearest quad search for snap mode.

	P: (K, 3) point positions
	xyz_det: (D, Hm, Wm, 3) detached mesh
	wind_d: (K,) int — target winding per point
	conn_prev/next: (D, Hm, Wm, 3) — conn direction unit vectors
	mask_conn: (D, 1, Hm, Wm, 3) — connection validity
	Returns: (h_idx, w_idx, u, v, direction) each (K,)
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	NQ = Qh * Qw

	best_h = torch.zeros(K, dtype=torch.long, device=dev)
	best_w = torch.zeros(K, dtype=torch.long, device=dev)
	best_u = torch.zeros(K, dtype=dt, device=dev)
	best_v = torch.zeros(K, dtype=dt, device=dev)
	best_dir = torch.zeros(K, dtype=torch.long, device=dev)
	best_dist = torch.full((K,), float("inf"), dtype=dt, device=dev)

	unique_d = torch.unique(wind_d)
	for d_val in unique_d.tolist():
		d = int(d_val)
		if d < 0 or d >= xyz_det.shape[0]:
			continue
		pmask = (wind_d == d)
		if not pmask.any():
			continue
		P_sub = P[pmask]  # (Ks, 3)
		Ks = P_sub.shape[0]

		# Get quads on layer d
		v00 = xyz_det[d, :-1, :-1].reshape(NQ, 3).unsqueeze(0)  # (1, NQ, 3)
		v10 = xyz_det[d, 1:, :-1].reshape(NQ, 3).unsqueeze(0)
		v01 = xyz_det[d, :-1, 1:].reshape(NQ, 3).unsqueeze(0)
		v11 = xyz_det[d, 1:, 1:].reshape(NQ, 3).unsqueeze(0)

		P_exp = P_sub.unsqueeze(1)  # (Ks, 1, 3)
		u_all, v_all = _bilinear_project(P_exp, v00, v10, v01, v11)  # (Ks, NQ)
		Q = _bilinear_interp(v00, v10, v01, v11, u_all, v_all)  # (Ks, NQ, 3)

		# Conn direction vectors at quad corners
		# For each quad (h, w), interpolate conn vectors at (u, v)
		h_q = torch.arange(Qh, device=dev).unsqueeze(1).expand(Qh, Qw).reshape(NQ)
		w_q = torch.arange(Qw, device=dev).unsqueeze(0).expand(Qh, Qw).reshape(NQ)

		for dir_idx, conn_dir in enumerate([conn_prev, conn_next]):
			# mask_conn indices: 0=prev, 2=next
			mc_idx = 0 if dir_idx == 0 else 2
			# Check validity at quad corners
			mc = mask_conn[d, 0, :, :, mc_idx]  # (Hm, Wm)
			mc_q = mc[:-1, :-1].reshape(NQ)  # (NQ,) — validity at v00 corner
			valid_q = (mc_q > 0).unsqueeze(0).expand(Ks, NQ)  # (Ks, NQ)

			# Interpolate conn direction at (u, v) within each quad
			cn00 = conn_dir[d, :-1, :-1].reshape(NQ, 3).unsqueeze(0)  # (1, NQ, 3)
			cn10 = conn_dir[d, 1:, :-1].reshape(NQ, 3).unsqueeze(0)
			cn01 = conn_dir[d, :-1, 1:].reshape(NQ, 3).unsqueeze(0)
			cn11 = conn_dir[d, 1:, 1:].reshape(NQ, 3).unsqueeze(0)
			n_interp = _bilinear_interp(cn00, cn10, cn01, cn11, u_all, v_all)  # (Ks, NQ, 3)
			n_len = n_interp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
			n_unit = n_interp / n_len

			dist_signed = ((P_exp - Q) * n_unit).sum(-1)  # (Ks, NQ)
			dist_abs = dist_signed.abs()
			dist_abs = torch.where(valid_q, dist_abs, torch.full_like(dist_abs, float("inf")))

			min_dist, min_qi = dist_abs.min(dim=1)  # (Ks,)
			# Update best for these points
			pidx = pmask.nonzero(as_tuple=True)[0]  # global indices
			better = min_dist < best_dist[pidx]
			if better.any():
				bi = better.nonzero(as_tuple=True)[0]
				gi = pidx[bi]
				qi = min_qi[bi]
				best_h[gi] = h_q[qi]
				best_w[gi] = w_q[qi]
				best_u[gi] = u_all[bi, qi]
				best_v[gi] = v_all[bi, qi]
				best_dir[gi] = dir_idx
				best_dist[gi] = min_dist[bi]

	return best_h, best_w, best_u, best_v, best_dir


def _snap_update_anchors(P: torch.Tensor, xyz_det: torch.Tensor, wind_d: torch.Tensor,
						 anchor_h: torch.Tensor, anchor_w: torch.Tensor,
						 anchor_u: torch.Tensor, anchor_v: torch.Tensor,
						 anchor_dir: torch.Tensor,
						 conn_prev: torch.Tensor, conn_next: torch.Tensor,
						 Qh: int, Qw: int,
						 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Local anchor update: re-project points onto current quads.

	If (u, v) goes out of [0, 1], shift to adjacent quad.
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	D, Hm, Wm, _ = xyz_det.shape

	h = anchor_h.clone()
	w = anchor_w.clone()
	d = wind_d

	# Gather current quad corners (detached)
	v00 = xyz_det[d, h, w]          # (K, 3)
	v10 = xyz_det[d, h + 1, w]
	v01 = xyz_det[d, h, w + 1]
	v11 = xyz_det[d, h + 1, w + 1]

	# Re-project
	u_new, v_new = _bilinear_project(P, v00, v10, v01, v11)  # (K,)

	# Shift if out of bounds
	h_shift = torch.zeros_like(h)
	w_shift = torch.zeros_like(w)
	h_shift[u_new > 1.0] = 1
	h_shift[u_new < 0.0] = -1
	w_shift[v_new > 1.0] = 1
	w_shift[v_new < 0.0] = -1

	h = (h + h_shift).clamp(0, Qh - 1)
	w = (w + w_shift).clamp(0, Qw - 1)

	# Re-project on shifted quads for points that moved
	moved = (h_shift != 0) | (w_shift != 0)
	if moved.any():
		mi = moved.nonzero(as_tuple=True)[0]
		v00_m = xyz_det[d[mi], h[mi], w[mi]]
		v10_m = xyz_det[d[mi], h[mi] + 1, w[mi]]
		v01_m = xyz_det[d[mi], h[mi], w[mi] + 1]
		v11_m = xyz_det[d[mi], h[mi] + 1, w[mi] + 1]
		u_m, v_m = _bilinear_project(P[mi], v00_m, v10_m, v01_m, v11_m)
		u_new[mi] = u_m
		v_new[mi] = v_m

	u_new = u_new.clamp(0.0, 1.0)
	v_new = v_new.clamp(0.0, 1.0)

	# Re-check direction: pick the one with smaller |dist|
	v00 = xyz_det[d, h, w]
	v10 = xyz_det[d, h + 1, w]
	v01 = xyz_det[d, h, w + 1]
	v11 = xyz_det[d, h + 1, w + 1]
	Q = _bilinear_interp(v00, v10, v01, v11, u_new, v_new)  # (K, 3)
	offset = P - Q  # (K, 3)

	new_dir = torch.zeros(K, dtype=torch.long, device=dev)  # start with prev=0
	for dir_idx, conn_dir in enumerate([conn_prev, conn_next]):
		cn00 = conn_dir[d, h, w]
		cn10 = conn_dir[d, h + 1, w]
		cn01 = conn_dir[d, h, w + 1]
		cn11 = conn_dir[d, h + 1, w + 1]
		n = _bilinear_interp(cn00, cn10, cn01, cn11, u_new, v_new)
		n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
		dist = (offset * n).sum(-1).abs()
		if dir_idx == 0:
			best_dist = dist.clone()
		else:
			use_this = dist < best_dist
			new_dir[use_this] = dir_idx

	return h, w, u_new, v_new, new_dir


def _corr_snap_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Snap-to-surface corr loss: penalize distance from surface to known-winding points."""
	global _dbg_call_count, _last_results
	global _snap_anchors_h, _snap_anchors_w, _snap_anchors_u, _snap_anchors_v
	global _snap_anchors_dir, _snap_initialized

	_dbg_call_count += 1
	dbg = (_dbg_call_count <= 2)

	dev = res.xyz_lr.device
	dt = res.xyz_lr.dtype

	pts_c = res.data.corr_points
	if pts_c is None or pts_c.points_xyz_winda.shape[0] == 0:
		if _dbg_call_count <= 2:
			print("[corr-snap] no correction points")
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	pts = pts_c.points_xyz_winda.to(device=dev, dtype=dt)   # (K, 4)
	col = pts_c.collection_idx.to(device=dev, dtype=torch.int64)
	pt_ids = pts_c.point_ids.to(device=dev, dtype=torch.int64)
	# wind_a (column 3) is the depth index from d.tif, stored as winding_annotation
	wind_d = pts[:, 3].round().to(torch.int64)  # (K,)
	K = int(pts.shape[0])
	P = pts[:, :3]  # (K, 3)

	xyz_lr = res.xyz_lr          # (D, Hm, Wm, 3) — has grad
	xy_conn = res.xy_conn        # (D, Hm, Wm, 3, 3) — [prev, self, next]
	mask_conn = res.mask_conn    # (D, 1, Hm, Wm, 3)
	D, Hm, Wm, _ = xyz_lr.shape
	Qh = Hm - 1
	Qw = Wm - 1

	if Qh <= 0 or Qw <= 0:
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	# Filter out points with invalid depth index
	valid_d = (wind_d >= 0) & (wind_d < D)
	if not valid_d.any():
		if dbg:
			print("[corr-snap] no points with valid depth index")
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	# Conn direction vectors (detached)
	xy_conn_det = xy_conn.detach()
	conn_prev = xy_conn_det[:, :, :, :, 0] - xy_conn_det[:, :, :, :, 1]  # (D, Hm, Wm, 3)
	conn_next = xy_conn_det[:, :, :, :, 2] - xy_conn_det[:, :, :, :, 1]  # (D, Hm, Wm, 3)
	# Normalize
	conn_prev = conn_prev / (conn_prev.norm(dim=-1, keepdim=True) + 1e-12)
	conn_next = conn_next / (conn_next.norm(dim=-1, keepdim=True) + 1e-12)

	xyz_det = xyz_lr.detach()

	# --- Anchor init or update ---
	if not _snap_initialized or _snap_anchors_h is None or _snap_anchors_h.shape[0] != K:
		# Brute force initial search
		if dbg:
			print(f"[corr-snap] brute-force init: K={K} points, mesh=({D},{Hm},{Wm})")
		_snap_anchors_h, _snap_anchors_w, _snap_anchors_u, _snap_anchors_v, _snap_anchors_dir = \
			_snap_brute_force(P, xyz_det, wind_d, conn_prev, conn_next, mask_conn.detach(), Qh, Qw)
		_snap_initialized = True
	else:
		# Local update
		_snap_anchors_h, _snap_anchors_w, _snap_anchors_u, _snap_anchors_v, _snap_anchors_dir = \
			_snap_update_anchors(P, xyz_det, wind_d,
								_snap_anchors_h, _snap_anchors_w,
								_snap_anchors_u, _snap_anchors_v,
								_snap_anchors_dir,
								conn_prev, conn_next, Qh, Qw)

	# --- Loss computation (with grad) ---
	h = _snap_anchors_h
	w = _snap_anchors_w
	u = _snap_anchors_u.detach()
	v = _snap_anchors_v.detach()
	d = wind_d

	# Check valid: d in range AND anchor found (best_dist < inf)
	anchor_valid = valid_d

	if not anchor_valid.any():
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	# Gather quad corners WITH grad
	v00_g = xyz_lr[d, h, w]          # (K, 3)
	v10_g = xyz_lr[d, h + 1, w]
	v01_g = xyz_lr[d, h, w + 1]
	v11_g = xyz_lr[d, h + 1, w + 1]

	Q = _bilinear_interp(v00_g, v10_g, v01_g, v11_g, u, v)  # (K, 3) — has grad

	# Pick conn direction (detached, normalized)
	use_next = (_snap_anchors_dir == 1)
	cn00_p = conn_prev[d, h, w]
	cn10_p = conn_prev[d, h + 1, w]
	cn01_p = conn_prev[d, h, w + 1]
	cn11_p = conn_prev[d, h + 1, w + 1]
	n_prev_i = _bilinear_interp(cn00_p, cn10_p, cn01_p, cn11_p, u, v)
	n_prev_i = n_prev_i / (n_prev_i.norm(dim=-1, keepdim=True) + 1e-12)

	cn00_n = conn_next[d, h, w]
	cn10_n = conn_next[d, h + 1, w]
	cn01_n = conn_next[d, h, w + 1]
	cn11_n = conn_next[d, h + 1, w + 1]
	n_next_i = _bilinear_interp(cn00_n, cn10_n, cn01_n, cn11_n, u, v)
	n_next_i = n_next_i / (n_next_i.norm(dim=-1, keepdim=True) + 1e-12)

	n_conn = torch.where(use_next.unsqueeze(-1), n_next_i, n_prev_i)  # (K, 3)

	# Signed distance along conn normal
	signed_dist = ((P - Q) * n_conn).sum(-1)  # (K,)

	err = torch.where(anchor_valid, signed_dist, torch.zeros_like(signed_dist))
	err_sq = err * err
	loss = err_sq[anchor_valid].mean()

	if dbg:
		rms = loss.item() ** 0.5
		print(f"[corr-snap] loss={loss.item():.6f}, valid={anchor_valid.sum().item()}/{K}, "
			  f"rms_dist={rms:.2f} voxels")

	# --- Build results ---
	_last_results = _build_snap_results(
		wind_d=wind_d, signed_dist=signed_dist,
		pt_ids=pt_ids, col=col, pts=pts, valid=anchor_valid,
	)

	mask = anchor_valid.to(dtype=dt)
	return loss, (err_sq,), (mask,)


def _build_snap_results(
	*, wind_d: torch.Tensor, signed_dist: torch.Tensor,
	pt_ids: torch.Tensor, col: torch.Tensor, pts: torch.Tensor,
	valid: torch.Tensor,
) -> dict:
	"""Build JSON-serializable dict of per-point snap results."""
	result: dict = {"points": {}}
	K = int(pts.shape[0])
	for i in range(K):
		pid = int(pt_ids[i].item())
		cid = int(col[i].item())
		wd = int(wind_d[i].item())
		sd = float(signed_dist[i].item()) if bool(valid[i]) else None
		entry: dict = {
			"collection_id": cid,
			"p": [round(float(pts[i, 0].item()), 2),
				  round(float(pts[i, 1].item()), 2),
				  round(float(pts[i, 2].item()), 2)],
			"winding_obs": wd if wd >= 0 else None,
			"winding_err": round(sd, 4) if sd is not None and math.isfinite(sd) else None,
			"valid": bool(valid[i]),
		}
		result["points"][str(pid)] = entry
	return result
