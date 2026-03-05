from __future__ import annotations

import math

import torch

import model as fit_model

_dbg_call_count = 0
_last_results: dict | None = None


def corr_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""3D correction point loss: point-to-quad nearest surface, collection-coupled winding error."""
	global _dbg_call_count, _last_results
	_dbg_call_count += 1
	dbg = (_dbg_call_count <= 2) or (_dbg_call_count % 100 == 0)

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
			if dbg:
				n_total = (col == int(cid)).sum().item()
				print(f"[corr] collection {cid}: {n_total} pts, 0 valid — skipped")
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
		if dbg:
			n_total = (col == int(cid)).sum().item()
			sign_str = "neg" if use_neg else "pos"
			avg_val = float(avg_neg.item()) if use_neg else float(avg_pos.item())
			mse_val = float(mse_neg.item()) if use_neg else float(mse_pos.item())
			print(f"[corr] collection {cid}: {n_total} pts, {m.sum().item()} valid, "
				  f"sign={sign_str}, avg_offset={avg_val:.3f}, mse={mse_val:.6f}")

	point_valid = torch.isfinite(err)

	# --- Debug per-point table ---
	if dbg:
		print(f"[corr] per-point details:")
		print(f"  {'pid':>5s}  {'col':>3s}  {'winda':>8s}  "
			  f"{'pt_x':>9s} {'pt_y':>9s} {'pt_z':>9s}  "
			  f"{'brk':>5s}  {'sd_lo':>8s} {'sd_hi':>8s}  "
			  f"{'frac':>5s}  {'obs':>9s}  {'avg':>9s}  {'err':>9s}  {'ok':>2s}")
		for i in range(K):
			obs_i = obs[i].item()
			avg_i = avg_per_point[i].item()
			err_i = err[i].item() if point_valid[i] else float("nan")
			bd_i = bracket_d[i].item()
			brk_str = f"{bd_i},{bd_i+1}" if bracket_valid[i] else "---"
			sd_lo_i = signed_dist_per_d[i, bd_i].item() if bracket_valid[i] else float("nan")
			sd_hi_i = signed_dist_per_d[i, bd_i + 1].item() if bracket_valid[i] else float("nan")
			frac_i = frac[i].item()
			print(f"  {pt_ids[i].item():5d}  {col[i].item():3d}  {winda[i].item():8.3f}  "
				  f"{P[i, 0].item():9.1f} {P[i, 1].item():9.1f} {P[i, 2].item():9.1f}  "
				  f"{brk_str:>5s}  {sd_lo_i:8.2f} {sd_hi_i:8.2f}  "
				  f"{frac_i:5.3f}  {obs_i:9.3f}  "
				  f"{avg_i:9.3f}  {err_i:9.3f}  "
				  f"{'Y' if valid[i] else 'N':>2s}")

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
