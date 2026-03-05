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

	# --- Step 3: point-to-quad distance search ---
	Qd = D
	Qh = Hm - 1
	Qw = Wm - 1
	if Qh <= 0 or Qw <= 0:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	# Quad corners (detached for nearest search)
	xyz_det = xyz_lr.detach()
	v00 = xyz_det[:, :-1, :-1]  # (D, Qh, Qw, 3)
	v10 = xyz_det[:, 1:, :-1]
	v01 = xyz_det[:, :-1, 1:]
	v11 = xyz_det[:, 1:, 1:]

	NQ = Qd * Qh * Qw
	v00_f = v00.reshape(NQ, 3)
	v10_f = v10.reshape(NQ, 3)
	v01_f = v01.reshape(NQ, 3)
	v11_f = v11.reshape(NQ, 3)

	# Point-to-quad closest point via affine projection
	P_exp = P.unsqueeze(1)          # (K, 1, 3)
	v00_exp = v00_f.unsqueeze(0)    # (1, NQ, 3)
	v10_exp = v10_f.unsqueeze(0)
	v01_exp = v01_f.unsqueeze(0)
	v11_exp = v11_f.unsqueeze(0)

	e1 = v10_exp - v00_exp
	e2 = v01_exp - v00_exp
	g = P_exp - v00_exp

	e1e1 = (e1 * e1).sum(-1)
	e1e2 = (e1 * e2).sum(-1)
	e2e2 = (e2 * e2).sum(-1)
	ge1 = (g * e1).sum(-1)
	ge2 = (g * e2).sum(-1)

	det = e1e1 * e2e2 - e1e2 * e1e2
	det_safe = det + (det.abs() < 1e-20).float() * 1e-20
	u = ((ge1 * e2e2 - ge2 * e1e2) / det_safe).clamp(0.0, 1.0)
	v = ((ge2 * e1e1 - ge1 * e1e2) / det_safe).clamp(0.0, 1.0)

	u1 = u.unsqueeze(-1)
	v1 = v.unsqueeze(-1)
	Q_closest = (v00_exp * (1 - u1) * (1 - v1) + v10_exp * u1 * (1 - v1) +
				 v01_exp * (1 - u1) * v1 + v11_exp * u1 * v1)

	diff = P_exp - Q_closest
	dist_sq = (diff * diff).sum(-1)  # (K, NQ)

	nearest_idx = dist_sq.argmin(dim=1)  # (K,)

	nearest_d = nearest_idx // (Qh * Qw)
	rem = nearest_idx % (Qh * Qw)
	nearest_h = rem // Qw
	nearest_w = rem % Qw

	kidx = torch.arange(K, device=dev)
	u_near = u[kidx, nearest_idx]
	v_near = v[kidx, nearest_idx]
	nearest_dist = dist_sq[kidx, nearest_idx].sqrt()

	# --- Step 4: boundary checks ---
	n00 = n_unit[nearest_d, nearest_h, nearest_w]
	n10 = n_unit[nearest_d, nearest_h + 1, nearest_w]
	n01 = n_unit[nearest_d, nearest_h, nearest_w + 1]
	n11 = n_unit[nearest_d, nearest_h + 1, nearest_w + 1]

	c00 = v00_f[nearest_idx]
	c10 = v10_f[nearest_idx]
	c01 = v01_f[nearest_idx]
	c11 = v11_f[nearest_idx]

	def _edge_check(va: torch.Tensor, vb: torch.Tensor, na: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
		edge = vb - va
		wn_a = torch.cross(na, edge, dim=-1)
		wn_b = torch.cross(nb, edge, dim=-1)
		side_a = ((P - va) * wn_a).sum(-1)
		side_b = ((P - vb) * wn_b).sum(-1)
		return (side_a >= 0) | (side_b >= 0)

	ok_e0 = _edge_check(c00, c10, n00, n10)
	ok_e1 = _edge_check(c10, c11, n10, n11)
	ok_e2 = _edge_check(c11, c01, n11, n01)
	ok_e3 = _edge_check(c01, c00, n01, n00)

	quad_n = 0.25 * (n00 + n10 + n01 + n11)
	quad_center = 0.25 * (c00 + c10 + c01 + c11)
	side_surf = ((P - quad_center) * quad_n).sum(-1)
	ok_surf = side_surf.abs() < (nearest_dist + 1e-6) * 10.0

	valid = ok_e0 & ok_e1 & ok_e2 & ok_e3 & ok_surf

	# --- Step 5: signed normal distance observation (with grad) ---
	v00_g = xyz_lr[nearest_d, nearest_h, nearest_w]
	v10_g = xyz_lr[nearest_d, nearest_h + 1, nearest_w]
	v01_g = xyz_lr[nearest_d, nearest_h, nearest_w + 1]
	v11_g = xyz_lr[nearest_d, nearest_h + 1, nearest_w + 1]

	u_d = u_near.detach().unsqueeze(-1)
	v_d = v_near.detach().unsqueeze(-1)
	Q_g = (v00_g * (1 - u_d) * (1 - v_d) + v10_g * u_d * (1 - v_d) +
		   v01_g * (1 - u_d) * v_d + v11_g * u_d * v_d)

	n_interp = (n00 * (1 - u_d) * (1 - v_d) + n10 * u_d * (1 - v_d) +
				n01 * (1 - u_d) * v_d + n11 * u_d * v_d)
	n_interp = n_interp / (n_interp.norm(dim=-1, keepdim=True) + 1e-12)

	obs = ((P - Q_g) * n_interp).sum(-1)  # (K,)

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
			  f"{'quad(d,h,w)':>12s}  {'u':>5s} {'v':>5s}  "
			  f"{'dist':>8s}  {'obs':>9s}  {'avg':>9s}  {'err':>9s}  "
			  f"{'e0':>2s} {'e1':>2s} {'e2':>2s} {'e3':>2s} {'sf':>2s} {'ok':>2s}")
		for i in range(K):
			obs_i = obs[i].item()
			avg_i = avg_per_point[i].item()
			err_i = err[i].item() if point_valid[i] else float("nan")
			print(f"  {pt_ids[i].item():5d}  {col[i].item():3d}  {winda[i].item():8.3f}  "
				  f"{P[i, 0].item():9.1f} {P[i, 1].item():9.1f} {P[i, 2].item():9.1f}  "
				  f"({nearest_d[i].item():2d},{nearest_h[i].item():3d},{nearest_w[i].item():3d})  "
				  f"{u_near[i].item():5.3f} {v_near[i].item():5.3f}  "
				  f"{nearest_dist[i].item():8.2f}  {obs_i:9.3f}  "
				  f"{avg_i:9.3f}  {err_i:9.3f}  "
				  f"{'Y' if ok_e0[i] else 'N':>2s} {'Y' if ok_e1[i] else 'N':>2s} "
				  f"{'Y' if ok_e2[i] else 'N':>2s} {'Y' if ok_e3[i] else 'N':>2s} "
				  f"{'Y' if ok_surf[i] else 'N':>2s} {'Y' if valid[i] else 'N':>2s}")

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
