from __future__ import annotations

import math

import torch

import fit_data
import model as fit_model

_dbg_call_count = 0
_last_results: dict | None = None
_corr_mode: str = "winding"

# Snap-mode anchor state (persists across calls)
_snap_anchors_h: torch.Tensor | None = None
_snap_anchors_w: torch.Tensor | None = None
_snap_anchors_u: torch.Tensor | None = None
_snap_anchors_v: torch.Tensor | None = None
_snap_anchors_dir: torch.Tensor | None = None  # 0=prev, 1=next
_snap_initialized: bool = False

# Winding-mode anchor state (persists across calls)
# Correspondence indices: 0=closest_low, 1=closest_up, 2=avg_low, 3=avg_up
_wind_anchors_d: torch.Tensor | None = None      # (K, 4) int — depth layer
_wind_anchors_h: torch.Tensor | None = None      # (K, 4) int — quad row
_wind_anchors_w: torch.Tensor | None = None      # (K, 4) int — quad col
_wind_anchors_valid: torch.Tensor | None = None   # (K, 4) bool — per-anchor validity
_wind_initialized: bool = False
_wind_target_per_point: torch.Tensor | None = None  # (K,) float — cached target winding


def set_corr_mode(mode: str) -> None:
	global _corr_mode
	assert mode in ("legacy", "snap", "winding"), f"Unknown corr mode: {mode}"
	_corr_mode = mode


def set_snap_mode(enabled: bool) -> None:
	"""Backward compat: 0=legacy, 1=snap."""
	set_corr_mode("snap" if enabled else "legacy")


def corr_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Dispatch to winding, snap, or legacy corr loss."""
	if _corr_mode == "winding":
		return _corr_winding_loss(res=res)
	if _corr_mode == "snap":
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


def print_detail(label: str = "") -> None:
	"""Print per-point corr detail: position, winding obs/target, anchors, validity."""
	tag = f"[corr-detail] {label}" if label else "[corr-detail]"
	if _last_results is None:
		print(f"{tag} no results yet")
		return

	pts = _last_results.get("points", {})
	col_avgs = _last_results.get("collection_avgs", {})
	if not pts:
		print(f"{tag} no points")
		return

	print(f"{tag} {len(pts)} points, mode={_corr_mode}")

	# Collection averages
	for cid, avg in sorted(col_avgs.items(), key=lambda x: int(x[0])):
		print(f"  collection {cid}: avg_winding={avg}")

	# Per-point table
	print(f"  {'pid':>6s}  {'col':>4s}  {'pos':>26s}  {'w_obs':>8s}  {'w_tgt':>8s}  {'w_err':>8s}  {'valid':>5s}", end="")
	# Anchor columns (winding mode only)
	has_anchors = _wind_anchors_d is not None and _corr_mode == "winding"
	if has_anchors:
		print(f"  {'cl_lo':>8s}  {'cl_up':>8s}  {'av_lo':>8s}  {'av_up':>8s}", end="")
	print()

	# Sort by point ID for stable output
	sorted_pts = sorted(pts.items(), key=lambda x: int(x[0]))
	# Build pid -> anchor-row mapping.  _build_winding_results iterates range(K)
	# using pt_ids[i] as dict key, so insertion order == tensor row order.
	pid_to_idx: dict[int, int] = {}
	if has_anchors:
		for idx, pid in enumerate(pts.keys()):
			pid_to_idx[int(pid)] = idx

	for pid, p in sorted_pts:
		pos = p.get("p", [0, 0, 0])
		pos_s = f"({pos[0]:8.1f},{pos[1]:8.1f},{pos[2]:8.1f})"
		w_obs = p.get("winding_obs")
		w_tgt = p.get("winding_target")
		w_err = p.get("winding_err")
		valid = p.get("valid", False)
		print(f"  {pid:>6s}  {p.get('collection_id', '?'):>4}  {pos_s}  "
			  f"{_fmt(w_obs):>8s}  {_fmt(w_tgt):>8s}  {_fmt(w_err):>8s}  "
			  f"{'  yes' if valid else '   no':>5s}", end="")
		if has_anchors:
			idx = pid_to_idx.get(int(pid))
			if idx is not None and idx < _wind_anchors_d.shape[0]:
				for ci in range(4):
					v = bool(_wind_anchors_valid[idx, ci])
					if v:
						d = int(_wind_anchors_d[idx, ci])
						h = int(_wind_anchors_h[idx, ci])
						w = int(_wind_anchors_w[idx, ci])
						print(f"  {d:2d},{h:3d},{w:3d}", end="")
					else:
						print(f"  {'---':>8s}", end="")
			else:
				print(f"  {'?':>8s}" * 4, end="")
		print()


def _fmt(v) -> str:
	if v is None:
		return "---"
	return f"{v:.4f}"


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

	h_q = torch.arange(Qh, device=dev).unsqueeze(1).expand(Qh, Qw).reshape(NQ)
	w_q = torch.arange(Qw, device=dev).unsqueeze(0).expand(Qh, Qw).reshape(NQ)

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

		# Euclidean distance to quad surface point
		diff = P_exp - Q  # (Ks, NQ, 3)
		dist_eucl = diff.norm(dim=-1)  # (Ks, NQ)

		min_dist, min_qi = dist_eucl.min(dim=1)  # (Ks,)
		pidx = pmask.nonzero(as_tuple=True)[0]
		better = min_dist < best_dist[pidx]
		if better.any():
			bi = better.nonzero(as_tuple=True)[0]
			gi = pidx[bi]
			qi = min_qi[bi]
			best_h[gi] = h_q[qi]
			best_w[gi] = w_q[qi]
			best_u[gi] = u_all[bi, qi]
			best_v[gi] = v_all[bi, qi]
			best_dist[gi] = min_dist[bi]

	# Pick best conn direction per point at its best quad
	v00_b = xyz_det[wind_d, best_h, best_w]
	v10_b = xyz_det[wind_d, best_h + 1, best_w]
	v01_b = xyz_det[wind_d, best_h, best_w + 1]
	v11_b = xyz_det[wind_d, best_h + 1, best_w + 1]
	Q_best = _bilinear_interp(v00_b, v10_b, v01_b, v11_b, best_u, best_v)
	offset = P - Q_best

	best_dir_dist = torch.full((K,), float("inf"), dtype=dt, device=dev)
	for dir_idx, conn_dir in enumerate([conn_prev, conn_next]):
		cn00 = conn_dir[wind_d, best_h, best_w]
		cn10 = conn_dir[wind_d, best_h + 1, best_w]
		cn01 = conn_dir[wind_d, best_h, best_w + 1]
		cn11 = conn_dir[wind_d, best_h + 1, best_w + 1]
		n = _bilinear_interp(cn00, cn10, cn01, cn11, best_u, best_v)
		n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
		dist = (offset * n).sum(-1).abs()
		use_this = dist < best_dir_dist
		best_dir[use_this] = dir_idx
		best_dir_dist[use_this] = dist[use_this]

	return best_h, best_w, best_u, best_v, best_dir


def _snap_update_anchors(P: torch.Tensor, xyz_det: torch.Tensor, wind_d: torch.Tensor,
						 anchor_h: torch.Tensor, anchor_w: torch.Tensor,
						 anchor_u: torch.Tensor, anchor_v: torch.Tensor,
						 anchor_dir: torch.Tensor,
						 conn_prev: torch.Tensor, conn_next: torch.Tensor,
						 Qh: int, Qw: int,
						 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Local anchor update: re-project points onto current quads.

	Computes unclamped (u, v) to get full integer shift, then re-projects.
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	D, Hm, Wm, _ = xyz_det.shape

	h_old = anchor_h.clone()
	w_old = anchor_w.clone()
	d = wind_d

	# --- Dist before update (for debug) ---
	v00_old = xyz_det[d, h_old, w_old]
	v10_old = xyz_det[d, h_old + 1, w_old]
	v01_old = xyz_det[d, h_old, w_old + 1]
	v11_old = xyz_det[d, h_old + 1, w_old + 1]
	Q_old = _bilinear_interp(v00_old, v10_old, v01_old, v11_old, anchor_u, anchor_v)
	dist_before = (P - Q_old).norm(dim=-1)

	# --- Unclamped bilinear project to get full shift ---
	e1 = v10_old - v00_old
	e2 = v01_old - v00_old
	g = P - v00_old
	e1e1 = (e1 * e1).sum(-1)
	e1e2 = (e1 * e2).sum(-1)
	e2e2 = (e2 * e2).sum(-1)
	ge1 = (g * e1).sum(-1)
	ge2 = (g * e2).sum(-1)
	det = e1e1 * e2e2 - e1e2 * e1e2
	det_safe = det + (det.abs() < 1e-20).float() * 1e-20
	u_raw = (ge1 * e2e2 - ge2 * e1e2) / det_safe
	v_raw = (ge2 * e1e1 - ge1 * e1e2) / det_safe

	# Full integer shift
	h = (h_old + u_raw.floor().to(torch.long)).clamp(0, Qh - 1)
	w = (w_old + v_raw.floor().to(torch.long)).clamp(0, Qw - 1)

	# Re-project on updated quads (clamped)
	v00 = xyz_det[d, h, w]
	v10 = xyz_det[d, h + 1, w]
	v01 = xyz_det[d, h, w + 1]
	v11 = xyz_det[d, h + 1, w + 1]
	u_new, v_new = _bilinear_project(P, v00, v10, v01, v11)  # clamped [0,1]

	# Re-check direction: pick the one with smaller |dist|
	Q_new = _bilinear_interp(v00, v10, v01, v11, u_new, v_new)
	offset = P - Q_new

	new_dir = torch.zeros(K, dtype=torch.long, device=dev)
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

	# --- Dist after update (for debug) ---
	# dist_after = offset.norm(dim=-1)
 #
	# # Debug: print per-point distances and indices before/after
	# def _fmt(t):
	# 	return " ".join(f"{v:.2f}" for v in t.tolist())
	# def _fmtpair(h_t, w_t):
	# 	return " ".join(f"({int(hi)},{int(wi)})" for hi, wi in zip(h_t.tolist(), w_t.tolist()))
	# print(f"[corr-snap-update] dist_before: {_fmt(dist_before)}")
	# print(f"[corr-snap-update] dist_after:  {_fmt(dist_after)}")
	# print(f"[corr-snap-update] idx_before:  {_fmtpair(h_old, w_old)}")
	# print(f"[corr-snap-update] idx_after:   {_fmtpair(h, w)}")

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


# ---------------------------------------------------------------------------
# Winding-observation corr loss
# ---------------------------------------------------------------------------

def _wind_nearest_quad_on_layer(
	P: torch.Tensor,           # (K, 3)
	xyz_det: torch.Tensor,     # (D, Hm, Wm, 3)
	d_layer: torch.Tensor,     # (K,) int — target depth layer per point
	Qh: int, Qw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""For each point, find nearest quad on its assigned layer.

	Returns: (h_idx, w_idx, u, v) each (K,).
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	NQ = Qh * Qw

	best_h = torch.zeros(K, dtype=torch.long, device=dev)
	best_w = torch.zeros(K, dtype=torch.long, device=dev)
	best_u = torch.zeros(K, dtype=dt, device=dev)
	best_v = torch.zeros(K, dtype=dt, device=dev)
	best_dist = torch.full((K,), float("inf"), dtype=dt, device=dev)

	h_q = torch.arange(Qh, device=dev).unsqueeze(1).expand(Qh, Qw).reshape(NQ)
	w_q = torch.arange(Qw, device=dev).unsqueeze(0).expand(Qh, Qw).reshape(NQ)

	unique_d = torch.unique(d_layer)
	for d_val in unique_d.tolist():
		d = int(d_val)
		if d < 0 or d >= xyz_det.shape[0]:
			continue
		pmask = (d_layer == d)
		if not pmask.any():
			continue
		P_sub = P[pmask]

		v00 = xyz_det[d, :-1, :-1].reshape(NQ, 3).unsqueeze(0)
		v10 = xyz_det[d, 1:, :-1].reshape(NQ, 3).unsqueeze(0)
		v01 = xyz_det[d, :-1, 1:].reshape(NQ, 3).unsqueeze(0)
		v11 = xyz_det[d, 1:, 1:].reshape(NQ, 3).unsqueeze(0)

		P_exp = P_sub.unsqueeze(1)
		u_all, v_all = _bilinear_project(P_exp, v00, v10, v01, v11)
		Q = _bilinear_interp(v00, v10, v01, v11, u_all, v_all)
		dist_sq = (P_exp - Q).square().sum(-1)

		min_dist, min_qi = dist_sq.min(dim=1)
		pidx = pmask.nonzero(as_tuple=True)[0]
		better = min_dist < best_dist[pidx]
		if better.any():
			bi = better.nonzero(as_tuple=True)[0]
			gi = pidx[bi]
			qi = min_qi[bi]
			best_h[gi] = h_q[qi]
			best_w[gi] = w_q[qi]
			best_u[gi] = u_all[bi, qi]
			best_v[gi] = v_all[bi, qi]
			best_dist[gi] = min_dist[bi]

	return best_h, best_w, best_u, best_v


def _wind_strip_integral(
	P: torch.Tensor,              # (K, 3)
	Q: torch.Tensor,              # (K, 3)
	gt_n: torch.Tensor,           # (K, 3)
	data: fit_data.FitData3D,
	strip_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Signed winding integral from P to Q.

	Returns:
		signed_winding: (K,)  sign(dot(Q-P, gt_n)) * strip_len * mean_mag
		unsigned_winding: (K,)  strip_len * mean_mag
		strip_valid: (K,) bool  all samples have grad_mag > 0
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	diff = Q - P
	t = torch.linspace(0.0, 1.0, strip_samples, device=dev, dtype=dt)
	strip = P.unsqueeze(1) + t.view(1, -1, 1) * diff.unsqueeze(1)  # (K, S, 3)

	strip_flat = strip.reshape(1, 1, K * strip_samples, 3)
	sampled = data.grid_sample_fullres(strip_flat)
	mag_raw = sampled.grad_mag  # (1, 1, 1, 1, K*S)
	mag = mag_raw.reshape(K, strip_samples)

	mean_mag = mag.mean(dim=-1).clamp(min=1e-4)
	strip_len = diff.square().sum(dim=-1).sqrt().clamp(min=1e-8)

	signed_normal_disp = (diff * gt_n).sum(dim=-1)
	int_sign = torch.sign(signed_normal_disp)

	unsigned_winding = strip_len * mean_mag
	signed_winding = int_sign * unsigned_winding
	strip_valid = (mag > 0).all(dim=-1)

	return signed_winding, unsigned_winding, strip_valid


def _wind_collection_average(
	winding_obs: torch.Tensor,    # (K,)
	winda: torch.Tensor,          # (K,)
	col: torch.Tensor,            # (K,) int
	obs_valid: torch.Tensor,      # (K,) bool — which points have reliable winding
) -> torch.Tensor:
	"""Collection-coupled target winding per point. Returns (K,) float, NaN for invalid."""
	K = winding_obs.shape[0]
	dev = winding_obs.device
	dt = winding_obs.dtype
	target = torch.full((K,), float("nan"), device=dev, dtype=dt)

	uc = torch.unique(col)
	for cid in uc.tolist():
		m = (col == int(cid)) & obs_valid
		if not m.any():
			continue
		obs_m = winding_obs[m]
		wa_m = winda[m]
		# Positive coupling: target = avg(obs - winda) + winda
		avg_pos = (obs_m - wa_m).mean()
		err_pos = obs_m - (avg_pos + wa_m)
		mse_pos = (err_pos * err_pos).mean()
		# Negative coupling: target = avg(obs + winda) - winda
		avg_neg = (obs_m + wa_m).mean()
		err_neg = obs_m - (avg_neg - wa_m)
		mse_neg = (err_neg * err_neg).mean()

		use_neg = bool((mse_neg < mse_pos).item())
		if use_neg:
			target[m] = avg_neg - wa_m
		else:
			target[m] = avg_pos + wa_m

		# Set target for all points in collection (including those excluded from avg)
		m_all = (col == int(cid)) & ~obs_valid
		if m_all.any():
			if use_neg:
				target[m_all] = avg_neg - winda[m_all]
			else:
				target[m_all] = avg_pos + winda[m_all]

	return target


def _wind_brute_force_init(
	P: torch.Tensor,              # (K, 3)
	gt_n: torch.Tensor,           # (K, 3)
	winda: torch.Tensor,          # (K,)
	col: torch.Tensor,            # (K,) int
	xyz_det: torch.Tensor,        # (D, Hm, Wm, 3)
	normals: torch.Tensor,        # (D, Hm, Wm, 3) — model surface normals
	data: fit_data.FitData3D,
	strip_samples: int,
	Qh: int, Qw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Brute-force init: bracket search, winding observation, collection avg, avg pair.

	Returns: (anchors_d, anchors_h, anchors_w, anchors_valid, target_per_point)
		anchors_d/h/w: (K, 4) int
		anchors_valid: (K, 4) bool
		target_per_point: (K,) float (NaN for fully invalid)
	"""
	K = P.shape[0]
	D = xyz_det.shape[0]
	dev = P.device
	dt = P.dtype
	NQ = Qh * Qw

	# --- Per-layer nearest quad + signed distance ---
	nearest_h = torch.zeros(K, D, dtype=torch.long, device=dev)
	nearest_w = torch.zeros(K, D, dtype=torch.long, device=dev)
	nearest_u = torch.zeros(K, D, dtype=dt, device=dev)
	nearest_v = torch.zeros(K, D, dtype=dt, device=dev)
	signed_dist = torch.zeros(K, D, dtype=dt, device=dev)

	h_q = torch.arange(Qh, device=dev).unsqueeze(1).expand(Qh, Qw).reshape(NQ)
	w_q = torch.arange(Qw, device=dev).unsqueeze(0).expand(Qh, Qw).reshape(NQ)
	P_exp = P.unsqueeze(1)  # (K, 1, 3)

	for d in range(D):
		v00 = xyz_det[d, :-1, :-1].reshape(NQ, 3).unsqueeze(0)
		v10 = xyz_det[d, 1:, :-1].reshape(NQ, 3).unsqueeze(0)
		v01 = xyz_det[d, :-1, 1:].reshape(NQ, 3).unsqueeze(0)
		v11 = xyz_det[d, 1:, 1:].reshape(NQ, 3).unsqueeze(0)

		u_all, v_all = _bilinear_project(P_exp, v00, v10, v01, v11)
		Q = _bilinear_interp(v00, v10, v01, v11, u_all, v_all)
		dist_sq = (P_exp - Q).square().sum(-1)  # (K, NQ)
		best_qi = dist_sq.argmin(dim=1)          # (K,)

		kidx = torch.arange(K, device=dev)
		nearest_h[:, d] = h_q[best_qi]
		nearest_w[:, d] = w_q[best_qi]
		nearest_u[:, d] = u_all[kidx, best_qi]
		nearest_v[:, d] = v_all[kidx, best_qi]

		# Signed distance along GT normal
		Q_best = Q[kidx, best_qi]  # (K, 3)
		signed_dist[:, d] = ((P - Q_best) * gt_n).sum(dim=-1)

	# --- Find bracket (two layers where signed_dist flips sign) ---
	bracket_lo = torch.zeros(K, dtype=torch.long, device=dev)
	bracket_valid = torch.zeros(K, dtype=torch.bool, device=dev)
	bracket_score = torch.full((K,), float("inf"), dtype=dt, device=dev)

	for d in range(D - 1):
		sd_lo = signed_dist[:, d]
		sd_hi = signed_dist[:, d + 1]
		between = (sd_lo * sd_hi) < 0
		score = sd_lo.abs() + sd_hi.abs()
		better = between & (score < bracket_score)
		bracket_lo[better] = d
		bracket_valid[better] = True
		bracket_score[better] = score[better]

	# --- Single-sided: nearest layer for non-bracketed points ---
	# Find the layer with smallest |signed_dist|
	abs_sd = signed_dist.abs()
	nearest_layer = abs_sd.argmin(dim=1)  # (K,)

	# --- Allocate anchor tensors ---
	anchors_d = torch.zeros(K, 4, dtype=torch.long, device=dev)
	anchors_h = torch.zeros(K, 4, dtype=torch.long, device=dev)
	anchors_w = torch.zeros(K, 4, dtype=torch.long, device=dev)
	anchors_valid = torch.zeros(K, 4, dtype=torch.bool, device=dev)

	# Fill closest pair
	kidx = torch.arange(K, device=dev)

	# Bracketed points: closest_low = bracket_lo, closest_up = bracket_lo + 1
	anchors_d[bracket_valid, 0] = bracket_lo[bracket_valid]
	anchors_d[bracket_valid, 1] = bracket_lo[bracket_valid] + 1
	anchors_h[bracket_valid, 0] = nearest_h[kidx[bracket_valid], bracket_lo[bracket_valid]]
	anchors_w[bracket_valid, 0] = nearest_w[kidx[bracket_valid], bracket_lo[bracket_valid]]
	anchors_h[bracket_valid, 1] = nearest_h[kidx[bracket_valid], bracket_lo[bracket_valid] + 1]
	anchors_w[bracket_valid, 1] = nearest_w[kidx[bracket_valid], bracket_lo[bracket_valid] + 1]
	anchors_valid[bracket_valid, 0] = True
	anchors_valid[bracket_valid, 1] = True

	# Single-sided: only closest_low anchor (index 0)
	single = ~bracket_valid
	anchors_d[single, 0] = nearest_layer[single]
	anchors_h[single, 0] = nearest_h[kidx[single], nearest_layer[single]]
	anchors_w[single, 0] = nearest_w[kidx[single], nearest_layer[single]]
	anchors_valid[single, 0] = True
	# closest_up invalid for single-sided (already False)

	# --- Winding observation ---
	winding_obs = torch.full((K,), float("nan"), dtype=dt, device=dev)
	obs_valid = torch.zeros(K, dtype=torch.bool, device=dev)

	# Bracketed: winding = d_low + integral_low / (integral_low + integral_up)
	if bracket_valid.any():
		bk = bracket_valid
		d_lo = anchors_d[bk, 0]
		h_lo = anchors_h[bk, 0]
		w_lo = anchors_w[bk, 0]
		u_lo = nearest_u[kidx[bk], d_lo]
		v_lo = nearest_v[kidx[bk], d_lo]
		v00_lo = xyz_det[d_lo, h_lo, w_lo]
		v10_lo = xyz_det[d_lo, h_lo + 1, w_lo]
		v01_lo = xyz_det[d_lo, h_lo, w_lo + 1]
		v11_lo = xyz_det[d_lo, h_lo + 1, w_lo + 1]
		Q_lo = _bilinear_interp(v00_lo, v10_lo, v01_lo, v11_lo, u_lo, v_lo)

		d_hi = anchors_d[bk, 1]
		h_hi = anchors_h[bk, 1]
		w_hi = anchors_w[bk, 1]
		u_hi = nearest_u[kidx[bk], d_hi]
		v_hi = nearest_v[kidx[bk], d_hi]
		v00_hi = xyz_det[d_hi, h_hi, w_hi]
		v10_hi = xyz_det[d_hi, h_hi + 1, w_hi]
		v01_hi = xyz_det[d_hi, h_hi, w_hi + 1]
		v11_hi = xyz_det[d_hi, h_hi + 1, w_hi + 1]
		Q_hi = _bilinear_interp(v00_hi, v10_hi, v01_hi, v11_hi, u_hi, v_hi)

		P_bk = P[bk]
		gt_n_bk = gt_n[bk]
		_, uint_lo, sv_lo = _wind_strip_integral(Q_lo, P_bk, gt_n_bk, data, strip_samples)
		_, uint_hi, sv_hi = _wind_strip_integral(P_bk, Q_hi, gt_n_bk, data, strip_samples)
		frac = uint_lo / (uint_lo + uint_hi + 1e-8)
		winding_obs[bk] = d_lo.to(dt) + frac
		obs_valid[bk] = sv_lo & sv_hi

	# Single-sided: winding = d_nearest +/- integral (valid only if integral < 1.0)
	if single.any():
		d_s = anchors_d[single, 0]
		h_s = anchors_h[single, 0]
		w_s = anchors_w[single, 0]
		u_s = nearest_u[kidx[single], d_s]
		v_s = nearest_v[kidx[single], d_s]
		v00_s = xyz_det[d_s, h_s, w_s]
		v10_s = xyz_det[d_s, h_s + 1, w_s]
		v01_s = xyz_det[d_s, h_s, w_s + 1]
		v11_s = xyz_det[d_s, h_s + 1, w_s + 1]
		Q_s = _bilinear_interp(v00_s, v10_s, v01_s, v11_s, u_s, v_s)

		P_s = P[single]
		gt_n_s = gt_n[single]
		_, uw, sv = _wind_strip_integral(P_s, Q_s, gt_n_s, data, strip_samples)
		# Use model surface normal at Q to determine sign consistently
		n00 = normals[d_s, h_s, w_s]
		n10 = normals[d_s, (h_s + 1).clamp(max=Qh), w_s]
		n01 = normals[d_s, h_s, (w_s + 1).clamp(max=Qw)]
		n11 = normals[d_s, (h_s + 1).clamp(max=Qh), (w_s + 1).clamp(max=Qw)]
		surf_n = _bilinear_interp(n00, n10, n01, n11, u_s, v_s)
		surf_n = surf_n / (surf_n.norm(dim=-1, keepdim=True) + 1e-8)
		# dot(P - Q, surf_n) > 0 → P is above surface (in normal direction)
		above = ((P_s - Q_s) * surf_n).sum(dim=-1) > 0
		w_est = torch.where(above, d_s.to(dt) + uw, d_s.to(dt) - uw)
		valid_single = sv & (uw < 1.0)
		winding_obs[single] = w_est
		obs_valid[single] = valid_single

	# --- Collection averaging ---
	target = _wind_collection_average(winding_obs, winda, col, obs_valid)

	# --- Avg pair anchors from target winding ---
	target_finite = torch.isfinite(target)
	# Replace NaN with 0 before floor/long to avoid garbage integers
	target_safe = torch.where(target_finite, target, torch.zeros_like(target))
	avg_lo_d = target_safe.floor().clamp(0, max(D - 2, 0)).long()
	avg_hi_d = (avg_lo_d + 1).clamp(max=D - 1)

	# Find quads on avg layers
	if target_finite.any():
		# avg_low — valid if target is finite and layer in range
		al_h, al_w, _, _ = _wind_nearest_quad_on_layer(P, xyz_det, avg_lo_d, Qh, Qw)
		anchors_d[:, 2] = avg_lo_d
		anchors_h[:, 2] = al_h
		anchors_w[:, 2] = al_w
		anchors_valid[:, 2] = target_finite & (avg_lo_d >= 0) & (avg_lo_d < D)

		# avg_up — only valid if it's a different layer from avg_lo (requires D >= 2)
		ah_h, ah_w, _, _ = _wind_nearest_quad_on_layer(P, xyz_det, avg_hi_d, Qh, Qw)
		anchors_d[:, 3] = avg_hi_d
		anchors_h[:, 3] = ah_h
		anchors_w[:, 3] = ah_w
		anchors_valid[:, 3] = target_finite & (avg_hi_d > avg_lo_d)

	return anchors_d, anchors_h, anchors_w, anchors_valid, target


def _wind_update_anchors(
	P: torch.Tensor,              # (K, 3)
	gt_n: torch.Tensor,           # (K, 3)
	xyz_det: torch.Tensor,        # (D, Hm, Wm, 3)
	anchors_d: torch.Tensor,      # (K, 4) int
	anchors_h: torch.Tensor,      # (K, 4) int
	anchors_w: torch.Tensor,      # (K, 4) int
	anchors_valid: torch.Tensor,  # (K, 4) bool
	target: torch.Tensor,         # (K,) float
	Qh: int, Qw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Local anchor update for all correspondences.

	Returns: (anchors_h, anchors_w, anchors_valid) — mutated in-place but also returned.
	"""
	K = P.shape[0]
	D = xyz_det.shape[0]
	dev = P.device
	dt = P.dtype

	for ci in range(4):
		valid = anchors_valid[:, ci]
		if not valid.any():
			continue
		vi = valid.nonzero(as_tuple=True)[0]
		d_ci = anchors_d[vi, ci]
		h_old = anchors_h[vi, ci]
		w_old = anchors_w[vi, ci]

		# Unclamped bilinear project on current quad
		v00 = xyz_det[d_ci, h_old, w_old]
		v10 = xyz_det[d_ci, (h_old + 1).clamp(max=Qh), w_old]
		v01 = xyz_det[d_ci, h_old, (w_old + 1).clamp(max=Qw)]
		v11 = xyz_det[d_ci, (h_old + 1).clamp(max=Qh), (w_old + 1).clamp(max=Qw)]

		e1 = v10 - v00
		e2 = v01 - v00
		g = P[vi] - v00
		e1e1 = (e1 * e1).sum(-1)
		e1e2 = (e1 * e2).sum(-1)
		e2e2 = (e2 * e2).sum(-1)
		ge1 = (g * e1).sum(-1)
		ge2 = (g * e2).sum(-1)
		det = e1e1 * e2e2 - e1e2 * e1e2
		det_safe = det + (det.abs() < 1e-20).float() * 1e-20
		u_raw = (ge1 * e2e2 - ge2 * e1e2) / det_safe
		v_raw = (ge2 * e1e1 - ge1 * e1e2) / det_safe

		# Integer shift + clamp
		h_new = (h_old + u_raw.floor().to(torch.long)).clamp(0, Qh - 1)
		w_new = (w_old + v_raw.floor().to(torch.long)).clamp(0, Qw - 1)

		anchors_h[vi, ci] = h_new
		anchors_w[vi, ci] = w_new

	# Re-check bracket for closest pair (indices 0, 1)
	has_bracket = anchors_valid[:, 0] & anchors_valid[:, 1]
	if has_bracket.any():
		bi = has_bracket.nonzero(as_tuple=True)[0]
		for ci in [0, 1]:
			d_ci = anchors_d[bi, ci]
			h_ci = anchors_h[bi, ci]
			w_ci = anchors_w[bi, ci]
			v00 = xyz_det[d_ci, h_ci, w_ci]
			v10 = xyz_det[d_ci, (h_ci + 1).clamp(max=Qh), w_ci]
			v01 = xyz_det[d_ci, h_ci, (w_ci + 1).clamp(max=Qw)]
			v11 = xyz_det[d_ci, (h_ci + 1).clamp(max=Qh), (w_ci + 1).clamp(max=Qw)]
			u_c, v_c = _bilinear_project(P[bi], v00, v10, v01, v11)
			Q_c = _bilinear_interp(v00, v10, v01, v11, u_c, v_c)
			sd = ((P[bi] - Q_c) * gt_n[bi]).sum(dim=-1)
			if ci == 0:
				sd_lo = sd
			else:
				sd_hi = sd
		# Check if bracket is still valid (signs differ)
		bracket_lost = (sd_lo * sd_hi) >= 0
		if bracket_lost.any():
			# Mark lost brackets invalid — will be re-found next brute-force
			lost = bi[bracket_lost]
			anchors_valid[lost, 0] = False
			anchors_valid[lost, 1] = False

	# Update avg pair layers if target changed
	D = xyz_det.shape[0]
	target_finite = torch.isfinite(target)
	target_safe = torch.where(target_finite, target, torch.zeros_like(target))
	new_avg_lo_d = target_safe.floor().clamp(0, max(D - 2, 0)).long()
	new_avg_hi_d = (new_avg_lo_d + 1).clamp(max=D - 1)

	# Re-init avg anchors where layer changed
	lo_changed = target_finite & (new_avg_lo_d != anchors_d[:, 2])
	hi_changed = target_finite & (new_avg_hi_d != anchors_d[:, 3])

	if lo_changed.any():
		lci = lo_changed.nonzero(as_tuple=True)[0]
		al_h, al_w, _, _ = _wind_nearest_quad_on_layer(
			P[lci], xyz_det, new_avg_lo_d[lci], Qh, Qw)
		anchors_d[lci, 2] = new_avg_lo_d[lci]
		anchors_h[lci, 2] = al_h
		anchors_w[lci, 2] = al_w
		anchors_valid[lci, 2] = True

	if hi_changed.any():
		hci = hi_changed.nonzero(as_tuple=True)[0]
		ah_h, ah_w, _, _ = _wind_nearest_quad_on_layer(
			P[hci], xyz_det, new_avg_hi_d[hci], Qh, Qw)
		anchors_d[hci, 3] = new_avg_hi_d[hci]
		anchors_h[hci, 3] = ah_h
		anchors_w[hci, 3] = ah_w
		anchors_valid[hci, 3] = (new_avg_hi_d[hci] > new_avg_lo_d[hci])

	# Ensure avg layer values are current even if not changed
	anchors_d[:, 2] = new_avg_lo_d
	anchors_d[:, 3] = new_avg_hi_d
	anchors_valid[:, 2] = anchors_valid[:, 2] & target_finite & (new_avg_lo_d >= 0) & (new_avg_lo_d < D)
	anchors_valid[:, 3] = anchors_valid[:, 3] & target_finite & (new_avg_hi_d >= 0) & (new_avg_hi_d < D)

	return anchors_h, anchors_w, anchors_valid


def _corr_winding_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Winding-observation corr loss with proxy correction."""
	global _dbg_call_count, _last_results
	global _wind_anchors_d, _wind_anchors_h, _wind_anchors_w
	global _wind_anchors_valid, _wind_initialized, _wind_target_per_point

	_dbg_call_count += 1
	dbg = (_dbg_call_count <= 2)

	dev = res.xyz_lr.device
	dt = res.xyz_lr.dtype

	pts_c = res.data.corr_points
	if pts_c is None or pts_c.points_xyz_winda.shape[0] == 0:
		if _dbg_call_count <= 2:
			print("[corr-wind] no correction points")
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	pts = pts_c.points_xyz_winda.to(device=dev, dtype=dt)
	col = pts_c.collection_idx.to(device=dev, dtype=torch.int64)
	pt_ids = pts_c.point_ids.to(device=dev, dtype=torch.int64)
	K = int(pts.shape[0])
	P = pts[:, :3]
	winda = pts[:, 3]

	xyz_lr = res.xyz_lr
	D, Hm, Wm, _ = xyz_lr.shape
	Qh, Qw = Hm - 1, Wm - 1
	if Qh <= 0 or Qw <= 0 or D < 1:
		if _dbg_call_count <= 2:
			print(f"[corr-wind] mesh too small D={D} Hm={Hm} Wm={Wm}")
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	xyz_det = xyz_lr.detach()
	strip_samples = max(2, int(res.params.subsample_mesh) + 1)

	# Sample GT normals at corr point positions
	gt_n_sampled = res.data.grid_sample_fullres(P.reshape(1, 1, K, 3))
	gt_n = gt_n_sampled.normal_3d  # (1, 1, K, 3) after squeeze in property
	gt_n = gt_n.reshape(K, 3)
	gt_n = gt_n / (gt_n.norm(dim=-1, keepdim=True) + 1e-8)

	# --- Initialize or update anchors ---
	if not _wind_initialized or _wind_anchors_d is None or _wind_anchors_d.shape[0] != K:
		if dbg:
			print(f"[corr-wind] brute-force init: K={K} points, mesh=({D},{Hm},{Wm})")
		with torch.no_grad():
			(
				_wind_anchors_d, _wind_anchors_h, _wind_anchors_w,
				_wind_anchors_valid, _wind_target_per_point,
			) = _wind_brute_force_init(
				P, gt_n, winda, col, xyz_det, res.normals, res.data, strip_samples, Qh, Qw)
		_wind_initialized = True
	else:
		# Phase A: Winding observation (closest pair, detached)
		with torch.no_grad():
			winding_obs = torch.full((K,), float("nan"), dtype=dt, device=dev)
			obs_valid = torch.zeros(K, dtype=torch.bool, device=dev)
			kidx = torch.arange(K, device=dev)

			# Bracketed points
			has_bracket = _wind_anchors_valid[:, 0] & _wind_anchors_valid[:, 1]
			if has_bracket.any():
				bk = has_bracket
				bki = bk.nonzero(as_tuple=True)[0]
				Q_pair = []
				for ci in [0, 1]:
					d_ci = _wind_anchors_d[bki, ci]
					h_ci = _wind_anchors_h[bki, ci]
					w_ci = _wind_anchors_w[bki, ci]
					v00 = xyz_det[d_ci, h_ci, w_ci]
					v10 = xyz_det[d_ci, (h_ci + 1).clamp(max=Qh), w_ci]
					v01 = xyz_det[d_ci, h_ci, (w_ci + 1).clamp(max=Qw)]
					v11 = xyz_det[d_ci, (h_ci + 1).clamp(max=Qh), (w_ci + 1).clamp(max=Qw)]
					u_c, v_c = _bilinear_project(P[bki], v00, v10, v01, v11)
					Q_pair.append(_bilinear_interp(v00, v10, v01, v11, u_c, v_c))

				Q_lo, Q_hi = Q_pair
				P_bk = P[bki]
				gt_n_bk = gt_n[bki]
				_, uint_lo, sv_lo = _wind_strip_integral(Q_lo, P_bk, gt_n_bk, res.data, strip_samples)
				_, uint_hi, sv_hi = _wind_strip_integral(P_bk, Q_hi, gt_n_bk, res.data, strip_samples)
				frac = uint_lo / (uint_lo + uint_hi + 1e-8)
				d_lo = _wind_anchors_d[bki, 0].to(dt)
				winding_obs[bki] = d_lo + frac
				obs_valid[bki] = sv_lo & sv_hi

			# Single-sided points
			single = _wind_anchors_valid[:, 0] & ~_wind_anchors_valid[:, 1]
			if single.any():
				si = single.nonzero(as_tuple=True)[0]
				d_s = _wind_anchors_d[si, 0]
				h_s = _wind_anchors_h[si, 0]
				w_s = _wind_anchors_w[si, 0]
				v00 = xyz_det[d_s, h_s, w_s]
				v10 = xyz_det[d_s, (h_s + 1).clamp(max=Qh), w_s]
				v01 = xyz_det[d_s, h_s, (w_s + 1).clamp(max=Qw)]
				v11 = xyz_det[d_s, (h_s + 1).clamp(max=Qh), (w_s + 1).clamp(max=Qw)]
				u_c, v_c = _bilinear_project(P[si], v00, v10, v01, v11)
				Q_s = _bilinear_interp(v00, v10, v01, v11, u_c, v_c)
				_, uw, sv = _wind_strip_integral(P[si], Q_s, gt_n[si], res.data, strip_samples)
				# Use model surface normal at Q for consistent sign
				mesh_normals = res.normals
				n00 = mesh_normals[d_s, h_s, w_s]
				n10 = mesh_normals[d_s, (h_s + 1).clamp(max=Qh), w_s]
				n01 = mesh_normals[d_s, h_s, (w_s + 1).clamp(max=Qw)]
				n11 = mesh_normals[d_s, (h_s + 1).clamp(max=Qh), (w_s + 1).clamp(max=Qw)]
				surf_n = _bilinear_interp(n00, n10, n01, n11, u_c, v_c)
				surf_n = surf_n / (surf_n.norm(dim=-1, keepdim=True) + 1e-8)
				above = ((P[si] - Q_s) * surf_n).sum(dim=-1) > 0
				w_est = torch.where(above, d_s.to(dt) + uw, d_s.to(dt) - uw)
				winding_obs[si] = w_est
				obs_valid[si] = sv & (uw < 1.0)

			# Phase B: Collection averaging
			_wind_target_per_point = _wind_collection_average(
				winding_obs, winda, col, obs_valid)

			# Phase C: Update anchors
			_wind_anchors_h, _wind_anchors_w, _wind_anchors_valid = _wind_update_anchors(
				P, gt_n, xyz_det,
				_wind_anchors_d, _wind_anchors_h, _wind_anchors_w,
				_wind_anchors_valid, _wind_target_per_point,
				Qh, Qw)

	# === Phase D: Proxy correction loss (avg pair, WITH gradients) ===
	target = _wind_target_per_point
	target_finite = torch.isfinite(target)
	if not target_finite.any():
		if dbg:
			print("[corr-wind] no valid targets")
		z = torch.zeros((), device=dev, dtype=dt)
		return z, (torch.zeros((1,), device=dev, dtype=dt),), (torch.zeros((1,), device=dev, dtype=dt),)

	frac = (target - _wind_anchors_d[:, 2].to(dt)).clamp(0.0, 1.0)

	total_loss = torch.zeros((), device=dev, dtype=dt)
	total_wsum = 0.0
	n_surfaces = 0
	all_err = torch.zeros(K, device=dev, dtype=dt)

	for ci, tgt_fn in [(2, lambda f: -f), (3, lambda f: 1.0 - f)]:
		valid = _wind_anchors_valid[:, ci] & target_finite
		if not valid.any():
			continue
		vi = valid.nonzero(as_tuple=True)[0]
		d_ci = _wind_anchors_d[vi, ci]
		h_ci = _wind_anchors_h[vi, ci]
		w_ci = _wind_anchors_w[vi, ci]

		# Gather 4 model quad corners WITH gradients
		M00 = xyz_lr[d_ci, h_ci, w_ci]
		M10 = xyz_lr[d_ci, h_ci + 1, w_ci]
		M01 = xyz_lr[d_ci, h_ci, w_ci + 1]
		M11 = xyz_lr[d_ci, h_ci + 1, w_ci + 1]

		with torch.no_grad():
			M00_det = M00.detach()
			M10_det = M10.detach()
			M01_det = M01.detach()
			M11_det = M11.detach()

			# Project P onto quad (detached u,v for weights)
			u_ci, v_ci = _bilinear_project(P[vi], M00_det, M10_det, M01_det, M11_det)

			# Bilinear model point (detached)
			uf = u_ci.unsqueeze(-1)
			vf = v_ci.unsqueeze(-1)
			Q = (1 - uf) * (1 - vf) * M00_det + uf * (1 - vf) * M10_det + \
				(1 - uf) * vf * M01_det + uf * vf * M11_det

			# Strip from P to Q: signed winding
			sw, _, sv = _wind_strip_integral(P[vi], Q, gt_n[vi], res.data, strip_samples)

			# Target offset
			tgt = tgt_fn(frac[vi])
			err = sw - tgt

			# Store error for results
			all_err[vi] = err

			# Proxies: shift each corner by gt_n * err
			we = err.unsqueeze(-1)
			gt_n_vi = gt_n[vi]
			proxy00 = M00_det - gt_n_vi * we
			proxy10 = M10_det - gt_n_vi * we
			proxy01 = M01_det - gt_n_vi * we
			proxy11 = M11_det - gt_n_vi * we

		# Bilinear weights
		w00 = (1 - u_ci) * (1 - v_ci)
		w10 = u_ci * (1 - v_ci)
		w01 = (1 - u_ci) * v_ci
		w11 = u_ci * v_ci

		# L2 loss (gradients flow through M00..M11)
		lm00 = (M00 - proxy00).square().sum(dim=-1)
		lm10 = (M10 - proxy10).square().sum(dim=-1)
		lm01 = (M01 - proxy01).square().sum(dim=-1)
		lm11 = (M11 - proxy11).square().sum(dim=-1)

		lm = w00 * lm00 + w10 * lm10 + w01 * lm01 + w11 * lm11

		# Mask by strip validity
		mask_ci = sv.to(dt)
		wsum = float(mask_ci.sum().detach().cpu())
		if wsum > 0:
			total_loss = total_loss + (lm * mask_ci).sum() / wsum
			total_wsum += wsum
		n_surfaces += 1

	if n_surfaces > 1:
		total_loss = total_loss / n_surfaces

	if dbg:
		n_valid = int(target_finite.sum().item())
		rms = float(total_loss.detach().sqrt().item()) if total_wsum > 0 else float("nan")
		print(f"[corr-wind] loss={float(total_loss.detach().item()):.6f}, "
			  f"valid={n_valid}/{K}, rms_proxy={rms:.4f}")

	# Build results
	_last_results = _build_winding_results(
		winding_obs=_wind_target_per_point, target=target,
		err=all_err, pt_ids=pt_ids, col=col, pts=pts, winda=winda,
		valid=target_finite,
	)
	if _dbg_call_count == 1:
		print_detail("INIT")

	err_sq = all_err * all_err
	mask_out = target_finite.to(dt)
	return total_loss, (err_sq,), (mask_out,)


def _build_winding_results(
	*, winding_obs: torch.Tensor, target: torch.Tensor,
	err: torch.Tensor, pt_ids: torch.Tensor, col: torch.Tensor,
	pts: torch.Tensor, winda: torch.Tensor, valid: torch.Tensor,
) -> dict:
	"""Build JSON-serializable dict of per-point winding results."""
	result: dict = {"points": {}, "collection_avgs": {}}
	K = int(pts.shape[0])
	for i in range(K):
		pid = int(pt_ids[i].item())
		cid = int(col[i].item())
		w_obs = float(winding_obs[i].item()) if math.isfinite(float(winding_obs[i].item())) else None
		w_tgt = float(target[i].item()) if math.isfinite(float(target[i].item())) else None
		e = float(err[i].item()) if bool(valid[i]) and math.isfinite(float(err[i].item())) else None
		entry: dict = {
			"collection_id": cid,
			"p": [round(float(pts[i, j].item()), 2) for j in range(3)],
			"winding_obs": round(w_obs, 6) if w_obs is not None else None,
			"winding_target": round(w_tgt, 6) if w_tgt is not None else None,
			"winding_err": round(e, 6) if e is not None else None,
			"valid": bool(valid[i]),
		}
		result["points"][str(pid)] = entry
	# Per-collection averages
	uc = torch.unique(col)
	for cid_t in uc.tolist():
		cid_int = int(cid_t)
		mask = (col == cid_int) & valid
		if mask.any():
			v = float(target[mask].mean().item())
			if math.isfinite(v):
				result["collection_avgs"][str(cid_int)] = round(v, 6)
	return result
