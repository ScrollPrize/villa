from __future__ import annotations

import math

import torch

import fit_data
import model as fit_model

_dbg_call_count = 0
_last_results: dict | None = None

# Winding-mode anchor state (persists across calls)
# Correspondence indices: 0=closest_low, 1=closest_up, 2=avg_low, 3=avg_up
_wind_anchors_d: torch.Tensor | None = None      # (K, 4) int — depth layer
_wind_anchors_h: torch.Tensor | None = None      # (K, 4) int — quad row
_wind_anchors_w: torch.Tensor | None = None      # (K, 4) int — quad col
_wind_anchors_valid: torch.Tensor | None = None   # (K, 4) bool — per-anchor validity
_wind_initialized: bool = False
_wind_target_per_point: torch.Tensor | None = None  # (K,) float — cached target winding
_wind_obs_per_point: torch.Tensor | None = None      # (K,) float — last raw winding observation
_wind_reinit_counter: int = 0
_wind_prev_any_valid: torch.Tensor | None = None     # (K,) bool — per-point "had valid anchor last step"


def corr_winding_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	return _corr_winding_loss(res=res)


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

	print(f"{tag} {len(pts)} points")

	# Collection averages
	for cid, avg in sorted(col_avgs.items(), key=lambda x: int(x[0])):
		print(f"  collection {cid}: avg_winding={avg}")

	# Per-point table
	print(f"  {'pid':>6s}  {'col':>4s}  {'pos':>26s}  {'w_obs':>8s}  {'w_tgt':>8s}  {'w_err':>8s}  {'valid':>5s}  {'abs':>3s}", end="")
	# Anchor columns (winding mode only)
	has_anchors = _wind_anchors_d is not None
	if has_anchors:
		print(f"  {'cl_lo':>8s}  {'cl_up':>8s}  {'tg_lo':>8s}  {'tg_up':>8s}", end="")
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
		is_abs = p.get("absolute", False)
		print(f"  {pid:>6s}  {p.get('collection_id', '?'):>4}  {pos_s}  "
			  f"{_fmt(w_obs):>8s}  {_fmt(w_tgt):>8s}  {_fmt(w_err):>8s}  "
			  f"{'  yes' if valid else '   no':>5s}  {'  Y' if is_abs else '  N':>3s}", end="")
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


# ---------------------------------------------------------------------------
# Winding-observation corr loss
# ---------------------------------------------------------------------------

def _wind_nearest_quad_on_layer(
	P: torch.Tensor,           # (K, 3)
	n: torch.Tensor,           # (K, 3) — ray direction (surface normal)
	xyz_det: torch.Tensor,     # (D, Hm, Wm, 3)
	d_layer: torch.Tensor,     # (K,) int — target depth layer per point
	Qh: int, Qw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""For each point, find nearest quad via ray-bilinear intersection.

	Shoots ray from P along n, intersects all quads on the assigned layer,
	picks the quad whose intersection point is closest to P.

	Returns: (h_idx, w_idx, u, v) each (K,).
	u, v are UNCLAMPED — caller checks [0,1] for validity.
	"""
	K = P.shape[0]
	dev = P.device
	dt = P.dtype
	Hm = Qh + 1
	Wm = Qw + 1
	NQ = Qh * Qw

	best_h = torch.zeros(K, dtype=torch.long, device=dev)
	best_w = torch.zeros(K, dtype=torch.long, device=dev)
	best_u = torch.zeros(K, dtype=dt, device=dev)
	best_v = torch.zeros(K, dtype=dt, device=dev)
	best_dist = torch.full((K,), float("inf"), dtype=dt, device=dev)

	h_q = torch.arange(Qh, device=dev).unsqueeze(1).expand(Qh, Qw).reshape(NQ)
	w_q = torch.arange(Qw, device=dev).unsqueeze(0).expand(Qh, Qw).reshape(NQ)
	# Fractional hints at quad center
	frac_h = torch.full((NQ,), 0.5, device=dev, dtype=dt)
	frac_w = torch.full((NQ,), 0.5, device=dev, dtype=dt)

	unique_d = torch.unique(d_layer)
	for d_val in unique_d.tolist():
		d = int(d_val)
		if d < 0 or d >= xyz_det.shape[0]:
			continue
		pmask = (d_layer == d)
		if not pmask.any():
			continue
		P_sub = P[pmask]       # (Ks, 3)
		n_sub = n[pmask]       # (Ks, 3)
		Ks = P_sub.shape[0]

		v00 = xyz_det[d, :-1, :-1].reshape(NQ, 3)
		v10 = xyz_det[d, 1:, :-1].reshape(NQ, 3)
		v01 = xyz_det[d, :-1, 1:].reshape(NQ, 3)
		v11 = xyz_det[d, 1:, 1:].reshape(NQ, 3)

		# Ray-bilinear intersect: (Ks, NQ) — broadcast P/n over quads
		u_all, v_all = fit_model.Model3D._ray_bilinear_intersect(
			P_sub.unsqueeze(1).expand(Ks, NQ, 3),
			n_sub.unsqueeze(1).expand(Ks, NQ, 3),
			v00.unsqueeze(0).expand(Ks, NQ, 3),
			v10.unsqueeze(0).expand(Ks, NQ, 3),
			v01.unsqueeze(0).expand(Ks, NQ, 3),
			v11.unsqueeze(0).expand(Ks, NQ, 3),
			frac_h.unsqueeze(0).expand(Ks, NQ),
			frac_w.unsqueeze(0).expand(Ks, NQ),
		)  # u_all, v_all: (Ks, NQ) unclamped

		# Compute intersection point and distance
		uc = u_all.unsqueeze(-1)
		vc = v_all.unsqueeze(-1)
		Q = ((1 - uc) * (1 - vc) * v00.unsqueeze(0) +
			 uc * (1 - vc) * v10.unsqueeze(0) +
			 (1 - uc) * vc * v01.unsqueeze(0) +
			 uc * vc * v11.unsqueeze(0))
		dist_sq = (P_sub.unsqueeze(1) - Q).square().sum(-1)  # (Ks, NQ)

		# Prefer in-bounds intersections; fall back to closest if none in-bounds
		in_bounds = (u_all >= 0) & (u_all <= 1) & (v_all >= 0) & (v_all <= 1)
		# Penalize out-of-bounds quads so in-bounds are preferred
		dist_sq_penalized = torch.where(in_bounds, dist_sq, dist_sq + 1e12)
		min_dist, min_qi = dist_sq_penalized.min(dim=1)

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
	is_absolute: torch.Tensor,    # (K,) bool — absolute points bypass averaging
) -> torch.Tensor:
	"""Collection-coupled target winding per point. Returns (K,) float, NaN for invalid.

	Absolute points: target = winda directly (no averaging).
	Relative points: observe, subtract winda, average within collection, add winda back.
	"""
	K = winding_obs.shape[0]
	dev = winding_obs.device
	dt = winding_obs.dtype
	target = torch.full((K,), float("nan"), device=dev, dtype=dt)

	# Absolute points: target is the specified winding directly
	target[is_absolute] = winda[is_absolute]

	# Relative points: collection-coupled +/- winda averaging
	uc = torch.unique(col)
	for cid in uc.tolist():
		m = (col == int(cid)) & obs_valid & ~is_absolute
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

		# Set target for relative points in collection excluded from avg
		m_all = (col == int(cid)) & ~obs_valid & ~is_absolute
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
	is_absolute: torch.Tensor,    # (K,) bool
	xyz_det: torch.Tensor,        # (D, Hm, Wm, 3)
	normals: torch.Tensor,        # (D, Hm, Wm, 3) — model surface normals
	data: fit_data.FitData3D,
	strip_samples: int,
	Qh: int, Qw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
	frac_half_h = torch.full((K, NQ), 0.5, device=dev, dtype=dt)
	frac_half_w = torch.full((K, NQ), 0.5, device=dev, dtype=dt)
	P_exp = P.unsqueeze(1).expand(K, NQ, 3)
	n_exp = gt_n.unsqueeze(1).expand(K, NQ, 3)

	for d in range(D):
		v00 = xyz_det[d, :-1, :-1].reshape(NQ, 3).unsqueeze(0).expand(K, NQ, 3)
		v10 = xyz_det[d, 1:, :-1].reshape(NQ, 3).unsqueeze(0).expand(K, NQ, 3)
		v01 = xyz_det[d, :-1, 1:].reshape(NQ, 3).unsqueeze(0).expand(K, NQ, 3)
		v11 = xyz_det[d, 1:, 1:].reshape(NQ, 3).unsqueeze(0).expand(K, NQ, 3)

		# Ray-bilinear intersection: unclamped (u, v)
		u_all, v_all = fit_model.Model3D._ray_bilinear_intersect(
			P_exp, n_exp, v00, v10, v01, v11, frac_half_h, frac_half_w)
		uc = u_all.unsqueeze(-1)
		vc = v_all.unsqueeze(-1)
		Q = ((1 - uc) * (1 - vc) * v00 + uc * (1 - vc) * v10 +
			 (1 - uc) * vc * v01 + uc * vc * v11)
		dist_sq = (P_exp - Q).square().sum(-1)  # (K, NQ)

		# Prefer in-bounds intersections
		in_bounds = (u_all >= 0) & (u_all <= 1) & (v_all >= 0) & (v_all <= 1)
		dist_sq_pen = torch.where(in_bounds, dist_sq, dist_sq + 1e12)
		best_qi = dist_sq_pen.argmin(dim=1)  # (K,)

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
	bv = bracket_valid
	bk_lo = bracket_lo[bv]
	bk_hi = bk_lo + 1
	bk_idx = kidx[bv]
	anchors_d[bv, 0] = bk_lo
	anchors_d[bv, 1] = bk_hi
	anchors_h[bv, 0] = nearest_h[bk_idx, bk_lo]
	anchors_w[bv, 0] = nearest_w[bk_idx, bk_lo]
	anchors_h[bv, 1] = nearest_h[bk_idx, bk_hi]
	anchors_w[bv, 1] = nearest_w[bk_idx, bk_hi]
	# Valid only if point projects within the quad (u,v in [0,1])
	u_lo = nearest_u[bk_idx, bk_lo]; v_lo = nearest_v[bk_idx, bk_lo]
	u_hi = nearest_u[bk_idx, bk_hi]; v_hi = nearest_v[bk_idx, bk_hi]
	anchors_valid[bv, 0] = (u_lo >= 0) & (u_lo <= 1) & (v_lo >= 0) & (v_lo <= 1)
	anchors_valid[bv, 1] = (u_hi >= 0) & (u_hi <= 1) & (v_hi >= 0) & (v_hi <= 1)

	# Single-sided: only closest_low anchor (index 0)
	single = ~bracket_valid
	s_idx = kidx[single]
	s_layer = nearest_layer[single]
	anchors_d[single, 0] = s_layer
	anchors_h[single, 0] = nearest_h[s_idx, s_layer]
	anchors_w[single, 0] = nearest_w[s_idx, s_layer]
	u_s = nearest_u[s_idx, s_layer]; v_s = nearest_v[s_idx, s_layer]
	anchors_valid[single, 0] = (u_s >= 0) & (u_s <= 1) & (v_s >= 0) & (v_s <= 1)
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
	target = _wind_collection_average(winding_obs, winda, col, obs_valid, is_absolute)

	# --- Avg pair anchors from target winding ---
	target_finite = torch.isfinite(target)
	# Replace NaN with 0 before floor/long to avoid garbage integers
	target_safe = torch.where(target_finite, target, torch.zeros_like(target))
	avg_lo_d = target_safe.floor().clamp(0, max(D - 2, 0)).long()
	avg_hi_d = (avg_lo_d + 1).clamp(max=D - 1)

	# Find quads on avg layers — only valid if point projects within quad (u,v in [0,1])
	if target_finite.any():
		# avg_low
		al_h, al_w, al_u, al_v = _wind_nearest_quad_on_layer(P, gt_n, xyz_det, avg_lo_d, Qh, Qw)
		al_in_quad = (al_u >= 0) & (al_u <= 1) & (al_v >= 0) & (al_v <= 1)
		anchors_d[:, 2] = avg_lo_d
		anchors_h[:, 2] = al_h
		anchors_w[:, 2] = al_w
		anchors_valid[:, 2] = target_finite & (avg_lo_d >= 0) & (avg_lo_d < D) & al_in_quad

		# avg_up — only valid if it's a different layer from avg_lo (requires D >= 2)
		ah_h, ah_w, ah_u, ah_v = _wind_nearest_quad_on_layer(P, gt_n, xyz_det, avg_hi_d, Qh, Qw)
		ah_in_quad = (ah_u >= 0) & (ah_u <= 1) & (ah_v >= 0) & (ah_v <= 1)
		anchors_d[:, 3] = avg_hi_d
		anchors_h[:, 3] = ah_h
		anchors_w[:, 3] = ah_w
		anchors_valid[:, 3] = target_finite & (avg_hi_d > avg_lo_d) & ah_in_quad

	return anchors_d, anchors_h, anchors_w, anchors_valid, target, winding_obs


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
	"""Two-pass ray-bilinear anchor update (same pattern as ext_offset).

	Pass 1: ray intersect on current quad → unclamped (u, v)
	Pass 2: shift quad idx based on pass-1, re-intersect → final (u, v)
	Mark invalid if final u,v outside [0,1].

	Returns: (anchors_h, anchors_w, anchors_valid) — mutated in-place but also returned.
	"""
	K = P.shape[0]
	D = xyz_det.shape[0]
	Hm = Qh + 1
	Wm = Qw + 1

	def _gather_quad(d, h, w):
		return (xyz_det[d, h, w],
				xyz_det[d, (h + 1).clamp(max=Qh), w],
				xyz_det[d, h, (w + 1).clamp(max=Qw)],
				xyz_det[d, (h + 1).clamp(max=Qh), (w + 1).clamp(max=Qw)])

	# --- Two-pass update for all 4 anchor types ---
	for ci in range(4):
		valid = anchors_valid[:, ci]
		if not valid.any():
			continue
		vi = valid.nonzero(as_tuple=True)[0]
		d_ci = anchors_d[vi, ci]
		row = anchors_h[vi, ci]
		col = anchors_w[vi, ci]
		P_vi = P[vi]
		n_vi = gt_n[vi]

		# Pass 1: intersect on current quad
		M00, M10, M01, M11 = _gather_quad(d_ci, row, col)
		frac_h = torch.full_like(row, 0.5, dtype=P.dtype)
		frac_w = torch.full_like(col, 0.5, dtype=P.dtype)
		u1, v1 = fit_model.Model3D._ray_bilinear_intersect(
			P_vi, n_vi, M00, M10, M01, M11, frac_h, frac_w)

		# Shift quad idx based on pass-1, clamp to valid range
		new_h = (row.float() + u1).clamp(0, Hm - 2)
		new_w = (col.float() + v1).clamp(0, Wm - 2)
		new_row = new_h.floor().clamp(0, Hm - 2).long()
		new_col = new_w.floor().clamp(0, Wm - 2).long()
		new_frac_h = new_h - new_row.float()
		new_frac_w = new_w - new_col.float()

		# Pass 2: re-intersect on shifted quad
		M00, M10, M01, M11 = _gather_quad(d_ci, new_row, new_col)
		u2, v2 = fit_model.Model3D._ray_bilinear_intersect(
			P_vi, n_vi, M00, M10, M01, M11, new_frac_h, new_frac_w)

		# Final position and validity
		final_h = (new_row.float() + u2).nan_to_num_(0.0).clamp(0, Hm - 2)
		final_w = (new_col.float() + v2).nan_to_num_(0.0).clamp(0, Wm - 2)
		anchors_h[vi, ci] = final_h.floor().long()
		anchors_w[vi, ci] = final_w.floor().long()
		# Mark invalid if final u,v outside [0,1]
		in_bounds = (u2 >= 0) & (u2 <= 1) & (v2 >= 0) & (v2 <= 1)
		anchors_valid[vi, ci] = in_bounds

	# --- Re-check bracket for closest pair (indices 0, 1) ---
	has_bracket = anchors_valid[:, 0] & anchors_valid[:, 1]
	if has_bracket.any():
		bi = has_bracket.nonzero(as_tuple=True)[0]
		for ci in [0, 1]:
			d_ci = anchors_d[bi, ci]
			h_ci = anchors_h[bi, ci]
			w_ci = anchors_w[bi, ci]
			M00, M10, M01, M11 = _gather_quad(d_ci, h_ci, w_ci)
			fh = torch.full_like(h_ci, 0.5, dtype=P.dtype)
			fw = torch.full_like(w_ci, 0.5, dtype=P.dtype)
			u_c, v_c = fit_model.Model3D._ray_bilinear_intersect(
				P[bi], gt_n[bi], M00, M10, M01, M11, fh, fw)
			Q_c = _bilinear_interp(M00, M10, M01, M11,
								   u_c.clamp(0, 1), v_c.clamp(0, 1))
			sd = ((P[bi] - Q_c) * gt_n[bi]).sum(dim=-1)
			if ci == 0:
				sd_lo = sd
			else:
				sd_hi = sd
		bracket_lost = (sd_lo * sd_hi) >= 0
		if bracket_lost.any():
			lost = bi[bracket_lost]
			anchors_valid[lost, 0] = False
			anchors_valid[lost, 1] = False

	# --- Update avg pair depth layers if target changed ---
	target_finite = torch.isfinite(target)
	target_safe = torch.where(target_finite, target, torch.zeros_like(target))
	new_avg_lo_d = target_safe.floor().clamp(0, max(D - 2, 0)).long()
	new_avg_hi_d = (new_avg_lo_d + 1).clamp(max=D - 1)
	anchors_d[:, 2] = new_avg_lo_d
	anchors_d[:, 3] = new_avg_hi_d
	# Invalidate avg anchors where target is not finite or layer out of range
	anchors_valid[:, 2] = anchors_valid[:, 2] & target_finite & (new_avg_lo_d >= 0) & (new_avg_lo_d < D)
	anchors_valid[:, 3] = anchors_valid[:, 3] & target_finite & (new_avg_hi_d > new_avg_lo_d)

	return anchors_h, anchors_w, anchors_valid


def _corr_winding_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Winding-observation corr loss with proxy correction."""
	global _dbg_call_count, _last_results
	global _wind_anchors_d, _wind_anchors_h, _wind_anchors_w
	global _wind_anchors_valid, _wind_initialized, _wind_target_per_point
	global _wind_obs_per_point, _wind_reinit_counter, _wind_prev_any_valid

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
	is_absolute = pts_c.is_absolute.to(device=dev)
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
				_wind_anchors_valid, _wind_target_per_point, _wind_obs_per_point,
			) = _wind_brute_force_init(
				P, gt_n, winda, col, is_absolute, xyz_det, res.normals, res.data, strip_samples, Qh, Qw)
		_wind_initialized = True
		_wind_reinit_counter = 0
		_wind_prev_any_valid = None
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
			_wind_obs_per_point = winding_obs.clone()
			_wind_target_per_point = _wind_collection_average(
				winding_obs, winda, col, obs_valid, is_absolute)

			# Phase C: Update anchors
			_wind_anchors_h, _wind_anchors_w, _wind_anchors_valid = _wind_update_anchors(
				P, gt_n, xyz_det,
				_wind_anchors_d, _wind_anchors_h, _wind_anchors_w,
				_wind_anchors_valid, _wind_target_per_point,
				Qh, Qw)

			# Phase C.5: Re-init points that lost all valid anchors
			any_valid = _wind_anchors_valid.any(dim=1)  # (K,)
			just_lost = (_wind_prev_any_valid & ~any_valid) if _wind_prev_any_valid is not None else ~any_valid
			persistent_lost = ~any_valid & ~just_lost
			_wind_reinit_counter += 1
			needs_reinit = just_lost | (persistent_lost & (_wind_reinit_counter % 100 == 0))
			_wind_prev_any_valid = any_valid.clone()

			if needs_reinit.any():
				ri = needs_reinit.nonzero(as_tuple=True)[0]
				n_ri = int(ri.shape[0])
				# Save old brackets before re-init
				old_d0 = _wind_anchors_d[ri, 0].tolist()
				old_d1 = _wind_anchors_d[ri, 1].tolist()
				old_v0 = _wind_anchors_valid[ri, 0].tolist()
				old_v1 = _wind_anchors_valid[ri, 1].tolist()
				(rd, rh, rw, rv, rt, ro) = _wind_brute_force_init(
					P[ri], gt_n[ri], winda[ri], col[ri], is_absolute[ri],
					xyz_det, res.normals, res.data, strip_samples, Qh, Qw)
				_wind_anchors_d[ri] = rd
				_wind_anchors_h[ri] = rh
				_wind_anchors_w[ri] = rw
				_wind_anchors_valid[ri] = rv
				_wind_target_per_point[ri] = rt
				_wind_obs_per_point[ri] = ro
				# Log old → new brackets
				new_d0 = rd[:, 0].tolist()
				new_d1 = rd[:, 1].tolist()
				new_v0 = rv[:, 0].tolist()
				new_v1 = rv[:, 1].tolist()
				print(f"[corr-wind] re-init {n_ri} points (just_lost={int(just_lost.sum())}, "
					  f"persistent={int(persistent_lost.sum())}, step={_wind_reinit_counter})")
				for j in range(n_ri):
					old_b = f"{old_d0[j]}-{old_d1[j]}" if old_v0[j] and old_v1[j] else "---"
					new_b = f"{new_d0[j]}-{new_d1[j]}" if new_v0[j] and new_v1[j] else "---"
					print(f"  pt {int(ri[j])}: bracket {old_b} -> {new_b}")

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
	all_too_far = torch.zeros(K, dtype=torch.bool, device=dev)

	# Per-surface frac weight: tgt_lo (ci=2) gets 1-frac, tgt_hi (ci=3) gets frac.
	# Integer targets naturally zero out the second surface.
	frac_weight = {2: (1.0 - frac), 3: frac}

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
			sw, uw, sv = _wind_strip_integral(P[vi], Q, gt_n[vi], res.data, strip_samples)

			# Target offset
			tgt = tgt_fn(frac[vi])
			err = sw - tgt

			# Single-sided distance cutoff: reject points > 2 windings from surface
			# when not bracketed (above top layer or below bottom layer).
			# Bracketed points skip this — large uw there is just density variation.
			is_bracketed = _wind_anchors_valid[vi, 0] & _wind_anchors_valid[vi, 1]
			too_far = ~is_bracketed & (uw > 2.0)
			all_too_far[vi] |= too_far

			# Store error for results (weighted by frac proximity)
			fw = frac_weight[ci][vi]
			all_err[vi] += err * fw

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

		# Mask by strip validity + distance cutoff, weighted by frac proximity to target
		mask_ci = sv.to(dt) * (~too_far).to(dt) * frac_weight[ci][vi]
		wsum = float(mask_ci.sum().detach().cpu())
		if wsum > 0:
			total_loss = total_loss + (lm * mask_ci).sum()
			total_wsum += wsum
		n_surfaces += 1

	if dbg:
		n_valid = int(target_finite.sum().item())
		rms = float(total_loss.detach().sqrt().item()) if total_wsum > 0 else float("nan")
		print(f"[corr-wind] loss={float(total_loss.detach().item()):.6f}, "
			  f"valid={n_valid}/{K}, rms_proxy={rms:.4f}")

	# Build results — point is valid only if it has a valid tgt anchor and is not too far
	has_tgt_anchor = _wind_anchors_valid[:, 2] | _wind_anchors_valid[:, 3]
	point_valid = target_finite & has_tgt_anchor & ~all_too_far
	_last_results = _build_winding_results(
		winding_obs=_wind_obs_per_point, target=target,
		err=all_err, pt_ids=pt_ids, col=col, pts=pts, winda=winda,
		valid=point_valid, is_absolute=is_absolute,
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
	is_absolute: torch.Tensor,
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
			"absolute": bool(is_absolute[i]),
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
