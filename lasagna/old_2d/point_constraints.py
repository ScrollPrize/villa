from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class PointConstraintsConfig:
	points: list[str]


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("point_constraints")
	g.add_argument(
		"--points",
		action="append",
		default=[],
		help="Point-annotation json file; can be passed multiple times.",
	)


def from_args(args: argparse.Namespace) -> PointConstraintsConfig:
	raw = getattr(args, "points", [])
	if raw in (None, ""):
		paths: list[str] = []
	elif isinstance(raw, str):
		paths = [str(raw)]
	else:
		paths = [str(p) for p in raw]
	return PointConstraintsConfig(points=paths)


def _collect_from_obj(obj: object, rows: list[list[float]], cids: list[int], pids: list[int]) -> None:
	"""Parse a collections dict and append points to rows/cids/pids."""
	cols = obj.get("collections", {}) if isinstance(obj, dict) else {}
	if not isinstance(cols, dict):
		return
	for _cid, col in cols.items():
		if not isinstance(col, dict):
			continue
		md = col.get("metadata", {})
		if not isinstance(md, dict):
			continue
		if bool(md.get("winding_is_absolute", True)):
			continue
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		for _pid, pd in pts.items():
			if not isinstance(pd, dict) or "wind_a" not in pd:
				continue
			wa = pd["wind_a"]
			if wa is None:
				continue
			pv = pd.get("p", None)
			if not isinstance(pv, (list, tuple)) or len(pv) < 3:
				continue
			try:
				cid_i = int(_cid)
			except Exception:
				cid_i = -1
			try:
				pid_i = int(_pid)
			except Exception:
				pid_i = -1
			rows.append([
				float(pv[0]),
				float(pv[1]),
				float(pv[2]),
				float(wa),
			])
			cids.append(cid_i)
			pids.append(pid_i)


def _rows_to_tensors(rows: list[list[float]], cids: list[int], pids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	pts = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 4), dtype=torch.float32)
	col_idx = torch.tensor(cids, dtype=torch.int64) if cids else torch.empty((0,), dtype=torch.int64)
	pt_ids = torch.tensor(pids, dtype=torch.int64) if pids else torch.empty((0,), dtype=torch.int64)
	return pts, col_idx, pt_ids


def load_points_tensor(cfg: PointConstraintsConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	rows: list[list[float]] = []
	cids: list[int] = []
	pids: list[int] = []
	for pth in cfg.points:
		obj = json.loads(Path(pth).read_text(encoding="utf-8"))
		_collect_from_obj(obj, rows, cids, pids)
	return _rows_to_tensors(rows, cids, pids)


def load_points_from_collections_dict(obj: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Load points from an inline collections dict (same format as file)."""
	rows: list[list[float]] = []
	cids: list[int] = []
	pids: list[int] = []
	_collect_from_obj(obj, rows, cids, pids)
	return _rows_to_tensors(rows, cids, pids)


def print_points_tensor(t: torch.Tensor) -> None:
	print("[point_constraints] points_xyz_winda", t)


def to_working_coords(
	*,
	points_xyz_winda: torch.Tensor,
	downscale: float,
	crop_xywh: tuple[int, int, int, int] | None,
	z0: int | None,
	z_step: int,
	z_size: int,
) -> torch.Tensor:
	"""Map full-res annotation points to working crop/scale coordinates.

	- x,y: subtract crop offset then divide by downscale
	- z: convert to local z-index using (z - z0) / z_step
	"""
	if points_xyz_winda.numel() == 0:
		return points_xyz_winda
	out = points_xyz_winda.clone().to(dtype=torch.float32)
	ds = float(downscale) if float(downscale) > 0.0 else 1.0
	if crop_xywh is not None:
		cx, cy, _cw, _ch = (int(v) for v in crop_xywh)
		out[:, 0] = (out[:, 0] - float(cx)) / ds
		out[:, 1] = (out[:, 1] - float(cy)) / ds
	else:
		out[:, 0] = out[:, 0] / ds
		out[:, 1] = out[:, 1] / ds
	zs = max(1, int(z_step))
	zbase = 0 if z0 is None else int(z0)
	out[:, 2] = (out[:, 2] - float(zbase)) / float(zs)
	if int(z_size) > 0:
		out[:, 2] = out[:, 2].clamp(0.0, float(int(z_size) - 1))
	return out


def closest_conn_segment_indices(
	*,
	points_xyz_winda: torch.Tensor,
	xy_conn: torch.Tensor,
) -> tuple[
	torch.Tensor,  # points_all (K,4)
	torch.Tensor, torch.Tensor, torch.Tensor,  # idx_left, valid_left, min_dist_left
	torch.Tensor, torch.Tensor, torch.Tensor,  # idx_right, valid_right, min_dist_right
	torch.Tensor,  # z_hi (K,)
	torch.Tensor,  # z_frac (K,)
]:
	"""Return closest enclosing LR-segment indices on z-interpolated mesh.

	For each point, linearly interpolates the mesh between z_lo=floor(z)
	and z_hi=ceil(z), then searches that single interpolated mesh.
	idx_left/idx_right store z_lo in [:, 0] for loss map distribution.

	- points_xyz_winda: (K,4) as [x,y,z,wind_a]
	- xy_conn: (N,Hm,Wm,3,2) in order [left,point,right]
	"""
	_empty = lambda *shape, dtype=torch.float32: torch.empty(shape, dtype=dtype, device=xy_conn.device)
	if points_xyz_winda.ndim != 2 or int(points_xyz_winda.shape[1]) < 3:
		return (
			_empty(0, 4),
			_empty(0, 3, dtype=torch.int64), _empty(0, dtype=torch.bool), _empty(0),
			_empty(0, 3, dtype=torch.int64), _empty(0, dtype=torch.bool), _empty(0),
			_empty(0, dtype=torch.int64),
			_empty(0),
		)
	if xy_conn.ndim != 5 or int(xy_conn.shape[3]) != 3 or int(xy_conn.shape[4]) != 2:
		raise ValueError("xy_conn must be (N,Hm,Wm,3,2)")

	pts = points_xyz_winda.to(device=xy_conn.device, dtype=torch.float32)
	n, hm, wm = int(xy_conn.shape[0]), int(xy_conn.shape[1]), int(xy_conn.shape[2])
	if hm <= 1 or wm <= 0:
		raise ValueError("xy_conn must have Hm>1 and Wm>0")

	center = xy_conn[:, :, :, 1, :]
	leftp = xy_conn[:, :, :, 0, :]
	rightp = xy_conn[:, :, :, 2, :]

	k = int(pts.shape[0])
	dev = xy_conn.device

	def _cross_z(a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
		return a2[..., 0] * b2[..., 1] - a2[..., 1] * b2[..., 0]

	def _batch_side_d2(*, conn_side: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
		"""Batched _side_d2: all inputs have leading K dimension.

		conn_side: (K, Hm, Wm, 2), p0/p1: (K, Hm-1, Wm, 2), q: (K, 1, 1, 2).
		Returns (K, Hm-1, Wm).
		"""
		c0 = conn_side[:, :-1, :, :]
		c1 = conn_side[:, 1:, :, :]

		s = p1 - p0
		l2 = (s * s).sum(dim=-1).clamp_min(1e-12)
		qp0 = q - p0
		a = ((qp0 * s).sum(dim=-1) / l2).clamp(0.0, 1.0)
		c = p0 + s * a.unsqueeze(-1)
		v0 = c0 - p0
		v1 = c1 - p1

		dot01 = (v0 * v1).sum(dim=-1)
		same_dir = dot01 > 0.0
		probe = 0.5 * (p0 + p1) + 0.5 * (v0 + v1)

		s_q = _cross_z(s, q - p0)
		s_p = _cross_z(s, probe - p0)
		u_q = _cross_z(v0, q - p0)
		u_p = _cross_z(v0, probe - p0)
		l_q = _cross_z(v1, q - p1)
		l_p = _cross_z(v1, probe - p1)
		conn_inside = (s_q * s_p >= 0.0) & (u_q * u_p >= 0.0) & (l_q * l_p >= 0.0) & same_dir

		nrm = torch.stack([-s[..., 1], s[..., 0]], dim=-1)
		n_u_q = ((q - p0) * nrm).sum(dim=-1)
		n_u_v = (v0 * nrm).sum(dim=-1)
		n_l_q = ((q - p1) * nrm).sum(dim=-1)
		n_l_v = (v1 * nrm).sum(dim=-1)
		normal_inside = (s_q * s_p >= 0.0) & ((n_u_q * n_u_v >= 0.0) | (n_l_q * n_l_v >= 0.0)) & same_dir

		inside = conn_inside | normal_inside

		d2 = ((q - c) * (q - c)).sum(dim=-1)
		return torch.where(inside, d2, torch.full_like(d2, float("inf")))

	def _batch_search_interp(*, center_k: torch.Tensor, leftp_k: torch.Tensor, rightp_k: torch.Tensor, z_ref: torch.Tensor):
		"""Search for best gap on pre-gathered per-point mesh slices.

		center_k, leftp_k, rightp_k: (K, Hm, Wm, 2) interpolated mesh.
		z_ref: (K,) long tensor stored in idx[:, 0] for loss map distribution.
		Returns (idx_l, ok_l, dist_l, idx_r, ok_r, dist_r) all with leading K dim.
		"""
		p0 = center_k[:, :-1, :, :]      # (K, Hm-1, Wm, 2)
		p1 = center_k[:, 1:, :, :]       # (K, Hm-1, Wm, 2)
		q = pts[:, 0:2].view(k, 1, 1, 2)

		d2_r = _batch_side_d2(conn_side=rightp_k, p0=p0, p1=p1, q=q)  # (K, Hm-1, Wm)
		d2_l = _batch_side_d2(conn_side=leftp_k, p0=p0, p1=p1, q=q)   # (K, Hm-1, Wm)

		min_r, argmin_r = d2_r.min(dim=1)  # (K, Wm)
		min_l, argmin_l = d2_l.min(dim=1)  # (K, Wm)

		ki = torch.arange(k, device=dev)

		if wm < 2:
			idx = torch.stack([z_ref, torch.full_like(z_ref, -1), torch.full_like(z_ref, -1)], dim=-1)
			ok = torch.zeros(k, dtype=torch.bool, device=dev)
			dist = torch.full((k,), float("inf"), dtype=torch.float32, device=dev)
			return idx, ok, dist, idx.clone(), ok.clone(), dist.clone()

		combined = min_r[:, :-1] + min_l[:, 1:]  # (K, Wm-1)
		best_gap = combined.argmin(dim=1)          # (K,)
		best_val = combined[ki, best_gap]          # (K,)
		gap_finite = torch.isfinite(best_val)

		c_r = best_gap          # (K,) right column
		c_l = best_gap + 1      # (K,) left column
		row_r = argmin_r[ki, c_r]  # (K,)
		row_l = argmin_l[ki, c_l]  # (K,)

		idx_l = torch.stack([z_ref, row_l, c_l], dim=-1)   # (K, 3)
		idx_r = torch.stack([z_ref, row_r, c_r], dim=-1)   # (K, 3)
		dist_l = torch.sqrt(min_l[ki, c_l])  # (K,)
		dist_r = torch.sqrt(min_r[ki, c_r])  # (K,)

		inv_idx = torch.stack([z_ref, torch.full_like(z_ref, -1), torch.full_like(z_ref, -1)], dim=-1)
		idx_l = torch.where(gap_finite.unsqueeze(-1), idx_l, inv_idx)
		idx_r = torch.where(gap_finite.unsqueeze(-1), idx_r, inv_idx)
		dist_l = torch.where(gap_finite, dist_l, torch.full_like(dist_l, float("inf")))
		dist_r = torch.where(gap_finite, dist_r, torch.full_like(dist_r, float("inf")))

		return idx_l, gap_finite.clone(), dist_l, idx_r, gap_finite.clone(), dist_r

	# Vectorized z computation for all K points
	z_f = pts[:, 2]
	z_lo = z_f.floor().long().clamp(0, n - 1)
	z_hi = z_f.ceil().long().clamp(0, n - 1)
	frac = z_f - z_f.floor()
	z_frac_out = torch.where(z_lo == z_hi, torch.zeros_like(frac), frac)

	# Interpolate mesh per-point
	z_lo_c = z_lo.clamp(0, n - 1)
	z_hi_c = z_hi.clamp(0, n - 1)
	frac_view = z_frac_out.view(k, 1, 1, 1)  # (K,1,1,1) for broadcasting
	center_interp = center[z_lo_c] * (1.0 - frac_view) + center[z_hi_c] * frac_view  # (K,Hm,Wm,2)
	leftp_interp = leftp[z_lo_c] * (1.0 - frac_view) + leftp[z_hi_c] * frac_view
	rightp_interp = rightp[z_lo_c] * (1.0 - frac_view) + rightp[z_hi_c] * frac_view

	# Single search on interpolated mesh
	idx_left, ok_left, dist_left, idx_right, ok_right, dist_right = _batch_search_interp(
		center_k=center_interp, leftp_k=leftp_interp, rightp_k=rightp_interp, z_ref=z_lo)

	return (
		pts,
		idx_left, ok_left, dist_left,
		idx_right, ok_right, dist_right,
		z_hi,
		z_frac_out,
	)


def print_closest_conn_segments(*, points_xyz_winda: torch.Tensor, xy_conn: torch.Tensor) -> None:
	(pts_all,
	 idx_l, ok_l, d_l, idx_r, ok_r, d_r,
	 z_hi, z_frac) = closest_conn_segment_indices(points_xyz_winda=points_xyz_winda, xy_conn=xy_conn)
	print("[point_constraints] points_xyz_winda_work_all", pts_all)
	print("[point_constraints] closest_conn_left[z_lo,row,colL]", idx_l)
	print("[point_constraints] closest_conn_left_valid", ok_l)
	print("[point_constraints] closest_conn_left_min_dist_px", d_l)
	print("[point_constraints] closest_conn_right[z_lo,row,colL]", idx_r)
	print("[point_constraints] closest_conn_right_valid", ok_r)
	print("[point_constraints] closest_conn_right_min_dist_px", d_r)
	print("[point_constraints] z_hi", z_hi)
	print("[point_constraints] z_frac", z_frac)


def winding_observed_and_error(
	*,
	points_xyz_winda: torch.Tensor,
	collection_idx: torch.Tensor,
	xy_conn: torch.Tensor,
	idx_left: torch.Tensor,
	valid_left: torch.Tensor,
	idx_right: torch.Tensor,
	valid_right: torch.Tensor,
	z_hi: torch.Tensor,
	z_frac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Return per-point observed winding, per-point collection average, and winding error.

	Uses z-interpolated mesh lookups (z_lo from idx[:, 0], z_hi, z_frac).

	- observed winding uses both sides when valid:
	  left estimate:  col_left - xfrac_left
	  right estimate: col_right + xfrac_right
	  observed = 0.5 * (left_est + right_est)
	- points without both valid sides are ignored for collection averages.
	- error = observed - (collection_avg + wind_a)
	"""
	if int(points_xyz_winda.shape[0]) <= 0:
		empty = torch.empty((0,), dtype=torch.float32, device=xy_conn.device)
		return empty, empty, empty
	pts = points_xyz_winda.to(device=xy_conn.device, dtype=torch.float32)
	col = collection_idx.to(device=xy_conn.device, dtype=torch.int64)
	idx_l = idx_left.to(device=xy_conn.device, dtype=torch.int64)
	idx_r = idx_right.to(device=xy_conn.device, dtype=torch.int64)
	ok_l = valid_left.to(device=xy_conn.device, dtype=torch.bool)
	ok_r = valid_right.to(device=xy_conn.device, dtype=torch.bool)
	z_hi_t = z_hi.to(device=xy_conn.device, dtype=torch.int64)
	z_frac_t = z_frac.to(device=xy_conn.device, dtype=torch.float32)

	center = xy_conn[:, :, :, 1, :]
	leftp = xy_conn[:, :, :, 0, :]
	rightp = xy_conn[:, :, :, 2, :]
	n, hm, wm = int(center.shape[0]), int(center.shape[1]), int(center.shape[2])

	def _xfrac_for_side(*, side_pts: torch.Tensor, idx: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
		kk = int(pts.shape[0])
		xf = torch.full((kk,), float("nan"), dtype=torch.float32, device=xy_conn.device)
		zi = idx[:, 0]
		ri = idx[:, 1]
		ci = idx[:, 2]
		valid = ok & (zi >= 0) & (zi < n) & (ri >= 0) & (ri + 1 < hm) & (ci >= 0) & (ci < wm)
		if not valid.any():
			return xf
		zi_s = zi.clamp(0, max(n - 1, 0))
		z_hi_s = z_hi_t.clamp(0, max(n - 1, 0))
		ri_s = ri.clamp(0, max(hm - 2, 0))
		ci_s = ci.clamp(0, max(wm - 1, 0))
		fk = z_frac_t.unsqueeze(-1)  # (K, 1)
		q = pts[:, 0:2]
		p0 = center[zi_s, ri_s, ci_s] * (1.0 - fk) + center[z_hi_s, ri_s, ci_s] * fk
		p1 = center[zi_s, ri_s + 1, ci_s] * (1.0 - fk) + center[z_hi_s, ri_s + 1, ci_s] * fk
		c0 = side_pts[zi_s, ri_s, ci_s] * (1.0 - fk) + side_pts[z_hi_s, ri_s, ci_s] * fk
		c1 = side_pts[zi_s, ri_s + 1, ci_s] * (1.0 - fk) + side_pts[z_hi_s, ri_s + 1, ci_s] * fk
		s = p1 - p0
		l2 = (s * s).sum(dim=-1)
		valid = valid & (l2 > 1e-12)
		l2 = l2.clamp_min(1e-12)
		a = (((q - p0) * s).sum(dim=-1) / l2).clamp(0.0, 1.0)
		cp = p0 + s * a.unsqueeze(-1)
		v0 = c0 - p0
		v1 = c1 - p1
		v = v0 * (1.0 - a).unsqueeze(-1) + v1 * a.unsqueeze(-1)
		v2 = (v * v).sum(dim=-1)
		valid = valid & (v2 > 1e-12)
		v2 = v2.clamp_min(1e-12)
		result = ((q - cp) * v).sum(dim=-1) / v2
		xf[valid] = result[valid]
		return xf

	xf_l = _xfrac_for_side(side_pts=leftp, idx=idx_l, ok=ok_l)
	xf_r = _xfrac_for_side(side_pts=rightp, idx=idx_r, ok=ok_r)
	w_left = idx_l[:, 2].to(dtype=torch.float32) - xf_l
	w_right = idx_r[:, 2].to(dtype=torch.float32) + xf_r
	both = ok_l & ok_r & torch.isfinite(w_left) & torch.isfinite(w_right)
	obs = torch.full((int(pts.shape[0]),), float("nan"), dtype=torch.float32, device=xy_conn.device)
	obs[both] = 0.5 * (w_left[both] + w_right[both])

	avg = torch.full_like(obs, float("nan"))
	if int(col.numel()) == int(obs.numel()) and int(obs.numel()) > 0:
		uc = torch.unique(col)
		for cid in uc.tolist():
			m = (col == int(cid)) & both
			if bool(m.any().item()):
				obs_m = obs[m]
				wa_m = pts[m, 3]
				avg_pos = torch.nanmean(obs_m - wa_m)
				err_pos = obs_m - (avg_pos + wa_m)
				mse_pos = torch.nanmean(err_pos * err_pos)
				avg_neg = torch.nanmean(obs_m + wa_m)
				err_neg = obs_m - (avg_neg - wa_m)
				mse_neg = torch.nanmean(err_neg * err_neg)
				use_neg = bool((mse_neg < mse_pos).item())
				avg[col == int(cid)] = avg_neg if use_neg else avg_pos

	err = torch.full_like(obs, float("nan"))
	if int(col.numel()) == int(obs.numel()) and int(obs.numel()) > 0:
		uc = torch.unique(col)
		for cid in uc.tolist():
			m = (col == int(cid)) & both
			if not bool(m.any().item()):
				continue
			obs_m = obs[m]
			wa_m = pts[m, 3]
			avg_m = avg[m]
			err_pos = obs_m - (avg_m + wa_m)
			err_neg = obs_m - (avg_m - wa_m)
			mse_pos = torch.nanmean(err_pos * err_pos)
			mse_neg = torch.nanmean(err_neg * err_neg)
			use_neg = bool((mse_neg < mse_pos).item())
			err[m] = err_neg if use_neg else err_pos
	return obs, avg, err
