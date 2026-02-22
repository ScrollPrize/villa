from __future__ import annotations

import argparse
import json
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


def load_points_tensor(cfg: PointConstraintsConfig) -> tuple[torch.Tensor, torch.Tensor]:
	rows: list[list[float]] = []
	cids: list[int] = []
	for pth in cfg.points:
		obj = json.loads(Path(pth).read_text(encoding="utf-8"))
		cols = obj.get("collections", {}) if isinstance(obj, dict) else {}
		if not isinstance(cols, dict):
			continue
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
				pv = pd.get("p", None)
				if not isinstance(pv, (list, tuple)) or len(pv) < 3:
					continue
				try:
					cid_i = int(_cid)
				except Exception:
					cid_i = -1
				rows.append([
					float(pv[0]),
					float(pv[1]),
					float(pv[2]),
					float(pd["wind_a"]),
				])
				cids.append(cid_i)
	pts = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 4), dtype=torch.float32)
	col_idx = torch.tensor(cids, dtype=torch.int64) if cids else torch.empty((0,), dtype=torch.int64)
	return pts, col_idx


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Return closest enclosing LR-segment indices for each annotation point.

	- points_xyz_winda: (K,4) as [x,y,z,wind_a]
	- xy_conn: (N,Hm,Wm,3,2) in order [left,point,right]
	Returns:
	- points_all: (K,4) all input points in working coords
	- idx_left: (K,3) [z,row_top,col]
	- valid_left: (K,) bool
	- min_dist_left: (K,) float32 (pixel distance to chosen segment projection; inf if invalid)
	- idx_right: (K,3) [z,row_top,col]
	- valid_right: (K,) bool
	- min_dist_right: (K,) float32 (pixel distance to chosen segment projection; inf if invalid)
	where each segment is between (row_top,col) and (row_top+1,col).
	"""
	if points_xyz_winda.ndim != 2 or int(points_xyz_winda.shape[1]) < 3:
		return (
			torch.empty((0, 4), dtype=torch.float32, device=xy_conn.device),
			torch.empty((0, 3), dtype=torch.int64, device=xy_conn.device),
			torch.empty((0,), dtype=torch.bool, device=xy_conn.device),
			torch.empty((0,), dtype=torch.float32, device=xy_conn.device),
			torch.empty((0, 3), dtype=torch.int64, device=xy_conn.device),
			torch.empty((0,), dtype=torch.bool, device=xy_conn.device),
			torch.empty((0,), dtype=torch.float32, device=xy_conn.device),
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
	idx_left = torch.empty((k, 3), dtype=torch.int64, device=xy_conn.device)
	ok_left = torch.zeros((k,), dtype=torch.bool, device=xy_conn.device)
	dist_left = torch.full((k,), float("inf"), dtype=torch.float32, device=xy_conn.device)
	idx_right = torch.empty((k, 3), dtype=torch.int64, device=xy_conn.device)
	ok_right = torch.zeros((k,), dtype=torch.bool, device=xy_conn.device)
	dist_right = torch.full((k,), float("inf"), dtype=torch.float32, device=xy_conn.device)

	def _cross_z(a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
		return a2[..., 0] * b2[..., 1] - a2[..., 1] * b2[..., 0]

	def _side_d2(*, conn_side: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
		"""Compute filtered d2 (hm-1, wm) for one side with inside test."""
		c0 = conn_side[:-1, :, :]
		c1 = conn_side[1:, :, :]

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

	for i in range(k):
		z = int(torch.round(pts[i, 2]).item())
		z = max(0, min(n - 1, z))
		q = pts[i, 0:2].view(1, 1, 2)

		p0 = center[z, :-1, :, :]
		p1 = center[z, 1:, :, :]

		d2_r = _side_d2(conn_side=rightp[z], p0=p0, p1=p1, q=q)  # (hm-1, wm)
		d2_l = _side_d2(conn_side=leftp[z], p0=p0, p1=p1, q=q)

		# Per-column (per-winding) minimum distance
		min_r, argmin_r = d2_r.min(dim=0)  # (wm,)
		min_l, argmin_l = d2_l.min(dim=0)

		# Joint selection: point is between winding c (right side) and
		# winding c+1 (left side).  Pick the pair with min combined dist.
		if wm < 2:
			idx_left[i] = torch.tensor([z, -1, -1], dtype=torch.int64)
			idx_right[i] = torch.tensor([z, -1, -1], dtype=torch.int64)
			continue

		combined = min_r[:-1] + min_l[1:]  # (wm-1,)
		best_gap = int(combined.argmin().item())

		if not torch.isfinite(combined[best_gap]):
			idx_left[i] = torch.tensor([z, -1, -1], dtype=torch.int64)
			idx_right[i] = torch.tensor([z, -1, -1], dtype=torch.int64)
			continue

		c_r = best_gap
		c_l = best_gap + 1
		idx_right[i, 0] = z
		idx_right[i, 1] = int(argmin_r[c_r].item())
		idx_right[i, 2] = c_r
		ok_right[i] = True
		dist_right[i] = torch.sqrt(min_r[c_r]).to(dtype=torch.float32)

		idx_left[i, 0] = z
		idx_left[i, 1] = int(argmin_l[c_l].item())
		idx_left[i, 2] = c_l
		ok_left[i] = True
		dist_left[i] = torch.sqrt(min_l[c_l]).to(dtype=torch.float32)
	return pts, idx_left, ok_left, dist_left, idx_right, ok_right, dist_right


def print_closest_conn_segments(*, points_xyz_winda: torch.Tensor, xy_conn: torch.Tensor) -> None:
	pts_all, idx_l, ok_l, d_l, idx_r, ok_r, d_r = closest_conn_segment_indices(points_xyz_winda=points_xyz_winda, xy_conn=xy_conn)
	print("[point_constraints] points_xyz_winda_work_all", pts_all)
	print("[point_constraints] closest_conn_left[z,row,colL]", idx_l)
	print("[point_constraints] closest_conn_left_valid", ok_l)
	print("[point_constraints] closest_conn_left_min_dist_px", d_l)
	print("[point_constraints] closest_conn_right[z,row,colL]", idx_r)
	print("[point_constraints] closest_conn_right_valid", ok_r)
	print("[point_constraints] closest_conn_right_min_dist_px", d_r)


def winding_observed_and_error(
	*,
	points_xyz_winda: torch.Tensor,
	collection_idx: torch.Tensor,
	xy_conn: torch.Tensor,
	idx_left: torch.Tensor,
	valid_left: torch.Tensor,
	idx_right: torch.Tensor,
	valid_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Return per-point observed winding, per-point collection average, and winding error.

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

	center = xy_conn[:, :, :, 1, :]
	leftp = xy_conn[:, :, :, 0, :]
	rightp = xy_conn[:, :, :, 2, :]
	n, hm, wm = int(center.shape[0]), int(center.shape[1]), int(center.shape[2])

	def _xfrac_for_side(*, side_pts: torch.Tensor, idx: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
		xf = torch.full((int(pts.shape[0]),), float("nan"), dtype=torch.float32, device=xy_conn.device)
		for i in range(int(pts.shape[0])):
			if not bool(ok[i].item()):
				continue
			z = int(idx[i, 0].item())
			r = int(idx[i, 1].item())
			c = int(idx[i, 2].item())
			if z < 0 or z >= n or r < 0 or (r + 1) >= hm or c < 0 or c >= wm:
				continue
			q = pts[i, 0:2]
			p0 = center[z, r, c]
			p1 = center[z, r + 1, c]
			c0 = side_pts[z, r, c]
			c1 = side_pts[z, r + 1, c]
			s = p1 - p0
			l2 = float((s * s).sum().item())
			if l2 <= 1e-12:
				continue
			a = float(((q - p0) * s).sum().item()) / l2
			a = max(0.0, min(1.0, a))
			cp = p0 + s * a
			v0 = c0 - p0
			v1 = c1 - p1
			v = v0 * (1.0 - a) + v1 * a
			v2 = float((v * v).sum().item())
			if v2 <= 1e-12:
				continue
			xf[i] = float(((q - cp) * v).sum().item()) / v2
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
