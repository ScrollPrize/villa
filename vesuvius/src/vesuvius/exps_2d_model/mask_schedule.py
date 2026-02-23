from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import model as fit_model


@dataclass(frozen=True)
class MaskSpec:
	label: str
	type: str
	params: dict
	for_losses: list[str]


def _lerp(a: float, b: float, t: float) -> float:
	t = float(max(0.0, min(1.0, t)))
	return (1.0 - t) * float(a) + t * float(b)


def _interp_cols_xy(*, xy_row: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
	"""Interpolate along the width axis of `xy_row`.

	- xy_row: (N,W,2)
	- x_pos: (N,) in [0, W-1] (float)
	Return: (N,2)
	"""
	n, w, c2 = (int(v) for v in xy_row.shape)
	if c2 != 2:
		raise ValueError("xy_row must be (N,W,2)")
	wf = float(max(1, w - 1))
	x = x_pos.clamp(0.0, wf)
	x0 = torch.floor(x).to(dtype=torch.int64)
	x1 = torch.minimum(x0 + 1, x0.new_full(x0.shape, w - 1))
	t = (x - x0.to(dtype=x.dtype)).view(n, 1)
	p0 = xy_row[torch.arange(n, device=xy_row.device), x0]
	p1 = xy_row[torch.arange(n, device=xy_row.device), x1]
	return p0 + (p1 - p0) * t


def _interp_rows_xy(*, xy_col: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
	"""Interpolate along the height axis of `xy_col`.

	- xy_col: (N,H,2)
	- y_pos: (N,) in [0, H-1] (float)
	Return: (N,2)
	"""
	n, h, c2 = (int(v) for v in xy_col.shape)
	if c2 != 2:
		raise ValueError("xy_col must be (N,H,2)")
	hf = float(max(1, h - 1))
	y = y_pos.clamp(0.0, hf)
	y0 = torch.floor(y).to(dtype=torch.int64)
	y1 = torch.minimum(y0 + 1, y0.new_full(y0.shape, h - 1))
	t = (y - y0.to(dtype=y.dtype)).view(n, 1)
	p0 = xy_col[torch.arange(n, device=xy_col.device), y0]
	p1 = xy_col[torch.arange(n, device=xy_col.device), y1]
	return p0 + (p1 - p0) * t


def _interp_cols_xy_conn_dir(*, xy_conn_row: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
	"""Interpolate conn direction (L->R) along width.

	- xy_conn_row: (N,W,3,2)
	- x_pos: (N,)
	Return: (N,2) direction vector
	"""
	n, w, k3, c2 = (int(v) for v in xy_conn_row.shape)
	if k3 != 3 or c2 != 2:
		raise ValueError("xy_conn_row must be (N,W,3,2)")
	wf = float(max(1, w - 1))
	x = x_pos.clamp(0.0, wf)
	x0 = torch.floor(x).to(dtype=torch.int64)
	x1 = torch.minimum(x0 + 1, x0.new_full(x0.shape, w - 1))
	t = (x - x0.to(dtype=x.dtype)).view(n, 1)
	idx = torch.arange(n, device=xy_conn_row.device)
	v0 = (xy_conn_row[idx, x0, 2] - xy_conn_row[idx, x0, 0]).to(dtype=torch.float32)
	v1 = (xy_conn_row[idx, x1, 2] - xy_conn_row[idx, x1, 0]).to(dtype=torch.float32)
	return v0 + (v1 - v0) * t


def _interp_rows_xy_conn_dir(*, xy_conn_col: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
	"""Interpolate conn direction (L->R) along height.

	- xy_conn_col: (N,H,3,2)
	- y_pos: (N,)
	Return: (N,2) direction vector
	"""
	n, h, k3, c2 = (int(v) for v in xy_conn_col.shape)
	if k3 != 3 or c2 != 2:
		raise ValueError("xy_conn_col must be (N,H,3,2)")
	hf = float(max(1, h - 1))
	y = y_pos.clamp(0.0, hf)
	y0 = torch.floor(y).to(dtype=torch.int64)
	y1 = torch.minimum(y0 + 1, y0.new_full(y0.shape, h - 1))
	t = (y - y0.to(dtype=y.dtype)).view(n, 1)
	idx = torch.arange(n, device=xy_conn_col.device)
	v0 = (xy_conn_col[idx, y0, 2] - xy_conn_col[idx, y0, 0]).to(dtype=torch.float32)
	v1 = (xy_conn_col[idx, y1, 2] - xy_conn_col[idx, y1, 0]).to(dtype=torch.float32)
	return v0 + (v1 - v0) * t


def _line_rect_intersections(*, p: tuple[float, float], v: tuple[float, float], w: int, h: int) -> list[tuple[float, float]]:
	"""Return intersections of infinite line p+t*v with image rectangle [0,w-1]x[0,h-1]."""
	x0, y0 = float(p[0]), float(p[1])
	vx, vy = float(v[0]), float(v[1])
	if w <= 1 or h <= 1:
		return []
	xmin, xmax = 0.0, float(w - 1)
	ymin, ymax = 0.0, float(h - 1)
	out: list[tuple[float, float]] = []

	def _add(t: float) -> None:
		x = x0 + t * vx
		y = y0 + t * vy
		if (xmin - 1e-6) <= x <= (xmax + 1e-6) and (ymin - 1e-6) <= y <= (ymax + 1e-6):
			out.append((x, y))

	if abs(vx) > 1e-12:
		_add((xmin - x0) / vx)
		_add((xmax - x0) / vx)
	if abs(vy) > 1e-12:
		_add((ymin - y0) / vy)
		_add((ymax - y0) / vy)

	uniq: list[tuple[float, float]] = []
	for x, y in out:
		ok = True
		for xu, yu in uniq:
			if abs(x - xu) < 1e-4 and abs(y - yu) < 1e-4:
				ok = False
				break
		if ok:
			uniq.append((x, y))
	return uniq


def _ray_rect_intersection(*, p: tuple[float, float], v: tuple[float, float], w: int, h: int) -> tuple[float, float] | None:
	"""Return the first intersection of the ray p+t*v (t>=0) with the image rectangle."""
	x0, y0 = float(p[0]), float(p[1])
	vx, vy = float(v[0]), float(v[1])
	if w <= 1 or h <= 1:
		return None
	xmin, xmax = 0.0, float(w - 1)
	ymin, ymax = 0.0, float(h - 1)
	best_t = None
	best_xy = None

	def _try(t: float) -> None:
		nonlocal best_t, best_xy
		if t < 0.0:
			return
		x = x0 + t * vx
		y = y0 + t * vy
		if (xmin - 1e-6) <= x <= (xmax + 1e-6) and (ymin - 1e-6) <= y <= (ymax + 1e-6):
			if best_t is None or t < best_t:
				best_t = float(t)
				best_xy = (float(x), float(y))

	if abs(vx) > 1e-12:
		_try((xmin - x0) / vx)
		_try((xmax - x0) / vx)
	if abs(vy) > 1e-12:
		_try((ymin - y0) / vy)
		_try((ymax - y0) / vy)
	return best_xy


def _rect_perimeter_s(*, p: tuple[float, float], w: int, h: int) -> float:
	"""Return clockwise perimeter coordinate for a point on the rectangle border."""
	x, y = float(p[0]), float(p[1])
	w1 = float(max(1, int(w) - 1))
	h1 = float(max(1, int(h) - 1))
	if abs(y - 0.0) <= 1e-3:
		return x
	if abs(x - w1) <= 1e-3:
		return w1 + y
	if abs(y - h1) <= 1e-3:
		return w1 + th1 + (w1 - x)
	return w1 + th1 + w1 + (th1 - y)


def _rect_border_path(
	*,
	p0: tuple[float, float],
	p1: tuple[float, float],
	w: int,
	h: int,
	cw: bool,
) -> list[tuple[float, float]]:
	"""Return border polyline from p0 to p1 along rectangle border (inclusive endpoints).

	Adds intermediate rectangle corners as needed.
	"""
	w1 = float(max(1, int(w) - 1))
	h1 = float(max(1, int(h) - 1))
	L = 2.0 * (w1 + th1)
	s0 = _rect_perimeter_s(p=p0, w=w, h=h)
	s1 = _rect_perimeter_s(p=p1, w=w, h=h)
	if cw:
		if s1 < s0:
			s1 += L
		corners = [(w1, 0.0), (w1, th1), (0.0, th1), (0.0, 0.0), (w1, 0.0)]
		spath = [w1, w1 + th1, w1 + th1 + w1, L, L + w1]
	else:
		if s1 > s0:
			s0 += L
		corners = [(0.0, 0.0), (0.0, th1), (w1, th1), (w1, 0.0), (0.0, 0.0)]
		spath = [L, w1 + th1 + w1, w1 + th1, w1, 0.0]

	out: list[tuple[float, float]] = [p0]
	for sc, c in zip(spath, corners, strict=True):
		sc2 = float(sc)
		if cw:
			if s0 < sc2 < s1:
				out.append(c)
		else:
			if s1 < sc2 < s0:
				out.append(c)
	out.append(p1)
	return out


def _clip_poly_halfplane(
	*,
	poly: list[tuple[float, float]],
	p: tuple[float, float],
	v: tuple[float, float],
	keep_xy: tuple[float, float],
) -> list[tuple[float, float]]:
	"""Clip polygon by the half-plane defined by the infinite line through p with direction v.

	The kept side is the one containing keep_xy.
	"""
	if not poly:
		return []
	px, py = float(p[0]), float(p[1])
	vx, vy = float(v[0]), float(v[1])
	# Line normal (perp to v).
	nx, ny = -vy, vx

	def s(x: float, y: float) -> float:
		return (x - px) * nx + (y - py) * ny

	ks = s(float(keep_xy[0]), float(keep_xy[1]))
	if abs(ks) < 1e-9:
		ks = 1.0

	def inside(x: float, y: float) -> bool:
		return s(x, y) * ks >= -1e-9

	def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
		ax, ay = float(a[0]), float(a[1])
		bx, by = float(b[0]), float(b[1])
		sa = s(ax, ay)
		sb = s(bx, by)
		d = (sb - sa)
		if abs(d) < 1e-12:
			t = 0.0
		else:
			t = (-sa) / d
		t = float(max(0.0, min(1.0, t)))
		return (ax + t * (bx - ax), ay + t * (by - ay))

	out: list[tuple[float, float]] = []
	prev = poly[-1]
	prev_in = inside(prev[0], prev[1])
	for cur in poly:
		cur_in = inside(cur[0], cur[1])
		if cur_in:
			if not prev_in:
				out.append(intersect(prev, cur))
			out.append(cur)
		elif prev_in:
			out.append(intersect(prev, cur))
		prev = cur
		prev_in = cur_in
	return out


def _intersect_lines(
	*,
	p0: tuple[float, float],
	v0: tuple[float, float],
	p1: tuple[float, float],
	v1: tuple[float, float],
) -> tuple[float, float] | None:
	"""Return intersection of two infinite lines, or None if parallel."""
	x0, y0 = float(p0[0]), float(p0[1])
	x1, y1 = float(p1[0]), float(p1[1])
	a, b = float(v0[0]), float(v0[1])
	c, d = float(v1[0]), float(v1[1])
	det = a * d - b * c
	if abs(det) < 1e-12:
		return None
	t = ((x1 - x0) * d - (y1 - y0) * c) / det
	return (x0 + t * a, y0 + t * b)


def _blur_sigma(*, img: np.ndarray, sigma: float) -> np.ndarray:
	s = float(sigma)
	if s <= 0.0:
		return img
	ks = int(max(3, int(round(6.0 * s)) | 1))
	return cv2.GaussianBlur(img, (ks, ks), s)


@torch.no_grad()
def central_winding_pie_ema(*, model: fit_model.Model2D, it: int, params: dict) -> torch.Tensor:
	"""Return an image-space (N,1,H,W) float mask in model pixel coords.

	Implementation note: currently rasterizes a triangle wedge (pie approximation).
	"""
	xy = model.xy_ema
	xyc = model.xy_conn_ema
	if xy.numel() == 0 or xyc.numel() == 0:
		return torch.ones((0, 1, 0, 0), device=xy.device, dtype=torch.float32)
	n, hm, wm, c2 = (int(v) for v in xy.shape)
	if c2 != 2:
		raise ValueError("model.xy_ema must be (N,Hm,Wm,2)")
	if xyc.ndim != 5 or int(xyc.shape[-2]) != 3 or int(xyc.shape[-1]) != 2:
		raise ValueError("model.xy_conn_ema must be (N,Hm,Wm,3,2)")
	# Use data dimensions when available (expanded data with margins);
	# fall back to crop dimensions for backward compat.
	dh, dw = model.params.data_size_modelpx
	if dh > 0 and dw > 0:
		h_img, w_img = int(dh), int(dw)
	else:
		if model.params.crop_xyzwhd is None:
			raise ValueError("central_winding_pie_ema requires params.crop_xyzwhd or data_size_modelpx")
		_cx, _cy, cw, ch, _z0, _d = model.params.crop_xyzwhd
		h_img, w_img = int(ch), int(cw)
	if n == 0:
		return torch.ones((0, 1, h_img, w_img), device=xy.device, dtype=torch.float32)

	ramp_start_it = int(params.get("ramp_start_it", 0))
	start_size_segs = float(params.get("start_size_segs", 1.0))
	ramp_stop_it = int(params.get("ramp_stop_it", ramp_start_it))
	blur_sigma = float(params.get("blur_sigma", 0.0))
	center_off = params.get("center_off", [0, 0])
	if not isinstance(center_off, (list, tuple)) or len(center_off) != 2:
		raise ValueError("center_off must be [dx, dy]")
	center_off_x = int(center_off[0])
	center_off_y = int(center_off[1])

	dbg_dump = bool(params.get("dbg_dump", False))
	dbg_dump_every_it = int(params.get("dbg_dump_every_it", 0))
	dbg_do = bool(dbg_dump) and (
		int(it) == 0
		or int(it) == int(ramp_start_it)
		or int(it) == int(ramp_stop_it)
		or (int(dbg_dump_every_it) > 0 and (int(it) % int(dbg_dump_every_it) == 0))
	)

	if int(ramp_stop_it) <= int(ramp_start_it):
		prog = 1.0
	else:
		prog = (float(it) - float(ramp_start_it)) / float(ramp_stop_it - ramp_start_it)
		prog = float(max(0.0, min(1.0, prog)))

	full_size_segs = float(max(1, hm - 1))
	seg_span = _lerp(float(start_size_segs), full_size_segs, prog)
	seg_span = float(max(1.0, seg_span))

	ix = int(wm // 2) + int(center_off_x)
	iy_c = float(int(hm // 2) + int(center_off_y))
	ix = int(max(0, min(int(wm - 1), ix)))
	iy_c = float(max(0.0, min(float(hm - 1), float(iy_c))))
	dh = float(0.5 * seg_span)
	iy0_f = float(max(0.0, min(float(hm - 1), iy_c - dh)))
	iy1_f = float(max(0.0, min(float(hm - 1), iy_c + dh)))
	iy0 = int(math.floor(iy0_f))
	iy1 = int(math.floor(iy1_f))

	xy_col = xy[:, :, ix, :].to(dtype=torch.float32)
	xc_col = xyc[:, :, ix, :, :].to(dtype=torch.float32)
	p_c = _interp_rows_xy(xy_col=xy_col, y_pos=torch.full((n,), float(iy_c), device=xy.device, dtype=torch.float32))
	p_l = _interp_rows_xy(xy_col=xy_col, y_pos=torch.full((n,), float(iy0_f), device=xy.device, dtype=torch.float32))
	p_r = _interp_rows_xy(xy_col=xy_col, y_pos=torch.full((n,), float(iy1_f), device=xy.device, dtype=torch.float32))

	# `xy_conn` encodes (left, mid, right) pixel positions along the normal direction.
	# The direction (left->right) is already the normal; do NOT rotate it.
	v_l = _interp_rows_xy_conn_dir(xy_conn_col=xc_col, y_pos=torch.full((n,), float(iy0_f), device=xy.device, dtype=torch.float32))
	v_r = _interp_rows_xy_conn_dir(xy_conn_col=xc_col, y_pos=torch.full((n,), float(iy1_f), device=xy.device, dtype=torch.float32))

	v_l = v_l / (torch.sqrt((v_l * v_l).sum(dim=1, keepdim=True) + 1e-12))
	v_r = v_r / (torch.sqrt((v_r * v_r).sum(dim=1, keepdim=True) + 1e-12))
	if dbg_do and int(n) > 0:
		pc0 = [float(x) for x in p_c[0].detach().cpu().tolist()]
		pl0 = [float(x) for x in p_l[0].detach().cpu().tolist()]
		pr0 = [float(x) for x in p_r[0].detach().cpu().tolist()]
		vl0 = [float(x) for x in v_l[0].detach().cpu().tolist()]
		vr0 = [float(x) for x in v_r[0].detach().cpu().tolist()]
		print(
			"mask central_winding_pie_ema dbg:",
			f"it={int(it)} ix={int(ix)} iy_c={float(iy_c):.3f} dh={float(dh):.3f} iy0={float(iy0_f):.3f} iy1={float(iy1_f):.3f}",
			f"p_c={pc0}",
			f"p_upper={pl0}",
			f"p_lower={pr0}",
			f"n_upper={vl0}",
			f"n_lower={vr0}",
		)

	out = torch.zeros((n, 1, h_img, w_img), device=xy.device, dtype=torch.float32)
	for i in range(n):
		pl = (float(p_l[i, 0].cpu()), float(p_l[i, 1].cpu()))
		pr = (float(p_r[i, 0].cpu()), float(p_r[i, 1].cpu()))
		vl = (float(v_l[i, 0].cpu()), float(v_l[i, 1].cpu()))
		vr = (float(v_r[i, 0].cpu()), float(v_r[i, 1].cpu()))

		pbl = _ray_rect_intersection(p=pl, v=vl, w=w_img, h=h_img)
		pbr = _ray_rect_intersection(p=pr, v=vr, w=w_img, h=h_img)
		if pbl is None or pbr is None:
			out[i, 0].fill_(1.0)
			continue
		apex = _intersect_lines(p0=pl, v0=vl, p1=pr, v1=vr)
		if apex is not None:
			ax, ay = float(apex[0]), float(apex[1])
			if not (0.0 <= ax <= float(w_img - 1) and 0.0 <= ay <= float(h_img - 1)):
				apex = None

		if dbg_do and i == 0:
			img_dbg = np.zeros((h_img, w_img, 3), dtype=np.uint8)
			col_pl = (255, 0, 255)
			col_pr = (255, 255, 0)
			col_edge = (0, 255, 0)
			col_apex = (0, 0, 255)
			th = 1
			cv2.line(img_dbg, (int(round(pl[0])), int(round(pl[1]))), (int(round(pbl[0])), int(round(pbl[1]))), col_edge, th)
			cv2.line(img_dbg, (int(round(pr[0])), int(round(pr[1]))), (int(round(pbr[0])), int(round(pbr[1]))), col_edge, th)
			cv2.circle(img_dbg, (int(round(pl[0])), int(round(pl[1]))), 4, col_pl, -1)
			cv2.circle(img_dbg, (int(round(pr[0])), int(round(pr[1]))), 4, col_pr, -1)
			cv2.circle(img_dbg, (int(round(pbl[0])), int(round(pbl[1]))), 3, col_edge, -1)
			cv2.circle(img_dbg, (int(round(pbr[0])), int(round(pbr[1]))), 3, col_edge, -1)
			cv2.putText(img_dbg, f"pl=({pl[0]:.1f},{pl[1]:.1f})", (int(round(pl[0])) + 6, int(round(pl[1])) + 6), cv2.FONT_HERSHEY_PLAIN, 1.0, col_pl, 1)
			cv2.putText(img_dbg, f"pr=({pr[0]:.1f},{pr[1]:.1f})", (int(round(pr[0])) + 6, int(round(pr[1])) + 6), cv2.FONT_HERSHEY_PLAIN, 1.0, col_pr, 1)
			cv2.putText(img_dbg, f"xy_ema idx: ix={int(ix)} iy0={int(iy0)} iy1={int(iy1)}", (10, 52), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
			cv2.putText(img_dbg, f"nl=({vl[0]:.3f},{vl[1]:.3f})", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, col_edge, 1)
			cv2.putText(img_dbg, f"nr=({vr[0]:.3f},{vr[1]:.3f})", (10, 36), cv2.FONT_HERSHEY_PLAIN, 1.0, col_edge, 1)
			if apex is not None:
				cv2.circle(img_dbg, (int(round(apex[0])), int(round(apex[1]))), 4, col_apex, -1)
				cv2.putText(img_dbg, f"apex=({apex[0]:.1f},{apex[1]:.1f})", (int(round(apex[0])) + 6, int(round(apex[1])) + 6), cv2.FONT_HERSHEY_PLAIN, 1.0, col_apex, 1)
			out_name = f"dbg_central_winding_pie_ema_overlay_it{int(it):06d}.png"
			cv2.imwrite(out_name, img_dbg)
		# Infinite wedge between the two infinite lines through support points.
		# Build from full-rectangle, clipped to the kept side of each line.
		w1 = float(max(1, int(w_img) - 1))
		h1 = float(max(1, int(h_img) - 1))
		rect = [(0.0, 0.0), (w1, 0.0), (w1, h1), (0.0, h1)]
		poly_f = rect
		poly_f = _clip_poly_halfplane(poly=poly_f, p=pl, v=vl, keep_xy=pr)
		poly_f = _clip_poly_halfplane(poly=poly_f, p=pr, v=vr, keep_xy=pl)
		if not poly_f:
			out[i, 0].fill_(1.0)
			continue
		poly = np.array([[int(round(x)), int(round(y))] for (x, y) in poly_f], dtype=np.int32)
		img = np.zeros((h_img, w_img), dtype=np.float32)
		cv2.fillPoly(img, [poly], 1.0)
		img = _blur_sigma(img=img, sigma=float(blur_sigma))
		img = np.clip(img, 0.0, 1.0)
		out[i, 0] = torch.from_numpy(img).to(device=out.device, dtype=out.dtype)
	return out


def _gaussian_blur_mask_nchw(*, x: torch.Tensor, sigma: float) -> torch.Tensor:
	"""2D separable Gaussian blur for (N,1,H,W) mask tensors."""
	if x.ndim != 4:
		raise ValueError("_gaussian_blur_mask_nchw: x must be (N,C,H,W)")
	if float(sigma) <= 0.0:
		return x
	ks = int(max(3, int(round(6.0 * float(sigma))) | 1))
	if ks <= 1:
		return x
	if (ks % 2) == 0:
		ks += 1
	device = x.device
	dtype = x.dtype
	r = ks // 2
	idx = torch.arange(-r, r + 1, device=device, dtype=dtype)
	k = torch.exp(-(idx * idx) / (2.0 * float(sigma) * float(sigma)))
	k = k / (k.sum() + 1e-12)
	kx = k.view(1, 1, 1, ks)
	ky = k.view(1, 1, ks, 1)
	c = int(x.shape[1])
	pad = (r, r, 0, 0)
	y = F.pad(x, pad, mode="reflect")
	y = F.conv2d(y, kx.expand(c, 1, 1, ks), groups=c)
	pad = (0, 0, r, r)
	y = F.pad(y, pad, mode="reflect")
	y = F.conv2d(y, ky.expand(c, 1, ks, 1), groups=c)
	return y


class DilationMaskState:
	"""Stateful mask that grows outward from a seed voxel via 3D morphological dilation."""

	def __init__(self, *, model: fit_model.Model2D, params: dict) -> None:
		xy = model.xy_ema  # (Z, Hm, Wm, 2)
		mesh_step_px = int(model.params.mesh_step_px)
		scaledown = float(model.params.scaledown)
		z_size = int(xy.shape[0])
		device = xy.device

		# Image resolution in model pixels (for upsampling target)
		dh, dw = model.params.data_size_modelpx
		if dh > 0 and dw > 0:
			h_img, w_img = int(dh), int(dw)
		else:
			if model.params.crop_xyzwhd is None:
				raise ValueError("DilationMaskState requires data_size_modelpx or crop_xyzwhd")
			_cx, _cy, cw, ch, _z0, _d = model.params.crop_xyzwhd
			h_img, w_img = int(ch), int(cw)

		# Mask grid at volume-scaled resolution.
		# Each mask voxel = mesh_step_px model pixels in XY, scaledown z-slices in Z.
		mask_w = max(1, math.ceil(w_img / mesh_step_px))
		mask_h = max(1, math.ceil(h_img / mesh_step_px))
		mask_z = max(1, math.ceil(z_size / scaledown))

		self.mesh_step_px = mesh_step_px
		self.scaledown = scaledown
		self.h_img = h_img
		self.w_img = w_img
		self.z_size = z_size

		self.mask = torch.zeros((mask_z, 1, mask_h, mask_w), device=device, dtype=torch.float32)
		self.accum = 0.0
		self.blur_sigma = float(params.get("blur_sigma", 3.0))

		# Seed: single voxel at center of mesh, converted to mask coords
		hm = int(xy.shape[1])
		wm = int(xy.shape[2])
		zi = z_size // 2
		yi_m = hm // 2
		xi_m = wm // 2
		center_xy = xy[zi, yi_m, xi_m]  # (2,) in model pixels
		x_px = float(center_xy[0].cpu())
		y_px = float(center_xy[1].cpu())
		mask_x = max(0, min(mask_w - 1, round(x_px / mesh_step_px)))
		mask_y = max(0, min(mask_h - 1, round(y_px / mesh_step_px)))
		mask_zi = max(0, min(mask_z - 1, round(zi / scaledown)))
		self.mask[mask_zi, 0, mask_y, mask_x] = 1.0

		self.blurred = self._recompute_blurred()

	def _recompute_blurred(self) -> torch.Tensor:
		"""Blur mask at mask resolution, then trilinear-upsample to image resolution."""
		m = self.mask
		if self.blur_sigma > 0.0:
			m = _gaussian_blur_mask_nchw(x=m, sigma=self.blur_sigma).clamp(0.0, 1.0)
		# Upsample (mask_z, 1, mask_h, mask_w) -> (z_size, 1, h_img, w_img)
		mz, _, mh, mw = (int(v) for v in m.shape)
		m5 = m.reshape(1, 1, mz, mh, mw)
		m5 = F.interpolate(m5, size=(self.z_size, self.h_img, self.w_img), mode="trilinear", align_corners=False)
		return m5.reshape(self.z_size, 1, self.h_img, self.w_img)

	def advance(self, mvx_per_it: float) -> None:
		"""Grow the mask by `mvx_per_it` mask-voxels (fractional accumulation)."""
		self.accum += float(mvx_per_it)
		n = int(math.floor(self.accum))
		self.accum -= float(n)
		if n > 0:
			ks = 2 * n + 1
			# Reshape (Z,1,H,W) -> (1,1,Z,H,W) for 3D max_pool
			z, c, h, w = (int(v) for v in self.mask.shape)
			m5 = self.mask.reshape(1, 1, z, h, w)
			m5 = F.max_pool3d(m5, kernel_size=ks, stride=1, padding=n)
			self.mask = m5.reshape(z, 1, h, w)
		self.blurred = self._recompute_blurred()

	def completed(self, model: fit_model.Model2D) -> float:
		"""Return fraction of valid mesh positions covered by the blurred mask."""
		xy = model.xy_ema  # (Z, Hm, Wm, 2)
		if xy.numel() == 0:
			return 1.0
		z, hm, wm, _ = (int(v) for v in xy.shape)

		# Compute validity mask at mesh positions
		valid = fit_model.xy_img_validity_mask(params=model.params, xy=xy)  # (Z, Hm, Wm)

		# Sample blurred mask at mesh positions via grid_sample
		dh, dw = model.params.data_size_modelpx
		if dh > 0 and dw > 0:
			h_img, w_img = int(dh), int(dw)
		else:
			if model.params.crop_xyzwhd is None:
				return 1.0
			_cx, _cy, cw, ch, _z0, _d = model.params.crop_xyzwhd
			h_img, w_img = int(ch), int(cw)

		grid = xy.detach().to(dtype=torch.float32).clone()  # (Z, Hm, Wm, 2)
		hd = float(max(1, h_img - 1))
		wd = float(max(1, w_img - 1))
		grid[..., 0] = (grid[..., 0] / wd) * 2.0 - 1.0
		grid[..., 1] = (grid[..., 1] / hd) * 2.0 - 1.0
		# grid_sample expects (N,C,H_in,W_in) and grid (N,H_out,W_out,2)
		s = F.grid_sample(
			self.blurred.to(dtype=torch.float32),
			grid,
			mode="bilinear",
			padding_mode="zeros",
			align_corners=True,
		)  # (Z, 1, Hm, Wm)
		mask_vals = s[:, 0, :, :]  # (Z, Hm, Wm)

		valid_sum = valid.sum()
		if float(valid_sum) == 0.0:
			return 1.0
		return float((mask_vals * valid).sum() / valid_sum)


@torch.no_grad()
def constant_velocity_dilation(*, model: fit_model.Model2D, it: int, params: dict) -> torch.Tensor:
	"""Return a growing dilation mask (Z, 1, H, W).

	The mask starts from a single seed voxel at the mesh center and grows
	outward via 3D morphological dilation at a configurable rate.
	"""
	if "_state" not in params:
		params["_state"] = DilationMaskState(model=model, params=params)
	state = params["_state"]
	state.advance(float(params.get("mvx_per_it", 1.0)))
	return state.blurred


def stage_mask_completed(*, model: fit_model.Model2D, masks: list[dict] | None) -> float:
	"""Return minimum completion fraction across all stateful masks in a stage.

	Returns 1.0 if there are no masks or no stateful masks.
	"""
	if not masks:
		return 1.0
	min_c = 1.0
	for m in masks:
		state = m.get("_state")
		if state is not None and hasattr(state, "completed"):
			min_c = min(min_c, state.completed(model))
	return min_c


@torch.no_grad()
def build_stage_img_masks(
	*,
	model: fit_model.Model2D,
	it: int,
	masks: list[dict],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
	"""Build scheduled image-space masks for a stage.

	Returns:
	- masks_by_label: label -> (N,1,H,W) float mask
	- losses_by_label: label -> list of loss names this mask applies to
	"""
	out_masks: dict[str, torch.Tensor] = {}
	out_losses: dict[str, list[str]] = {}
	for mi, m0 in enumerate(masks):
		if not isinstance(m0, dict):
			continue
		m = dict(m0)
		label = str(m.pop("label", ""))
		mtype = str(m.pop("type", ""))
		losses = m.pop("losses", [])
		if not isinstance(losses, list):
			losses = []
		losses = [str(x) for x in losses]
		if not label:
			raise ValueError(f"stage masks[{mi}]: missing/empty label")
		if not mtype:
			raise ValueError(f"stage masks[{mi}]: missing/empty type")
		if mtype == "central_winding_pie_ema":
			mask = central_winding_pie_ema(model=model, it=int(it), params=m)
		elif mtype == "constant_velocity_dilation":
			mask = constant_velocity_dilation(model=model, it=int(it), params=m0)
		else:
			raise ValueError(f"stage masks[{mi}]: unknown type '{mtype}'")
		mask = mask.detach()
		mask.requires_grad_(False)
		out_masks[label] = mask
		out_losses[label] = losses
	return out_masks, out_losses
