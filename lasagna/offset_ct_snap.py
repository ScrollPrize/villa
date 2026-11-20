"""offset_ct_snap — put a Lasagna Offset surface back on the papyrus, check it, and recover missed vertices.

Input : the start surface (tifxyz grid on winding N), the surface produced by a Lasagna Offset fit (same grid, meant to
		sit on winding N+1 or N-1), and the CT volume around them.
Output: the fitted surface snapped onto the nearest papyrus ridge along its normal, a per-vertex state
		(0 not delivered, 1 anchored, 2 lifted by propagation, 3 recovered, 4 marched by opt-in fallback), and a verdict
		telling whether the surface as a whole crossed exactly one air gap from the start winding (1 = accepted). By default,
		a surface that is not accepted is reported, not repaired: this module never walks to a winding on its own unless
		allow_march_fallback=True is explicitly requested. One air gap is taken as one winding step; an air gap inside a
		winding (delamination) or windings fused without air can make that wrong (see docs/offset_ct_snap.md).

Only NumPy and SciPy are needed for the core. All lengths are in voxels of the CT level that is passed in.
This is a clean rewrite of the "accepted surface" branch of a research engine; it was checked against that engine
vertex for vertex on 211 surfaces (186 accepted identical, 25 refusals with the same verdict); that check needs the
research data and is not part of this repository.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

PARAMS = {
	# dense grid and normal profiles
	"dense_step_vox": 4,            # spacing of the working grid
	"reach_vox": 60,                # half length of the CT profile read along each normal
	"profile_smooth_vox": 1.0,
	"ridge_prominence": 8.0,
	# snapping
	"anchor_snap_vox": 6.0,         # a vertex already on paper anchors to a ridge at most this far
	"follow_tol_vox": 6.0,          # a neighbour follows if its ridge is within this of the local median
	"min_corrected_neighbors": 2,
	"island_step_vox": 6.0,         # largest step between neighbours of one continuous patch
	"consensus_window": 7,
	"consensus_tol_vox": 6.0,
	"consensus_iters": 3,
	# check: rays marched from the start winding along the step normal
	"air_min_run_vox": 14,          # an air gap counts when at least this long
	"segment_step_vox": 1.0,
	"end_margin_vox": 2,
	"march_max_vox": 220.0,
	"ridge_window_vox": 12,
	"start_margin_vox": 2,
	"next_winding_consensus_window": 11,
	"next_winding_consensus_tol_vox": 12.0,
	"proj_radius_vox": 20.0,
	"min_forward_vox": 6.0,
	"double_step_excess_vox": 20.0,
	"min_valid_rays": 30,
	"min_delivered_fraction": 0.55,
	"accept_margin_share": 0.05,        # "one air gap" must lead "no gap" and "two gaps" by more than this share of the rays
	"fallback_min_fit_move_vox": 0.0,   # legacy absolute guard on the CT march (0 = off; V6-FR used 45), replaced by the evidence below
	# fallback 1: deliver the sheet the fit itself designates when the air-gap check could not confirm it (touching windings)
	"fit_sheet_min_step_vox": 18.0,     # one sheet thickness: a ridge closer to the start than this is the start winding itself
	"fit_sheet_stayed_fraction": 0.5,   # a vertex closer to the start than this fraction of the local step is left out
	"fit_sheet_max_double_share": 0.35, # not when this share of the rays crossed two air gaps (the fit is one winding too far)
	"fit_sheet_max_left_share": 0.8,    # nor when the raw fit lies above the start packet on this share of the rays (it floats)
	# fallback 2: CT march, only for a fit that went past the middle of the first air gap without landing on a sheet
	"march_min_past_mid_gap_share": 0.5,  # share of start rays where the raw fit is beyond the middle of the first air gap
	# pocket repair is transactional: a repaired patch is kept only if it does not open a step with its unrepaired neighbours
	"repair_depression_fraction": 0.6,  # a pocket is a vertex lower than this fraction of the level set by the one-gap vertices
	"repair_level_tol_fraction": 0.3,   # its repair must bring it back within this fraction of that level (at least 12 voxels)
	"repair_jump_vox": 8.0,             # and the whole repair must not add steps longer than this
	# recovery of vertices the snap left out although a ridge exists
	"recover_tol_vox": 12.0,
	"recover_window": 7,
	"recover_min_neighbors": 3,
	"recover_max_passes": 30,
	# optional post-regularization and fallback march (disabled by default)
	"smooth_window": 5,
	"smooth_max_shift_vox": 4.0,
	"smooth_iters": 3,
	"march_jump_margin_vox": 28.0,
}

ACCEPTED = 1
__version__ = "0.2.0rc4"
MIN_EDGE_VOX = 1e-6          # two neighbouring vertices closer than this are the same point


def _mark(stages, name, **data):
	"""Optional export for viewers: record that a real step just finished, with the arrays it produced.
	Does nothing when no collector is passed; never changes a result."""
	if stages is None:
		return
	stages.setdefault("events", []).append({"stage": name, "unix_time": time.time()})
	stages.update(data)


@dataclass
class Snap:
	points: np.ndarray      # (H, W, 3) input grid
	snapped: np.ndarray     # (H, W, 3) grid moved onto the ridge where a ridge was taken
	state: np.ndarray       # (H, W) int8
	shift: np.ndarray       # (H, W) signed move along the normal, NaN where none
	inside: np.ndarray      # (H, W) whole profile lies in the CT crop
	threshold: float        # paper / air grey level
	normals: np.ndarray     # (H, W, 3)
	ridges: list            # ridges[i][j] : ridge positions along the normal


def otsu(values: np.ndarray) -> float:
	h, _ = np.histogram(values, bins=256, range=(0, 256))
	p = h / h.sum()
	w = np.cumsum(p)
	m = np.cumsum(p * np.arange(256))
	s = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
	return float(np.argmax(s))


def _nanmedian_filter2d(arr: np.ndarray, size: int, mask: np.ndarray | None = None) -> np.ndarray:
	"""Exact 2D sliding-window nanmedian (mode='nearest') without Python per-pixel callbacks or All-NaN RuntimeWarnings."""
	pad = size // 2
	padded = np.pad(arr, pad, mode="edge")
	has_any = ndimage.maximum_filter(np.isfinite(padded), size=size, mode="constant", cval=False)[pad:-pad, pad:-pad]
	if mask is not None:
		has_any = has_any & mask
	out = np.full(arr.shape, np.nan, dtype=arr.dtype)
	if not has_any.any():
		return out
	win = sliding_window_view(padded, (size, size))[has_any].reshape(-1, size * size)
	cnt = np.isfinite(win).sum(axis=1)
	s = np.sort(np.where(np.isnan(win), np.inf, win), axis=1)
	rows = np.arange(len(cnt))
	lo_v = s[rows, (cnt - 1) >> 1]
	hi_v = s[rows, cnt >> 1]
	out[has_any] = np.where(cnt & 1, lo_v, 0.5 * (lo_v + hi_v)).astype(arr.dtype, copy=False)
	return out


def _batched_find_peaks(profiles: np.ndarray, prominence: float, height: float | None = None):
	"""Run scipy.signal.find_peaks across a 2D stack of 1D profiles in a single C call using +1e9 boundary barriers."""
	M, L = profiles.shape
	if M == 0:
		return []
	S = L + 2
	padded = np.empty((M, S), dtype=np.float32)
	padded[:, 0] = 1e9
	padded[:, 1:-1] = profiles
	padded[:, -1] = 1e9
	h_arg = (height, 1e8) if height is not None else (None, 1e8)
	all_pk, _ = find_peaks(padded.ravel(), prominence=prominence, height=h_arg)
	row_ids = all_pk // S
	col_ids = (all_pk % S) - 1
	splits = np.searchsorted(row_ids, np.arange(1, M))
	return np.split(col_ids, splits)


def dense_grid(xyz: np.ndarray, step_src: float, step: float) -> np.ndarray:
	"""Bilinear refinement of an (H, W, 3) grid from spacing step_src to about step."""
	f = max(1, int(round(step_src / step)))
	H, W, _ = xyz.shape
	gi = np.linspace(0, H - 1, (H - 1) * f + 1)
	gj = np.linspace(0, W - 1, (W - 1) * f + 1)
	A, B = np.meshgrid(gi, gj, indexing="ij")
	return np.stack([ndimage.map_coordinates(xyz[..., k], [A, B], order=1) for k in range(3)], -1)


def grid_step(xyz: np.ndarray) -> float:
	return float(np.mean([np.linalg.norm(np.diff(xyz, axis=0), axis=-1).mean(), np.linalg.norm(np.diff(xyz, axis=1), axis=-1).mean()]))


def step_normal(P: np.ndarray, offset_sign: float, sigma: float = 3.0) -> np.ndarray:
	"""Smoothed unit normal of the grid, oriented by the sign of the offset."""
	Ps = np.stack([ndimage.gaussian_filter(P[..., k].astype(np.float32), sigma) for k in range(3)], -1)
	U = np.cross(np.gradient(Ps, axis=0), np.gradient(Ps, axis=1))
	U = U / (np.linalg.norm(U, axis=-1, keepdims=True) + 1e-12)
	return (U * float(np.sign(offset_sign))).astype(np.float32)


def snap_to_paper(
	P: np.ndarray,
	vol: np.ndarray,
	org: np.ndarray,
	params: dict = PARAMS,
	q0_foot: np.ndarray | None = None,
	u0_foot: np.ndarray | None = None,
	b1_foot: np.ndarray | None = None,
	d1_foot: np.ndarray | None = None,
) -> Snap:
	"""Move each vertex of P along its normal onto a papyrus ridge of the CT, keeping one continuous patch.

	vol is the CT crop indexed (z, y, x); org is the xyz position of its first voxel.
	Optional barrier arguments (q0_foot, u0_foot, b1_foot, d1_foot) restrict candidate ridges during opt-in fallback marching.
	"""
	p = params
	H, W, _ = P.shape
	R = p["reach_vox"]
	Ps = np.stack([ndimage.gaussian_filter(P[..., k].astype(np.float32), 2.0) for k in range(3)], -1)
	n = np.cross(np.gradient(Ps, axis=0), np.gradient(Ps, axis=1))
	n = (n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)).astype(np.float32)
	t = np.arange(-R, R + 1, 1.0, dtype=np.float32)
	lim = np.array(vol.shape[::-1]) - 2
	p_rel = P - org
	rn = R * np.abs(n)
	inside = ((p_rel - rn >= 1) & (p_rel + rn < lim)).all(-1)
	if inside.any():
		pts_in = P[inside][:, None, :] + t[None, :, None] * n[inside][:, None, :] - org
		exact_in = ((pts_in >= 1) & (pts_in < lim)).all(-1).all(-1)
		if not exact_in.all():
			idx_in = np.argwhere(inside)
			for k_bad in np.nonzero(~exact_in)[0]:
				inside[tuple(idx_in[k_bad])] = False
			pts_in = pts_in[exact_in]
	on_paper = np.zeros((H, W), dtype=bool)
	ridges = [[np.empty(0, dtype=np.float32)] * W for _ in range(H)]
	use_barrier = (b1_foot is not None) and (q0_foot is not None) and (u0_foot is not None)
	if use_barrier:
		h0 = np.einsum("ijk,ijk->ij", P - q0_foot, u0_foot)
		cos_nu = np.einsum("ijk,ijk->ij", n, u0_foot)
		jump_margin = p.get("march_jump_margin_vox", 28.0)
	if inside.any():
		prof_in = ndimage.map_coordinates(vol, [pts_in[..., 2], pts_in[..., 1], pts_in[..., 0]], order=1, mode="nearest").astype(np.float32)
		prof_in = ndimage.gaussian_filter1d(prof_in, p["profile_smooth_vox"], axis=1)
		thr = otsu(prof_in)
		on_paper[inside] = prof_in[:, R] > thr
		pk_list = _batched_find_peaks(prof_in, prominence=p["ridge_prominence"], height=thr)
		for (i, j), pk in zip(np.argwhere(inside), pk_list):
			rt = t[pk]
			if use_barrier and len(rt) and np.isfinite(b1_foot[i, j]):
				h_cand = h0[i, j] + rt * cos_nu[i, j]
				keep_r = h_cand >= b1_foot[i, j]
				if d1_foot is not None and np.isfinite(d1_foot[i, j]):
					keep_r &= (h_cand <= d1_foot[i, j] + jump_margin)
				rt = rt[keep_r]
			ridges[i][j] = rt
	else:
		thr = 80.0

	d = np.full((H, W), np.nan, dtype=np.float32)
	state = np.zeros((H, W), np.int8)
	queue = deque()
	for i, j in zip(*np.nonzero(inside & on_paper)):
		if len(ridges[i][j]):
			k = np.abs(ridges[i][j]).argmin()
			if abs(ridges[i][j][k]) <= p["anchor_snap_vox"]:
				d[i, j] = ridges[i][j][k]
				state[i, j] = 1
				queue.append((i, j))

	def propagate(queue):
		while queue:
			i, j = queue.popleft()
			for di in (-1, 0, 1):
				for dj in (-1, 0, 1):
					a, b = i + di, j + dj
					if (di or dj) and 0 <= a < H and 0 <= b < W and state[a, b] == 0 and inside[a, b] and len(ridges[a, b] if isinstance(ridges, np.ndarray) else ridges[a][b]):
						nb = d[max(0, a - 1):a + 2, max(0, b - 1):b + 2]
						nb = nb[np.isfinite(nb)]
						if len(nb) < p["min_corrected_neighbors"]:
							continue
						pred = float(np.median(nb))
						k = np.abs(ridges[a][b] - pred).argmin()
						if abs(ridges[a][b][k] - pred) <= p["follow_tol_vox"]:
							d[a, b] = ridges[a][b][k]
							state[a, b] = 2
							queue.append((a, b))

	propagate(queue)

	# keep the largest patch where the snapped surface is continuous; undo the rest and regrow from the main patch
	T3 = np.stack([P[..., k] + np.nan_to_num(d) * n[..., k] for k in range(3)], -1)
	ok = state > 0
	lab = np.zeros((H, W), int)
	cur = 0
	for i0, j0 in zip(*np.nonzero(ok)):
		if lab[i0, j0]:
			continue
		cur += 1
		lab[i0, j0] = cur
		q = deque([(i0, j0)])
		while q:
			i, j = q.popleft()
			for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
				a, b = i + di, j + dj
				if 0 <= a < H and 0 <= b < W and ok[a, b] and not lab[a, b] and abs(float((T3[a, b] - T3[i, j]) @ n[i, j])) <= p["island_step_vox"]:
					lab[a, b] = cur
					q.append((a, b))
	if cur > 1:
		sizes = np.bincount(lab.ravel())[1:]
		main = sizes.argmax() + 1
		isl = ok & (lab != main)
		d[isl] = np.nan
		state[isl] = 0
		propagate(deque(zip(*np.nonzero(lab == main))))

	# each kept vertex re-hooks to the ridge nearest the median move of its neighbourhood
	for _ in range(p["consensus_iters"]):
		active = state > 0
		dm = _nanmedian_filter2d(np.where(active, d, np.nan), p["consensus_window"], mask=active)
		changed = 0
		for i, j in zip(*np.nonzero(active)):
			rg = ridges[i][j]
			if not len(rg) or not np.isfinite(dm[i, j]):
				continue
			k = np.abs(rg - dm[i, j]).argmin()
			if abs(rg[k] - dm[i, j]) <= p["consensus_tol_vox"] and rg[k] != d[i, j]:
				d[i, j] = rg[k]
				changed += 1
		if not changed:
			break
	Q = np.where(np.isfinite(d)[..., None], P + np.nan_to_num(d)[..., None] * n, P)
	return Snap(P, Q, state, d, inside, thr, n, ridges)


def _inpaint_nearest(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
	if not valid.any():
		return np.zeros_like(arr, dtype=np.float32)
	if valid.all():
		return arr.astype(np.float32)
	idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
	return arr[tuple(idx)].astype(np.float32)


def march_rays(A: np.ndarray, U: np.ndarray, vol: np.ndarray, org: np.ndarray, thr: float, params: dict = PARAMS):
	"""Read the CT along rays A + s U. For each ray: ridges met right after an air gap and the width of that gap,
	plus the middle of the first and second air gaps."""
	p = params
	s = np.arange(0.0, int(p["march_max_vox"]) + 1.0, 1.0, dtype=np.float32)
	N = len(A)
	if N == 0:
		return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32)
	pts = A[:, None, :] + s[None, :, None] * U[:, None, :] - org
	ok = ((pts >= 1) & (pts < np.array(vol.shape[::-1]) - 2)).all(axis=-1)
	n_valid = np.where(ok.all(axis=1), len(s), ok.argmin(axis=1))
	b1 = np.full(N, np.nan, dtype=np.float32)
	b2 = np.full(N, np.nan, dtype=np.float32)
	events = [(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))] * N
	use_rays = np.nonzero(n_valid >= 20)[0]
	if len(use_rays) == 0:
		return events, b1, b2
	pts_u = pts[use_rays]
	coords = [np.clip(pts_u[..., 2], 0, vol.shape[0] - 1).ravel(), np.clip(pts_u[..., 1], 0, vol.shape[1] - 1).ravel(), np.clip(pts_u[..., 0], 0, vol.shape[2] - 1).ravel()]
	V = ndimage.map_coordinates(vol, coords, order=1, mode="nearest").reshape(len(use_rays), len(s)).astype(np.float32)
	V = ndimage.gaussian_filter1d(V, 1.0, axis=1)
	air_all = V <= thr
	air_all[:, :p["start_margin_vox"]] = False
	prom = p["ridge_prominence"] / 2.0
	min_run = p["air_min_run_vox"]
	rw = p["ridge_window_vox"]
	for idx_u, k in enumerate(use_rays):
		n = int(n_valid[k])
		air = air_all[idx_u, :n]
		dd = np.diff(np.r_[0, air.astype(np.int8), 0])
		st = np.nonzero(dd == 1)[0]
		en = np.nonzero(dd == -1)[0]
		ev, widths, mids = [], [], []
		vk = V[idx_u]
		for a, e in zip(st, en):
			w_air = e - a
			if w_air < min_run or e >= n:
				continue
			mids.append(0.5 * (a + e))
			w = vk[e:e + rw + 1]
			pk, _ = find_peaks(w, prominence=prom)
			ev.append(float(e + (pk[0] if len(pk) else int(w.argmax()))))
			widths.append(float(w_air))
		if len(mids) >= 1:
			b1[k] = mids[0]
		if len(mids) >= 2:
			b2[k] = mids[1]
		events[k] = (np.array(ev, dtype=np.float32), np.array(widths, dtype=np.float32))
	return events, b1, b2


def consensus_target(events, b1, b2, mask: np.ndarray, params: dict = PARAMS):
	"""Maps on the start grid: half-gap barrier and distance to the next winding, by spatial consensus.
	A thin internal split of the start winding (a small gap followed by a clearly wider one) is skipped."""
	p = params
	H, W = mask.shape
	first = np.array([e[0] if len(e) else np.nan for e, _ in events], dtype=np.float32)
	second = np.array([e[1] if len(e) >= 2 else np.nan for e, _ in events], dtype=np.float32)
	w0 = np.array([w[0] if len(w) else np.nan for _, w in events], dtype=np.float32)
	w1 = np.array([w[1] if len(w) >= 2 else np.nan for _, w in events], dtype=np.float32)
	valid = np.isfinite(first)
	use = events
	b1_use = b1.copy()
	if valid.sum() > 100:
		blister = valid & np.isfinite(second) & (w0 <= 18.0) & (w1 >= 26.0) & (w1 >= 1.8 * w0)
		if blister.sum() > 0.15 * valid.sum():
			med_blister = float(np.nanmedian(first[blister]))
			high = valid & (w0 >= 26.0) & (first >= med_blister + 28.0)
			if high.sum() > 0.12 * valid.sum():
				cutoff = 0.5 * (med_blister + float(np.nanmedian(first[high])))
				use = [(e[e >= cutoff], w[e >= cutoff]) for e, w in events]
				b1_use = np.where((first < cutoff) & np.isfinite(b2), b2, b1)
	first_map = np.full((H, W), np.nan, dtype=np.float32)
	b1_raw = np.full((H, W), np.nan, dtype=np.float32)
	idx = np.argwhere(mask)
	for k, ((i, j), (ev, _)) in enumerate(zip(idx, use)):
		b1_raw[i, j] = b1_use[k]
		if len(ev) >= 1:
			first_map[i, j] = ev[0]
	med = _nanmedian_filter2d(first_map, p["next_winding_consensus_window"], mask=mask)
	d1_raw = np.full((H, W), np.nan, dtype=np.float32)
	for (i, j), (ev, _) in zip(idx, use):
		if len(ev) and np.isfinite(med[i, j]):
			q = np.abs(ev - med[i, j]).argmin()
			if abs(ev[q] - med[i, j]) <= p["next_winding_consensus_tol_vox"]:
				d1_raw[i, j] = ev[q]
	b1_med = _nanmedian_filter2d(b1_raw, p["next_winding_consensus_window"])
	b1_map = ndimage.gaussian_filter(_inpaint_nearest(b1_med, np.isfinite(b1_med)), sigma=3.0)
	d1_smooth = ndimage.gaussian_filter(_inpaint_nearest(d1_raw, np.isfinite(d1_raw)), sigma=2.5)
	b1_map = np.clip(np.minimum(b1_map, 0.6 * d1_smooth), 6.0, 85.0)
	return b1_map, d1_smooth, d1_raw


def consensus_target_v6f(events, b1, b2, mask: np.ndarray, params: dict = PARAMS):
	"""Delamination-aware multi-event ray consensus for V6-F:
	Detects local and global start-winding delamination splits (narrow gap w0 followed by wider inter-winding gap w1
	where the second event matches the clean target-sheet height h_ref within +-26 voxels) and promotes the second
	event and its half-gap barrier b2."""
	p = params
	H, W = mask.shape
	first = np.array([e[0] if len(e) else np.nan for e, _ in events], dtype=np.float32)
	second = np.array([e[1] if len(e) >= 2 else np.nan for e, _ in events], dtype=np.float32)
	w0 = np.array([w[0] if len(w) else np.nan for _, w in events], dtype=np.float32)
	w1 = np.array([w[1] if len(w) >= 2 else np.nan for _, w in events], dtype=np.float32)
	valid = np.isfinite(first)
	use = list(events)
	b1_use = b1.copy()
	promoted = np.zeros(len(events), dtype=bool)
	if valid.sum() > 100:
		clean = valid & ((~np.isfinite(w1)) | (w0 >= w1) | (w0 >= 28.0))
		if clean.sum() >= 0.15 * valid.sum():
			h_clean_med = float(np.nanmedian(first[clean]))
			h_clean_p65 = float(np.nanpercentile(first[clean], 65))
			if h_clean_med >= 78.0:
				low_split = valid & np.isfinite(second) & (w0 <= 36.0) & (w1 >= 18.0) & (w1 >= 1.15 * w0) & (first <= h_clean_p65 - 20.0)
				if low_split.sum() >= 0.03 * valid.sum():
					h_ref = h_clean_p65 if (low_split.sum() > 0.20 * valid.sum()) else h_clean_med
					h_low1 = float(np.nanmedian(first[low_split]))
					h_low2 = float(np.nanmedian(second[low_split]))
					if (h_ref - h_low1 >= 40.0) and (abs(h_low2 - h_ref) <= 26.0):
						cutoff = 0.5 * (h_low1 + h_ref)
						promote_k = valid & np.isfinite(second) & (first < cutoff) & (w1 >= w0) & (second <= h_ref + 35.0)
						for k in np.nonzero(promote_k)[0]:
							ev_k, w_k = use[k]
							use[k] = (ev_k[1:], w_k[1:])
							if np.isfinite(b2[k]):
								b1_use[k] = b2[k]
							promoted[k] = True
	first_map = np.full((H, W), np.nan, dtype=np.float32)
	b1_raw = np.full((H, W), np.nan, dtype=np.float32)
	prom_map = np.zeros((H, W), dtype=bool)
	idx = np.argwhere(mask)
	for k, ((i, j), (ev, _)) in enumerate(zip(idx, use)):
		b1_raw[i, j] = b1_use[k]
		prom_map[i, j] = promoted[k]
		if len(ev) >= 1:
			first_map[i, j] = ev[0]
	med = _nanmedian_filter2d(first_map, p["next_winding_consensus_window"], mask=mask)
	d1_raw = np.full((H, W), np.nan, dtype=np.float32)
	for (i, j), (ev, _) in zip(idx, use):
		if len(ev) and np.isfinite(med[i, j]):
			q = np.abs(ev - med[i, j]).argmin()
			if abs(ev[q] - med[i, j]) <= p["next_winding_consensus_tol_vox"]:
				d1_raw[i, j] = ev[q]
	b1_med = _nanmedian_filter2d(b1_raw, p["next_winding_consensus_window"])
	b1_map = ndimage.gaussian_filter(_inpaint_nearest(b1_med, np.isfinite(b1_med)), sigma=3.0)
	d1_smooth = ndimage.gaussian_filter(_inpaint_nearest(d1_raw, np.isfinite(d1_raw)), sigma=2.5)
	b1_map = np.clip(np.minimum(b1_map, 0.6 * d1_smooth), 6.0, 95.0)
	return b1_map, d1_smooth, d1_raw, prom_map


def _tangent_frame(u0: np.ndarray):
	u_mean = u0.mean(axis=0)
	u_mean = u_mean / (np.linalg.norm(u_mean) + 1e-12)
	ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(u_mean[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
	t1 = np.cross(u_mean, ref)
	t1 /= np.linalg.norm(t1) + 1e-12
	return t1, np.cross(u_mean, t1)


def project_on_rays(Q0, U0, mask0, Q1, mask1, r_max: float):
	"""Height of the arrival surface Q1 above each start vertex, measured along the start ray (Q0, U0)."""
	H, W = mask0.shape
	h_map = np.full((H, W), np.nan, dtype=np.float32)
	q1_proj = np.zeros_like(Q0, dtype=np.float32)
	if not mask0.any() or not mask1.any():
		return h_map, q1_proj
	pts0, u0, pts1 = Q0[mask0], U0[mask0], Q1[mask1]
	t1, t2 = _tangent_frame(u0)
	c0 = np.c_[pts0 @ t1, pts0 @ t2]
	c1 = np.c_[pts1 @ t1, pts1 @ t2]
	dists, nn = cKDTree(c1).query(c0, k=min(8, len(pts1)), distance_upper_bound=r_max)
	if dists.ndim == 1:
		dists, nn = dists[:, None], nn[:, None]
	valid_k = dists < r_max
	idx0 = np.argwhere(mask0)
	for m in np.nonzero(valid_k.any(axis=1))[0]:
		diff = pts1[nn[m, valid_k[m]]] - pts0[m]
		h = diff @ u0[m]
		perp = np.linalg.norm(diff - h[:, None] * u0[m], axis=1)
		keep = perp <= r_max
		if not keep.any():
			continue
		w = 1.0 / (perp[keep] + 1.0) ** 2
		w /= w.sum()
		h_val = float((w * h[keep]).sum())
		i, j = idx0[m]
		h_map[i, j] = h_val
		q1_proj[i, j] = pts0[m] + h_val * u0[m]
	return h_map, q1_proj


def footpoints(Q0, U0, mask0, b1_map, P):
	"""For each vertex of P: nearest start vertex in the tangent plane, its ray direction and its barrier height."""
	H, W, _ = P.shape
	if not mask0.any():
		return np.zeros_like(P), np.zeros_like(P), np.full((H, W), np.nan, dtype=np.float32)
	pts0, u0, b0 = Q0[mask0], U0[mask0], b1_map[mask0]
	t1, t2 = _tangent_frame(u0)
	flat = P.reshape(-1, 3)
	_, nn = cKDTree(np.c_[pts0 @ t1, pts0 @ t2]).query(np.c_[flat @ t1, flat @ t2], k=1)
	return pts0[nn].reshape(H, W, 3), u0[nn].reshape(H, W, 3), b0[nn].reshape(H, W)


def count_air(A, B, vol, org, thr, params: dict = PARAMS) -> np.ndarray:
	"""Number of air runs of at least air_min_run_vox on each segment A[k] -> B[k]."""
	p = params
	if len(A) == 0:
		return np.zeros(0, int)
	L = np.linalg.norm(B - A, axis=1)
	M = np.maximum(2, np.ceil(L / p["segment_step_vox"]).astype(int) + 1)
	offsets = np.r_[0, np.cumsum(M)]
	total_pts = int(offsets[-1])
	idx = np.arange(total_pts)
	seg_id = np.repeat(np.arange(len(A)), M)
	local_i = idx - offsets[seg_id]
	s = (local_i / np.maximum(1, M[seg_id] - 1))[:, None]
	pts = A[seg_id] + s * (B[seg_id] - A[seg_id]) - org
	air_all = ndimage.map_coordinates(vol, [pts[:, 2], pts[:, 1], pts[:, 0]], order=1, mode="nearest") <= thr
	e = p["end_margin_vox"]
	out = np.zeros(len(A), int)
	step_scale = L / np.maximum(1, M - 1)
	min_run = p["air_min_run_vox"]
	for k in range(len(A)):
		air = air_all[offsets[k]:offsets[k + 1]]
		if e:
			air[:e] = False
			air[-e:] = False
		dd = np.diff(np.r_[0, air.astype(np.int8), 0])
		run = (np.nonzero(dd == -1)[0] - np.nonzero(dd == 1)[0]) * step_scale[k]
		out[k] = int((run >= min_run).sum())
	return out


def recover(snap: Snap, Q, state, shift, params: dict = PARAMS):
	"""Hook vertices the snap left out although a ridge exists: take the ridge nearest the median move of the already
	placed neighbours, if close enough. Recovered vertices get state 3."""
	p = params
	state = state.copy()
	d = shift.copy()
	h = p["recover_window"] // 2
	todo = [(i, j) for i, j in zip(*np.nonzero((state == 0) & snap.inside)) if len(snap.ridges[i][j])]
	for _ in range(p["recover_max_passes"]):
		changed = 0
		rest = []
		for i, j in todo:
			nb = d[max(0, i - h):i + h + 1, max(0, j - h):j + h + 1]
			v = nb[np.isfinite(nb)]
			if len(v) < p["recover_min_neighbors"]:
				rest.append((i, j))
				continue
			med = float(np.median(v))
			rg = snap.ridges[i][j]
			c = float(rg[np.abs(rg - med).argmin()])
			if abs(c - med) > p["recover_tol_vox"]:
				rest.append((i, j))
				continue
			d[i, j] = c
			state[i, j] = 3
			changed += 1
		todo = rest
		if not changed:
			break
	rec = state == 3
	Q = np.where(rec[..., None], snap.points + np.nan_to_num(d)[..., None] * snap.normals, Q)
	return Q, state, d, int(rec.sum())


def _jump_vertices(Q: np.ndarray, valid: np.ndarray, thr_step: float = 8.0) -> np.ndarray:
	has_jump = np.zeros(valid.shape, dtype=bool)
	for ax in (0, 1):
		a = [slice(None)] * 2
		b = [slice(None)] * 2
		a[ax] = slice(0, -1)
		b[ax] = slice(1, None)
		m = valid[tuple(a)] & valid[tuple(b)]
		jmp = m & (np.linalg.norm(Q[tuple(b)] - Q[tuple(a)], axis=-1) > thr_step)
		has_jump[tuple(a)] |= jmp
		has_jump[tuple(b)] |= jmp
	return has_jump


def regularize_within_paper(
	s1: Snap,
	Q: np.ndarray,
	state: np.ndarray,
	domain: np.ndarray,
	vol: np.ndarray,
	org: np.ndarray,
	thr: float,
	q0_f: np.ndarray | None = None,
	u0_f: np.ndarray | None = None,
	b1_f: np.ndarray | None = None,
	params: dict = PARAMS,
) -> np.ndarray:
	"""Targeted papyrus-core normal regularizer:
	- Leaves all already-continuous vertices (4-connected 3D step <= 8 voxels) untouched on their exact CT ridge peak.
	- On vertices participating in a >8-voxel step, prefers a true CT ridge in s1.ridges near the 5x5 median if one exists;
	  otherwise damps the normal offset toward the 5x5 median within +-smooth_max_shift_vox.
	- Requires both midpoint and endpoint to remain in the dense CT papyrus core (CT >= max(thr + 2, 0.80 * CT_orig)),
	  above the half-air barrier b1_f, and reverts any move that would create a new >8-voxel edge."""
	p = params
	P = s1.points
	normals = s1.normals
	valid = domain & (state > 0)
	if not valid.any():
		return Q
	win_size = int(p.get("smooth_window", 5))
	max_shift = float(p.get("smooth_max_shift_vox", 4.0))
	iters = int(p.get("smooth_iters", 3))
	Q_work = Q.copy()
	d_orig = np.einsum("ijk,ijk->ij", Q_work - P, normals).astype(np.float32)
	d_work = d_orig.copy()
	lim = np.array(vol.shape[::-1]) - 2
	pts0 = Q_work[valid] - org
	ct_orig = np.zeros(state.shape, dtype=np.float32)
	ct_orig[valid] = ndimage.map_coordinates(vol, [pts0[:, 2], pts0[:, 1], pts0[:, 0]], order=1, mode="nearest")
	use_barrier = (b1_f is not None) and (q0_f is not None) and (u0_f is not None)

	for _ in range(iters):
		target_mask = valid & _jump_vertices(Q_work, valid, 8.0)
		if not target_mask.any():
			break
		d_med = _nanmedian_filter2d(np.where(valid, d_work, np.nan), win_size, mask=target_mask)
		idx_t = np.argwhere(target_mask)
		d_cand = np.clip(d_med[target_mask], d_orig[target_mask] - max_shift, d_orig[target_mask] + max_shift)
		for k, (i, j) in enumerate(idx_t):
			mv = float(d_med[i, j])
			if not np.isfinite(mv):
				d_cand[k] = d_work[i, j]
				continue
			rg = s1.ridges[i][j]
			if len(rg):
				rk = float(rg[np.abs(rg - mv).argmin()])
				if abs(rk - mv) <= 4.0 and abs(rk - d_orig[i, j]) <= 8.0:
					d_cand[k] = rk
		Q_cand = P[target_mask] + d_cand[:, None] * normals[target_mask]
		Q_mid = 0.5 * (Q_work[target_mask] + Q_cand)
		pts_c = Q_cand - org
		pts_m = Q_mid - org
		in_b = ((pts_c >= 1) & (pts_c < lim)).all(-1) & ((pts_m >= 1) & (pts_m < lim)).all(-1)
		val_c = ndimage.map_coordinates(vol, [pts_c[:, 2], pts_c[:, 1], pts_c[:, 0]], order=1, mode="nearest")
		val_m = ndimage.map_coordinates(vol, [pts_m[:, 2], pts_m[:, 1], pts_m[:, 0]], order=1, mode="nearest")
		min_ct = np.maximum(thr + 2.0, 0.80 * ct_orig[target_mask])
		ok_p = in_b & (val_c >= min_ct) & (val_m > thr) & np.isfinite(d_cand)
		if use_barrier:
			h_c = np.einsum("ij,ij->i", Q_cand - q0_f[target_mask], u0_f[target_mask])
			b1_t = b1_f[target_mask]
			ok_p &= (~np.isfinite(b1_t) | (h_c >= np.minimum(10.0, b1_t - 6.0)))
		if not ok_p.any():
			break
		Q_next = Q_work.copy()
		d_next = d_work.copy()
		upd_idx = idx_t[ok_p]
		Q_next[upd_idx[:, 0], upd_idx[:, 1]] = Q_cand[ok_p]
		d_next[upd_idx[:, 0], upd_idx[:, 1]] = d_cand[ok_p]
		for ax in (0, 1):
			a = [slice(None)] * 2
			b = [slice(None)] * 2
			a[ax] = slice(0, -1)
			b[ax] = slice(1, None)
			m = valid[tuple(a)] & valid[tuple(b)]
			was_ok = m & (np.linalg.norm(Q_work[tuple(b)] - Q_work[tuple(a)], axis=-1) <= 8.0)
			now_bad = was_ok & (np.linalg.norm(Q_next[tuple(b)] - Q_next[tuple(a)], axis=-1) > 8.0)
			if now_bad.any():
				bad_v = np.zeros(valid.shape, dtype=bool)
				bad_v[tuple(a)] |= now_bad
				bad_v[tuple(b)] |= now_bad
				Q_next[bad_v] = Q_work[bad_v]
				d_next[bad_v] = d_work[bad_v]
		Q_work = Q_next
		d_work = d_next
	return Q_work


def _local_step(height: np.ndarray, placed: np.ndarray, block: int = 16) -> np.ndarray:
	"""Step height the surface itself claims locally: median of `height` over blocks of `block` vertices, smoothed and
	interpolated back to the grid. NaN everywhere when no block holds enough placed vertices."""
	H, W = height.shape
	hb, wb = -(-H // block), -(-W // block)
	med = np.full((hb, wb), np.nan, dtype=np.float32)
	for a in range(hb):
		for b in range(wb):
			sl = (slice(a * block, (a + 1) * block), slice(b * block, (b + 1) * block))
			v = height[sl][placed[sl]]
			if len(v) >= 8:
				med[a, b] = np.median(v)
	if not np.isfinite(med).any():
		return np.full((H, W), np.nan, dtype=np.float32)
	med = ndimage.gaussian_filter(_inpaint_nearest(med, np.isfinite(med)), 1.0)
	gi, gj = np.meshgrid((np.arange(H) + 0.5) / block - 0.5, (np.arange(W) + 0.5) / block - 0.5, indexing="ij")
	return ndimage.map_coordinates(med, [gi, gj], order=1, mode="nearest").astype(np.float32)


def _count_steps(Q: np.ndarray, valid: np.ndarray, thr_step: float) -> int:
	n = 0
	for ax in (0, 1):
		a = [slice(None)] * 2
		b = [slice(None)] * 2
		a[ax] = slice(0, -1)
		b[ax] = slice(1, None)
		a, b = tuple(a), tuple(b)
		n += int((valid[a] & valid[b] & (np.linalg.norm(Q[b] - Q[a], axis=-1) > thr_step)).sum())
	return n


def commit_repairs(Q_before, state_before, height_before, Q_after, state_after, height_after, level, params: dict = PARAMS):
	"""Transactional pocket repair: compare before and after, keep only what is justified, roll back the rest.

	level : (H, W) step height set locally by the vertices that crossed exactly one air gap (the evidence the surface was
	        accepted on). A start pocket is a depression below that level; repairing it means bringing it back to it.
	A repaired vertex is kept only if it was lower than repair_depression_fraction x level before, and lies within
	repair_level_tol_fraction x level (at least 12 voxels) of it after. A vertex that was already at the level of the
	accepted surface was not trapped: moving it by about one winding is exactly the destructive repair this guards against.
	If what is kept adds steps longer than repair_jump_vox to the surface, everything is rolled back.
	Returns (Q, state, vertices kept, vertices rolled back)."""
	p = params
	moved = (state_after != state_before) | (Q_after != Q_before).any(-1)
	if not moved.any():
		return Q_after, state_after, 0, 0
	tol = np.maximum(12.0, p["repair_level_tol_fraction"] * level)
	was_low = (state_before == 0) | (height_before < p["repair_depression_fraction"] * level)
	keep = moved & np.isfinite(level) & was_low & (state_after > 0) & (np.abs(height_after - level) <= tol)
	Q = np.where(keep[..., None], Q_after, Q_before)
	state = np.where(keep, state_after, state_before)
	if keep.any() and _count_steps(Q, state > 0, p["repair_jump_vox"]) > 1.05 * _count_steps(Q_before, state_before > 0, p["repair_jump_vox"]) + 20:
		return Q_before, state_before, 0, int(moved.sum())
	return Q, state, int(keep.sum()), int((moved & ~keep).sum())


def _check_inputs(start_xyz, fit_xyz, vol, org, offset_sign):
	"""Refuse inputs the method was never measured on, with a message that says what is wrong."""
	start_xyz, fit_xyz = np.asarray(start_xyz), np.asarray(fit_xyz)
	for name, g in (("start", start_xyz), ("fit", fit_xyz)):
		if g.ndim != 3 or g.shape[-1] != 3:
			raise ValueError(f"{name} grid must have shape (h, w, 3), got {g.shape}")
		if g.shape[0] < 2 or g.shape[1] < 2:
			raise ValueError(f"{name} grid must be at least 2 x 2 vertices, got {g.shape[:2]}")
		if not np.isfinite(g).all():
			raise ValueError(f"{name} grid has non-finite coordinates")
		if (g <= 0).all(axis=-1).any():
			raise ValueError(f"{name} grid has absent vertices (the (-1, -1, -1) sentinel); pass a window where every vertex is present")
	if fit_xyz.shape != start_xyz.shape:
		raise ValueError(f"start and fit grids must have the same shape, got {start_xyz.shape} and {fit_xyz.shape}")
	for name, g in (("start", start_xyz), ("fit", fit_xyz)):
		g64 = g.astype(np.float64)
		edges = min(float(np.linalg.norm(np.diff(g64, axis=0), axis=-1).min()), float(np.linalg.norm(np.diff(g64, axis=1), axis=-1).min()))
		if edges < MIN_EDGE_VOX:
			raise ValueError(f"{name} grid has coincident neighbouring vertices (an edge shorter than {MIN_EDGE_VOX} voxel)")
		cells = np.linalg.norm(np.cross(g64[1:, :-1] - g64[:-1, :-1], g64[:-1, 1:] - g64[:-1, :-1]), axis=-1)
		if float(cells.max()) < MIN_EDGE_VOX ** 2:
			raise ValueError(f"{name} grid is degenerate: all its vertices lie on a line, it spans no surface")
	vol = np.asarray(vol)
	if vol.ndim != 3:
		raise ValueError(f"CT crop must be a 3D array indexed (z, y, x), got {vol.ndim} dimensions")
	if vol.dtype != np.uint8:
		raise ValueError(f"CT crop must be uint8 (thresholds are 8-bit grey levels), got {vol.dtype}")
	if vol.size == 0 or min(vol.shape) < 2:
		raise ValueError(f"CT crop is empty or flat, shape {vol.shape}")
	if int(vol.min()) == int(vol.max()):
		raise ValueError(f"CT crop has a single grey level ({int(vol.min())}): an empty or masked region, there is no papyrus to snap onto")
	org = np.asarray(org, dtype=float)
	if org.shape != (3,) or not np.isfinite(org).all():
		raise ValueError("org must be the xyz position of the first voxel of the crop (3 finite numbers)")
	if offset_sign not in (1, -1):
		raise ValueError("offset_sign must be +1 or -1")
	return start_xyz, fit_xyz, vol, org


def _march_out(out, Q_m, state_m, dom_m, s1, p, stages):
	"""V6-FS: deliver a fallback march only if it places at least min_delivered_fraction of its domain."""
	if (state_m > 0).sum() < p["min_delivered_fraction"] * max(1, int(dom_m.sum())):
		out.update(xyz=s1.snapped, state=np.zeros_like(s1.state), fallback_refused="march placed too few vertices",
		           refusal_reason="march placed too few vertices")
		out.pop("fallback_height_from", None)
		_mark(stages, "surface_refused")
		return out
	out.update(xyz=Q_m, state=state_m, domain=dom_m, fallback_marched=True, delivery_basis="ct_march")
	_mark(stages, "surface_fallback_marched", xyz_final=Q_m, state_final=state_m)
	return out


def snap_offset_surface(
	start_xyz,
	fit_xyz,
	vol,
	org,
	offset_sign: float,
	params: dict = PARAMS,
	stages: dict | None = None,
	*,
	smooth_within_paper: bool = False,
	allow_march_fallback: bool = False,
	auto_orient: bool = False,
	repair_start_pockets: bool = False,
) -> dict:
	"""Snap, check and complete the surface of an Offset fit.

	stages : optional dict filled, for viewers, with the arrays each real step produced and the wall-clock time at which
			 each step finished (see _mark). The steps are whole passes over the surface; the order in which vertices are
			 visited inside a pass is not recorded. Passing it does not change any result.

	start_xyz, fit_xyz   : (h, w, 3) grids in voxel xyz of the CT level of `vol`; same shape.
	vol, org             : CT crop indexed (z, y, x) and the xyz position of its first voxel.
	offset_sign          : sign of the offset that produced fit_xyz from start_xyz (+1 or -1).
	smooth_within_paper  : opt-in (default False); smooth normal displacements within the CT papyrus ridge to reduce jumps.
	allow_march_fallback : opt-in (default False, experimental); when verdict is 0 or 2, march along the start normal with
	                       barrier guidance (heights from the CT dark-gap consensus, or from the fit on a drifted start or
	                       a blister), only if the fit moved at least fallback_min_fit_move_vox (45) along the start normal.
	auto_orient          : opt-in (default False); flip offset_sign if fit moved strongly to the opposite side of start.
	repair_start_pockets : opt-in (default False); on ACCEPTED surfaces, re-place vertices trapped in a start-winding pocket
	                       (V6-F behaviour that was tied to allow_march_fallback; separated in V6-FS).
	V6-FS safety rules for allow_march_fallback: the march runs only when the fit itself moved forward along the start
	normal (median >= min_forward_vox), so a wrong offset sign or a fit that stayed on the start winding is refused, not
	marched; a march that places fewer than min_delivered_fraction of the domain is refused; accepted surfaces are never
	modified by allow_march_fallback. A refused march is reported in fallback_refused.

	Returns a dict: xyz (dense snapped grid), state (0 / 1 / 2 / 3 / 4), domain (vertices whose start and arrival profiles
	lie in the crop), verdict (1 accepted, 0 stayed on the start winding, 2 double step or too few vertices placed,
	-1 undecided), accepted, recovered, grid_step, median_fit_normal_move_vox, auto_oriented, fallback_marched.
	When verdict is not 1 and allow_march_fallback is False, nothing is delivered: state is all zeros.
	"""
	p = params
	start_xyz, fit_xyz, vol, org = _check_inputs(start_xyz, fit_xyz, vol, org, offset_sign)
	step = grid_step(start_xyz)
	P0 = dense_grid(start_xyz, step, p["dense_step_vox"])
	P1 = dense_grid(fit_xyz, step, p["dense_step_vox"])
	_mark(stages, "grids_densified", start_dense=P0, fit_dense=P1, grid_step=step)
	s0 = snap_to_paper(P0, vol, org, p)
	_mark(stages, "start_snapped", start_snapped=s0.snapped, start_state=s0.state)
	s1 = snap_to_paper(P1, vol, org, p)
	_mark(stages, "fit_snapped", fit_snapped=s1.snapped, fit_state_after_snap=s1.state, fit_shift_after_snap=s1.shift, fit_normals=s1.normals, fit_inside=s1.inside)
	thr = 0.5 * (s0.threshold + s1.threshold)
	U0 = step_normal(P0, offset_sign)
	raw_move = float(np.median(np.einsum("ijk,ijk->ij", P1 - P0, U0)))
	auto_oriented = False
	if auto_orient and raw_move <= -2.0 * p["min_forward_vox"]:
		U0 = -U0
		raw_move = -raw_move
		auto_oriented = True
	mask0 = s0.inside & (s0.state > 0)
	mask1 = s1.inside & (s1.state > 0)
	events, b1, b2 = march_rays(s0.snapped[mask0], U0[mask0], vol, org, thr, p)
	b1_map, d1_smooth, d1_raw = consensus_target(events, b1, b2, mask0, p)
	_mark(stages, "rays_marched", rays=int(mask0.sum()), paper_threshold=thr)

	h_map, q1_proj = project_on_rays(s0.snapped, U0, mask0, s1.snapped, mask1, p["proj_radius_vox"])
	valid = mask0 & np.isfinite(h_map)
	domain = s0.inside & s1.inside
	raw_c0_frac = 0.0
	h_c0_med = 999.0
	h_c1_med = 0.0
	c_air_raw = np.zeros(0, dtype=int)
	if valid.sum() >= p["min_valid_rays"]:
		crossed = np.zeros(valid.sum(), dtype=int)
		crossed_raw = np.zeros(valid.sum(), dtype=int)
		h_vals = h_map[valid]
		fwd = h_vals >= p["min_forward_vox"]
		if fwd.any():
			c_air_raw = count_air(s0.snapped[valid][fwd], q1_proj[valid][fwd], vol, org, thr, p)
			crossed_raw[fwd] = np.minimum(c_air_raw, 2)
			c_air = np.minimum(c_air_raw, 2)
			c_air = np.where((c_air == 1) & (h_vals[fwd] - d1_smooth[valid][fwd] > p["double_step_excess_vox"]), 2, c_air)
			crossed[fwd] = c_air
		raw_c0_frac = float((crossed_raw == 0).sum()) / max(1, valid.sum())
		if (crossed_raw == 0).any() and (crossed_raw >= 1).any():
			h_c0_med = float(np.median(h_vals[crossed_raw == 0]))
			h_c1_med = float(np.median(h_vals[crossed_raw >= 1]))
		counts = np.bincount(crossed, minlength=3)
		# accepted when "one air gap" leads both rivals by more than accept_margin_share of the rays: a lead of a few rays
		# is a tie the check cannot break (with accept_margin_share = 0 this is the strict plurality of the first versions)
		if counts[1] - max(counts[0], counts[2]) > p["accept_margin_share"] * valid.sum():
			verdict = ACCEPTED
		elif counts[2] > counts[0]:
			verdict = 2
		else:
			verdict = 0
		if verdict == ACCEPTED and (domain & (s1.state > 0)).sum() < p["min_delivered_fraction"] * max(1, domain.sum()):
			verdict = 2
	else:
		verdict = -1
		counts = np.zeros(3, dtype=int)

	# evidence the decision rests on; all of it is measured on the CT, the start and the fit of this very surface
	n_valid = int(valid.sum())
	decided = verdict != -1
	w_first = np.array([w[0] if len(w) else np.nan for _, w in events], dtype=np.float32)
	packet_top = b1 - 0.5 * w_first                                # per start ray: where the paper of the start packet ends
	h_raw = np.einsum("ijk,ijk->ij", P1 - s0.snapped, U0)[mask0]   # raw fit above the snapped start, same grid index
	left_share = float((np.isfinite(packet_top) & (h_raw >= packet_top)).sum()) / max(1, int(mask0.sum()))
	past_mid_share = float((np.isfinite(b1) & (h_raw >= b1)).sum()) / max(1, int(mask0.sum()))   # nearer the next sheet than the start packet
	evidence = {
		"fit_move_vox": round(raw_move, 2),
		"one_gap_share": round(float(counts[1]) / n_valid, 4) if decided else None,
		"no_gap_share": round(float(counts[0]) / n_valid, 4) if decided else None,
		"double_share": round(float(counts[2]) / n_valid, 4) if decided else None,
		"acceptance_margin": round(float(counts[1] - max(counts[0], counts[2])) / n_valid, 4) if decided else None,
		"fit_left_start_packet_share": round(left_share, 4),
		"fit_past_mid_gap_share": round(past_mid_share, 4),
		"start_on_paper_share": round(float(mask0.mean()), 4),
		"fit_anchored_share": round(float((s1.state > 0).mean()), 4),
		"step_height_vox": None,
	}
	out = {
		"verdict": verdict,
		"accepted": verdict == ACCEPTED,
		"domain": domain,
		"grid_step": step,
		"recovered": 0,
		"median_fit_normal_move_vox": round(raw_move, 2),
		"auto_oriented": auto_oriented,
		"fallback_marched": False,
		"fallback_refused": None,
		"delivery_basis": None,
		"refusal_reason": None,
		"evidence": evidence,
		"valid_rays": n_valid,
		"start_vertices_on_paper": int(mask0.sum()),
		"start_pockets_repaired": 0,
		"start_pockets_rolled_back": 0,
	}
	_mark(stages, "surface_checked", verdict=verdict, domain=domain, valid_rays=n_valid)

	def refuse(reason):
		out.update(xyz=s1.snapped, state=np.zeros_like(s1.state), refusal_reason=reason)
		out.pop("fallback_height_from", None)
		if allow_march_fallback:
			out["fallback_refused"] = reason
		_mark(stages, "surface_refused")
		return out

	if verdict != ACCEPTED:
		why = {0: "the snapped fit did not cross one air gap from the start winding (it stayed on it, or the windings touch without air)",
		       2: "double step, or too few vertices placed",
		       -1: "the check was undecided (too few valid rays)"}[verdict]
		if decided and 0 < counts[1] - max(counts[0], counts[2]) <= p["accept_margin_share"] * n_valid:
			why = (f"near tie: {100 * counts[1] / n_valid:.1f} % of the rays crossed one air gap, {100 * counts[0] / n_valid:.1f} % none and "
			       f"{100 * counts[2] / n_valid:.1f} % two; a lead under accept_margin_share = {100 * p['accept_margin_share']:.0f} % is not an acceptance")
		if not allow_march_fallback:
			return refuse(why)
		if not mask0.any():
			return refuse("no start vertex on papyrus")
		if verdict == -1:
			return refuse("the check was undecided (too few valid rays): nothing to complete or march against")
		if raw_move < p["min_forward_vox"]:
			return refuse("fit did not move forward along the start normal (sign or stayed fit)")

		# fallback 1: the sheet the fit itself designates. Where two windings touch there is no air gap to count, yet the fit
		# sits on a papyrus ridge a full step above the start: deliver that sheet, never look for another one.
		if raw_move < p["fit_sheet_min_step_vox"]:
			why_fit = (f"the fit moved {raw_move:.1f} voxels along the start normal, less than one sheet thickness (fit_sheet_min_step_vox = "
			           f"{p['fit_sheet_min_step_vox']:g}): it designates no other sheet than the start")
		elif left_share >= p["fit_sheet_max_left_share"]:
			why_fit = (f"the raw fit lies above the top of the start packet on {100 * left_share:.0f} % of the rays: it left that packet "
			           "as a whole and is not on a sheet pressed against the start")
		elif evidence["double_share"] <= p["fit_sheet_max_double_share"]:
			# Candidate ridges closer to the start than one sheet thickness are the start winding itself: they are excluded
			# before snapping, so that a vertex is hooked to the sheet the fit designates or left out, never pulled back.
			start_map = np.zeros(mask0.shape, dtype=np.float32)
			q0_p, u0_p, _ = footpoints(s0.snapped, U0, mask0, start_map, P1)
			barrier = np.full(mask0.shape, p["fit_sheet_min_step_vox"], dtype=np.float32)
			sf = snap_to_paper(P1, vol, org, p, q0_foot=q0_p, u0_foot=u0_p, b1_foot=barrier)
			Qf, state_f, _, n_rec = recover(sf, sf.snapped, sf.state, sf.shift, p)
			q0_f, u0_f, b1_f = footpoints(s0.snapped, U0, mask0, b1_map, Qf)
			height = np.einsum("ijk,ijk->ij", Qf - q0_f, u0_f)
			placed = domain & sf.inside & (state_f > 0)
			need = p["min_delivered_fraction"] * max(1, int(domain.sum()))
			if placed.sum() < need:
				why_fit = "the snapped fit places too few vertices to be a sheet"
			else:
				step_h = float(np.median(height[placed]))
				evidence["step_height_vox"] = round(step_h, 2)
				if step_h < p["fit_sheet_min_step_vox"]:
					why_fit = (f"the snapped fit sits {step_h:.1f} voxels above the start, less than fit_sheet_min_step_vox = "
					           f"{p['fit_sheet_min_step_vox']:g}: it is still on the start winding")
				else:
					local = _local_step(height, placed)
					floor = p["fit_sheet_stayed_fraction"] * np.maximum(np.nan_to_num(local, nan=step_h), step_h)
					state_f = np.where(height < floor, 0, state_f).astype(np.int8)      # vertices left on the start side
					if (domain & (state_f > 0)).sum() < need:
						why_fit = "too few vertices of the snapped fit lie a full step above the start"
					else:
						if smooth_within_paper:
							_mark(stages, "before_smooth", xyz_unsmoothed=Qf.copy(), state_unsmoothed=state_f.copy(), domain_unsmoothed=domain.copy())
							Qf = regularize_within_paper(sf, Qf, state_f, domain, vol, org, thr, q0_f, u0_f, b1_f, p)
						kept = domain & (state_f > 0)
						evidence["fit_to_sheet_shift_vox"] = round(float(np.median(np.linalg.norm(Qf[kept] - P1[kept], axis=-1))), 2)
						out.update(xyz=Qf, state=state_f, recovered=n_rec, fallback_marched=True, delivery_basis="fit_sheet")
						_mark(stages, "surface_fit_sheet_delivered", xyz_final=Qf, state_final=state_f)
						return out
		else:
			why_fit = (f"{100 * evidence['double_share']:.0f} % of the rays crossed two air gaps: the fit is one winding too far")

		# fallback 2: CT march (V6-F), only for a fit that went past the middle of the first air gap: a fit floating in the
		# lower half of that gap is nearer the start packet than the next sheet, and says nothing a march could rely on
		if past_mid_share < p["march_min_past_mid_gap_share"]:
			return refuse(f"{why_fit}; no march either: the raw fit is past the middle of the first air gap on {100 * past_mid_share:.0f} % "
			              f"of the rays, less than march_min_past_mid_gap_share = {100 * p['march_min_past_mid_gap_share']:.0f} %")
		b1_map_v6f, d1_smooth_v6f, d1_raw_v6f, prom_map = consensus_target_v6f(events, b1, b2, mask0, p)
		d1_med = float(np.nanmedian(np.where(mask0, d1_smooth_v6f, np.nan))) if np.isfinite(d1_smooth_v6f[mask0]).any() else np.nan
		out["fallback_target_height_vox"] = round(d1_med, 2) if np.isfinite(d1_med) else None
		if not np.isfinite(d1_med):
			return refuse(f"{why_fit}; no march either: the CT shows no sheet beyond an air gap")
		if raw_move < p["fallback_min_fit_move_vox"]:
			return refuse(f"{why_fit}; no march either: the fit moved {raw_move:.1f} voxels along the start normal, less than "
			              f"fallback_min_fit_move_vox = {p['fallback_min_fit_move_vox']:g}")
		d0_smooth = ndimage.gaussian_filter(_inpaint_nearest(s0.shift, s0.state > 0), sigma=2.0)
		Q0_full = np.where((s0.state > 0)[..., None], s0.snapped, P0 + d0_smooth[..., None] * s0.normals)
		d1_filled = np.where(np.isfinite(d1_raw_v6f), d1_raw_v6f, d1_smooth_v6f)
		s0_u0 = np.einsum("ijk,ijk->ij", s0.snapped - P0, U0)
		s0_drifted = (float(np.nanpercentile(s0_u0[mask0], 85)) >= 18.0) and (float(np.nanmedian(s0_u0[mask0])) >= 8.0) and (raw_move >= 20.0)
		blister_rescued = False
		if verdict == 2 and valid.sum() >= p["min_valid_rays"] and prom_map.sum() >= 0.25 * mask0.sum() and len(c_air_raw):
			prom_fwd = prom_map[valid][fwd]
			d1_fwd = d1_smooth_v6f[valid][fwd]
			b1_fwd = b1_map_v6f[valid][fwd]
			c_adj = np.where(prom_fwd & (c_air_raw >= 2) & (h_vals[fwd] >= b1_fwd) & (h_vals[fwd] <= d1_fwd + p["double_step_excess_vox"]), c_air_raw - 1, c_air_raw)
			c_adj = np.minimum(c_adj, 2)
			c_adj = np.where((c_adj == 1) & (h_vals[fwd] - d1_fwd > p["double_step_excess_vox"]), 2, c_adj)
			cr_adj = np.zeros(valid.sum(), dtype=int)
			cr_adj[fwd] = c_adj
			cnt_adj = np.bincount(cr_adj, minlength=3)
			if cnt_adj[1] > cnt_adj[0] and cnt_adj[1] > cnt_adj[2]:
				blister_rescued = True
		out["fallback_refused"] = None
		if s0_drifted or blister_rescued:
			Q0_ref = P0 if s0_drifted else Q0_full
			out["fallback_height_from"] = "fit"   # heights projected from the snapped fit (drifted start or blister)
			h_p0, _ = project_on_rays(Q0_ref, U0, s0.inside, s1.snapped, mask1, 24.0)
			valid_h = np.isfinite(h_p0) & (h_p0 >= 15.0)
			if valid_h.any():
				h_smooth = ndimage.gaussian_filter(_inpaint_nearest(h_p0, valid_h), sigma=3.0)
				b1_h = np.clip(0.55 * h_smooth, 12.0, 85.0)
				Pc = Q0_ref + h_smooth[..., None] * U0
				sm = snap_to_paper(Pc, vol, org, p, q0_foot=Q0_ref, u0_foot=U0, b1_foot=b1_h, d1_foot=h_smooth)
				dom_m = s0.inside & sm.inside
				state_m = np.where(dom_m & (sm.state > 0), np.int8(4), np.int8(0))
				if smooth_within_paper:
					_mark(stages, "before_smooth", xyz_unsmoothed=sm.snapped.copy(), state_unsmoothed=state_m.copy(), domain_unsmoothed=dom_m.copy())
				Q_m = regularize_within_paper(sm, sm.snapped, state_m, dom_m, vol, org, thr, Q0_ref, U0, b1_h, p) if smooth_within_paper else sm.snapped
				return _march_out(out, Q_m, state_m, dom_m, s1, p, stages)
		out["fallback_height_from"] = "ct_consensus"   # heights from the CT barrier consensus, not from the fit
		Pc = Q0_full + d1_filled[..., None] * U0
		sm = snap_to_paper(Pc, vol, org, p, q0_foot=Q0_full, u0_foot=U0, b1_foot=b1_map_v6f, d1_foot=d1_filled)
		dom_m = s0.inside & sm.inside
		state_m = np.where(dom_m & (sm.state > 0), np.int8(4), np.int8(0))
		if smooth_within_paper:
			_mark(stages, "before_smooth", xyz_unsmoothed=sm.snapped.copy(), state_unsmoothed=state_m.copy(), domain_unsmoothed=dom_m.copy())
		Q_m = regularize_within_paper(sm, sm.snapped, state_m, dom_m, vol, org, thr, Q0_full, U0, b1_map_v6f, p) if smooth_within_paper else sm.snapped
		return _march_out(out, Q_m, state_m, dom_m, s1, p, stages)

	# accepted: one air gap crossed
	if repair_start_pockets and mask0.any():
		b1_map_v6f, d1_smooth_v6f, d1_raw_v6f, prom_map = consensus_target_v6f(events, b1, b2, mask0, p)
	else:
		b1_map_v6f, d1_smooth_v6f, d1_raw_v6f = b1_map, d1_smooth, d1_raw
		prom_map = np.zeros(mask0.shape, dtype=bool)
	Q, state, _, n_rec = recover(s1, s1.snapped, s1.state, s1.shift, p)
	_mark(stages, "vertices_recovered", xyz_after_recover=Q, state_after_recover=state, recovered=n_rec)
	q0_f, u0_f, b1_f = footpoints(s0.snapped, U0, mask0, b1_map_v6f, Q)
	height = np.einsum("ijk,ijk->ij", Q - q0_f, u0_f)
	Q_before_repair, state_before_repair, height_before_repair = Q.copy(), state.copy(), height.copy()
	if repair_start_pockets and mask0.any():
		has_start_pocket = (prom_map.sum() >= 0.14 * max(1, mask0.sum())) or (raw_c0_frac >= 0.25 and h_c0_med <= 27.5 and (h_c1_med - h_c0_med) <= 45.0)
		if has_start_pocket and raw_c0_frac >= 0.14:
			d1_filled = np.where(np.isfinite(d1_raw_v6f), d1_raw_v6f, d1_smooth_v6f)
			_, _, d1_f = footpoints(s0.snapped, U0, mask0, d1_filled, Q)
			_, _, prom_f = footpoints(s0.snapped, U0, mask0, prom_map.astype(np.float32), Q)
			act_low = domain & (state > 0) & np.isfinite(b1_f) & (height < b1_f - 2.0)
			if act_low.any():
				c_low = count_air(q0_f[act_low], Q[act_low], vol, org, thr, p)
				req_air = np.where(prom_f[act_low] > 0.5, 2, 1)
				really_trapped = (c_low < req_air) | (height[act_low] < np.minimum(14.0, b1_f[act_low] - 6.0))
				trapped = np.zeros(domain.shape, dtype=bool)
				for (i, j), is_t in zip(np.argwhere(act_low), really_trapped):
					if is_t:
						trapped[i, j] = True
				if trapped.sum() >= 0.04 * max(1, domain.sum()):
					trapped |= (domain & (state == 0) & np.isfinite(b1_f))
				if trapped.any():
					for i, j in zip(*np.nonzero(trapped)):
						rg = s1.ridges[i][j]
						if len(rg):
							pts_r = s1.points[i, j][None, :] + rg[:, None] * s1.normals[i, j][None, :]
							h_r = (pts_r - q0_f[i, j]) @ u0_f[i, j]
							ok_r = (h_r >= b1_f[i, j]) & (np.abs(h_r - d1_f[i, j]) <= 22.0)
							if ok_r.any():
								cand_rg = rg[ok_r]
								cand_hr = h_r[ok_r]
								best = np.abs(cand_hr - d1_f[i, j]).argmin()
								Q[i, j] = s1.points[i, j] + float(cand_rg[best]) * s1.normals[i, j]
								state[i, j] = 3
								height[i, j] = float(cand_hr[best])
								trapped[i, j] = False
					still_trapped = trapped & (height < b1_f - 2.0)
					if still_trapped.sum() >= 0.02 * max(1, domain.sum()):
						Pc_acc = q0_f + d1_f[..., None] * u0_f
						sm_acc = snap_to_paper(Pc_acc, vol, org, p, q0_foot=q0_f, u0_foot=u0_f, b1_foot=b1_f, d1_foot=d1_f)
						rep = still_trapped & sm_acc.inside & (sm_acc.state > 0)
						Q = np.where(rep[..., None], sm_acc.snapped, Q)
						state = np.where(rep, np.int8(4), state)
		# transaction: only depressions below the level of the one-gap vertices may be repaired, and only back to that level
		if ((state != state_before_repair) | (Q != Q_before_repair).any(-1)).any():
			placed_b = domain & (state_before_repair > 0)
			one_gap = np.zeros(domain.shape, dtype=bool)
			one_gap[placed_b] = count_air(q0_f[placed_b], Q_before_repair[placed_b], vol, org, thr, p) == 1
			level = _local_step(height_before_repair, one_gap)
			if one_gap.any():
				level = np.where(np.isfinite(level), level, float(np.median(height_before_repair[one_gap])))
			height_after = np.einsum("ijk,ijk->ij", Q - q0_f, u0_f)
			Q, state, kept, rolled = commit_repairs(Q_before_repair, state_before_repair, height_before_repair, Q, state, height_after, level, p)
			out["start_pockets_repaired"], out["start_pockets_rolled_back"] = kept, rolled
		height = np.einsum("ijk,ijk->ij", Q - q0_f, u0_f)
	stayed = np.isfinite(b1_f) & (height < np.minimum(10.0, b1_f - 6.0))     # vertices left on the start winding
	if stayed.any():
		state = np.where(stayed, 0, state)
	if smooth_within_paper:
		_mark(stages, "before_smooth", xyz_unsmoothed=Q.copy(), state_unsmoothed=state.copy(), domain_unsmoothed=domain.copy())
		Q = regularize_within_paper(s1, Q, state, domain, vol, org, thr, q0_f, u0_f, b1_f, p)
	placed = domain & (state > 0)
	if placed.any():
		evidence["step_height_vox"] = round(float(np.median(height[placed])), 2)
	out.update(xyz=Q, state=state, recovered=n_rec, delivery_basis="air_gap")
	_mark(stages, "surface_delivered", stayed=stayed, xyz_final=Q, state_final=state)
	return out
