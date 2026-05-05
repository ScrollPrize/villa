import argparse
import copy
import dataclasses
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import cli_data
import cli_json
import cli_model
import cli_opt
import fit_data
import model
import opt_loss_corr
import opt_loss_dir
import optimizer


def _grid_center(mdl: "model.Model3D") -> torch.Tensor:
	"""Bilinear center of the model grid — matches (Hm-1)/2, (Wm-1)/2 in station loss."""
	xyz = mdl._grid_xyz()  # (D, Hm, Wm, 3)
	Hm, Wm = xyz.shape[1], xyz.shape[2]
	h_mid, w_mid = (Hm - 1) / 2.0, (Wm - 1) / 2.0
	h0, w0 = int(h_mid), int(w_mid)
	h1, w1 = min(h0 + 1, Hm - 1), min(w0 + 1, Wm - 1)
	fh, fw = h_mid - h0, w_mid - w0
	return ((1 - fh) * (1 - fw) * xyz[0, h0, w0]
	      + fh * (1 - fw) * xyz[0, h1, w0]
	      + (1 - fh) * fw * xyz[0, h0, w1]
	      + fh * fw * xyz[0, h1, w1])


def _arc_params_from_seed(
	seed: tuple[int, int, int],
	model_w: int,
	volume_extent_fullres: tuple[int, int, int],
	umbilicus_center: tuple[float, float] | None = None,
) -> dict:
	"""Derive arc params from seed point and model width.

	seed: (cx, cy, cz) in fullres voxels
	model_w: model width in fullres voxels (circumferential extent)
	volume_extent_fullres: (X_total, Y_total, Z_total) in fullres voxels
	Returns dict with keys matching ModelConfig arc fields.
	"""
	seed_cx, seed_cy, seed_cz = float(seed[0]), float(seed[1]), float(seed[2])
	vol_x, vol_y, _vol_z = volume_extent_fullres

	if umbilicus_center is not None:
		arc_cx, arc_cy = umbilicus_center
	else:
		# Volume center XY = fallback estimate of scroll axis
		arc_cx = vol_x / 2.0
		arc_cy = vol_y / 2.0

	# Arc radius = distance from volume center to seed point
	dx = seed_cx - arc_cx
	dy = seed_cy - arc_cy
	arc_radius = math.sqrt(dx * dx + dy * dy)
	arc_radius = max(arc_radius, 100.0)

	# Seed angle
	seed_angle = math.atan2(dy, dx)

	# Angular half-extent from model width
	half_angle = float(model_w) / (2.0 * arc_radius)
	half_angle = max(half_angle, 0.05)

	return {
		"arc_cx": arc_cx,
		"arc_cy": arc_cy,
		"arc_radius": arc_radius,
		"arc_angle0": seed_angle - half_angle,
		"arc_angle1": seed_angle + half_angle,
		"z_center": seed_cz,
	}


def _load_umbilicus_points(path: str) -> list[tuple[float, float, float]]:
	p = Path(path)
	if p.suffix.lower() == ".json":
		document = json.loads(p.read_text(encoding="utf-8"))
		entries = (
			document.get("points", document.get("control_points", document))
			if isinstance(document, dict)
			else document
		)
		if not isinstance(entries, list):
			raise ValueError(
				"umbilicus JSON root must be an array or contain a "
				"'points' or 'control_points' array"
			)
		points: list[tuple[float, float, float]] = []
		for i, entry in enumerate(entries):
			if isinstance(entry, dict):
				points.append((float(entry["x"]), float(entry["y"]), float(entry["z"])))
			elif isinstance(entry, list) and len(entry) >= 3:
				z, y, x = entry[:3]
				points.append((float(x), float(y), float(z)))
			else:
				raise ValueError(f"unsupported umbilicus JSON entry at index {i}")
		if not points:
			raise ValueError("umbilicus file contained no points")
		return points

	points = []
	for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		parts = [part.strip() for part in line.split(",") if part.strip()]
		if len(parts) != 3:
			raise ValueError(f"umbilicus text line {line_no} must contain z,y,x")
		z, y, x = (float(v) for v in parts)
		points.append((x, y, z))
	if not points:
		raise ValueError("umbilicus file contained no points")
	return points


def _umbilicus_center_at_z(path: str, z: float, *, scale: float = 1.0) -> tuple[float, float]:
	points = _load_umbilicus_points(path)
	points = [(x * scale, y * scale, z0 * scale) for x, y, z0 in points]
	points.sort(key=lambda p: p[2])
	if z <= points[0][2]:
		return points[0][0], points[0][1]
	if z >= points[-1][2]:
		return points[-1][0], points[-1][1]
	for p0, p1 in zip(points, points[1:]):
		if p0[2] <= z <= p1[2]:
			dz = p1[2] - p0[2]
			t = 0.0 if dz == 0.0 else (z - p0[2]) / dz
			return p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
	return points[-1][0], points[-1][1]


def _straight_params_from_seed(
	seed: tuple[int, int, int],
	model_w: int,
) -> dict:
	"""Derive straight params from seed point and model width.

	The line is centered at the seed XY, oriented at angle 0 (along X),
	with half-width = model_w / 2. The optimizer will adjust angle.
	"""
	return {
		"straight_cx": float(seed[0]),
		"straight_cy": float(seed[1]),
		"straight_angle": 0.0,
		"straight_half_w": float(model_w) / 2.0,
		"z_center": float(seed[2]),
	}


def _parse_corr_points(obj: dict, device: torch.device) -> fit_data.CorrPoints3D | None:
	"""Parse a VC3D corr_points collections dict into CorrPoints3D."""
	cols = obj.get("collections", {})
	print(f"[fit] _parse_corr_points: {len(cols) if isinstance(cols, dict) else 0} collections in input", flush=True)
	if not isinstance(cols, dict):
		print(f"[fit] _parse_corr_points: collections is not a dict: {type(cols).__name__}", flush=True)
		return None
	rows: list[list[float]] = []
	cids: list[int] = []
	pids: list[int] = []
	abs_flags: list[bool] = []
	for _cid, col in cols.items():
		if not isinstance(col, dict):
			print(f"[fit] _parse_corr_points: col {_cid} is not a dict", flush=True)
			continue
		md = col.get("metadata", {})
		if not isinstance(md, dict):
			md = {}
		is_abs = bool(md.get("winding_is_absolute", True))
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		try:
			cid_i = int(_cid)
		except Exception:
			cid_i = -1
		n_pts = 0
		for _pid, pd in pts.items():
			if not isinstance(pd, dict):
				continue
			pv = pd.get("p", None)
			if not isinstance(pv, (list, tuple)) or len(pv) < 3:
				continue
			wa = pd.get("wind_a", None)
			if wa is None:
				print(f"[fit] WARNING: corr point {_pid} in collection {_cid} has no wind_a, skipping")
				continue
			try:
				pid_i = int(_pid)
			except Exception:
				pid_i = -1
			rows.append([float(pv[0]), float(pv[1]), float(pv[2]), float(wa)])
			cids.append(cid_i)
			pids.append(pid_i)
			abs_flags.append(is_abs)
			n_pts += 1
		print(f"[fit] _parse_corr_points: col {_cid}: {n_pts} points, "
			  f"absolute={is_abs}", flush=True)
	if not rows:
		print(f"[fit] _parse_corr_points: no valid points found after parsing", flush=True)
		return None
	pts_t = torch.tensor(rows, dtype=torch.float32, device=device)
	col_t = torch.tensor(cids, dtype=torch.int64, device=device)
	pid_t = torch.tensor(pids, dtype=torch.int64, device=device)
	abs_t = torch.tensor(abs_flags, dtype=torch.bool, device=device)
	n_abs = int(abs_t.sum().item())
	print(f"[fit] loaded {pts_t.shape[0]} corr_points from config "
		  f"({len(set(cids))} collections, {n_abs} absolute, {len(rows) - n_abs} relative)")
	return fit_data.CorrPoints3D(points_xyz_winda=pts_t, collection_idx=col_t,
								 point_ids=pid_t, is_absolute=abs_t)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="3D fit entrypoint",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	p.add_argument("--out-dir", default=None, help="Output directory for snapshots and debug")
	p.add_argument("--tifxyz-init", default=None, help="Initialize model from tifxyz directory instead of model.pt or new model")
	p.add_argument("--window-size", type=int, default=None,
		help="Window size in fullres voxels for windowed tifxyz optimization (0 or omit = no windowing)")
	p.add_argument("--window-overlap", type=int, default=0,
		help="Overlap between windows in fullres voxels")
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def _optimizer_stage_cfg(cfg: dict) -> dict:
	stage_cfg = dict(cfg)
	stage_cfg.pop("args", None)
	stage_cfg.pop("voxel_size_um", None)
	stage_cfg.pop("external_surfaces", None)
	stage_cfg.pop("tifxyz_data", None)
	stage_cfg.pop("offset_value", None)
	stage_cfg.pop("output_scale", None)
	stage_cfg.pop("mode", None)
	stage_cfg.pop("boundary_anchor", None)
	return stage_cfg


def _valid_xyz_np(xyz) -> "np.ndarray":
	import numpy as np
	arr = xyz.detach().cpu().numpy()
	return np.isfinite(arr).all(axis=-1) & (arr != -1.0).all(axis=-1)


def _point_grid_index(pd: dict, full_xyz_np, full_valid_np) -> tuple[int, int] | None:
	import numpy as np
	grid = pd.get("grid", None)
	if isinstance(grid, (list, tuple)) and len(grid) >= 2:
		try:
			return int(grid[0]), int(grid[1])
		except Exception:
			pass
	p = pd.get("p", None)
	if not isinstance(p, (list, tuple)) or len(p) < 3:
		return None
	pt = np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
	if not np.isfinite(pt).all() or not full_valid_np.any():
		return None
	diff = full_xyz_np - pt.reshape(1, 1, 3)
	dist2 = np.einsum("...i,...i->...", diff, diff)
	dist2 = np.where(full_valid_np, dist2, np.inf)
	idx = int(np.argmin(dist2))
	if not np.isfinite(dist2.reshape(-1)[idx]):
		return None
	return tuple(int(v) for v in np.unravel_index(idx, dist2.shape))


def _filter_corr_points_for_window(corr_points_obj: dict | None, *,
								   h0: int, h1: int, w0: int, w1: int,
								   full_xyz_np, full_valid_np) -> dict | None:
	if not isinstance(corr_points_obj, dict):
		return None
	cols = corr_points_obj.get("collections", {})
	if not isinstance(cols, dict):
		return None
	out_cols: dict = {}
	for cid, col in cols.items():
		if not isinstance(col, dict):
			continue
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		out_pts: dict = {}
		for pid, pd in pts.items():
			if not isinstance(pd, dict):
				continue
			idx = _point_grid_index(pd, full_xyz_np, full_valid_np)
			if idx is None:
				continue
			r, c = idx
			if h0 <= r < h1 and w0 <= c < w1:
				out_pts[pid] = copy.deepcopy(pd)
		if out_pts:
			out_col = copy.deepcopy(col)
			out_col["points"] = out_pts
			md = out_col.setdefault("metadata", {})
			if isinstance(md, dict):
				md["winding_is_absolute"] = True
			out_cols[cid] = out_col
	if not out_cols:
		return None
	out = copy.deepcopy(corr_points_obj)
	out["collections"] = out_cols
	return out


def _merge_window_tifxyz(*, windows: list[Path], out_dir: Path,
						 base_xyz, base_valid, scale: float,
						 voxel_size_um: float | None,
						 preserve_existing: bool = True) -> None:
	import json as _json
	import numpy as np
	import tifffile
	import fit2tifxyz as _f2t

	base_np = base_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
	base_valid_np = base_valid.detach().cpu().numpy().astype(bool, copy=False)
	H, W, _ = base_np.shape
	x_all = np.full((H, W), -1.0, dtype=np.float32)
	y_all = np.full((H, W), -1.0, dtype=np.float32)
	z_all = np.full((H, W), -1.0, dtype=np.float32)
	locked = np.zeros((H, W), dtype=bool)
	if preserve_existing:
		x_all[base_valid_np] = base_np[..., 0][base_valid_np]
		y_all[base_valid_np] = base_np[..., 1][base_valid_np]
		z_all[base_valid_np] = base_np[..., 2][base_valid_np]
		locked = base_valid_np.copy()
	best = np.full((H, W), -np.inf, dtype=np.float32)

	for win_dir in windows:
		meta = _json.loads((win_dir / "meta.json").read_text(encoding="utf-8"))
		h0, w0 = [int(v) for v in meta["window_origin_verts"]]
		wh, ww = [int(v) for v in meta["window_size_verts"]]
		x = tifffile.imread(str(win_dir / "x.tif")).astype(np.float32)
		y = tifffile.imread(str(win_dir / "y.tif")).astype(np.float32)
		z = tifffile.imread(str(win_dir / "z.tif")).astype(np.float32)
		wh = min(wh, x.shape[0], H - h0)
		ww = min(ww, x.shape[1], W - w0)
		if wh <= 0 or ww <= 0:
			continue
		x = x[:wh, :ww]
		y = y[:wh, :ww]
		z = z[:wh, :ww]
		valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & ~((x == -1.0) & (y == -1.0) & (z == -1.0))
		rr = np.arange(wh, dtype=np.float32)[:, None]
		cc = np.arange(ww, dtype=np.float32)[None, :]
		row_weight = np.minimum(rr + 1.0, wh - rr)
		col_weight = np.minimum(cc + 1.0, ww - cc)
		weight = np.minimum(row_weight, col_weight)
		dst = (slice(h0, h0 + wh), slice(w0, w0 + ww))
		can_write = valid & ~locked[dst] & (weight > best[dst])
		x_dst = x_all[dst]
		y_dst = y_all[dst]
		z_dst = z_all[dst]
		best_dst = best[dst]
		x_dst[can_write] = x[can_write]
		y_dst[can_write] = y[can_write]
		z_dst[can_write] = z[can_write]
		best_dst[can_write] = weight[can_write]

	area = _f2t._get_area(x_all, y_all, z_all, 1.0 / scale, voxel_size_um)
	_f2t._write_tifxyz(out_dir=out_dir, x=x_all, y=y_all, z=z_all,
						scale=scale, area=area)
	meta_path = out_dir / "meta.json"
	meta = _json.loads(meta_path.read_text(encoding="utf-8"))
	meta["extend_merged"] = True
	meta["preserved_existing_vertices"] = bool(preserve_existing)
	meta["source_grid_size_verts"] = [int(H), int(W)]
	meta["window_count"] = len(windows)
	meta_path.write_text(_json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	_f2t._print_area(area)


def _extend_padding_from_cfg(cfg: dict) -> tuple[int, int, int, int]:
	direction = str(cfg.get("extend_direction", "all")).lower()
	steps = max(1, int(cfg.get("extend_steps", 1)))
	if direction == "left":
		return 0, 0, steps, 0
	if direction == "right":
		return 0, 0, 0, steps
	if direction == "up":
		return steps, 0, 0, 0
	if direction == "down":
		return 0, steps, 0, 0
	return steps, steps, steps, steps


def _pad_tifxyz_for_extend(full_xyz: torch.Tensor,
						   full_valid: torch.Tensor,
						   cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
	top, bottom, left, right = _extend_padding_from_cfg(cfg)
	if top == 0 and bottom == 0 and left == 0 and right == 0:
		return full_xyz, full_valid
	H, W, _ = full_xyz.shape
	new_h = H + top + bottom
	new_w = W + left + right
	padded_xyz = torch.full((new_h, new_w, 3), -1.0, dtype=full_xyz.dtype, device=full_xyz.device)
	padded_valid = torch.zeros((new_h, new_w), dtype=full_valid.dtype, device=full_valid.device)
	padded_xyz[top:top + H, left:left + W] = full_xyz
	padded_valid[top:top + H, left:left + W] = full_valid
	print(f"[fit] extend mode: padded grid {H}x{W} -> {new_h}x{new_w} "
		  f"(top={top} bottom={bottom} left={left} right={right})", flush=True)
	return padded_xyz, padded_valid


def _optional_channels_to_skip(stages: list["optimizer.Stage"]) -> set[str]:
	def _term_active(term: str) -> bool:
		return any(
			s.global_opt.steps > 0 and float(s.global_opt.eff.get(term, 0.0)) > 0.0
			for s in stages
		)

	skip: set[str] = set()
	if not (_term_active("data") or _term_active("data_plain")):
		skip.add("cos")
	if not _term_active("pred_dt"):
		skip.add("pred_dt")
	return skip


def _compute_window_grid(
	H: int, W: int, mesh_step: int, window_size: int, overlap: int,
) -> list[tuple[int, int, int, int]]:
	"""Compute window tiles over a (H, W) vertex grid.

	window_size and overlap are in fullres voxels.
	Returns list of (h0, h1, w0, w1) in vertex indices.
	"""
	if overlap >= window_size:
		raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")
	win_verts = window_size // mesh_step + 1
	overlap_verts = overlap // mesh_step
	stride = max(1, win_verts - overlap_verts)
	windows = []
	h = 0
	while h < H:
		h1 = min(h + win_verts, H)
		w = 0
		while w < W:
			w1 = min(w + win_verts, W)
			windows.append((h, h1, w, w1))
			if w1 == W:
				break
			w += stride
		if h1 == H:
			break
		h += stride
	return windows


def _slice_tifxyz_with_invalid_padding(
	xyz: torch.Tensor,
	valid: torch.Tensor,
	*,
	h0: int,
	h1: int,
	w0: int,
	w1: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Slice a vertex grid, filling out-of-bounds halo with invalid vertices."""
	H, W, C = xyz.shape
	out_h = max(0, h1 - h0)
	out_w = max(0, w1 - w0)
	out_xyz = torch.full((out_h, out_w, C), -1.0, dtype=xyz.dtype, device=xyz.device)
	out_valid = torch.zeros((out_h, out_w), dtype=valid.dtype, device=valid.device)

	src_h0 = max(0, h0)
	src_h1 = min(H, h1)
	src_w0 = max(0, w0)
	src_w1 = min(W, w1)
	if src_h1 <= src_h0 or src_w1 <= src_w0:
		return out_xyz, out_valid

	dst_h0 = src_h0 - h0
	dst_w0 = src_w0 - w0
	dst_h1 = dst_h0 + (src_h1 - src_h0)
	dst_w1 = dst_w0 + (src_w1 - src_w0)
	out_xyz[dst_h0:dst_h1, dst_w0:dst_w1] = xyz[src_h0:src_h1, src_w0:src_w1]
	out_valid[dst_h0:dst_h1, dst_w0:dst_w1] = valid[src_h0:src_h1, src_w0:src_w1]
	return out_xyz, out_valid


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]

	parser = _build_parser()
	cfg_paths, argv_rest = cli_json.split_cfg_argv(argv)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = cli_json.merge_cfgs(cfg_paths)
	fit_config = copy.deepcopy(cfg)
	cli_json.apply_defaults_from_cfg_args(parser, cfg)
	args = parser.parse_args(argv_rest or [])
	# Merge final parsed args into fit_config so checkpoint has all values
	fit_config.setdefault("args", {}).update(
		{k.replace("_", "-"): v for k, v in vars(args).items()})

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	progress_enabled = bool(args.progress)
	_out_dir = args.out_dir
	output_scale = float(cfg.get("output_scale", 1.0))
	if not math.isfinite(output_scale) or output_scale <= 0.0:
		raise ValueError(f"output_scale must be positive and finite, got {output_scale}")
	stage_cfg = _optimizer_stage_cfg(cfg)
	stages = optimizer.load_stages_cfg(stage_cfg)
	skip_channels = _optional_channels_to_skip(stages)
	if "cos" in skip_channels:
		print("[fit] skipping optional channel 'cos': data/data_plain losses are disabled", flush=True)
	if "pred_dt" in skip_channels:
		print("[fit] skipping optional channel 'pred_dt': pred_dt loss is disabled", flush=True)

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)

	# Probe preprocessed data for scaledown and volume extent (in base/VC3D coords)
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input), skip_channels=skip_channels)
	source_to_base = float(prep_params.get("source_to_base", 1.0))
	# Model scaledown in base coords = channel_scaledown * source_to_base
	scaledown = float(prep_params["scaledown"]) * source_to_base
	volume_extent_fullres = prep_params.get("volume_extent_fullres")
	print(f"[fit] scaledown={scaledown} (source_sd={prep_params['scaledown']} "
		  f"source_to_base={source_to_base}) volume_extent={volume_extent_fullres}", flush=True)

	# --- Init from seed (new model only) ---
	is_new_model = model_cfg.model_input is None
	if is_new_model and data_cfg.seed is not None and data_cfg.model_w is not None:
		if model_cfg.init_mode == "straight":
			sp = _straight_params_from_seed(data_cfg.seed, data_cfg.model_w)
			model_cfg = dataclasses.replace(model_cfg, **sp)
			print(f"[fit] straight from seed: cx={sp['straight_cx']:.1f} cy={sp['straight_cy']:.1f} "
				  f"angle={sp['straight_angle']:.3f} half_w={sp['straight_half_w']:.1f} "
				  f"z={sp['z_center']:.1f}", flush=True)
		elif volume_extent_fullres is not None:
			umb_center = None
			if data_cfg.umbilicus:
				umb_center = _umbilicus_center_at_z(
					data_cfg.umbilicus,
					float(data_cfg.seed[2]),
					scale=float(data_cfg.umbilicus_scale),
				)
				print(f"[fit] umbilicus center at z={data_cfg.seed[2]}: "
					  f"cx={umb_center[0]:.1f} cy={umb_center[1]:.1f} "
					  f"path={data_cfg.umbilicus}", flush=True)
			arc = _arc_params_from_seed(
				data_cfg.seed, data_cfg.model_w, volume_extent_fullres,
				umbilicus_center=umb_center,
			)
			model_cfg = dataclasses.replace(model_cfg, **arc)
			print(f"[fit] arc from seed: cx={arc['arc_cx']:.1f} cy={arc['arc_cy']:.1f} "
				  f"r={arc['arc_radius']:.1f} a0={arc['arc_angle0']:.3f} a1={arc['arc_angle1']:.3f} "
				  f"z={arc['z_center']:.1f}", flush=True)

	# --- Size mesh from model_w, model_h, windings (new model only) ---
	if is_new_model and data_cfg.model_w is not None and data_cfg.model_h is not None and data_cfg.windings is not None:
		auto_mesh_w = max(2, int(data_cfg.model_w / model_cfg.mesh_step) + 1)
		auto_mesh_h = max(2, int(data_cfg.model_h / model_cfg.mesh_step) + 1)
		auto_depth = max(1, data_cfg.windings)

		model_cfg = dataclasses.replace(model_cfg, depth=auto_depth, mesh_h=auto_mesh_h, mesh_w=auto_mesh_w)
		print(f"[fit] model size: depth={auto_depth} mesh_h={auto_mesh_h} mesh_w={auto_mesh_w}", flush=True)

	# --- Windowed tifxyz mode ---
	tifxyz_init = getattr(args, "tifxyz_init", None)
	window_size = getattr(args, "window_size", None) or 0
	window_overlap = getattr(args, "window_overlap", 0)
	extend_mode = str(cfg.get("mode", "")).lower() == "extend"
	boundary_anchor_cfg = cfg.get("boundary_anchor")
	boundary_anchor_enabled = isinstance(boundary_anchor_cfg, dict) and bool(boundary_anchor_cfg.get("enabled", False))
	boundary_anchor_ring = int(boundary_anchor_cfg.get("ring_width", 1)) if isinstance(boundary_anchor_cfg, dict) else 1

	if tifxyz_init and window_size > 0:
		from tifxyz_io import load_tifxyz
		import fit2tifxyz as _f2t
		import json as _json

		# Load full tifxyz to CPU (save GPU mem)
		full_xyz, full_valid, full_meta = load_tifxyz(tifxyz_init, device="cpu")
		if extend_mode:
			full_xyz, full_valid = _pad_tifxyz_for_extend(full_xyz, full_valid, cfg)
		full_xyz_np = full_xyz.detach().cpu().numpy()
		full_valid_np = full_valid.detach().cpu().numpy().astype(bool)
		H_full, W_full, _ = full_xyz.shape
		mesh_step = model_cfg.mesh_step
		scale = full_meta.get("scale")
		if scale is not None and isinstance(scale, list) and len(scale) >= 1 and float(scale[0]) > 0:
			mesh_step = max(1, int(round(1.0 / float(scale[0]))))

		# Get offset from external_surfaces config
		ext_surfaces_cfg = cfg.pop("external_surfaces", None)
		offset_val = 1.0
		if isinstance(ext_surfaces_cfg, list) and ext_surfaces_cfg:
			offset_val = float(ext_surfaces_cfg[0].get("offset", 1.0))
		ext_margin = max(4, int(2 * abs(offset_val) / mesh_step) + 2)

		# Parse stages and channel skipping (shared across windows)
		extend_corr_points_obj = cfg.get("corr_points") if extend_mode else None
		if not extend_mode:
			cfg.pop("corr_points", None)

		windows = _compute_window_grid(H_full, W_full, mesh_step, window_size, window_overlap)
		n_windows = len(windows)
		overlap_verts = window_overlap // mesh_step
		opt_halo_verts = max(overlap_verts, ext_margin)
		print(f"[fit] windowed mode: {n_windows} windows, mode={'extend' if extend_mode else 'default'}, window_size={window_size} "
			  f"overlap={window_overlap} mesh_step={mesh_step} opt_halo={opt_halo_verts} grid={H_full}x{W_full}",
			  flush=True)

		# Output directory for window tifxyz exports
		output_dir = model_cfg.model_output
		if output_dir is not None:
			output_dir = str(Path(output_dir).parent)
		elif _out_dir is not None:
			output_dir = _out_dir
		else:
			raise ValueError("windowed mode requires --model-output or --out-dir")
		Path(output_dir).mkdir(parents=True, exist_ok=True)

		voxel_size_um = fit_config.get("voxel_size_um")
		window_output_dirs: list[Path] = []

		for wi, (h0, h1, w0, w1) in enumerate(windows):
			print(f"\n[fit] === window {wi+1}/{n_windows}: rows [{h0}:{h1}], cols [{w0}:{w1}] "
				  f"({h1-h0}x{w1-w0} verts) ===", flush=True)
			window_corr_points_3d = None
			if extend_mode:
				window_corr_obj = _filter_corr_points_for_window(
					extend_corr_points_obj, h0=h0, h1=h1, w0=w0, w1=w1,
					full_xyz_np=full_xyz_np, full_valid_np=full_valid_np)
				if window_corr_obj is not None:
					window_corr_points_3d = _parse_corr_points(window_corr_obj, device)

			# Optimize a halo around the intended output window. If the
			# halo extends beyond the source surface, keep it invalid
			# instead of duplicating/clamping edge coordinates.
			oh0 = h0 - opt_halo_verts
			oh1 = h1 + opt_halo_verts
			ow0 = w0 - opt_halo_verts
			ow1 = w1 + opt_halo_verts
			crop_xyz_cpu, crop_valid_cpu = _slice_tifxyz_with_invalid_padding(
				full_xyz, full_valid, h0=oh0, h1=oh1, w0=ow0, w1=ow1)
			crop_xyz = crop_xyz_cpu.to(device)
			crop_valid = crop_valid_cpu.to(device)

			# Create model from crop
			mdl = model.Model3D.from_tifxyz_crop(
				crop_xyz, crop_valid, device=device, mesh_step=mesh_step,
				winding_step=model_cfg.winding_step,
				subsample_mesh=model_cfg.subsample_mesh,
				subsample_winding=model_cfg.subsample_winding,
			)
			if boundary_anchor_enabled:
				anchor_idx = mdl.add_boundary_anchor_surface(
					crop_xyz, crop_valid, ring_width=boundary_anchor_ring)
				print(f"[fit] boundary anchor {anchor_idx}: ring_width={boundary_anchor_ring} "
					  f"anchors={int(mdl._boundary_anchors[anchor_idx][1].sum())}", flush=True)

			# External reference surface gets its own halo. Out-of-bounds
			# regions remain invalid rather than clamped to the source edge.
			eh0 = oh0 - ext_margin
			eh1 = oh1 + ext_margin
			ew0 = ow0 - ext_margin
			ew1 = ow1 + ext_margin
			ext_xyz_cpu, ext_valid_cpu = _slice_tifxyz_with_invalid_padding(
				full_xyz, full_valid, h0=eh0, h1=eh1, w0=ew0, w1=ew1)
			ext_xyz = ext_xyz_cpu.to(device)
			ext_valid = ext_valid_cpu.to(device)
			ext_idx = mdl.add_external_surface(ext_xyz, valid=ext_valid, offset=offset_val)
			# ext->model mapping: ext corner r -> model grid r + h_off
			# ext grid 0 = fullres eh0, model grid 0 = fullres oh0
			# so model_h = r + (eh0 - oh0)
			mdl._ext_conn_offsets[ext_idx][0] = float(eh0 - oh0)
			mdl._ext_conn_offsets[ext_idx][1] = float(ew0 - ow0)

			# CUDA uses sparse streaming; CPU uses the existing dense cropped
			# loader because SparseChunkGroupCache stores CUDA pointers.
			def _load_data_win() -> fit_data.FitData3D:
				if device.type != "cuda":
					d = fit_data.load_3d_for_model(
						path=str(data_cfg.input),
						device=device,
						model=mdl,
						erode_valid_mask=data_cfg.erode_valid_mask,
						cuda_gridsample=data_cfg.cuda_gridsample,
						skip_channels=skip_channels,
					)
					Z, Y, X = d.size
					sx, sy, sz = d.spacing
					volume_extent = (
						d.origin_fullres[0], d.origin_fullres[1], d.origin_fullres[2],
						d.origin_fullres[0] + (X - 1) * sx,
						d.origin_fullres[1] + (Y - 1) * sy,
						d.origin_fullres[2] + (Z - 1) * sz,
					)
					mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
					if window_corr_points_3d is not None:
						d = dataclasses.replace(d, corr_points=window_corr_points_3d)
					return d

				d = fit_data.load_3d_streaming(
					path=str(data_cfg.input),
					device=device,
					skip_channels=skip_channels,
				)
				Z, Y, X = d.size
				sx, sy, sz = d.spacing
				volume_extent = (
					d.origin_fullres[0], d.origin_fullres[1], d.origin_fullres[2],
					d.origin_fullres[0] + (X - 1) * sx,
					d.origin_fullres[1] + (Y - 1) * sy,
					d.origin_fullres[2] + (Z - 1) * sz,
				)
				mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
				if window_corr_points_3d is not None:
					d = dataclasses.replace(d, corr_points=window_corr_points_3d)
				return d

			def _ensure_data_win(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
				if data is None:
					return _load_data_win()
				return data

			data = _ensure_data_win(None, set())

			# Progress wrapper: prefix window index, scale overall progress
			def _make_progress(wi_=wi, n_=n_windows):
				def _progress_win(*, step: int, total: int, loss: float, **kw: object) -> None:
					if progress_enabled:
						inner = float(kw.get("overall_progress", 0.0))
						overall = (wi_ + inner) / n_
						stage_name = kw.get("stage_name", "")
						print(f"PROGRESS {step} {total} {loss:.6f} win={wi_+1}/{n_} "
							  f"overall={overall:.3f} {stage_name}", flush=True)
				return _progress_win

			opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

			# Seed from center of model grid (matches h_mid/w_mid in station loss)
			center_pt = _grid_center(mdl).detach()
			win_seed = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
			print(f"[fit] window seed: ({win_seed[0]:.0f}, {win_seed[1]:.0f}, {win_seed[2]:.0f})",
				  flush=True)

			optimizer.optimize(
				model=mdl,
				data=data,
				stages=stages,
				snapshot_interval=0,
				snapshot_fn=lambda **kw: None,
				progress_fn=_make_progress(),
				ensure_data_fn=_ensure_data_win,
				seed_xyz=win_seed,
			)

			# Export this window's tifxyz
			mesh = mdl.mesh_coarse()  # (3, 1, Hm, Wm)
			mesh_np = mesh.detach().cpu().numpy()
			Hm, Wm = mesh_np.shape[2], mesh_np.shape[3]
			core_h0 = h0 - oh0
			core_h1 = core_h0 + (h1 - h0)
			core_w0 = w0 - ow0
			core_w1 = core_w0 + (w1 - w0)
			x_out = mesh_np[0, 0, core_h0:core_h1, core_w0:core_w1]
			y_out = mesh_np[1, 0, core_h0:core_h1, core_w0:core_w1]
			z_out = mesh_np[2, 0, core_h0:core_h1, core_w0:core_w1]
			if output_scale != 1.0:
				x_out = x_out * output_scale
				y_out = y_out * output_scale
				z_out = z_out * output_scale
			mesh_step_output = float(mesh_step) * output_scale
			meta_scale = 1.0 / mesh_step_output

			win_name = f"window_{wi:04d}.tifxyz"
			win_dir = Path(output_dir) / win_name
			window_output_dirs.append(win_dir)
			area = _f2t._get_area(x_out, y_out, z_out, mesh_step_output,
								  float(voxel_size_um) if voxel_size_um else None)
			_f2t._write_tifxyz(
				out_dir=win_dir, x=x_out, y=y_out, z=z_out,
				scale=meta_scale, area=area,
			)
			# Add window metadata to meta.json
			meta_path = win_dir / "meta.json"
			meta = _json.loads(meta_path.read_text(encoding="utf-8"))
			meta["window_index"] = wi
			meta["window_origin_verts"] = [h0, w0]
			meta["window_size_verts"] = [h1 - h0, w1 - w0]
			meta["optimized_origin_verts"] = [oh0, ow0]
			meta["optimized_size_verts"] = [oh1 - oh0, ow1 - ow0]
			meta["source_grid_size_verts"] = [H_full, W_full]
			meta["overlap_verts"] = overlap_verts
			meta["optimization_halo_verts"] = opt_halo_verts
			meta_path.write_text(_json.dumps(meta, indent=2) + "\n", encoding="utf-8")

			_f2t._print_area(area)
			print(f"[fit] exported {win_name}", flush=True)

			# Free GPU memory before next window
			del mdl, data, crop_xyz, crop_valid, crop_xyz_cpu, crop_valid_cpu
			del ext_xyz, ext_valid, ext_xyz_cpu, ext_valid_cpu, mesh, mesh_np
			if device.type == "cuda":
				torch.cuda.empty_cache()

		if extend_mode:
			merge_name = "extend_merged.tifxyz"
			print(f"\n[fit] extend mode: merging {len(window_output_dirs)} windows into {merge_name}",
				  flush=True)
			_merge_window_tifxyz(
				windows=window_output_dirs,
				out_dir=Path(output_dir) / merge_name,
				base_xyz=full_xyz,
				base_valid=full_valid,
				scale=1.0 / (float(mesh_step) * output_scale),
				voxel_size_um=float(voxel_size_um) if voxel_size_um else None,
				preserve_existing=True)

		print(f"\n[fit] windowed mode complete: {n_windows} windows exported to {output_dir}",
			  flush=True)
		return 0

	# --- Construct / load model (before data, so we can compute bbox) ---
	if tifxyz_init:
		mdl = model.Model3D.from_tifxyz(
			tifxyz_init, device=device,
			mesh_step=model_cfg.mesh_step,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
		)
		print(f"[fit] initialized from tifxyz: {tifxyz_init}", flush=True)
	elif is_new_model:
		mdl = model.Model3D(
			device=device,
			depth=model_cfg.depth,
			mesh_h=model_cfg.mesh_h,
			mesh_w=model_cfg.mesh_w,
			mesh_step=model_cfg.mesh_step,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
			scaledown=scaledown,
			z_step_eff=int(round(scaledown)),
			z_center=model_cfg.z_center,
			arc_cx=model_cfg.arc_cx,
			arc_cy=model_cfg.arc_cy,
			arc_radius=model_cfg.arc_radius,
			arc_angle0=model_cfg.arc_angle0,
			arc_angle1=model_cfg.arc_angle1,
			straight_cx=model_cfg.straight_cx,
			straight_cy=model_cfg.straight_cy,
			straight_angle=model_cfg.straight_angle,
			straight_half_w=model_cfg.straight_half_w,
			init_mode=model_cfg.init_mode,
			volume_extent=None,
			pyramid_d=model_cfg.pyramid_d,
		)
	else:
		st = torch.load(model_cfg.model_input, map_location=device, weights_only=False)
		mdl = model.Model3D.from_checkpoint(st, device=device)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"arc_enabled={mdl.arc_enabled}")

	# Load external reference surfaces
	ext_surfaces_cfg = cfg.pop("external_surfaces", None)
	if isinstance(ext_surfaces_cfg, list) and ext_surfaces_cfg:
		from tifxyz_io import load_tifxyz
		for es in ext_surfaces_cfg:
			es_path = str(es["path"])
			es_offset = float(es.get("offset", 1.0))
			xyz_ext, valid_ext, meta_ext = load_tifxyz(es_path, device=device)
			idx = mdl.add_external_surface(xyz_ext, valid=valid_ext, offset=es_offset)
			print(f"[fit] external surface {idx}: path={es_path} offset={es_offset} "
				  f"shape={tuple(xyz_ext.shape)} valid={int(valid_ext.sum())}/{valid_ext.numel()}", flush=True)
			if boundary_anchor_enabled and abs(es_offset) <= 1e-9:
				anchor_idx = mdl.add_boundary_anchor_surface(
					xyz_ext, valid_ext, ring_width=boundary_anchor_ring)
				print(f"[fit] boundary anchor {anchor_idx}: ring_width={boundary_anchor_ring} "
					  f"anchors={int(mdl._boundary_anchors[anchor_idx][1].sum())}", flush=True)

	# Parse correction points from config (injected by VC3D)
	corr_points_obj = cfg.pop("corr_points", None)
	corr_points_3d: fit_data.CorrPoints3D | None = None
	if isinstance(corr_points_obj, dict):
		corr_points_3d = _parse_corr_points(corr_points_obj, device)
	else:
		print(f"[fit] corr_points: not found in config (type={type(corr_points_obj).__name__})", flush=True)

	# CUDA uses sparse streaming; CPU uses the existing dense cropped loader
	# because SparseChunkGroupCache stores CUDA pointers.
	def _load_data() -> fit_data.FitData3D:
		if device.type != "cuda":
			d = fit_data.load_3d_for_model(
				path=str(data_cfg.input),
				device=device,
				model=mdl,
				erode_valid_mask=data_cfg.erode_valid_mask,
				cuda_gridsample=data_cfg.cuda_gridsample,
				skip_channels=skip_channels,
			)
			Z, Y, X = d.size
			sx, sy, sz = d.spacing
			volume_extent = (
				d.origin_fullres[0],
				d.origin_fullres[1],
				d.origin_fullres[2],
				d.origin_fullres[0] + (X - 1) * sx,
				d.origin_fullres[1] + (Y - 1) * sy,
				d.origin_fullres[2] + (Z - 1) * sz,
			)
			mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
			if corr_points_3d is not None:
				d = dataclasses.replace(d, corr_points=corr_points_3d)
			if data_cfg.winding_volume is not None:
				wv_t, wv_min, wv_max = fit_data.load_winding_volume(
					path=data_cfg.winding_volume, device=device,
					crop=None, downscale=scaledown)
				d = dataclasses.replace(d, winding_volume=wv_t,
							winding_min=wv_min, winding_max=wv_max)
			return d

		d = fit_data.load_3d_streaming(
			path=str(data_cfg.input),
			device=device,
			skip_channels=skip_channels,
		)
		Z, Y, X = d.size
		# Volume extent covers the full zarr volume
		sx, sy, sz = d.spacing
		volume_extent = (
			d.origin_fullres[0],
			d.origin_fullres[1],
			d.origin_fullres[2],
			d.origin_fullres[0] + (X - 1) * sx,
			d.origin_fullres[1] + (Y - 1) * sy,
			d.origin_fullres[2] + (Z - 1) * sz,
		)
		mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
		if corr_points_3d is not None:
			d = dataclasses.replace(d, corr_points=corr_points_3d)
		if data_cfg.winding_volume is not None:
			wv_t, wv_min, wv_max = fit_data.load_winding_volume(
				path=data_cfg.winding_volume, device=device,
				crop=None, downscale=scaledown)
			d = dataclasses.replace(d, winding_volume=wv_t,
						winding_min=wv_min, winding_max=wv_max)
		return d

	def _ensure_data(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
		if data is None:
			return _load_data()
		# Streaming covers full volume — no border checks or channel loading needed
		return data

	data = _ensure_data(None, set())

	# Print loaded data summary
	Z, Y, X = data.size
	if data.sparse_caches:
		_cache_table_bytes = sum(c.chunk_table.nbytes for c in data.sparse_caches.values())
		print(f"[fit] data (streaming): vol_size=({Z},{Y},{X}) origin={data.origin_fullres} "
			  f"spacing={data.spacing} groups={list(data.sparse_caches.keys())} "
			  f"table_mem={_cache_table_bytes / 2**20:.1f} MiB", flush=True)
	else:
		_data_bytes = sum(t.nbytes for t in [data.cos, data.grad_mag, data.nx, data.ny] if t is not None)
		if data.pred_dt is not None:
			_data_bytes += data.pred_dt.nbytes
		if data.winding_volume is not None:
			_data_bytes += data.winding_volume.nbytes
		print(f"[fit] data: size=({Z},{Y},{X}) origin={data.origin_fullres} spacing={data.spacing} "
			  f"pred_dt={data.pred_dt is not None} winding_volume={data.winding_volume is not None} "
			  f"corr_points={data.corr_points is not None} "
			  f"mem={_data_bytes / 2**30:.2f} GiB", flush=True)

	# Print initial mesh stats
	with torch.no_grad():
		xyz = mdl._grid_xyz()
		mn = xyz.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
		print(f"initial mesh: mean={[round(v, 1) for v in mean]} "
			  f"min={[round(v, 1) for v in mn]} max={[round(v, 1) for v in mx]}")

	def _save_model(path: str) -> None:
		if mdl.arc_enabled:
			mdl.bake_arc_into_mesh()
		if mdl.straight_enabled:
			mdl.bake_straight_into_mesh()
		st = dict(mdl.state_dict())
		# Store flat mesh instead of pyramid levels
		ms_keys = [k for k in st if k.startswith("mesh_ms.")]
		for k in ms_keys:
			del st[k]
		with torch.no_grad():
			st["mesh_flat"] = mdl.mesh_coarse().detach().clone()
		st["_model_params_"] = asdict(mdl.params)
		st["_fit_config_"] = fit_config
		corr_results = opt_loss_corr.get_last_results()
		if corr_results is not None:
			st["_corr_points_results_"] = corr_results
		# Store winding volume auto-offset if computed
		from opt_loss_winding_volume import _winding_offset, _winding_direction
		if _winding_offset is not None:
			st["_winding_offset_"] = _winding_offset
			st["_winding_direction_"] = _winding_direction
		torch.save(st, path)

	def _snapshot(*, stage: str, step: int, loss: float, data, res=None) -> None:
		if _out_dir is not None:
			out = Path(_out_dir)
			out.mkdir(parents=True, exist_ok=True)
			snaps = out / "model_snapshots"
			snaps.mkdir(parents=True, exist_ok=True)
			_save_model(str(snaps / f"model_{stage}_{step:06d}.pt"))

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

	# Run optimization
	seed_xyz = tuple(float(v) for v in data_cfg.seed) if data_cfg.seed is not None else None
	# tifxyz init: always use model grid center as seed (overrides CLI seed)
	if tifxyz_init:
		center_pt = _grid_center(mdl)
		seed_xyz = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
		print(f"[fit] tifxyz seed: ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	# Re-optimize from checkpoint: derive seed from model grid center
	if seed_xyz is None and not is_new_model:
		center_pt = _grid_center(mdl)
		seed_xyz = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
		print(f"[fit] checkpoint seed (grid center): ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		ensure_data_fn=_ensure_data,
		seed_xyz=seed_xyz,
	)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

	# Save final model
	if model_cfg.model_output is not None:
		_save_model(str(model_cfg.model_output))
		print(f"[fit] saved model to {model_cfg.model_output}")

	# Save snapshot
	if _out_dir is not None:
		out = Path(_out_dir)
		out.mkdir(parents=True, exist_ok=True)
		_save_model(str(out / "model_final.pt"))

	# Export tifxyz
	model_out = model_cfg.model_output
	if model_out is None and _out_dir is not None:
		model_out = str(Path(_out_dir) / "model_final.pt")
	if model_out is not None and _out_dir is not None:
		import fit2tifxyz
		export_dir = str(Path(_out_dir) / "tifxyz")
		tifxyz_argv = ["--input", str(model_out), "--output", export_dir]
		voxel_size_um = cfg.get("voxel_size_um")
		if voxel_size_um is not None:
			tifxyz_argv += ["--voxel-size-um", str(float(voxel_size_um))]
		fit2tifxyz.main(tifxyz_argv)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
