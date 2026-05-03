from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dense_batch_flow
import model as fit_model
from opt_loss_dir import _vertex_normals
from opt_loss_station import _intersect_single_quad

_INNER_FACTOR = 0.25  # penalty reduction for points inside the predicted surface
_FLOW_GATE_THRESHOLD = 100.0

_flow_gate_cfg: dict | None = None
_flow_gate_stage: str = "stage"
_flow_gate_seed_xyz: tuple[float, float, float] | None = None
_flow_gate_out_dir: Path | None = None
_flow_gate_debug_counts: dict[str, int] = {}
_flow_gate_last_stats: dict[str, float] = {}
_flow_gate_seed_hw_cache: tuple[int, int, float, float] | None = None
_flow_gate_jpg_warned: bool = False


def _seed_gt_normal(*, seed_xyz: torch.Tensor, res: fit_model.FitResult3D) -> torch.Tensor:
	query = seed_xyz.view(1, 1, 1, 3)
	sampled = res.data.grid_sample_fullres(query)
	nx = sampled.nx.squeeze()
	ny = sampled.ny.squeeze()
	nz = (1.0 - nx * nx - ny * ny).clamp(min=0.0).sqrt()
	n = torch.stack([nx, ny, nz]).to(device=seed_xyz.device, dtype=seed_xyz.dtype)
	return n / n.norm().clamp(min=1e-8)


def _seed_surface_intersection_xy(
	*,
	xyz_img: torch.Tensor,
	seed_xyz: torch.Tensor,
	n_gt: torch.Tensor,
) -> tuple[float, float] | None:
	"""Intersect seed + t*n_gt with the current 2D rendered surface.

	Returns image-space (x, y), using the high-resolution model render.
	"""
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return None
	H, W, _ = xyz_img.shape
	if H < 2 or W < 2:
		return None

	best: tuple[float, float, float] | None = None
	best_score = float("inf")
	for y in range(H - 1):
		for x in range(W - 1):
			P00 = xyz_img[y, x]
			P10 = xyz_img[y + 1, x]
			P01 = xyz_img[y, x + 1]
			P11 = xyz_img[y + 1, x + 1]
			try:
				u, v, p = _intersect_single_quad(
					seed_xyz, n_gt, P00, P10, P01, P11, 0.5, 0.5)
			except Exception:
				continue
			if u < -1.0e-4 or u > 1.0 + 1.0e-4 or v < -1.0e-4 or v > 1.0 + 1.0e-4:
				continue
			delta = p - seed_xyz
			t = torch.dot(delta, n_gt).item()
			residual = (delta - t * n_gt).norm().item()
			score = residual * 1000.0 + abs(t)
			if score < best_score:
				best_score = score
				best = (float(x) + float(v), float(y) + float(u), t)
	if best is None:
		return None
	return best[0], best[1]


def _seed_surface_intersection_xy_from_cache(
	*,
	xyz_img: torch.Tensor,
	seed_xyz: torch.Tensor,
	n_gt: torch.Tensor,
	h_frac: float,
	w_frac: float,
) -> tuple[float, float] | None:
	"""One station-style cached ray/surface update on the high-res render grid."""
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return None
	H, W, _ = xyz_img.shape
	if H < 2 or W < 2:
		return None
	if not (np.isfinite(h_frac) and np.isfinite(w_frac)):
		return None

	row = max(0, min(int(h_frac), H - 2))
	col = max(0, min(int(w_frac), W - 2))
	frac_h = h_frac - float(row)
	frac_w = w_frac - float(col)

	try:
		u1, v1, _ = _intersect_single_quad(
			seed_xyz, n_gt,
			xyz_img[row, col],
			xyz_img[row + 1, col],
			xyz_img[row, col + 1],
			xyz_img[row + 1, col + 1],
			frac_h, frac_w,
		)
	except Exception:
		return None
	if not (np.isfinite(u1) and np.isfinite(v1)):
		return None

	new_h = max(0.0, min(float(H - 2), float(row) + float(u1)))
	new_w = max(0.0, min(float(W - 2), float(col) + float(v1)))
	new_row = max(0, min(int(new_h), H - 2))
	new_col = max(0, min(int(new_w), W - 2))
	new_frac_h = new_h - float(new_row)
	new_frac_w = new_w - float(new_col)

	try:
		u2, v2, _ = _intersect_single_quad(
			seed_xyz, n_gt,
			xyz_img[new_row, new_col],
			xyz_img[new_row + 1, new_col],
			xyz_img[new_row, new_col + 1],
			xyz_img[new_row + 1, new_col + 1],
			new_frac_h, new_frac_w,
		)
	except Exception:
		return None
	if not (np.isfinite(u2) and np.isfinite(v2)):
		return None
	if u2 < 0.0 or u2 > 1.0 or v2 < 0.0 or v2 > 1.0:
		return None

	return float(new_col) + float(v2), float(new_row) + float(u2)


def _dilate_binary_3x3(mask: np.ndarray, iterations: int) -> np.ndarray:
	out = mask.astype(bool, copy=False)
	for _ in range(max(0, int(iterations))):
		p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
		out = (
			p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
			p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
			p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
		)
	return out


def _erode_binary_3x3(mask: np.ndarray, iterations: int) -> np.ndarray:
	out = mask.astype(bool, copy=False)
	for _ in range(max(0, int(iterations))):
		p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
		out = (
			p[:-2, :-2] & p[:-2, 1:-1] & p[:-2, 2:] &
			p[1:-1, :-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
			p[2:, :-2] & p[2:, 1:-1] & p[2:, 2:]
		)
	return out


def _flow_gate_white_domain_u8(pred_u8: np.ndarray, threshold: float) -> np.ndarray:
	white_domain = pred_u8 > threshold
	white_domain = _dilate_binary_3x3(white_domain, 2)
	white_domain = _erode_binary_3x3(white_domain, 2)
	return white_domain.astype(np.uint8) * 255


def _flow_gate_seed_dt(pred_u8: np.ndarray, threshold: float, source_x: int, source_y: int) -> float:
	if source_x < 0 or source_x >= pred_u8.shape[1] or source_y < 0 or source_y >= pred_u8.shape[0]:
		return 0.0
	white_domain = _flow_gate_white_domain_u8(pred_u8, threshold)
	try:
		import cv2
		dt = cv2.distanceTransform(white_domain, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
		return float(dt[source_y, source_x])
	except Exception:
		# Fallback is only for environments missing cv2; the C++ path uses OpenCV.
		if white_domain[source_y, source_x] == 0:
			return 0.0
		ys, xs = np.nonzero(white_domain == 0)
		if xs.size == 0:
			return 0.0
		dx = xs.astype(np.float32) - float(source_x)
		dy = ys.astype(np.float32) - float(source_y)
		return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _sample_pred_dt_max3d(
	*,
	res: fit_model.FitResult3D,
	xyz_hr: torch.Tensor,
	radius: int,
	step_scale: float,
) -> torch.Tensor:
	"""Pred-dt render for flow gating, max-pooled in 3D around surface samples."""
	r = int(max(0, radius))
	if r <= 0:
		pred_hr = res.data_s.pred_dt.squeeze(0).squeeze(0)
		if pred_hr.ndim != 3 or pred_hr.shape[0] != 1:
			raise RuntimeError(f"pred_dt_flow_gate expected pred_dt render shape (1,H,W), got {tuple(pred_hr.shape)}")
		return pred_hr[0]

	offsets = torch.tensor(
		[(dx, dy, dz)
		 for dz in range(-r, r + 1)
		 for dy in range(-r, r + 1)
		 for dx in range(-r, r + 1)],
		device=xyz_hr.device,
		dtype=xyz_hr.dtype,
	)
	spacing = torch.tensor(
		res.data._spacing_for("pred_dt"),
		device=xyz_hr.device,
		dtype=xyz_hr.dtype,
	)
	offsets = offsets * spacing.view(1, 3) * float(step_scale)
	query = xyz_hr.unsqueeze(0) + offsets.view(-1, 1, 1, 3)
	sampled = res.data.grid_sample_fullres(query).pred_dt
	if sampled is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")
	pred = sampled.squeeze(0).squeeze(0)
	if pred.ndim != 3:
		raise RuntimeError(f"pred_dt_flow_gate expected pooled pred_dt shape (N,H,W), got {tuple(pred.shape)}")
	return pred.amax(dim=0)


def configure_flow_gate(
	*,
	cfg: dict | None,
	stage_name: str,
	seed_xyz: tuple[float, float, float] | None,
	out_dir: str | None,
) -> None:
	global _flow_gate_cfg, _flow_gate_stage, _flow_gate_seed_xyz, _flow_gate_out_dir, _flow_gate_debug_counts, _flow_gate_last_stats, _flow_gate_seed_hw_cache
	_flow_gate_last_stats = {}
	_flow_gate_seed_hw_cache = None
	if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
		_flow_gate_cfg = None
		_flow_gate_stage = str(stage_name)
		_flow_gate_seed_xyz = seed_xyz
		_flow_gate_out_dir = Path(out_dir) if out_dir else None
		return
	_flow_gate_cfg = dict(cfg)
	_flow_gate_stage = str(stage_name)
	_flow_gate_seed_xyz = seed_xyz
	debug_out_dir = _flow_gate_cfg.get("debug_out_dir", None)
	_flow_gate_out_dir = Path(debug_out_dir) if debug_out_dir else (Path(out_dir) if out_dir else None)
	_flow_gate_debug_counts[str(stage_name)] = 0


def flow_gate_last_stats() -> dict[str, float]:
	return dict(_flow_gate_last_stats)


def _write_flow_gate_debug(
	*,
	stage_name: str,
	debug_index: int,
	pred_u8: np.ndarray,
	flow_hr: np.ndarray | None,
	smooth_grid_flow: np.ndarray | None,
	graph_edge_flow_rgb: np.ndarray | None,
	weight_hr: np.ndarray | None,
	out_dir: Path | None,
	threshold: float,
	source_xy: tuple[int, int] | None = None,
) -> None:
	if out_dir is None:
		return
	try:
		import tifffile
	except Exception as exc:
		print(f"[pred_dt_flow_gate] debug write skipped: tifffile import failed: {exc}", flush=True)
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	# Match dense_batch_min_cut's fixed binarization white distance domain.
	thresholded_u8 = _flow_gate_white_domain_u8(pred_u8, threshold)
	pred_raw_u8 = pred_u8.astype(np.uint8, copy=True)
	flow = np.zeros_like(thresholded_u8, dtype=np.float32) if flow_hr is None else np.asarray(flow_hr, dtype=np.float32)
	grid_flow = None if smooth_grid_flow is None else np.asarray(smooth_grid_flow, dtype=np.float32)
	graph_flow = None if graph_edge_flow_rgb is None else np.asarray(graph_edge_flow_rgb, dtype=np.float32)
	weights = np.zeros_like(thresholded_u8, dtype=np.float32) if weight_hr is None else np.asarray(weight_hr, dtype=np.float32)
	def write_named_layer(tw, image: np.ndarray, *, name: str) -> None:
		arr = np.asarray(image)
		if arr.ndim == 2:
			arr = np.repeat(arr.astype(np.float32, copy=False)[..., None], 3, axis=2)
		elif arr.ndim == 3 and arr.shape[2] == 3:
			arr = arr.astype(np.float32, copy=False)
		else:
			raise ValueError(f"flow gate debug layer {name!r} has unsupported shape {arr.shape}")
		tw.write(
			arr,
			photometric="rgb",
			metadata=None,
			extratags=[(285, "s", 0, name, False)],  # TIFFTAG_PAGENAME
		)
	for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
		with tifffile.TiffWriter(str(out_dir / f"pred_dt_flow_gate_{suffix}_layers.tif")) as tw:
			write_named_layer(tw, pred_raw_u8, name="pred_dt")
			write_named_layer(tw, thresholded_u8, name="thresholded_pred_dt")
			write_named_layer(tw, flow, name="raw_flow_bilinear")
			if grid_flow is not None:
				write_named_layer(tw, grid_flow, name="smooth_grid_flow")
			if graph_flow is not None:
				write_named_layer(tw, graph_flow, name="graph_edge_flow")
			write_named_layer(tw, weights, name="flow_gate_weight")
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_pred_dt.tif"), pred_raw_u8)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_thresholded.tif"), thresholded_u8)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_raw_flow.tif"), flow)
		if grid_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_smooth_grid_flow.tif"), grid_flow)
		if graph_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_graph_edge_flow.tif"), graph_flow)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_weight.tif"), weights)


def _write_flow_gate_weight_jpg(
	*,
	stage_name: str,
	debug_index: int,
	used_weight_lr: torch.Tensor,
	out_dir: Path | None,
) -> None:
	global _flow_gate_jpg_warned
	if out_dir is None:
		return
	arr = used_weight_lr.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
	if arr.ndim != 2:
		return
	img = (arr * 255.0 + 0.5).astype(np.uint8)
	jpg_dir = out_dir / "pred_dt_flow_gate_weight_jpg"
	try:
		jpg_dir.mkdir(parents=True, exist_ok=True)
		for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
			path = jpg_dir / f"{suffix}_used_weight.jpg"
			try:
				import cv2
				cv2.imwrite(str(path), img)
			except Exception:
				from PIL import Image
				Image.fromarray(img, mode="L").save(str(path), quality=95)
	except Exception as exc:
		if not _flow_gate_jpg_warned:
			print(f"[pred_dt_flow_gate] jpg weight write skipped: {exc}", flush=True)
			_flow_gate_jpg_warned = True


def _flow_gate_weight(res: fit_model.FitResult3D) -> torch.Tensor | None:
	global _flow_gate_last_stats, _flow_gate_seed_hw_cache
	_flow_gate_last_stats = {}
	cfg = _flow_gate_cfg
	if cfg is None:
		return None
	if res.xyz_lr.shape[0] != 1:
		raise RuntimeError("pred_dt_flow_gate currently supports only single-winding models (D == 1)")
	if _flow_gate_seed_xyz is None:
		raise RuntimeError("pred_dt_flow_gate requires a fit seed so it can project the flow source")
	if res.data_s.pred_dt is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")

	threshold = _FLOW_GATE_THRESHOLD
	flow_zero = float(cfg.get("flow_zero", 20.0))
	flow_one = float(cfg.get("flow_one", 100.0))
	backtrack_distance = float(cfg.get("backtrack_distance", 10.0))
	pred_dt_pool_radius = int(cfg.get("pred_dt_pool_radius", 1))
	pred_dt_pool_step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
	if flow_one <= flow_zero:
		raise ValueError("pred_dt_flow_gate requires flow_one > flow_zero")
	debug = bool(cfg.get("debug", True))
	debug_index = 0
	if debug:
		debug_index = _flow_gate_debug_counts.get(_flow_gate_stage, 0)
		_flow_gate_debug_counts[_flow_gate_stage] = debug_index + 1
	write_layer_debug = debug and (debug_index % 10) == 0

	with torch.no_grad():
		xyz_hr = res.xyz_hr[0].detach()
		pred_hr = _sample_pred_dt_max3d(
			res=res,
			xyz_hr=xyz_hr,
			radius=pred_dt_pool_radius,
			step_scale=pred_dt_pool_step_scale,
		)
		pred_img = pred_hr.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()
		He, We = pred_img.shape

		seed = torch.tensor(_flow_gate_seed_xyz, device=xyz_hr.device, dtype=xyz_hr.dtype)
		n_gt = _seed_gt_normal(seed_xyz=seed, res=res)
		source_xy = None
		if _flow_gate_seed_hw_cache is not None:
			cache_h, cache_w, cache_y, cache_x = _flow_gate_seed_hw_cache
			if cache_h == He and cache_w == We:
				source_xy = _seed_surface_intersection_xy_from_cache(
					xyz_img=xyz_hr,
					seed_xyz=seed,
					n_gt=n_gt,
					h_frac=cache_y,
					w_frac=cache_x,
				)
		if source_xy is None:
			source_xy = _seed_surface_intersection_xy(
				xyz_img=xyz_hr,
				seed_xyz=seed,
				n_gt=n_gt,
			)
		if source_xy is None:
			dist2 = ((xyz_hr - seed.view(1, 1, 3)) ** 2).sum(dim=-1)
			source_flat = int(torch.argmin(dist2).detach().cpu())
			source_y = source_flat // We
			source_x = source_flat % We
			_flow_gate_seed_hw_cache = (He, We, float(source_y), float(source_x))
		else:
			_flow_gate_seed_hw_cache = (He, We, float(source_xy[1]), float(source_xy[0]))
			source_x = int(round(source_xy[0]))
			source_y = int(round(source_xy[1]))
			source_x = max(0, min(We - 1, source_x))
			source_y = max(0, min(He - 1, source_y))

		Hm = int(res.xyz_lr.shape[1])
		Wm = int(res.xyz_lr.shape[2])
		sub_h = int(res.params.subsample_mesh)
		sub_w = int(res.params.subsample_winding)
		grid_step = max(1, int(round(0.5 * (sub_h + sub_w))))
		yy, xx = np.meshgrid(
			np.arange(Hm, dtype=np.float32) * float(sub_h),
			np.arange(Wm, dtype=np.float32) * float(sub_w),
			indexing="ij",
		)
		query_xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
		grid_y, grid_x = torch.meshgrid(
			torch.arange(Hm, device=res.xyz_lr.device, dtype=torch.float32),
			torch.arange(Wm, device=res.xyz_lr.device, dtype=torch.float32),
			indexing="ij",
		)
		source_grid_y = float(source_y) / float(max(1, sub_h))
		source_grid_x = float(source_x) / float(max(1, sub_w))
		seed_area = (
			(grid_y - source_grid_y).square() + (grid_x - source_grid_x).square()
		).le(4.0).view(1, 1, Hm, Wm)

		if write_layer_debug:
			_write_flow_gate_debug(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				pred_u8=pred_img,
				flow_hr=None,
				smooth_grid_flow=None,
				graph_edge_flow_rgb=None,
				weight_hr=None,
				out_dir=_flow_gate_out_dir,
				threshold=threshold,
				source_xy=(source_x, source_y),
			)
		try:
			flow_outputs = dense_batch_flow.compute_flow_grid(
				pred_img,
				source_xy=(source_x, source_y),
				query_xy=query_xy,
				verbose=False,
				return_debug=write_layer_debug,
				grid_step=grid_step,
				backtrack_distance=backtrack_distance,
			)
			if write_layer_debug:
				query_flow, dense_flow, smooth_grid_flow, graph_edge_flow_rgb = flow_outputs
			else:
				query_flow, dense_flow = flow_outputs
				smooth_grid_flow = None
				graph_edge_flow_rgb = None
		except RuntimeError as exc:
			white_domain = _flow_gate_white_domain_u8(pred_img, threshold)
			source_value = int(pred_img[source_y, source_x]) if 0 <= source_y < He and 0 <= source_x < We else -1
			source_in_domain = bool(
				0 <= source_y < He and 0 <= source_x < We and
				white_domain[source_y, source_x] > 0
			)
			domain_pixels = int(np.count_nonzero(white_domain))
			if not source_in_domain:
				weight = seed_area.to(dtype=torch.float32)
				valid = res.mask_lr > 0.0
				valid_count = max(1.0, float(valid.sum().detach().cpu()))
				used_weight = weight * res.mask_lr
				_flow_gate_last_stats = {
					"pred_dt_gate_gt0": float(((weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_eq1": float(((weight >= 1.0) & valid).sum().detach().cpu()) / valid_count,
				}
				print(
					f"[pred_dt_flow_gate] {_flow_gate_stage}: skipped flow "
					f"(source outside white domain, value={source_value}, "
					f"domain={domain_pixels}/{We * He})",
					flush=True,
				)
				if write_layer_debug:
					weight_hr = F.interpolate(
						weight,
						size=(He, We),
						mode="bilinear",
						align_corners=True,
					)[0, 0].detach().cpu().numpy().astype(np.float32)
					_write_flow_gate_debug(
						stage_name=_flow_gate_stage,
						debug_index=debug_index,
						pred_u8=pred_img,
						flow_hr=None,
						smooth_grid_flow=None,
						graph_edge_flow_rgb=None,
						weight_hr=weight_hr,
						out_dir=_flow_gate_out_dir,
						threshold=threshold,
						source_xy=(source_x, source_y),
					)
				if debug:
					_write_flow_gate_weight_jpg(
						stage_name=_flow_gate_stage,
						debug_index=debug_index,
						used_weight_lr=used_weight,
						out_dir=_flow_gate_out_dir,
					)
				return weight
			raise RuntimeError(
				f"{exc}; pred_dt source_value={source_value} "
				f"source_in_flow_domain={source_in_domain} "
				f"domain_pixels={domain_pixels}/{We * He}; "
				f"wrote threshold debug to {_flow_gate_out_dir}"
			) from exc
		flow_lr = torch.as_tensor(
			query_flow.reshape(Hm, Wm),
			device=res.xyz_lr.device,
			dtype=torch.float32,
		).view(1, 1, Hm, Wm)
		seed_dt = _flow_gate_seed_dt(pred_img, threshold, source_x, source_y)
		effective_flow_one = flow_one
		effective_flow_zero = flow_zero
		if seed_dt > 0.0 and seed_dt < flow_one:
			scale = seed_dt / flow_one
			effective_flow_one = seed_dt
			effective_flow_zero = flow_zero * scale
		if effective_flow_one <= effective_flow_zero:
			effective_flow_zero = max(0.0, effective_flow_one - 1.0)
		weight = ((flow_lr - effective_flow_zero) / (effective_flow_one - effective_flow_zero)).clamp(0.0, 1.0)
		weight = torch.where(seed_area, torch.ones_like(weight), weight)
		valid = res.mask_lr > 0.0
		valid_count = max(1.0, float(valid.sum().detach().cpu()))
		used_weight = weight * res.mask_lr
		_flow_gate_last_stats = {
			"pred_dt_gate_gt0": float(((weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_eq1": float(((weight >= 1.0) & valid).sum().detach().cpu()) / valid_count,
		}

		if write_layer_debug:
			flow_hr = F.interpolate(
				flow_lr,
				size=(He, We),
				mode="bilinear",
				align_corners=True,
			)[0, 0].detach().cpu().numpy().astype(np.float32)
			weight_hr = F.interpolate(
				weight,
				size=(He, We),
				mode="bilinear",
				align_corners=True,
			)[0, 0].detach().cpu().numpy().astype(np.float32)
			_write_flow_gate_debug(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				pred_u8=pred_img,
				flow_hr=flow_hr,
				smooth_grid_flow=smooth_grid_flow,
				graph_edge_flow_rgb=graph_edge_flow_rgb,
				weight_hr=weight_hr,
				out_dir=_flow_gate_out_dir,
				threshold=threshold,
				source_xy=(source_x, source_y),
			)
		if debug:
			_write_flow_gate_weight_jpg(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				used_weight_lr=used_weight,
				out_dir=_flow_gate_out_dir,
			)
		return weight


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: two clamped L1 terms pushing mesh into the prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)      — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	# Project gradients onto surface normal only (prevents tangential crimping)
	n = _vertex_normals(res.xyz_lr.detach())
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	xyz = xyz_normal + xyz_tangential.detach()

	# Sample pred_dt using common sampling with per-channel spacing and diff gradients
	sampled = res.data.grid_sample_fullres(xyz, diff=True)
	sampled_raw = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	lm_out = (127.0 - sampled_raw).clamp(min=0)      # outside: 1–47, inside: 0 (no grad)
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = lm_out + _INNER_FACTOR * lm_in

	mask = res.mask_lr
	flow_gate = _flow_gate_weight(res)
	wsum = mask.sum()
	if float(wsum) > 0.0:
		if flow_gate is not None:
			loss = (lm * mask * flow_gate).sum() / wsum
		else:
			loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	if flow_gate is not None:
		return loss, (lm,), (mask * flow_gate,)
	return loss, (lm,), (mask,)
