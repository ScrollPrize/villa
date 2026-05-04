from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import dense_batch_flow
import model as fit_model
from opt_loss_dir import _vertex_normals
from opt_loss_station import _intersect_single_quad

_INNER_FACTOR = 0.0  # penalty reduction for points inside the predicted surface
_flow_gate_cfg: dict | None = None
_flow_gate_stage: str = "stage"
_flow_gate_seed_xyz: tuple[float, float, float] | None = None
_flow_gate_out_dir: Path | None = None
_flow_gate_debug_counts: dict[str, int] = {}
_flow_gate_last_stats: dict[str, float] = {}
_flow_gate_seed_hw_cache: tuple[int, int, float, float] | None = None
_flow_gate_jpg_warned: bool = False


def flow_gate_prefetch_points(
	*,
	data,
	xyz_hr: torch.Tensor,
	xyz_lr: torch.Tensor | None = None,
	cfg: dict | None,
) -> torch.Tensor | None:
	"""Extra pred-dt sample positions used by flow-gate render and pull fitting."""
	if cfg is None or not bool(cfg.get("enabled", False)):
		return None
	if xyz_hr.shape[0] != 1:
		return None
	extra: list[torch.Tensor] = []
	r = int(max(0, int(cfg.get("pred_dt_pool_radius", 0))))
	if r > 0:
		step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
		xyz0 = xyz_hr[0].detach()
		offsets = torch.tensor(
			[(dx, dy, dz)
			 for dz in range(-r, r + 1)
			 for dy in range(-r, r + 1)
			 for dx in range(-r, r + 1)],
			device=xyz_hr.device,
			dtype=xyz_hr.dtype,
		)
		spacing = torch.tensor(
			data._spacing_for("pred_dt"),
			device=xyz0.device,
			dtype=xyz0.dtype,
		)
		offsets = offsets * spacing.view(1, 3) * step_scale
		extra.append((xyz0.unsqueeze(0) + offsets.view(-1, 1, 1, 3)).reshape(-1, 3))

	pull_cfg = cfg.get("anticipatory_pull", None)
	if isinstance(pull_cfg, dict) and bool(pull_cfg.get("enabled", False)) and xyz_lr is not None and xyz_lr.shape[0] == 1:
		pull_pf = _anticipatory_pull_prefetch_points(xyz_lr=xyz_lr, cfg=pull_cfg)
		if pull_pf is not None:
			extra.append(pull_pf.reshape(-1, 3))
	if not extra:
		return None
	return torch.cat(extra, dim=0).view(1, 1, -1, 3)


def _seed_gt_normal(*, seed_xyz: torch.Tensor, res: fit_model.FitResult3D) -> torch.Tensor:
	query = seed_xyz.view(1, 1, 1, 3)
	sampled = res.data.grid_sample_fullres(query, channels={"nx", "ny"})
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

	M00 = xyz_img[:-1, :-1]
	M10 = xyz_img[1:, :-1]
	M01 = xyz_img[:-1, 1:]
	M11 = xyz_img[1:, 1:]
	frac_h = torch.tensor(0.5, device=xyz_img.device, dtype=xyz_img.dtype)
	frac_w = torch.tensor(0.5, device=xyz_img.device, dtype=xyz_img.dtype)

	u, v = fit_model.Model3D._ray_bilinear_intersect(
		seed_xyz, n_gt, M00, M10, M01, M11, frac_h, frac_w)
	valid = (
		torch.isfinite(u) & torch.isfinite(v) &
		(u >= -1.0e-4) & (u <= 1.0 + 1.0e-4) &
		(v >= -1.0e-4) & (v <= 1.0 + 1.0e-4)
	)
	p = (
		(1.0 - u).unsqueeze(-1) * (1.0 - v).unsqueeze(-1) * M00 +
		u.unsqueeze(-1) * (1.0 - v).unsqueeze(-1) * M10 +
		(1.0 - u).unsqueeze(-1) * v.unsqueeze(-1) * M01 +
		u.unsqueeze(-1) * v.unsqueeze(-1) * M11
	)
	delta = p - seed_xyz.view(1, 1, 3)
	t = (delta * n_gt.view(1, 1, 3)).sum(dim=-1)
	residual = (delta - t.unsqueeze(-1) * n_gt.view(1, 1, 3)).norm(dim=-1)
	score = residual * 1000.0 + t.abs()
	score = score.masked_fill(~valid, float("inf"))
	best_flat_t = torch.argmin(score)
	best_score = score.reshape(-1)[best_flat_t]
	if not torch.isfinite(best_score).detach().cpu().item():
		return None
	best_flat = int(best_flat_t.detach().cpu())
	y = best_flat // (W - 1)
	x = best_flat % (W - 1)
	return (
		float((torch.as_tensor(float(x), device=xyz_img.device, dtype=xyz_img.dtype) + v[y, x]).detach().cpu()),
		float((torch.as_tensor(float(y), device=xyz_img.device, dtype=xyz_img.dtype) + u[y, x]).detach().cpu()),
	)


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
	sampled = res.data.grid_sample_fullres(query, channels={"pred_dt"}).pred_dt
	if sampled is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")
	pred = sampled.squeeze(0).squeeze(0)
	if pred.ndim != 3:
		raise RuntimeError(f"pred_dt_flow_gate expected pooled pred_dt shape (N,H,W), got {tuple(pred.shape)}")
	return pred.amax(dim=0)


def _pred_dt_loss_sample_xyz(res: fit_model.FitResult3D) -> torch.Tensor:
	"""Exact LR positions used by pred_dt_loss for differentiable pred-dt sampling."""
	n = _vertex_normals(res.xyz_lr.detach())
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	return xyz_normal + xyz_tangential.detach()


def _neighbor_candidate_indices(*, Hm: int, Wm: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Directed 8-neighbor root->tip pairs on one LR winding grid."""
	tip_hs: list[torch.Tensor] = []
	tip_ws: list[torch.Tensor] = []
	root_hs: list[torch.Tensor] = []
	root_ws: list[torch.Tensor] = []
	for dh in (-1, 0, 1):
		for dw in (-1, 0, 1):
			if dh == 0 and dw == 0:
				continue
			h0 = max(0, -dh)
			h1 = Hm - max(0, dh)
			w0 = max(0, -dw)
			w1 = Wm - max(0, dw)
			if h1 <= h0 or w1 <= w0:
				continue
			h, w = torch.meshgrid(
				torch.arange(h0, h1, device=device, dtype=torch.long),
				torch.arange(w0, w1, device=device, dtype=torch.long),
				indexing="ij",
			)
			tip_hs.append(h.reshape(-1))
			tip_ws.append(w.reshape(-1))
			root_hs.append((h + dh).reshape(-1))
			root_ws.append((w + dw).reshape(-1))
	return (
		torch.cat(tip_hs, dim=0),
		torch.cat(tip_ws, dim=0),
		torch.cat(root_hs, dim=0),
		torch.cat(root_ws, dim=0),
	)


def _anticipatory_pull_cfg(cfg: dict | None) -> dict | None:
	if not isinstance(cfg, dict):
		return None
	pull_cfg = cfg.get("anticipatory_pull", None)
	if not isinstance(pull_cfg, dict) or not bool(pull_cfg.get("enabled", False)):
		return None
	return pull_cfg


def _anticipatory_normal_offset_factors(*, cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	steps = max(1, int(cfg.get("search_steps", 21)))
	if steps % 2 == 0:
		steps += 1
	deg = float(cfg.get("search_angle_degrees", 60.0))
	rad = np.deg2rad(max(0.0, deg))
	return torch.linspace(-float(np.tan(rad)), float(np.tan(rad)), steps, device=device, dtype=dtype)


def _anticipatory_reference_step(
	*,
	cfg: dict,
	device: torch.device,
	dtype: torch.dtype,
	params: fit_model.ModelParams3D | None = None,
	xyz0: torch.Tensor | None = None,
) -> torch.Tensor:
	if "search_step_length" in cfg:
		return torch.tensor(float(cfg["search_step_length"]), device=device, dtype=dtype).clamp_min(1e-6)
	if params is not None:
		return torch.tensor(float(params.mesh_step), device=device, dtype=dtype).clamp_min(1e-6)
	if xyz0 is not None:
		dh = (xyz0[1:] - xyz0[:-1]).norm(dim=-1).median() if xyz0.shape[0] > 1 else torch.tensor(1.0, device=device, dtype=dtype)
		dw = (xyz0[:, 1:] - xyz0[:, :-1]).norm(dim=-1).median() if xyz0.shape[1] > 1 else torch.tensor(1.0, device=device, dtype=dtype)
		return (0.5 * (dh + dw)).to(device=device, dtype=dtype).clamp_min(1e-6)
	return torch.tensor(1.0, device=device, dtype=dtype)


def _anticipatory_pull_prefetch_points(*, xyz_lr: torch.Tensor, cfg: dict) -> torch.Tensor | None:
	"""Conservative line-fit sample positions for sparse pred-dt cache prefetch."""
	if xyz_lr.shape[0] != 1:
		return None
	xyz0 = xyz_lr[0].detach()
	Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
	if Hm <= 1 or Wm <= 1:
		return None
	samples_n = max(2, int(cfg.get("samples", 8)))
	tip_h, tip_w, root_h, root_w = _neighbor_candidate_indices(Hm=Hm, Wm=Wm, device=xyz0.device)
	root = xyz0[root_h, root_w]
	tip = xyz0[tip_h, tip_w]
	normals = _vertex_normals(xyz_lr.detach())[0]
	offset_factors = _anticipatory_normal_offset_factors(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype)
	t = torch.linspace(0.0, 1.0, samples_n, device=xyz0.device, dtype=xyz0.dtype).view(1, 1, samples_n, 1)
	line_vec = tip - root
	ref_step = _anticipatory_reference_step(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype, xyz0=xyz0)
	n = normals[tip_h, tip_w]
	offset = ref_step * offset_factors.view(1, -1)
	target_vec = line_vec.view(-1, 1, 3) + offset.unsqueeze(-1) * n.view(-1, 1, 3)
	line = root.view(-1, 1, 1, 3) + t * target_vec.view(-1, int(offset_factors.numel()), 1, 3)
	return line.reshape(-1, 3)


def flow_gate_prefetch_points_for_result(
	*,
	res: fit_model.FitResult3D,
	cfg: dict | None,
) -> torch.Tensor | None:
	items = flow_gate_prefetch_items_for_result(res=res, cfg=cfg)
	return items.get("pred_dt")


def flow_gate_prefetch_items_for_result(
	*,
	res: fit_model.FitResult3D,
	cfg: dict | None,
) -> dict[str, torch.Tensor]:
	"""Exact channel sample positions for active flow-gate losses after model forward."""
	out: dict[str, torch.Tensor] = {}
	if cfg is None or not bool(cfg.get("enabled", False)):
		return out
	if res.xyz_lr.shape[0] != 1:
		return out
	xyz_hr = res.xyz_hr[0].detach()
	device = xyz_hr.device
	dtype = xyz_hr.dtype

	if _flow_gate_seed_xyz is not None:
		seed = torch.tensor(_flow_gate_seed_xyz, device=device, dtype=dtype).view(1, 1, 1, 3)
		out["nx"] = seed
		out["ny"] = seed

	loss_xyz = _pred_dt_loss_sample_xyz(res).detach().reshape(1, 1, -1, 3)
	out["pred_dt"] = loss_xyz

	r = int(max(0, int(cfg.get("pred_dt_pool_radius", 0))))
	if r > 0:
		step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
		offsets = torch.tensor(
			[(dx, dy, dz)
			 for dz in range(-r, r + 1)
			 for dy in range(-r, r + 1)
			 for dx in range(-r, r + 1)],
			device=device,
			dtype=dtype,
		)
		spacing = torch.tensor(res.data._spacing_for("pred_dt"), device=device, dtype=dtype)
		offsets = offsets * spacing.view(1, 3) * step_scale
		pool_points = (xyz_hr.unsqueeze(0) + offsets.view(-1, 1, 1, 3)).reshape(1, 1, -1, 3)
		out["pred_dt"] = torch.cat([out["pred_dt"], pool_points], dim=2)

	pull_cfg = _anticipatory_pull_cfg(cfg)
	if pull_cfg is not None:
		xyz0 = res.xyz_lr[0].detach()
		Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
		if Hm > 1 and Wm > 1:
			samples_n = max(2, int(pull_cfg.get("samples", 8)))
			tip_h, tip_w, root_h, root_w = _neighbor_candidate_indices(Hm=Hm, Wm=Wm, device=xyz0.device)
			root = xyz0[root_h, root_w]
			tip = xyz0[tip_h, tip_w]
			line_vec = tip - root
			n = _tip_gt_normals_from_result(res=res, tip_h=tip_h, tip_w=tip_w)
			offset_factors = _anticipatory_normal_offset_factors(cfg=pull_cfg, device=xyz0.device, dtype=xyz0.dtype)
			ref_step = _anticipatory_reference_step(cfg=pull_cfg, device=xyz0.device, dtype=xyz0.dtype, params=res.params)
			offset = ref_step * offset_factors.view(1, -1)
			target_vec = line_vec.view(-1, 1, 3) + offset.unsqueeze(-1) * n.view(-1, 1, 3)
			t = torch.linspace(0.0, 1.0, samples_n, device=xyz0.device, dtype=xyz0.dtype).view(1, 1, samples_n, 1)
			line = root.view(-1, 1, 1, 3) + t * target_vec.view(-1, int(offset_factors.numel()), 1, 3)
			pull_points = line.reshape(1, 1, -1, 3)
			out["pred_dt"] = torch.cat([out["pred_dt"], pull_points], dim=2) if "pred_dt" in out else pull_points
	return out


def _tip_gt_normals_from_result(
	*,
	res: fit_model.FitResult3D,
	tip_h: torch.Tensor,
	tip_w: torch.Tensor,
) -> torch.Tensor:
	if res.gt_normal_lr is None:
		raise RuntimeError("anticipatory_pull requires gt_normal_lr; GT nx/ny normals are missing at LR mesh points")
	n = res.gt_normal_lr[0, tip_h, tip_w].detach().to(device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
	return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _score_anticipatory_pull_candidates(
	*,
	res: fit_model.FitResult3D,
	cfg: dict,
) -> dict | None:
	"""Fit all one-step straight root->tip candidates, independent of flow."""
	if res.xyz_lr.shape[0] != 1:
		return None
	xyz0 = res.xyz_lr[0].detach()
	Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
	if Hm <= 1 or Wm <= 1:
		return None
	samples_n = max(2, int(cfg.get("samples", 8)))
	inlier_zero = float(cfg.get("inlier_zero", 80.0))
	inlier_one = float(cfg.get("inlier_one", 120.0))
	if inlier_one <= inlier_zero:
		raise ValueError("anticipatory_pull requires inlier_one > inlier_zero")
	chunk_candidates = max(256, int(cfg.get("chunk_candidates", 4096)))
	tip_h, tip_w, root_h, root_w = _neighbor_candidate_indices(Hm=Hm, Wm=Wm, device=xyz0.device)
	n_candidates = int(tip_h.numel())
	if n_candidates <= 0:
		return None
	offset_factors = _anticipatory_normal_offset_factors(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype)
	ref_step = _anticipatory_reference_step(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype, params=res.params)
	if offset_factors.ndim != 1:
		raise RuntimeError(f"anticipatory_pull expected 1D offset factors, got {tuple(offset_factors.shape)}")
	t = torch.linspace(0.0, 1.0, samples_n, device=xyz0.device, dtype=xyz0.dtype).view(1, 1, samples_n, 1)
	targets: list[torch.Tensor] = []
	prefixes: list[torch.Tensor] = []
	best_inliers: list[torch.Tensor] = []
	best_offsets_all: list[torch.Tensor] = []
	with torch.no_grad():
		for c0 in range(0, n_candidates, chunk_candidates):
			c1 = min(n_candidates, c0 + chunk_candidates)
			th = tip_h[c0:c1]
			tw = tip_w[c0:c1]
			rh = root_h[c0:c1]
			rw = root_w[c0:c1]
			root = xyz0[rh, rw]
			tip = xyz0[th, tw]
			line_vec = tip - root
			n = _tip_gt_normals_from_result(res=res, tip_h=th, tip_w=tw)
			offsets = ref_step * offset_factors
			target_vec = line_vec.view(-1, 1, 3) + offsets.view(1, -1, 1) * n.view(-1, 1, 3)
			query = root.view(-1, 1, 1, 3) + t * target_vec.view(c1 - c0, int(offset_factors.numel()), 1, 3)
			flat_query = query.reshape(1, 1, -1, 3)
			sampled = res.data.grid_sample_fullres(flat_query, channels={"pred_dt"}).pred_dt
			if sampled is None:
				raise RuntimeError("anticipatory_pull requires pred_dt to be loaded")
			pred = sampled.reshape(c1 - c0, int(offset_factors.numel()), samples_n)
			inlier = ((pred - inlier_zero) / (inlier_one - inlier_zero)).clamp(0.0, 1.0)
			prefix = inlier.cumprod(dim=2).mean(dim=2)
			best_score, best_offset_idx = prefix.max(dim=1)
			best_offsets = offsets[best_offset_idx]
			targets.append(tip + best_offsets.unsqueeze(-1) * n)
			prefixes.append(best_score)
			best_inliers.append(inlier[torch.arange(c1 - c0, device=xyz0.device), best_offset_idx])
			best_offsets_all.append(best_offsets)
	candidate_idx = torch.arange(n_candidates, device=xyz0.device, dtype=torch.long)
	return {
		"candidate_idx": candidate_idx,
		"tip_h": tip_h,
		"tip_w": tip_w,
		"root_h": root_h,
		"root_w": root_w,
		"target_xyz": torch.cat(targets, dim=0),
		"prefix": torch.cat(prefixes, dim=0),
		"inliers": torch.cat(best_inliers, dim=0),
		"offset": torch.cat(best_offsets_all, dim=0),
	}


def _activate_anticipatory_pull(
	*,
	candidates: dict | None,
	flow_weight: torch.Tensor,
	mask_lr: torch.Tensor,
	cfg: dict,
) -> dict | None:
	if candidates is None:
		return None
	tip_h = candidates["tip_h"]
	tip_w = candidates["tip_w"]
	root_h = candidates["root_h"]
	root_w = candidates["root_w"]
	prefix = candidates["prefix"].detach()
	root_weight = flow_weight[0, 0, root_h, root_w].detach()
	tip_weight = flow_weight[0, 0, tip_h, tip_w].detach()
	tip_mask = mask_lr[0, 0, tip_h, tip_w].detach()
	active = (tip_weight < 1.0) & (root_weight > 0.0) & (root_weight > tip_weight) & (prefix > 0.0) & (tip_mask > 0.0)
	if not bool(active.any().detach().cpu()):
		return {
			"candidate_idx": candidates["candidate_idx"][:0],
			"tip_h": tip_h[:0],
			"tip_w": tip_w[:0],
			"root_h": root_h[:0],
			"root_w": root_w[:0],
			"target_xyz": candidates["target_xyz"][:0],
			"candidate_weight": prefix[:0],
			"prefix": prefix[:0],
			"root_weight": root_weight[:0],
		}
	loss_weight = float(cfg.get("loss_weight", 1.0))
	candidate_weight = root_weight[active] * prefix[active] * tip_mask[active] * loss_weight
	return {
		"candidate_idx": candidates["candidate_idx"][active],
		"tip_h": tip_h[active],
		"tip_w": tip_w[active],
		"root_h": root_h[active],
		"root_w": root_w[active],
		"target_xyz": candidates["target_xyz"][active].detach(),
		"candidate_weight": candidate_weight.detach(),
		"prefix": prefix[active].detach(),
		"root_weight": root_weight[active].detach(),
	}


def _anticipatory_pull_loss_map(*, res: fit_model.FitResult3D, pull: dict | None) -> torch.Tensor:
	out = torch.zeros_like(res.mask_lr)
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return out
	live_tip = res.xyz_lr[0, pull["tip_h"], pull["tip_w"]]
	target = pull["target_xyz"].to(device=live_tip.device, dtype=live_tip.dtype)
	candidate_weight = pull["candidate_weight"].to(device=live_tip.device, dtype=live_tip.dtype)
	per_candidate = F.smooth_l1_loss(live_tip, target, reduction="none").sum(dim=-1)
	idx = pull["tip_h"] * int(res.xyz_lr.shape[2]) + pull["tip_w"]
	out_flat = out[0, 0].reshape(-1)
	out_flat.index_add_(0, idx, per_candidate * candidate_weight)
	return out


def _anticipatory_pull_debug_lr(
	*,
	pull: dict | None,
	Hm: int,
	Wm: int,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	weight = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	prefix = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	root_weight = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return weight, prefix, root_weight
	idx = pull["tip_h"] * Wm + pull["tip_w"]
	flat_weight = weight.reshape(-1)
	flat_prefix = prefix.reshape(-1)
	flat_root_weight = root_weight.reshape(-1)
	flat_weight.index_add_(0, idx, pull["candidate_weight"].to(device=device, dtype=torch.float32))
	flat_prefix.index_add_(0, idx, pull["prefix"].to(device=device, dtype=torch.float32))
	flat_root_weight.index_add_(0, idx, pull["root_weight"].to(device=device, dtype=torch.float32))
	count = torch.zeros(Hm * Wm, device=device, dtype=torch.float32)
	count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
	den = count.clamp_min(1.0)
	flat_prefix /= den
	flat_root_weight /= den
	return weight, prefix, root_weight


def _anticipatory_pull_overlay(
	*,
	pull: dict | None,
	height: int,
	width: int,
	sub_h: int,
	sub_w: int,
) -> np.ndarray | None:
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return None
	img = np.zeros((height, width, 3), dtype=np.float32)
	try:
		import cv2
	except Exception:
		cv2 = None
	tip_h = pull["tip_h"].detach().cpu().numpy()
	tip_w = pull["tip_w"].detach().cpu().numpy()
	root_h = pull["root_h"].detach().cpu().numpy()
	root_w = pull["root_w"].detach().cpu().numpy()
	candidate_weight = pull["candidate_weight"].detach().cpu().numpy()
	if candidate_weight.size == 0:
		return img
	scale = float(candidate_weight.max()) if float(candidate_weight.max()) > 0.0 else 1.0
	for th, tw, rh, rw, cw in zip(tip_h, tip_w, root_h, root_w, candidate_weight):
		x0 = int(round(float(rw) * float(sub_w)))
		y0 = int(round(float(rh) * float(sub_h)))
		x1 = int(round(float(tw) * float(sub_w)))
		y1 = int(round(float(th) * float(sub_h)))
		x0 = max(0, min(width - 1, x0))
		x1 = max(0, min(width - 1, x1))
		y0 = max(0, min(height - 1, y0))
		y1 = max(0, min(height - 1, y1))
		color = (0.0, 255.0 * min(1.0, float(cw) / scale), 255.0)
		if cv2 is not None:
			cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
		else:
			steps = max(abs(x1 - x0), abs(y1 - y0), 1)
			for i in range(steps + 1):
				a = i / steps
				x = int(round((1.0 - a) * x0 + a * x1))
				y = int(round((1.0 - a) * y0 + a * y1))
				img[y, x] = color
	return img


def _anticipatory_debug_points(cfg: dict) -> list[tuple[int, int]]:
	points = cfg.get("debug_points", None)
	if points is None:
		points = cfg.get("debug_fit_points", None)
	if not isinstance(points, (list, tuple)):
		return []
	out: list[tuple[int, int]] = []
	for p in points:
		if not isinstance(p, (list, tuple)) or len(p) < 2:
			continue
		try:
			out.append((int(p[0]), int(p[1])))
		except Exception:
			continue
	return out


def _anticipatory_debug_roi_center(cfg: dict) -> tuple[float, float, float] | None:
	center = cfg.get("debug_roi_center_xyz", None)
	if center is None:
		center = cfg.get("debug_center_xyz", None)
	if not isinstance(center, (list, tuple)) or len(center) < 3:
		return None
	try:
		return (float(center[0]), float(center[1]), float(center[2]))
	except Exception:
		return None


def _write_anticipatory_fit_debug_mosaic(
	*,
	res: fit_model.FitResult3D,
	cfg: dict,
	candidates: dict | None,
	pull: dict | None,
	flow_weight: torch.Tensor | None,
	stage_name: str,
	debug_index: int,
	out_dir: Path | None,
) -> None:
	points = _anticipatory_debug_points(cfg)
	roi_center = _anticipatory_debug_roi_center(cfg)
	if out_dir is None or (not points and roi_center is None) or candidates is None:
		return
	try:
		import cv2
	except Exception as exc:
		print(f"[pred_dt_flow_gate] anticipatory fit debug skipped: cv2 import failed: {exc}", flush=True)
		return
	xyz0 = res.xyz_lr[0].detach()
	Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
	if Hm <= 1 or Wm <= 1:
		return
	normals = _vertex_normals(res.xyz_lr.detach())[0]
	up = max(1, int(cfg.get("debug_slice_upsample", 8)))
	slice_w = max(24, int(cfg.get("debug_slice_width", 96)))
	slice_h = max(16, int(cfg.get("debug_slice_height", 48)))
	inlier_zero = float(cfg.get("inlier_zero", 80.0))
	inlier_one = float(cfg.get("inlier_one", 120.0))
	if inlier_one <= inlier_zero:
		return
	active_idx = None
	if pull is not None and int(pull["candidate_idx"].numel()) > 0:
		active_idx = pull["candidate_idx"].detach()

	def _candidate_for_point(h: int, w: int) -> int | None:
		if h < 0 or h >= Hm or w < 0 or w >= Wm:
			return None
		tip_match = (candidates["tip_h"] == h) & (candidates["tip_w"] == w)
		if active_idx is not None and int(active_idx.numel()) > 0:
			active_match = tip_match[active_idx]
			if bool(active_match.any().detach().cpu()):
				sub_idx = active_idx[active_match]
				prefix = candidates["prefix"][sub_idx]
				return int(sub_idx[int(torch.argmax(prefix).detach().cpu())].detach().cpu())
		if not bool(tip_match.any().detach().cpu()):
			return None
		sub_idx = tip_match.nonzero(as_tuple=True)[0]
		prefix = candidates["prefix"][sub_idx]
		return int(sub_idx[int(torch.argmax(prefix).detach().cpu())].detach().cpu())

	debug_candidates: list[tuple[int | None, str]] = []
	for tip_h, tip_w in points:
		cand_i = _candidate_for_point(tip_h, tip_w)
		debug_candidates.append((cand_i, f"tip=({tip_h},{tip_w})"))

	if roi_center is not None and flow_weight is not None:
		k = max(1, int(cfg.get("debug_roi_k", 8)))
		root_threshold = float(cfg.get("debug_roi_root_min", 0.5))
		tip_threshold = float(cfg.get("debug_roi_tip_max", 0.5))
		center_t = torch.tensor(roi_center, device=xyz0.device, dtype=xyz0.dtype)
		rh = candidates["root_h"]
		rw = candidates["root_w"]
		th = candidates["tip_h"]
		tw = candidates["tip_w"]
		root_weight = flow_weight[0, 0, rh, rw].detach()
		tip_weight = flow_weight[0, 0, th, tw].detach()
		eligible = (root_weight > root_threshold) & (tip_weight < tip_threshold)
		if bool(eligible.any().detach().cpu()):
			dist2 = ((xyz0[rh, rw] - center_t.view(1, 3)) ** 2).sum(dim=-1)
			dist2 = dist2.masked_fill(~eligible, float("inf"))
			n_pick = min(k, int(eligible.sum().detach().cpu()))
			picked = torch.topk(-dist2, k=n_pick, largest=True).indices
			for cand_t in picked:
				cand_i = int(cand_t.detach().cpu())
				d = float(dist2[cand_i].sqrt().detach().cpu())
				debug_candidates.append((cand_i, f"roi d={d:.1f}"))

	tiles: list[np.ndarray] = []
	xs_unit = torch.linspace(-0.25, 1.25, slice_w, device=xyz0.device, dtype=xyz0.dtype)
	ys_unit = torch.linspace(-1.0, 1.0, slice_h, device=xyz0.device, dtype=xyz0.dtype)
	def draw_text_bg(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float, color: tuple[int, int, int]) -> None:
		font = cv2.FONT_HERSHEY_SIMPLEX
		thickness = 1
		(size_w, size_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
		x, y = org
		x0 = max(0, x - 2)
		y0 = max(0, y - size_h - 3)
		x1 = min(img.shape[1] - 1, x + size_w + 2)
		y1 = min(img.shape[0] - 1, y + baseline + 3)
		cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
		cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

	for cand_i, label in debug_candidates:
		if cand_i is None:
			tile = np.zeros((slice_h * up, slice_w * up, 3), dtype=np.uint8)
			draw_text_bg(tile, f"{label} no cand", (6, 18), 0.45, (255, 255, 255))
			tiles.append(tile)
			continue
		rh = int(candidates["root_h"][cand_i].detach().cpu())
		rw = int(candidates["root_w"][cand_i].detach().cpu())
		th = int(candidates["tip_h"][cand_i].detach().cpu())
		tw = int(candidates["tip_w"][cand_i].detach().cpu())
		root = xyz0[rh, rw]
		tip = xyz0[th, tw]
		line = tip - root
		line_len = line.norm().clamp_min(1e-6)
		line_dir = line / line_len
		normal = _tip_gt_normals_from_result(
			res=res,
			tip_h=torch.tensor([th], device=xyz0.device, dtype=torch.long),
			tip_w=torch.tensor([tw], device=xyz0.device, dtype=torch.long),
		)[0]
		normal = normal / normal.norm().clamp_min(1e-6)
		offset_factors = _anticipatory_normal_offset_factors(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype)
		ref_step = _anticipatory_reference_step(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype, params=res.params)
		search_radius = float((ref_step * offset_factors.abs().max()).detach().cpu()) if offset_factors.numel() > 0 else float(ref_step.detach().cpu())
		search_radius = max(search_radius, 1.0e-6)
		xs = xs_unit * line_len
		ys = ys_unit * search_radius
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
		query = root.view(1, 1, 3) + grid_x.unsqueeze(-1) * line_dir.view(1, 1, 3) + grid_y.unsqueeze(-1) * normal.view(1, 1, 3)
		sampled = res.data.grid_sample_fullres(
			query.view(1, slice_h, slice_w, 3),
			channels={"pred_dt"},
		).pred_dt
		if sampled is None:
			continue
		pred = sampled.squeeze(0).squeeze(0).squeeze(0)
		gray = ((pred - inlier_zero) / (inlier_one - inlier_zero)).clamp(0.0, 1.0)
		img = (gray.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
		rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		rgb = cv2.resize(rgb, (slice_w * up, slice_h * up), interpolation=cv2.INTER_NEAREST)
		y0 = int(round((0.0 + 1.0) * 0.5 * (slice_h - 1) * up))
		best_offset = float(candidates["offset"][cand_i].detach().cpu()) if "offset" in candidates else 0.0
		x_root = int(round((0.0 + 0.25) / 1.5 * (slice_w - 1) * up))
		x_tip = int(round((1.0 + 0.25) / 1.5 * (slice_w - 1) * up))
		y_tip_offset = int(round(((best_offset / search_radius) + 1.0) * 0.5 * (slice_h - 1) * up))
		cv2.line(rgb, (x_root, y0), (x_tip, y0), (140, 140, 140), 1, cv2.LINE_AA)
		cv2.circle(rgb, (x_root, y0), 3, (255, 128, 0), -1, cv2.LINE_AA)
		cv2.circle(rgb, (x_tip, y0), 3, (255, 0, 128), -1, cv2.LINE_AA)
		draw_text_bg(rgb, "R0", (max(0, x_root - 8), max(10, y0 - 6)), 0.28, (255, 128, 0))
		draw_text_bg(rgb, "T0", (min(rgb.shape[1] - 20, x_tip + 3), max(10, y0 - 6)), 0.28, (255, 0, 128))
		cv2.line(rgb, (x_root, y0), (x_tip, y_tip_offset), (0, 255, 255), 1, cv2.LINE_AA)
		inliers = candidates["inliers"][cand_i].detach().cpu().numpy()
		baseline_scores: list[float] | None = None
		if 0 <= y0 // up < gray.shape[0]:
			baseline_scores = []
		for si, score in enumerate(inliers):
			a = si / max(1, len(inliers) - 1)
			xp = int(round(((a + 0.25) / 1.5) * (slice_w - 1) * up))
			yp = int(round(((a * best_offset / search_radius) + 1.0) * 0.5 * (slice_h - 1) * up))
			cv2.circle(rgb, (xp, yp), 2, (0, 0, 255), -1, cv2.LINE_AA)
			if si < 8:
				draw_text_bg(rgb, f"{score:.2f}", (max(0, xp - 14), min(rgb.shape[0] - 4, yp + 16 + (si % 2) * 12)), 0.28, (255, 255, 255))
			if baseline_scores is not None:
				xg = max(0, min(gray.shape[1] - 1, int(round(xp / up))))
				yg = max(0, min(gray.shape[0] - 1, int(round(y0 / up))))
				baseline_scores.append(float(gray[yg, xg].detach().cpu()))
				cv2.circle(rgb, (xp, y0), 2, (255, 180, 0), -1, cv2.LINE_AA)
				if si < 8:
					draw_text_bg(rgb, f"{baseline_scores[-1]:.2f}", (max(0, xp - 14), max(10, y0 - 10 - (si % 2) * 12)), 0.28, (255, 180, 0))
		prefix = float(candidates["prefix"][cand_i].detach().cpu())
		draw_text_bg(rgb, f"{label} tip=({th},{tw}) root=({rh},{rw})", (6, 12), 0.35, (255, 255, 255))
		eq_angle = np.rad2deg(np.arctan2(best_offset, float(ref_step.detach().cpu())))
		draw_text_bg(rgb, f"line={prefix:.3f} off={best_offset:.1f} eq={eq_angle:.1f}", (6, 26), 0.35, (0, 255, 255))
		cv2.line(rgb, (0, y0), (rgb.shape[1] - 1, y0), (96, 96, 96), 1, cv2.LINE_AA)
		tiles.append(rgb)
	if not tiles:
		return
	n = len(tiles)
	cols = max(1, int(np.ceil(np.sqrt(2.0 * n))))
	rows = int(np.ceil(n / cols))
	tile_h, tile_w = tiles[0].shape[:2]
	mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
	for i, tile in enumerate(tiles):
		r = i // cols
		c = i % cols
		mosaic[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
	out_dir.mkdir(parents=True, exist_ok=True)
	for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
		path = out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_fit_points.jpg"
		cv2.imwrite(str(path), mosaic)


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
	pull_weight_hr: np.ndarray | None = None,
	pull_prefix_hr: np.ndarray | None = None,
	pull_root_weight_hr: np.ndarray | None = None,
	pull_overlay_rgb: np.ndarray | None = None,
) -> None:
	if out_dir is None:
		return
	try:
		import tifffile
	except Exception as exc:
		print(f"[pred_dt_flow_gate] debug write skipped: tifffile import failed: {exc}", flush=True)
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	pred_raw_u8 = pred_u8.astype(np.uint8, copy=True)
	flow = np.zeros_like(pred_raw_u8, dtype=np.float32) if flow_hr is None else np.asarray(flow_hr, dtype=np.float32)
	grid_flow = None if smooth_grid_flow is None else np.asarray(smooth_grid_flow, dtype=np.float32)
	graph_flow = None if graph_edge_flow_rgb is None else np.asarray(graph_edge_flow_rgb, dtype=np.float32)
	weights = np.zeros_like(pred_raw_u8, dtype=np.float32) if weight_hr is None else np.asarray(weight_hr, dtype=np.float32)
	pull_weight = None if pull_weight_hr is None else np.asarray(pull_weight_hr, dtype=np.float32)
	pull_prefix = None if pull_prefix_hr is None else np.asarray(pull_prefix_hr, dtype=np.float32)
	pull_root_weight = None if pull_root_weight_hr is None else np.asarray(pull_root_weight_hr, dtype=np.float32)
	pull_overlay = None if pull_overlay_rgb is None else np.asarray(pull_overlay_rgb, dtype=np.float32)
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
		layers_path = out_dir / f"pred_dt_flow_gate_{suffix}_layers.tif"
		with tifffile.TiffWriter(str(layers_path)) as tw:
			write_named_layer(tw, pred_raw_u8, name="pred_dt")
			write_named_layer(tw, flow, name="raw_flow_bilinear")
			if grid_flow is not None:
				write_named_layer(tw, grid_flow, name="smooth_grid_flow")
			if graph_flow is not None:
				write_named_layer(tw, graph_flow, name="graph_edge_flow")
			write_named_layer(tw, weights, name="flow_gate_weight")
			if pull_weight is not None:
				write_named_layer(tw, pull_weight, name="anticipatory_pull_weight")
			if pull_prefix is not None:
				write_named_layer(tw, pull_prefix, name="anticipatory_pull_prefix")
			if pull_root_weight is not None:
				write_named_layer(tw, pull_root_weight, name="anticipatory_pull_root_weight")
			if pull_overlay is not None:
				write_named_layer(tw, pull_overlay, name="anticipatory_pull_overlay")
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_pred_dt.tif"), pred_raw_u8)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_raw_flow.tif"), flow)
		if grid_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_smooth_grid_flow.tif"), grid_flow)
		if graph_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_graph_edge_flow.tif"), graph_flow)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_weight.tif"), weights)
		if pull_weight is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_weight.tif"), pull_weight)
		if pull_prefix is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_prefix.tif"), pull_prefix)
		if pull_root_weight is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_root_weight.tif"), pull_root_weight)
		if pull_overlay is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_overlay.tif"), pull_overlay)


def _write_flow_gate_weight_jpg(
	*,
	stage_name: str,
	debug_index: int,
	pred_u8: np.ndarray,
	used_weight_hr: np.ndarray,
	out_dir: Path | None,
) -> None:
	global _flow_gate_jpg_warned
	if out_dir is None:
		return
	pred = np.asarray(pred_u8)
	weight = np.asarray(used_weight_hr, dtype=np.float32)
	if pred.ndim != 2 or weight.ndim != 2:
		return
	if pred.shape != weight.shape:
		return
	pred_vis = ((pred.astype(np.float32) - 80.0) / (127.0 - 80.0)).clip(0.0, 1.0)
	weight_vis = weight.clip(0.0, 1.0)
	img = np.concatenate([
		(pred_vis * 255.0 + 0.5).astype(np.uint8),
		(weight_vis * 255.0 + 0.5).astype(np.uint8),
	], axis=1)
	jpg_dir = out_dir / "pred_dt_flow_gate_weight_jpg"
	try:
		jpg_dir.mkdir(parents=True, exist_ok=True)
		for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
			path = jpg_dir / f"{suffix}_pred_dt_and_used_weight.jpg"
			try:
				import cv2
				if not cv2.imwrite(str(path), img):
					raise RuntimeError("cv2.imwrite returned false")
			except Exception:
				from PIL import Image
				Image.fromarray(img, mode="L").save(str(path), quality=95)
	except Exception as exc:
		if not _flow_gate_jpg_warned:
			print(f"[pred_dt_flow_gate] jpg weight write skipped: {exc}", flush=True)
			_flow_gate_jpg_warned = True


def _flow_gate_weight(res: fit_model.FitResult3D) -> torch.Tensor | tuple[torch.Tensor, dict | None] | None:
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

	flow_zero = float(cfg.get("flow_zero", 20.0))
	flow_one = float(cfg.get("flow_one", 100.0))
	gate_factor = float(cfg.get("gate_factor", 1.0))
	backtrack_distance = float(cfg.get("backtrack_distance", 10.0))
	pred_dt_pool_radius = int(cfg.get("pred_dt_pool_radius", 0))
	pred_dt_pool_step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
	pull_cfg = _anticipatory_pull_cfg(cfg)
	if flow_one <= flow_zero:
		raise ValueError("pred_dt_flow_gate requires flow_one > flow_zero")
	if not 0.0 <= gate_factor <= 1.0:
		raise ValueError("pred_dt_flow_gate requires gate_factor in [0, 1]")
	debug = bool(cfg.get("debug", True))
	debug_index = 0
	if debug:
		debug_index = _flow_gate_debug_counts.get(_flow_gate_stage, 0)
		_flow_gate_debug_counts[_flow_gate_stage] = debug_index + 1
	write_layer_debug = debug and (debug_index % 10) == 0
	def mark(label: str) -> float:
		return time.perf_counter()

	def done(label: str, t0: float) -> None:
		return None

	with torch.no_grad():
		xyz_hr = res.xyz_hr[0].detach()
		_t = mark("sample_pred_dt_max3d")
		pred_hr = _sample_pred_dt_max3d(
			res=res,
			xyz_hr=xyz_hr,
			radius=pred_dt_pool_radius,
			step_scale=pred_dt_pool_step_scale,
		)
		pred_img = pred_hr.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()
		done("sample_pred_dt_max3d", _t)
		He, We = pred_img.shape

		_t = mark("project_seed")
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
		done("project_seed", _t)

		_t = mark("build_query_grid")
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
		done("build_query_grid", _t)

		if write_layer_debug:
			_t = mark("write_initial_debug")
			_write_flow_gate_debug(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				pred_u8=pred_img,
				flow_hr=None,
				smooth_grid_flow=None,
				graph_edge_flow_rgb=None,
				weight_hr=None,
				out_dir=_flow_gate_out_dir,
			)
			done("write_initial_debug", _t)
		def _compute_flow_outputs():
			return dense_batch_flow.compute_flow_grid(
					pred_img,
					source_xy=(source_x, source_y),
					query_xy=query_xy,
					verbose=False,
					return_debug=write_layer_debug,
					return_metadata=True,
					grid_step=grid_step,
					backtrack_distance=backtrack_distance,
				)

		pull_candidates = None
		try:
			_t = mark("compute_flow_grid")
			if pull_cfg is not None:
				with ThreadPoolExecutor(max_workers=1) as executor:
					flow_future = executor.submit(_compute_flow_outputs)
					pull_candidates = _score_anticipatory_pull_candidates(res=res, cfg=pull_cfg)
					flow_outputs = flow_future.result()
			else:
				flow_outputs = _compute_flow_outputs()
			done("compute_flow_grid", _t)
			if write_layer_debug:
				query_flow, dense_flow, smooth_grid_flow, graph_edge_flow_rgb, flow_metadata = flow_outputs
			else:
				query_flow, dense_flow, flow_metadata = flow_outputs
				smooth_grid_flow = None
				graph_edge_flow_rgb = None
		except RuntimeError as exc:
			message = str(exc)
			source_value = int(pred_img[source_y, source_x]) if 0 <= source_y < He and 0 <= source_x < We else -1
			if "white distance domain" in message:
				gate_weight = seed_area.to(dtype=torch.float32)
				weight = gate_weight
				if gate_factor < 1.0:
					weight = gate_factor * gate_weight + (1.0 - gate_factor)
				valid = res.mask_lr > 0.0
				valid_count = max(1.0, float(valid.sum().detach().cpu()))
				used_weight = weight * res.mask_lr
				_flow_gate_last_stats = {
					"pred_dt_gate_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_eq1": float(((gate_weight >= 1.0) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_n_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_n_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_gate_n_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
					"pred_dt_pull_active_frac": 0.0,
					"pred_dt_pull_weight_mean": 0.0,
					"pred_dt_pull_prefix_mean": 0.0,
				}
				print(
					f"[pred_dt_flow_gate] {_flow_gate_stage}: skipped flow "
					f"(source outside C++ flow domain, value={source_value})",
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
					)
				if debug:
					used_weight_hr = F.interpolate(
						used_weight,
						size=(He, We),
						mode="nearest",
					)[0, 0].detach().cpu().numpy().astype(np.float32)
					_write_flow_gate_weight_jpg(
						stage_name=_flow_gate_stage,
						debug_index=debug_index,
						pred_u8=pred_img,
						used_weight_hr=used_weight_hr,
						out_dir=_flow_gate_out_dir,
					)
				return weight, None
			raise
		flow_lr = torch.as_tensor(
			query_flow.reshape(Hm, Wm),
			device=res.xyz_lr.device,
			dtype=torch.float32,
		).view(1, 1, Hm, Wm)
		_t = mark("compute_weight")
		seed_capacity = float(flow_metadata.get("source_capacity", 0.0))
		effective_flow_one = flow_one
		effective_flow_zero = flow_zero
		if seed_capacity > 0.0 and seed_capacity < flow_one:
			scale = seed_capacity / flow_one
			effective_flow_one = seed_capacity
			effective_flow_zero = flow_zero * scale
		if effective_flow_one <= effective_flow_zero:
			effective_flow_zero = max(0.0, effective_flow_one - 1.0)
		gate_weight = ((flow_lr - effective_flow_zero) / (effective_flow_one - effective_flow_zero)).clamp(0.0, 1.0)
		gate_weight = torch.where(seed_area, torch.ones_like(gate_weight), gate_weight)
		weight = gate_weight
		if gate_factor < 1.0:
			weight = gate_factor * gate_weight + (1.0 - gate_factor)
		valid = res.mask_lr > 0.0
		valid_count = max(1.0, float(valid.sum().detach().cpu()))
		used_weight = weight * res.mask_lr
		_flow_gate_last_stats = {
			"pred_dt_gate_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_eq1": float(((gate_weight >= 1.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
		}
		pull = None
		if pull_cfg is not None:
			pull = _activate_anticipatory_pull(
				candidates=pull_candidates,
				flow_weight=weight,
				mask_lr=res.mask_lr,
				cfg=pull_cfg,
			)
			active_count = 0 if pull is None else int(pull["tip_h"].numel())
			candidate_count = 0 if pull_candidates is None else int(pull_candidates["tip_h"].numel())
			_flow_gate_last_stats.update({
				"pred_dt_pull_active_frac": float(active_count) / float(max(1, candidate_count)),
				"pred_dt_pull_weight_mean": (
					float(pull["candidate_weight"].mean().detach().cpu()) if active_count > 0 else 0.0
				),
				"pred_dt_pull_prefix_mean": (
					float(pull["prefix"].mean().detach().cpu()) if active_count > 0 else 0.0
				),
			})
		done("compute_weight", _t)

		if write_layer_debug:
			_t = mark("write_layer_debug")
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
			pull_weight_hr = None
			pull_prefix_hr = None
			pull_root_weight_hr = None
			pull_overlay_rgb = None
			if pull_cfg is not None:
				pull_weight_lr, pull_prefix_lr, pull_root_weight_lr = _anticipatory_pull_debug_lr(
					pull=pull,
					Hm=Hm,
					Wm=Wm,
					device=res.xyz_lr.device,
				)
				pull_weight_hr = pull_weight_lr.detach().cpu().numpy().astype(np.float32)
				pull_prefix_hr = pull_prefix_lr.detach().cpu().numpy().astype(np.float32)
				pull_root_weight_hr = pull_root_weight_lr.detach().cpu().numpy().astype(np.float32)
				pull_overlay_rgb = _anticipatory_pull_overlay(
					pull=pull,
					height=He,
					width=We,
					sub_h=sub_h,
					sub_w=sub_w,
				)
			_write_flow_gate_debug(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				pred_u8=pred_img,
				flow_hr=flow_hr,
				smooth_grid_flow=smooth_grid_flow,
				graph_edge_flow_rgb=graph_edge_flow_rgb,
				weight_hr=weight_hr,
				pull_weight_hr=pull_weight_hr,
				pull_prefix_hr=pull_prefix_hr,
				pull_root_weight_hr=pull_root_weight_hr,
				pull_overlay_rgb=pull_overlay_rgb,
				out_dir=_flow_gate_out_dir,
			)
			if pull_cfg is not None:
				_write_anticipatory_fit_debug_mosaic(
					res=res,
					cfg=pull_cfg,
					candidates=pull_candidates,
					pull=pull,
					flow_weight=weight,
					stage_name=_flow_gate_stage,
					debug_index=debug_index,
					out_dir=_flow_gate_out_dir,
				)
			done("write_layer_debug", _t)
		if debug:
			_t = mark("write_weight_jpg")
			used_weight_hr = F.interpolate(
				used_weight,
				size=(He, We),
				mode="nearest",
			)[0, 0].detach().cpu().numpy().astype(np.float32)
			_write_flow_gate_weight_jpg(
				stage_name=_flow_gate_stage,
				debug_index=debug_index,
				pred_u8=pred_img,
				used_weight_hr=used_weight_hr,
				out_dir=_flow_gate_out_dir,
			)
			done("write_weight_jpg", _t)
		return weight, pull


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: clamped outside L2 plus inside L1 pushing mesh into prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)^2    — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	# Project gradients onto surface normal only (prevents tangential crimping)
	xyz = _pred_dt_loss_sample_xyz(res)

	# Sample pred_dt using common sampling with per-channel spacing and diff gradients
	sampled = res.data.grid_sample_fullres(xyz, diff=True, channels={"pred_dt"})
	sampled_raw = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	lm_out_l1 = (127.0 - sampled_raw).clamp(min=0)   # outside: 1–47, inside: 0 (no grad)
	lm_out = lm_out_l1 * lm_out_l1
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = lm_out + _INNER_FACTOR * lm_in

	mask = res.mask_lr
	flow_result = _flow_gate_weight(res)
	pull = None
	if isinstance(flow_result, tuple):
		flow_gate, pull = flow_result
	else:
		flow_gate = flow_result
	wsum = mask.sum()
	if flow_gate is not None:
		pull_lm = _anticipatory_pull_loss_map(res=res, pull=pull)
		loss = (lm * mask * flow_gate + pull_lm).mean()
	else:
		if float(wsum) > 0.0:
			loss = (lm * mask).sum() / wsum
		else:
			loss = lm.mean()
	if flow_gate is not None:
		return loss, (lm, pull_lm), (mask * flow_gate, torch.ones_like(pull_lm))
	return loss, (lm,), (mask,)
