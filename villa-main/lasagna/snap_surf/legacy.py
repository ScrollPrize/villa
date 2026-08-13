from __future__ import annotations

import math

import torch

import opt_loss_station

from .config import SnapSurfConfig
from .tensor import _CORNERS_2D, _dihedral_transforms, _normalized_seed_quad, _transform_det_sign


def _closest_external_seed_surface(
	*,
	seed: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	chunk_quads: int = 262144,
) -> tuple[tuple[int, int] | None, torch.Tensor | None, float]:
	"""Closest point on any valid external tifxyz quad to the seed."""
	if ext_valid.numel() == 0 or not bool(ext_valid.any().detach().cpu()):
		return None, None, float("inf")
	if ext_xyz.ndim != 3 or int(ext_xyz.shape[-1]) != 3:
		return None, None, float("inf")
	H, W, _ = ext_xyz.shape
	if H < 2 or W < 2 or ext_quad_valid.numel() == 0 or not bool(ext_quad_valid.any().detach().cpu()):
		pts = ext_xyz[ext_valid & torch.isfinite(ext_xyz).all(dim=-1)]
		if pts.numel() == 0:
			return None, None, float("inf")
		dist2 = (pts - seed.view(1, 3)).square().sum(dim=-1)
		best = int(torch.argmin(dist2).detach().cpu())
		return None, pts[best].detach(), math.sqrt(float(dist2[best].detach().cpu()))

	valid_ids = ext_quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, None, float("inf")
	Wq = W - 1
	rows_all = torch.div(valid_ids, Wq, rounding_mode="floor")
	cols_all = valid_ids - rows_all * Wq
	best = torch.full((), float("inf"), device=ext_xyz.device, dtype=ext_xyz.dtype)
	best_hw: tuple[int, int] | None = None
	best_point: torch.Tensor | None = None
	chunk = max(1, int(chunk_quads))
	for start in range(0, int(valid_ids.numel()), chunk):
		end = min(start + chunk, int(valid_ids.numel()))
		rows = rows_all[start:end]
		cols = cols_all[start:end]
		p00 = ext_xyz[rows, cols]
		p10 = ext_xyz[rows + 1, cols]
		p01 = ext_xyz[rows, cols + 1]
		p11 = ext_xyz[rows + 1, cols + 1]
		finite = (
			torch.isfinite(p00).all(dim=-1)
			& torch.isfinite(p10).all(dim=-1)
			& torch.isfinite(p01).all(dim=-1)
			& torch.isfinite(p11).all(dim=-1)
		)
		if not bool(finite.any().detach().cpu()):
			continue
		rows_f = rows[finite]
		cols_f = cols[finite]
		p00 = p00[finite]
		p10 = p10[finite]
		p01 = p01[finite]
		p11 = p11[finite]
		cp0, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p10, p11)
		cp1, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p11, p01)
		d20 = (cp0 - seed.view(1, 3)).square().sum(dim=-1)
		d21 = (cp1 - seed.view(1, 3)).square().sum(dim=-1)
		use_first = d20 <= d21
		d2 = torch.where(use_first, d20, d21)
		local = int(torch.argmin(d2).detach().cpu())
		local_best = d2[local]
		if float(local_best.detach().cpu()) < float(best.detach().cpu()):
			best = local_best
			best_hw = (int(rows_f[local].detach().cpu()), int(cols_f[local].detach().cpu()))
			best_point = (cp0 if bool(use_first[local].detach().cpu()) else cp1)[local].detach()
	if not bool(torch.isfinite(best).detach().cpu()):
		return None, None, float("inf")
	return best_hw, best_point, math.sqrt(float(best.detach().cpu()))


def _closest_model_surface_quad(
	*,
	point: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
) -> tuple[tuple[int, int, int] | None, float]:
	D, H, W, _ = model_xyz.shape
	if H < 2 or W < 2:
		return None, float("inf")
	quad_valid = (
		model_valid[:, :-1, :-1]
		& model_valid[:, 1:, :-1]
		& model_valid[:, :-1, 1:]
		& model_valid[:, 1:, 1:]
	)
	valid_ids = quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, float("inf")
	Hq, Wq = H - 1, W - 1
	d = torch.div(valid_ids, Hq * Wq, rounding_mode="floor")
	rem = valid_ids - d * Hq * Wq
	h = torch.div(rem, Wq, rounding_mode="floor")
	w = rem - h * Wq
	p00 = model_xyz[d, h, w]
	p10 = model_xyz[d, h + 1, w]
	p01 = model_xyz[d, h, w + 1]
	p11 = model_xyz[d, h + 1, w + 1]
	cp0, _ = opt_loss_station._closest_points_on_triangles(point, p00, p10, p11)
	cp1, _ = opt_loss_station._closest_points_on_triangles(point, p00, p11, p01)
	d2 = torch.minimum(
		(cp0 - point.view(1, 3)).square().sum(dim=-1),
		(cp1 - point.view(1, 3)).square().sum(dim=-1),
	)
	best = int(torch.argmin(d2).detach().cpu())
	return (
		int(d[best].detach().cpu()),
		int(h[best].detach().cpu()),
		int(w[best].detach().cpu()),
	), math.sqrt(float(d2[best].detach().cpu()))


def _closest_point_uv_on_model_quad(
	*,
	point: torch.Tensor,
	model_xyz: torch.Tensor,
	model_quad: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, float]:
	d, h, w = model_quad
	p00 = model_xyz[d, h, w].view(1, 3)
	p10 = model_xyz[d, h + 1, w].view(1, 3)
	p01 = model_xyz[d, h, w + 1].view(1, 3)
	p11 = model_xyz[d, h + 1, w + 1].view(1, 3)
	cp0, bary0 = opt_loss_station._closest_points_on_triangles(point, p00, p10, p11)
	cp1, bary1 = opt_loss_station._closest_points_on_triangles(point, p00, p11, p01)
	d20 = (cp0 - point.view(1, 3)).square().sum(dim=-1)
	d21 = (cp1 - point.view(1, 3)).square().sum(dim=-1)
	if float(d20[0].detach().cpu()) <= float(d21[0].detach().cpu()):
		b = bary0[0]
		uv = torch.stack([
			torch.as_tensor(float(h), device=point.device, dtype=point.dtype) + b[1] + b[2],
			torch.as_tensor(float(w), device=point.device, dtype=point.dtype) + b[2],
		], dim=0)
		return cp0[0].detach(), uv.detach(), math.sqrt(float(d20[0].detach().cpu()))
	b = bary1[0]
	uv = torch.stack([
		torch.as_tensor(float(h), device=point.device, dtype=point.dtype) + b[1],
		torch.as_tensor(float(w), device=point.device, dtype=point.dtype) + b[1] + b[2],
	], dim=0)
	return cp1[0].detach(), uv.detach(), math.sqrt(float(d21[0].detach().cpu()))


def _map_init_seed_quad_uv_for_points(
	points: torch.Tensor,
	*,
	ext_xyz: torch.Tensor,
	model_xyz: torch.Tensor,
	ext_quad: tuple[int, int],
	model_quad: tuple[int, int, int],
	transform: tuple[tuple[int, int], ...],
	ext_anchor: torch.Tensor,
	model_anchor_uv: torch.Tensor,
	eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
	out = torch.full((*points.shape[:-1], 2), float("nan"), device=points.device, dtype=points.dtype)
	ok = torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
	if points.numel() == 0:
		return out, ok, None
	eh, ew = ext_quad
	d, mh, mw = model_quad
	ext_pts = torch.stack([ext_xyz[eh + th, ew + tw] for th, tw in transform], dim=0)
	model_pts = torch.stack([model_xyz[d, mh + sh, mw + sw] for sh, sw in _CORNERS_2D], dim=0)
	if not bool(torch.isfinite(ext_pts).all().detach().cpu()):
		return out, ok, "non-finite external seed quad"
	if not bool(torch.isfinite(model_pts).all().detach().cpu()):
		return out, ok, "non-finite model seed quad"
	if not bool(torch.isfinite(ext_anchor).all().detach().cpu()):
		return out, ok, "non-finite external seed anchor"
	if not bool(torch.isfinite(model_anchor_uv).all().detach().cpu()):
		return out, ok, "non-finite model seed anchor"

	ext_h = ext_pts[1] - ext_pts[0]
	ext_w = ext_pts[2] - ext_pts[0]
	model_h = model_pts[1] - model_pts[0]
	model_w = model_pts[2] - model_pts[0]
	ext_h_len = ext_h.norm()
	ext_w_len = ext_w.norm()
	model_h_len = model_h.norm()
	model_w_len = model_w.norm()
	lengths = torch.stack([ext_h_len, ext_w_len, model_h_len, model_w_len])
	if not bool(torch.isfinite(lengths).all().detach().cpu()):
		return out, ok, "non-finite seed quad edge length"
	if float(lengths.min().detach().cpu()) <= float(eps):
		return out, ok, "degenerate seed quad edge"

	ext_basis = torch.stack([ext_h / ext_h_len, ext_w / ext_w_len], dim=1)
	model_unit = torch.stack([model_h / model_h_len, model_w / model_w_len], dim=1)
	model_edges = torch.stack([model_h, model_w], dim=1)
	flat = points.reshape(-1, 3)
	point_ok = torch.isfinite(flat).all(dim=-1)
	if not bool(point_ok.any().detach().cpu()):
		return out, ok, None
	rel = (flat[point_ok] - ext_anchor.view(1, 3)).transpose(0, 1)
	try:
		ext_coeff = torch.linalg.lstsq(ext_basis, rel).solution.transpose(0, 1)
		model_disp = (ext_coeff @ model_unit.transpose(0, 1)).transpose(0, 1)
		uv_delta = torch.linalg.lstsq(model_edges, model_disp).solution.transpose(0, 1)
	except RuntimeError as exc:
		return out, ok, f"seed quad solve failed: {exc}"
	uv = model_anchor_uv.view(1, 2) + uv_delta
	local_ok = torch.isfinite(uv).all(dim=-1)
	flat_out = out.reshape(-1, 2)
	flat_ok = ok.reshape(-1)
	idx = point_ok.nonzero(as_tuple=False).flatten()
	flat_out[idx] = torch.where(local_ok.unsqueeze(-1), uv, flat_out[idx])
	flat_ok[idx] = local_ok
	return out, ok, None


def _choose_seed_transform(
	*,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	model_quad: tuple[int, int, int],
	ext_quad: tuple[int, int],
	cfg: SnapSurfConfig,
) -> tuple[tuple[tuple[int, int], ...], int]:
	transforms = [_dihedral_transforms()[0]] if cfg.orientation in {"identity", "none"} else _dihedral_transforms()
	d, mh, mw = model_quad
	eh, ew = ext_quad
	model_pts = torch.stack([model_xyz[d, mh + sh, mw + sw] for sh, sw in _CORNERS_2D], dim=0)
	model_norm = _normalized_seed_quad(model_pts)
	best = transforms[0]
	best_score = float("inf")
	for transform in transforms:
		ext_pts = torch.stack([ext_xyz[eh + th, ew + tw] for th, tw in transform], dim=0)
		ext_norm = _normalized_seed_quad(ext_pts)
		score = float((model_norm - ext_norm).norm(dim=-1).sum().detach().cpu())
		if score < best_score:
			best_score = score
			best = transform
	return best, _transform_det_sign(best)


def _huber(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	abs_r = residual.abs()
	d = float(delta)
	return torch.where(abs_r <= d, 0.5 * residual.square(), d * (abs_r - 0.5 * d))


__all__ = [name for name in globals() if not name.startswith("__")]
