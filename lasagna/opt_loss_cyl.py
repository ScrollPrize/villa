from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import model as fit_model

_last_stats: dict[str, float] = {}
_candidate_terms: dict[str, torch.Tensor] = {}
_candidate_normal_stats: dict[str, torch.Tensor] = {}
_sample_cache_key: int | None = None
_sample_cache_value: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
_shell_geometry_cache_key: tuple[int, int, bool] | None = None
_shell_geometry_cache_value: dict[str, torch.Tensor | bool | None] | None = None
_shell_sample_cache_key: int | None = None
_shell_sample_cache_value: dict[tuple[int, tuple[int, ...]], tuple[torch.Tensor | None, torch.Tensor | None]] = {}
_compile_shell_normal = False
_compile_shell_normal_backend: str | None = None
_compile_shell_normal_mode: str | None = None
_compile_shell_normal_dynamic = False
_compile_shell_normal_fullgraph = False
_compile_shell_normal_key: tuple[bool, str | None, str | None, bool, bool] | None = None
_compiled_shell_normal_core = None
_compile_shell_normal_disabled_reason: str | None = None
_BEND_THRESHOLD_RAD = math.radians(60.0)
_BEND_COS_THRESHOLD = math.cos(math.pi - _BEND_THRESHOLD_RAD)


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def reset_candidate_terms() -> None:
	global _candidate_terms, _candidate_normal_stats, _sample_cache_key, _sample_cache_value, _last_stats
	global _shell_geometry_cache_key, _shell_geometry_cache_value, _shell_sample_cache_key, _shell_sample_cache_value
	_candidate_terms = {}
	_candidate_normal_stats = {}
	_sample_cache_key = None
	_sample_cache_value = None
	_shell_geometry_cache_key = None
	_shell_geometry_cache_value = None
	_shell_sample_cache_key = None
	_shell_sample_cache_value = {}
	_last_stats = {}


def configure_compile(
	*,
	shell_normal: bool = False,
	backend: str | None = None,
	mode: str | None = None,
	dynamic: bool = False,
	fullgraph: bool = False,
) -> None:
	"""Configure opt-in torch.compile use for isolated cylinder loss kernels."""
	global _compile_shell_normal, _compile_shell_normal_backend, _compile_shell_normal_mode
	global _compile_shell_normal_dynamic, _compile_shell_normal_fullgraph, _compile_shell_normal_key
	global _compiled_shell_normal_core, _compile_shell_normal_disabled_reason
	backend = str(backend).strip() if backend is not None and str(backend).strip() else None
	mode = str(mode).strip() if mode is not None and str(mode).strip() else None
	key = (bool(shell_normal), backend, mode, bool(dynamic), bool(fullgraph))
	if key != _compile_shell_normal_key:
		_compiled_shell_normal_core = None
		_compile_shell_normal_disabled_reason = None
		_compile_shell_normal_key = key
	_compile_shell_normal = bool(shell_normal)
	_compile_shell_normal_backend = backend
	_compile_shell_normal_mode = mode
	_compile_shell_normal_dynamic = bool(dynamic)
	_compile_shell_normal_fullgraph = bool(fullgraph)


def _active_terms(weights: dict[str, float]) -> dict[str, float]:
	return {
		name: float(weights.get(name, 0.0))
		for name in (
			"cyl_normal",
			"cyl_center",
			"cyl_smooth",
			"cyl_z_smooth",
			"cyl_z_center",
			"cyl_step",
			"cyl_radial_mean",
			"cyl_bend",
			"cyl_spheres",
			"cyl_spheres_outside",
			"cyl_conn_mesh",
			"cyl_conn_gt",
			"cyl_base_mesh",
			"cyl_base_gt",
		)
		if float(weights.get(name, 0.0)) != 0.0 and name in _candidate_terms
	}


def _candidate_totals(weights: dict[str, float]) -> tuple[torch.Tensor | None, dict[str, float]]:
	active = _active_terms(weights)
	if not active:
		return None, {}
	ref = next(iter(_candidate_terms.values()))
	total = torch.zeros_like(ref)
	for name, weight in active.items():
		total = total + weight * _candidate_terms[name]
	return total, active


def top_candidates(weights: dict[str, float], *, limit: int) -> list[dict[str, float | int]]:
	total, active = _candidate_totals(weights)
	if total is None:
		return []
	valid = torch.isfinite(total)
	if not bool(valid.any().detach().cpu()):
		return []
	total_valid = torch.where(valid, total, torch.full_like(total, float("inf")))
	k = min(max(0, int(limit)), int(valid.sum().detach().cpu()))
	if k == 0:
		return []
	values, indices = torch.topk(total_valid, k=k, largest=False)
	out: list[dict[str, float | int]] = []
	for rank, (value_t, idx_t) in enumerate(zip(values, indices, strict=True), start=1):
		idx = int(idx_t.detach().cpu())
		row: dict[str, float | int] = {
			"rank": rank,
			"idx": idx,
			"cyl_min": float(value_t.detach().cpu()),
		}
		for name in active:
			row[name] = float(_candidate_terms[name][idx_t].detach().cpu())
		for name, stat in _candidate_normal_stats.items():
			row[name] = float(stat[idx_t].detach().cpu())
		out.append(row)
	return out


def top_candidate_indices(weights: dict[str, float], *, limit: int) -> list[int]:
	return [int(row["idx"]) for row in top_candidates(weights, limit=limit)]


def display_stats(weights: dict[str, float]) -> tuple[int | None, float | None, dict[str, float]]:
	total, active = _candidate_totals(weights)
	if total is None:
		return None, None, {}
	valid = torch.isfinite(total)
	if not bool(valid.any().detach().cpu()):
		return None, None, {"cyl_min": float("inf")}
	total_valid = torch.where(valid, total, torch.full_like(total, float("inf")))
	best_t = torch.argmin(total_valid)
	best = int(best_t.detach().cpu())
	out = {"cyl_min": float(total_valid[best_t].detach().cpu())}
	for name in active:
		out[name] = float(_candidate_terms[name][best_t].detach().cpu())
	for name, values in _candidate_normal_stats.items():
		out[name] = float(values[best_t].detach().cpu())
	max_values = _candidate_normal_stats.get("cyl_nerr_max")
	if max_values is not None:
		max_valid = torch.isfinite(max_values)
		if bool(max_valid.any().detach().cpu()):
			out["cyl_nerr_max_min"] = float(max_values[max_valid].amin().detach().cpu())
		else:
			out["cyl_nerr_max_min"] = float("inf")
	return best, out["cyl_min"], out


def cyl_normal_prefetch_items_for_result(*, res: fit_model.FitResult3D) -> dict[str, torch.Tensor]:
	if res.cyl_xyz is None or res.cyl_count <= 0:
		return {}
	if bool(getattr(res, "cyl_shell_mode", False)):
		geom = _shell_geometry(res=res, factor=4)
		if geom is None:
			return {}
		pts = geom["xyz"].unsqueeze(0).detach()
		base = geom.get("base")
		if base is not None and bool(geom.get("has_base", False)):
			pts = torch.cat([pts.reshape(1, 1, -1, 3), base.unsqueeze(0).detach().reshape(1, 1, -1, 3)], dim=2)
	else:
		pts = res.cyl_xyz.detach()
	return {"grad_mag": pts, "nx": pts, "ny": pts}


def _zero_loss(res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	zero = res.xyz_lr.sum() * 0.0
	return zero, (zero.view(1, 1, 1, 1),), (zero.view(1, 1, 1, 1),)


def _shell_step(res: fit_model.FitResult3D) -> float:
	return float(getattr(res, "cyl_shell_step", 500.0))


def _shell_xyz(res: fit_model.FitResult3D) -> torch.Tensor | None:
	if not bool(getattr(res, "cyl_shell_mode", False)) or res.cyl_xyz is None:
		return None
	if res.cyl_xyz.ndim == 4:
		return res.cyl_xyz[0]
	if res.cyl_xyz.ndim == 3:
		return res.cyl_xyz
	return None


def _shell_spacing_scales(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	with torch.no_grad():
		if int(xyz.shape[0]) > 1:
			h_scale = (xyz[1:] - xyz[:-1]).norm(dim=-1).mean().clamp(min=1.0)
		else:
			h_scale = xyz.new_tensor(1.0)
		w_scale = (torch.roll(xyz, shifts=-1, dims=1) - xyz).norm(dim=-1).mean().clamp(min=1.0)
	return h_scale, w_scale


def _periodic_interp_width(xyz: torch.Tensor, target_w: int) -> torch.Tensor:
	target_w = max(3, int(target_w))
	src_w = int(xyz.shape[1])
	if src_w == target_w:
		return xyz
	device = xyz.device
	dtype = xyz.dtype
	phase = torch.arange(target_w, device=device, dtype=dtype) * (float(src_w) / float(target_w))
	i0 = torch.floor(phase).to(dtype=torch.long)
	frac = (phase - i0.to(dtype=dtype)).view(1, target_w, 1)
	i0 = torch.remainder(i0, src_w)
	i1 = torch.remainder(i0 + 1, src_w)
	p0 = xyz.index_select(1, i0)
	p1 = xyz.index_select(1, i1)
	return p0 + frac * (p1 - p0)


def _shell_supersampled_field(field: torch.Tensor, *, factor: int = 4) -> torch.Tensor:
	factor = max(1, int(factor))
	if factor <= 1:
		return field
	if field.ndim == 2:
		field_in = field.unsqueeze(-1)
		squeeze = True
	else:
		field_in = field
		squeeze = False
	H, W = int(field_in.shape[0]), int(field_in.shape[1])
	target_h = max(2, (H - 1) * factor + 1)
	t = field_in.permute(2, 1, 0).unsqueeze(0)
	t = F.interpolate(t, size=(W, target_h), mode="bilinear", align_corners=True)
	field_h = t.squeeze(0).permute(2, 1, 0).contiguous()
	out = _periodic_interp_width(field_h, W * factor)
	return out.squeeze(-1) if squeeze else out


def _interp_width_at_offsets(field: torch.Tensor, offsets: torch.Tensor, *, offset_scale: float = 1.0) -> torch.Tensor:
	if field.ndim == 2:
		field_in = field.unsqueeze(-1)
		squeeze = True
	else:
		field_in = field
		squeeze = False
	H = int(field_in.shape[0])
	W = int(field_in.shape[1])
	C = int(field_in.shape[2])
	if W <= 1:
		out = field_in.expand(H, max(1, W), C)
		return out.squeeze(-1) if squeeze else out
	device = field_in.device
	dtype = field_in.dtype
	base_w = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
	phase = base_w + offsets.to(device=device, dtype=dtype) * float(offset_scale)
	i0_floor = torch.floor(phase)
	frac = (phase - i0_floor).unsqueeze(-1)
	i0 = torch.remainder(i0_floor.to(dtype=torch.long), W)
	i1 = torch.remainder(i0 + 1, W)
	i0e = i0.unsqueeze(-1).expand(H, W, C)
	i1e = i1.unsqueeze(-1).expand(H, W, C)
	p0 = torch.gather(field_in, 1, i0e)
	p1 = torch.gather(field_in, 1, i1e)
	out = p0 + frac * (p1 - p0)
	return out.squeeze(-1) if squeeze else out


def _shell_supersampled_xyz(xyz: torch.Tensor, *, factor: int = 4) -> torch.Tensor:
	factor = max(1, int(factor))
	if factor <= 1:
		return xyz
	H, W = int(xyz.shape[0]), int(xyz.shape[1])
	target_h = max(2, (H - 1) * factor + 1)
	t = xyz.permute(2, 1, 0).unsqueeze(0)  # (1, 3, W, H)
	t = F.interpolate(t, size=(W, target_h), mode="bilinear", align_corners=True)
	xyz_h = t.squeeze(0).permute(2, 1, 0).contiguous()
	return _periodic_interp_width(xyz_h, W * factor)


def _shell_geometry_tensors(
	xyz_lr: torch.Tensor,
	base: torch.Tensor | None,
	offsets: torch.Tensor | None,
	delta_xyz: torch.Tensor | None,
	*,
	factor: int = 4,
	has_base: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
	if base is None or offsets is None or delta_xyz is None:
		xyz = _shell_supersampled_xyz(xyz_lr, factor=factor)
		return xyz, None, None

	base_s = _shell_supersampled_field(base.to(device=xyz_lr.device, dtype=xyz_lr.dtype), factor=factor)
	offsets_s = _shell_supersampled_field(offsets.to(device=xyz_lr.device, dtype=xyz_lr.dtype), factor=factor)
	delta_s = _shell_supersampled_field(delta_xyz.to(device=xyz_lr.device, dtype=xyz_lr.dtype), factor=factor)
	if bool(has_base):
		base_conn = _interp_width_at_offsets(base_s, offsets_s, offset_scale=float(factor))
	else:
		base_conn = base_s
	xyz = base_conn + delta_s
	return xyz, base_conn, xyz - base_conn


def _shell_fields_for_result(
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool] | None:
	xyz_lr = _shell_xyz(res)
	if xyz_lr is None:
		return None
	return (
		xyz_lr,
		getattr(res, "cyl_shell_base_xyz", None),
		getattr(res, "cyl_shell_w_offsets", None),
		getattr(res, "cyl_shell_delta_xyz", None),
		int(getattr(res, "cyl_shell_index", 0)) > 0,
	)


def _shell_geometry(*, res: fit_model.FitResult3D, factor: int = 4) -> dict[str, torch.Tensor | bool | None] | None:
	global _shell_geometry_cache_key, _shell_geometry_cache_value
	cache_key = (id(res), int(factor), bool(torch.is_grad_enabled()))
	if _shell_geometry_cache_key == cache_key and _shell_geometry_cache_value is not None:
		return _shell_geometry_cache_value
	fields = _shell_fields_for_result(res)
	if fields is None:
		return None
	xyz_lr, base, offsets, delta_xyz, has_base = fields
	xyz, base_conn, conn = _shell_geometry_tensors(
		xyz_lr,
		base,
		offsets,
		delta_xyz,
		factor=factor,
		has_base=has_base,
	)
	_shell_geometry_cache_key = cache_key
	_shell_geometry_cache_value = {"xyz": xyz, "base": base_conn, "conn": conn, "has_base": has_base}
	return _shell_geometry_cache_value


def _unit_normals_for_shell_xyz(xyz: torch.Tensor) -> torch.Tensor:
	dh = torch.zeros_like(xyz)
	dh[1:-1] = xyz[2:] - xyz[:-2]
	dh[0] = xyz[1] - xyz[0]
	dh[-1] = xyz[-1] - xyz[-2]
	dw = torch.roll(xyz, shifts=-1, dims=1) - torch.roll(xyz, shifts=1, dims=1)
	n = torch.cross(dh, dw, dim=-1)
	return F.normalize(n, dim=-1, eps=1.0e-8)


def _register_shell_term(
	name: str,
	lm: torch.Tensor,
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	mask = torch.ones_like(lm, dtype=lm.dtype)
	err = lm.mean()
	loss = _register_candidate_term(name, err.view(1))
	lm_out = lm
	mask_out = mask
	while lm_out.ndim < 2:
		lm_out = lm_out.unsqueeze(0)
		mask_out = mask_out.unsqueeze(0)
	return loss, (lm_out.unsqueeze(0).unsqueeze(0),), (mask_out.unsqueeze(0).unsqueeze(0),)


def _register_shell_masked_term(
	name: str,
	lm: torch.Tensor,
	mask: torch.Tensor,
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	mask = mask.to(device=lm.device, dtype=lm.dtype)
	wsum = mask.sum()
	err = (lm * mask).sum() / wsum.clamp(min=1.0)
	err = torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))
	loss = _register_candidate_term(name, err.view(1))
	lm_out = lm
	mask_out = mask
	while lm_out.ndim < 2:
		lm_out = lm_out.unsqueeze(0)
		mask_out = mask_out.unsqueeze(0)
	return loss, (lm_out.unsqueeze(0).unsqueeze(0),), (mask_out.unsqueeze(0).unsqueeze(0),)


def _normal_align_lm(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	a = F.normalize(a, dim=-1, eps=1.0e-8)
	b = F.normalize(b, dim=-1, eps=1.0e-8)
	dot_abs = (a * b).sum(dim=-1).clamp(min=-1.0, max=1.0).abs()
	return 1.0 - dot_abs * dot_abs, dot_abs


def _shell_normal_geometry_core(
	xyz_lr: torch.Tensor,
	base: torch.Tensor | None,
	offsets: torch.Tensor | None,
	delta_xyz: torch.Tensor | None,
	*,
	factor: int = 4,
	has_base: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
	xyz, base_conn, _conn = _shell_geometry_tensors(
		xyz_lr,
		base,
		offsets,
		delta_xyz,
		factor=factor,
		has_base=has_base,
	)
	return xyz, base_conn


def _shell_normal_compute_core(
	xyz_lr: torch.Tensor,
	base: torch.Tensor | None,
	offsets: torch.Tensor | None,
	delta_xyz: torch.Tensor | None,
	target: torch.Tensor,
	mask: torch.Tensor,
	factor: int = 4,
	has_base: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	xyz, _base_conn = _shell_normal_geometry_core(
		xyz_lr,
		base,
		offsets,
		delta_xyz,
		factor=factor,
		has_base=has_base,
	)
	normal = _unit_normals_for_shell_xyz(xyz)
	normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
	dot = (normal * target).sum(dim=-1).clamp(min=-1.0, max=1.0)
	dot_abs = dot.abs()
	lm = 1.0 - dot_abs * dot_abs
	wsum = mask.sum()
	err = (lm * mask).sum() / wsum.clamp(min=1.0)
	err = torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))
	return err, lm, dot_abs


def _compiled_shell_normal_compute_core():
	global _compiled_shell_normal_core, _compile_shell_normal, _compile_shell_normal_disabled_reason
	if not _compile_shell_normal:
		return _shell_normal_compute_core
	if _compile_shell_normal_disabled_reason is not None:
		return _shell_normal_compute_core
	if not hasattr(torch, "compile"):
		_compile_shell_normal_disabled_reason = "torch.compile is unavailable"
		print(f"[opt_loss_cyl] compile_cyl_normal disabled: {_compile_shell_normal_disabled_reason}", flush=True)
		return _shell_normal_compute_core
	if _compiled_shell_normal_core is None:
		kwargs = {
			"dynamic": _compile_shell_normal_dynamic,
			"fullgraph": _compile_shell_normal_fullgraph,
		}
		if _compile_shell_normal_backend is not None:
			kwargs["backend"] = _compile_shell_normal_backend
		if _compile_shell_normal_mode is not None:
			kwargs["mode"] = _compile_shell_normal_mode
		try:
			_compiled_shell_normal_core = torch.compile(_shell_normal_compute_core, **kwargs)
		except Exception as exc:
			_compile_shell_normal_disabled_reason = f"{type(exc).__name__}: {exc}"
			print(f"[opt_loss_cyl] compile_cyl_normal disabled: {_compile_shell_normal_disabled_reason}", flush=True)
			return _shell_normal_compute_core
	return _compiled_shell_normal_core


def _run_shell_normal_compute_core(
	xyz_lr: torch.Tensor,
	base: torch.Tensor | None,
	offsets: torch.Tensor | None,
	delta_xyz: torch.Tensor | None,
	target: torch.Tensor,
	mask: torch.Tensor,
	*,
	factor: int = 4,
	has_base: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	global _compile_shell_normal_disabled_reason
	fn = _compiled_shell_normal_compute_core()
	if fn is _shell_normal_compute_core:
		return fn(xyz_lr, base, offsets, delta_xyz, target, mask, factor, has_base)
	try:
		return fn(xyz_lr, base, offsets, delta_xyz, target, mask, factor, has_base)
	except Exception as exc:
		_compile_shell_normal_disabled_reason = f"{type(exc).__name__}: {exc}"
		print(f"[opt_loss_cyl] compile_cyl_normal disabled after failure: {_compile_shell_normal_disabled_reason}", flush=True)
		return _shell_normal_compute_core(xyz_lr, base, offsets, delta_xyz, target, mask, factor, has_base)


def _shell_normal_sample_xyz(
	xyz_lr: torch.Tensor,
	base: torch.Tensor | None,
	offsets: torch.Tensor | None,
	delta_xyz: torch.Tensor | None,
	*,
	factor: int = 4,
	has_base: bool = False,
) -> torch.Tensor:
	with torch.no_grad():
		xyz, _base_conn = _shell_normal_geometry_core(
			xyz_lr,
			base,
			offsets,
			delta_xyz,
			factor=factor,
			has_base=has_base,
		)
	return xyz


def _sample_shell_gt(
	*,
	res: fit_model.FitResult3D,
	xyz: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
	global _shell_sample_cache_key, _shell_sample_cache_value
	res_key = id(res)
	if _shell_sample_cache_key != res_key:
		_shell_sample_cache_key = res_key
		_shell_sample_cache_value = {}
	cache_key = (int(xyz.data_ptr()), tuple(int(v) for v in xyz.shape))
	if cache_key in _shell_sample_cache_value:
		return _shell_sample_cache_value[cache_key]
	sampled = res.data.grid_sample_fullres(
		xyz.unsqueeze(0).detach(),
		channels={"grad_mag", "nx", "ny"},
	)
	target = sampled.normal_3d
	grad_mag = sampled.grad_mag
	if target is None or grad_mag is None:
		_shell_sample_cache_value[cache_key] = (None, None)
		return _shell_sample_cache_value[cache_key]
	_shell_sample_cache_value[cache_key] = (
		target.squeeze(0),
		(grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=xyz.dtype),
	)
	return _shell_sample_cache_value[cache_key]


def _sample_gt(res: fit_model.FitResult3D) -> tuple[torch.Tensor | None, torch.Tensor | None]:
	global _sample_cache_key, _sample_cache_value
	key = id(res)
	if _sample_cache_key == key and _sample_cache_value is not None:
		return _sample_cache_value
	if res.cyl_xyz is None or res.cyl_normals is None or res.cyl_count <= 0:
		return None, None
	sampled = res.data.grid_sample_fullres(
		res.cyl_xyz.detach(),
		diff=False,
		channels={"grad_mag", "nx", "ny"},
	)
	_sample_cache_key = key
	_sample_cache_value = (sampled.normal_3d, sampled.grad_mag)
	return _sample_cache_value


def _cyl_dims(res: fit_model.FitResult3D) -> tuple[int, int, int, int, int]:
	N = int(res.cyl_count)
	S = max(1, int(getattr(res, "cyl_sample_count", 1)))
	D = max(1, int(res.xyz_lr.shape[0]))
	H = int(res.cyl_xyz.shape[1])
	W = int(res.cyl_xyz.shape[2])
	return N, S, D, H, W


def _candidate_mean(lm: torch.Tensor, mask: torch.Tensor, *, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	N, S, D, H, W = _cyl_dims(res)
	lm_c = lm.reshape(N, S, D, H, W)
	mask_c = mask.reshape(N, S, D, H, W) * _mesh_distance_falloff(res=res, dtype=lm.dtype, device=lm.device)
	wsum = mask_c.sum(dim=(1, 2, 3, 4))
	err = (lm_c * mask_c).sum(dim=(1, 2, 3, 4)) / wsum.clamp(min=1.0)
	err = torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))
	return err, lm_c, mask_c


def _mesh_distance_falloff(*, res: fit_model.FitResult3D, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
	N, S, D, H, W = _cyl_dims(res)
	with torch.no_grad():
		h_idx = torch.arange(H, device=device, dtype=dtype)
		w_idx = torch.arange(W, device=device, dtype=dtype)
		h_center = float(H // 2)
		w_center = float(W // 2)
		h_edge = max(h_center, float(H - 1) - h_center, 1.0)
		w_edge = max(w_center, float(W - 1) - w_center, 1.0)
		h_dist = ((h_idx - h_center).abs() / h_edge).clamp(max=1.0)
		w_dist = ((w_idx - w_center).abs() / w_edge).clamp(max=1.0)
		grid_dist = torch.maximum(h_dist.view(1, H, 1), w_dist.view(1, 1, W))
		floor = 0.1
		falloff = floor + (1.0 - floor) * (1.0 - grid_dist).pow(2.0)
		return falloff.view(1, 1, 1, H, W).expand(N, S, D, H, W)


def _register_normal_angle_stats(dot: torch.Tensor, mask: torch.Tensor, *, res: fit_model.FitResult3D) -> None:
	N, S, D, H, W = _cyl_dims(res)
	with torch.no_grad():
		dot_c = dot.detach().reshape(N, S, D, H, W).clamp(min=-1.0, max=1.0)
		mask_f = mask.detach().reshape(N, S, D, H, W) * _mesh_distance_falloff(res=res, dtype=dot_c.dtype, device=dot_c.device)
		mask_c = mask_f > 0.0
		angle = torch.acos(dot_c) * (180.0 / math.pi)
		wsum = mask_f.sum(dim=(1, 2, 3, 4))
		avg = (angle * mask_f).sum(dim=(1, 2, 3, 4)) / wsum.clamp(min=1.0)
		avg = torch.where(wsum > 0.0, avg, torch.full_like(avg, float("inf")))
		inf = torch.full_like(angle, float("inf"))
		ninf = torch.full_like(angle, float("-inf"))
		min_v = torch.where(mask_c, angle, inf).amin(dim=(1, 2, 3, 4))
		max_v = torch.where(mask_c, angle, ninf).amax(dim=(1, 2, 3, 4))
		max_v = torch.where(wsum > 0.0, max_v, torch.full_like(max_v, float("inf")))
		_candidate_normal_stats["cyl_nerr_avg"] = avg.detach()
		_candidate_normal_stats["cyl_nerr_min"] = min_v.detach()
		_candidate_normal_stats["cyl_nerr_max"] = max_v.detach()


def _register_candidate_term(name: str, err: torch.Tensor) -> torch.Tensor:
	_candidate_terms[name] = err.detach()
	valid = torch.isfinite(err)
	if bool(valid.any().detach().cpu()):
		return err[valid].mean()
	return err.new_zeros(()) + err[~valid].new_zeros(()).sum() * 0.0


def _register_shell_normal_angle_stats(dot_abs: torch.Tensor, mask: torch.Tensor) -> None:
	with torch.no_grad():
		dot_abs = dot_abs.detach().clamp(min=0.0, max=1.0)
		mask_f = mask.detach().to(dtype=dot_abs.dtype)
		mask_c = mask_f > 0.0
		angle = torch.acos(dot_abs) * (180.0 / math.pi)
		wsum = mask_f.sum()
		avg = (angle * mask_f).sum() / wsum.clamp(min=1.0)
		avg = torch.where(wsum > 0.0, avg, torch.full_like(avg, float("inf")))
		inf = torch.full_like(angle, float("inf"))
		ninf = torch.full_like(angle, float("-inf"))
		min_v = torch.where(mask_c, angle, inf).amin()
		max_v = torch.where(mask_c, angle, ninf).amax()
		max_v = torch.where(wsum > 0.0, max_v, torch.full_like(max_v, float("inf")))
		_candidate_normal_stats["cyl_nerr_avg"] = avg.detach().view(1)
		_candidate_normal_stats["cyl_nerr_min"] = min_v.detach().view(1)
		_candidate_normal_stats["cyl_nerr_max"] = max_v.detach().view(1)


def _unit_xy(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	length = v.norm(dim=-1).clamp(min=1.0e-8)
	return v / length.unsqueeze(-1), length


def _signed_xy_normal_alignment(
	*,
	res: fit_model.FitResult3D,
	target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	cyl_xy, _cyl_xy_len = _unit_xy(res.cyl_normals[..., :2])
	with torch.no_grad():
		xyz = res.cyl_xyz.detach()
		target_xy_raw = target[..., :2].detach()
		target_xy, target_xy_len = _unit_xy(target_xy_raw)
		umb_xy = res.data.umbilicus_xy_at_z(xyz[..., 2])
		radial_xy, radial_len = _unit_xy(xyz[..., :2] - umb_xy)
		target_dot_radial = (target_xy * radial_xy).sum(dim=-1)
		flip = target_dot_radial < 0.0
		target_oriented = torch.where(flip.unsqueeze(-1), -target_xy, target_xy)
		radial_weight = (target_oriented * radial_xy).sum(dim=-1).clamp(min=0.0, max=1.0)
		in_plane_weight = target_xy_raw.norm(dim=-1).clamp(min=0.0, max=1.0)
		valid = (target_xy_len > 1.0e-7) & (radial_len > 1.0e-7)
		weight = radial_weight * in_plane_weight * valid.to(dtype=target.dtype)
	dot = (cyl_xy * target_oriented).sum(dim=-1).clamp(min=-1.0, max=1.0)
	return dot, weight


def cyl_normal_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""GT normal alignment for analytic cylinders and optimized shells.

	GT normal samples are intentionally non-differentiable; gradients flow only
	through the cylinder/shell geometry. Analytic candidates orient GT normals
	away from the umbilicus in the local xy slice before comparison.
	"""
	if res.cyl_xyz is None or res.cyl_count <= 0:
		return _zero_loss(res)

	if bool(getattr(res, "cyl_shell_mode", False)):
		fields = _shell_fields_for_result(res)
		if fields is None:
			return _zero_loss(res)
		xyz_lr, base, offsets, delta_xyz, has_base = fields
		xyz = _shell_normal_sample_xyz(
			xyz_lr,
			base,
			offsets,
			delta_xyz,
			factor=4,
			has_base=has_base,
		)
		target, mask = _sample_shell_gt(res=res, xyz=xyz)
		if target is None or mask is None:
			return _zero_loss(res)
		err, lm, dot_abs = _run_shell_normal_compute_core(
			xyz_lr,
			base,
			offsets,
			delta_xyz,
			target,
			mask,
			factor=4,
			has_base=has_base,
		)
		_register_shell_normal_angle_stats(dot_abs, mask)
		loss = _register_candidate_term("cyl_normal", err.view(1))
		return loss, (lm.unsqueeze(0).unsqueeze(0),), (mask.unsqueeze(0).unsqueeze(0),)

	if res.cyl_normals is None:
		return _zero_loss(res)

	target, grad_mag = _sample_gt(res)
	if target is None or grad_mag is None:
		return _zero_loss(res)

	dot, normal_weight = _signed_xy_normal_alignment(res=res, target=target)
	lm = 1.0 - dot
	mask = (grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype) * normal_weight.to(dtype=lm.dtype)

	_register_normal_angle_stats(dot, mask, res=res)
	err, lm_c, mask_c = _candidate_mean(lm, mask, res=res)
	loss = _register_candidate_term("cyl_normal", err)
	return loss, (lm_c,), (mask_c,)


def cyl_center_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Distance from sampled GT normal rays to each candidate cylinder axis."""
	if res.cyl_xyz is None or res.cyl_centers is None or res.cyl_axes is None or res.cyl_count <= 0:
		return _zero_loss(res)

	target, grad_mag = _sample_gt(res)
	if target is None or grad_mag is None:
		return _zero_loss(res)

	N, S, D, H, W = _cyl_dims(res)
	xyz = res.cyl_xyz.reshape(N, S, D, H, W, 3)
	ray_dir = target.reshape(N, S, D, H, W, 3).detach()
	axis = res.cyl_axes.view(N, 1, 1, 1, 1, 3)
	center = res.cyl_centers.view(N, 1, 1, 1, 1, 3)
	delta = center - xyz
	cross_dir = torch.cross(ray_dir, axis.expand_as(ray_dir), dim=-1)
	cross_sq = (cross_dir * cross_dir).sum(dim=-1)
	line_num = ((delta * cross_dir).sum(dim=-1) ** 2)
	line_dist_sq = line_num / cross_sq.clamp(min=1.0e-8)
	parallel = torch.cross(delta, axis.expand_as(delta), dim=-1)
	parallel_dist_sq = (parallel * parallel).sum(dim=-1)
	dist_sq = torch.where(cross_sq > 1.0e-6, line_dist_sq, parallel_dist_sq)
	axis_delta = delta - (delta * axis).sum(dim=-1, keepdim=True) * axis
	radius_sq = (axis_delta * axis_delta).sum(dim=-1).clamp(min=1.0)
	lm = dist_sq / radius_sq
	mask = (grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype).reshape(N, S, D, H, W)
	mask = mask * ((ray_dir * ray_dir).sum(dim=-1) > 1.0e-6).to(dtype=lm.dtype)

	err, lm_c, mask_c = _candidate_mean(lm, mask, res=res)
	loss = _register_candidate_term("cyl_center", err)
	return loss, (lm_c,), (mask_c,)


def cyl_smooth_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Shell Laplacian loss: each point stays near the average of its neighbors."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)
	h_scale, w_scale = _shell_spacing_scales(xyz)
	terms: list[torch.Tensor] = []
	if int(xyz.shape[0]) > 2:
		h_mid = xyz[1:-1]
		h_avg = 0.5 * (xyz[:-2] + xyz[2:])
		terms.append(((h_mid - h_avg).norm(dim=-1) / h_scale).square().reshape(-1))
	w_avg = 0.5 * (torch.roll(xyz, shifts=1, dims=1) + torch.roll(xyz, shifts=-1, dims=1))
	terms.append(((xyz - w_avg).norm(dim=-1) / w_scale).square().reshape(-1))
	lm = torch.cat(terms, dim=0)
	return _register_shell_term("cyl_smooth", lm, res=res)


def cyl_z_smooth_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Shell Laplacian loss along the height/z index only."""
	xyz = _shell_xyz(res)
	if xyz is None or int(xyz.shape[0]) <= 2:
		return _zero_loss(res)
	h_scale, _w_scale = _shell_spacing_scales(xyz)
	h_mid = xyz[1:-1]
	h_avg = 0.5 * (xyz[:-2] + xyz[2:])
	lm = ((h_mid - h_avg).norm(dim=-1) / h_scale).square()
	return _register_shell_term("cyl_z_smooth", lm, res=res)


def cyl_z_center_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Keep the shell's average z near the prepared init z center."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)
	target = xyz.new_tensor(float(getattr(res, "cyl_z_center_target", getattr(res, "cyl_seed_z", 0.0))))
	scale = xyz.new_tensor(max(1.0, float(getattr(res, "cyl_shell_height_step", 1.0))))
	lm = ((xyz[..., 2].mean() - target) / scale).square().view(1)
	return _register_shell_term("cyl_z_center", lm, res=res)


def cyl_step_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Match shell H/W edges and wrapped quad diagonals to the current target."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)
	w_step = torch.roll(xyz, shifts=-1, dims=1) - xyz
	w_len = w_step.norm(dim=-1)
	h_len = (xyz[1:] - xyz[:-1]).norm(dim=-1) if int(xyz.shape[0]) > 1 else None
	w_target_value = float(getattr(res, "cyl_shell_width_step", 0.0))
	if w_target_value > 0.0:
		w_target = w_len.new_tensor(w_target_value).clamp(min=1.0e-6)
	else:
		target_terms = [w_len.reshape(-1)]
		if h_len is not None:
			target_terms.append(h_len.reshape(-1))
		w_target = torch.cat(target_terms, dim=0).mean().clamp(min=1.0e-6)
	h_target_value = float(getattr(res, "cyl_shell_height_step", 0.0))
	if h_target_value > 0.0:
		h_target = w_len.new_tensor(h_target_value).clamp(min=1.0e-6)
	elif h_len is not None:
		h_target = h_len.mean().clamp(min=1.0e-6)
	else:
		h_target = w_target
	terms = [((w_len - w_target) / w_target).square().reshape(-1)]
	if h_len is not None:
		terms.append(((h_len - h_target) / h_target).square().reshape(-1))
		p00 = xyz[:-1]
		p10 = xyz[1:]
		p01 = torch.roll(p00, shifts=-1, dims=1)
		p11 = torch.roll(p10, shifts=-1, dims=1)
		diag_target = torch.sqrt(h_target.square() + w_target.square()).clamp(min=1.0e-6)
		diag_fwd = (p11 - p00).norm(dim=-1)
		diag_back = (p10 - p01).norm(dim=-1)
		terms.append(((diag_fwd - diag_target) / diag_target).square().reshape(-1))
		terms.append(((diag_back - diag_target) / diag_target).square().reshape(-1))
	lm = torch.cat(terms, dim=0)
	return _register_shell_term("cyl_step", lm, res=res)


def cyl_radial_mean_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Keep the initial shell near its configured umbilicus radius."""
	delta_xyz = getattr(res, "cyl_shell_delta_xyz", None)
	if delta_xyz is None:
		return _zero_loss(res)
	if bool(getattr(res, "cyl_shell_mode", False)) and int(getattr(res, "cyl_shell_index", 0)) > 0:
		return _zero_loss(res)
	length = delta_xyz.norm(dim=-1)
	target = length.new_tensor(max(1.0, float(getattr(res, "cyl_shell_step", 1.0))))
	mean_len = length.mean()
	lm = ((mean_len - target) / target).square().view(1)
	return _register_shell_term("cyl_radial_mean", lm, res=res)


def _umbilicus_sphere_distances(res: fit_model.FitResult3D, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	z = xyz[..., 2].detach()
	z0 = float(z.amin().detach().cpu())
	z1 = float(z.amax().detach().cpu())
	if not math.isfinite(z0) or not math.isfinite(z1):
		raise ValueError("invalid cylinder shell z range for sphere loss")
	if z1 < z0:
		z0, z1 = z1, z0
	H = int(xyz.shape[0])
	sample_count = max(8, min(257, max(1, H - 1) * 4 + 1))
	z_samples = torch.linspace(z0, z1, sample_count, device=xyz.device, dtype=xyz.dtype)
	umb_xy = res.data.umbilicus_xy_at_z(z_samples).to(device=xyz.device, dtype=xyz.dtype)
	centers = torch.cat([umb_xy, z_samples[:, None]], dim=-1)
	dist = torch.cdist(xyz.reshape(-1, 3), centers).amin(dim=1)
	target_radius = getattr(res, "cyl_shell_sphere_radius", getattr(res, "cyl_shell_step", 1.0))
	target = dist.new_tensor(max(1.0, float(target_radius)))
	return dist, target


def cyl_spheres_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Keep the shell near the surface of the closest umbilicus-centered sphere."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)
	dist, target = _umbilicus_sphere_distances(res, xyz)
	lm = ((dist - target) / target).square()
	return _register_shell_term("cyl_spheres", lm, res=res)


def cyl_spheres_outside_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Prevent the shell from entering any sampled umbilicus-centered sphere."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)
	dist, target = _umbilicus_sphere_distances(res, xyz)
	lm = ((target - dist).clamp(min=0.0) / target).square()
	return _register_shell_term("cyl_spheres_outside", lm, res=res)


def cyl_bend_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize sharp shell bends using the same angle threshold as mesh bend."""
	xyz = _shell_xyz(res)
	if xyz is None:
		return _zero_loss(res)

	def _bend_penalty(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
		e1n = F.normalize(e1, dim=-1, eps=1.0e-12)
		e2n = F.normalize(e2, dim=-1, eps=1.0e-12)
		cos_angle = (-e1n * e2n).sum(dim=-1)
		excess = (cos_angle - _BEND_COS_THRESHOLD).clamp(min=0.0)
		return excess * excess

	terms: list[torch.Tensor] = []
	if int(xyz.shape[0]) > 2:
		e1_h = xyz[1:-1] - xyz[:-2]
		e2_h = xyz[2:] - xyz[1:-1]
		terms.append(_bend_penalty(e1_h, e2_h).reshape(-1))
	if int(xyz.shape[1]) > 2:
		e1_w = xyz - torch.roll(xyz, shifts=1, dims=1)
		e2_w = torch.roll(xyz, shifts=-1, dims=1) - xyz
		terms.append(_bend_penalty(e1_w, e2_w).reshape(-1))
	if int(xyz.shape[0]) > 2 and int(xyz.shape[1]) > 2:
		mid = xyz[1:-1]
		e1_d0 = mid - torch.roll(xyz[:-2], shifts=1, dims=1)
		e2_d0 = torch.roll(xyz[2:], shifts=-1, dims=1) - mid
		e1_d1 = mid - torch.roll(xyz[:-2], shifts=-1, dims=1)
		e2_d1 = torch.roll(xyz[2:], shifts=1, dims=1) - mid
		terms.append(_bend_penalty(e1_d0, e2_d0).reshape(-1))
		terms.append(_bend_penalty(e1_d1, e2_d1).reshape(-1))
	if not terms:
		return _zero_loss(res)
	lm = torch.cat(terms, dim=0)
	return _register_shell_term("cyl_bend", lm, res=res)


def _shell_offset_geometry_for_loss(res: fit_model.FitResult3D) -> dict[str, torch.Tensor | bool] | None:
	geom = _shell_geometry(res=res, factor=4)
	if geom is None or not bool(geom.get("has_base", False)):
		return None
	if geom.get("base") is None or geom.get("conn") is None:
		return None
	return geom


def cyl_conn_mesh_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Align the base-to-shell connection vector with both shells' mesh normals."""
	geom = _shell_offset_geometry_for_loss(res)
	if geom is None:
		return _zero_loss(res)
	xyz = geom["xyz"]
	base = geom["base"]
	conn = geom["conn"]
	if not isinstance(xyz, torch.Tensor) or not isinstance(base, torch.Tensor) or not isinstance(conn, torch.Tensor):
		return _zero_loss(res)
	conn_n = F.normalize(conn, dim=-1, eps=1.0e-8)
	new_mesh_n = _unit_normals_for_shell_xyz(xyz)
	base_mesh_n = _unit_normals_for_shell_xyz(base)
	lm_new, _dot_new = _normal_align_lm(conn_n, new_mesh_n)
	lm_base, _dot_base = _normal_align_lm(conn_n, base_mesh_n)
	lm = torch.cat([lm_new.reshape(-1), lm_base.reshape(-1)], dim=0)
	return _register_shell_term("cyl_conn_mesh", lm, res=res)


def cyl_conn_gt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Align the base-to-shell connection vector with GT normals at both shell positions."""
	geom = _shell_offset_geometry_for_loss(res)
	if geom is None:
		return _zero_loss(res)
	xyz = geom["xyz"]
	base = geom["base"]
	conn = geom["conn"]
	if not isinstance(xyz, torch.Tensor) or not isinstance(base, torch.Tensor) or not isinstance(conn, torch.Tensor):
		return _zero_loss(res)
	conn_n = F.normalize(conn, dim=-1, eps=1.0e-8)
	new_gt, new_mask = _sample_shell_gt(res=res, xyz=xyz)
	base_gt, base_mask = _sample_shell_gt(res=res, xyz=base)
	if new_gt is None or new_mask is None or base_gt is None or base_mask is None:
		return _zero_loss(res)
	lm_new, _dot_new = _normal_align_lm(conn_n, new_gt)
	lm_base, _dot_base = _normal_align_lm(conn_n, base_gt)
	lm = torch.cat([lm_new.reshape(-1), lm_base.reshape(-1)], dim=0)
	mask = torch.cat([new_mask.reshape(-1), base_mask.reshape(-1)], dim=0)
	return _register_shell_masked_term("cyl_conn_gt", lm, mask, res=res)


def cyl_base_mesh_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Align the offset-sampled base shell mesh normal with the new shell mesh normal."""
	geom = _shell_offset_geometry_for_loss(res)
	if geom is None:
		return _zero_loss(res)
	xyz = geom["xyz"]
	base = geom["base"]
	if not isinstance(xyz, torch.Tensor) or not isinstance(base, torch.Tensor):
		return _zero_loss(res)
	lm, _dot = _normal_align_lm(_unit_normals_for_shell_xyz(base), _unit_normals_for_shell_xyz(xyz))
	return _register_shell_term("cyl_base_mesh", lm, res=res)


def cyl_base_gt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Align the offset-sampled base GT normal with the new shell mesh normal."""
	geom = _shell_offset_geometry_for_loss(res)
	if geom is None:
		return _zero_loss(res)
	xyz = geom["xyz"]
	base = geom["base"]
	if not isinstance(xyz, torch.Tensor) or not isinstance(base, torch.Tensor):
		return _zero_loss(res)
	base_gt, base_mask = _sample_shell_gt(res=res, xyz=base)
	if base_gt is None or base_mask is None:
		return _zero_loss(res)
	lm, _dot = _normal_align_lm(_unit_normals_for_shell_xyz(xyz), base_gt)
	return _register_shell_masked_term("cyl_base_gt", lm, base_mask, res=res)
