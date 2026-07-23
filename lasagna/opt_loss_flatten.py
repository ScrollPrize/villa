from __future__ import annotations

import torch

import model as fit_model


_sdir_eps = 1.0e-8
_orient_min_det = 0.0
_diagnostics_enabled = True
_last_stats: dict[str, float] = {}
_prev_point_mask: torch.Tensor | None = None
_compile_flatten = False
_compile_backend: str | None = None
_compile_mode: str | None = None
_compile_dynamic = False
_compile_fullgraph = False
_compile_key: tuple[bool, str | None, str | None, bool, bool] | None = None
_compiled_forward_sdir_core = None
_compiled_orient_core = None
_compiled_combined_core = None
_compile_disabled_reason: str | None = None


def configure(
	*,
	sdir_eps: float | None = None,
	orient_min_det: float | None = None,
	diagnostics: bool = True,
	reset_history: bool = True,
) -> None:
	global _sdir_eps, _orient_min_det, _diagnostics_enabled, _last_stats, _prev_point_mask
	if sdir_eps is not None:
		_sdir_eps = max(1.0e-12, float(sdir_eps))
	if orient_min_det is not None:
		_orient_min_det = float(orient_min_det)
	_diagnostics_enabled = bool(diagnostics)
	if reset_history or not _diagnostics_enabled:
		_last_stats = {}
		_prev_point_mask = None


def configure_compile(
	*,
	enabled: bool = False,
	backend: str | None = None,
	mode: str | None = None,
	dynamic: bool = False,
	fullgraph: bool = False,
) -> None:
	"""Configure opt-in torch.compile use for the expensive flatten loss kernels."""
	global _compile_flatten, _compile_backend, _compile_mode, _compile_dynamic, _compile_fullgraph
	global _compile_key, _compiled_forward_sdir_core, _compiled_orient_core, _compiled_combined_core
	global _compile_disabled_reason
	backend = str(backend).strip() if backend is not None and str(backend).strip() else None
	mode = str(mode).strip() if mode is not None and str(mode).strip() else None
	key = (bool(enabled), backend, mode, bool(dynamic), bool(fullgraph))
	if key != _compile_key:
		_compiled_forward_sdir_core = None
		_compiled_orient_core = None
		_compiled_combined_core = None
		_compile_disabled_reason = None
		_compile_key = key
	_compile_flatten = bool(enabled)
	_compile_backend = backend
	_compile_mode = mode
	_compile_dynamic = bool(dynamic)
	_compile_fullgraph = bool(fullgraph)


def _compile_kwargs() -> dict[str, object]:
	kwargs: dict[str, object] = {
		"dynamic": _compile_dynamic,
		"fullgraph": _compile_fullgraph,
	}
	if _compile_backend is not None:
		kwargs["backend"] = _compile_backend
	if _compile_mode is not None:
		kwargs["mode"] = _compile_mode
	return kwargs


def _compiled_flatten_core(which: str, eager_fn):
	global _compiled_forward_sdir_core, _compiled_orient_core, _compiled_combined_core
	global _compile_disabled_reason
	if not _compile_flatten or _compile_disabled_reason is not None:
		return eager_fn
	if not hasattr(torch, "compile"):
		_compile_disabled_reason = "torch.compile is unavailable"
		print(f"[opt_loss_flatten] compile_flatten disabled: {_compile_disabled_reason}", flush=True)
		return eager_fn
	if which == "forward_sdir":
		compiled = _compiled_forward_sdir_core
	elif which == "orient":
		compiled = _compiled_orient_core
	elif which == "combined":
		compiled = _compiled_combined_core
	else:
		raise ValueError(f"unknown flatten compile core {which!r}")
	if compiled is None:
		try:
			compiled = torch.compile(eager_fn, **_compile_kwargs())
		except Exception as exc:
			_compile_disabled_reason = f"{type(exc).__name__}: {exc}"
			print(f"[opt_loss_flatten] compile_flatten disabled: {_compile_disabled_reason}", flush=True)
			return eager_fn
		if which == "forward_sdir":
			_compiled_forward_sdir_core = compiled
		elif which == "orient":
			_compiled_orient_core = compiled
		else:
			_compiled_combined_core = compiled
	return compiled


def _run_compiled_flatten_core(which: str, eager_fn, *args):
	global _compile_disabled_reason
	fn = _compiled_flatten_core(which, eager_fn)
	if fn is eager_fn:
		return fn(*args)
	try:
		return fn(*args)
	except Exception as exc:
		_compile_disabled_reason = f"{type(exc).__name__}: {exc}"
		print(f"[opt_loss_flatten] compile_flatten disabled after failure: {_compile_disabled_reason}", flush=True)
		return eager_fn(*args)


def _diagnostic_maps(
	lm: torch.Tensor,
	mask: torch.Tensor,
	*,
	leading_dims: int = 2,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	if not _diagnostics_enabled:
		return (), ()
	for _ in range(leading_dims):
		lm = lm.unsqueeze(0)
		mask = mask.unsqueeze(0)
	return (lm,), (mask,)


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def _is_forward(res: fit_model.FitResult3D) -> bool:
	return str(getattr(res, "flatten_direction", "inverse")).strip().lower() == "forward"


def _forward_source_fields(
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	uv = res.flatten_map
	xyz = res.flatten_source_xyz
	vertex_valid = res.flatten_source_valid
	cell_valid = res.flatten_source_cell_valid
	if uv is None or xyz is None or vertex_valid is None or cell_valid is None:
		raise RuntimeError("forward flatten loss requires source UV, xyz, vertex mask, and cell mask")
	if uv.ndim != 3 or int(uv.shape[-1]) != 2:
		raise RuntimeError(f"forward flatten UV map must have shape (H,W,2), got {tuple(uv.shape)}")
	if xyz.ndim != 3 or int(xyz.shape[-1]) != 3:
		raise RuntimeError(f"forward flatten source xyz must have shape (H,W,3), got {tuple(xyz.shape)}")
	if tuple(uv.shape[:2]) != tuple(xyz.shape[:2]):
		raise RuntimeError("forward flatten UV map shape does not match source xyz")
	if tuple(vertex_valid.shape) != tuple(uv.shape[:2]):
		raise RuntimeError("forward flatten source vertex mask shape does not match UV map")
	if tuple(cell_valid.shape) != (max(0, int(uv.shape[0]) - 1), max(0, int(uv.shape[1]) - 1)):
		raise RuntimeError("forward flatten source cell mask shape does not match UV map")
	return uv, xyz, vertex_valid.to(dtype=torch.bool), cell_valid.to(dtype=torch.bool)


def _identity_vectors(
	res: fit_model.FitResult3D,
	map_yx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	identity_y = res.flatten_identity_y
	identity_x = res.flatten_identity_x
	if identity_y is None or int(identity_y.numel()) != int(map_yx.shape[0]):
		identity_y = torch.arange(int(map_yx.shape[0]), device=map_yx.device, dtype=map_yx.dtype)
	else:
		identity_y = identity_y.to(device=map_yx.device, dtype=map_yx.dtype)
	if identity_x is None or int(identity_x.numel()) != int(map_yx.shape[1]):
		identity_x = torch.arange(int(map_yx.shape[1]), device=map_yx.device, dtype=map_yx.dtype)
	else:
		identity_x = identity_x.to(device=map_yx.device, dtype=map_yx.dtype)
	return identity_y, identity_x


def _flatten_forward_sdir_core(
	uv: torch.Tensor,
	source_metric: torch.Tensor,
	cell_valid: torch.Tensor,
	domain_step: torch.Tensor,
	eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	u00 = uv[:-1, :-1]
	u10 = uv[1:, :-1]
	u01 = uv[:-1, 1:]
	u11 = uv[1:, 1:]
	Uy = 0.5 * ((u10 - u00) + (u11 - u01))
	Ux = 0.5 * ((u01 - u00) + (u11 - u10))

	g00 = source_metric[..., 0]
	g01 = source_metric[..., 1]
	g11 = source_metric[..., 2]
	c00 = (Uy * Uy).sum(dim=-1)
	c01 = (Uy * Ux).sum(dim=-1)
	c11 = (Ux * Ux).sum(dim=-1)
	det_g = g00 * g11 - g01 * g01
	det_c = c00 * c11 - c01 * c01
	inv_g00 = g11 / det_g.clamp_min(eps)
	inv_g01 = -g01 / det_g.clamp_min(eps)
	inv_g11 = g00 / det_g.clamp_min(eps)
	inv_c00 = c11 / det_c.clamp_min(eps)
	inv_c01 = -c01 / det_c.clamp_min(eps)
	inv_c11 = c00 / det_c.clamp_min(eps)
	tr_j = (domain_step * domain_step) * (c00 * inv_g00 + 2.0 * c01 * inv_g01 + c11 * inv_g11)
	tr_inv = (g00 * inv_c00 + 2.0 * g01 * inv_c01 + g11 * inv_c11) / (domain_step * domain_step)
	lm = torch.nan_to_num(tr_j + tr_inv - 4.0, nan=0.0, posinf=1.0e12, neginf=0.0)
	det_uv = Uy[..., 0] * Ux[..., 1] - Uy[..., 1] * Ux[..., 0]
	valid = (
		cell_valid.to(device=uv.device)
		& torch.isfinite(lm)
		& torch.isfinite(det_g)
		& torch.isfinite(det_c)
		& (det_g > eps)
		& (det_c > eps)
		& torch.isfinite(det_uv)
		& (det_uv > eps)
	)
	mask = valid.to(dtype=uv.dtype)
	wsum = mask.sum()
	weighted = (lm * mask).sum()
	loss = torch.where(wsum > 0, weighted / wsum.clamp_min(1.0), weighted * 0.0)
	return loss, lm, mask, valid


def _flatten_forward_combined_core(
	uv: torch.Tensor,
	source_metric: torch.Tensor,
	vertex_valid: torch.Tensor,
	cell_valid: torch.Tensor,
	domain_step: torch.Tensor,
	avg_mask: torch.Tensor,
	avg_target: torch.Tensor,
	identity_y: torch.Tensor,
	identity_x: torch.Tensor,
	weights: torch.Tensor,
	eps: float,
	orient_min_det: float,
) -> torch.Tensor:
	"""Combined diagnostics-free forward flatten objective.

	The source metric and identity vectors are immutable model buffers. UV cell
	derivatives and determinants are shared by symmetric Dirichlet and orientation.
	"""
	m00 = uv[:-1, :-1]
	m10 = uv[1:, :-1]
	m01 = uv[:-1, 1:]
	m11 = uv[1:, 1:]
	uy = 0.5 * ((m10 - m00) + (m11 - m01))
	ux = 0.5 * ((m01 - m00) + (m11 - m10))
	c00 = (uy * uy).sum(dim=-1)
	c01 = (uy * ux).sum(dim=-1)
	c11 = (ux * ux).sum(dim=-1)
	det_c = c00 * c11 - c01 * c01
	det_uv = uy[..., 0] * ux[..., 1] - uy[..., 1] * ux[..., 0]

	g00 = source_metric[..., 0]
	g01 = source_metric[..., 1]
	g11 = source_metric[..., 2]
	det_g = g00 * g11 - g01 * g01
	inv_g00 = g11 / det_g.clamp_min(eps)
	inv_g01 = -g01 / det_g.clamp_min(eps)
	inv_g11 = g00 / det_g.clamp_min(eps)
	inv_c00 = c11 / det_c.clamp_min(eps)
	inv_c01 = -c01 / det_c.clamp_min(eps)
	inv_c11 = c00 / det_c.clamp_min(eps)
	domain_step2 = domain_step * domain_step
	tr_j = domain_step2 * (c00 * inv_g00 + 2.0 * c01 * inv_g01 + c11 * inv_g11)
	tr_inv = (g00 * inv_c00 + 2.0 * g01 * inv_c01 + g11 * inv_c11) / domain_step2
	sdir_lm = torch.nan_to_num(tr_j + tr_inv - 4.0, nan=0.0, posinf=1.0e12, neginf=0.0)
	sdir_valid = (
		cell_valid
		& torch.isfinite(sdir_lm)
		& torch.isfinite(det_g)
		& torch.isfinite(det_c)
		& (det_g > eps)
		& (det_c > eps)
		& torch.isfinite(det_uv)
		& (det_uv > eps)
	)
	sdir_mask = sdir_valid.to(dtype=uv.dtype)
	sdir_weight = sdir_mask.sum()
	sdir_sum = (sdir_lm * sdir_mask).sum()
	sdir_loss = torch.where(
		sdir_weight > 0,
		sdir_sum / sdir_weight.clamp_min(1.0),
		sdir_sum * 0.0,
	)

	dy0 = uv[1:, :, 0] - uv[:-1, :, 0] - 1.0
	dy1 = uv[1:, :, 1] - uv[:-1, :, 1]
	dy_lm = dy0 * dy0 + dy1 * dy1
	dy_valid = torch.isfinite(dy_lm) & vertex_valid[1:, :] & vertex_valid[:-1, :]
	dy_mask = dy_valid.to(dtype=uv.dtype)
	dy_safe = torch.nan_to_num(dy_lm, nan=0.0, posinf=1.0e12, neginf=0.0)
	dx0 = uv[:, 1:, 0] - uv[:, :-1, 0]
	dx1 = uv[:, 1:, 1] - uv[:, :-1, 1] - 1.0
	dx_lm = dx0 * dx0 + dx1 * dx1
	dx_valid = torch.isfinite(dx_lm) & vertex_valid[:, 1:] & vertex_valid[:, :-1]
	dx_mask = dx_valid.to(dtype=uv.dtype)
	dx_safe = torch.nan_to_num(dx_lm, nan=0.0, posinf=1.0e12, neginf=0.0)
	map_step_sum = (dy_safe * dy_mask).sum() + (dx_safe * dx_mask).sum()
	map_step_weight = dy_mask.sum() + dx_mask.sum()
	map_step_loss = torch.where(
		map_step_weight > 0,
		map_step_sum / map_step_weight.clamp_min(1.0),
		map_step_sum * 0.0,
	)

	avg_mask_f = avg_mask.to(dtype=uv.dtype)
	avg_weight = avg_mask_f.sum()
	offset_y = uv[..., 0] - identity_y.reshape(-1, 1)
	offset_x = uv[..., 1] - identity_x.reshape(1, -1)
	avg_candidate = torch.stack(
		((offset_y * avg_mask_f).sum(), (offset_x * avg_mask_f).sum()),
	) / avg_weight.clamp_min(1.0)
	avg_diff_candidate = avg_candidate - avg_target
	avg_diff = torch.where(avg_weight > 0, avg_diff_candidate, torch.zeros_like(avg_diff_candidate))
	avg_loss = (avg_diff * avg_diff).sum()

	orient_lm = torch.nan_to_num(
		torch.relu(orient_min_det - det_uv) ** 2,
		nan=0.0,
		posinf=1.0e12,
		neginf=0.0,
	)
	orient_active = cell_valid & torch.isfinite(det_uv) & (det_uv < orient_min_det)
	orient_loss = (orient_lm * orient_active.to(dtype=uv.dtype)).sum()

	return (
		weights[0] * sdir_loss
		+ weights[1] * map_step_loss
		+ weights[2] * avg_loss
		+ weights[3] * orient_loss
	)


def flatten_combined_loss(
	*,
	res: fit_model.FitResult3D,
	weights: torch.Tensor,
) -> torch.Tensor:
	"""Evaluate all four forward flatten terms in one compiled autograd graph."""
	if not _is_forward(res):
		raise RuntimeError("combined flatten loss currently requires the forward solver")
	uv, xyz, vertex_valid, cell_valid = _forward_source_fields(res)
	if int(uv.shape[0]) < 2 or int(uv.shape[1]) < 2:
		raise RuntimeError("combined flatten loss requires a map of at least 2x2")
	metric = res.flatten_source_metric
	if metric is None or int(metric.numel()) == 0:
		metric = fit_model.Model3D._flatten_source_metric(xyz)
	metric = metric.to(device=uv.device, dtype=uv.dtype)
	if tuple(metric.shape) != (int(uv.shape[0]) - 1, int(uv.shape[1]) - 1, 3):
		raise RuntimeError(f"flatten source metric shape {tuple(metric.shape)} does not match map {tuple(uv.shape)}")
	avg_mask = res.flatten_avg_offset_mask
	avg_target = res.flatten_initial_avg_offset
	if avg_mask is None or avg_target is None:
		raise RuntimeError("combined flatten loss requires average-offset mask and target")
	if tuple(avg_mask.shape) != tuple(uv.shape[:2]):
		raise RuntimeError("combined flatten average-offset mask shape does not match map")
	identity_y, identity_x = _identity_vectors(res, uv)
	if int(weights.numel()) != 4:
		raise ValueError(f"combined flatten weights must have four values, got {tuple(weights.shape)}")
	domain_step = (
		uv.new_tensor(float(res.params.mesh_step))
		if res.flatten_target_step is None
		else res.flatten_target_step.to(device=uv.device, dtype=uv.dtype)
	).clamp_min(1.0e-12)
	return _run_compiled_flatten_core(
		"combined",
		_flatten_forward_combined_core,
		uv,
		metric,
		vertex_valid.to(device=uv.device, dtype=torch.bool),
		cell_valid.to(device=uv.device, dtype=torch.bool),
		domain_step,
		avg_mask.to(device=uv.device, dtype=torch.bool),
		avg_target.to(device=uv.device, dtype=uv.dtype).reshape(2),
		identity_y,
		identity_x,
		weights.to(device=uv.device, dtype=uv.dtype).reshape(4),
		float(_sdir_eps),
		float(_orient_min_det),
	)


def _flatten_forward_sdir_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	global _last_stats, _prev_point_mask
	uv, xyz, vertex_valid, cell_valid = _forward_source_fields(res)
	if int(uv.shape[0]) < 2 or int(uv.shape[1]) < 2:
		zero = uv.sum() * 0.0
		if _diagnostics_enabled:
			return zero, (zero.reshape(1, 1, 1, 1),), (zero.reshape(1, 1, 1, 1),)
		return zero, (), ()
	if res.flatten_target_step is None:
		domain_step = torch.tensor(float(res.params.mesh_step), device=uv.device, dtype=uv.dtype)
	else:
		domain_step = res.flatten_target_step.to(device=uv.device, dtype=uv.dtype)
	domain_step = domain_step.clamp_min(1.0e-12)
	metric = res.flatten_source_metric
	if metric is None or int(metric.numel()) == 0:
		metric = fit_model.Model3D._flatten_source_metric(xyz)
	metric = metric.to(device=uv.device, dtype=uv.dtype)
	if tuple(metric.shape) != (int(uv.shape[0]) - 1, int(uv.shape[1]) - 1, 3):
		raise RuntimeError(f"flatten source metric shape {tuple(metric.shape)} does not match map {tuple(uv.shape)}")
	eps = float(_sdir_eps)
	loss, lm, mask, valid = _run_compiled_flatten_core(
		"forward_sdir",
		_flatten_forward_sdir_core,
		uv,
		metric,
		cell_valid,
		domain_step,
		eps,
	)

	if _diagnostics_enabled:
		with torch.no_grad():
			_last_stats = {
				"flatten_point_valid": float(vertex_valid.float().mean().detach().cpu()),
				"flatten_quad_valid": float(valid.float().mean().detach().cpu()) if valid.numel() else 0.0,
				"flatten_tgt_step": float(domain_step.detach().cpu()),
				"flatten_valid_to_invalid": 0.0,
				"flatten_invalid_to_valid": 0.0,
				"flatten_sdir_no_new": float(loss.detach().cpu()),
			}
			_prev_point_mask = None
	lms, masks = _diagnostic_maps(lm, mask)
	return loss, lms, masks


def flatten_sdir_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Symmetric Dirichlet energy for the flatten inverse map output surface.

	The surface samples live in fullres coordinates.  The 2D output grid domain
	uses the measured average spacing of the source tifxyz grid, so flattening
	does not impose global scaling from metadata.
	"""
	if _is_forward(res):
		return _flatten_forward_sdir_loss(res=res)
	global _last_stats, _prev_point_mask
	xyz = res.flatten_xyz
	point_mask = res.flatten_point_mask
	quad_mask = res.flatten_quad_mask
	if xyz is None or point_mask is None or quad_mask is None:
		raise RuntimeError("flatten_sdir requires flatten forward artifacts")
	if xyz.ndim != 4 or int(xyz.shape[0]) != 1 or int(xyz.shape[-1]) != 3:
		raise RuntimeError(f"flatten_xyz must have shape (1,H,W,3), got {tuple(xyz.shape)}")
	if int(xyz.shape[1]) < 2 or int(xyz.shape[2]) < 2:
		zero = xyz.sum() * 0.0
		if _diagnostics_enabled:
			return zero, (zero.reshape(1, 1, 1, 1),), (zero.reshape(1, 1, 1, 1),)
		return zero, (), ()

	p00 = xyz[:, :-1, :-1]
	p10 = xyz[:, 1:, :-1]
	p01 = xyz[:, :-1, 1:]
	p11 = xyz[:, 1:, 1:]
	if res.flatten_target_step is None:
		domain_step = torch.tensor(float(res.params.mesh_step), device=xyz.device, dtype=xyz.dtype)
	else:
		domain_step = res.flatten_target_step.to(device=xyz.device, dtype=xyz.dtype)
	domain_step = domain_step.clamp_min(1.0e-12)
	du = 0.5 * ((p10 - p00) + (p11 - p01)) / domain_step
	dv = 0.5 * ((p01 - p00) + (p11 - p10)) / domain_step

	a = (du * du).sum(dim=-1)
	b = (du * dv).sum(dim=-1)
	c = (dv * dv).sum(dim=-1)
	det = (a * c - b * b).clamp_min(float(_sdir_eps))
	tr_g = a + c
	tr_inv = (a + c) / det
	lm = torch.nan_to_num(tr_g + tr_inv - 4.0, nan=0.0, posinf=1.0e12, neginf=0.0)
	mask = quad_mask.to(device=lm.device, dtype=lm.dtype)
	wsum = mask.sum()
	weighted = (lm * mask).sum()
	loss = torch.where(wsum > 0, weighted / wsum.clamp_min(1.0), weighted * 0.0)

	if _diagnostics_enabled:
		with torch.no_grad():
			pm = point_mask.to(dtype=torch.bool)
			qm = quad_mask.to(dtype=torch.bool)
			prev_pm = _prev_point_mask
			if prev_pm is None or tuple(prev_pm.shape) != tuple(pm.shape):
				valid_to_invalid = torch.zeros_like(pm)
				invalid_to_valid = torch.zeros_like(pm)
				no_new_loss = loss.detach()
			else:
				prev_pm = prev_pm.to(device=pm.device, dtype=torch.bool)
				valid_to_invalid = prev_pm & ~pm
				invalid_to_valid = ~prev_pm & pm
				stable_pm = pm & prev_pm
				if int(stable_pm.shape[0]) > 1 and int(stable_pm.shape[1]) > 1:
					stable_qm = (
						stable_pm[:-1, :-1] &
						stable_pm[1:, :-1] &
						stable_pm[:-1, 1:] &
						stable_pm[1:, 1:]
					).unsqueeze(0) & qm
				else:
					stable_qm = torch.zeros_like(qm)
				stable_mask = stable_qm.to(device=lm.device, dtype=lm.dtype)
				stable_wsum = stable_mask.sum()
				if bool((stable_wsum > 0).detach().cpu()):
					no_new_loss = (lm * stable_mask).sum() / stable_wsum
				else:
					no_new_loss = loss.detach()
			_last_stats = {
				"flatten_point_valid": float(pm.float().mean().detach().cpu()),
				"flatten_quad_valid": float(qm.float().mean().detach().cpu()) if qm.numel() else 0.0,
				"flatten_tgt_step": float(domain_step.detach().cpu()),
				"flatten_valid_to_invalid": float(valid_to_invalid.float().mean().detach().cpu()),
				"flatten_invalid_to_valid": float(invalid_to_valid.float().mean().detach().cpu()),
				"flatten_sdir_no_new": float(no_new_loss.detach().cpu()),
			}
			_prev_point_mask = pm.detach().clone()
	lms, masks = _diagnostic_maps(lm, mask, leading_dims=1)
	return loss, lms, masks


def flatten_map_step_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Regularize the inverse map to advance one source-grid cell per output step."""
	map_yx = res.flatten_map
	if map_yx is None:
		raise RuntimeError("flatten_map_step requires flatten_map")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")

	H, W = int(map_yx.shape[0]), int(map_yx.shape[1])
	maps: list[torch.Tensor] = []
	masks: list[torch.Tensor] = []
	sum_loss = map_yx.sum() * 0.0
	sum_weight = map_yx.new_zeros(())

	def _accumulate(lm: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
		nonlocal sum_loss, sum_weight
		mask = torch.isfinite(lm)
		if valid_mask is not None:
			mask = mask & valid_mask.to(device=lm.device, dtype=torch.bool)
		mask_f = mask.to(dtype=lm.dtype)
		lm_safe = torch.nan_to_num(lm, nan=0.0, posinf=1.0e12, neginf=0.0)
		if _diagnostics_enabled:
			maps.append(lm_safe.unsqueeze(0).unsqueeze(1))
			masks.append(mask_f.unsqueeze(0).unsqueeze(1))
		sum_loss = sum_loss + (lm_safe * mask_f).sum()
		sum_weight = sum_weight + mask_f.sum()

	source_valid = None
	if _is_forward(res):
		if res.flatten_source_valid is None:
			raise RuntimeError("forward flatten_map_step requires flatten_source_valid")
		source_valid = res.flatten_source_valid.to(device=map_yx.device, dtype=torch.bool)
		if tuple(source_valid.shape) != tuple(map_yx.shape[:2]):
			raise RuntimeError("forward flatten source mask shape does not match map")

	if H > 1:
		target_y = torch.tensor([1.0, 0.0], device=map_yx.device, dtype=map_yx.dtype)
		dy = map_yx[1:, :] - map_yx[:-1, :] - target_y
		valid_edge = None if source_valid is None else (source_valid[1:, :] & source_valid[:-1, :])
		_accumulate((dy * dy).sum(dim=-1), valid_edge)
	if W > 1:
		target_x = torch.tensor([0.0, 1.0], device=map_yx.device, dtype=map_yx.dtype)
		dx = map_yx[:, 1:] - map_yx[:, :-1] - target_x
		valid_edge = None if source_valid is None else (source_valid[:, 1:] & source_valid[:, :-1])
		_accumulate((dx * dx).sum(dim=-1), valid_edge)

	loss = torch.where(sum_weight > 0, sum_loss / sum_weight.clamp_min(1.0), sum_loss * 0.0)
	if _diagnostics_enabled and not maps:
		zero = loss.reshape(1, 1, 1, 1)
		maps = [zero]
		masks = [zero]
	return loss, tuple(maps), tuple(masks)


def flatten_avg_offset_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Keep the mean inverse-map offset fixed over the init-valid source area."""
	global _last_stats
	map_yx = res.flatten_map
	mask = res.flatten_avg_offset_mask
	target = res.flatten_initial_avg_offset
	if map_yx is None or mask is None or target is None:
		raise RuntimeError("flatten_avg_offset requires flatten map offset anchor artifacts")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")
	if tuple(mask.shape) != tuple(map_yx.shape[:2]):
		raise RuntimeError(f"flatten_avg_offset_mask shape {tuple(mask.shape)} does not match map {tuple(map_yx.shape[:2])}")
	if target.numel() != 2:
		raise RuntimeError(f"flatten_initial_avg_offset must have 2 values, got {tuple(target.shape)}")

	identity_y, identity_x = _identity_vectors(res, map_yx)
	mask_f = mask.to(device=map_yx.device, dtype=map_yx.dtype)
	weight = mask_f.sum()
	target_t = target.to(device=map_yx.device, dtype=map_yx.dtype).reshape(2)
	offset_y = map_yx[..., 0] - identity_y.reshape(-1, 1)
	offset_x = map_yx[..., 1] - identity_x.reshape(1, -1)
	weighted_offset = torch.stack(
		((offset_y * mask_f).sum(), (offset_x * mask_f).sum()),
	)
	avg_candidate = weighted_offset / weight.clamp_min(1.0)
	diff_candidate = avg_candidate - target_t
	diff = torch.where(weight > 0, diff_candidate, torch.zeros_like(diff_candidate))
	loss = (diff * diff).sum()
	if _diagnostics_enabled:
		lm = torch.nan_to_num(
			(offset_y - target_t[0]).square() + (offset_x - target_t[1]).square(),
			nan=0.0,
			posinf=1.0e12,
			neginf=0.0,
		)
		with torch.no_grad():
			_last_stats = {
				**_last_stats,
				"flatten_avg_offset_norm": float(torch.linalg.vector_norm(diff).detach().cpu()),
			}
		return loss, (lm.unsqueeze(0).unsqueeze(1),), (mask_f.unsqueeze(0).unsqueeze(1),)
	return loss, (), ()


def _flatten_orient_core(
	map_yx: torch.Tensor,
	valid_cells: torch.Tensor,
	min_det_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	m00 = map_yx[:-1, :-1]
	m10 = map_yx[1:, :-1]
	m01 = map_yx[:-1, 1:]
	m11 = map_yx[1:, 1:]
	dy = 0.5 * ((m10 - m00) + (m11 - m01))
	dx = 0.5 * ((m01 - m00) + (m11 - m10))
	det = dy[..., 0] * dx[..., 1] - dy[..., 1] * dx[..., 0]
	min_det = torch.tensor(min_det_value, device=map_yx.device, dtype=map_yx.dtype)
	lm = torch.nan_to_num(torch.relu(min_det - det) ** 2, nan=0.0, posinf=1.0e12, neginf=0.0)
	active = valid_cells & torch.isfinite(det) & (det < min_det)
	mask = active.to(dtype=lm.dtype)
	loss = (lm * mask).sum()
	return loss, lm, mask, det


def flatten_orient_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Source-map signed-area hinge for fold-in prevention."""
	global _last_stats
	map_yx = res.flatten_map
	if map_yx is None:
		raise RuntimeError("flatten_orient requires flatten_map")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")

	H, W = int(map_yx.shape[0]), int(map_yx.shape[1])
	if H < 2 or W < 2:
		zero = map_yx.sum() * 0.0
		if _diagnostics_enabled:
			return zero, (zero.reshape(1, 1, 1, 1),), (zero.reshape(1, 1, 1, 1),)
		return zero, (), ()

	valid_cells = torch.ones((H - 1, W - 1), device=map_yx.device, dtype=torch.bool)
	if _is_forward(res):
		if res.flatten_source_cell_valid is None:
			raise RuntimeError("forward flatten_orient requires flatten_source_cell_valid")
		valid_cells = res.flatten_source_cell_valid.to(device=map_yx.device, dtype=torch.bool)
		if tuple(valid_cells.shape) != (H - 1, W - 1):
			raise RuntimeError("forward flatten source cell mask shape does not match orient determinant")
	loss, lm, mask, det = _run_compiled_flatten_core(
		"orient",
		_flatten_orient_core,
		map_yx,
		valid_cells,
		float(_orient_min_det),
	)

	if _diagnostics_enabled:
		with torch.no_grad():
			min_det = torch.tensor(float(_orient_min_det), device=map_yx.device, dtype=map_yx.dtype)
			valid_det = det[valid_cells & torch.isfinite(det)]
			if valid_det.numel():
				fold_frac = float((valid_det <= 0.0).float().mean().detach().cpu())
				lowdet_frac = float((valid_det < min_det).float().mean().detach().cpu())
				min_det_val = float(valid_det.min().detach().cpu())
				mean_det_val = float(valid_det.mean().detach().cpu())
			else:
				fold_frac = 0.0
				lowdet_frac = 0.0
				min_det_val = 0.0
				mean_det_val = 0.0
			_last_stats = {
				**_last_stats,
				"flatten_orient_fold_frac": fold_frac,
				"flatten_orient_lowdet_frac": lowdet_frac,
				"flatten_orient_min_det": min_det_val,
				"flatten_orient_mean_det": mean_det_val,
			}
	lms, masks = _diagnostic_maps(lm, mask)
	return loss, lms, masks
