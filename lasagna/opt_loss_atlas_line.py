from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import model as fit_model
import opt_loss_corr


_rows: torch.Tensor | None = None
_cols: torch.Tensor | None = None
_frac_h: torch.Tensor | None = None
_frac_w: torch.Tensor | None = None
_last_stats: dict[str, float] = {}
_last_debug_payload: "AtlasLineDebugPayload | None" = None


@dataclass(frozen=True)
class AtlasLineDebugPayload:
	model_xyz: torch.Tensor
	valid: torch.Tensor
	target_xyz: torch.Tensor
	hit_xyz: torch.Tensor
	model_normal: torch.Tensor
	signed_delta: torch.Tensor
	is_control: torch.Tensor
	is_snap: torch.Tensor
	sample_model_h: torch.Tensor
	sample_model_w: torch.Tensor
	normal_proxy_target_xyz: torch.Tensor
	normal_proxy_valid: torch.Tensor
	object_ids: tuple[str, ...]
	atlas_winding_model_ranges: tuple[tuple[int, float, float], ...] = ()


def reset_state() -> None:
	global _rows, _cols, _frac_h, _frac_w, _last_stats, _last_debug_payload
	_rows = None
	_cols = None
	_frac_h = None
	_frac_w = None
	_last_stats = {}
	_last_debug_payload = None


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def last_debug_payload() -> AtlasLineDebugPayload | None:
	return _last_debug_payload


def _atlas_object_ids(lines, count: int) -> tuple[str, ...]:
	raw = tuple(str(v) for v in (getattr(lines, "object_ids", ()) or ()))
	if len(raw) == int(count):
		return raw
	return tuple("unknown" for _ in range(int(count)))


def _atlas_source_indices(lines, count: int) -> tuple[int, ...]:
	raw = tuple(int(v) for v in (getattr(lines, "source_indices", ()) or ()))
	if len(raw) == int(count):
		return raw
	return tuple(int(i) for i in range(int(count)))


def _finite_float(value: object) -> float | None:
	try:
		v = float(value)
	except (TypeError, ValueError):
		return None
	return v if torch.isfinite(torch.tensor(v)).item() else None


def _xyz_list(values: torch.Tensor) -> list[float | None]:
	vals = values.detach().cpu().reshape(3).tolist()
	return [_finite_float(v) for v in vals]


def _snap_status(*, valid: bool | None, signed_delta: float | None) -> str:
	if valid is None:
		return "-"
	if not valid:
		return "invalid"
	if signed_delta is None:
		return "valid"
	return "valid fw" if signed_delta >= 0.0 else "valid bw"


def atlas_control_points_results(*, lines, payload: AtlasLineDebugPayload | None = None) -> dict | None:
	"""Build JSON-safe final atlas-control feedback from an atlas-line debug payload."""
	payload = _last_debug_payload if payload is None else payload
	if payload is None or lines is None:
		return None

	K = int(lines.target_xyz.shape[0])
	if K <= 0:
		return {
			"format": "lasagna_atlas_control_points_results",
			"version": 1,
			"summary": _atlas_control_summary([]),
			"fibers": [],
			"records": [],
		}

	is_control = getattr(lines, "is_control_point", None)
	if is_control is None:
		control_k = torch.zeros(K, dtype=torch.bool)
	else:
		control_k = is_control.detach().cpu().to(dtype=torch.bool).view(K)
	is_snap = getattr(lines, "is_snap_point", None)
	if is_snap is None:
		snap_k = torch.zeros(K, dtype=torch.bool)
	else:
		snap_k = is_snap.detach().cpu().to(dtype=torch.bool).view(K)
	object_ids = _atlas_object_ids(lines, K)
	source_indices = _atlas_source_indices(lines, K)

	valid = payload.valid.detach().cpu().to(dtype=torch.bool)
	target_xyz = payload.target_xyz.detach().cpu().to(dtype=torch.float32)
	hit_xyz = payload.hit_xyz.detach().cpu().to(dtype=torch.float32)
	signed_delta = payload.signed_delta.detach().cpu().to(dtype=torch.float32)
	model_h = payload.sample_model_h.detach().cpu().to(dtype=torch.float32)
	model_w = payload.sample_model_w.detach().cpu().to(dtype=torch.float32)
	D = int(valid.shape[0])

	snap_by_key: dict[tuple[str, int], int] = {}
	for k in range(K):
		if bool(snap_k[k]):
			snap_by_key.setdefault((object_ids[k], int(source_indices[k])), k)

	records: list[dict] = []
	for d in range(D):
		for k in range(K):
			if not bool(control_k[k]):
				continue
			target = target_xyz[d, k]
			mesh = hit_xyz[d, k]
			ok = bool(valid[d, k]) and bool(torch.isfinite(target).all()) and bool(torch.isfinite(mesh).all())
			distance = None
			if ok:
				distance = float(torch.linalg.vector_norm(target - mesh).item())
			snap_idx = snap_by_key.get((object_ids[k], int(source_indices[k])))
			snap_valid = None
			snap_delta = None
			snap_target_xyz = None
			if snap_idx is not None:
				snap_target = target_xyz[d, snap_idx]
				snap_mesh = hit_xyz[d, snap_idx]
				snap_valid = (
					bool(valid[d, snap_idx])
					and bool(torch.isfinite(snap_target).all())
					and bool(torch.isfinite(snap_mesh).all())
				)
				snap_delta = _finite_float(signed_delta[d, snap_idx].item())
				snap_target_xyz = _xyz_list(snap_target)
			record = {
				"fiber_id": object_ids[k],
				"object_id": object_ids[k],
				"source_index": int(source_indices[k]),
				"control_index": -1,
				"target_xyz": _xyz_list(target),
				"mesh_xyz": _xyz_list(mesh),
				"model_h": _finite_float(model_h[d, k].item()),
				"model_w": _finite_float(model_w[d, k].item()),
				"distance": _finite_float(distance),
				"signed_delta": _finite_float(signed_delta[d, k].item()),
				"valid": ok,
				"snap_valid": snap_valid,
				"snap_direction": None if snap_valid is None or snap_delta is None else ("fw" if snap_delta >= 0.0 else "bw"),
				"snap_signed_delta": snap_delta,
				"snap_target_xyz": snap_target_xyz,
				"snap_status": _snap_status(valid=snap_valid, signed_delta=snap_delta),
			}
			if D != 1:
				record["layer_index"] = int(d)
			records.append(record)

	records.sort(key=lambda r: (str(r["fiber_id"]), int(r.get("layer_index", 0)), int(r["source_index"])))
	fibers: list[dict] = []
	by_fiber: dict[str, list[dict]] = {}
	for record in records:
		by_fiber.setdefault(str(record["fiber_id"]), []).append(record)
	for fiber_id in sorted(by_fiber):
		points = by_fiber[fiber_id]
		for i, record in enumerate(points):
			record["control_index"] = int(i)
		fibers.append({
			"fiber_id": fiber_id,
			"object_id": fiber_id,
			"control_points": points,
		})

	return {
		"format": "lasagna_atlas_control_points_results",
		"version": 1,
		"summary": _atlas_control_summary(records),
		"fibers": fibers,
		"records": records,
	}


def _atlas_control_summary(records: list[dict]) -> dict:
	distances = [
		float(r["distance"])
		for r in records
		if bool(r.get("valid")) and r.get("distance") is not None
	]
	total = int(len(records))
	valid_count = int(len(distances))
	max_distance = max(distances) if distances else 0.0
	rms_distance = (
		float(sum(d * d for d in distances) / len(distances)) ** 0.5
		if distances else 0.0
	)
	return {
		"total_count": total,
		"control_count": total,
		"valid_count": valid_count,
		"max_distance": float(max_distance),
		"rms_distance": float(rms_distance),
	}


def _empty_debug_payload(*, res: fit_model.FitResult3D, lines) -> AtlasLineDebugPayload:
	D = int(res.xyz_lr.shape[0])
	K = int(lines.target_xyz.shape[0])
	return AtlasLineDebugPayload(
		model_xyz=res.xyz_lr.detach().cpu(),
		valid=torch.zeros((D, K), dtype=torch.bool),
		target_xyz=torch.full((D, K, 3), float("nan"), dtype=torch.float32),
		hit_xyz=torch.full((D, K, 3), float("nan"), dtype=torch.float32),
		model_normal=torch.full((D, K, 3), float("nan"), dtype=torch.float32),
		signed_delta=torch.full((D, K), float("nan"), dtype=torch.float32),
		is_control=torch.zeros((D, K), dtype=torch.bool),
		is_snap=torch.zeros((D, K), dtype=torch.bool),
		sample_model_h=torch.full((D, K), float("nan"), dtype=torch.float32),
		sample_model_w=torch.full((D, K), float("nan"), dtype=torch.float32),
		normal_proxy_target_xyz=torch.full(tuple(res.xyz_lr.shape), float("nan"), dtype=torch.float32),
		normal_proxy_valid=torch.zeros(tuple(res.xyz_lr.shape[:3]), dtype=torch.bool),
		object_ids=_atlas_object_ids(lines, K),
		atlas_winding_model_ranges=tuple(getattr(lines, "atlas_winding_model_ranges", ()) or ()),
	)


def _write_obj_mesh(path: Path, xyz: torch.Tensor) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	xyz_cpu = xyz.detach().cpu()
	if xyz_cpu.ndim == 3:
		xyz_cpu = xyz_cpu.unsqueeze(0)
	if xyz_cpu.ndim != 4:
		raise ValueError(f"debug mesh expects (H,W,3) or (D,H,W,3), got {tuple(xyz_cpu.shape)}")
	D, H, W, _ = xyz_cpu.shape
	lines = ["o model_surface\n"]
	vertex_index = 1
	for d in range(D):
		index = torch.zeros((H, W), dtype=torch.long)
		lines.append(f"g layer_{d:03d}\n")
		for h in range(H):
			for w in range(W):
				p = xyz_cpu[d, h, w]
				if not bool(torch.isfinite(p).all()):
					continue
				index[h, w] = vertex_index
				vertex_index += 1
				lines.append(f"v {float(p[0]):.9g} {float(p[1]):.9g} {float(p[2]):.9g}\n")
		for h in range(max(0, H - 1)):
			for w in range(max(0, W - 1)):
				v00 = int(index[h, w])
				v10 = int(index[h + 1, w])
				v11 = int(index[h + 1, w + 1])
				v01 = int(index[h, w + 1])
				if v00 and v10 and v11 and v01:
					lines.append(f"f {v00} {v10} {v11} {v01}\n")
	path.write_text("".join(lines), encoding="utf-8")


def _write_obj_mesh_legacy(path: Path, xyz: torch.Tensor) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	xyz_cpu = xyz.detach().cpu()
	H, W, _ = xyz_cpu.shape
	index = torch.zeros((H, W), dtype=torch.long)
	next_index = 1
	lines = ["o model_surface\n"]
	for h in range(H):
		for w in range(W):
			p = xyz_cpu[h, w]
			if not bool(torch.isfinite(p).all()):
				continue
			index[h, w] = next_index
			next_index += 1
			lines.append(f"v {float(p[0]):.9g} {float(p[1]):.9g} {float(p[2]):.9g}\n")
	for h in range(max(0, H - 1)):
		for w in range(max(0, W - 1)):
			v00 = int(index[h, w])
			v10 = int(index[h + 1, w])
			v11 = int(index[h + 1, w + 1])
			v01 = int(index[h, w + 1])
			if v00 and v10 and v11 and v01:
				lines.append(f"f {v00} {v10} {v11} {v01}\n")
	path.write_text("".join(lines), encoding="utf-8")


def _write_obj_lines(path: Path, start: torch.Tensor, end: torch.Tensor, valid: torch.Tensor, *, label: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	start_cpu = start.detach().cpu().reshape(-1, 3)
	end_cpu = end.detach().cpu().reshape(-1, 3)
	valid_cpu = valid.detach().cpu().reshape(-1).to(dtype=torch.bool)
	lines = [f"o {label}\n"]
	vertex_index = 1
	for a, b, ok in zip(start_cpu, end_cpu, valid_cpu):
		if not bool(ok):
			continue
		if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
			continue
		lines.append(f"v {float(a[0]):.9g} {float(a[1]):.9g} {float(a[2]):.9g}\n")
		lines.append(f"v {float(b[0]):.9g} {float(b[1]):.9g} {float(b[2]):.9g}\n")
		lines.append(f"l {vertex_index} {vertex_index + 1}\n")
		vertex_index += 2
	path.write_text("".join(lines), encoding="utf-8")


def _debug_stage_label(stage: str) -> str:
	return str(stage).replace("/", "_").replace("\\", "_")


def _debug_obj_name(name: str) -> str:
	out = []
	for ch in str(name):
		if ch.isalnum() or ch in ("-", "_", "."):
			out.append(ch)
		else:
			out.append("_")
	s = "".join(out).strip("._")
	return s or "unknown"


def _column_slice_for_winding(start_w: float, end_w: float, width: int) -> slice:
	lo = min(float(start_w), float(end_w))
	hi = max(float(start_w), float(end_w))
	c0 = max(0, int(torch.floor(torch.tensor(lo)).item()))
	c1 = min(int(width), int(torch.ceil(torch.tensor(hi)).item()) + 1)
	if c1 <= c0:
		c1 = min(int(width), c0 + 1)
	return slice(c0, c1)


def _write_control_connections_by_fiber(root: Path, payload: AtlasLineDebugPayload) -> None:
	out_dir = root / "control_connections_by_fiber"
	valid_control = payload.valid.to(dtype=torch.bool) & payload.is_control.to(dtype=torch.bool)
	object_ids = tuple(payload.object_ids)
	if not object_ids:
		return
	seen: set[str] = set()
	for object_id in object_ids:
		if object_id in seen:
			continue
		seen.add(object_id)
		k_mask = torch.tensor([oid == object_id for oid in object_ids], dtype=torch.bool).view(1, -1)
		mask = valid_control & k_mask.expand_as(valid_control)
		if not bool(mask.any()):
			continue
		_write_obj_lines(
			out_dir / f"{_debug_obj_name(object_id)}.obj",
			payload.hit_xyz,
			payload.target_xyz,
			mask,
			label=f"control_connections_{_debug_obj_name(object_id)}",
		)


def write_debug_objs(*, stage: str, step: int, interval: int = 1,
					 payload: AtlasLineDebugPayload | None = None) -> Path | None:
	payload = _last_debug_payload if payload is None else payload
	if payload is None:
		return None
	interval = max(1, int(interval))
	step_i = int(step)
	if step_i % interval != 0:
		return None
	root = Path.cwd() / "atlas_debug_objs" / f"{_debug_stage_label(stage)}_step{step_i:06d}"
	model_xyz = payload.model_xyz
	valid = payload.valid.to(dtype=torch.bool)
	control = payload.is_control.to(dtype=torch.bool)
	proxy_valid = payload.normal_proxy_valid.to(dtype=torch.bool)
	_write_control_connections_by_fiber(root, payload)
	ranges = tuple(payload.atlas_winding_model_ranges or ())
	if ranges:
		D, _H, W, _ = model_xyz.shape
		for winding, start_w, end_w in ranges:
			col_slice = _column_slice_for_winding(start_w, end_w, int(W))
			wdir = root / f"atlas_winding_{int(winding):+04d}"
			wdir.mkdir(parents=True, exist_ok=True)
			_write_obj_mesh(wdir / "model_surface.obj", model_xyz[:, :, col_slice])
			w_sample = payload.sample_model_w
			if int(winding) == int(ranges[-1][0]):
				wmask = (w_sample >= min(start_w, end_w)) & (w_sample <= max(start_w, end_w))
			else:
				wmask = (w_sample >= min(start_w, end_w)) & (w_sample < max(start_w, end_w))
			_write_obj_lines(
				wdir / "control_connections.obj",
				payload.hit_xyz,
				payload.target_xyz,
				valid & control & wmask,
				label="control_connections",
			)
			_write_obj_lines(
				wdir / "other_connections.obj",
				payload.hit_xyz,
				payload.target_xyz,
				valid & ~control & wmask,
				label="other_connections",
			)
			_write_obj_lines(
				wdir / "normal_proxy.obj",
				model_xyz[:, :, col_slice],
				payload.normal_proxy_target_xyz[:, :, col_slice],
				proxy_valid[:, :, col_slice],
				label="normal_proxy",
			)
		return root

	for d in range(int(model_xyz.shape[0])):
		wdir = root / f"winding_{d:03d}"
		wdir.mkdir(parents=True, exist_ok=True)
		_write_obj_mesh_legacy(wdir / "model_surface.obj", model_xyz[d])
		_write_obj_lines(
			wdir / "control_connections.obj",
			payload.hit_xyz[d],
			payload.target_xyz[d],
			valid[d] & control[d],
			label="control_connections",
		)
		_write_obj_lines(
			wdir / "other_connections.obj",
			payload.hit_xyz[d],
			payload.target_xyz[d],
			valid[d] & ~control[d],
			label="other_connections",
		)
		_write_obj_lines(
			wdir / "normal_proxy.obj",
			model_xyz[d],
			payload.normal_proxy_target_xyz[d],
			proxy_valid[d],
			label="normal_proxy",
		)
	return root


def _ensure_state(*, res: fit_model.FitResult3D) -> None:
	global _rows, _cols, _frac_h, _frac_w
	lines = getattr(res.data, "atlas_lines", None)
	if lines is None:
		raise ValueError("atlas_line loss requires FitData3D.atlas_lines")
	D, H, W, _ = res.xyz_lr.shape
	K = int(lines.target_xyz.shape[0])
	device = res.xyz_lr.device
	if (
		_rows is not None
		and tuple(_rows.shape) == (D, K)
		and _rows.device == device
	):
		return
	h = lines.model_h.to(device=device, dtype=torch.float32).view(1, K).expand(D, K)
	w = lines.model_w.to(device=device, dtype=torch.float32).view(1, K).expand(D, K)
	hc = h.clamp(0.0, float(max(0, H - 1)))
	wc = w.clamp(0.0, float(max(0, W - 1)))
	_rows = torch.floor(hc).to(dtype=torch.long).clamp(0, max(0, H - 2))
	_cols = torch.floor(wc).to(dtype=torch.long).clamp(0, max(0, W - 2))
	_frac_h = (hc - _rows.to(dtype=torch.float32)).clamp(0.0, 1.0)
	_frac_w = (wc - _cols.to(dtype=torch.float32)).clamp(0.0, 1.0)


def _gather_quads(xyz: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> tuple[torch.Tensor, ...]:
	D, _H, _W, _ = xyz.shape
	K = int(row.shape[1])
	d_idx = torch.arange(D, device=xyz.device, dtype=torch.long).view(D, 1).expand(D, K)
	M00 = xyz[d_idx, row, col]
	M10 = xyz[d_idx, row + 1, col]
	M01 = xyz[d_idx, row, col + 1]
	M11 = xyz[d_idx, row + 1, col + 1]
	return M00, M10, M01, M11


def _bilinear(M00: torch.Tensor, M10: torch.Tensor, M01: torch.Tensor, M11: torch.Tensor,
			  u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	return (
		M00 * (1.0 - u).unsqueeze(-1) * (1.0 - v).unsqueeze(-1)
		+ M10 * u.unsqueeze(-1) * (1.0 - v).unsqueeze(-1)
		+ M01 * (1.0 - u).unsqueeze(-1) * v.unsqueeze(-1)
		+ M11 * u.unsqueeze(-1) * v.unsqueeze(-1)
	)


def _ray_residual(P: torch.Tensor, O: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
	n_len = torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(1.0e-12)
	n_unit = n / n_len
	d = P - O
	t = (d * n_unit).sum(dim=-1, keepdim=True)
	closest = O + t * n_unit
	return torch.linalg.vector_norm(P - closest, dim=-1)


def _intersect_quad(
	O: torch.Tensor,
	n: torch.Tensor,
	M00: torch.Tensor,
	M10: torch.Tensor,
	M01: torch.Tensor,
	M11: torch.Tensor,
	frac_h: torch.Tensor,
	frac_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	u, v = fit_model.Model3D._ray_bilinear_intersect(O, n, M00, M10, M01, M11, frac_h, frac_w)
	P = _bilinear(M00, M10, M01, M11, u, v)
	res = _ray_residual(P, O, n)

	a = M10 - M00
	b = M01 - M00
	A = torch.stack((a, b, -n), dim=-1)
	rhs = (O - M00).unsqueeze(-1)
	sol = torch.matmul(torch.linalg.pinv(A), rhs).squeeze(-1)
	u_aff = sol[..., 0]
	v_aff = sol[..., 1]
	P_aff = _bilinear(M00, M10, M01, M11, u_aff, v_aff)
	res_aff = _ray_residual(P_aff, O, n)
	use_aff = torch.isfinite(res_aff) & (~torch.isfinite(res) | (res_aff < res))
	return torch.where(use_aff, u_aff, u), torch.where(use_aff, v_aff, v)


def _unit_with_valid(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	n_len = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
	valid = torch.isfinite(n).all(dim=-1) & (n_len.squeeze(-1) > 1.0e-8)
	return n / n_len.clamp_min(1.0e-8), valid


def _zero_item(res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	D, H, W, _ = res.xyz_lr.shape
	z = res.xyz_lr.sum() * 0.0
	lm = torch.zeros((D, 1, H, W), device=res.xyz_lr.device, dtype=res.xyz_lr.dtype) + z
	mask = torch.zeros_like(lm)
	return z, (lm,), (mask,)


def _loss_from_splats(
	*,
	res: fit_model.FitResult3D,
	d_p: torch.Tensor,
	h_floor_p: torch.Tensor,
	w_floor_p: torch.Tensor,
	h_cont_p: torch.Tensor,
	w_cont_p: torch.Tensor,
	signed_delta_p: torch.Tensor,
	normal_p: torch.Tensor,
	mask_p: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], float, torch.Tensor, torch.Tensor]:
	D, H, W, _ = res.xyz_lr.shape
	dev = res.xyz_lr.device
	dt = res.xyz_lr.dtype
	z = res.xyz_lr.sum() * 0.0
	sq_map = torch.zeros((D, H, W), device=dev, dtype=dt) + z
	mask_map = torch.zeros((D, H, W), device=dev, dtype=dt)
	proxy_target_map = torch.full((D, H, W, 3), float("nan"), device=dev, dtype=dt)
	proxy_valid_map = torch.zeros((D, H, W), device=dev, dtype=torch.bool)

	valid = mask_p > 1.0e-8
	if not bool(valid.any().detach().cpu()):
		return z, (sq_map.unsqueeze(1),), (mask_map.unsqueeze(1),), 0.0, proxy_target_map, proxy_valid_map

	d_p = d_p[valid]
	h_floor_p = h_floor_p[valid]
	w_floor_p = w_floor_p[valid]
	h_cont_p = h_cont_p[valid]
	w_cont_p = w_cont_p[valid]
	signed_delta_p = signed_delta_p[valid]
	normal_p = normal_p[valid]
	mask_p = mask_p[valid]

	H_map = torch.zeros((D, H, W), device=dev, dtype=dt)
	V_map = torch.zeros((D, H, W, 3), device=dev, dtype=dt)
	W_map = torch.zeros((D, H, W), device=dev, dtype=dt)
	W_max_map = torch.zeros((D, H, W), device=dev, dtype=dt)
	sigma = max(1.0e-6, float(getattr(opt_loss_corr, "_corr_splat_sigma", 1.0)))
	opt_loss_corr._height_map_splat(
		d_p, h_floor_p, w_floor_p, h_cont_p, w_cont_p,
		signed_delta_p, normal_p, mask_p, sigma,
		D, H, W, H_map, V_map, W_map, W_max_map,
	)

	active = W_map > 1.0e-8
	active_count = float(active.to(dtype=torch.float32).sum().detach().cpu())
	if not bool(active.any().detach().cpu()):
		return z, (sq_map.unsqueeze(1),), (mask_map.unsqueeze(1),), 0.0, proxy_target_map, proxy_valid_map

	d_idx, h_idx, w_idx = active.nonzero(as_tuple=True)
	live = res.xyz_lr[d_idx, h_idx, w_idx]
	det = res.xyz_lr.detach()[d_idx, h_idx, w_idx]
	disp = V_map[d_idx, h_idx, w_idx] / W_map[d_idx, h_idx, w_idx].clamp_min(1.0e-8).unsqueeze(-1)
	target = det + disp
	sq = (live - target).square().sum(dim=-1)
	weights = W_max_map[d_idx, h_idx, w_idx]
	loss = (weights * sq).sum() / weights.sum().clamp_min(1.0e-8)
	sq_map[d_idx, h_idx, w_idx] = sq
	mask_map[d_idx, h_idx, w_idx] = weights
	proxy_target_map[d_idx, h_idx, w_idx] = target.detach()
	proxy_valid_map[d_idx, h_idx, w_idx] = True
	return loss, (sq_map.unsqueeze(1),), (mask_map.unsqueeze(1),), active_count, proxy_target_map, proxy_valid_map


def atlas_line_loss(
	*, res: fit_model.FitResult3D,
	stage_eff: dict[str, float] | None = None,
	debug_payload: bool = False,
) -> dict[str, tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
	global _rows, _cols, _frac_h, _frac_w, _last_stats, _last_debug_payload
	_last_debug_payload = None
	lines = getattr(res.data, "atlas_lines", None)
	if lines is None:
		raise ValueError("atlas_line loss requires FitData3D.atlas_lines")
	if int(lines.target_xyz.shape[0]) <= 0:
		item = _zero_item(res)
		if debug_payload:
			_last_debug_payload = _empty_debug_payload(res=res, lines=lines)
		return {"atlas_line": item, "atlas_line_control": item, "atlas_line_other": item, "atlas_line_snap": item}

	_ensure_state(res=res)
	assert _rows is not None and _cols is not None and _frac_h is not None and _frac_w is not None
	D, H, W, _ = res.xyz_lr.shape
	K = int(lines.target_xyz.shape[0])
	device = res.xyz_lr.device
	if H < 2 or W < 2:
		raise ValueError(f"atlas_line requires model H/W >= 2, got {H}x{W}")
	if res.normals is None:
		raise ValueError("atlas_line loss requires model normals")

	is_control = getattr(lines, "is_control_point", None)
	if is_control is None:
		control_k = torch.zeros(K, device=device, dtype=torch.bool)
	else:
		control_k = is_control.to(device=device, dtype=torch.bool).view(K)
	is_snap = getattr(lines, "is_snap_point", None)
	if is_snap is None:
		snap_k = torch.zeros(K, device=device, dtype=torch.bool)
	else:
		snap_k = is_snap.to(device=device, dtype=torch.bool).view(K)
	other_k = ~(control_k | snap_k)
	if stage_eff is None:
		process_control = True
		process_other = True
		process_snap = True
	else:
		legacy_weight = float(stage_eff.get("atlas_line", 0.0)) > 0.0
		process_control = legacy_weight or float(stage_eff.get("atlas_line_control", 0.0)) > 0.0
		process_other = legacy_weight or float(stage_eff.get("atlas_line_other", 0.0)) > 0.0
		process_snap = float(stage_eff.get("atlas_line_snap", 0.0)) > 0.0
	active_k = (control_k & process_control) | (other_k & process_other) | (snap_k & process_snap)
	zero_item = _zero_item(res)

	valid = torch.zeros((D, K), device=device, dtype=torch.bool)
	valid_control = torch.zeros_like(valid)
	valid_other = torch.zeros_like(valid)
	valid_snap = torch.zeros_like(valid)
	combined_item = zero_item
	control_item = zero_item
	other_item = zero_item
	snap_item = zero_item
	combined_active_verts = 0.0
	control_active_verts = 0.0
	other_active_verts = 0.0
	snap_active_verts = 0.0
	signed_delta_mean = 0.0
	if debug_payload:
		_last_debug_payload = _empty_debug_payload(res=res, lines=lines)

	if bool(active_k.any().detach().cpu()):
		active_idx = torch.nonzero(active_k, as_tuple=False).flatten()
		K_active = int(active_idx.shape[0])
		rows_active = _rows.index_select(1, active_idx)
		cols_active = _cols.index_select(1, active_idx)
		frac_h_active = _frac_h.index_select(1, active_idx)
		frac_w_active = _frac_w.index_select(1, active_idx)
		target = lines.target_xyz.to(device=device, dtype=res.xyz_lr.dtype).index_select(0, active_idx).view(1, K_active, 3).expand(D, K_active, 3)
		t_valid = torch.isfinite(target).all(dim=-1)
		in_bounds = (rows_active >= 0) & (rows_active < H - 1) & (cols_active >= 0) & (cols_active < W - 1)

		xyz_det = res.xyz_lr.detach()
		norm_det = res.normals.detach()
		N00, N10, N01, N11 = _gather_quads(norm_det, rows_active, cols_active)
		n0, n0_valid = _unit_with_valid(_bilinear(N00, N10, N01, N11, frac_h_active, frac_w_active))

		M00, M10, M01, M11 = _gather_quads(xyz_det, rows_active, cols_active)
		cached_hit = _bilinear(M00, M10, M01, M11, frac_h_active, frac_w_active)
		u1, v1 = _intersect_quad(target, n0, M00, M10, M01, M11, frac_h_active, frac_w_active)
		finite1 = torch.isfinite(u1) & torch.isfinite(v1)
		step_h = torch.where(u1 < 0.0, torch.full_like(rows_active, -1), torch.where(u1 > 1.0, torch.ones_like(rows_active), torch.zeros_like(rows_active)))
		step_w = torch.where(v1 < 0.0, torch.full_like(cols_active, -1), torch.where(v1 > 1.0, torch.ones_like(cols_active), torch.zeros_like(cols_active)))
		row2 = torch.where(finite1, (rows_active + step_h).clamp(0, H - 2), rows_active)
		col2 = torch.where(finite1, (cols_active + step_w).clamp(0, W - 2), cols_active)
		frac_h1 = torch.where(finite1, u1.clamp(0.0, 1.0), frac_h_active)
		frac_w1 = torch.where(finite1, v1.clamp(0.0, 1.0), frac_w_active)

		Q00, Q10, Q01, Q11 = _gather_quads(xyz_det, row2, col2)
		u2, v2 = _intersect_quad(target, n0, Q00, Q10, Q01, Q11, frac_h1, frac_w1)
		finite2 = torch.isfinite(u2) & torch.isfinite(v2)
		u = torch.where(finite2, u2.clamp(0.0, 1.0), frac_h_active).detach()
		v = torch.where(finite2, v2.clamp(0.0, 1.0), frac_w_active).detach()
		hit_rows = torch.where(finite2, row2, rows_active)
		hit_cols = torch.where(finite2, col2, cols_active)
		H00, H10, H01, H11 = _gather_quads(xyz_det, hit_rows, hit_cols)
		hit_xyz = torch.where(
			finite2.unsqueeze(-1),
			_bilinear(H00, H10, H01, H11, u, v),
			cached_hit,
		)
		HN00, HN10, HN01, HN11 = _gather_quads(norm_det, hit_rows, hit_cols)
		hit_n, hit_n_valid = _unit_with_valid(_bilinear(HN00, HN10, HN01, HN11, u, v))
		model_n = torch.where(finite2.unsqueeze(-1), hit_n, n0)
		model_n_valid = torch.where(finite2, hit_n_valid, n0_valid)
		valid_active = t_valid & in_bounds & n0_valid & model_n_valid & torch.isfinite(hit_xyz).all(dim=-1)
		valid = valid.scatter(1, active_idx.view(1, K_active).expand(D, K_active), valid_active)
		active_control = control_k.index_select(0, active_idx).view(1, K_active).expand(D, K_active)
		active_other = other_k.index_select(0, active_idx).view(1, K_active).expand(D, K_active)
		active_snap = snap_k.index_select(0, active_idx).view(1, K_active).expand(D, K_active)
		valid_control = valid_control.scatter(1, active_idx.view(1, K_active).expand(D, K_active), valid_active & active_control)
		valid_other = valid_other.scatter(1, active_idx.view(1, K_active).expand(D, K_active), valid_active & active_other)
		valid_snap = valid_snap.scatter(1, active_idx.view(1, K_active).expand(D, K_active), valid_active & active_snap)

		signed_delta = ((target - hit_xyz) * model_n).sum(dim=-1)
		h_cont = hit_rows.to(dtype=res.xyz_lr.dtype) + u
		w_cont = hit_cols.to(dtype=res.xyz_lr.dtype) + v
		d_flat = torch.arange(D, device=device, dtype=torch.long).view(D, 1).expand(D, K_active).reshape(-1)
		h_floor_flat = torch.floor(h_cont).to(dtype=torch.long).clamp(0, H - 1).reshape(-1)
		w_floor_flat = torch.floor(w_cont).to(dtype=torch.long).clamp(0, W - 1).reshape(-1)
		h_cont_flat = h_cont.reshape(-1)
		w_cont_flat = w_cont.reshape(-1)
		signed_flat = signed_delta.reshape(-1)
		normal_flat = model_n.reshape(-1, 3)
		mask_flat = valid_active.to(dtype=res.xyz_lr.dtype).reshape(-1)
		control_flat = active_control.reshape(-1)
		other_flat = active_other.reshape(-1)
		snap_flat = active_snap.reshape(-1)
		with torch.no_grad():
			if bool(valid_active.any().detach().cpu()):
				signed_delta_mean = float(signed_delta[valid_active].mean().detach().cpu())

		combined_loss, combined_maps, combined_masks, combined_active_verts, proxy_target_map, proxy_valid_map = _loss_from_splats(
			res=res,
			d_p=d_flat,
			h_floor_p=h_floor_flat,
			w_floor_p=w_floor_flat,
			h_cont_p=h_cont_flat,
			w_cont_p=w_cont_flat,
			signed_delta_p=signed_flat,
			normal_p=normal_flat,
			mask_p=mask_flat,
		)
		combined_item = (combined_loss, combined_maps, combined_masks)
		if debug_payload:
			with torch.no_grad():
				idx_cpu = active_idx.detach().cpu()
				target_full = torch.full((D, K, 3), float("nan"), dtype=torch.float32)
				hit_full = torch.full((D, K, 3), float("nan"), dtype=torch.float32)
				normal_full = torch.full((D, K, 3), float("nan"), dtype=torch.float32)
				signed_full = torch.full((D, K), float("nan"), dtype=torch.float32)
				sample_h_full = torch.full((D, K), float("nan"), dtype=torch.float32)
				sample_w_full = torch.full((D, K), float("nan"), dtype=torch.float32)
				valid_full = torch.zeros((D, K), dtype=torch.bool)
				control_full = control_k.view(1, K).expand(D, K).detach().cpu().to(dtype=torch.bool)
				snap_full = snap_k.view(1, K).expand(D, K).detach().cpu().to(dtype=torch.bool)
				target_full[:, idx_cpu] = target.detach().cpu().to(dtype=torch.float32)
				hit_full[:, idx_cpu] = hit_xyz.detach().cpu().to(dtype=torch.float32)
				normal_full[:, idx_cpu] = model_n.detach().cpu().to(dtype=torch.float32)
				signed_full[:, idx_cpu] = signed_delta.detach().cpu().to(dtype=torch.float32)
				sample_h_full[:, idx_cpu] = h_cont.detach().cpu().to(dtype=torch.float32)
				sample_w_full[:, idx_cpu] = w_cont.detach().cpu().to(dtype=torch.float32)
				valid_full[:, idx_cpu] = valid_active.detach().cpu().to(dtype=torch.bool)
				_last_debug_payload = AtlasLineDebugPayload(
					model_xyz=res.xyz_lr.detach().cpu(),
					valid=valid_full,
					target_xyz=target_full,
					hit_xyz=hit_full,
					model_normal=normal_full,
					signed_delta=signed_full,
					is_control=control_full,
					is_snap=snap_full,
					sample_model_h=sample_h_full,
					sample_model_w=sample_w_full,
					normal_proxy_target_xyz=proxy_target_map.detach().cpu().to(dtype=torch.float32),
					normal_proxy_valid=proxy_valid_map.detach().cpu().to(dtype=torch.bool),
					object_ids=_atlas_object_ids(lines, K),
					atlas_winding_model_ranges=tuple(getattr(lines, "atlas_winding_model_ranges", ()) or ()),
				)
		control_loss, control_maps, control_masks, control_active_verts, _control_proxy_target_map, _control_proxy_valid_map = _loss_from_splats(
			res=res,
			d_p=d_flat,
			h_floor_p=h_floor_flat,
			w_floor_p=w_floor_flat,
			h_cont_p=h_cont_flat,
			w_cont_p=w_cont_flat,
			signed_delta_p=signed_flat,
			normal_p=normal_flat,
			mask_p=mask_flat * control_flat.to(dtype=res.xyz_lr.dtype),
		)
		control_item = (control_loss, control_maps, control_masks)
		other_loss, other_maps, other_masks, other_active_verts, _other_proxy_target_map, _other_proxy_valid_map = _loss_from_splats(
			res=res,
			d_p=d_flat,
			h_floor_p=h_floor_flat,
			w_floor_p=w_floor_flat,
			h_cont_p=h_cont_flat,
			w_cont_p=w_cont_flat,
			signed_delta_p=signed_flat,
			normal_p=normal_flat,
			mask_p=mask_flat * other_flat.to(dtype=res.xyz_lr.dtype),
		)
		other_item = (other_loss, other_maps, other_masks)
		snap_loss, snap_maps, snap_masks, snap_active_verts, _snap_proxy_target_map, _snap_proxy_valid_map = _loss_from_splats(
			res=res,
			d_p=d_flat,
			h_floor_p=h_floor_flat,
			w_floor_p=w_floor_flat,
			h_cont_p=h_cont_flat,
			w_cont_p=w_cont_flat,
			signed_delta_p=signed_flat,
			normal_p=normal_flat,
			mask_p=mask_flat * snap_flat.to(dtype=res.xyz_lr.dtype),
		)
		snap_item = (snap_loss, snap_maps, snap_masks)

		with torch.no_grad():
			next_rows = _rows.clone()
			next_cols = _cols.clone()
			next_frac_h = _frac_h.clone()
			next_frac_w = _frac_w.clone()
			cache_ok = finite2 & t_valid & n0_valid
			next_rows[:, active_idx] = torch.where(cache_ok, row2, rows_active).detach()
			next_cols[:, active_idx] = torch.where(cache_ok, col2, cols_active).detach()
			next_frac_h[:, active_idx] = torch.where(cache_ok, u, frac_h_active).detach()
			next_frac_w[:, active_idx] = torch.where(cache_ok, v, frac_w_active).detach()
			_rows = next_rows.detach()
			_cols = next_cols.detach()
			_frac_h = next_frac_h.detach()
			_frac_w = next_frac_w.detach()

	with torch.no_grad():
		valid_count = float(valid.to(dtype=torch.float32).sum().detach().cpu())
		control_count = float(valid_control.to(dtype=torch.float32).sum().detach().cpu())
		other_count = float(valid_other.to(dtype=torch.float32).sum().detach().cpu())
		snap_count = float(valid_snap.to(dtype=torch.float32).sum().detach().cpu())
		_last_stats = {
			"atlas_line_samples": float(D * K),
			"atlas_line_valid": valid_count,
			"atlas_line_rms": float(torch.sqrt(combined_item[0]).detach().cpu()),
			"atlas_line_active_vertices": combined_active_verts,
			"atlas_line_signed_delta_mean": signed_delta_mean,
			"atlas_line_control_valid": control_count,
			"atlas_line_control_rms": float(torch.sqrt(control_item[0]).detach().cpu()),
			"atlas_line_control_active_vertices": control_active_verts,
			"atlas_line_other_valid": other_count,
			"atlas_line_other_rms": float(torch.sqrt(other_item[0]).detach().cpu()),
			"atlas_line_other_active_vertices": other_active_verts,
			"atlas_line_snap_valid": snap_count,
			"atlas_line_snap_rms": float(torch.sqrt(snap_item[0]).detach().cpu()),
			"atlas_line_snap_active_vertices": snap_active_verts,
		}

	return {
		"atlas_line": combined_item,
		"atlas_line_control": control_item,
		"atlas_line_other": other_item,
		"atlas_line_snap": snap_item,
	}
