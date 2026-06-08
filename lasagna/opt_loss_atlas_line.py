from __future__ import annotations

import torch

import model as fit_model


_rows: torch.Tensor | None = None
_cols: torch.Tensor | None = None
_frac_h: torch.Tensor | None = None
_frac_w: torch.Tensor | None = None
_last_stats: dict[str, float] = {}


def reset_state() -> None:
	global _rows, _cols, _frac_h, _frac_w, _last_stats
	_rows = None
	_cols = None
	_frac_h = None
	_frac_w = None
	_last_stats = {}


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


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


def atlas_line_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	global _rows, _cols, _frac_h, _frac_w, _last_stats
	lines = getattr(res.data, "atlas_lines", None)
	if lines is None:
		raise ValueError("atlas_line loss requires FitData3D.atlas_lines")
	if int(lines.target_xyz.shape[0]) <= 0:
		z = res.xyz_lr.sum() * 0.0
		return z, (z.view(1, 1, 1, 1),), (torch.zeros_like(z).view(1, 1, 1, 1),)

	_ensure_state(res=res)
	assert _rows is not None and _cols is not None and _frac_h is not None and _frac_w is not None
	D, H, W, _ = res.xyz_lr.shape
	K = int(lines.target_xyz.shape[0])
	device = res.xyz_lr.device
	if H < 2 or W < 2:
		raise ValueError(f"atlas_line requires model H/W >= 2, got {H}x{W}")

	target = lines.target_xyz.to(device=device, dtype=res.xyz_lr.dtype).view(1, K, 3).expand(D, K, 3)
	normal = lines.normal_xyz.to(device=device, dtype=res.xyz_lr.dtype).view(1, K, 3).expand(D, K, 3)
	n_valid = torch.isfinite(normal).all(dim=-1) & (torch.linalg.vector_norm(normal, dim=-1) > 1.0e-8)
	t_valid = torch.isfinite(target).all(dim=-1)
	in_bounds = (_rows >= 0) & (_rows < H - 1) & (_cols >= 0) & (_cols < W - 1)
	valid0 = n_valid & t_valid & in_bounds

	xyz_det = res.xyz_lr.detach()
	M00, M10, M01, M11 = _gather_quads(xyz_det, _rows, _cols)
	u1, v1 = _intersect_quad(target, normal, M00, M10, M01, M11, _frac_h, _frac_w)
	finite1 = torch.isfinite(u1) & torch.isfinite(v1)
	step_h = torch.where(u1 < 0.0, torch.full_like(_rows, -1), torch.where(u1 > 1.0, torch.ones_like(_rows), torch.zeros_like(_rows)))
	step_w = torch.where(v1 < 0.0, torch.full_like(_cols, -1), torch.where(v1 > 1.0, torch.ones_like(_cols), torch.zeros_like(_cols)))
	row2 = torch.where(finite1, (_rows + step_h).clamp(0, H - 2), _rows)
	col2 = torch.where(finite1, (_cols + step_w).clamp(0, W - 2), _cols)
	frac_h1 = torch.where(finite1, u1.clamp(0.0, 1.0), _frac_h)
	frac_w1 = torch.where(finite1, v1.clamp(0.0, 1.0), _frac_w)

	Q00, Q10, Q01, Q11 = _gather_quads(xyz_det, row2, col2)
	u2, v2 = _intersect_quad(target, normal, Q00, Q10, Q01, Q11, frac_h1, frac_w1)
	finite2 = torch.isfinite(u2) & torch.isfinite(v2)
	u = torch.where(finite2, u2.clamp(0.0, 1.0), frac_h1).detach()
	v = torch.where(finite2, v2.clamp(0.0, 1.0), frac_w1).detach()
	valid = valid0 & finite1 & finite2

	G00, G10, G01, G11 = _gather_quads(res.xyz_lr, row2, col2)
	model_pt = _bilinear(G00, G10, G01, G11, u, v)
	sq = ((model_pt - target) ** 2).sum(dim=-1)
	sq = torch.where(valid, sq, torch.zeros_like(sq))
	den = valid.to(dtype=res.xyz_lr.dtype).sum().clamp_min(1.0)
	loss = sq.sum() / den

	with torch.no_grad():
		_rows = torch.where(valid, row2, _rows).detach()
		_cols = torch.where(valid, col2, _cols).detach()
		_frac_h = torch.where(valid, u, _frac_h).detach()
		_frac_w = torch.where(valid, v, _frac_w).detach()
		valid_count = float(valid.to(dtype=torch.float32).sum().detach().cpu())
		_last_stats = {
			"atlas_line_samples": float(D * K),
			"atlas_line_valid": valid_count,
			"atlas_line_rms": float(torch.sqrt(sq.sum() / den).detach().cpu()),
		}

	lm = sq.reshape(D, 1, 1, K)
	mask = valid.to(dtype=res.xyz_lr.dtype).reshape(D, 1, 1, K)
	return loss, (lm,), (mask,)
