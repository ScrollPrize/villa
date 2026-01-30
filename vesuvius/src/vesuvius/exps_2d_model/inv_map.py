from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_xy(*, xy_lr: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
	"""Sample xy_lr (N,Hm,Wm,2) at uv (N,H,W,2) in index units (x=ix, y=iy)."""
	if xy_lr.ndim != 4 or int(xy_lr.shape[-1]) != 2:
		raise ValueError("xy_lr must be (N,Hm,Wm,2)")
	if uv.ndim != 4 or int(uv.shape[-1]) != 2:
		raise ValueError("uv must be (N,H,W,2)")

	n, hm, wm, _c2 = (int(v) for v in xy_lr.shape)
	uvx = uv[..., 0]
	uvy = uv[..., 1]
	xn = (2.0 * (uvx / float(max(1, wm - 1)))) - 1.0
	yn = (2.0 * (uvy / float(max(1, hm - 1)))) - 1.0
	grid = torch.stack([xn, yn], dim=-1)
	src = xy_lr.permute(0, 3, 1, 2).contiguous()
	out = F.grid_sample(src, grid, mode="bilinear", align_corners=True, padding_mode="border")
	return out.permute(0, 2, 3, 1).contiguous()


def inverse_map_autograd(
	*,
	xy_lr: torch.Tensor,
	h_out: int,
	w_out: int,
	iters: int = 50,
	step_size: float = 0.3,
	eps_det: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Invert xy_lr by optimizing uv using autograd.

	We minimize per-pixel squared reprojection error:
		E(uv) = 0.5 * ||sample(xy_lr, uv) - target_xy||^2

	Returns:
	- uv: (N,h_out,w_out,2) in index units (x=ix in [0,Wm-1], y=iy in [0,Hm-1]).
	- mask: (N,1,h_out,w_out) float mask, 1 where the local Jacobian is usable.
	"""
	if xy_lr.ndim != 4 or int(xy_lr.shape[-1]) != 2:
		raise ValueError("xy_lr must be (N,Hm,Wm,2)")
	n, hm, wm, _c2 = (int(v) for v in xy_lr.shape)
	h_out = int(max(1, h_out))
	w_out = int(max(1, w_out))
	iters = int(max(0, iters))
	step_size = float(step_size)

	def clamp_uv(u: torch.Tensor) -> torch.Tensor:
		x = u[..., 0].clamp(0.0, float(max(1, wm - 1)))
		y = u[..., 1].clamp(0.0, float(max(1, hm - 1)))
		return torch.stack([x, y], dim=-1)

	device = xy_lr.device
	dtype = xy_lr.dtype
	xs = torch.linspace(0.0, float(max(1, w_out - 1)), w_out, device=device, dtype=dtype)
	ys = torch.linspace(0.0, float(max(1, h_out - 1)), h_out, device=device, dtype=dtype)
	yy, xx = torch.meshgrid(ys, xs, indexing="ij")
	tgt_xy = torch.stack([xx, yy], dim=-1).view(1, h_out, w_out, 2).expand(n, h_out, w_out, 2)

	xy_flat = xy_lr.view(n, -1, 2)
	min_xy = xy_flat.amin(dim=1).view(n, 1, 1, 2)
	max_xy = xy_flat.amax(dim=1).view(n, 1, 1, 2)
	scale_xy = (max_xy - min_xy).clamp_min(1e-6)
	uv0 = (tgt_xy - min_xy) / scale_xy
	uv0x = uv0[..., 0] * float(max(1, wm - 1))
	uv0y = uv0[..., 1] * float(max(1, hm - 1))
	uv = torch.stack([uv0x, uv0y], dim=-1)
	uv = clamp_uv(uv)
	uv = uv.detach().requires_grad_(True)

	for _ in range(iters):
		xy = _sample_xy(xy_lr=xy_lr, uv=uv)
		err = xy - tgt_xy
		loss = 0.5 * (err * err).sum(dim=-1)
		g = torch.autograd.grad(loss.sum(), uv, create_graph=False, retain_graph=False)[0]
		uv = clamp_uv(uv - step_size * g).detach().requires_grad_(True)

	# Jacobian usability mask (finite-diff is fine here; only for visualization gating).
	du = 1.0
	with torch.no_grad():
		xy = _sample_xy(xy_lr=xy_lr, uv=uv)
		uv_u = uv.clone()
		uv_v = uv.clone()
		uv_u[..., 0] = (uv_u[..., 0] + du).clamp(0.0, float(max(1, wm - 1)))
		uv_v[..., 1] = (uv_v[..., 1] + du).clamp(0.0, float(max(1, hm - 1)))
		xy_u = _sample_xy(xy_lr=xy_lr, uv=uv_u)
		xy_v = _sample_xy(xy_lr=xy_lr, uv=uv_v)
		j_u = (xy_u - xy)
		j_v = (xy_v - xy)
		a = j_u[..., 0]
		b = j_v[..., 0]
		c = j_u[..., 1]
		d = j_v[..., 1]
		det = a * d - b * c
		ok = det.abs() > float(eps_det)
		mask = ok.to(dtype=torch.float32).view(n, 1, h_out, w_out)
		return uv.detach(), mask


def warp_nchw_from_uv(
	*,
	src: torch.Tensor,
	uv: torch.Tensor,
	uv_mask: torch.Tensor | None = None,
	fill: float = 0.5,
) -> torch.Tensor:
	"""Warp a base-grid tensor `src` into image space using `uv`.

	- src: (N,C,Hm,Wm)
	- uv: (N,H,W,2) index units for src (x in [0,Wm-1], y in [0,Hm-1])
	- uv_mask: (N,1,H,W) optional mask; invalid pixels are set to `fill`.
	"""
	if src.ndim != 4:
		raise ValueError("src must be (N,C,Hm,Wm)")
	if uv.ndim != 4 or int(uv.shape[-1]) != 2:
		raise ValueError("uv must be (N,H,W,2)")
	n, _c, hm, wm = (int(v) for v in src.shape)
	if int(uv.shape[0]) != n:
		raise ValueError("uv batch must match src")
	xn = (2.0 * (uv[..., 0] / float(max(1, wm - 1)))) - 1.0
	yn = (2.0 * (uv[..., 1] / float(max(1, hm - 1)))) - 1.0
	grid = torch.stack([xn, yn], dim=-1)
	out = F.grid_sample(src, grid, mode="bilinear", align_corners=True, padding_mode="border")
	if uv_mask is None:
		return out
	return out * uv_mask + float(fill) * (1.0 - uv_mask)
