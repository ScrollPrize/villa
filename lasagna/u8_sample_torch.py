"""Device-agnostic uint8 3D trilinear sampling in plain PyTorch ops.

These match the CUDA kernels' arithmetic (``grid_sample_3d_u8*_kernel.cu``,
``sparse_grid_sample_3d_u8*_kernel.cu`` and ``sparse_prefetch_chunks_kernel.cu``):
the same float32 coordinate transform, chunk and corner choice, clamping, zeroing
rules and round-half-away-from-zero for uint8 output. They run on the CPU and on
MPS, serve as the fallback when no native kernel is available, and as the readable
reference the Metal kernels are tested against.

Sparse layout off CUDA: the chunk table holds int32 pool slots (``slot + 1``, 0 =
not loaded) into one ``(S, C, 34, 34, 34)`` uint8 pool, instead of raw device
pointers. Gradients flow only to the sample positions, as in the CUDA kernels.
"""
from __future__ import annotations

import torch

CHUNK = 32
PADDED = CHUNK + 2
PADDED3 = PADDED ** 3


def _corner_weights(f: torch.Tensor) -> torch.Tensor:
	"""(N, 8) trilinear weights in the kernels' corner order: 000 100 010 110 001 101 011 111 (x, y, z)."""
	fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
	gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
	return torch.stack(
		[gx * gy * gz, fx * gy * gz, gx * fy * gz, fx * fy * gz,
		 gx * gy * fz, fx * gy * fz, gx * fy * fz, fx * fy * fz],
		dim=1,
	)


def _round_half_away_u8(val: torch.Tensor) -> torch.Tensor:
	"""CUDA's ``(uint8)clamp((int)roundf(v), 0, 255)``; torch.round would round ties to even."""
	r = torch.floor(val)
	r = r + ((val - r) >= 0.5).to(val.dtype)
	neg = val < 0
	# roundf rounds -x.5 away from zero too
	r = torch.where(neg, -torch.floor(-val + 0.5), r)
	return r.clamp_(0.0, 255.0).to(torch.uint8)


def _local_coords(grid: torch.Tensor, offset: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
	"""(N, 3) float32 volume-index coordinates (x, y, z), computed per component as the kernels do."""
	g = grid.reshape(-1, 3)
	return (g - offset.view(1, 3).to(g.dtype)) * inv_scale.view(1, 3).to(g.dtype)


def sparse_sample(
	table: torch.Tensor,
	pool: torch.Tensor,
	grid: torch.Tensor,
	offset: torch.Tensor,
	inv_scale: torch.Tensor,
	*,
	diff: bool,
) -> torch.Tensor:
	"""Trilinear sample from a slot-indexed sparse chunk pool.

	Args:
		table: (cZ, cY, cX) int32, slot + 1 per loaded chunk, 0 if not loaded
		pool: (S, C, 34, 34, 34) uint8 padded chunks
		grid: (D, H, W, 3) float32 fullres coordinates (x, y, z)
		offset, inv_scale: (3,) float32

	Returns:
		(C, D, H, W): float32 raw values when ``diff`` (differentiable w.r.t. ``grid``), else uint8
	"""
	D, H, W, _ = grid.shape
	C = int(pool.shape[1]) if pool.dim() == 5 else 1
	cZ, cY, cX = (int(v) for v in table.shape)
	p = _local_coords(grid if diff else grid.detach(), offset, inv_scale)
	finite = torch.isfinite(p).all(dim=1)
	p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
	pd = p.detach()
	ci = torch.floor(pd / float(CHUNK))
	lim = torch.tensor([cX, cY, cZ], dtype=ci.dtype, device=ci.device)
	inb = finite & (ci >= 0).all(dim=1) & (ci < lim).all(dim=1)
	cil = torch.where(inb.unsqueeze(1), ci, torch.zeros_like(ci)).to(torch.long)
	slot = table[cil[:, 2], cil[:, 1], cil[:, 0]].to(torch.long)
	valid = inb & (slot > 0)
	base = (slot - 1).clamp_min(0) * (C * PADDED3)

	local = p - ci * float(CHUNK) + 1.0
	l0 = torch.floor(local.detach())
	f = local - l0
	i0 = l0.clamp(0, PADDED - 1).to(torch.long)
	i1 = (i0 + 1).clamp(0, PADDED - 1)
	xs = torch.stack([i0[:, 0], i1[:, 0]], 1)
	ys = torch.stack([i0[:, 1], i1[:, 1]], 1)
	zs = torch.stack([i0[:, 2], i1[:, 2]], 1)
	# corner order 000 100 010 110 001 101 011 111 over (x, y, z)
	bx = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=p.device)
	by = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=p.device)
	bz = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=p.device)
	corner = zs[:, bz] * (PADDED * PADDED) + ys[:, by] * PADDED + xs[:, bx]  # (N, 8)
	chan = torch.arange(C, device=p.device).view(1, C, 1) * PADDED3
	idx = base.view(-1, 1, 1) + chan + corner.unsqueeze(1)  # (N, C, 8)
	vals = pool.reshape(-1)[idx.reshape(-1)].view(idx.shape).to(torch.float32)

	w = _corner_weights(f).unsqueeze(1)  # (N, 1, 8)
	out = (w * vals).sum(dim=2)  # (N, C)
	out = torch.where(valid.unsqueeze(1), out, torch.zeros_like(out))
	out = out.t().reshape(C, D, H, W)
	if diff:
		return out
	return _round_half_away_u8(out)


def dense_sample(
	volume: torch.Tensor,
	grid: torch.Tensor,
	offset: torch.Tensor,
	inv_scale: torch.Tensor,
	*,
	diff: bool,
) -> torch.Tensor:
	"""Trilinear sample of a dense (C, Z, Y, X) uint8 volume with zeroed out-of-range corners.

	Returns (C, D, H, W): float32 raw values when ``diff`` (differentiable w.r.t. ``grid``), else uint8.
	"""
	D, H, W, _ = grid.shape
	C, Z, Y, X = (int(v) for v in volume.shape)
	p = _local_coords(grid if diff else grid.detach(), offset, inv_scale)
	# CUDA's corner checks all fail for a non-finite point; zero it here so no NaN reaches the
	# weights, where 0 * NaN would poison the backward pass
	finite = torch.isfinite(p).all(dim=1)
	p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
	l0 = torch.floor(p.detach())
	f = p - l0
	i0 = l0.to(torch.long)
	i1 = i0 + 1
	lim = torch.tensor([X, Y, Z], dtype=torch.long, device=p.device)
	ok0 = (i0 >= 0) & (i0 < lim)
	ok1 = (i1 >= 0) & (i1 < lim)
	xs = torch.stack([i0[:, 0], i1[:, 0]], 1)
	ys = torch.stack([i0[:, 1], i1[:, 1]], 1)
	zs = torch.stack([i0[:, 2], i1[:, 2]], 1)
	oxs = torch.stack([ok0[:, 0], ok1[:, 0]], 1)
	oys = torch.stack([ok0[:, 1], ok1[:, 1]], 1)
	ozs = torch.stack([ok0[:, 2], ok1[:, 2]], 1)
	bx = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=p.device)
	by = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=p.device)
	bz = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=p.device)
	ok = oxs[:, bx] & oys[:, by] & ozs[:, bz] & finite.unsqueeze(1)  # (N, 8)
	lin = (zs[:, bz] * Y + ys[:, by]) * X + xs[:, bx]
	lin = torch.where(ok, lin, torch.zeros_like(lin))
	chan = torch.arange(C, device=p.device).view(1, C, 1) * (Z * Y * X)
	idx = chan + lin.unsqueeze(1)  # (N, C, 8)
	vals = volume.reshape(-1)[idx.reshape(-1)].view(idx.shape).to(torch.float32)
	vals = torch.where(ok.unsqueeze(1), vals, torch.zeros_like(vals))
	w = _corner_weights(f).unsqueeze(1)
	out = (w * vals).sum(dim=2)
	out = out.t().reshape(C, D, H, W)
	if diff:
		return out
	return _round_half_away_u8(out)


def missing_chunks(
	xyz_fullres: torch.Tensor,
	table: torch.Tensor,
	origin: torch.Tensor,
	spacing: torch.Tensor,
) -> torch.Tensor:
	"""(M, 3) int64 (cz, cy, cx) of unloaded chunks in the 26-neighbourhood of each sample's chunk.

	Matches ``sparse_prefetch_chunks_kernel.cu``: chunk index by truncation toward zero, non-finite
	components mapped to chunk 0, then clamped into the grid.
	"""
	cZ, cY, cX = (int(v) for v in table.shape)
	xyz = xyz_fullres.detach().reshape(-1, 3).to(torch.float32)
	c = (xyz - origin.view(1, 3)) / spacing.view(1, 3) / float(CHUNK)
	c = torch.where(torch.isfinite(xyz), c, torch.zeros_like(c))
	c = torch.trunc(c).clamp(-1, max(cX, cY, cZ)).to(torch.long)
	c = torch.stack([c[:, 0].clamp(0, cX - 1), c[:, 1].clamp(0, cY - 1), c[:, 2].clamp(0, cZ - 1)], 1)
	lin = torch.unique((c[:, 2] * cY + c[:, 1]) * cX + c[:, 0])
	needed = torch.zeros(cZ * cY * cX, dtype=torch.bool, device=table.device)
	cz, rem = lin // (cY * cX), lin % (cY * cX)
	cy, cx = rem // cX, rem % cX
	for dz in (-1, 0, 1):
		zz = cz + dz
		for dy in (-1, 0, 1):
			yy = cy + dy
			for dx in (-1, 0, 1):
				xx = cx + dx
				ok = (zz >= 0) & (zz < cZ) & (yy >= 0) & (yy < cY) & (xx >= 0) & (xx < cX)
				needed[((zz * cY + yy) * cX + xx)[ok]] = True
	missing = needed.view(cZ, cY, cX) & (table == 0)
	return torch.nonzero(missing)
