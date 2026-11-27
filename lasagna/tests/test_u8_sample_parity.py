"""Parity of the portable uint8 samplers against a float64 reference of the CUDA kernels' math.

The reference computes sample coordinates in float32 exactly as the ``.cu`` kernels do (so the chunk, corner
and floor choices match) and everything after that in float64. Both the PyTorch fallback (CPU and MPS) and the
Metal kernels (MPS) must meet the same bars, and two deliberately wrong variants must fail them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import u8_sample_torch as ref_torch

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="needs an Apple GPU")

FWD_TOL = 2e-4  # absolute, on the 0-255 value scale
BWD_REL, BWD_ABS = 1e-4, 1e-5
TIE_BAND = 1e-3


# ---------------------------------------------------------------- reference (numpy, independent of the code under test)

def _f32_coords(grid: np.ndarray, offset: np.ndarray, inv_scale: np.ndarray) -> np.ndarray:
	g = grid.reshape(-1, 3).astype(np.float32)
	return (g - offset.astype(np.float32)) * inv_scale.astype(np.float32)


def _weights(f: np.ndarray) -> np.ndarray:
	fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
	gx, gy, gz = 1 - fx, 1 - fy, 1 - fz
	return np.stack([gx * gy * gz, fx * gy * gz, gx * fy * gz, fx * fy * gz,
					 gx * gy * fz, fx * gy * fz, gx * fy * fz, fx * fy * fz], 1)


def _dweights(f: np.ndarray) -> np.ndarray:
	"""(N, 3, 8): derivative of each corner weight w.r.t. fx, fy, fz."""
	fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
	gx, gy, gz = 1 - fx, 1 - fy, 1 - fz
	dx = np.stack([-gy * gz, gy * gz, -fy * gz, fy * gz, -gy * fz, gy * fz, -fy * fz, fy * fz], 1)
	dy = np.stack([-gx * gz, -fx * gz, gx * gz, fx * gz, -gx * fz, -fx * fz, gx * fz, fx * fz], 1)
	dz = np.stack([-gx * gy, -fx * gy, -gx * fy, -fx * fy, gx * gy, fx * gy, gx * fy, fx * fy], 1)
	return np.stack([dx, dy, dz], 1)


BX = np.array([0, 1, 0, 1, 0, 1, 0, 1])
BY = np.array([0, 0, 1, 1, 0, 0, 1, 1])
BZ = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def ref_sparse(table, pool, grid, offset, inv_scale, grad_out=None, *, halo=1.0):
	"""float64 values (C, N) and, with grad_out (C, N), grid gradients (N, 3) of the sparse kernels."""
	cZ, cY, cX = table.shape
	C = pool.shape[1]
	p = _f32_coords(grid, offset, inv_scale)
	N = p.shape[0]
	out = np.zeros((C, N))
	grad = np.zeros((N, 3))
	with np.errstate(invalid="ignore"):
		finite = np.isfinite(p).all(1)
		ps = np.where(finite[:, None], p, 0).astype(np.float32)
		ci = np.floor(ps / np.float32(32)).astype(np.int64)
	inb = finite & (ci >= 0).all(1) & (ci < np.array([cX, cY, cZ])).all(1)
	cil = np.where(inb[:, None], ci, 0)
	slot = table[cil[:, 2], cil[:, 1], cil[:, 0]].astype(np.int64)
	valid = inb & (slot > 0)
	local = (ps - (cil * 32).astype(np.float32) + np.float32(halo)).astype(np.float32)
	l0 = np.floor(local)
	f = (local - l0).astype(np.float64)
	i0 = np.clip(l0.astype(np.int64), 0, 33)
	i1 = np.clip(i0 + 1, 0, 33)
	xs, ys, zs = np.stack([i0[:, 0], i1[:, 0]], 1), np.stack([i0[:, 1], i1[:, 1]], 1), np.stack([i0[:, 2], i1[:, 2]], 1)
	w = _weights(f)
	dw = _dweights(f)
	isc = inv_scale.astype(np.float64)
	for c in range(C):
		vals = pool[np.maximum(slot - 1, 0)[:, None], c, zs[:, BZ], ys[:, BY], xs[:, BX]].astype(np.float64)
		out[c] = np.where(valid, (w * vals).sum(1), 0)
		if grad_out is not None:
			d = (dw * vals[:, None, :]).sum(2)  # (N, 3)
			grad += np.where(valid[:, None], grad_out[c].astype(np.float64)[:, None] * d * isc[None, :], 0)
	return out, grad


def ref_dense(volume, grid, offset, inv_scale, grad_out=None):
	"""float64 values (C, N) and, with grad_out, grid gradients (N, 3) of the dense kernels."""
	C, Z, Y, X = volume.shape
	p = _f32_coords(grid, offset, inv_scale)
	N = p.shape[0]
	out = np.zeros((C, N))
	grad = np.zeros((N, 3))
	with np.errstate(invalid="ignore"):
		finite = np.isfinite(p).all(1)
		ps = np.where(finite[:, None], p, 0).astype(np.float32)
	l0 = np.floor(ps)
	f = (ps - l0).astype(np.float64)
	i0 = l0.astype(np.int64)
	lim = np.array([X, Y, Z])
	xs = np.stack([i0[:, 0], i0[:, 0] + 1], 1)
	ys = np.stack([i0[:, 1], i0[:, 1] + 1], 1)
	zs = np.stack([i0[:, 2], i0[:, 2] + 1], 1)
	ok = (finite[:, None] & (xs[:, BX] >= 0) & (xs[:, BX] < lim[0]) & (ys[:, BY] >= 0) & (ys[:, BY] < lim[1])
		  & (zs[:, BZ] >= 0) & (zs[:, BZ] < lim[2]))
	w = _weights(f)
	dw = _dweights(f)
	isc = inv_scale.astype(np.float64)
	for c in range(C):
		vals = np.where(ok, volume[c, np.where(ok, zs[:, BZ], 0), np.where(ok, ys[:, BY], 0),
								   np.where(ok, xs[:, BX], 0)].astype(np.float64), 0)
		out[c] = (w * vals).sum(1)
		if grad_out is not None:
			d = (dw * vals[:, None, :]).sum(2)
			grad += grad_out[c].astype(np.float64)[:, None] * d * isc[None, :]
	return out, grad


def ref_missing(xyz, table, origin, spacing):
	cZ, cY, cX = table.shape
	x = xyz.reshape(-1, 3).astype(np.float32)
	with np.errstate(invalid="ignore"):
		c = (x - origin.astype(np.float32)) / spacing.astype(np.float32) / np.float32(32)
		c = np.where(np.isfinite(x), c, 0)
		c = np.trunc(np.clip(c, -2, max(cX, cY, cZ) + 1)).astype(np.int64)
	c = np.stack([np.clip(c[:, 0], 0, cX - 1), np.clip(c[:, 1], 0, cY - 1), np.clip(c[:, 2], 0, cZ - 1)], 1)
	needed = np.zeros((cZ, cY, cX), bool)
	for dz in (-1, 0, 1):
		for dy in (-1, 0, 1):
			for dx in (-1, 0, 1):
				zz, yy, xx = c[:, 2] + dz, c[:, 1] + dy, c[:, 0] + dx
				ok = (zz >= 0) & (zz < cZ) & (yy >= 0) & (yy < cY) & (xx >= 0) & (xx < cX)
				needed[zz[ok], yy[ok], xx[ok]] = True
	return {tuple(v) for v in np.argwhere(needed & (table == 0)).tolist()}


# ---------------------------------------------------------------- bars

def fwd_ok(got: np.ndarray, want: np.ndarray) -> tuple[bool, float]:
	err = float(np.abs(got.astype(np.float64) - want).max()) if want.size else 0.0
	return err <= FWD_TOL, err


def bwd_ok(got: np.ndarray, want: np.ndarray) -> tuple[bool, float]:
	err = np.abs(got.astype(np.float64) - want)
	bound = BWD_REL * np.abs(want).max(axis=0, keepdims=True) + BWD_ABS
	return bool((err <= bound).all()), float(err.max())


def u8_ok(got: np.ndarray, want: np.ndarray) -> tuple[bool, int]:
	"""Exact uint8 match except where the f64 value lies within TIE_BAND of a .5 tie."""
	exact = np.clip(np.floor(want + 0.5), 0, 255)
	near_tie = np.abs((want - np.floor(want)) - 0.5) < TIE_BAND
	bad = (got.astype(np.float64) != exact) & ~near_tie
	excused = int(((got.astype(np.float64) != exact) & near_tie).sum())
	return (not bad.any()) and excused <= max(1, int(1e-4 * want.size)), int(bad.sum())


# ---------------------------------------------------------------- fixtures

def _volume(rng, C=3, shape=(70, 80, 66)):
	return rng.integers(0, 256, (C, *shape), dtype=np.uint8)


def _bricks(volume, coords):
	"""(S, C, 34, 34, 34) padded bricks of `coords` (cz, cy, cx), zero outside the volume."""
	C = volume.shape[0]
	pool = np.zeros((len(coords), C, 34, 34, 34), np.uint8)
	for i, coord in enumerate(coords):
		src = tuple(slice(max(0, c * 32 - 1), min(n, c * 32 + 33)) for c, n in zip(coord, volume.shape[1:]))
		dst = tuple(slice(s.start - (c * 32 - 1), s.stop - (c * 32 - 1)) for s, c in zip(src, coord))
		pool[(slice(i, i + 1), slice(None), *dst)] = volume[(slice(None), *src)]
	return pool


def _sparse_fixture(rng, volume, drop=0.25):
	C, Z, Y, X = volume.shape
	grid_shape = tuple((n + 31) // 32 for n in (Z, Y, X))
	coords = [tuple(c) for c in np.ndindex(*grid_shape) if rng.random() > drop]
	rng.shuffle(coords)
	table = np.zeros(grid_shape, np.int32)
	for i, c in enumerate(coords):
		table[c] = i + 1
	return table, _bricks(volume, coords)


def _grid(rng, extent_xyz, offset, spacing, n=(5, 17, 23)):
	"""(D, H, W, 3) fullres points: random plus chunk borders, integers, halves, NaN and Inf, non-contiguous."""
	lo = offset - 6 * spacing
	hi = offset + (np.array(extent_xyz) + 6) * spacing
	pts = rng.uniform(lo, hi, (int(np.prod(n)), 3)).astype(np.float32)
	k = pts.shape[0]
	special = []
	for b in (0, 31, 32, 33, 63, 64, 65):
		special.append(offset + np.array([b, b - 0.5, b + 0.25]) * spacing)
		special.append(offset + np.array([b + 0.5, 3.0, 7.0]) * spacing)
	special += [offset + np.array([-0.5, 2, 2]) * spacing, offset + np.array([-1.0, 2, 2]) * spacing,
				[np.nan, 1, 1], [1, np.inf, 1], [1, 1, -np.inf]]
	special = np.array(special, np.float32)
	pts[: len(special)] = special
	pts[len(special):len(special) + k // 4] = np.round(pts[len(special):len(special) + k // 4] / spacing) * spacing
	g = pts.reshape(*n, 3)
	# non-contiguous: stored channel-first and viewed as (D, H, W, 3)
	return np.ascontiguousarray(g.transpose(3, 0, 1, 2))


def _as_grid(t_cdhw: torch.Tensor) -> torch.Tensor:
	return t_cdhw.permute(1, 2, 3, 0)


OFFSET = np.array([3.25, -7.5, 1.0], np.float32)
SPACING = np.array([2.0, 2.0, 2.0], np.float32)
INV = (1.0 / SPACING).astype(np.float32)


def _impls():
	yield "torch-cpu", "cpu", ref_torch
	if MPS:
		yield "torch-mps", "mps", ref_torch
		import u8_sample_metal
		yield "metal", "mps", u8_sample_metal


IMPLS = list(_impls())


@pytest.fixture(params=[i[0] for i in IMPLS])
def impl(request):
	return next(i for i in IMPLS if i[0] == request.param)


def _run_sparse(mod, device, table, pool, grid_np, grad_out_np):
	t = torch.from_numpy(table).to(device)
	pl = torch.from_numpy(pool).to(device)
	g = _as_grid(torch.from_numpy(grid_np).to(device)).requires_grad_(True)
	off = torch.from_numpy(OFFSET).to(device)
	inv = torch.from_numpy(INV).to(device)
	out = mod.sparse_sample(t, pl, g, off, inv, diff=True)
	out.backward(torch.from_numpy(grad_out_np).to(device))
	u8 = mod.sparse_sample(t, pl, g.detach(), off, inv, diff=False)
	return out.detach().cpu().numpy(), g.grad.cpu().numpy(), u8.cpu().numpy()


def _run_dense(mod, device, volume, grid_np, grad_out_np):
	v = torch.from_numpy(volume).to(device)
	g = _as_grid(torch.from_numpy(grid_np).to(device)).requires_grad_(True)
	off = torch.from_numpy(OFFSET).to(device)
	inv = torch.from_numpy(INV).to(device)
	out = mod.dense_sample(v, g, off, inv, diff=True)
	out.backward(torch.from_numpy(grad_out_np).to(device))
	u8 = mod.dense_sample(v, g.detach(), off, inv, diff=False)
	return out.detach().cpu().numpy(), g.grad.cpu().numpy(), u8.cpu().numpy()


# ---------------------------------------------------------------- tests

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_matches_reference(impl, seed):
	name, device, mod = impl
	rng = np.random.default_rng(seed)
	volume = _volume(rng)
	table, pool = _sparse_fixture(rng, volume)
	grid_np = _grid(rng, volume.shape[:0:-1], OFFSET, SPACING)
	C = volume.shape[0]
	n = grid_np.shape[1:]
	grad_out = rng.standard_normal((C, *n)).astype(np.float32)
	out, grad, u8 = _run_sparse(mod, device, table, pool, grid_np, grad_out)
	want, want_grad = ref_sparse(table, pool, np.moveaxis(grid_np, 0, -1), OFFSET, INV, grad_out.reshape(C, -1))
	ok, err = fwd_ok(out.reshape(C, -1), want)
	assert ok, f"{name} sparse fwd max err {err}"
	ok, err = bwd_ok(grad.reshape(-1, 3), want_grad)
	assert ok, f"{name} sparse bwd max err {err}"
	ok, bad = u8_ok(u8.reshape(C, -1), want)
	assert ok, f"{name} sparse u8 mismatches {bad}"
	# the fixture must exercise every zeroing rule and real samples
	assert (want != 0).mean() > 0.3 and (want == 0).mean() > 0.05


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dense_matches_reference(impl, seed):
	name, device, mod = impl
	rng = np.random.default_rng(10 + seed)
	volume = _volume(rng)
	grid_np = _grid(rng, volume.shape[:0:-1], OFFSET, SPACING)
	C = volume.shape[0]
	n = grid_np.shape[1:]
	grad_out = rng.standard_normal((C, *n)).astype(np.float32)
	out, grad, u8 = _run_dense(mod, device, volume, grid_np, grad_out)
	want, want_grad = ref_dense(volume, np.moveaxis(grid_np, 0, -1), OFFSET, INV, grad_out.reshape(C, -1))
	ok, err = fwd_ok(out.reshape(C, -1), want)
	assert ok, f"{name} dense fwd max err {err}"
	ok, err = bwd_ok(grad.reshape(-1, 3), want_grad)
	assert ok, f"{name} dense bwd max err {err}"
	ok, bad = u8_ok(u8.reshape(C, -1), want)
	assert ok, f"{name} dense u8 mismatches {bad}"


def test_missing_chunks_match_reference(impl):
	name, device, mod = impl
	rng = np.random.default_rng(5)
	table = (rng.random((5, 6, 7)) < 0.4).astype(np.int32)
	xyz = rng.uniform(-200, 700, (4000, 3)).astype(np.float32)
	xyz[:3] = [[np.nan, 5, 5], [5, np.inf, 5], [-0.5, -40, 1e12]]
	origin = np.array([1.5, -3.0, 2.0], np.float32)
	spacing = np.array([2.0, 2.0, 2.0], np.float32)
	got = mod.missing_chunks(torch.from_numpy(xyz).to(device), torch.from_numpy(table).to(device),
							 torch.from_numpy(origin).to(device), torch.from_numpy(spacing).to(device))
	assert {tuple(v) for v in got.cpu().tolist()} == ref_missing(xyz, table, origin, spacing)


@needs_mps
def test_missing_chunk_collector_matches_reference():
	"""Several adds per round, a list overflow that has to be replayed, and a second round on a changed table."""
	import u8_sample_metal
	rng = np.random.default_rng(6)
	table = (rng.random((5, 6, 7)) < 0.4).astype(np.int32)
	xyz = rng.uniform(-200, 700, (4000, 3)).astype(np.float32)
	xyz[:3] = [[np.nan, 5, 5], [5, np.inf, 5], [-0.5, -40, 1e12]]
	origin, spacing = (1.5, -3.0, 2.0), (2.0, 2.0, 2.0)
	o, sp = np.array(origin, np.float32), np.array(spacing, np.float32)
	collector = u8_sample_metal.MissingChunkCollector(table.shape, torch.device("mps"), capacity=8)

	table_t = torch.from_numpy(table).to("mps")
	for part in np.array_split(xyz, 3):
		collector.add(torch.from_numpy(part).to("mps"), table_t, origin, spacing)
	got = collector.take()
	want = ref_missing(xyz, table, o, sp)
	assert len(want) > 8, "the fixture must overflow the initial list"
	assert [tuple(v) for v in got.tolist()] == sorted(want)

	# chunks listed last round and still unloaded must be listed again
	table[tuple(got[: len(got) // 2].T)] = 1
	few = xyz[:500]
	collector.add(torch.from_numpy(few).to("mps"), torch.from_numpy(table).to("mps"), origin, spacing)
	assert [tuple(v) for v in collector.take().tolist()] == sorted(ref_missing(few, table, o, sp))
	assert collector.take().shape == (0, 3)


def test_u8_ties_round_half_away_from_zero(impl):
	name, device, mod = impl
	volume = np.zeros((1, 4, 4, 8), np.uint8)
	volume[0, 1, 1, :] = np.arange(10, 18)
	# x = i + 0.5 at integer y, z: the value is exactly (v_i + v_{i+1}) / 2 = k + 0.5
	xs = np.arange(0, 7, dtype=np.float32) + 0.5
	grid = np.stack([xs, np.ones_like(xs), np.ones_like(xs)], 1).reshape(1, 1, -1, 3)
	g = torch.from_numpy(grid).to(device)
	zero = torch.zeros(3, device=device)
	one = torch.ones(3, device=device)
	got = mod.dense_sample(torch.from_numpy(volume).to(device), g, zero, one, diff=False).cpu().numpy().ravel()
	np.testing.assert_array_equal(got, np.arange(11, 18))


def test_known_negatives_fail_the_bars():
	"""A halo-less sampler and half-to-even uint8 rounding must both fail the parity bars."""
	rng = np.random.default_rng(3)
	volume = _volume(rng)
	table, pool = _sparse_fixture(rng, volume, drop=0.0)
	grid_np = np.moveaxis(_grid(rng, volume.shape[:0:-1], OFFSET, SPACING), 0, -1)
	want, _ = ref_sparse(table, pool, grid_np, OFFSET, INV)
	no_halo, _ = ref_sparse(table, pool, grid_np, OFFSET, INV, halo=0.0)
	assert not fwd_ok(no_halo, want)[0]

	ties = np.arange(256, dtype=np.float64)[None, :255] + 0.5
	half_even = np.round(ties)  # numpy, like torch.round, rounds ties to even
	assert not u8_ok(half_even, ties)[0]
	assert u8_ok(np.floor(ties + 0.5), ties)[0]
