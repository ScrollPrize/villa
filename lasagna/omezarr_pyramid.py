from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from vc_reader.reader import ZarrWriter, create_group, open_zarr, write_pyramid_level

from vesuvius.neural_tracing.normals import decode_normals as _decode_normals

_NATIVE_THREAD_ENV = (
	"OPENBLAS_NUM_THREADS",
	"OMP_NUM_THREADS",
	"MKL_NUM_THREADS",
	"BLIS_NUM_THREADS",
	"VECLIB_MAXIMUM_THREADS",
	"NUMEXPR_NUM_THREADS",
	"BLOSC_NTHREADS",
)


def _set_native_thread_env_one() -> None:
	for name in _NATIVE_THREAD_ENV:
		os.environ[name] = "1"


@contextmanager
def _single_threaded_native_runtime():
	"""Temporarily constrain parent and newly spawned native runtimes."""
	missing = object()
	previous = {name: os.environ.get(name, missing) for name in _NATIVE_THREAD_ENV}
	_set_native_thread_env_one()
	try:
		with threadpool_limits(limits=1):
			yield
	finally:
		for name, value in previous.items():
			if value is missing:
				os.environ.pop(name, None)
			else:
				os.environ[name] = value


def shape_div2(shape: tuple[int, int, int], n: int) -> tuple[int, int, int]:
	z, y, x = (int(v) for v in shape)
	for _ in range(max(0, int(n))):
		z = max(1, (z + 1) // 2)
		y = max(1, (y + 1) // 2)
		x = max(1, (x + 1) // 2)
	return z, y, x


def print_progress(*, prefix: str, done: int, total: int, t0: float, suffix: str = "") -> None:
	d = max(0, int(done))
	t = max(1, int(total))
	elapsed = max(1e-6, float(time.time() - t0))
	per = elapsed / float(max(1, d)) if d > 0 else 0.0
	eta = max(0.0, per * float(max(0, t - d))) if d > 0 else 0.0
	eta_m = int(eta // 60.0)
	eta_s = int(eta % 60.0)
	bar_w = 30
	fill = int(round((float(d) / float(t)) * float(bar_w)))
	bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
	print(
		f"\r{prefix} [{bar}] {d}/{t} ({(100.0 * d / float(t)):.1f}%) eta {eta_m:02d}:{eta_s:02d}{suffix}",
		end="",
		flush=True,
	)


def set_pyramid_metadata(group, *, method: str) -> None:
	group.attrs["lasagna_pyramid_downsample"] = method


_U8_SUM8_LUT = np.rint(np.arange(2041, dtype=np.float32) / np.float32(8.0)).clip(0.0, 255.0).astype(np.uint8)

_NORMAL_LUTS: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.uint8, np.uint8] | None = None
_NORMAL_SLAB_Z = 8


def _mean_pool2x_u8_general(slab: np.ndarray, *, zero_overrides: bool = False) -> np.ndarray:
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	acc = np.zeros(out_shape, dtype=np.float32)
	cnt = np.zeros(out_shape, dtype=np.float32)
	zero = np.zeros(out_shape, dtype=bool) if zero_overrides else None
	for dz in (0, 1):
		for dy in (0, 1):
			for dx in (0, 1):
				part = slab[dz::2, dy::2, dx::2]
				if part.size == 0:
					continue
				sz, sy, sx = part.shape
				acc[:sz, :sy, :sx] += part.astype(np.float32, copy=False)
				cnt[:sz, :sy, :sx] += 1.0
				if zero is not None:
					zero[:sz, :sy, :sx] |= part == 0
	out = np.rint(acc / np.maximum(cnt, 1.0)).clip(0.0, 255.0).astype(np.uint8)
	if zero is not None:
		out[zero] = 0
	return out


def _mean_pool2x_u8_even(slab: np.ndarray, *, zero_overrides: bool = False) -> np.ndarray:
	c000 = slab[0::2, 0::2, 0::2]
	c001 = slab[0::2, 0::2, 1::2]
	c010 = slab[0::2, 1::2, 0::2]
	c011 = slab[0::2, 1::2, 1::2]
	c100 = slab[1::2, 0::2, 0::2]
	c101 = slab[1::2, 0::2, 1::2]
	c110 = slab[1::2, 1::2, 0::2]
	c111 = slab[1::2, 1::2, 1::2]
	# Eight uint8 values sum to at most 2040 (< 2^24), so the uint16 sum is exact
	# and _U8_SUM8_LUT reproduces np.rint(float32(sum) / 8.0) bit-for-bit.
	s = c000.astype(np.uint16)
	s += c001
	s += c010
	s += c011
	s += c100
	s += c101
	s += c110
	s += c111
	out = _U8_SUM8_LUT[s]
	if zero_overrides:
		m = c000.copy()
		np.minimum(m, c001, out=m)
		np.minimum(m, c010, out=m)
		np.minimum(m, c011, out=m)
		np.minimum(m, c100, out=m)
		np.minimum(m, c101, out=m)
		np.minimum(m, c110, out=m)
		np.minimum(m, c111, out=m)
		out[m == 0] = 0
	return out


def _mean_pool2x_u8(slab: np.ndarray, *, zero_overrides: bool = False) -> np.ndarray:
	if slab.ndim != 3:
		raise ValueError(f"expected 3D slab, got shape={slab.shape}")
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	if slab.size == 0:
		return np.empty(out_shape, dtype=np.uint8)
	if slab.dtype != np.uint8:
		return _mean_pool2x_u8_general(slab, zero_overrides=zero_overrides)

	ez, ey, ex = z & ~1, y & ~1, x & ~1
	if ez == z and ey == y and ex == x:
		return _mean_pool2x_u8_even(slab, zero_overrides=zero_overrides)

	out = np.empty(out_shape, dtype=np.uint8)
	if ez and ey and ex:
		out[: ez // 2, : ey // 2, : ex // 2] = _mean_pool2x_u8_even(
			slab[:ez, :ey, :ex], zero_overrides=zero_overrides
		)
	if ez < z:
		out[ez // 2 :, :, :] = _mean_pool2x_u8_general(slab[ez:, :, :], zero_overrides=zero_overrides)
	if ey < y and ez:
		out[: ez // 2, ey // 2 :, :] = _mean_pool2x_u8_general(slab[:ez, ey:, :], zero_overrides=zero_overrides)
	if ex < x and ez and ey:
		out[: ez // 2, : ey // 2, ex // 2 :] = _mean_pool2x_u8_general(slab[:ez, :ey, ex:], zero_overrides=zero_overrides)
	return out


def _mean_pool2x_f32_general(slab: np.ndarray) -> np.ndarray:
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	acc = np.zeros(out_shape, dtype=np.float32)
	cnt = np.zeros(out_shape, dtype=np.float32)
	for dz in (0, 1):
		for dy in (0, 1):
			for dx in (0, 1):
				part = slab[dz::2, dy::2, dx::2]
				if part.size == 0:
					continue
				sz, sy, sx = part.shape
				acc[:sz, :sy, :sx] += part.astype(np.float32, copy=False)
				cnt[:sz, :sy, :sx] += 1.0
	return acc / np.maximum(cnt, 1.0)


def _mean_pool2x_f32_even(slab: np.ndarray) -> np.ndarray:
	z, y, x = (int(v) for v in slab.shape)
	src = slab.astype(np.float32, copy=False)
	# Preserve exact (dz, dy, dx) float32 accumulation order while avoiding the cnt array.
	acc = src[0::2, 0::2, 0::2].copy()
	acc += src[0::2, 0::2, 1::2]
	acc += src[0::2, 1::2, 0::2]
	acc += src[0::2, 1::2, 1::2]
	acc += src[1::2, 0::2, 0::2]
	acc += src[1::2, 0::2, 1::2]
	acc += src[1::2, 1::2, 0::2]
	acc += src[1::2, 1::2, 1::2]
	acc *= np.float32(0.125)
	return acc


def _mean_pool2x_f32(slab: np.ndarray) -> np.ndarray:
	if slab.ndim != 3:
		raise ValueError(f"expected 3D slab, got shape={slab.shape}")
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	if slab.size == 0:
		return np.empty(out_shape, dtype=np.float32)
	ez, ey, ex = z & ~1, y & ~1, x & ~1
	if ez == z and ey == y and ex == x:
		return _mean_pool2x_f32_even(slab)
	out = np.empty(out_shape, dtype=np.float32)
	if ez and ey and ex:
		out[: ez // 2, : ey // 2, : ex // 2] = _mean_pool2x_f32_even(slab[:ez, :ey, :ex])
	if ez < z:
		out[ez // 2 :, :, :] = _mean_pool2x_f32_general(slab[ez:, :, :])
	if ey < y and ez:
		out[: ez // 2, ey // 2 :, :] = _mean_pool2x_f32_general(slab[:ez, ey:, :])
	if ex < x and ez and ey:
		out[: ez // 2, : ey // 2, ex // 2 :] = _mean_pool2x_f32_general(slab[:ez, :ey, ex:])
	return out


def _solve_principal_normal_3x3_batch(
	xx: np.ndarray,
	xy: np.ndarray,
	xz: np.ndarray,
	yy: np.ndarray,
	yz: np.ndarray,
	zz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Principal eigenvector of 3x3 symmetric PSD moment matrices, quantized to uint8 (nx, ny).

	Uses Cardano's closed-form eigenvalues in float64 and the largest-norm row cross-product
	of (M - lambda_max * I), falling back to np.linalg.eigh on near-degenerate or equatorial
	(|vz| < 1e-4) matrices so sign tie-breaking matches np.linalg.eigh bit-for-bit.
	"""
	if xx.size == 0:
		empty = np.empty(0, dtype=np.uint8)
		return empty, empty

	a = xx.astype(np.float64)
	d = xy.astype(np.float64)
	e = xz.astype(np.float64)
	b = yy.astype(np.float64)
	f = yz.astype(np.float64)
	c = zz.astype(np.float64)

	p1 = d * d + e * e + f * f
	q = (a + b + c) * (1.0 / 3.0)
	aq = a - q
	bq = b - q
	cq = c - q
	p = np.sqrt((aq * aq + bq * bq + cq * cq + 2.0 * p1) * (1.0 / 6.0))
	inv_p = np.where(p > 0.0, 1.0 / np.where(p > 0.0, p, 1.0), 0.0)
	b00 = aq * inv_p
	b11 = bq * inv_p
	b22 = cq * inv_p
	b01 = d * inv_p
	b02 = e * inv_p
	b12 = f * inv_p

	det_b = (
		b00 * (b11 * b22 - b12 * b12)
		- b01 * (b01 * b22 - b12 * b02)
		+ b02 * (b01 * b12 - b11 * b02)
	)
	phi = np.arccos(np.clip(0.5 * det_b, -1.0, 1.0)) * (1.0 / 3.0)
	cos_phi = np.cos(phi)
	sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
	lam_max = q + 2.0 * p * cos_phi
	lam_mid = q + p * (-cos_phi + 1.7320508075688772 * sin_phi)

	a0 = a - lam_max
	b0 = b - lam_max
	c0 = c - lam_max

	u0x = d * f - e * b0
	u0y = e * d - a0 * f
	u0z = a0 * b0 - d * d
	n0 = u0x * u0x + u0y * u0y + u0z * u0z

	u1x = d * c0 - e * f
	u1y = e * e - a0 * c0
	u1z = a0 * f - d * e
	n1 = u1x * u1x + u1y * u1y + u1z * u1z

	u2x = b0 * c0 - f * f
	u2y = f * e - d * c0
	u2z = d * f - b0 * e
	n2 = u2x * u2x + u2y * u2y + u2z * u2z

	use1 = n1 > n0
	vx = np.where(use1, u1x, u0x)
	vy = np.where(use1, u1y, u0y)
	vz = np.where(use1, u1z, u0z)
	nmax = np.where(use1, n1, n0)

	use2 = n2 > nmax
	vx = np.where(use2, u2x, vx)
	vy = np.where(use2, u2y, vy)
	vz = np.where(use2, u2z, vz)
	nmax = np.where(use2, n2, nmax)

	scale = np.maximum(np.abs(lam_max), 1e-12)
	degen = (p <= 1e-10 * scale) | ((lam_max - lam_mid) <= 1e-4 * scale) | (nmax <= 1e-22 * scale * scale)
	inv_n = 1.0 / np.sqrt(np.where(degen, 1.0, nmax))
	vxf = (vx * inv_n).astype(np.float32)
	vyf = (vy * inv_n).astype(np.float32)
	vzf = (vz * inv_n).astype(np.float32)
	# Equatorial eigenvectors (|vz| ~ 0) sit on the hemisphere flip boundary; delegate to eigh
	# so arbitrary +/-v tie-breaking matches np.linalg.eigh exactly.
	degen |= np.abs(vzf) < np.float32(1e-4)

	n_degen = int(np.count_nonzero(degen))
	if n_degen > 0:
		idx = np.flatnonzero(degen)
		mat = np.empty((n_degen, 3, 3), dtype=np.float32)
		mat[:, 0, 0] = xx[idx]
		mat[:, 0, 1] = xy[idx]
		mat[:, 0, 2] = xz[idx]
		mat[:, 1, 0] = xy[idx]
		mat[:, 1, 1] = yy[idx]
		mat[:, 1, 2] = yz[idx]
		mat[:, 2, 0] = xz[idx]
		mat[:, 2, 1] = yz[idx]
		mat[:, 2, 2] = zz[idx]
		vals, vecs = np.linalg.eigh(mat)
		vd = np.take_along_axis(vecs, np.argmax(vals, axis=-1)[..., None, None], axis=-1)[..., 0]
		vxf[idx] = vd[:, 0]
		vyf[idx] = vd[:, 1]
		vzf[idx] = vd[:, 2]

	flip = np.where(vzf < np.float32(0.0), np.float32(-1.0), np.float32(1.0))
	vxf *= flip
	vyf *= flip
	vzf *= flip
	norm = np.sqrt(np.maximum(vxf * vxf + vyf * vyf + vzf * vzf, np.float32(1e-12)))
	vxf /= norm
	vyf /= norm
	nx_out = np.rint(vxf * np.float32(127.0) + np.float32(128.0)).clip(0.0, 255.0).astype(np.uint8)
	ny_out = np.rint(vyf * np.float32(127.0) + np.float32(128.0)).clip(0.0, 255.0).astype(np.uint8)
	return nx_out, ny_out


def _get_normal_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.uint8, np.uint8]:
	global _NORMAL_LUTS
	if _NORMAL_LUTS is not None:
		return _NORMAL_LUTS
	vals = np.arange(256, dtype=np.uint8)
	gx_u8, gy_u8 = np.meshgrid(vals, vals, indexing="ij")
	gx, gy, nz_lut = _decode_normals(gx_u8, gy_u8)
	comp_lut = gx[:, 0].copy()
	# Exact output for any uniform 2x2x2 block of identical (nx_u8, ny_u8).
	u_nx, u_ny = _solve_principal_normal_3x3_batch(
		(gx * gx).ravel(),
		(gx * gy).ravel(),
		(gx * nz_lut).ravel(),
		(gy * gy).ravel(),
		(gy * nz_lut).ravel(),
		(nz_lut * nz_lut).ravel(),
	)
	uniform_nx = u_nx.reshape(256, 256)
	uniform_ny = u_ny.reshape(256, 256)
	z_nx, z_ny = _solve_principal_normal_3x3_batch(
		np.zeros(1, dtype=np.float32),
		np.zeros(1, dtype=np.float32),
		np.zeros(1, dtype=np.float32),
		np.zeros(1, dtype=np.float32),
		np.zeros(1, dtype=np.float32),
		np.zeros(1, dtype=np.float32),
	)
	_NORMAL_LUTS = (comp_lut, nz_lut, uniform_nx, uniform_ny, z_nx[0], z_ny[0])
	return _NORMAL_LUTS


def _moment_pool2x_normals_general(
	nx_u8: np.ndarray,
	ny_u8: np.ndarray,
	*,
	validity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	nx, ny, nz = _decode_normals(nx_u8, ny_u8)
	if validity is not None:
		valid = validity != 0
		nx, ny, nz = (v * valid for v in (nx, ny, nz))
	xx = _mean_pool2x_f32(nx * nx)
	xy = _mean_pool2x_f32(nx * ny)
	xz = _mean_pool2x_f32(nx * nz)
	yy = _mean_pool2x_f32(ny * ny)
	yz = _mean_pool2x_f32(ny * nz)
	zz = _mean_pool2x_f32(nz * nz)
	onx, ony = _solve_principal_normal_3x3_batch(
		xx.ravel(), xy.ravel(), xz.ravel(), yy.ravel(), yz.ravel(), zz.ravel()
	)
	return onx.reshape(xx.shape), ony.reshape(xx.shape)


def _moment_pool2x_normals(nx_u8: np.ndarray, ny_u8: np.ndarray, *, validity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
	if nx_u8.shape != ny_u8.shape:
		raise ValueError(f"normal channel shape mismatch: nx={nx_u8.shape} ny={ny_u8.shape}")
	if nx_u8.ndim != 3:
		raise ValueError(f"expected 3D slab, got shape={nx_u8.shape}")
	if validity is not None and validity.shape != nx_u8.shape:
		raise ValueError("normal validity shape mismatch")

	z, y, x = (int(v) for v in nx_u8.shape)
	oz, oy, ox = (z + 1) // 2, (y + 1) // 2, (x + 1) // 2
	if nx_u8.size == 0:
		return np.empty((oz, oy, ox), dtype=np.uint8), np.empty((oz, oy, ox), dtype=np.uint8)
	if nx_u8.dtype != np.uint8 or ny_u8.dtype != np.uint8:
		return _moment_pool2x_normals_general(nx_u8, ny_u8, validity=validity)

	comp_lut, nz_lut, uniform_nx, uniform_ny, zero_nx, zero_ny = _get_normal_luts()
	nx_out = np.empty((oz, oy, ox), dtype=np.uint8)
	ny_out = np.empty((oz, oy, ox), dtype=np.uint8)
	inv8 = np.float32(0.125)

	for z0 in range(0, z, _NORMAL_SLAB_Z):
		z1 = min(z, z0 + _NORMAL_SLAB_Z)
		oz0 = z0 // 2
		oz1 = (z1 + 1) // 2
		snx = nx_u8[z0:z1]
		sny = ny_u8[z0:z1]
		sval = None if validity is None else validity[z0:z1]
		sz, sy, sx = snx.shape

		if (sz & 1) or (sy & 1) or (sx & 1):
			# Small Z-slab with odd boundary dimension: exact general pooling + fast 3x3 solver.
			onx, ony = _moment_pool2x_normals_general(snx, sny, validity=sval)
			nx_out[oz0:oz1] = onx
			ny_out[oz0:oz1] = ony
			continue

		sh = (sz // 2, sy // 2, sx // 2)
		r = (sh[0], 2, sh[1], 2, sh[2], 2)
		bx = snx.reshape(r)
		by = sny.reshape(r)
		x000 = bx[:, 0, :, 0, :, 0]
		y000 = by[:, 0, :, 0, :, 0]

		uniform = (
			(bx[:, 0, :, 0, :, 1] == x000) & (by[:, 0, :, 0, :, 1] == y000)
			& (bx[:, 0, :, 1, :, 0] == x000) & (by[:, 0, :, 1, :, 0] == y000)
			& (bx[:, 0, :, 1, :, 1] == x000) & (by[:, 0, :, 1, :, 1] == y000)
			& (bx[:, 1, :, 0, :, 0] == x000) & (by[:, 1, :, 0, :, 0] == y000)
			& (bx[:, 1, :, 0, :, 1] == x000) & (by[:, 1, :, 0, :, 1] == y000)
			& (bx[:, 1, :, 1, :, 0] == x000) & (by[:, 1, :, 1, :, 0] == y000)
			& (bx[:, 1, :, 1, :, 1] == x000) & (by[:, 1, :, 1, :, 1] == y000)
		)

		if sval is None:
			onx = uniform_nx[x000, y000]
			ony = uniform_ny[x000, y000]
			mixed = ~uniform
			n_mix = int(np.count_nonzero(mixed))
			if n_mix > 0:
				if n_mix * 2 > uniform.size:
					nxf = comp_lut[snx]
					nyf = comp_lut[sny]
					nzf = nz_lut[snx, sny]
					xx = np.zeros(sh, dtype=np.float32)
					xy = np.zeros(sh, dtype=np.float32)
					xz = np.zeros(sh, dtype=np.float32)
					yy = np.zeros(sh, dtype=np.float32)
					yz = np.zeros(sh, dtype=np.float32)
					zz = np.zeros(sh, dtype=np.float32)
					for dz in (0, 1):
						for dy in (0, 1):
							for dx in (0, 1):
								px = nxf[dz::2, dy::2, dx::2]
								py = nyf[dz::2, dy::2, dx::2]
								pz = nzf[dz::2, dy::2, dx::2]
								xx += px * px
								xy += px * py
								xz += px * pz
								yy += py * py
								yz += py * pz
								zz += pz * pz
					xx *= inv8
					xy *= inv8
					xz *= inv8
					yy *= inv8
					yz *= inv8
					zz *= inv8
					rnx, rny = _solve_principal_normal_3x3_batch(
						xx.ravel(), xy.ravel(), xz.ravel(), yy.ravel(), yz.ravel(), zz.ravel()
					)
					onx = rnx.reshape(sh)
					ony = rny.reshape(sh)
				else:
					mx = bx.transpose(0, 2, 4, 1, 3, 5)[mixed].reshape(n_mix, 8)
					my = by.transpose(0, 2, 4, 1, 3, 5)[mixed].reshape(n_mix, 8)
					px = comp_lut[mx]
					py = comp_lut[my]
					pz = nz_lut[mx, my]
					xx = np.zeros(n_mix, dtype=np.float32)
					xy = np.zeros(n_mix, dtype=np.float32)
					xz = np.zeros(n_mix, dtype=np.float32)
					yy = np.zeros(n_mix, dtype=np.float32)
					yz = np.zeros(n_mix, dtype=np.float32)
					zz = np.zeros(n_mix, dtype=np.float32)
					for k in range(8):
						pkx = px[:, k]
						pky = py[:, k]
						pkz = pz[:, k]
						xx += pkx * pkx
						xy += pkx * pky
						xz += pkx * pkz
						yy += pky * pky
						yz += pky * pkz
						zz += pkz * pkz
					xx *= inv8
					xy *= inv8
					xz *= inv8
					yy *= inv8
					yz *= inv8
					zz *= inv8
					rnx, rny = _solve_principal_normal_3x3_batch(xx, xy, xz, yy, yz, zz)
					onx[mixed] = rnx
					ony[mixed] = rny
		else:
			vbool = (sval != 0).reshape(r)
			v000 = vbool[:, 0, :, 0, :, 0]
			v001 = vbool[:, 0, :, 0, :, 1]
			v010 = vbool[:, 0, :, 1, :, 0]
			v011 = vbool[:, 0, :, 1, :, 1]
			v100 = vbool[:, 1, :, 0, :, 0]
			v101 = vbool[:, 1, :, 0, :, 1]
			v110 = vbool[:, 1, :, 1, :, 0]
			v111 = vbool[:, 1, :, 1, :, 1]
			all_invalid = ~(v000 | v001 | v010 | v011 | v100 | v101 | v110 | v111)
			all_valid = v000 & v001 & v010 & v011 & v100 & v101 & v110 & v111
			shortcut_uniform = all_valid & uniform
			onx = np.full(sh, zero_nx, dtype=np.uint8)
			ony = np.full(sh, zero_ny, dtype=np.uint8)
			if np.any(shortcut_uniform):
				onx[shortcut_uniform] = uniform_nx[x000[shortcut_uniform], y000[shortcut_uniform]]
				ony[shortcut_uniform] = uniform_ny[x000[shortcut_uniform], y000[shortcut_uniform]]
			mixed = ~(all_invalid | shortcut_uniform)
			n_mix = int(np.count_nonzero(mixed))
			if n_mix > 0:
				mx = bx.transpose(0, 2, 4, 1, 3, 5)[mixed].reshape(n_mix, 8)
				my = by.transpose(0, 2, 4, 1, 3, 5)[mixed].reshape(n_mix, 8)
				mv = vbool.transpose(0, 2, 4, 1, 3, 5)[mixed].reshape(n_mix, 8)
				px = comp_lut[mx] * mv
				py = comp_lut[my] * mv
				pz = nz_lut[mx, my] * mv
				xx = np.zeros(n_mix, dtype=np.float32)
				xy = np.zeros(n_mix, dtype=np.float32)
				xz = np.zeros(n_mix, dtype=np.float32)
				yy = np.zeros(n_mix, dtype=np.float32)
				yz = np.zeros(n_mix, dtype=np.float32)
				zz = np.zeros(n_mix, dtype=np.float32)
				for k in range(8):
					pkx = px[:, k]
					pky = py[:, k]
					pkz = pz[:, k]
					xx += pkx * pkx
					xy += pkx * pky
					xz += pkx * pkz
					yy += pky * pky
					yz += pky * pkz
					zz += pkz * pkz
				xx *= inv8
				xy *= inv8
				xz *= inv8
				yy *= inv8
				yz *= inv8
				zz *= inv8
				rnx, rny = _solve_principal_normal_3x3_batch(xx, xy, xz, yy, yz, zz)
				onx[mixed] = rnx
				ony[mixed] = rny

		nx_out[oz0:oz1] = onx
		ny_out[oz0:oz1] = ony
	return nx_out, ny_out


def clear_level_chunks(level_path: str | Path) -> None:
	"""Remove a level's chunks, keeping its metadata and attributes."""
	level_path = Path(level_path)
	if not level_path.is_dir():
		return
	level = ZarrWriter.open(level_path)
	attrs = dict(level.attrs)
	cleared = ZarrWriter.create_like(level_path, level, overwrite=True)
	if attrs:
		cleared.attrs.update(attrs)


def clear_coarser_levels(omezarr_path: str | Path, data_level: int, n_levels: int) -> None:
	for lv in range(int(data_level) + 1, int(n_levels)):
		clear_level_chunks(Path(omezarr_path) / str(lv))


def _scaled_crop_for_source_level(
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	levels_above_data: int,
) -> tuple[int, int, int, int, int, int] | None:
	if crop_zyx is None:
		return None
	z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_zyx)
	scale = 2 ** max(0, int(levels_above_data))
	return (
		z0 // scale,
		y0 // scale,
		x0 // scale,
		(z1 + scale - 1) // scale,
		(y1 + scale - 1) // scale,
		(x1 + scale - 1) // scale,
	)


def _crop_chunk_range(
	dst_shape: tuple[int, int, int],
	dst_chunks: tuple[int, int, int],
	crop_zyx: tuple[int, int, int, int, int, int] | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	"""The output chunks ``[begin, end)`` under a crop given in source-level voxels."""
	grid = tuple((n + c - 1) // c for n, c in zip(dst_shape, dst_chunks))
	if crop_zyx is None:
		return (0, 0, 0), grid
	begin = tuple(max(0, int(a)) // (2 * c) for a, c in zip(crop_zyx[:3], dst_chunks))
	end = tuple(min(n, (int(b) + 2 * c - 1) // (2 * c)) for b, c, n in zip(crop_zyx[3:], dst_chunks, grid))
	return begin, end


def _stored_output_chunks(source_path: str | Path, dst_chunks) -> set[tuple[int, int, int]]:
	"""The output chunks over any chunk stored in the source level."""
	src_chunks = open_zarr(source_path).chunks
	return {tuple(i * s // 2 // d for i, s, d in zip(index, src_chunks, dst_chunks)) for index in ZarrWriter.open(source_path).stored_chunks()}


def _pool_chunks(
	source_paths,
	dst_shape,
	dst_chunks,
	crop_zyx,
	require_source_chunks: bool,
) -> list[tuple[int, int, int]]:
	"""Output chunks to consider, in chunk order. With ``require_source_chunks``, every source must store a chunk under each."""
	begin, end = _crop_chunk_range(dst_shape, dst_chunks, crop_zyx)
	if require_source_chunks:
		stored = None
		for path in source_paths:
			found = _stored_output_chunks(path, dst_chunks)
			stored = found if stored is None else stored & found
		return sorted(c for c in stored if all(b <= i < e for b, i, e in zip(begin, c, end)))
	return [
		(z, y, x)
		for z in range(begin[0], end[0])
		for y in range(begin[1], end[1])
		for x in range(begin[2], end[2])
	]


def _pyramid_status_suffix(counts: dict[str, int]) -> str:
	parts = []
	for key, label in (
		("written", "write"),
		("skipped_existing", "skip_existing"),
		("skipped_empty_source", "skip_empty"),
		("skipped_empty_slice", "skip_empty_slice"),
	):
		value = int(counts.get(key, 0))
		if value:
			parts.append(f"{label}={value}")
	return " " + " ".join(parts) if parts else ""


_PROCESS_WORKER = None


def _init_process_worker(worker) -> None:
	global _PROCESS_WORKER
	_PROCESS_WORKER = worker


def _call_process_worker(item):
	return _PROCESS_WORKER(item)


def _run_pool(work, worker, *, workers: int, tag: str, total: int | None = None, processes: bool = False) -> None:
	"""``worker`` over ``work``, on threads, or with ``processes`` on spawned processes for work that keeps the GIL
	(``worker`` then is picklable).
	"""
	n_work = int(total) if total is not None else len(work)
	if n_work == 0:
		return
	t0 = time.time()
	done_count = [0]
	status_counts: dict[str, int] = {}
	lock = threading.Lock()
	stop = threading.Event()

	def _prog() -> None:
		while not stop.is_set():
			with lock:
				d = done_count[0]
				suffix = _pyramid_status_suffix(status_counts)
			print_progress(prefix=tag, done=d, total=n_work, t0=t0, suffix=suffix)
			stop.wait(0.5)

	prog_thread = threading.Thread(
		target=_prog, daemon=True, name=f"pyramid-progress:{tag}"
	)
	prog_thread.start()
	complete = False
	pool_size = min(max(1, int(workers)), n_work)
	print(f"\n{tag} workers={pool_size}", flush=True)
	try:
		with _single_threaded_native_runtime():
			if processes and pool_size > 1:
				pool = ProcessPoolExecutor(
					max_workers=pool_size, mp_context=multiprocessing.get_context("spawn"),
					initializer=_init_process_worker, initargs=(worker,),
				)
				statuses = lambda: pool.map(_call_process_worker, work)
			else:
				pool = ThreadPoolExecutor(max_workers=pool_size)
				statuses = lambda: pool.map(worker, work)
			with pool:
				for status in statuses():
					with lock:
						done_count[0] += 1
						if status:
							status_counts[str(status)] = status_counts.get(str(status), 0) + 1
		complete = True
	finally:
		stop.set()
		prog_thread.join(timeout=2)
		if complete:
			print_progress(prefix=tag, done=n_work, total=n_work, t0=t0, suffix=_pyramid_status_suffix(status_counts))
		print("", flush=True)


def invalidate_coarser_chunks(level_writers: dict[int, ZarrWriter], level: int, index: tuple[int, int, int]) -> None:
	"""Coarser chunks made from this chunk are out of date once it changes: an interrupted build recomputes them."""
	for lv in sorted(lv for lv in level_writers if lv > level):
		index = tuple(i // 2 for i in index)
		level_writers[lv].remove_chunk(index)


class _PoolChunk:
	"""Writes one output chunk of each root's level from the level below. Picklable."""

	def __init__(self, roots, src_level: int, dst_level: int, n_levels: int, reduce, skip_existing: bool):
		self.sources = [open_zarr(Path(root) / str(src_level), threads=1) for root in roots]
		self.targets = [ZarrWriter.open(Path(root) / str(dst_level), skip_empty=True) for root in roots]
		self.coarser = [
			{lv: ZarrWriter.open(Path(root) / str(lv)) for lv in range(dst_level + 1, n_levels) if (Path(root) / str(lv)).is_dir()}
			for root in roots
		]
		self.dst_level = dst_level
		self.reduce = reduce
		self.skip_existing = skip_existing

	def __call__(self, index) -> str:
		if self.skip_existing and all(target.chunk_exists(index) for target in self.targets):
			return "skipped_existing"
		chunks, shape = self.targets[0].chunks, self.targets[0].shape
		start = [i * c * 2 for i, c in zip(index, chunks)]
		size = [min(c * 2, n - a) for c, n, a in zip(chunks, self.sources[0].shape, start)]
		if min(size) <= 0:
			return "skipped_empty_slice"
		blocks = self.reduce(*(source.read_region(start, size) for source in self.sources))
		origin = [i * c for i, c in zip(index, chunks)]
		for target, levels, block in zip(self.targets, self.coarser, blocks):
			invalidate_coarser_chunks(levels, self.dst_level, index)
			region = tuple(slice(a, min(n, a + b)) for a, n, b in zip(origin, shape, block.shape))
			target[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
		return "written"


def _pool_level(
	roots,
	src_level: int,
	dst_level: int,
	n_levels: int,
	reduce,
	*,
	crop_zyx,
	skip_existing: bool,
	require_source_chunks: bool,
	workers: int,
	tag: str,
	processes: bool = False,
) -> None:
	"""Write ``dst_level`` of each root from the level below, chunk by chunk, with an application reduction:
	``reduce(*source_blocks)`` returns one block per root.
	"""
	worker = _PoolChunk(roots, src_level, dst_level, n_levels, reduce, skip_existing)
	work = _pool_chunks(
		[Path(root) / str(src_level) for root in roots], worker.targets[0].shape, worker.targets[0].chunks,
		crop_zyx, require_source_chunks,
	)
	_run_pool(work, worker, workers=workers, tag=tag, processes=processes)


def _reduce_zero_overrides(slab: np.ndarray) -> tuple[np.ndarray]:
	return (_mean_pool2x_u8(np.asarray(slab, dtype=np.uint8), zero_overrides=True),)


def _reduce_normals(nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return _moment_pool2x_normals(np.asarray(nx, dtype=np.uint8), np.asarray(ny, dtype=np.uint8))


def build_scalar_omezarr_pyramid(
	omezarr_path: str | Path,
	data_level: int,
	n_levels: int,
	chunk: int | tuple[int, int, int] | None = None,
	*,
	workers: int = 0,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	label: str = "",
	force: bool = False,
	zero_overrides: bool = False,
	scan_existing_source_chunks: bool = False,
) -> None:
	"""Coarser levels as 2x means. Plain means are rewritten over the crop (or the whole grid with
	``scan_existing_source_chunks``); ``zero_overrides`` keeps existing chunks unless ``force``.
	``chunk`` is unused: each level's own chunks decide.
	"""
	root = Path(omezarr_path)
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	if force:
		clear_coarser_levels(root, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		src_crop = None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		tag = f"[pyramid {label} L{lv}]" if label else f"[pyramid L{lv}]"
		if zero_overrides:
			_pool_level(
				(root,), src_lv, lv, int(n_levels),
				_reduce_zero_overrides,
				crop_zyx=src_crop,
				skip_existing=not force,
				require_source_chunks=scan_existing_source_chunks,
				workers=workers,
				tag=tag,
			)
		else:
			print(f"{tag} workers={workers}", flush=True)
			level = open_zarr(root / str(lv))
			write_pyramid_level(root, lv, threads=workers, chunk_range=_crop_chunk_range(level.shape, level.chunks, src_crop))
	set_pyramid_metadata(create_group(root), method="mean_pool2x_zero_overrides" if zero_overrides else "mean_pool2x")


def build_normal_omezarr_pyramid(
	nx_omezarr_path: str | Path,
	ny_omezarr_path: str | Path,
	data_level: int,
	n_levels: int,
	chunk: int | tuple[int, int, int] | None = None,
	*,
	workers: int = 0,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	label: str = "normal",
	force: bool = False,
	scan_existing_source_chunks: bool = False,
) -> None:
	"""Coarser levels of a normal pair by the second-moment reduction. Existing chunks stay unless ``force``.
	``chunk`` is unused: each level's own chunks decide.
	"""
	roots = (Path(nx_omezarr_path), Path(ny_omezarr_path))
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	if force:
		for root in roots:
			clear_coarser_levels(root, data_level, n_levels)

	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		_pool_level(
			roots, src_lv, lv, int(n_levels), _reduce_normals,
			crop_zyx=None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level)),
			skip_existing=not force,
			require_source_chunks=scan_existing_source_chunks,
			workers=workers,
			tag=f"[pyramid {label} L{lv}]",
			processes=True,  # numpy's eigh keeps the GIL
		)
	for root in roots:
		set_pyramid_metadata(create_group(root), method="normal_second_moment_mean_pool2x")
