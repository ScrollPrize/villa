"""Bit-exact regression tests for vectorized OME-Zarr scalar and normal pooling."""
from __future__ import annotations

import numpy as np
import pytest

from omezarr_pyramid import (
	_decode_normals,
	_get_normal_luts,
	_mean_pool2x_f32,
	_mean_pool2x_u8,
	_moment_pool2x_normals,
)


def _ref_mean_pool2x_u8(slab: np.ndarray, *, zero_overrides: bool = False) -> np.ndarray:
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


def _ref_mean_pool2x_f32(slab: np.ndarray) -> np.ndarray:
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


def _ref_moment_pool2x_normals(
	nx_u8: np.ndarray,
	ny_u8: np.ndarray,
	*,
	validity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	nx, ny, nz = _decode_normals(nx_u8, ny_u8)
	if validity is not None:
		valid = validity != 0
		nx, ny, nz = (v * valid for v in (nx, ny, nz))
	xx = _ref_mean_pool2x_f32(nx * nx)
	xy = _ref_mean_pool2x_f32(nx * ny)
	xz = _ref_mean_pool2x_f32(nx * nz)
	yy = _ref_mean_pool2x_f32(ny * ny)
	yz = _ref_mean_pool2x_f32(ny * nz)
	zz = _ref_mean_pool2x_f32(nz * nz)
	mat = np.empty(xx.shape + (3, 3), dtype=np.float32)
	mat[..., 0, 0] = xx
	mat[..., 0, 1] = xy
	mat[..., 0, 2] = xz
	mat[..., 1, 0] = xy
	mat[..., 1, 1] = yy
	mat[..., 1, 2] = yz
	mat[..., 2, 0] = xz
	mat[..., 2, 1] = yz
	mat[..., 2, 2] = zz
	vals, vecs = np.linalg.eigh(mat)
	v = np.take_along_axis(vecs, np.argmax(vals, axis=-1)[..., None, None], axis=-1)[..., 0]
	flip = np.where(v[..., 2:3] < 0.0, -1.0, 1.0).astype(np.float32)
	v = v * flip
	norm = np.sqrt(np.maximum(np.sum(v * v, axis=-1, keepdims=True), 1e-12))
	v = v / norm
	nx_out = np.rint(v[..., 0] * 127.0 + 128.0).clip(0.0, 255.0).astype(np.uint8)
	ny_out = np.rint(v[..., 1] * 127.0 + 128.0).clip(0.0, 255.0).astype(np.uint8)
	return nx_out, ny_out


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 9, 16), (1, 1, 1), (3, 5, 7), (10, 11, 12), (0, 4, 4)])
@pytest.mark.parametrize("zero_overrides", [False, True])
def test_mean_pool2x_u8_bit_exact(shape, zero_overrides):
	rng = np.random.default_rng(42)
	data = rng.integers(0, 256, size=shape, dtype=np.uint8)
	if data.size:
		data.ravel()[::13] = 0
	np.testing.assert_array_equal(
		_mean_pool2x_u8(data, zero_overrides=zero_overrides),
		_ref_mean_pool2x_u8(data, zero_overrides=zero_overrides),
	)


def test_mean_pool2x_u8_exhaustive_sums_and_non_u8():
	# Cover every possible 8-corner sum in [0, 2040].
	blocks = np.zeros((2041, 2, 2, 2), dtype=np.uint8)
	for s in range(2041):
		rem = s
		flat = blocks[s].ravel()
		for i in range(8):
			v = min(255, rem)
			flat[i] = v
			rem -= v
	slab = blocks.transpose(0, 1, 2, 3).reshape(2041 * 2, 2, 2)
	np.testing.assert_array_equal(_mean_pool2x_u8(slab), _ref_mean_pool2x_u8(slab))

	# Non-uint8 fallback must also match identically.
	arr_f32 = np.random.default_rng(1).uniform(-5.0, 300.0, size=(6, 7, 8)).astype(np.float32)
	np.testing.assert_array_equal(_mean_pool2x_u8(arr_f32), _ref_mean_pool2x_u8(arr_f32))


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 9, 15), (1, 1, 1), (0, 4, 4)])
def test_mean_pool2x_f32_bit_exact(shape):
	rng = np.random.default_rng(7)
	data = rng.standard_normal(size=shape).astype(np.float32)
	np.testing.assert_array_equal(_mean_pool2x_f32(data), _ref_mean_pool2x_f32(data))


def test_uniform_normal_lut_matches_eigh_all_65536_pairs():
	_, _, uniform_nx, uniform_ny, _, _ = _get_normal_luts()
	vals = np.arange(256, dtype=np.uint8)
	gx, gy = np.meshgrid(vals, vals, indexing="ij")
	# Build a (2, 512, 512) volume where each 2x2x2 block is uniform (gx, gy).
	nx_vol = np.repeat(np.repeat(np.repeat(gx[None, :, :], 2, axis=0), 2, axis=1), 2, axis=2)
	ny_vol = np.repeat(np.repeat(np.repeat(gy[None, :, :], 2, axis=0), 2, axis=1), 2, axis=2)
	ref_nx, ref_ny = _ref_moment_pool2x_normals(nx_vol, ny_vol)
	np.testing.assert_array_equal(uniform_nx, ref_nx[0])
	np.testing.assert_array_equal(uniform_ny, ref_ny[0])


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 9, 15), (10, 12, 14), (2, 2, 2), (1, 3, 5)])
@pytest.mark.parametrize("use_validity", [False, True])
def test_moment_pool2x_normals_bit_exact(shape, use_validity):
	rng = np.random.default_rng(123)
	nx = rng.integers(0, 256, size=shape, dtype=np.uint8)
	ny = rng.integers(0, 256, size=shape, dtype=np.uint8)
	# Inject uniform zero blocks, uniform non-zero blocks, and equatorial normals.
	if shape[0] >= 4 and shape[1] >= 4 and shape[2] >= 4:
		nx[:2, :2, :2] = 0
		ny[:2, :2, :2] = 0
		nx[2:4, :2, :2] = 255
		ny[2:4, :2, :2] = 128
		nx[:2, 2:4, :2] = 128
		ny[:2, 2:4, :2] = 128
	validity = None
	if use_validity:
		validity = rng.integers(0, 2, size=shape, dtype=np.uint8)
		if shape[0] >= 4 and shape[1] >= 4 and shape[2] >= 4:
			validity[:2, :2, :2] = 0
			validity[2:4, 2:4, :2] = 1
	got_nx, got_ny = _moment_pool2x_normals(nx, ny, validity=validity)
	ref_nx, ref_ny = _ref_moment_pool2x_normals(nx, ny, validity=validity)
	np.testing.assert_array_equal(got_nx, ref_nx)
	np.testing.assert_array_equal(got_ny, ref_ny)


def test_moment_pool2x_normals_degenerate_orthogonal_block():
	# Three orthogonal unit normals in the same 2x2x2 block produce an isotropic/degenerate tensor.
	nx = np.full((2, 2, 2), 128, dtype=np.uint8)
	ny = np.full((2, 2, 2), 128, dtype=np.uint8)
	nx[0, 0, 0] = 255
	ny[0, 0, 1] = 255
	nx[0, 1, 0] = 1
	ny[0, 1, 1] = 1
	got_nx, got_ny = _moment_pool2x_normals(nx, ny)
	ref_nx, ref_ny = _ref_moment_pool2x_normals(nx, ny)
	np.testing.assert_array_equal(got_nx, ref_nx)
	np.testing.assert_array_equal(got_ny, ref_ny)


def test_moment_pool2x_normals_errors():
	a = np.zeros((4, 4, 4), dtype=np.uint8)
	with pytest.raises(ValueError, match="mismatch"):
		_moment_pool2x_normals(a, np.zeros((4, 4, 2), dtype=np.uint8))
	with pytest.raises(ValueError, match="validity"):
		_moment_pool2x_normals(a, a, validity=np.zeros((2, 4, 4), dtype=np.uint8))
	with pytest.raises(ValueError, match="3D"):
		_moment_pool2x_normals(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
