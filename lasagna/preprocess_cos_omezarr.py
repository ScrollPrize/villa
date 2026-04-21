from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
del _os

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
try:
	import cupy as cp
	from cupyx.scipy import ndimage as cnd
	_HAS_CUPY = True
except ImportError:
	_HAS_CUPY = False
try:
	import edt as edt_mod
	_HAS_EDT = True
except ImportError:
	edt_mod = None
	_HAS_EDT = False
import numpy as np
try:
	import numba
	_HAS_NUMBA = True
except ImportError:
	numba = None
	_HAS_NUMBA = False
import torch
import torch.nn.functional as F
import zarr

from common import load_unet, unet_infer_tiled
from train_unet_3d import build_model as build_model_3d


def _crop_xyzwhd_bounds(*, shape_zyx: tuple[int, int, int], crop_xyzwhd: tuple[int, int, int, int, int, int] | None) -> tuple[int, int, int, int, int, int]:
	zs, ys, xs = (int(v) for v in shape_zyx)
	if crop_xyzwhd is None:
		return 0, zs, 0, ys, 0, xs
	x, y, z, w, h, d = (int(v) for v in crop_xyzwhd)
	x0 = max(0, min(x, xs))
	y0 = max(0, min(y, ys))
	z0 = max(0, min(z, zs))
	x1 = max(x0, min(x + max(0, w), xs))
	y1 = max(y0, min(y + max(0, h), ys))
	z1 = max(z0, min(z + max(0, d), zs))
	return z0, z1, y0, y1, x0, x1


def _ds_size(v: int, f: int) -> int:
	# Match interpolate(scale_factor=1/f) floor behavior.
	return max(1, int(v) // int(f))


def _ds_index(v: int, f: int) -> int:
	return max(0, int(v) // int(f))


def _pyrdown2d(arr: np.ndarray, *, factor: int) -> np.ndarray:
	"""Gaussian pyramid downscale using repeated cv2.pyrDown for power-of-2 factors."""
	f = int(factor)
	if f <= 1:
		return arr
	if (f & (f - 1)) != 0:
		raise ValueError("downscale factor must be a power of 2 for pyramid scaling")
	out = arr.astype(np.float32, copy=False)
	while f > 1:
		out = cv2.pyrDown(out)
		f //= 2
	return out


def _pyrdown3d(t: torch.Tensor, *, factor: int) -> torch.Tensor:
	"""Gaussian pyramid downscale for 3D volume tensors.
	Uses the same [1,4,6,4,1]/16 kernel as cv2.pyrDown, applied separably."""
	f = int(factor)
	if f <= 1:
		return t
	if (f & (f - 1)) != 0:
		raise ValueError("downscale factor must be a power of 2 for pyramid scaling")
	k = torch.tensor([1, 4, 6, 4, 1], dtype=t.dtype, device=t.device) / 16.0
	while f > 1:
		C = t.shape[0]
		for dim, pad_arg in enumerate([(0,0,0,0,2,2), (0,0,2,2,0,0), (2,2,0,0,0,0)]):
			shape = [1, 1, 1, 1, 1]
			shape[dim + 2] = 5
			kd = k.view(*shape).expand(C, 1, *shape[2:])
			t = F.conv3d(F.pad(t.unsqueeze(0), pad_arg, mode='reflect'), kd, groups=C)[0]
		t = t[:, ::2, ::2, ::2]
		f //= 2
	return t


def _decode_dir_angle(dir0: np.ndarray, dir1: np.ndarray) -> np.ndarray:
	"""Decode dir0+dir1 (in [0,1]) to angle θ ∈ (-π/2, π/2]."""
	cos2t = 2.0 * dir0 - 1.0
	sin2t = cos2t - np.sqrt(2.0) * (2.0 * dir1 - 1.0)
	return np.arctan2(sin2t, cos2t) * 0.5


def _estimate_normal(
	dir0_z: np.ndarray, dir1_z: np.ndarray,
	dir0_y: np.ndarray, dir1_y: np.ndarray,
	dir0_x: np.ndarray, dir1_x: np.ndarray,
	eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Estimate 3D surface normal and fusion weights from three axis dir channel pairs.

	Uses iterative observation-weighted fitting:
	  Pass 1: Score 3 cross-product candidates against observed dir channels,
	          weighted average → initial estimate.
	  Pass 2: Re-weight constraint rows by axis reliability from the estimate,
	          sign-align, sum, normalize → final normal.

	Returns (w_z, w_y, w_x, nx_n, ny_n, nz_n) — fusion weights and unit normal.
	"""
	theta_z = _decode_dir_angle(dir0_z, dir1_z)
	theta_y = _decode_dir_angle(dir0_y, dir1_y)
	theta_x = _decode_dir_angle(dir0_x, dir1_x)

	sz, cz = np.sin(theta_z), np.cos(theta_z)
	sy, cy = np.sin(theta_y), np.cos(theta_y)
	sx, cx = np.sin(theta_x), np.cos(theta_x)

	# Cross products (candidate normals):
	n1_x = cz * cy
	n1_y = sz * cy
	n1_z = cz * sy

	n2_x = cz * cx
	n2_y = sz * cx
	n2_z = sz * sx

	n3_x = cy * sx
	n3_y = sy * cx
	n3_z = sy * sx

	# Align signs: flip n2, n3 so dot(ni, n1) >= 0
	dot2 = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z
	sign2 = np.where(dot2 >= 0, 1.0, -1.0)
	n2_x = n2_x * sign2
	n2_y = n2_y * sign2
	n2_z = n2_z * sign2

	dot3 = n1_x * n3_x + n1_y * n3_y + n1_z * n3_z
	sign3 = np.where(dot3 >= 0, 1.0, -1.0)
	n3_x = n3_x * sign3
	n3_y = n3_y * sign3
	n3_z = n3_z * sign3

	# --- Pass 1: Score candidates against observed dir channels ---
	def _enc(gx, gy):
		r2 = gx * gx + gy * gy + eps
		c2 = (gx * gx - gy * gy) / r2
		s2 = 2.0 * gx * gy / r2
		isq2 = 1.0 / np.sqrt(2.0)
		return 0.5 + 0.5 * c2, 0.5 + 0.5 * (c2 - s2) * isq2

	scores = []
	for (ncx, ncy, ncz) in [(n1_x, n1_y, n1_z), (n2_x, n2_y, n2_z), (n3_x, n3_y, n3_z)]:
		pz0, pz1 = _enc(ncx, ncy)
		py0, py1 = _enc(ncx, ncz)
		px0, px1 = _enc(ncy, ncz)
		err_z = (pz0 - dir0_z) ** 2 + (pz1 - dir1_z) ** 2
		err_y = (py0 - dir0_y) ** 2 + (py1 - dir1_y) ** 2
		err_x = (px0 - dir0_x) ** 2 + (px1 - dir1_x) ** 2
		wz_c = ncx ** 2 + ncy ** 2
		wy_c = ncx ** 2 + ncz ** 2
		wx_c = ncy ** 2 + ncz ** 2
		total_err = wz_c * err_z + wy_c * err_y + wx_c * err_x
		scores.append(1.0 / (total_err + eps))

	s1, s2_s, s3_s = scores
	est_x = s1 * n1_x + s2_s * n2_x + s3_s * n3_x
	est_y = s1 * n1_y + s2_s * n2_y + s3_s * n3_y
	est_z = s1 * n1_z + s2_s * n2_z + s3_s * n3_z
	norm_e = np.sqrt(est_x ** 2 + est_y ** 2 + est_z ** 2) + eps
	est_x = est_x / norm_e
	est_y = est_y / norm_e
	est_z = est_z / norm_e

	# --- Pass 2: Re-weight constraint rows ---
	wz2 = np.sqrt(est_x ** 2 + est_y ** 2 + eps)
	wy2 = np.sqrt(est_x ** 2 + est_z ** 2 + eps)
	wx2 = np.sqrt(est_y ** 2 + est_z ** 2 + eps)

	wzy = wz2 * wy2
	wzx = wz2 * wx2
	wyx = wy2 * wx2

	rn1_x = wzy * n1_x; rn1_y = wzy * n1_y; rn1_z = wzy * n1_z
	rn2_x = wzx * n2_x; rn2_y = wzx * n2_y; rn2_z = wzx * n2_z
	rn3_x = wyx * n3_x; rn3_y = wyx * n3_y; rn3_z = wyx * n3_z

	dot2r = rn1_x * rn2_x + rn1_y * rn2_y + rn1_z * rn2_z
	s2r = np.where(dot2r >= 0, 1.0, -1.0)
	rn2_x = rn2_x * s2r; rn2_y = rn2_y * s2r; rn2_z = rn2_z * s2r

	dot3r = rn1_x * rn3_x + rn1_y * rn3_y + rn1_z * rn3_z
	s3r = np.where(dot3r >= 0, 1.0, -1.0)
	rn3_x = rn3_x * s3r; rn3_y = rn3_y * s3r; rn3_z = rn3_z * s3r

	nx_f = rn1_x + rn2_x + rn3_x
	ny_f = rn1_y + rn2_y + rn3_y
	nz_f = rn1_z + rn2_z + rn3_z
	norm_f = np.sqrt(nx_f ** 2 + ny_f ** 2 + nz_f ** 2) + eps
	nx_n = nx_f / norm_f
	ny_n = ny_f / norm_f
	nz_n = nz_f / norm_f

	w_z = np.sqrt(nx_n * nx_n + ny_n * ny_n + eps)
	w_y = np.sqrt(nx_n * nx_n + nz_n * nz_n + eps)
	w_x = np.sqrt(ny_n * ny_n + nz_n * nz_n + eps)

	return w_z, w_y, w_x, nx_n, ny_n, nz_n


def run_preprocess(
	*,
	input_path: str,
	output_path: str,
	unet_checkpoint: str,
	device: str | None,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	axis: str = "z",
	tile_size: int,
	overlap: int,
	border: int,
	scaledown: int,
	chunk_z: int,
	chunk_yx: int,
	measure_cuda_timings: bool = False,
) -> None:
	a_in = zarr.open(str(input_path), mode="r")
	if not hasattr(a_in, "shape"):
		raise ValueError(f"input must point to an OME-Zarr array, got non-array: {input_path}")
	sh = tuple(int(v) for v in a_in.shape)
	if len(sh) != 3:
		raise ValueError(f"input array must be (Z,Y,X), got {sh}")

	z0, z1, y0, y1, x0, x1 = _crop_xyzwhd_bounds(shape_zyx=sh, crop_xyzwhd=crop_xyzwhd)
	nz = z1 - z0
	ny = y1 - y0
	nx = x1 - x0
	if nz <= 0 or ny <= 0 or nx <= 0:
		raise ValueError(f"empty crop: x=[{x0},{x1}) y=[{y0},{y1}) z=[{z0},{z1}) in shape={sh}")

	if scaledown <= 0:
		raise ValueError("scaledown must be >= 1")

	# --- Axis-dependent dimension mapping ---
	# Everything in ZYX order (matching zarr layout, indices 0=Z, 1=Y, 2=X)
	dim_names = ["z", "y", "x"]
	axis_to_dim = {"z": 0, "y": 1, "x": 2}
	if axis not in axis_to_dim:
		raise ValueError(f"axis must be 'z', 'y', or 'x', got '{axis}'")
	slice_dim = axis_to_dim[axis]
	plane_dims = [d for d in range(3) if d != slice_dim]
	plane_dim0, plane_dim1 = plane_dims

	crop_ranges = [(z0, z1), (y0, y1), (x0, x1)]
	full_sizes = [int(sh[0]), int(sh[1]), int(sh[2])]

	slice_start, slice_end = crop_ranges[slice_dim]

	slice_sel = list(range(int(slice_start), int(slice_end), int(scaledown)))
	if len(slice_sel) <= 0:
		raise ValueError(
			f"empty {dim_names[slice_dim]} selection after downscale: "
			f"{dim_names[slice_dim]}=[{slice_start},{slice_end}) scaledown={scaledown}"
		)
	proc_count = len(slice_sel)

	# Output sizes in ZYX order — uniform scaledown in all dims
	out_sizes = [0, 0, 0]
	out_sizes[slice_dim] = _ds_size(full_sizes[slice_dim], scaledown)
	out_sizes[plane_dim0] = _ds_size(full_sizes[plane_dim0], scaledown)
	out_sizes[plane_dim1] = _ds_size(full_sizes[plane_dim1], scaledown)
	out_z, out_y, out_x = out_sizes

	# Output offsets in ZYX order
	out_offsets = [0, 0, 0]
	out_offsets[slice_dim] = _ds_index(crop_ranges[slice_dim][0], scaledown)
	out_offsets[plane_dim0] = _ds_index(crop_ranges[plane_dim0][0], scaledown)
	out_offsets[plane_dim1] = _ds_index(crop_ranges[plane_dim1][0], scaledown)

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	torch_device = torch.device(device)

	model = load_unet(
		device=torch_device,
		weights=str(unet_checkpoint),
		strict=True,
		in_channels=1,
		out_channels=4,
		base_channels=32,
		num_levels=6,
		max_channels=1024,
	)
	model.eval()

	# Output chunk sizes in ZYX order
	chunk_sizes = [0, 0, 0]
	chunk_sizes[slice_dim] = max(1, int(chunk_z))
	chunk_sizes[plane_dim0] = min(out_sizes[plane_dim0], max(1, int(chunk_yx)))
	chunk_sizes[plane_dim1] = min(out_sizes[plane_dim1], max(1, int(chunk_yx)))

	arr = zarr.open(
		str(output_path),
		mode="w",
		shape=(5, int(out_z), int(out_y), int(out_x)),
		chunks=(1, chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]),
		dtype=np.uint8,
		fill_value=0,
		zarr_format=2,
	)
	arr.attrs["preprocess_params"] = {
		"axis": axis,
		"scaledown": int(scaledown),
		"grad_mag_encode_scale": float(1000.0),
		"processed_z_slices": int(proc_count),
		"crop_xyzwhd": [int(x0), int(y0), int(z0), int(nx), int(ny), int(nz)],
		"output_full_scaled": True,
		"channels": ["cos", "grad_mag", "dir0", "dir1", "valid"],
	}

	ax_name = dim_names[slice_dim]
	print(
		f"[preprocess_cos_omezarr] input={input_path} axis={axis} crop_xyzwhd=({x0},{y0},{z0},{nx},{ny},{nz}) "
		f"scaledown={scaledown} proc_slices={proc_count} out_shape_full={(out_z, out_y, out_x)} in_shape={sh} "
		f"-> out={output_path} out_shape={(5, out_z, out_y, out_x)} dtype=uint8"
	)
	t0 = time.time()
	t_read_sum = 0.0
	t_infer_sum = 0.0
	t_write_sum = 0.0

	# Read chunk size along slice axis from input zarr
	raw_chunks = getattr(a_in, "chunks", None)
	if isinstance(raw_chunks, tuple) and len(raw_chunks) > slice_dim:
		read_chunk = max(1, int(raw_chunks[slice_dim]))
	else:
		read_chunk = max(1, int(chunk_z))

	read0 = (int(slice_start) // int(read_chunk)) * int(read_chunk)
	done = 0
	for sr0 in range(read0, int(slice_end), int(read_chunk)):
		sr1 = min(int(slice_end), sr0 + int(read_chunk))
		if sr1 <= int(slice_start):
			continue
		slo = max(int(slice_start), sr0)
		shi = sr1
		s_keep = [ss for ss in range(slo, shi) if ((ss - int(slice_start)) % int(scaledown)) == 0]
		if len(s_keep) <= 0:
			continue
		idx_keep = np.asarray([ss - sr0 for ss in s_keep], dtype=np.int64)

		# Build zarr read slices: full crop for plane dims, chunk range for slice dim
		read_ranges = list(crop_ranges)
		read_ranges[slice_dim] = (sr0, sr1)
		zarr_sel = tuple(slice(s, e) for s, e in read_ranges)

		t_read0 = time.time()
		raw_chunk_np = np.asarray(a_in[zarr_sel])

		# Move slice dimension to axis 0 for uniform processing
		if slice_dim != 0:
			raw_chunk_np = np.moveaxis(raw_chunk_np, slice_dim, 0)

		raw_blk_np = raw_chunk_np[idx_keep, :, :]
		if raw_blk_np.dtype == np.uint16:
			raw_blk_np = (raw_blk_np // 257).astype(np.uint8)
		raw_blk = torch.from_numpy(raw_blk_np.astype(np.float32)).to(device=torch_device)
		if raw_blk.numel() > 0:
			mx = raw_blk.amax(dim=(1, 2), keepdim=True)
			raw_blk = torch.where(mx > 0.0, raw_blk / mx, raw_blk)
		raw_blk = raw_blk[:, None, :, :]
		t_read_sum += float(time.time() - t_read0)

		for bi, ss in enumerate(s_keep):
			raw_i = raw_blk[bi : bi + 1]
			if measure_cuda_timings and torch_device.type == "cuda":
				torch.cuda.synchronize(torch_device)
			t_inf0 = time.time()
			with torch.inference_mode(), torch.autocast(device_type=torch_device.type):
				pred_i = unet_infer_tiled(
					model,
					raw_i,
					tile_size=int(tile_size),
					overlap=int(overlap),
					border=int(border),
				)
			if measure_cuda_timings and torch_device.type == "cuda":
				torch.cuda.synchronize(torch_device)
			t_infer_sum += float(time.time() - t_inf0)

			cos = pred_i[:, 0:1]
			grad_mag = pred_i[:, 1:2] if int(pred_i.shape[1]) > 1 else pred_i[:, 0:1]
			dir0 = pred_i[:, 2:3] if int(pred_i.shape[1]) > 2 else pred_i[:, 0:1]
			dir1 = pred_i[:, 3:4] if int(pred_i.shape[1]) > 3 else pred_i[:, 0:1]
			cos_np = cos[0, 0].detach().cpu().numpy().astype(np.float32)
			grad_mag_np = grad_mag[0, 0].detach().cpu().numpy().astype(np.float32)
			dir0_np = dir0[0, 0].detach().cpu().numpy().astype(np.float32)
			dir1_np = dir1[0, 0].detach().cpu().numpy().astype(np.float32)
			if scaledown > 1:
				cos_np = _pyrdown2d(cos_np, factor=int(scaledown))
				grad_mag_np = _pyrdown2d(grad_mag_np, factor=int(scaledown))
				dir0_np = _pyrdown2d(dir0_np, factor=int(scaledown))
				dir1_np = _pyrdown2d(dir1_np, factor=int(scaledown))

			cos_u8 = np.clip(cos_np * 255.0, 0.0, 255.0).astype(np.uint8)
			grad_mag_u8 = np.clip(grad_mag_np * 1000.0, 0.0, 255.0).astype(np.uint8)
			dir0_u8 = np.clip(dir0_np * 255.0, 0.0, 255.0).astype(np.uint8)
			dir1_u8 = np.clip(dir1_np * 255.0, 0.0, 255.0).astype(np.uint8)
			t_wr0 = time.time()

			# Output index along slice dimension
			oi = int(ss) // int(scaledown)
			# Output ranges for plane dimensions (shape[0]=plane_dim0, shape[1]=plane_dim1)
			p0_start = out_offsets[plane_dim0]
			p1_start = out_offsets[plane_dim1]
			p0_end = min(out_sizes[plane_dim0], p0_start + int(cos_u8.shape[0]))
			p1_end = min(out_sizes[plane_dim1], p1_start + int(cos_u8.shape[1]))

			if oi >= 0 and oi < out_sizes[slice_dim] and p0_end > p0_start and p1_end > p1_start:
				p0_h = p0_end - p0_start
				p1_w = p1_end - p1_start
				# Build write index [z_idx, y_idx, x_idx]
				write_idx: list = [None, None, None]
				write_idx[slice_dim] = oi
				write_idx[plane_dim0] = slice(p0_start, p0_end)
				write_idx[plane_dim1] = slice(p1_start, p1_end)
				widx = tuple(write_idx)
				arr[(0,) + widx] = cos_u8[:p0_h, :p1_w]
				arr[(1,) + widx] = grad_mag_u8[:p0_h, :p1_w]
				arr[(2,) + widx] = dir0_u8[:p0_h, :p1_w]
				arr[(3,) + widx] = dir1_u8[:p0_h, :p1_w]
				arr[(4,) + widx] = 255
			t_write_sum += float(time.time() - t_wr0)

			done += 1
			elapsed = max(1e-6, float(time.time() - t0))
			per = elapsed / float(done)
			eta = max(0.0, per * float(proc_count - done))
			eta_m = int(eta // 60.0)
			eta_s = int(eta % 60.0)
			bar_w = 30
			fill = int(round((float(done) / float(max(1, proc_count))) * float(bar_w)))
			bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
			print(
				f"\r[preprocess_cos_omezarr] [{bar}] {done}/{proc_count} ({(100.0 * done / max(1, proc_count)):.1f}%) "
				f"eta {eta_m:02d}:{eta_s:02d} read_avg={((1000.0 * t_read_sum) / max(1, done)):.1f}ms "
				f"infer_avg={((1000.0 * t_infer_sum) / max(1, done)):.1f}ms "
				f"write_avg={((1000.0 * t_write_sum) / max(1, done)):.1f}ms (src {ax_name}={ss})",
				end="",
				flush=True,
			)
	print("", flush=True)
	print(
		f"[preprocess_cos_omezarr] profile: processed_slices={proc_count} output_depth={out_sizes[slice_dim]} "
		f"read_avg={((1000.0 * t_read_sum) / max(1, proc_count)):.2f}ms "
		f"infer_avg={((1000.0 * t_infer_sum) / max(1, proc_count)):.2f}ms "
		f"write_avg={((1000.0 * t_write_sum) / max(1, proc_count)):.2f}ms"
	)


# ---------------------------------------------------------------------------
# OME-Zarr helpers
# ---------------------------------------------------------------------------

def _omezarr_level_shape(
	base_shape: tuple[int, int, int], level: int,
) -> tuple[int, int, int]:
	"""Shape at a given pyramid level (halving with ceil, like OME-Zarr)."""
	z, y, x = (int(v) for v in base_shape)
	for _ in range(max(0, int(level))):
		z = max(1, (z + 1) // 2)
		y = max(1, (y + 1) // 2)
		x = max(1, (x + 1) // 2)
	return z, y, x


def _create_omezarr(
	path: str,
	base_shape_zyx: tuple[int, int, int],
	first_level: int,
	n_levels: int,
	chunk: int,
	channel_name: str,
) -> zarr.Group:
	"""Create an OME-Zarr group with pyramid level arrays.

	Creates levels from ``first_level`` to ``n_levels - 1`` (coarser only).
	Each level is a 3D (Z, Y, X) uint8 array.
	"""
	g = zarr.open_group(str(path), mode="w", zarr_format=2)
	datasets = []
	for lv in range(first_level, n_levels):
		sh = _omezarr_level_shape(base_shape_zyx, lv)
		g.create_array(
			str(lv), shape=sh,
			chunks=(min(sh[0], chunk), min(sh[1], chunk), min(sh[2], chunk)),
			dtype=np.uint8, fill_value=0, overwrite=True,
		)
		datasets.append({
			"path": str(lv),
			"coordinateTransformations": [{"type": "scale", "scale": [float(2 ** lv)] * 3}],
		})
	g.attrs["multiscales"] = [{
		"version": "0.4",
		"name": channel_name,
		"axes": [
			{"name": "z", "type": "space", "unit": "pixel"},
			{"name": "y", "type": "space", "unit": "pixel"},
			{"name": "x", "type": "space", "unit": "pixel"},
		],
		"datasets": datasets,
	}]
	return g


def _open_or_create_omezarr(
	path: str,
	base_shape_zyx: tuple[int, int, int],
	first_level: int,
	n_levels: int,
	chunk: int,
	channel_name: str,
) -> zarr.Group:
	"""Open existing OME-Zarr group or create a new one."""
	if os.path.exists(path):
		try:
			g = zarr.open_group(str(path), mode="r+")
			# Verify the data level exists and has correct shape
			expected = _omezarr_level_shape(base_shape_zyx, first_level)
			arr = g[str(first_level)]
			if tuple(int(v) for v in arr.shape) == expected:
				return g
		except (KeyError, Exception):
			pass
		print(f"[predict3d] {path} shape mismatch, recreating", flush=True)
	return _create_omezarr(path, base_shape_zyx, first_level, n_levels, chunk, channel_name)


def _build_omezarr_pyramid(
	omezarr_path: str,
	data_level: int,
	n_levels: int,
	chunk: int,
) -> None:
	"""Build coarser pyramid levels by 2x downsampling from the data level."""
	g = zarr.open_group(str(omezarr_path), mode="r+")
	for lv in range(data_level + 1, n_levels):
		src = g[str(lv - 1)]
		dst = g[str(lv)]
		sh = tuple(int(v) for v in src.shape)
		cz2 = chunk * 2
		t0 = time.time()
		for z0 in range(0, sh[0], cz2):
			z1 = min(sh[0], z0 + cz2)
			for y0 in range(0, sh[1], cz2):
				y1 = min(sh[1], y0 + cz2)
				for x0 in range(0, sh[2], cz2):
					x1 = min(sh[2], x0 + cz2)
					slab = np.asarray(src[z0:z1, y0:y1, x0:x1])
					down = slab[::2, ::2, ::2]
					if down.size > 0:
						dz, dy, dx = z0 // 2, y0 // 2, x0 // 2
						dst[dz:dz + down.shape[0], dy:dy + down.shape[1], dx:dx + down.shape[2]] = down
		print(f"[predict3d] pyramid L{lv} ({tuple(int(v) for v in dst.shape)}) "
			  f"from L{lv-1} in {time.time() - t0:.1f}s", flush=True)


def _find_resume_z(omezarr_path: str, level: int) -> int:
	"""Find the highest z-index with non-zero data in an OME-Zarr level.

	Returns 0 if no data found or OME-Zarr doesn't exist.
	Uses binary search on z-slabs for efficiency.
	"""
	if not os.path.exists(omezarr_path):
		return 0
	try:
		g = zarr.open_group(str(omezarr_path), mode="r")
		arr = g[str(level)]
		z_total = int(arr.shape[0])
		if z_total == 0:
			return 0
		# Binary search: find highest z with any non-zero data
		lo, hi = 0, z_total
		# Quick check: is there any data at all?
		mid_z = z_total // 2
		sample = np.asarray(arr[mid_z])
		if not np.any(sample != 0):
			# Check if there's data in the first half
			sample = np.asarray(arr[0])
			if not np.any(sample != 0):
				return 0
			hi = mid_z
		# Binary search for the last z with data
		while lo < hi - 1:
			mid = (lo + hi) // 2
			sample = np.asarray(arr[mid])
			if np.any(sample != 0):
				lo = mid
			else:
				hi = mid
		return lo + 1  # return count of written z-slices
	except Exception:
		return 0


import ctypes
import ctypes.util

_libc = None


def _get_libc():
	global _libc
	if _libc is None:
		_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
	return _libc


def _release_memmap_pages(arr: np.ndarray, z0: int, z1: int) -> None:
	"""Release memmap pages for z-slice range [z0, z1) via madvise(DONTNEED).

	For a 4D array (C, Z, Y, X), z is axis 1. For the wsum arrays (1, Z, Y, X)
	same layout. This frees both RAM and disk backing.
	"""
	if z1 <= z0 or not hasattr(arr, 'ctypes'):
		return
	try:
		libc = _get_libc()
		# Bytes per z-slice: product of dims after z * itemsize
		# For shape (C, Z, Y, X), stride along z-axis
		bytes_per_z = int(np.prod(arr.shape[2:])) * arr.shape[0] * arr.itemsize
		offset = z0 * bytes_per_z
		length = (z1 - z0) * bytes_per_z
		page = 4096
		aligned_offset = (offset // page) * page
		aligned_end = ((offset + length + page - 1) // page) * page
		aligned_length = aligned_end - aligned_offset
		if aligned_length <= 0:
			return
		addr = ctypes.c_void_p(arr.ctypes.data + aligned_offset)
		MADV_DONTNEED = 4
		libc.madvise(addr, ctypes.c_size_t(aligned_length), ctypes.c_int(MADV_DONTNEED))
	except Exception:
		pass  # best effort — non-critical


def _edt_read_worker(args):
	"""Multiprocessing worker: read chunk from zarr with overlap, binarize.

	Each worker opens its own zarr handle (zarr3 safe).
	Returns (chunk_index, binary_bool_array).
	"""
	idx, pred_path, cz0, cz1, cy0, cy1, cx0, cx1, overlap, pZ, pY, pX = args
	pred = zarr.open(str(pred_path), mode="r")
	rz0 = max(0, cz0 - overlap)
	rz1 = min(pZ, cz1 + overlap)
	ry0 = max(0, cy0 - overlap)
	ry1 = min(pY, cy1 + overlap)
	rx0 = max(0, cx0 - overlap)
	rx1 = min(pX, cx1 + overlap)
	raw = np.asarray(pred[rz0:rz1, ry0:ry1, rx0:rx1])
	return idx, (raw > 0)


def _edt_write_worker(args):
	"""Multiprocessing worker: downsample encoded uint8 and write to zarr.

	Each worker opens its own zarr handle (zarr3 safe).
	"""
	(idx, dt_u8, out_path, out_level,
	 cz0, cz1, cy0, cy1, cx0, cx1,
	 overlap, pZ, pY, pX,
	 pred_z0, pred_y0, pred_x0,
	 out_z0, out_y0, out_x0,
	 scaledown) = args

	rz0 = max(0, cz0 - overlap)
	ry0 = max(0, cy0 - overlap)
	rx0 = max(0, cx0 - overlap)
	sz, sy, sx = cz1 - cz0, cy1 - cy0, cx1 - cx0
	center = dt_u8[cz0 - rz0:cz0 - rz0 + sz,
				   cy0 - ry0:cy0 - ry0 + sy,
				   cx0 - rx0:cx0 - rx0 + sx]

	out_arr = zarr.open(str(out_path), mode="r+")[str(out_level)]
	out_shape = tuple(int(v) for v in out_arr.shape)

	for zl in range(0, sz, scaledown):
		slc = center[zl]
		if scaledown > 1:
			oh = max(1, slc.shape[0] // scaledown)
			ow = max(1, slc.shape[1] // scaledown)
			slc = cv2.resize(slc, (ow, oh), interpolation=cv2.INTER_AREA)
		ozi = out_z0 + (cz0 - pred_z0 + zl) // scaledown
		oyi = out_y0 + (cy0 - pred_y0) // scaledown
		oxi = out_x0 + (cx0 - pred_x0) // scaledown
		oy_end = min(out_shape[1], oyi + slc.shape[0])
		ox_end = min(out_shape[2], oxi + slc.shape[1])
		wy = oy_end - oyi
		wx = ox_end - oxi
		if ozi < out_shape[0] and wy > 0 and wx > 0:
			out_arr[ozi, oyi:oy_end, oxi:ox_end] = slc[:wy, :wx]

	return idx


def _compute_pred_dt_slab(
	*,
	pred_zarr,
	pred_path: str,
	output_level_arr,
	output_omezarr_path: str,
	output_level_key: str,
	pred_z0: int, pred_z1: int,
	pred_y0: int, pred_y1: int,
	pred_x0: int, pred_x1: int,
	out_z0: int, out_y0: int, out_x0: int,
	scaledown: int,
	overlap: int = 48,
	edt_chunk: int = 448,
	ome_chunk: int = 32,
	progress: dict | None = None,
	read_workers: int = 0,
) -> None:
	"""Compute signed distance-to-surface for a region, chunked for GPU EDT.

	Pipeline: multiprocessing workers read+binarize chunks from zarr in
	parallel (each opens its own handle to avoid zarr3 concurrency issues),
	feeding a buffer of ready chunks.  The main process pulls chunks in
	order and runs GPU EDT, keeping the GPU saturated while I/O happens
	in the background.
	"""
	if not _HAS_CUPY:
		raise ImportError("pred-dt requires CuPy for GPU EDT")

	import multiprocessing as _mp

	_MAX_DT = 48
	pZ, pY, pX = (int(v) for v in pred_zarr.shape)

	# Align edt_chunk to output zarr chunk boundaries to avoid write collisions
	# and enable per-chunk resume.  Output chunk in pred coords = ome_chunk * scaledown.
	# Use largest multiple of out_chunk_in_pred that fits in GPU memory.
	# EDT needs ~12 bytes/voxel peak (bool + 2×float32 + uint8).
	# Query available GPU memory to size chunks appropriately.
	out_chunk_in_pred = max(1, scaledown) * max(1, ome_chunk)
	try:
		_gpu_free = cp.cuda.Device().mem_info[0]
		# Use at most 60% of free memory, 12 bytes per voxel
		max_voxels = int(_gpu_free * 0.6 / 12)
		max_side = max(out_chunk_in_pred, int(max_voxels ** (1.0 / 3.0)))
	except Exception:
		max_side = 512
	edt_chunk = max(out_chunk_in_pred,
					(min(edt_chunk, max_side) // out_chunk_in_pred) * out_chunk_in_pred)
	print(f"[pred_dt] edt_chunk={edt_chunk} (out_chunk_in_pred={out_chunk_in_pred}, "
		  f"gpu_free={_gpu_free / 2**30:.1f}GiB, max_side={max_side})", flush=True)
	# Align region starts to output chunk boundaries (round down start, round up end)
	aligned_z0 = (pred_z0 // out_chunk_in_pred) * out_chunk_in_pred
	aligned_y0 = (pred_y0 // out_chunk_in_pred) * out_chunk_in_pred
	aligned_x0 = (pred_x0 // out_chunk_in_pred) * out_chunk_in_pred
	aligned_z1 = min(pZ, ((pred_z1 + out_chunk_in_pred - 1) // out_chunk_in_pred) * out_chunk_in_pred)
	aligned_y1 = min(pY, ((pred_y1 + out_chunk_in_pred - 1) // out_chunk_in_pred) * out_chunk_in_pred)
	aligned_x1 = min(pX, ((pred_x1 + out_chunk_in_pred - 1) // out_chunk_in_pred) * out_chunk_in_pred)

	# Helper: check if an output chunk file already exists (zarr v2, dim_sep="/")
	def _out_chunk_exists(cz0, cy0, cx0):
		"""Check if the output zarr chunk for this pred-src region exists."""
		ozi = out_z0 + (cz0 - pred_z0) // scaledown
		oyi = out_y0 + (cy0 - pred_y0) // scaledown
		oxi = out_x0 + (cx0 - pred_x0) // scaledown
		iz = ozi // ome_chunk
		iy = oyi // ome_chunk
		ix = oxi // ome_chunk
		chunk_path = os.path.join(output_omezarr_path, output_level_key, str(iz), str(iy), str(ix))
		return os.path.isfile(chunk_path)

	# Build work list, skipping chunks whose output already exists
	work: list[tuple[int, int, int, int, int, int]] = []
	skipped = 0
	for cz0 in range(aligned_z0, aligned_z1, edt_chunk):
		cz1 = min(aligned_z1, cz0 + edt_chunk)
		for cy0 in range(aligned_y0, aligned_y1, edt_chunk):
			cy1 = min(aligned_y1, cy0 + edt_chunk)
			for cx0 in range(aligned_x0, aligned_x1, edt_chunk):
				cx1 = min(aligned_x1, cx0 + edt_chunk)
				if _out_chunk_exists(cz0, cy0, cx0):
					skipped += 1
					continue
				work.append((cz0, cz1, cy0, cy1, cx0, cx1))

	n_chunks = len(work)
	if skipped > 0:
		print(f"[pred_dt] skipped {skipped} already-processed chunks", flush=True)
	if n_chunks == 0:
		return

	t0 = time.time()
	n_workers = min(read_workers if read_workers > 0 else max(1, (os.cpu_count() or 4) // 2),
				   n_chunks)

	print(f"[pred_dt] {n_chunks} chunks to process, {n_workers} workers, edt_chunk={edt_chunk}",
		  flush=True)

	# --- Pipelined: parallel READ → streamed GPU (H2D|EDT|D2H) → parallel WRITE ---
	# Three CUDA streams overlap transfers with compute:
	#   stream_h2d: host→device transfer of next chunk
	#   stream_compute: EDT + encode on current chunk
	#   stream_d2h: device→host transfer of previous result
	# Reader/writer processes run in parallel for I/O.

	read_args = []
	for idx, (cz0, cz1, cy0, cy1, cx0, cx1) in enumerate(work):
		read_args.append((idx, pred_path, cz0, cz1, cy0, cy1, cx0, cx1, overlap, pZ, pY, pX))

	pending: dict[int, np.ndarray] = {}
	next_gpu_idx = 0
	write_pool = _mp.Pool(processes=n_workers)
	write_futures = []
	done_count = [0]

	stream_h2d = cp.cuda.Stream(non_blocking=True)
	stream_compute = cp.cuda.Stream(non_blocking=True)
	stream_d2h = cp.cuda.Stream(non_blocking=True)

	# Pipeline state: staged GPU buffers
	staged_gpu = None       # (idx, m_gpu) ready for compute after H2D done
	computing = None        # (idx, dt_enc_gpu) being computed
	transferring = None     # (idx, dt_enc_gpu, pinned_buf) being D2H transferred

	def _gpu_encode(m_gpu):
		"""EDT + signed encode on current compute stream. Returns uint8 GPU array."""
		outer_dt = cnd.distance_transform_edt(~m_gpu)
		inner_dt = cnd.distance_transform_edt(m_gpu)
		outer_enc = (128 - cp.clip(cp.round(outer_dt), 1, _MAX_DT)).astype(cp.uint8)
		inner_enc = (127 + cp.clip(cp.round(inner_dt), 1, _MAX_DT)).astype(cp.uint8)
		result = cp.where(m_gpu, inner_enc, outer_enc)
		del outer_dt, inner_dt, outer_enc, inner_enc
		return result

	def _dispatch_write(idx, dt_u8_np):
		"""Send result to write worker pool."""
		cz0, cz1, cy0, cy1, cx0, cx1 = work[idx]
		wa = (idx, dt_u8_np, output_omezarr_path, output_level_key,
			  cz0, cz1, cy0, cy1, cx0, cx1,
			  overlap, pZ, pY, pX,
			  pred_z0, pred_y0, pred_x0,
			  out_z0, out_y0, out_x0, scaledown)
		write_futures.append(write_pool.apply_async(_edt_write_worker, (wa,)))

	def _collect_writes():
		nonlocal write_futures
		new = []
		for wf in write_futures:
			if wf.ready():
				wf.get()
				done_count[0] += 1
				if progress is not None:
					progress["edt_done"] = progress.get("edt_done", 0) + 1
			else:
				new.append(wf)
		write_futures = new

	def _print_progress():
		gpu_done = next_gpu_idx
		if gpu_done == 0:
			return
		elapsed = max(1e-6, time.time() - t0)
		eta = elapsed / gpu_done * (n_chunks - gpu_done)
		overall = ""
		if progress is not None:
			oe = max(1e-6, time.time() - progress["t0"])
			frac = progress.get("tiles_done", 0) / max(1, progress.get("tiles_total", 1))
			edt_frac = progress.get("edt_done", 0) / max(1, progress.get("edt_total_est", 1))
			overall_frac = 0.7 * frac + 0.3 * edt_frac
			if overall_frac > 0.01:
				overall_eta = oe / overall_frac * (1.0 - overall_frac)
				overall = f" | overall eta {int(overall_eta // 60):02d}:{int(overall_eta % 60):02d}"
		print(f"\r[pred_dt] gpu {gpu_done}/{n_chunks} "
			  f"written {done_count[0]}/{n_chunks} "
			  f"({100.0 * gpu_done / n_chunks:.0f}%) "
			  f"eta {int(eta // 60):02d}:{int(eta % 60):02d}"
			  f"{overall}  ",
			  end="", flush=True)

	with _mp.Pool(processes=n_workers) as read_pool:
		read_iter = read_pool.imap_unordered(_edt_read_worker, read_args)

		for idx, binary in read_iter:
			pending[idx] = binary

			while next_gpu_idx in pending:
				cur_binary = pending.pop(next_gpu_idx)
				cur_idx = next_gpu_idx
				next_gpu_idx += 1

				# Stage 1: H2D transfer (async) — wait for previous H2D first
				stream_h2d.synchronize()
				with stream_h2d:
					new_m_gpu = cp.asarray(cur_binary)

				# Stage 2: if we have a staged buffer, start compute (async)
				if staged_gpu is not None:
					prev_idx, prev_m_gpu = staged_gpu
					# Wait for previous compute to finish before reusing stream
					stream_compute.synchronize()
					with stream_compute:
						# Wait for that buffer's H2D to be done
						stream_compute.wait_event(stream_h2d.record())
						prev_result = _gpu_encode(prev_m_gpu)
						del prev_m_gpu

					# Stage 3: if we have a computing result, start D2H (async)
					if computing is not None:
						comp_idx, comp_result = computing
						stream_d2h.synchronize()
						with stream_d2h:
							stream_d2h.wait_event(stream_compute.record())
							dt_u8 = cp.asnumpy(comp_result)
						# D2H is synchronous for asnumpy — result is ready
						del comp_result
						_dispatch_write(comp_idx, dt_u8)
						_collect_writes()
						_print_progress()

					computing = (prev_idx, prev_result)

				staged_gpu = (cur_idx, new_m_gpu)

		# Drain pipeline: process staged buffer
		if staged_gpu is not None:
			s_idx, s_m_gpu = staged_gpu
			stream_h2d.synchronize()
			with stream_compute:
				s_result = _gpu_encode(s_m_gpu)
				del s_m_gpu
			if computing is not None:
				comp_idx, comp_result = computing
				stream_compute.synchronize()
				dt_u8 = cp.asnumpy(comp_result)
				del comp_result
				_dispatch_write(comp_idx, dt_u8)
			computing = (s_idx, s_result)

		# Drain: transfer last computed result
		if computing is not None:
			comp_idx, comp_result = computing
			stream_compute.synchronize()
			dt_u8 = cp.asnumpy(comp_result)
			del comp_result
			_dispatch_write(comp_idx, dt_u8)

	# Wait for all writes
	for wf in write_futures:
		wf.get()
		done_count[0] += 1
		if progress is not None:
			progress["edt_done"] = progress.get("edt_done", 0) + 1
	write_pool.close()
	write_pool.join()

	cp.get_default_memory_pool().free_all_blocks()
	_print_progress()
	print(f"\r[pred_dt] {n_chunks} chunks in {time.time() - t0:.1f}s" + " " * 30,
		  flush=True)


def _compute_pred_dt_channel(
	*,
	pred_path: str,
	output_arr: zarr.Array,
	channel_idx: int,
	ref_z: int, ref_y: int, ref_x: int,
	scaledown: int,
	crop_xyzwhd: list[int] | None,
	chunk_depth: int = 256,
	chunk_yx: int = 256,
	overlap: int = 64,
) -> None:
	"""Compute signed distance-to-surface channel from a prediction zarr.

	Encoding (uint8): outside=[80,127], inside=[128,175], no data=0.
	Boundary jump is exactly 1 (127→128).
	"""
	# Overlap must cover MAX_DT (48) so chunk-boundary EDT is accurate
	overlap = max(overlap, 48)
	pred = zarr.open(str(pred_path), mode="r")
	if not hasattr(pred, "shape"):
		raise ValueError(f"pred-dt must point to a zarr array, got group: {pred_path}")
	pred_shape = tuple(int(v) for v in pred.shape)
	if len(pred_shape) != 3:
		raise ValueError(f"pred-dt array must be (Z,Y,X), got shape {pred_shape}")
	pZ, pY, pX = pred_shape

	# Apply crop to prediction volume (same coordinate space as input volume)
	if crop_xyzwhd is not None:
		cx, cy, cz, cw, ch, cd = (int(v) for v in crop_xyzwhd)
		p_z0 = max(0, min(cz, pZ))
		p_z1 = max(p_z0, min(cz + max(0, cd), pZ))
		p_y0 = max(0, min(cy, pY))
		p_y1 = max(p_y0, min(cy + max(0, ch), pY))
		p_x0 = max(0, min(cx, pX))
		p_x1 = max(p_x0, min(cx + max(0, cw), pX))
	else:
		p_z0, p_z1 = 0, pZ
		p_y0, p_y1 = 0, pY
		p_x0, p_x1 = 0, pX

	total_z = p_z1 - p_z0
	total_y = p_y1 - p_y0
	total_x = p_x1 - p_x0
	if total_z <= 0:
		print(f"[pred_dt] WARNING: empty z range after crop, skipping")
		return

	# Round up chunk_depth to a multiple of scaledown so z sampling phase
	# stays aligned across chunk boundaries
	if scaledown > 1:
		chunk_depth = ((chunk_depth + scaledown - 1) // scaledown) * scaledown

	# Build chunk grid for all 3 axes
	z_starts = list(range(p_z0, p_z1, chunk_depth))
	y_starts = list(range(p_y0, p_y1, chunk_yx))
	x_starts = list(range(p_x0, p_x1, chunk_yx))
	n_chunks = len(z_starts) * len(y_starts) * len(x_starts)

	print(
		f"[pred_dt] pred={pred_path} shape={pred_shape} crop_z=[{p_z0},{p_z1}) "
		f"crop_y=[{p_y0},{p_y1}) crop_x=[{p_x0},{p_x1}) "
		f"scaledown={scaledown} "
		f"chunk_depth={chunk_depth} chunk_yx={chunk_yx} overlap={overlap}",
		flush=True,
	)
	print(
		f"[pred_dt] {len(z_starts)}z x {len(y_starts)}y x {len(x_starts)}x = {n_chunks} chunk(s)",
		flush=True,
	)

	t0 = time.time()
	chunk_i = 0

	for z_pos in z_starts:
		z_chunk_end = min(p_z1, z_pos + chunk_depth)
		# Padded read range in Z
		read_z0 = max(p_z0, z_pos - overlap)
		read_z1 = min(p_z1, z_chunk_end + overlap)

		for y_pos in y_starts:
			y_chunk_end = min(p_y1, y_pos + chunk_yx)
			# Padded read range in Y
			read_y0 = max(p_y0, y_pos - overlap)
			read_y1 = min(p_y1, y_chunk_end + overlap)

			for x_pos in x_starts:
				x_chunk_end = min(p_x1, x_pos + chunk_yx)
				# Padded read range in X
				read_x0 = max(p_x0, x_pos - overlap)
				read_x1 = min(p_x1, x_chunk_end + overlap)

				chunk_i += 1
				read_sz = (read_z1 - read_z0, read_y1 - read_y0, read_x1 - read_x0)
				print(
					f"[pred_dt] chunk {chunk_i}/{n_chunks}  "
					f"z=[{z_pos},{z_chunk_end}) y=[{y_pos},{y_chunk_end}) x=[{x_pos},{x_chunk_end})  "
					f"reading {read_sz} ...",
					end="", flush=True,
				)
				t_read = time.time()
				chunk_np = np.asarray(pred[read_z0:read_z1, read_y0:read_y1, read_x0:read_x1])
				print(f" {time.time() - t_read:.1f}s", flush=True)

				# Binarize and compute signed distance transform
				binary = chunk_np > 0
				del chunk_np
				_MAX_DT = 48  # max distance in voxels for encoding

				def _run_edt(mask_np):
					"""Run EDT on a boolean mask, return float32 numpy."""
					if _HAS_CUPY:
						cp.get_default_memory_pool().free_all_blocks()
						try:
							m_gpu = cp.asarray(mask_np)
							d_gpu = cnd.distance_transform_edt(m_gpu)
							d = cp.asnumpy(d_gpu).astype(np.float32)
							del m_gpu, d_gpu
							cp.get_default_memory_pool().free_all_blocks()
							return d
						except cp.cuda.memory.OutOfMemoryError:
							cp.get_default_memory_pool().free_all_blocks()
							if not _HAS_EDT:
								raise
					if _HAS_EDT:
						return edt_mod.edt(mask_np, parallel=32).astype(np.float32)
					raise ImportError("pred-dt requires CuPy (GPU) or 'edt' package (CPU)")

				nvoxels = binary.size
				if _HAS_CUPY:
					gpu_free = cp.cuda.Device().mem_info[0]
					print(
						f"[pred_dt] chunk {chunk_i}/{n_chunks}  signed EDT (GPU) "
						f"voxels={nvoxels:,} free={gpu_free / 2**30:.1f}GiB ...",
						end="", flush=True,
					)
				else:
					print(
						f"[pred_dt] chunk {chunk_i}/{n_chunks}  signed EDT (CPU) ...",
						end="", flush=True,
					)
				t_edt = time.time()
				outer_dt = _run_edt(~binary)  # bg → distance to fg
				inner_dt = _run_edt(binary)   # fg → distance to bg
				print(f" {time.time() - t_edt:.1f}s", flush=True)

				# Signed distance encoding:
				#   Outside (binary=0): 128 - clip(round(outer_dt), 1, 48) → [80, 127]
				#   Inside  (binary=1): 127 + clip(round(inner_dt), 1, 48) → [128, 175]
				#   Boundary jump: 127 → 128 (exactly 1)
				#   Value 0 = no data (zarr fill_value)
				outer_enc = (128 - np.clip(np.round(outer_dt), 1, _MAX_DT)).astype(np.uint8)
				inner_enc = (127 + np.clip(np.round(inner_dt), 1, _MAX_DT)).astype(np.uint8)
				dt_encoded = np.where(binary, inner_enc, outer_enc)
				del binary, outer_dt, inner_dt, outer_enc, inner_enc

				# Crop off overlap padding to keep center region
				pad_z = z_pos - read_z0
				pad_y = y_pos - read_y0
				pad_x = x_pos - read_x0
				center_z = z_chunk_end - z_pos
				center_y = y_chunk_end - y_pos
				center_x = x_chunk_end - x_pos
				dt_u8 = dt_encoded[pad_z:pad_z + center_z, pad_y:pad_y + center_y, pad_x:pad_x + center_x]
				del dt_encoded

				# Downscale to output grid
				# Z: subsample at scaledown spacing
				z_indices_full = list(range(z_pos, z_chunk_end, scaledown))
				if not z_indices_full:
					continue

				for zf in z_indices_full:
					local_z = zf - z_pos
					if local_z < 0 or local_z >= dt_u8.shape[0]:
						continue
					slc = dt_u8[local_z]

					# YX downscale
					if scaledown > 1:
						out_h = max(1, slc.shape[0] // scaledown)
						out_w = max(1, slc.shape[1] // scaledown)
						slc = cv2.resize(slc, (out_w, out_h), interpolation=cv2.INTER_AREA)

					# Output indices
					out_zi = zf // scaledown
					if out_zi < 0 or out_zi >= ref_z:
						continue
					out_y0 = y_pos // max(1, scaledown)
					out_x0 = x_pos // max(1, scaledown)
					out_y1 = min(ref_y, out_y0 + slc.shape[0])
					out_x1 = min(ref_x, out_x0 + slc.shape[1])
					wy = out_y1 - out_y0
					wx = out_x1 - out_x0
					if wy > 0 and wx > 0:
						output_arr[channel_idx, out_zi, out_y0:out_y1, out_x0:out_x1] = slc[:wy, :wx]

				elapsed = max(1e-6, time.time() - t0)
				progress = float(chunk_i) / float(max(1, n_chunks))
				eta = elapsed / max(1e-6, progress) * (1.0 - progress)
				print(
					f"[pred_dt] chunk {chunk_i}/{n_chunks} done — "
					f"{100.0 * progress:.1f}%  "
					f"elapsed {int(elapsed // 60):02d}:{int(elapsed % 60):02d}  "
					f"eta {int(eta // 60):02d}:{int(eta % 60):02d}",
					flush=True,
				)

	print(f"[pred_dt] done in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 3D UNet predict mode
# ---------------------------------------------------------------------------

def _read_tile_zarr(
	zarr_arr,
	volume_shape: tuple[int, int, int],
	crop_offset: tuple[int, int, int],
	tz: int, ty: int, tx: int,
	tile_size: int,
	border: int,
) -> np.ndarray:
	"""Read a single tile from zarr, using reflect-padding only at volume boundaries.

	The tile grid is defined in padded-crop space (crop + border on each side).
	We map tile coords back to zarr coords: zarr_coord = tile_coord + crop_offset - border.
	Where zarr coords fall outside [0, vol_dim), we reflect-pad.
	"""
	Zv, Yv, Xv = volume_shape
	oz, oy, ox = crop_offset

	# Map tile position in padded space to zarr coordinates
	src_z0 = tz + oz - border
	src_y0 = ty + oy - border
	src_x0 = tx + ox - border

	src_z1 = src_z0 + tile_size
	src_y1 = src_y0 + tile_size
	src_x1 = src_x0 + tile_size

	# Clamp to valid zarr range
	rz0 = max(0, src_z0)
	ry0 = max(0, src_y0)
	rx0 = max(0, src_x0)
	rz1 = min(Zv, src_z1)
	ry1 = min(Yv, src_y1)
	rx1 = min(Xv, src_x1)

	if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0:
		return np.zeros((tile_size, tile_size, tile_size), dtype=np.uint8)

	chunk = np.asarray(zarr_arr[rz0:rz1, ry0:ry1, rx0:rx1])

	# Pad if we went out of bounds
	pad_before = (rz0 - src_z0, ry0 - src_y0, rx0 - src_x0)
	pad_after = (src_z1 - rz1, src_y1 - ry1, src_x1 - rx1)
	needs_pad = any(p > 0 for p in pad_before + pad_after)
	if needs_pad:
		chunk = np.pad(
			chunk,
			[(pad_before[0], pad_after[0]),
			 (pad_before[1], pad_after[1]),
			 (pad_before[2], pad_after[2])],
			mode="reflect",
		)
	return chunk


def _calibrate_instance_norm(
	model,
	zarr_arr,
	*,
	crop_slices: tuple[int, int, int, int, int, int],
	device: torch.device,
	tile_size: int,
	n_tiles: int = 16,
) -> None:
	"""Calibrate InstanceNorm3d running statistics from representative tiles.

	Enables track_running_stats on all InstanceNorm layers, runs a few forward
	passes in train mode to accumulate running mean/var, then switches back to
	eval mode so inference uses fixed statistics instead of per-tile stats.
	"""
	# Find all InstanceNorm layers
	in_layers = [m for m in model.modules() if isinstance(m, torch.nn.InstanceNorm3d)]
	if not in_layers:
		print("[calibrate_norm] no InstanceNorm3d layers found, skipping")
		return

	print(f"[calibrate_norm] calibrating {len(in_layers)} InstanceNorm3d layers with {n_tiles} tiles")

	# Enable running stats
	for m in in_layers:
		m.track_running_stats = True
		m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
		n = m.num_features
		m.running_mean = torch.zeros(n, device=device)
		m.running_var = torch.ones(n, device=device)

	z0, z1, y0, y1, x0, x1 = crop_slices
	volume_shape = tuple(int(v) for v in zarr_arr.shape)
	crop_offset = (z0, y0, x0)

	# Sample random tile positions within the crop
	rng = np.random.default_rng(42)
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	max_tz = max(0, nz - tile_size)
	max_ty = max(0, ny - tile_size)
	max_tx = max(0, nx - tile_size)

	model.train()
	with torch.inference_mode():
		for i in range(n_tiles):
			tz = int(rng.integers(0, max_tz + 1)) if max_tz > 0 else 0
			ty = int(rng.integers(0, max_ty + 1)) if max_ty > 0 else 0
			tx = int(rng.integers(0, max_tx + 1)) if max_tx > 0 else 0
			tile_np = _read_tile_zarr(zarr_arr, volume_shape, crop_offset, tz, ty, tx, tile_size, 0)
			if tile_np.dtype == np.uint16:
				tile_np = (tile_np // 257).astype(np.uint8)
			tile_t = torch.from_numpy(tile_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
			model(tile_t)
	model.eval()
	print("[calibrate_norm] done")


def _infer_tiled_3d(
	model,
	zarr_arr,
	*,
	crop_slices: tuple[int, int, int, int, int, int],
	device: torch.device,
	tile_size: int = 256,
	overlap: int = 64,
	border: int = 16,
	out_channels: int = 8,
	cos_scaledown: int = 2,
	other_scaledown: int = 4,
	tmp_dir: str | None = None,
	output_sigmoid: bool = True,
	on_z_complete=None,
	skip_z_positions: int = 0,
	progress: dict | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Run 3D UNet inference with dual-resolution accumulators.

	Channel 0 (cos) is accumulated at cos_scaledown resolution.
	Channels 1..out_channels-1 are accumulated at other_scaledown resolution.

	If *on_z_complete* is provided, it is called after each z-row of tiles
	with ``(acc_fine, wsum_fine, acc_coarse, wsum_coarse, complete_z_padded,
	pad0)``.  The callback should normalize, post-process, write output, and
	release memmap pages for the flushed region.  When a callback is
	provided the function returns ``None`` (output is consumed by the
	callback).

	*skip_z_positions* skips the first N z-positions (for resume).

	Returns:
		(cos_result, other_result) where:
		  cos_result:   (1, Z_fine, Y_fine, X_fine) float32
		  other_result: (out_channels-1, Z_coarse, Y_coarse, X_coarse) float32
	"""
	z0, z1, y0, y1, x0, x1 = crop_slices
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	volume_shape = tuple(int(v) for v in zarr_arr.shape)

	sd_fine = max(1, int(cos_scaledown))
	sd_coarse = max(1, int(other_scaledown))
	# Use finest scaledown for tiling alignment
	sd_min = min(sd_fine, sd_coarse)
	stride = max(1, tile_size - overlap)

	# Validate alignment
	for sd_label, sd_val in [("cos_scaledown", sd_fine), ("other_scaledown", sd_coarse)]:
		if sd_val > 1:
			for name, val in [("tile_size", tile_size), ("stride", stride), ("border", border)]:
				if val % sd_val != 0:
					raise ValueError(f"{name}={val} must be divisible by {sd_label}={sd_val}")

	# Padded crop dimensions (border on each side)
	pad0 = max(0, int(border))
	Zp = nz + 2 * pad0
	Yp = ny + 2 * pad0
	Xp = nx + 2 * pad0

	# Round up to coarsest scaledown-multiple
	sd_max = max(sd_fine, sd_coarse)
	if sd_max > 1:
		Zp = ((Zp + sd_max - 1) // sd_max) * sd_max
		Yp = ((Yp + sd_max - 1) // sd_max) * sd_max
		Xp = ((Xp + sd_max - 1) // sd_max) * sd_max

	# Output dimensions for each accumulator
	Zo_f, Yo_f, Xo_f = Zp // sd_fine, Yp // sd_fine, Xp // sd_fine
	Zo_c, Yo_c, Xo_c = Zp // sd_coarse, Yp // sd_coarse, Xp // sd_coarse

	ov_eff = max(0, overlap - 2 * border)

	def _build_positions(size, tile, s):
		if size <= tile:
			return [0]
		positions = list(range(0, size - tile + 1, s))
		last = size - tile
		if positions[-1] != last:
			positions.append(last)
		return positions

	z_positions = _build_positions(Zp, tile_size, stride)
	y_positions = _build_positions(Yp, tile_size, stride)
	x_positions = _build_positions(Xp, tile_size, stride)

	def _blend_ramp(length, ov, b):
		ramp = np.zeros(length, dtype=np.float32)
		if length <= 0:
			return ramp
		core_start = min(b, length)
		core_end = max(core_start, length - b)
		core_len = core_end - core_start
		if core_len <= 0:
			return ramp
		core = np.ones(core_len, dtype=np.float32)
		if ov > 0:
			ov_core = min(ov, core_len // 2)
			if ov_core > 0:
				edges = np.linspace(0.0, 1.0, ov_core + 1, dtype=np.float32)[1:]
				core[:ov_core] = edges
				core[-ov_core:] = edges[::-1]
		ramp[core_start:core_end] = core
		return ramp

	# Precompute full-tile blend weight on GPU
	rz_full = _blend_ramp(tile_size, ov_eff, border)
	ry_full = _blend_ramp(tile_size, ov_eff, border)
	rx_full = _blend_ramp(tile_size, ov_eff, border)
	w_full = torch.from_numpy(
		rz_full[:, None, None] * ry_full[None, :, None] * rx_full[None, None, :]
	).to(device)  # (tile, tile, tile)

	# Precompute downscaled weights for each accumulator
	w_fine = (_pyrdown3d(w_full.unsqueeze(0), factor=sd_fine).squeeze(0).cpu().numpy()
			  if sd_fine > 1 else w_full.cpu().numpy())
	w_coarse = (_pyrdown3d(w_full.unsqueeze(0), factor=sd_coarse).squeeze(0).cpu().numpy()
				if sd_coarse > 1 else w_full.cpu().numpy())

	# Memmap accumulators — place next to output to avoid /tmp overflow
	def _make_memmap(suffix, shape):
		if tmp_dir:
			p = os.path.join(tmp_dir, f".predict3d_{suffix}.tmp")
		else:
			p = tempfile.mktemp(suffix=f".{suffix}")
		mm = np.memmap(p, dtype=np.float32, mode="w+", shape=shape)
		try:
			os.unlink(p)
		except OSError:
			pass
		return mm

	n_other = out_channels - 1
	acc_fine = _make_memmap("acc_fine", (1, Zo_f, Yo_f, Xo_f))
	wsum_fine = _make_memmap("wsum_fine", (1, Zo_f, Yo_f, Xo_f))
	acc_coarse = _make_memmap("acc_coarse", (n_other, Zo_c, Yo_c, Xo_c))
	wsum_coarse = _make_memmap("wsum_coarse", (1, Zo_c, Yo_c, Xo_c))

	fine_bytes = (np.prod(acc_fine.shape) + np.prod(wsum_fine.shape)) * 4
	coarse_bytes = (np.prod(acc_coarse.shape) + np.prod(wsum_coarse.shape)) * 4
	print(
		f"[predict3d] accumulators: fine ({1},{Zo_f},{Yo_f},{Xo_f}) sd={sd_fine} "
		f"({fine_bytes / (1024**3):.2f} GiB) + "
		f"coarse ({n_other},{Zo_c},{Yo_c},{Xo_c}) sd={sd_coarse} "
		f"({coarse_bytes / (1024**3):.2f} GiB)",
		flush=True,
	)

	tiles_per_zrow = len(y_positions) * len(x_positions)
	total_tiles = len(z_positions) * tiles_per_zrow
	skipped_tiles = skip_z_positions * tiles_per_zrow
	done = skipped_tiles
	t0 = time.time()
	crop_offset = (z0, y0, x0)
	if progress is not None:
		progress["tiles_total"] = total_tiles
		progress["tiles_done"] = done

	for i_tz, tz in enumerate(z_positions):
		if i_tz < skip_z_positions:
			continue  # resume: skip already-processed z-rows

		for ty in y_positions:
			for tx in x_positions:
				# Read tile from zarr (lazy)
				tile_np = _read_tile_zarr(
					zarr_arr, volume_shape, crop_offset,
					tz, ty, tx, tile_size, border,
				)
				if tile_np.dtype == np.uint16:
					tile_np = (tile_np // 257).astype(np.uint8)

				tile_f = tile_np.astype(np.float32) / 255.0
				tile_t = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)

				with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
					pred = model(tile_t)
				if isinstance(pred, dict):
					pred = pred["output"]

				# Diagnostics
				raw_nan = torch.isnan(pred).sum().item()
				if raw_nan > 0 or done == skipped_tiles:
					print(flush=True)
					print(
						f"  tile {done}/{total_tiles} "
						f"pos=({tz},{ty},{tx}) "
						f"input: min={tile_f.min():.4f} max={tile_f.max():.4f} "
						f"raw_out: min={pred.min().item():.4f} max={pred.max().item():.4f} "
						f"nan={raw_nan}/{pred.numel()} "
						f"dtype={pred.dtype}",
						flush=True,
					)

				# Activate on GPU
				if output_sigmoid:
					pred = torch.sigmoid(pred.float())  # (1, C, tz, ty, tx)
				else:
					pred = pred.float().clamp(0.0, 1.0)

				# Split channels and accumulate at different resolutions
				pred_cos = pred[0, 0:1] * w_full    # (1, tz, ty, tx)
				pred_other = pred[0, 1:] * w_full    # (C-1, tz, ty, tx)

				# Downscale cos channel
				if sd_fine > 1:
					pred_cos = _pyrdown3d(pred_cos, factor=sd_fine)
				cos_np = pred_cos.cpu().numpy()
				ts_f = tile_size // sd_fine
				azl_f, ayl_f, axl_f = tz // sd_fine, ty // sd_fine, tx // sd_fine
				azr_f = min(azl_f + ts_f, Zo_f)
				ayr_f = min(ayl_f + ts_f, Yo_f)
				axr_f = min(axl_f + ts_f, Xo_f)
				pz_f, py_f, px_f = azr_f - azl_f, ayr_f - ayl_f, axr_f - axl_f
				acc_fine[:, azl_f:azr_f, ayl_f:ayr_f, axl_f:axr_f] += cos_np[:, :pz_f, :py_f, :px_f]
				wsum_fine[0, azl_f:azr_f, ayl_f:ayr_f, axl_f:axr_f] += w_fine[:pz_f, :py_f, :px_f]

				# Downscale other channels
				if sd_coarse > 1:
					pred_other = _pyrdown3d(pred_other, factor=sd_coarse)
				other_np = pred_other.cpu().numpy()
				ts_c = tile_size // sd_coarse
				azl_c, ayl_c, axl_c = tz // sd_coarse, ty // sd_coarse, tx // sd_coarse
				azr_c = min(azl_c + ts_c, Zo_c)
				ayr_c = min(ayl_c + ts_c, Yo_c)
				axr_c = min(axl_c + ts_c, Xo_c)
				pz_c, py_c, px_c = azr_c - azl_c, ayr_c - ayl_c, axr_c - axl_c
				acc_coarse[:, azl_c:azr_c, ayl_c:ayr_c, axl_c:axr_c] += other_np[:, :pz_c, :py_c, :px_c]
				wsum_coarse[0, azl_c:azr_c, ayl_c:ayr_c, axl_c:axr_c] += w_coarse[:pz_c, :py_c, :px_c]

				done += 1
				if progress is not None:
					progress["tiles_done"] = done
				elapsed = max(1e-6, time.time() - t0)
				actual_done = done - skipped_tiles
				per = elapsed / max(1, actual_done)
				remaining = total_tiles - done
				eta = max(0.0, per * remaining)
				overall = ""
				if progress is not None:
					oe = max(1e-6, time.time() - progress["t0"])
					frac = done / max(1, total_tiles)
					overall_frac = 0.7 * frac  # tiles ~70% of total work
					if overall_frac > 0.01:
						overall_eta = oe / overall_frac * (1.0 - overall_frac)
						overall = f" | overall eta {int(overall_eta // 60):02d}:{int(overall_eta % 60):02d}"
				bar_w = 30
				fill = int(round(done / max(1, total_tiles) * bar_w))
				bar = "#" * fill + "-" * (bar_w - fill)
				print(
					f"\r[predict3d] [{bar}] {done}/{total_tiles} tiles "
					f"({100.0 * done / max(1, total_tiles):.1f}%) "
					f"eta {int(eta // 60):02d}:{int(eta % 60):02d} "
					f"avg={1000.0 * per:.0f}ms/tile"
					f"{overall}",
					end="", flush=True,
				)

		# --- After all (ty, tx) for this tz: flush completed z-band ---
		if on_z_complete is not None:
			next_tz = z_positions[i_tz + 1] if i_tz + 1 < len(z_positions) else Zp
			on_z_complete(acc_fine, wsum_fine, acc_coarse, wsum_coarse, next_tz, pad0)

	print("", flush=True)
	print(f"[predict3d] inference done in {time.time() - t0:.1f}s ({done - skipped_tiles} tiles)", flush=True)

	# If streaming mode (callback), output was consumed incrementally
	if on_z_complete is not None:
		del acc_fine, wsum_fine, acc_coarse, wsum_coarse
		return None

	# Legacy mode: normalize and return full memmap views
	acc_fine /= np.maximum(wsum_fine, 1e-7)
	acc_coarse /= np.maximum(wsum_coarse, 1e-7)
	del wsum_fine, wsum_coarse

	b_f = pad0 // sd_fine
	b_c = pad0 // sd_coarse
	nz_f, ny_f, nx_f = nz // sd_fine, ny // sd_fine, nx // sd_fine
	nz_c, ny_c, nx_c = nz // sd_coarse, ny // sd_coarse, nx // sd_coarse

	result_fine = acc_fine[:, b_f:b_f + nz_f, b_f:b_f + ny_f, b_f:b_f + nx_f]
	result_coarse = acc_coarse[:, b_c:b_c + nz_c, b_c:b_c + ny_c, b_c:b_c + nx_c]
	return result_fine, result_coarse


def _find_zarr_group_root(path: str) -> Path | None:
	"""Walk up from a zarr array path to find the group root (.zattrs or .zgroup).

	Handles trailing slashes, numeric level dirs, and nested structures
	like ``volpkg/volumes/vol.zarr/2``.
	"""
	p = Path(str(path).rstrip("/")).resolve()
	# Start from p itself, walk up until we find .zattrs or .zgroup
	check = p
	for _ in range(5):  # don't walk up forever
		if (check / ".zattrs").is_file() or (check / ".zgroup").is_file():
			return check
		if check.parent == check:
			break
		check = check.parent
	return None


def _download_one_path(
	zarr_path: str,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
) -> None:
	"""Download chunks for a single zarr path from its S3 source.

	Walks up from *zarr_path* to find the group root with ``_download``
	metadata.  Downloads only the level indicated by the path's trailing
	numeric component (e.g. ``vol.zarr/2`` → scale 2).
	"""
	import json as _json
	import sys as _sys
	_lasagna_dir = str(Path(__file__).resolve().parent)
	if _lasagna_dir not in _sys.path:
		_sys.path.insert(0, _lasagna_dir)
	from scripts.download_omezarr import download

	p = Path(str(zarr_path).rstrip("/")).resolve()

	# Find group root with _download metadata
	group_root = None
	dl_meta = None
	check = p
	for _ in range(5):
		zattrs_path = check / ".zattrs"
		if zattrs_path.is_file():
			zattrs = _json.loads(zattrs_path.read_text(encoding="utf-8"))
			if "_download" in zattrs:
				group_root = check
				dl_meta = zattrs["_download"]
				break
		if check.parent == check:
			break
		check = check.parent

	if group_root is None or dl_meta is None:
		raise ValueError(
			f"no _download metadata found walking up from {zarr_path} — "
			"run download_omezarr.py on this volume first "
			"(it records the S3 source), or pass --no-download to skip"
		)

	# Determine scale from path (e.g. vol.zarr/2 → scale 2)
	scales: list[int] | None = None
	if p.name.isdigit():
		scales = [int(p.name)]

	# Convert crop to bbox (base coords)
	bbox: tuple[int, int, int, int, int, int] | None = None
	if crop_xyzwhd is not None:
		x, y, z, w, h, d = crop_xyzwhd
		bbox = (x, y, z, x + w, y + h, z + d)

	source_uri = dl_meta["source"]
	anon = dl_meta.get("anon", False)
	region = dl_meta.get("region")

	print(f"[predict3d] downloading {source_uri} "
		  f"scales={scales or 'all'} dest={group_root} ...", flush=True)
	ret = download(
		source=source_uri,
		dest=str(group_root),
		scales=scales,
		bbox_xyzxyz=bbox,
		anon=anon,
		region=region,
	)
	if ret != 0:
		raise RuntimeError(f"download from {source_uri} failed (exit {ret})")


def _auto_download(
	input_path: str,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	pred_dt_path: str | None,
) -> None:
	"""Auto-download input and pred-dt data from S3.

	Each path is resolved independently to its own zarr group root —
	they may come from different S3 sources.
	"""
	_download_one_path(input_path, crop_xyzwhd)
	if pred_dt_path:
		_download_one_path(pred_dt_path, crop_xyzwhd)
	print("[predict3d] all downloads complete", flush=True)


def _resolve_base_shape(
	input_path: str,
	base_ref: str | None,
	base_scale: int | None,
) -> tuple[int, int, int] | None:
	"""Resolve base_shape_zyx from --base-ref/--base-scale or auto-detect.

	Three modes (checked in order):
	1. base_ref + base_scale: read zarr shape, multiply by 2^base_scale
	2. base_ref alone: zarr IS 1× base, use shape directly
	3. Neither: try to open the input as a zarr group and read level 0
	"""
	if base_ref is not None:
		ref = zarr.open(str(base_ref), mode="r")
		if hasattr(ref, "shape"):
			sh = tuple(int(v) for v in ref.shape)
			# Strip leading channel dim if present
			if len(sh) == 4:
				sh = sh[1:]
			if len(sh) != 3:
				raise ValueError(f"--base-ref array must be 3D or 4D (CZYX), got shape={sh}")
		else:
			raise ValueError(f"--base-ref must point to a zarr array, got group: {base_ref}")
		scale = base_scale if base_scale is not None else 0
		f = 2 ** scale
		return (sh[0] * f, sh[1] * f, sh[2] * f)

	# Auto-detect: find the finest available level in the input's zarr group
	# and scale up to base (level 0) resolution.
	try:
		inp = Path(str(input_path).rstrip("/"))
		# Walk up to find the zarr root (directory containing .zgroup or .zattrs)
		group_path = inp.parent if inp.name.isdigit() else inp
		grp = zarr.open_group(str(group_path), mode="r")
		# Find the finest (lowest-numbered) level
		level_keys = sorted(int(k) for k in grp.keys() if k.isdigit())
		if level_keys:
			finest_lv = level_keys[0]
			arr = grp[str(finest_lv)]
			sh = tuple(int(v) for v in arr.shape)
			if len(sh) == 3:
				# Scale up: level N shape × 2^N ≈ base shape
				f = 2 ** finest_lv
				base = (sh[0] * f, sh[1] * f, sh[2] * f)
				print(f"[predict3d] auto-detected base shape: level {finest_lv} "
					  f"shape={sh} × {f} → base={base}", flush=True)
				return base
	except Exception:
		pass
	return None


def run_preprocess_3d(
	*,
	input_path: str,
	output_path: str,
	unet3d_checkpoint: str,
	device: str | None,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	tile_size: int,
	overlap: int,
	border: int,
	cos_scaledown: int = 2,
	scaledown: int = 4,
	source_to_base: float = 1.0,
	pred_dt_path: str | None = None,
	chunk_z: int = 32,
	chunk_yx: int = 32,
	edt_chunk_depth: int = 448,
	edt_chunk_yx: int = 448,
	calibrate_norm: bool = False,
	base_ref: str | None = None,
	base_scale: int | None = None,
	n_levels: int = 5,
	ome_chunk: int = 32,
) -> None:
	"""Run 3D UNet inference and write .lasagna.json with OME-Zarr pyramids.

	Output: <output_dir>/<name>.lasagna.json + per-channel .ome.zarr groups.
	Streaming: tiles processed in z-order, completed z-bands flushed to
	OME-Zarr progressively.  Memmap pages released after each flush.

	cos channel is stored at cos_scaledown resolution.
	grad_mag, nx, ny are stored at scaledown resolution.
	pred_dt (if provided) is stored at cos_scaledown resolution with signed
	distance encoding: outside=[80,127], inside=[128,175], no data=0.
	"""
	from lasagna_volume import LasagnaVolume, ChannelGroup

	if not output_path.endswith(".lasagna.json"):
		raise ValueError(f"output must be .lasagna.json, got: {output_path}")

	a_in = zarr.open(str(input_path), mode="r")
	if not hasattr(a_in, "shape"):
		raise ValueError(f"input must point to a zarr array, got: {input_path}")
	sh = tuple(int(v) for v in a_in.shape)
	if len(sh) != 3:
		raise ValueError(f"input array must be (Z,Y,X), got shape {sh}")

	# Resolve base shape first (needed to scale crop from base to input coords)
	base_shape_zyx = _resolve_base_shape(input_path, base_ref, base_scale)
	if base_shape_zyx is None:
		raise ValueError(
			"cannot determine base_shape_zyx — required for OME-Zarr output. "
			"Pass --base-ref or ensure the input is inside an OME-Zarr group."
		)

	import math as _math
	input_sd = max(1, round(base_shape_zyx[0] / sh[0]))

	# crop_xyzwhd is in BASE coordinates — scale to input resolution
	crop_input: tuple[int, int, int, int, int, int] | None = None
	if crop_xyzwhd is not None:
		bx, by, bz, bw, bh, bd = (int(v) for v in crop_xyzwhd)
		crop_input = (
			bx // input_sd, by // input_sd, bz // input_sd,
			max(1, bw // input_sd), max(1, bh // input_sd), max(1, bd // input_sd),
		)

	z0, z1, y0, y1, x0, x1 = _crop_xyzwhd_bounds(shape_zyx=sh, crop_xyzwhd=crop_input)
	nz = z1 - z0
	ny = y1 - y0
	nx_dim = x1 - x0
	if nz <= 0 or ny <= 0 or nx_dim <= 0:
		raise ValueError(
			f"empty crop: x=[{x0},{x1}) y=[{y0},{y1}) z=[{z0},{z1}) in shape={sh}\n"
			f"  base crop={crop_xyzwhd} → input crop={crop_input} (input_sd={input_sd})"
		)

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	torch_device = torch.device(device)

	print(
		f"[predict3d] input={input_path} shape={sh} input_sd={input_sd}\n"
		f"  base_shape={base_shape_zyx} base_crop={crop_xyzwhd}\n"
		f"  input_crop=({x0},{y0},{z0},{nx_dim},{ny},{nz}) "
		f"cos_scaledown={cos_scaledown} scaledown={scaledown}",
		flush=True,
	)

	out_dir = os.path.dirname(os.path.abspath(output_path))
	os.makedirs(out_dir, exist_ok=True)

	# Derive zarr name prefix from JSON filename: "s5.lasagna.json" → "s5_"
	json_stem = os.path.basename(output_path).removesuffix(".lasagna.json")
	prefix = f"{json_stem}_" if json_stem else ""

	# Output dimensions for each resolution
	cos_sd = max(1, int(cos_scaledown))
	other_sd = max(1, int(scaledown))

	full_cos_z = _ds_size(sh[0], cos_sd)
	full_cos_y = _ds_size(sh[1], cos_sd)
	full_cos_x = _ds_size(sh[2], cos_sd)

	full_other_z = _ds_size(sh[0], other_sd)
	full_other_y = _ds_size(sh[1], other_sd)
	full_other_x = _ds_size(sh[2], other_sd)

	# Crop offsets at each output resolution (in input-relative coords)
	cos_oz0 = _ds_index(z0, cos_sd)
	cos_oy0 = _ds_index(y0, cos_sd)
	cos_ox0 = _ds_index(x0, cos_sd)
	cos_oz1 = min(full_cos_z, cos_oz0 + _ds_size(nz, cos_sd))
	cos_oy1 = min(full_cos_y, cos_oy0 + _ds_size(ny, cos_sd))
	cos_ox1 = min(full_cos_x, cos_ox0 + _ds_size(nx_dim, cos_sd))

	other_oz0 = _ds_index(z0, other_sd)
	other_oy0 = _ds_index(y0, other_sd)
	other_ox0 = _ds_index(x0, other_sd)
	other_oz1 = min(full_other_z, other_oz0 + _ds_size(nz, other_sd))
	other_oy1 = min(full_other_y, other_oy0 + _ds_size(ny, other_sd))
	other_ox1 = min(full_other_x, other_ox0 + _ds_size(nx_dim, other_sd))

	# Effective scaledowns from base and OME-Zarr level numbers
	effective_cos_sd = input_sd * cos_sd
	effective_other_sd = input_sd * other_sd
	cos_level = round(_math.log2(effective_cos_sd)) if effective_cos_sd > 1 else 0
	other_level = round(_math.log2(effective_other_sd)) if effective_other_sd > 1 else 0
	# Ensure enough levels
	n_levels = max(n_levels, cos_level + 2, other_level + 2)
	print(f"[predict3d] input_sd={input_sd} cos_level={cos_level} other_level={other_level} "
		  f"n_levels={n_levels}", flush=True)

	# Load or create the .lasagna.json manifest
	json_path = Path(output_path)
	if json_path.exists():
		vol = LasagnaVolume.load(json_path)
		print(f"[predict3d] loaded existing manifest: {output_path}", flush=True)
		vol.base_shape_zyx = base_shape_zyx
	else:
		vol = LasagnaVolume(
			path=json_path.resolve(),
			source_to_base=source_to_base,
			base_shape_zyx=base_shape_zyx,
		)
	# Record this crop in base coordinates (appends if new, deduplicates)
	if crop_xyzwhd is not None:
		vol.add_crop(tuple(int(v) for v in crop_xyzwhd))

	# Validate pred-dt source and determine its scale relative to base
	pred_dt_zarr = None
	pred_sd = 1       # pred zarr scaledown from base (1 = full res)
	pred_per_cos = effective_cos_sd  # pred voxels per cos output voxel
	if pred_dt_path:
		pred_dt_path = pred_dt_path.rstrip("/")
		pred_dt_zarr = zarr.open(str(pred_dt_path), mode="r")
		if not hasattr(pred_dt_zarr, "shape"):
			raise ValueError(f"pred-dt must point to a zarr array, got group: {pred_dt_path}")
		_pred_shape = tuple(int(v) for v in pred_dt_zarr.shape)
		if len(_pred_shape) != 3:
			raise ValueError(f"pred-dt array must be 3D (Z,Y,X), got shape {_pred_shape}")
		# pred zarr scaledown relative to base
		pred_sd = max(1, round(base_shape_zyx[0] / _pred_shape[0]))
		# How many pred voxels per cos output voxel
		pred_per_cos = max(1, effective_cos_sd // pred_sd)
		print(f"[predict3d] pred-dt={pred_dt_path} shape={_pred_shape} "
			  f"pred_sd={pred_sd} pred_per_cos={pred_per_cos}", flush=True)

	# --- Create OME-Zarr outputs ---
	oc = max(1, int(ome_chunk))
	cos_omezarr_path = os.path.join(out_dir, f"{prefix}cos.ome.zarr")
	gm_omezarr_path = os.path.join(out_dir, f"{prefix}grad_mag.ome.zarr")
	nx_omezarr_path = os.path.join(out_dir, f"{prefix}nx.ome.zarr")
	ny_omezarr_path = os.path.join(out_dir, f"{prefix}ny.ome.zarr")
	dt_omezarr_path = os.path.join(out_dir, f"{prefix}pred_dt.ome.zarr") if pred_dt_path else None

	cos_grp = _open_or_create_omezarr(cos_omezarr_path, base_shape_zyx, cos_level, n_levels, oc, "cos")
	gm_grp = _open_or_create_omezarr(gm_omezarr_path, base_shape_zyx, other_level, n_levels, oc, "grad_mag")
	nx_grp = _open_or_create_omezarr(nx_omezarr_path, base_shape_zyx, other_level, n_levels, oc, "nx")
	ny_grp = _open_or_create_omezarr(ny_omezarr_path, base_shape_zyx, other_level, n_levels, oc, "ny")
	dt_grp = _open_or_create_omezarr(dt_omezarr_path, base_shape_zyx, cos_level, n_levels, oc, "pred_dt") if dt_omezarr_path else None

	# Level arrays (3D) for writing
	cos_lv_arr = cos_grp[str(cos_level)]
	gm_lv_arr = gm_grp[str(other_level)]
	nx_lv_arr = nx_grp[str(other_level)]
	ny_lv_arr = ny_grp[str(other_level)]
	dt_lv_arr = dt_grp[str(cos_level)] if dt_grp else None

	# --- Resume logic ---
	resume_z_fine = _find_resume_z(cos_omezarr_path, cos_level)
	if resume_z_fine > cos_oz0:
		print(f"[predict3d] resume: cos OME-Zarr has data up to z={resume_z_fine}, "
			  f"will skip completed z-bands", flush=True)

	# How many z_positions to skip for resume?  Map resume_z_fine back to
	# padded accumulator space, then find the earliest tile z-row that
	# produces output beyond resume_z_fine.
	sd_fine = cos_sd
	sd_coarse = other_sd
	pad0 = max(0, int(border))
	stride = max(1, tile_size - overlap)

	resume_z_padded = (resume_z_fine - cos_oz0) * sd_fine + pad0  # approx
	# We need to re-run tiles whose output overlaps the resume boundary
	# (for correct blending).  Back up by one tile_size.
	safe_resume_z = max(0, resume_z_padded - tile_size)
	skip_z_positions = max(0, safe_resume_z // stride)

	# --- Pause training if running (free GPU) ---
	from gpu_pause import gpu_pause_context
	_gpu_ctx = gpu_pause_context()
	if _gpu_ctx is not None:
		_gpu_ctx.__enter__()

	# --- Build model ---
	model, _norm_type, _upsample_mode, _output_sigmoid = build_model_3d(
		tile_size, str(torch_device), weights=str(unet3d_checkpoint))
	model.eval()

	if calibrate_norm:
		_calibrate_instance_norm(
			model, a_in,
			crop_slices=(z0, z1, y0, y1, x0, x1),
			device=torch_device,
			tile_size=tile_size,
		)

	# --- Streaming flush callback ---
	# Captures OME-Zarr arrays, offsets, pred_dt state from enclosing scope.
	_prev_flush_fine = [0]   # accumulator z (fine res) already flushed
	_prev_flush_coarse = [0]
	_flush_t0 = [time.time()]
	sd_ratio = other_sd // cos_sd

	# Output region sizes
	cos_wz = cos_oz1 - cos_oz0
	cos_wy = cos_oy1 - cos_oy0
	cos_wx = cos_ox1 - cos_ox0
	other_wz = other_oz1 - other_oz0
	other_wy = other_oy1 - other_oy0
	other_wx = other_ox1 - other_ox0

	# Accumulator-to-output coordinate helpers
	b_f = pad0 // sd_fine   # fine accumulator index where output starts
	b_c = pad0 // sd_coarse
	out_end_f = b_f + cos_wz   # fine accumulator index where output ends
	out_end_c = b_c + other_wz

	def _on_z_complete(acc_fine, wsum_fine, acc_coarse, wsum_coarse,
					   complete_z_padded, pad0_inner):
		"""Flush completed z-bands to OME-Zarr, compute pred_dt, release pages."""
		# --- Flush fine (cos + pred_dt) ---
		complete_z_f = complete_z_padded // sd_fine
		flush_from_f = max(_prev_flush_fine[0], b_f)
		flush_to_f = min(complete_z_f, out_end_f)

		if flush_to_f > flush_from_f:
			# Skip bands that are already written (resume)
			out_zs = flush_from_f - b_f   # output z-start
			out_ze = flush_to_f - b_f     # output z-end
			# Only write bands beyond resume point
			effective_start = max(out_zs, resume_z_fine - cos_oz0)
			if out_ze > effective_start:
				# Normalize accumulator in-place for this band
				acc_band = acc_fine[:, flush_from_f:flush_to_f, :, :]
				ws_band = wsum_fine[:, flush_from_f:flush_to_f, :, :]
				acc_band /= np.maximum(ws_band, 1e-7)

				# Map to output region
				eff_from = max(flush_from_f, b_f + effective_start)
				eff_to = flush_to_f
				eff_out_zs = eff_from - b_f
				eff_out_ze = eff_to - b_f

				if eff_out_ze > eff_out_zs:
					local_from = eff_from - flush_from_f
					local_to = eff_to - flush_from_f
					# Trim to crop region (Y, X)
					yf = pad0_inner // sd_fine
					xf = pad0_inner // sd_fine
					cos_slab = np.ascontiguousarray(
						acc_band[0, local_from:local_to, yf:yf + cos_wy, xf:xf + cos_wx])
					cos_u8 = np.clip(cos_slab * 255.0, 0.0, 255.0).astype(np.uint8)
					oz = cos_oz0 + eff_out_zs
					cos_lv_arr[oz:oz + cos_u8.shape[0],
							   cos_oy0:cos_oy0 + cos_u8.shape[1],
							   cos_ox0:cos_ox0 + cos_u8.shape[2]] = cos_u8

					# --- pred_dt for this z-band ---
					if pred_dt_zarr is not None and dt_lv_arr is not None:
						# cos output z → base z → pred_src z
						# All coords go through base as common frame
						base_z0 = (cos_oz0 + eff_out_zs) * effective_cos_sd
						base_z1 = (cos_oz0 + eff_out_ze) * effective_cos_sd
						pdt_z0 = base_z0 // pred_sd
						pdt_z1 = base_z1 // pred_sd
						# Crop YX: source coords → base → pred_src
						if crop_xyzwhd is not None:
							pdt_y0 = y0 * input_sd // pred_sd
							pdt_y1 = (y0 + ny) * input_sd // pred_sd
							pdt_x0 = x0 * input_sd // pred_sd
							pdt_x1 = (x0 + nx_dim) * input_sd // pred_sd
						else:
							pdt_y0 = 0
							pdt_y1 = int(pred_dt_zarr.shape[1])
							pdt_x0 = 0
							pdt_x1 = int(pred_dt_zarr.shape[2])
						_compute_pred_dt_slab(
							pred_zarr=pred_dt_zarr,
							pred_path=pred_dt_path,
							output_level_arr=dt_lv_arr,
							output_omezarr_path=dt_omezarr_path,
							output_level_key=str(cos_level),
							pred_z0=pdt_z0, pred_z1=pdt_z1,
							pred_y0=pdt_y0, pred_y1=pdt_y1,
							pred_x0=pdt_x0, pred_x1=pdt_x1,
							out_z0=cos_oz0 + eff_out_zs,
							out_y0=cos_oy0, out_x0=cos_ox0,
							scaledown=pred_per_cos,
							ome_chunk=oc,
							progress=_progress,
						)

			# Release memmap pages
			_release_memmap_pages(acc_fine, flush_from_f, flush_to_f)
			_release_memmap_pages(wsum_fine, flush_from_f, flush_to_f)

		_prev_flush_fine[0] = max(_prev_flush_fine[0], complete_z_f)

		# --- Flush coarse (grad_mag, nx, ny) ---
		complete_z_c = complete_z_padded // sd_coarse
		flush_from_c = max(_prev_flush_coarse[0], b_c)
		flush_to_c = min(complete_z_c, out_end_c)

		if flush_to_c > flush_from_c:
			out_zs_c = flush_from_c - b_c
			out_ze_c = flush_to_c - b_c
			# Resume: check against coarse resume point
			resume_z_coarse = (resume_z_fine - cos_oz0) // sd_ratio if resume_z_fine > cos_oz0 else 0
			effective_start_c = max(out_zs_c, resume_z_coarse)
			if out_ze_c > effective_start_c:
				# Normalize
				acc_band_c = acc_coarse[:, flush_from_c:flush_to_c, :, :]
				ws_band_c = wsum_coarse[:, flush_from_c:flush_to_c, :, :]
				acc_band_c /= np.maximum(ws_band_c, 1e-7)

				eff_from_c = max(flush_from_c, b_c + effective_start_c)
				eff_to_c = flush_to_c
				eff_out_zs_c = eff_from_c - b_c
				eff_out_ze_c = eff_to_c - b_c

				if eff_out_ze_c > eff_out_zs_c:
					local_from_c = eff_from_c - flush_from_c
					local_to_c = eff_to_c - flush_from_c
					yc = pad0_inner // sd_coarse
					xc = pad0_inner // sd_coarse
					slab = np.ascontiguousarray(
						acc_band_c[:, local_from_c:local_to_c, yc:yc + other_wy, xc:xc + other_wx])

					# grad_mag
					gm_u8 = np.clip(slab[0] * 1000.0, 0.0, 255.0).astype(np.uint8)
					oz_c = other_oz0 + eff_out_zs_c
					gm_lv_arr[oz_c:oz_c + gm_u8.shape[0],
							  other_oy0:other_oy0 + gm_u8.shape[1],
							  other_ox0:other_ox0 + gm_u8.shape[2]] = gm_u8

					# Normals
					_, _, _, nx_n, ny_n, nz_n = _estimate_normal(
						slab[1], slab[2], slab[3], slab[4], slab[5], slab[6])
					flip = np.where(nz_n < 0, -1.0, 1.0)
					nx_u8 = np.clip(np.round(nx_n * flip * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)
					ny_u8 = np.clip(np.round(ny_n * flip * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)
					nx_lv_arr[oz_c:oz_c + nx_u8.shape[0],
							  other_oy0:other_oy0 + nx_u8.shape[1],
							  other_ox0:other_ox0 + nx_u8.shape[2]] = nx_u8
					ny_lv_arr[oz_c:oz_c + ny_u8.shape[0],
							  other_oy0:other_oy0 + ny_u8.shape[1],
							  other_ox0:other_ox0 + ny_u8.shape[2]] = ny_u8

			# Release memmap pages
			_release_memmap_pages(acc_coarse, flush_from_c, flush_to_c)
			_release_memmap_pages(wsum_coarse, flush_from_c, flush_to_c)

		_prev_flush_coarse[0] = max(_prev_flush_coarse[0], complete_z_c)

	# --- Streaming inference + flush ---
	# Shared progress tracker for unified ETA across tiles + EDT
	_progress = {
		"t0": time.time(),
		"tiles_done": 0,
		"tiles_total": 0,  # set by _infer_tiled_3d
		"edt_done": 0,
		"edt_total_est": 1,  # updated below
	}
	# Estimate total EDT chunks (for overall ETA weighting)
	if pred_dt_zarr is not None:
		_psh = tuple(int(v) for v in pred_dt_zarr.shape)
		import math as _m2
		_edt_nz = max(1, _m2.ceil((pred_z1_total := (cos_oz1 - cos_oz0) * effective_cos_sd // max(1, pred_sd)) / 448))
		_edt_ny = max(1, _m2.ceil((_psh[1]) / 448))
		_edt_nx = max(1, _m2.ceil((_psh[2]) / 448))
		_progress["edt_total_est"] = _edt_nz * _edt_ny * _edt_nx

	_infer_tiled_3d(
		model, a_in,
		crop_slices=(z0, z1, y0, y1, x0, x1),
		device=torch_device,
		tile_size=tile_size,
		overlap=overlap,
		border=border,
		cos_scaledown=cos_sd,
		other_scaledown=other_sd,
		tmp_dir=out_dir,
		output_sigmoid=_output_sigmoid,
		on_z_complete=_on_z_complete,
		skip_z_positions=skip_z_positions,
		progress=_progress,
	)
	del model
	torch.cuda.empty_cache()

	# --- Build pyramids ---
	print("[predict3d] building OME-Zarr pyramids ...", flush=True)
	for path, data_lv, name in [
		(cos_omezarr_path, cos_level, "cos"),
		(gm_omezarr_path, other_level, "grad_mag"),
		(nx_omezarr_path, other_level, "nx"),
		(ny_omezarr_path, other_level, "ny"),
	]:
		_build_omezarr_pyramid(path, data_lv, n_levels, oc)
	if dt_omezarr_path:
		_build_omezarr_pyramid(dt_omezarr_path, cos_level, n_levels, oc)

	# --- Update manifest ---
	vol.update_group("cos", ChannelGroup(
		zarr_path=f"{prefix}cos.ome.zarr/{cos_level}", scaledown=cos_sd, channels=["cos"]))
	vol.update_group("grad_mag", ChannelGroup(
		zarr_path=f"{prefix}grad_mag.ome.zarr/{other_level}", scaledown=other_sd, channels=["grad_mag"]))
	vol.update_group("nx", ChannelGroup(
		zarr_path=f"{prefix}nx.ome.zarr/{other_level}", scaledown=other_sd, channels=["nx"]))
	vol.update_group("ny", ChannelGroup(
		zarr_path=f"{prefix}ny.ome.zarr/{other_level}", scaledown=other_sd, channels=["ny"]))
	if pred_dt_path:
		vol.update_group("pred_dt", ChannelGroup(
			zarr_path=f"{prefix}pred_dt.ome.zarr/{cos_level}", scaledown=cos_sd, channels=["pred_dt"]))

	# --- Resume training ---
	if _gpu_ctx is not None:
		_gpu_ctx.__exit__(None, None, None)

	vol.save()
	print(f"[predict3d] done. manifest: {output_path}", flush=True)


_N_WORKERS = min(16, os.cpu_count() or 4)


def _make_fuse_tile_3axis():
	"""Create numba-jitted fuse kernel if numba is available."""
	if not _HAS_NUMBA:
		return None

	@numba.njit(cache=True)
	def _fuse_tile_3axis(z, y, x, out, eps):
		"""Fuse one spatial tile from 3 axis volumes.

		z, y, x: (5, nz, ny, nx) float32 — [cos, grad_mag, dir0, dir1, valid]
		out: (4, nz, ny, nx) uint8 — [cos, gm, nx_u8, ny_u8]
		"""
		nz = z.shape[1]
		ny = z.shape[2]
		nx = z.shape[3]
		inv255 = np.float32(1.0 / 255.0)
		sqrt2 = np.float32(np.sqrt(2.0))
		inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
		for zi in range(nz):
			for yi in range(ny):
				for xi in range(nx):
					# Read raw uint8-scale values
					z_cos = z[0, zi, yi, xi]
					z_gm = z[1, zi, yi, xi]
					z_d0 = z[2, zi, yi, xi] * inv255
					z_d1 = z[3, zi, yi, xi] * inv255

					y_cos = y[0, zi, yi, xi]
					y_gm = y[1, zi, yi, xi]
					y_d0 = y[2, zi, yi, xi] * inv255
					y_d1 = y[3, zi, yi, xi] * inv255

					x_cos = x[0, zi, yi, xi]
					x_gm = x[1, zi, yi, xi]
					x_d0 = x[2, zi, yi, xi] * inv255
					x_d1 = x[3, zi, yi, xi] * inv255

					# Decode dir angles: θ = 0.5 * arctan2(sin2t, cos2t)
					cos2t_z = np.float32(2.0) * z_d0 - np.float32(1.0)
					sin2t_z = cos2t_z - sqrt2 * (np.float32(2.0) * z_d1 - np.float32(1.0))
					theta_z = np.arctan2(sin2t_z, cos2t_z) * np.float32(0.5)

					cos2t_y = np.float32(2.0) * y_d0 - np.float32(1.0)
					sin2t_y = cos2t_y - sqrt2 * (np.float32(2.0) * y_d1 - np.float32(1.0))
					theta_y = np.arctan2(sin2t_y, cos2t_y) * np.float32(0.5)

					cos2t_x = np.float32(2.0) * x_d0 - np.float32(1.0)
					sin2t_x = cos2t_x - sqrt2 * (np.float32(2.0) * x_d1 - np.float32(1.0))
					theta_x = np.arctan2(sin2t_x, cos2t_x) * np.float32(0.5)

					sz = np.sin(theta_z)
					cz = np.cos(theta_z)
					sy = np.sin(theta_y)
					cy = np.cos(theta_y)
					sx = np.sin(theta_x)
					cx = np.cos(theta_x)

					# Cross products (candidate normals)
					n1_x = cz * cy
					n1_y = sz * cy
					n1_z = cz * sy

					n2_x = cz * cx
					n2_y = sz * cx
					n2_z = sz * sx

					n3_x = cy * sx
					n3_y = sy * cx
					n3_z = sy * sx

					# Align signs
					dot2 = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z
					if dot2 < np.float32(0.0):
						n2_x = -n2_x
						n2_y = -n2_y
						n2_z = -n2_z

					dot3 = n1_x * n3_x + n1_y * n3_y + n1_z * n3_z
					if dot3 < np.float32(0.0):
						n3_x = -n3_x
						n3_y = -n3_y
						n3_z = -n3_z

					# --- Pass 1: Score candidates against observations ---
					# Inline encode_dir: (a,b) -> d0=0.5+0.5*c2, d1=0.5+0.5*(c2-s2)*inv_sqrt2
					# where c2=(a²-b²)/(a²+b²+eps), s2=2ab/(a²+b²+eps)
					total_err1 = np.float32(0.0)
					total_err2 = np.float32(0.0)
					total_err3 = np.float32(0.0)

					# Score all 3 candidates against z-axis obs (nx, ny)
					for ci in range(3):
						if ci == 0:
							ca = n1_x; cb = n1_y; cc = n1_z
						elif ci == 1:
							ca = n2_x; cb = n2_y; cc = n2_z
						else:
							ca = n3_x; cb = n3_y; cc = n3_z

						# z-axis: encode(ca, cb)
						r2 = ca * ca + cb * cb + eps
						c2 = (ca * ca - cb * cb) / r2
						s2 = np.float32(2.0) * ca * cb / r2
						pz0 = np.float32(0.5) + np.float32(0.5) * c2
						pz1 = np.float32(0.5) + np.float32(0.5) * (c2 - s2) * inv_sqrt2
						ez = (pz0 - z_d0) ** 2 + (pz1 - z_d1) ** 2
						wz_c = ca * ca + cb * cb

						# y-axis: encode(ca, cc)
						r2 = ca * ca + cc * cc + eps
						c2 = (ca * ca - cc * cc) / r2
						s2 = np.float32(2.0) * ca * cc / r2
						py0 = np.float32(0.5) + np.float32(0.5) * c2
						py1 = np.float32(0.5) + np.float32(0.5) * (c2 - s2) * inv_sqrt2
						ey = (py0 - y_d0) ** 2 + (py1 - y_d1) ** 2
						wy_c = ca * ca + cc * cc

						# x-axis: encode(cb, cc)
						r2 = cb * cb + cc * cc + eps
						c2 = (cb * cb - cc * cc) / r2
						s2 = np.float32(2.0) * cb * cc / r2
						px0 = np.float32(0.5) + np.float32(0.5) * c2
						px1 = np.float32(0.5) + np.float32(0.5) * (c2 - s2) * inv_sqrt2
						ex = (px0 - x_d0) ** 2 + (px1 - x_d1) ** 2
						wx_c = cb * cb + cc * cc

						te = wz_c * ez + wy_c * ey + wx_c * ex
						if ci == 0:
							total_err1 = te
						elif ci == 1:
							total_err2 = te
						else:
							total_err3 = te

					sc1 = np.float32(1.0) / (total_err1 + eps)
					sc2 = np.float32(1.0) / (total_err2 + eps)
					sc3 = np.float32(1.0) / (total_err3 + eps)

					est_x = sc1 * n1_x + sc2 * n2_x + sc3 * n3_x
					est_y = sc1 * n1_y + sc2 * n2_y + sc3 * n3_y
					est_z = sc1 * n1_z + sc2 * n2_z + sc3 * n3_z
					norm_e = np.sqrt(est_x * est_x + est_y * est_y + est_z * est_z) + eps
					est_x = est_x / norm_e
					est_y = est_y / norm_e
					est_z = est_z / norm_e

					# --- Pass 2: Re-weight constraint rows ---
					wz2 = np.sqrt(est_x * est_x + est_y * est_y + eps)
					wy2 = np.sqrt(est_x * est_x + est_z * est_z + eps)
					wx2 = np.sqrt(est_y * est_y + est_z * est_z + eps)

					wzy = wz2 * wy2
					wzx = wz2 * wx2
					wyx = wy2 * wx2

					rn1_x = wzy * n1_x; rn1_y = wzy * n1_y; rn1_z = wzy * n1_z
					rn2_x = wzx * n2_x; rn2_y = wzx * n2_y; rn2_z = wzx * n2_z
					rn3_x = wyx * n3_x; rn3_y = wyx * n3_y; rn3_z = wyx * n3_z

					dot2r = rn1_x * rn2_x + rn1_y * rn2_y + rn1_z * rn2_z
					if dot2r < np.float32(0.0):
						rn2_x = -rn2_x; rn2_y = -rn2_y; rn2_z = -rn2_z
					dot3r = rn1_x * rn3_x + rn1_y * rn3_y + rn1_z * rn3_z
					if dot3r < np.float32(0.0):
						rn3_x = -rn3_x; rn3_y = -rn3_y; rn3_z = -rn3_z

					nnx = rn1_x + rn2_x + rn3_x
					nny = rn1_y + rn2_y + rn3_y
					nnz = rn1_z + rn2_z + rn3_z
					norm = np.sqrt(nnx * nnx + nny * nny + nnz * nnz) + eps

					nnx_n = nnx / norm
					nny_n = nny / norm
					nnz_n = nnz / norm

					# In-plane projection weights
					w_z = np.sqrt(nnx_n * nnx_n + nny_n * nny_n + eps)
					w_y = np.sqrt(nnx_n * nnx_n + nnz_n * nnz_n + eps)
					w_x = np.sqrt(nny_n * nny_n + nnz_n * nnz_n + eps)

					# Weighted fusion
					w_sum = w_z + w_y + w_x + eps
					cos_fused = (w_z * z_cos + w_y * y_cos + w_x * x_cos) / w_sum
					gm_fused = (z_gm + y_gm + x_gm) / w_sum

					# Flip to +z hemisphere
					if nnz_n < np.float32(0.0):
						nnx_n = -nnx_n
						nny_n = -nny_n

					# Encode normal as uint8
					nx_val = nnx_n * np.float32(127.0) + np.float32(128.5)
					ny_val = nny_n * np.float32(127.0) + np.float32(128.5)

					out[0, zi, yi, xi] = np.uint8(min(np.float32(255.0), max(np.float32(0.0), cos_fused)))
					out[1, zi, yi, xi] = np.uint8(min(np.float32(255.0), max(np.float32(0.0), gm_fused)))
					out[2, zi, yi, xi] = np.uint8(min(np.float32(255.0), max(np.float32(0.0), nx_val)))
					out[3, zi, yi, xi] = np.uint8(min(np.float32(255.0), max(np.float32(0.0), ny_val)))

	return _fuse_tile_3axis


_fuse_tile_3axis = _make_fuse_tile_3axis()


def _fuse_tile_3axis_numpy(z, y, x, out, eps):
	"""Numpy fallback for _fuse_tile_3axis when numba is unavailable."""
	dir0_z = z[2] / 255.0
	dir1_z = z[3] / 255.0
	dir0_y = y[2] / 255.0
	dir1_y = y[3] / 255.0
	dir0_x = x[2] / 255.0
	dir1_x = x[3] / 255.0

	w_z, w_y, w_x, nx_n, ny_n, nz_n = _estimate_normal(
		dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x, eps=eps)

	w_sum = w_z + w_y + w_x + eps
	cos_fused = (w_z * z[0] + w_y * y[0] + w_x * x[0]) / w_sum
	gm_fused = (z[1] + y[1] + x[1]) / w_sum

	# Flip to +z hemisphere
	flip = np.where(nz_n < 0, -1.0, 1.0)
	nx_n = nx_n * flip
	ny_n = ny_n * flip

	out[0] = np.clip(cos_fused, 0, 255).astype(np.uint8)
	out[1] = np.clip(gm_fused, 0, 255).astype(np.uint8)
	out[2] = np.clip(np.round(nx_n * 127.0 + 128.0), 0, 255).astype(np.uint8)
	out[3] = np.clip(np.round(ny_n * 127.0 + 128.0), 0, 255).astype(np.uint8)



def _warmup_fuse_kernel():
	"""Trigger numba JIT compilation with a tiny dummy tile."""
	if _fuse_tile_3axis is None:
		return
	dummy = np.zeros((5, 1, 1, 1), dtype=np.float32)
	out = np.zeros((4, 1, 1, 1), dtype=np.uint8)
	_fuse_tile_3axis(dummy, dummy, dummy, out, np.float32(1e-7))


def _run_integrate_tile_parallel(
	*, z_vol, y_vol, x_vol, out,
	n_out_ch, out_chunks, eps,
	zi_lo, zi_hi, yi_lo, yi_hi, xi_lo, xi_hi,
):
	"""Tile-parallel path: numba fused kernel releases GIL, threads run in parallel."""
	_, cz_out, cy_out, cx_out = out_chunks
	# Align iteration to zarr chunk boundaries so each tile falls within
	# exactly one chunk per axis — prevents concurrent write races.
	z_base = (zi_lo // cz_out) * cz_out
	y_base = (yi_lo // cy_out) * cy_out
	x_base = (xi_lo // cx_out) * cx_out
	tiles = []
	for zs_chunk in range(z_base, zi_hi, cz_out):
		zs = max(zi_lo, zs_chunk)
		ze = min(zi_hi, zs_chunk + cz_out)
		for ys_chunk in range(y_base, yi_hi, cy_out):
			ys = max(yi_lo, ys_chunk)
			ye = min(yi_hi, ys_chunk + cy_out)
			for xs_chunk in range(x_base, xi_hi, cx_out):
				xs = max(xi_lo, xs_chunk)
				xe = min(xi_hi, xs_chunk + cx_out)
				tiles.append((zs, ze, ys, ye, xs, xe))

	n_tiles = len(tiles)
	print(f"[integrate_directions] {n_tiles} tiles, {_N_WORKERS} workers, compute=numba")

	print("[integrate_directions] warming up numba kernel...", end="", flush=True)
	t_warmup = time.time()
	_warmup_fuse_kernel()
	print(f" {time.time() - t_warmup:.1f}s", flush=True)

	lock = threading.Lock()
	done_count = [0]
	t_read_sum = [0.0]
	t_compute_sum = [0.0]
	t_write_sum = [0.0]

	def process_tile(coords):
		zs, ze, ys, ye, xs, xe = coords
		t_r0 = time.time()
		z_chunk = np.asarray(z_vol[:5, zs:ze, ys:ye, xs:xe]).astype(np.float32)
		y_chunk = np.asarray(y_vol[:5, zs:ze, ys:ye, xs:xe]).astype(np.float32)
		x_chunk = np.asarray(x_vol[:5, zs:ze, ys:ye, xs:xe]).astype(np.float32)
		dt_read = time.time() - t_r0

		t_c0 = time.time()
		out_tile = np.empty((4, ze - zs, ye - ys, xe - xs), dtype=np.uint8)
		_fuse_tile_3axis(z_chunk, y_chunk, x_chunk, out_tile, eps)
		dt_compute = time.time() - t_c0

		t_w0 = time.time()
		for ch in range(4):
			out[ch, zs:ze, ys:ye, xs:xe] = out_tile[ch]
		dt_write = time.time() - t_w0

		with lock:
			t_read_sum[0] += dt_read
			t_compute_sum[0] += dt_compute
			t_write_sum[0] += dt_write
			done_count[0] += 1

	t0 = time.time()
	with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
		futures = [pool.submit(process_tile, t) for t in tiles]
		for f in as_completed(futures):
			f.result()
			with lock:
				dc = done_count[0]
			elapsed = max(1e-6, time.time() - t0)
			per = elapsed / max(1, dc)
			eta = per * max(0, n_tiles - dc)
			bar_w = 30
			fill = int(round(float(dc) / float(max(1, n_tiles)) * bar_w))
			bar = "#" * fill + "-" * (bar_w - fill)
			print(
				f"\r[integrate] [{bar}] {dc}/{n_tiles} "
				f"({100.0 * dc / max(1, n_tiles):.1f}%) "
				f"eta {int(eta // 60):02d}:{int(eta % 60):02d} "
				f"avg={1000.0 * elapsed / max(1, dc):.2f}ms/tile",
				end="", flush=True,
			)

	print("", flush=True)
	total = time.time() - t0
	avg_ms = 1000.0 * total / max(1, n_tiles)
	avg_read = 1000.0 * t_read_sum[0] / max(1, n_tiles)
	avg_compute = 1000.0 * t_compute_sum[0] / max(1, n_tiles)
	avg_write = 1000.0 * t_write_sum[0] / max(1, n_tiles)
	print(f"[integrate_directions] done in {total:.1f}s "
		  f"({n_tiles} tiles, avg {avg_ms:.2f}ms/tile: "
		  f"read={avg_read:.2f}ms compute={avg_compute:.2f}ms write={avg_write:.2f}ms)")


def _run_integrate_slab(
	*, z_vol, y_vol, x_vol, out,
	n_out_ch, z_chunks, eps,
	zi_lo, zi_hi, yi_lo, yi_hi, xi_lo, xi_hi,
	crop_z_count,
):
	"""Slab-based path: read/compute full z-slabs with numpy, pipelined I/O."""
	src_z_chunk = z_chunks[1]
	batch_size = src_z_chunk

	vols_to_read = [z_vol, y_vol, x_vol]

	ys, ye, xs, xe = yi_lo, yi_hi, xi_lo, xi_hi
	io_pool = ThreadPoolExecutor(max_workers=len(vols_to_read) + 1)

	def _do_read(vol, s, e):
		return np.asarray(vol[:5, s:e, ys:ye, xs:xe]).astype(np.float32)

	def _submit_reads(zi_s, zi_e):
		return [io_pool.submit(_do_read, vol, zi_s, zi_e) for vol in vols_to_read]

	def _do_write(out_arr, ch_data, zi_s, zi_e):
		for ch, data in ch_data:
			out_arr[ch, zi_s:zi_e, ys:ye, xs:xe] = data

	# Align z-batches to chunk boundaries to avoid write races
	z_base = (zi_lo // batch_size) * batch_size
	batches = [(max(zi_lo, zi_s), min(zi_hi, zi_s + batch_size))
			   for zi_s in range(z_base, zi_hi, batch_size)]
	n_batches_total = len(batches)
	print(f"[integrate_directions] {n_batches_total} z-slabs, batch_size={batch_size}, "
		  f"compute=numpy (slab), pipeline read||compute||write")

	t0 = time.time()
	t_read_wait = 0.0
	t_compute_total = 0.0
	t_write_wait = 0.0
	done_batches = 0

	read_futs = _submit_reads(*batches[0]) if batches else []
	write_fut = None

	for bi, (zi_start, zi_end) in enumerate(batches):
		t_rw0 = time.time()
		read_results = [f.result() for f in read_futs]
		t_read_wait += time.time() - t_rw0

		z_batch, y_batch, x_batch = read_results

		if bi + 1 < len(batches):
			read_futs = _submit_reads(*batches[bi + 1])

		if write_fut is not None:
			t_ww0 = time.time()
			write_fut.result()
			t_write_wait += time.time() - t_ww0

		t_c0 = time.time()
		ch_data = []

		dir0_z = z_batch[2] / 255.0
		dir1_z = z_batch[3] / 255.0
		dir0_y = y_batch[2] / 255.0
		dir1_y = y_batch[3] / 255.0
		dir0_x = x_batch[2] / 255.0
		dir1_x = x_batch[3] / 255.0
		w_z, w_y, w_x, nx_n, ny_n, nz_n = _estimate_normal(
			dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x)
		w_sum = w_z + w_y + w_x + eps
		cos_fused = (w_z * z_batch[0] + w_y * y_batch[0] + w_x * x_batch[0]) / w_sum
		gm_fused = (z_batch[1] + y_batch[1] + x_batch[1]) / w_sum

		flip = np.where(nz_n < 0, -1.0, 1.0)
		nx_enc = nx_n * flip
		ny_enc = ny_n * flip
		ch_data.append((0, np.clip(cos_fused, 0, 255).astype(np.uint8)))
		ch_data.append((1, np.clip(gm_fused, 0, 255).astype(np.uint8)))
		ch_data.append((2, np.clip(np.round(nx_enc * 127.0 + 128.0), 0, 255).astype(np.uint8)))
		ch_data.append((3, np.clip(np.round(ny_enc * 127.0 + 128.0), 0, 255).astype(np.uint8)))

		t_compute_total += time.time() - t_c0

		write_fut = io_pool.submit(_do_write, out, ch_data, zi_start, zi_end)

		done_batches += 1
		done = zi_end - zi_lo
		elapsed = max(1e-6, time.time() - t0)
		per = elapsed / max(1, done)
		eta = per * max(0, crop_z_count - done)
		bar_w = 30
		fill = int(round(float(done) / float(max(1, crop_z_count)) * bar_w))
		bar = "#" * fill + "-" * (bar_w - fill)
		nb = max(1, done_batches)
		print(
			f"\r[integrate] [{bar}] {done}/{crop_z_count} "
			f"({100.0 * done / max(1, crop_z_count):.1f}%) "
			f"eta {int(eta // 60):02d}:{int(eta % 60):02d} "
			f"read={t_read_wait / nb:.2f}s "
			f"compute={t_compute_total / nb:.2f}s "
			f"write={t_write_wait / nb:.2f}s",
			end="", flush=True,
		)

	if write_fut is not None:
		t_ww0 = time.time()
		write_fut.result()
		t_write_wait += time.time() - t_ww0

	io_pool.shutdown(wait=False)

	print("", flush=True)
	total = time.time() - t0
	print(f"[integrate_directions] done in {total:.1f}s "
		  f"({n_batches_total} slabs: "
		  f"read={t_read_wait:.1f}s compute={t_compute_total:.1f}s write={t_write_wait:.1f}s)")


def run_integrate_directions(
	*,
	z_volume_path: str,
	y_volume_path: str,
	x_volume_path: str,
	output_path: str,
	batch_size: int = 32,
	pred_dt_path: str | None = None,
) -> None:
	"""Fuse cos/grad_mag and estimate 3D normal from three axis volumes (z, y, x).

	All three axis volumes are required and must be preprocessed with the same
	uniform scaledown. The z-volume shape is the reference.

	The estimated normal is stored as hemisphere-encoded (nx, ny) uint8 pair.
	nz is reconstructed as sqrt(1 - nx² - ny²) >= 0 by convention.

	grad_mag == 0 marks invalid voxels (no separate valid channel).

	Output channels: [cos, grad_mag, nx, ny] (pred_dt appended if given)
	"""
	z_vol = zarr.open(str(z_volume_path), mode="r")
	z_shape = tuple(int(v) for v in z_vol.shape)
	if len(z_shape) != 4 or z_shape[0] != 5:
		raise ValueError(f"z-volume must have shape (5, Z, Y, X), got {z_shape}")
	_, ref_z, ref_y, ref_x = z_shape

	z_params = dict(z_vol.attrs.get("preprocess_params", {}))
	scaledown = int(z_params.get("scaledown", 1))

	crop_param = z_params.get("crop_xyzwhd", None)
	if crop_param is not None:
		cx, cy, cz, cw, ch, cd = (int(v) for v in crop_param)
		zi_lo = max(0, min(cz // scaledown, ref_z))
		zi_hi = max(zi_lo, min((cz + cd + scaledown - 1) // scaledown, ref_z))
		yi_lo = max(0, min(cy // scaledown, ref_y))
		yi_hi = max(yi_lo, min((cy + ch + scaledown - 1) // scaledown, ref_y))
		xi_lo = max(0, min(cx // scaledown, ref_x))
		xi_hi = max(xi_lo, min((cx + cw + scaledown - 1) // scaledown, ref_x))
	else:
		zi_lo, zi_hi = 0, ref_z
		yi_lo, yi_hi = 0, ref_y
		xi_lo, xi_hi = 0, ref_x
	crop_z_count = zi_hi - zi_lo

	y_vol = zarr.open(str(y_volume_path), mode="r")
	x_vol = zarr.open(str(x_volume_path), mode="r")

	channel_names: list[str] = ["cos", "grad_mag", "nx", "ny"]

	# Validate pred-dt early, before any heavy processing
	if pred_dt_path:
		pred_dt_path = pred_dt_path.rstrip("/")
		_pred_check = zarr.open(str(pred_dt_path), mode="r")
		if not hasattr(_pred_check, "shape"):
			raise ValueError(f"pred-dt must point to a zarr array, got group: {pred_dt_path}")
		_pred_shape = tuple(int(v) for v in _pred_check.shape)
		if len(_pred_shape) != 3:
			raise ValueError(f"pred-dt array must be 3D (Z,Y,X), got shape {_pred_shape}")
		print(f"[integrate_directions] pred-dt={pred_dt_path} shape={_pred_shape}", flush=True)
		del _pred_check, _pred_shape
		channel_names.append("pred_dt")

	n_out_ch = len(channel_names)
	z_chunks = tuple(int(v) for v in z_vol.chunks)
	out_chunks = (1,) + z_chunks[1:]

	out = zarr.open(
		str(output_path),
		mode="w",
		shape=(n_out_ch, ref_z, ref_y, ref_x),
		chunks=out_chunks,
		dtype=np.uint8,
		fill_value=0,
		zarr_format=2,
	)
	out_params = dict(z_params)
	out_params["channels"] = channel_names
	out.attrs["preprocess_params"] = out_params

	print(f"[integrate_directions] z_volume={z_volume_path} shape={z_shape} scaledown={scaledown}")
	print(f"[integrate_directions] y_volume shape={tuple(int(v) for v in y_vol.shape)}")
	print(f"[integrate_directions] x_volume shape={tuple(int(v) for v in x_vol.shape)}")
	print(f"[integrate_directions] fusion=3-axis normal-weighted")
	print(f"[integrate_directions] -> {output_path} shape=({n_out_ch}, {ref_z}, {ref_y}, {ref_x})")
	if crop_param is not None:
		print(f"[integrate_directions] crop z=[{zi_lo},{zi_hi}) y=[{yi_lo},{yi_hi}) x=[{xi_lo},{xi_hi}) "
			  f"of ({ref_z},{ref_y},{ref_x}) => {crop_z_count}x{yi_hi-yi_lo}x{xi_hi-xi_lo} slices")
	else:
		print(f"[integrate_directions] no crop — processing full {ref_z}x{ref_y}x{ref_x}")

	eps = np.float32(1e-7)

	# Choose processing strategy: tile-parallel (numba releases GIL) vs slab-based (numpy)
	use_numba = _fuse_tile_3axis is not None

	if use_numba:
		_run_integrate_tile_parallel(
			z_vol=z_vol, y_vol=y_vol, x_vol=x_vol, out=out,
			n_out_ch=n_out_ch, out_chunks=out_chunks, eps=eps,
			zi_lo=zi_lo, zi_hi=zi_hi, yi_lo=yi_lo, yi_hi=yi_hi,
			xi_lo=xi_lo, xi_hi=xi_hi,
		)
	else:
		_run_integrate_slab(
			z_vol=z_vol, y_vol=y_vol, x_vol=x_vol, out=out,
			n_out_ch=n_out_ch, z_chunks=z_chunks, eps=eps,
			zi_lo=zi_lo, zi_hi=zi_hi, yi_lo=yi_lo, yi_hi=yi_hi,
			xi_lo=xi_lo, xi_hi=xi_hi,
			crop_z_count=crop_z_count,
		)

	if pred_dt_path:
		pred_dt_ch = channel_names.index("pred_dt")
		crop_param = z_params.get("crop_xyzwhd", None)
		_compute_pred_dt_channel(
			pred_path=pred_dt_path,
			output_arr=out,
			channel_idx=pred_dt_ch,
			ref_z=ref_z, ref_y=ref_y, ref_x=ref_x,
			scaledown=scaledown,
			crop_xyzwhd=crop_param,
		)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
		pass

	p = argparse.ArgumentParser(
		description="Run tiled 2D UNet inference on an OME-Zarr volume (per-axis slicing).",
		epilog=(
			"─── integrate mode ───────────────────────────────────────────────\n"
			"Fuse 3-axis 2D results into cos/grad_mag/nx/ny.\n"
			"  preprocess_cos_omezarr.py integrate [options]\n"
			"\n"
			"  --z-volume PATH       Axis-z preprocessed zarr (reference shape). Required.\n"
			"  --y-volume PATH       Axis-y preprocessed zarr. Required.\n"
			"  --x-volume PATH       Axis-x preprocessed zarr. Required.\n"
			"  --output PATH         Output zarr path. Required.\n"
			"  --batch-size N        Z-slices per batch for resize (default: 32).\n"
			"  --pred-dt PATH        Surface prediction zarr for distance-to-skeleton channel.\n"
			"\n"
			"─── predict3d mode ───────────────────────────────────────────────\n"
			"3D UNet single-pass inference → cos/grad_mag/nx/ny zarr.\n"
			"Uses CUDA by default when available.\n"
			"  preprocess_cos_omezarr.py predict3d [options]\n"
			"\n"
			"  --input PATH          Input zarr array (3D ZYX). Required.\n"
			"  --output PATH         Output zarr path. Required.\n"
			"  --unet-checkpoint P   3D UNet checkpoint (.pt). Required.\n"
			"  --tile-size N         Tile cube size (default: 256).\n"
			"  --overlap N           Tile overlap in voxels (default: 64).\n"
			"  --border N            Hard discard border at tile edges (default: 16).\n"
			"  --scaledown N         Output downsample factor (default: 4).\n"
			"  --crop X Y Z W H D   Crop region in absolute input coordinates.\n"
			"  --pred-dt PATH        Prediction zarr for distance-to-surface channel.\n"
			"  --device DEV          Device, e.g. cuda or cpu (default: cuda if available).\n"
			"  --chunk-z N           Output zarr chunk size along Z (default: 32).\n"
			"  --chunk-yx N          Output zarr chunk size for Y and X (default: 32).\n"
			"  --edt-chunk-depth N   EDT chunk depth in Z (default: 256).\n"
			"  --edt-chunk-yx N      EDT chunk size in Y/X (default: 256)."
		),
		formatter_class=_Fmt,
	)
	p.add_argument("--input", required=True, help="Input OME-Zarr array path (must be Z,Y,X array).")
	p.add_argument("--output", required=True, help="Output OME-Zarr group path.")
	p.add_argument("--unet-checkpoint", required=True, help="UNet checkpoint path (.pt).")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu" (default: cuda if available).')
	p.add_argument("--axis", choices=["z", "y", "x"], default="z",
		help="Dimension to slice along.")
	p.add_argument("--crop", "--crop-xyzwhd", dest="crop_xyzwhd", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"), help="Crop in absolute input coordinates: x y z w h d.")
	p.add_argument("--tile-size", type=int, default=2048, help="Tile size.")
	p.add_argument("--overlap", type=int, default=128, help="Tile overlap.")
	p.add_argument("--border", type=int, default=32, help="Tile border discard width.")
	p.add_argument("--scaledown", type=int, default=4,
		help="Uniform downscale factor for all three dimensions.")
	p.add_argument("--chunk-z", "--chunk-slice", dest="chunk_z", type=int, default=32,
		help="Output chunk size along the slice axis.")
	p.add_argument("--chunk-yx", "--chunk-plane", dest="chunk_yx", type=int, default=32,
		help="Output chunk size for the plane axes.")
	p.add_argument("--measure-cuda-timings", action="store_true", default=False,
		help="Insert cuda.synchronize() calls to measure per-step timings accurately (slower).")
	args = p.parse_args(argv)

	run_preprocess(
		input_path=str(args.input),
		output_path=str(args.output),
		unet_checkpoint=str(args.unet_checkpoint),
		device=args.device,
		crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd is not None else None,
		axis=str(args.axis),
		tile_size=int(args.tile_size),
		overlap=int(args.overlap),
		border=int(args.border),
		scaledown=int(args.scaledown),
		chunk_z=int(args.chunk_z),
		chunk_yx=int(args.chunk_yx),
		measure_cuda_timings=bool(args.measure_cuda_timings),
	)
	return 0


def main_integrate(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Integrate direction channels from axis-y / axis-x preprocessed volumes into the axis-z reference volume.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--z-volume", required=True, help="Axis-z preprocessed zarr (reference shape).")
	p.add_argument("--y-volume", required=True, help="Axis-y preprocessed zarr.")
	p.add_argument("--x-volume", required=True, help="Axis-x preprocessed zarr.")
	p.add_argument("--output", required=True, help="Output zarr path.")
	p.add_argument("--batch-size", type=int, default=32, help="Z-slices per batch for resize.")
	p.add_argument("--pred-dt", default=None, help="Surface prediction zarr for distance-to-skeleton channel.")
	args = p.parse_args(argv)

	run_integrate_directions(
		z_volume_path=str(args.z_volume),
		y_volume_path=str(args.y_volume),
		x_volume_path=str(args.x_volume),
		output_path=str(args.output),
		batch_size=int(args.batch_size),
		pred_dt_path=str(args.pred_dt) if args.pred_dt else None,
	)
	return 0


def main_predict3d(argv: list[str] | None = None) -> int:
	# Make this process the OOM killer's first target so the parent session survives
	try:
		with open("/proc/self/oom_score_adj", "w") as f:
			f.write("1000")
	except (OSError, PermissionError):
		pass

	p = argparse.ArgumentParser(
		description="Run 3D UNet inference and write .lasagna.json with per-group zarrs (cos, prediction, pred_dt).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--input", required=True, help="Input zarr array (3D ZYX).")
	p.add_argument("--output", required=True, help="Output .lasagna.json path.")
	p.add_argument("--unet-checkpoint", required=True, help="3D UNet checkpoint (.pt).")
	p.add_argument("--tile-size", type=int, default=256,
		help="Tile size, must be compatible with model architecture.")
	p.add_argument("--overlap", type=int, default=64, help="Tile overlap in voxels.")
	p.add_argument("--border", type=int, default=16, help="Hard discard border at tile edges.")
	p.add_argument("--cos-scaledown", type=int, default=2, help="Downsample factor for cos channel.")
	p.add_argument("--scaledown", type=int, default=4, help="Downsample factor for other channels.")
	p.add_argument("--source-to-base", type=float, default=1.0,
		help="Source volume to base (VC3D) coordinate factor.")
	p.add_argument("--crop", "--crop-xyzwhd", dest="crop_xyzwhd", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"), help="Crop region: x y z w h d.")
	p.add_argument("--pred-dt", default=None, help="Prediction zarr for distance-to-surface channel.")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu" (default: cuda if available).')
	p.add_argument("--chunk-z", type=int, default=32, help="Output zarr chunk size along Z.")
	p.add_argument("--chunk-yx", type=int, default=32, help="Output zarr chunk size for Y and X.")
	p.add_argument("--edt-chunk-depth", type=int, default=256, help="EDT chunk depth in Z (default 256).")
	p.add_argument("--edt-chunk-yx", type=int, default=256, help="EDT chunk size in Y/X (default 256).")
	p.add_argument("--calibrate-norm", action="store_true", default=False,
		help="Calibrate InstanceNorm running stats before inference for tile consistency.")
	p.add_argument("--base-ref", default=None,
		help="Reference zarr for base shape. If given with --base-scale N, "
			 "base = ref_shape * 2^N. If given alone, ref IS base (1x). "
			 "If omitted, auto-detect from input zarr group level 0.")
	p.add_argument("--base-scale", type=int, default=None,
		help="How many 2x downsamples the --base-ref zarr is from the true base.")
	p.add_argument("--no-download", action="store_true", default=False,
		help="Skip automatic S3 download of input data. By default, predict3d "
			 "checks for _download metadata in the zarr's .zattrs and downloads "
			 "needed chunks before inference.")
	p.add_argument("--levels", type=int, default=5,
		help="Number of OME-Zarr pyramid levels to generate (default 5).")
	p.add_argument("--ome-chunk", type=int, default=32,
		help="Chunk size for OME-Zarr output levels (default 128).")
	args = p.parse_args(argv)

	if not args.no_download:
		_auto_download(
			input_path=str(args.input),
			crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd else None,
			pred_dt_path=str(args.pred_dt) if args.pred_dt else None,
		)

	run_preprocess_3d(
		input_path=str(args.input),
		output_path=str(args.output),
		unet3d_checkpoint=str(args.unet_checkpoint),
		device=args.device,
		crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd else None,
		tile_size=int(args.tile_size),
		overlap=int(args.overlap),
		border=int(args.border),
		cos_scaledown=int(args.cos_scaledown),
		scaledown=int(args.scaledown),
		source_to_base=float(args.source_to_base),
		pred_dt_path=str(args.pred_dt) if args.pred_dt else None,
		chunk_z=int(args.chunk_z),
		chunk_yx=int(args.chunk_yx),
		edt_chunk_depth=int(args.edt_chunk_depth),
		edt_chunk_yx=int(args.edt_chunk_yx),
		calibrate_norm=bool(args.calibrate_norm),
		base_ref=args.base_ref,
		base_scale=args.base_scale,
		n_levels=int(args.levels),
		ome_chunk=int(args.ome_chunk),
	)
	return 0


if __name__ == "__main__":
	import sys
	if "--help" in sys.argv or "-h" in sys.argv:
		raise SystemExit(main(["--help"]))
	if len(sys.argv) > 1 and sys.argv[1] == "integrate":
		raise SystemExit(main_integrate(sys.argv[2:]))
	if len(sys.argv) > 1 and sys.argv[1] == "predict3d":
		raise SystemExit(main_predict3d(sys.argv[2:]))
	raise SystemExit(main())
