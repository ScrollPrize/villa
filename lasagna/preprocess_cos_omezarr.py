from __future__ import annotations

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
	"""Compute distance-to-surface channel from a prediction zarr and write into output_arr."""
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

				# Binarize and compute distance transform
				binary = chunk_np > 0
				del chunk_np
				if _HAS_CUPY:
					cp.get_default_memory_pool().free_all_blocks()
					mempool = cp.get_default_memory_pool()
					nvoxels = binary.size
					# PBA 3D needs ~24 bytes/voxel (bool input + int64 encoded + int64 working + float64 output)
					est_bytes = nvoxels * 24
					gpu_free = cp.cuda.Device().mem_info[0]
					print(
						f"[pred_dt] chunk {chunk_i}/{n_chunks}  distance_transform_edt (GPU) "
						f"voxels={nvoxels:,} est={est_bytes / 2**30:.1f}GiB "
						f"free={gpu_free / 2**30:.1f}GiB pool_used={mempool.used_bytes() / 2**30:.2f}GiB ...",
						end="", flush=True,
					)
					t_edt = time.time()
					try:
						binary_gpu = cp.asarray(binary)
						dt_gpu = cnd.distance_transform_edt(~binary_gpu)
						dt = cp.asnumpy(dt_gpu).astype(np.float32)
						del binary_gpu, dt_gpu
						cp.get_default_memory_pool().free_all_blocks()
					except cp.cuda.memory.OutOfMemoryError as e:
						cp.get_default_memory_pool().free_all_blocks()
						gpu_free2 = cp.cuda.Device().mem_info[0]
						gpu_total = cp.cuda.Device().mem_info[1]
						print(flush=True)
						print(
							f"[pred_dt] GPU OOM!\n"
							f"  chunk shape     : {binary.shape}\n"
							f"  voxels          : {nvoxels:,}\n"
							f"  est. GPU need   : {est_bytes / 2**30:.1f} GiB\n"
							f"  GPU free/total  : {gpu_free2 / 2**30:.1f} / {gpu_total / 2**30:.1f} GiB\n"
							f"  pool used       : {mempool.used_bytes() / 2**30:.2f} GiB\n"
							f"  overlap={overlap} \u2192 max padded side = chunk + 2*overlap\n"
							f"  Try reducing --edt-chunk-depth / --edt-chunk-yx or overlap.",
							flush=True,
						)
						raise
				elif _HAS_EDT:
					print(f"[pred_dt] chunk {chunk_i}/{n_chunks}  distance_transform_edt (CPU) ...", end="", flush=True)
					t_edt = time.time()
					dt = edt_mod.edt(~binary, parallel=32).astype(np.float32)
				else:
					raise ImportError("pred-dt requires either CuPy (GPU) or the 'edt' package (CPU). Install with: pip install edt")
				print(f" {time.time() - t_edt:.1f}s", flush=True)

				# Crop off overlap padding to keep center region
				pad_z = z_pos - read_z0
				pad_y = y_pos - read_y0
				pad_x = x_pos - read_x0
				center_z = z_chunk_end - z_pos
				center_y = y_chunk_end - y_pos
				center_x = x_chunk_end - x_pos
				dt_center = dt[pad_z:pad_z + center_z, pad_y:pad_y + center_y, pad_x:pad_x + center_x]

				# Clamp to 255, cast to uint8
				dt_u8 = np.clip(dt_center, 0.0, 255.0).astype(np.uint8)

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
	scaledown: int = 1,
) -> np.ndarray:
	"""Run 3D UNet inference on a volume using overlapping tiles with linear blending.

	Lazy zarr reads per tile, memmap accumulator on disk, sigmoid + optional
	downscale applied per tile on GPU.

	Args:
		model: 3D UNet model in eval mode.
		zarr_arr: zarr array (Z, Y, X), read lazily per tile.
		crop_slices: (z0, z1, y0, y1, x0, x1) crop region in zarr coords.
		device: torch device for inference.
		tile_size: cube tile edge length.
		overlap: tile overlap in voxels.
		border: hard-discard border at tile edges.
		out_channels: number of model output channels.
		scaledown: spatial downscale factor applied per tile (must divide
			tile_size, stride, and border evenly).

	Returns:
		(out_channels, Z_out, Y_out, X_out) float32 numpy array of
		sigmoid-activated (and optionally downscaled) predictions.
	"""
	z0, z1, y0, y1, x0, x1 = crop_slices
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	volume_shape = tuple(int(v) for v in zarr_arr.shape)

	sd = max(1, int(scaledown))
	stride = max(1, tile_size - overlap)

	# Validate alignment
	if sd > 1:
		for name, val in [("tile_size", tile_size), ("stride", stride), ("border", border)]:
			if val % sd != 0:
				raise ValueError(f"{name}={val} must be divisible by scaledown={sd}")

	# Padded crop dimensions (border on each side)
	pad0 = max(0, int(border))
	Zp = nz + 2 * pad0
	Yp = ny + 2 * pad0
	Xp = nx + 2 * pad0

	# Round up to scaledown-multiple
	if sd > 1:
		Zp = ((Zp + sd - 1) // sd) * sd
		Yp = ((Yp + sd - 1) // sd) * sd
		Xp = ((Xp + sd - 1) // sd) * sd

	# Output dimensions
	Zo = Zp // sd
	Yo = Yp // sd
	Xo = Xp // sd

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

	# Precompute downscaled weight (or full-res weight) as numpy for accumulation
	if sd > 1:
		w_lr = F.avg_pool3d(
			w_full.unsqueeze(0).unsqueeze(0), sd, sd
		).squeeze(0).squeeze(0).cpu().numpy()  # (tile//sd, tile//sd, tile//sd)
	else:
		w_lr = w_full.cpu().numpy()  # (tile, tile, tile)

	# Memmap accumulators
	acc_file = tempfile.NamedTemporaryFile(suffix=".acc", delete=False)
	wsum_file = tempfile.NamedTemporaryFile(suffix=".wsum", delete=False)
	acc_path = acc_file.name
	wsum_path = wsum_file.name
	acc_file.close()
	wsum_file.close()

	acc_shape = (out_channels, Zo, Yo, Xo)
	wsum_shape = (1, Zo, Yo, Xo)
	print(
		f"[predict3d] memmap accumulator shape={acc_shape} "
		f"({np.prod(acc_shape) * 4 / (1024**3):.2f} GiB) + wsum "
		f"({np.prod(wsum_shape) * 4 / (1024**3):.2f} GiB)",
		flush=True,
	)
	acc = np.memmap(acc_path, dtype=np.float32, mode="w+", shape=acc_shape)
	wsum = np.memmap(wsum_path, dtype=np.float32, mode="w+", shape=wsum_shape)

	# Delete temp files immediately — memmap keeps them open, OS cleans up after
	try:
		os.unlink(acc_path)
		os.unlink(wsum_path)
	except OSError:
		pass

	total_tiles = len(z_positions) * len(y_positions) * len(x_positions)
	done = 0
	t0 = time.time()
	crop_offset = (z0, y0, x0)

	for tz in z_positions:
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

				with torch.inference_mode():
					pred = model(tile_t)
				# Model returns dict {"output": tensor} or plain tensor
				if isinstance(pred, dict):
					pred = pred["output"]

				# Diagnostics: check for NaN at each stage
				raw_nan = torch.isnan(pred).sum().item()
				if raw_nan > 0 or done == 0:
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

				# Sigmoid on GPU
				pred = torch.sigmoid(pred.float())  # (1, C, tz, ty, tx)

				# Apply blend weight on GPU
				pred_w = pred[0] * w_full  # (C, tz, ty, tx)

				# Downscale on GPU if needed
				if sd > 1:
					pred_w = F.avg_pool3d(pred_w.unsqueeze(0), sd, sd)[0]  # (C, tz/sd, ty/sd, tx/sd)

				pred_np = pred_w.cpu().numpy()  # (C, tz_out, ty_out, tx_out)

				# Accumulator coords in output space
				azl = tz // sd
				ayl = ty // sd
				axl = tx // sd
				ts_out = tile_size // sd
				azr = min(azl + ts_out, Zo)
				ayr = min(ayl + ts_out, Yo)
				axr = min(axl + ts_out, Xo)
				pzo = azr - azl
				pyo = ayr - ayl
				pxo = axr - axl

				acc[:, azl:azr, ayl:ayr, axl:axr] += pred_np[:, :pzo, :pyo, :pxo]
				wsum[0, azl:azr, ayl:ayr, axl:axr] += w_lr[:pzo, :pyo, :pxo]

				done += 1
				elapsed = max(1e-6, time.time() - t0)
				per = elapsed / done
				eta = max(0.0, per * (total_tiles - done))
				bar_w = 30
				fill = int(round(done / max(1, total_tiles) * bar_w))
				bar = "#" * fill + "-" * (bar_w - fill)
				print(
					f"\r[predict3d] [{bar}] {done}/{total_tiles} tiles "
					f"({100.0 * done / max(1, total_tiles):.1f}%) "
					f"eta {int(eta // 60):02d}:{int(eta % 60):02d} "
					f"avg={1000.0 * per:.0f}ms/tile",
					end="", flush=True,
				)

	print("", flush=True)
	print(f"[predict3d] inference done in {time.time() - t0:.1f}s ({total_tiles} tiles)", flush=True)

	# Normalize
	acc /= np.maximum(wsum, 1e-7)

	# Trim padding (border // sd on each side) back to crop-sized output
	b_out = pad0 // sd
	nz_out = nz // sd if sd > 1 else nz
	ny_out = ny // sd if sd > 1 else ny
	nx_out = nx // sd if sd > 1 else nx
	result = np.array(acc[:, b_out:b_out + nz_out, b_out:b_out + ny_out, b_out:b_out + nx_out])

	# Clean up memmaps
	del acc, wsum

	return result


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
	scaledown: int,
	pred_dt_path: str | None = None,
	chunk_z: int = 32,
	chunk_yx: int = 32,
	edt_chunk_depth: int = 448,
	edt_chunk_yx: int = 448,
) -> None:
	"""Run 3D UNet inference on a volume and write preprocessed zarr.

	Output format matches fit_data.load_3d(): channels [cos, grad_mag, nx, ny]
	(+ optional pred_dt) as uint8 with preprocess_params metadata.

	Memory-efficient: reads tiles lazily from zarr, accumulates in a
	disk-backed memmap, and post-processes in Z-chunks.
	"""
	a_in = zarr.open(str(input_path), mode="r")
	if not hasattr(a_in, "shape"):
		raise ValueError(f"input must point to a zarr array, got: {input_path}")
	sh = tuple(int(v) for v in a_in.shape)
	if len(sh) != 3:
		raise ValueError(f"input array must be (Z,Y,X), got shape {sh}")

	z0, z1, y0, y1, x0, x1 = _crop_xyzwhd_bounds(shape_zyx=sh, crop_xyzwhd=crop_xyzwhd)
	nz = z1 - z0
	ny = y1 - y0
	nx_dim = x1 - x0
	if nz <= 0 or ny <= 0 or nx_dim <= 0:
		raise ValueError(f"empty crop: x=[{x0},{x1}) y=[{y0},{y1}) z=[{z0},{z1}) in shape={sh}")
	if scaledown <= 0:
		raise ValueError("scaledown must be >= 1")

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	torch_device = torch.device(device)

	print(
		f"[predict3d] input={input_path} shape={sh} "
		f"crop=({x0},{y0},{z0},{nx_dim},{ny},{nz}) scaledown={scaledown}",
		flush=True,
	)

	# Output dimensions
	full_out_z = _ds_size(sh[0], scaledown)
	full_out_y = _ds_size(sh[1], scaledown)
	full_out_x = _ds_size(sh[2], scaledown)

	oz0 = _ds_index(z0, scaledown)
	oy0 = _ds_index(y0, scaledown)
	ox0 = _ds_index(x0, scaledown)
	out_nz = _ds_size(nz, scaledown)
	out_ny = _ds_size(ny, scaledown)
	out_nx = _ds_size(nx_dim, scaledown)
	oz1 = min(full_out_z, oz0 + out_nz)
	oy1 = min(full_out_y, oy0 + out_ny)
	ox1 = min(full_out_x, ox0 + out_nx)
	wz = oz1 - oz0
	wy = oy1 - oy0
	wx = ox1 - ox0

	channel_names: list[str] = ["cos", "grad_mag", "nx", "ny"]

	# Validate pred-dt
	if pred_dt_path:
		pred_dt_path = pred_dt_path.rstrip("/")
		_pred_check = zarr.open(str(pred_dt_path), mode="r")
		if not hasattr(_pred_check, "shape"):
			raise ValueError(f"pred-dt must point to a zarr array, got group: {pred_dt_path}")
		_pred_shape = tuple(int(v) for v in _pred_check.shape)
		if len(_pred_shape) != 3:
			raise ValueError(f"pred-dt array must be 3D (Z,Y,X), got shape {_pred_shape}")
		print(f"[predict3d] pred-dt={pred_dt_path} shape={_pred_shape}", flush=True)
		del _pred_check, _pred_shape
		channel_names.append("pred_dt")

	n_ch = len(channel_names)
	chunk_sizes = (
		1,
		min(full_out_z, max(1, chunk_z)),
		min(full_out_y, max(1, chunk_yx)),
		min(full_out_x, max(1, chunk_yx)),
	)

	arr = zarr.open(
		str(output_path),
		mode="w",
		shape=(n_ch, full_out_z, full_out_y, full_out_x),
		chunks=chunk_sizes,
		dtype=np.uint8,
		fill_value=0,
		zarr_format=2,
	)
	arr.attrs["preprocess_params"] = {
		"scaledown": int(scaledown),
		"grad_mag_encode_scale": 1000.0,
		"channels": channel_names,
		"crop_xyzwhd": [int(x0), int(y0), int(z0), int(nx_dim), int(ny), int(nz)],
		"output_full_scaled": True,
		"source": "predict3d",
	}

	print(
		f"[predict3d] writing {output_path} shape=({n_ch},{full_out_z},{full_out_y},{full_out_x}) "
		f"crop=[{oz0}:{oz1},{oy0}:{oy1},{ox0}:{ox1}]",
		flush=True,
	)

	# --- Phase 1: pred_dt (runs first so GPU OOM fails fast) ---
	if pred_dt_path:
		pred_dt_ch = channel_names.index("pred_dt")
		_compute_pred_dt_channel(
			pred_path=pred_dt_path,
			output_arr=arr,
			channel_idx=pred_dt_ch,
			ref_z=full_out_z, ref_y=full_out_y, ref_x=full_out_x,
			scaledown=scaledown,
			crop_xyzwhd=[int(x0), int(y0), int(z0), int(nx_dim), int(ny), int(nz)]
				if crop_xyzwhd is not None else None,
			chunk_depth=edt_chunk_depth,
			chunk_yx=edt_chunk_yx,
		)

	# --- Phase 2: tiled 3D inference ---
	model = build_model_3d(tile_size, str(torch_device), weights=str(unet3d_checkpoint))
	model.eval()

	pred = _infer_tiled_3d(
		model, a_in,
		crop_slices=(z0, z1, y0, y1, x0, x1),
		device=torch_device,
		tile_size=tile_size,
		overlap=overlap,
		border=border,
		scaledown=scaledown,
	)  # (8, out_nz, out_ny, out_nx) float32, sigmoid already applied
	del model
	torch.cuda.empty_cache()

	# Diagnostic: verify inference produced meaningful data
	print(
		f"[predict3d] inference result shape={pred.shape} "
		f"min={pred.min():.4f} max={pred.max():.4f} mean={pred.mean():.4f}",
		flush=True,
	)
	for ci in range(min(pred.shape[0], 8)):
		ch = pred[ci]
		print(
			f"  ch{ci}: min={ch.min():.4f} max={ch.max():.4f} "
			f"mean={ch.mean():.4f} nonzero={np.count_nonzero(ch)}/{ch.size}",
			flush=True,
		)

	# --- Phase 3: post-process and write channels 0–3 ---
	post_chunk_z = max(1, chunk_z)
	for zs in range(0, wz, post_chunk_z):
		ze = min(wz, zs + post_chunk_z)
		# Channels 0-1: cos, grad_mag — simple uint8 encode
		cos_slab = pred[0, zs:ze, :wy, :wx]
		gm_slab = pred[1, zs:ze, :wy, :wx]
		arr[0, oz0 + zs:oz0 + ze, oy0:oy1, ox0:ox1] = np.clip(
			cos_slab * 255.0, 0.0, 255.0).astype(np.uint8)
		arr[1, oz0 + zs:oz0 + ze, oy0:oy1, ox0:ox1] = np.clip(
			gm_slab * 1000.0, 0.0, 255.0).astype(np.uint8)

		# Channels 2-7: dir channels -> normal estimation -> nx, ny uint8
		d0z = pred[2, zs:ze, :wy, :wx]
		d1z = pred[3, zs:ze, :wy, :wx]
		d0y = pred[4, zs:ze, :wy, :wx]
		d1y = pred[5, zs:ze, :wy, :wx]
		d0x = pred[6, zs:ze, :wy, :wx]
		d1x = pred[7, zs:ze, :wy, :wx]
		_, _, _, nx_n, ny_n, nz_n = _estimate_normal(d0z, d1z, d0y, d1y, d0x, d1x)
		# Flip to +z hemisphere
		flip = np.where(nz_n < 0, -1.0, 1.0)
		nx_n = nx_n * flip
		ny_n = ny_n * flip
		arr[2, oz0 + zs:oz0 + ze, oy0:oy1, ox0:ox1] = np.clip(
			np.round(nx_n * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)
		arr[3, oz0 + zs:oz0 + ze, oy0:oy1, ox0:ox1] = np.clip(
			np.round(ny_n * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)

	del pred

	print("[predict3d] done.", flush=True)


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
			"  --chunk-yx N          Output zarr chunk size for Y and X (default: 32)."
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
		description="Run 3D UNet inference on a volume and write preprocessed zarr with cos/grad_mag/nx/ny channels.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--input", required=True, help="Input zarr array (3D ZYX).")
	p.add_argument("--output", required=True, help="Output zarr path.")
	p.add_argument("--unet-checkpoint", required=True, help="3D UNet checkpoint (.pt).")
	p.add_argument("--tile-size", type=int, default=256,
		help="Tile size, must be compatible with model architecture.")
	p.add_argument("--overlap", type=int, default=64, help="Tile overlap in voxels.")
	p.add_argument("--border", type=int, default=16, help="Hard discard border at tile edges.")
	p.add_argument("--scaledown", type=int, default=4, help="Output downsample factor.")
	p.add_argument("--crop", "--crop-xyzwhd", dest="crop_xyzwhd", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"), help="Crop region: x y z w h d.")
	p.add_argument("--pred-dt", default=None, help="Prediction zarr for distance-to-surface channel.")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu" (default: cuda if available).')
	p.add_argument("--chunk-z", type=int, default=32, help="Output zarr chunk size along Z.")
	p.add_argument("--chunk-yx", type=int, default=32, help="Output zarr chunk size for Y and X.")
	p.add_argument("--edt-chunk-depth", type=int, default=256, help="EDT chunk depth in Z (default 256).")
	p.add_argument("--edt-chunk-yx", type=int, default=256, help="EDT chunk size in Y/X (default 256).")
	args = p.parse_args(argv)

	run_preprocess_3d(
		input_path=str(args.input),
		output_path=str(args.output),
		unet3d_checkpoint=str(args.unet_checkpoint),
		device=args.device,
		crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd else None,
		tile_size=int(args.tile_size),
		overlap=int(args.overlap),
		border=int(args.border),
		scaledown=int(args.scaledown),
		pred_dt_path=str(args.pred_dt) if args.pred_dt else None,
		chunk_z=int(args.chunk_z),
		chunk_yx=int(args.chunk_yx),
		edt_chunk_depth=int(args.edt_chunk_depth),
		edt_chunk_yx=int(args.edt_chunk_yx),
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
