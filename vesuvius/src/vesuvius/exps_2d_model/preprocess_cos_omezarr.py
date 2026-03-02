from __future__ import annotations

import argparse
from pathlib import Path
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
import torch
import zarr

from common import load_unet, unet_infer_tiled
from fit_data import _gaussian_blur_nchw


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


def run_preprocess(
	*,
	input_path: str,
	output_path: str,
	unet_checkpoint: str,
	device: str | None,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	axis: str = "z",
	z_step: int,
	tile_size: int,
	overlap: int,
	border: int,
	downscale_xy: int,
	grad_mag_blur_sigma: float,
	dir_blur_sigma: float,
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

	if downscale_xy <= 0:
		raise ValueError("downscale must be >= 1")

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
	step_raw = max(1, int(z_step))
	step_eff = step_raw * max(1, int(downscale_xy))

	slice_sel = list(range(int(slice_start), int(slice_end), int(step_eff)))
	if len(slice_sel) <= 0:
		raise ValueError(
			f"empty {dim_names[slice_dim]} selection after downscale: "
			f"{dim_names[slice_dim]}=[{slice_start},{slice_end}) step={step_eff}"
		)
	proc_count = len(slice_sel)

	# Output sizes in ZYX order
	out_sizes = [0, 0, 0]
	out_sizes[slice_dim] = _ds_size(full_sizes[slice_dim], step_eff)
	out_sizes[plane_dim0] = _ds_size(full_sizes[plane_dim0], downscale_xy)
	out_sizes[plane_dim1] = _ds_size(full_sizes[plane_dim1], downscale_xy)
	out_z, out_y, out_x = out_sizes

	# Output offsets in ZYX order
	out_offsets = [0, 0, 0]
	out_offsets[slice_dim] = _ds_index(crop_ranges[slice_dim][0], step_eff)
	out_offsets[plane_dim0] = _ds_index(crop_ranges[plane_dim0][0], downscale_xy)
	out_offsets[plane_dim1] = _ds_index(crop_ranges[plane_dim1][0], downscale_xy)

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
		dimension_separator="/",
	)
	arr.attrs["preprocess_params"] = {
		"axis": axis,
		"downscale_xy": int(downscale_xy),
		"z_step": int(step_raw),
		"z_step_eff": int(step_eff),
		"grad_mag_blur_sigma": float(grad_mag_blur_sigma),
		"grad_mag_encode_scale": float(1000.0),
		"dir_blur_sigma": float(dir_blur_sigma),
		"processed_z_slices": int(proc_count),
		"crop_xyzwhd": [int(x0), int(y0), int(z0), int(nx), int(ny), int(nz)],
		"output_full_scaled": True,
		"channels": ["cos", "grad_mag", "dir0", "dir1", "valid"],
	}

	ax_name = dim_names[slice_dim]
	print(
		f"[preprocess_cos_omezarr] input={input_path} axis={axis} crop_xyzwhd=({x0},{y0},{z0},{nx},{ny},{nz}) "
		f"step={step_raw} step_eff={step_eff} proc_slices={proc_count} out_shape_full={(out_z, out_y, out_x)} in_shape={sh} "
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
		s_keep = [ss for ss in range(slo, shi) if ((ss - int(slice_start)) % int(step_eff)) == 0]
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
			if downscale_xy > 1:
				cos_np = _pyrdown2d(cos_np, factor=int(downscale_xy))
				grad_mag_np = _pyrdown2d(grad_mag_np, factor=int(downscale_xy))
				dir0_np = _pyrdown2d(dir0_np, factor=int(downscale_xy))
				dir1_np = _pyrdown2d(dir1_np, factor=int(downscale_xy))

			grad_mag = torch.from_numpy(grad_mag_np).to(device=torch_device, dtype=torch.float32)[None, None, :, :]
			dir0 = torch.from_numpy(dir0_np).to(device=torch_device, dtype=torch.float32)[None, None, :, :]
			dir1 = torch.from_numpy(dir1_np).to(device=torch_device, dtype=torch.float32)[None, None, :, :]
			if float(grad_mag_blur_sigma) > 0.0:
				grad_mag = _gaussian_blur_nchw(x=grad_mag, sigma=float(grad_mag_blur_sigma))
			if float(dir_blur_sigma) > 0.0:
				dir0 = _gaussian_blur_nchw(x=dir0, sigma=float(dir_blur_sigma))
				dir1 = _gaussian_blur_nchw(x=dir1, sigma=float(dir_blur_sigma))
			cos_u8 = np.clip(cos_np * 255.0, 0.0, 255.0).astype(np.uint8)
			grad_mag_u8 = np.clip(grad_mag[0, 0].detach().cpu().numpy().astype(np.float32) * 1000.0, 0.0, 255.0).astype(np.uint8)
			dir0_u8 = np.clip(dir0[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
			dir1_u8 = np.clip(dir1[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
			t_wr0 = time.time()

			# Output index along slice dimension
			oi = int(ss) // int(step_eff)
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
	downscale_xy: int,
	z_step_eff: int,
	crop_xyzwhd: list[int] | None,
	chunk_depth: int = 1000,
	chunk_yx: int = 1024,
	overlap: int = 255,
) -> None:
	"""Compute distance-to-surface channel from a prediction zarr and write into output_arr."""
	pred = zarr.open(str(pred_path), mode="r")
	if not hasattr(pred, "shape"):
		# OME-Zarr group — use highest-resolution array (level 0)
		if hasattr(pred, "__getitem__") and "0" in pred:
			pred = pred["0"]
		else:
			raise ValueError(f"pred-dt must point to a zarr array or OME-Zarr group, got: {pred_path}")
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

	# Round up chunk_depth to a multiple of z_step_eff so z sampling phase
	# stays aligned across chunk boundaries
	if z_step_eff > 1:
		chunk_depth = ((chunk_depth + z_step_eff - 1) // z_step_eff) * z_step_eff

	# Build chunk grid for all 3 axes
	z_starts = list(range(p_z0, p_z1, chunk_depth))
	y_starts = list(range(p_y0, p_y1, chunk_yx))
	x_starts = list(range(p_x0, p_x1, chunk_yx))
	n_chunks = len(z_starts) * len(y_starts) * len(x_starts)

	print(
		f"[pred_dt] pred={pred_path} shape={pred_shape} crop_z=[{p_z0},{p_z1}) "
		f"crop_y=[{p_y0},{p_y1}) crop_x=[{p_x0},{p_x1}) "
		f"downscale_xy={downscale_xy} z_step_eff={z_step_eff} "
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
				if _HAS_CUPY:
					print(f"[pred_dt] chunk {chunk_i}/{n_chunks}  distance_transform_edt (GPU) ...", end="", flush=True)
					t_edt = time.time()
					binary_gpu = cp.asarray(binary)
					dt_gpu = cnd.distance_transform_edt(~binary_gpu)
					dt = cp.asnumpy(dt_gpu).astype(np.float32)
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
				# Z: subsample at z_step_eff spacing
				z_indices_full = list(range(z_pos, z_chunk_end, z_step_eff))
				if not z_indices_full:
					continue

				for zf in z_indices_full:
					local_z = zf - z_pos
					if local_z < 0 or local_z >= dt_u8.shape[0]:
						continue
					slc = dt_u8[local_z]

					# YX downscale
					if downscale_xy > 1:
						out_h = max(1, slc.shape[0] // downscale_xy)
						out_w = max(1, slc.shape[1] // downscale_xy)
						slc = cv2.resize(slc, (out_w, out_h), interpolation=cv2.INTER_AREA)

					# Output indices
					out_zi = zf // z_step_eff
					if out_zi < 0 or out_zi >= ref_z:
						continue
					out_y0 = y_pos // max(1, downscale_xy)
					out_x0 = x_pos // max(1, downscale_xy)
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


def run_integrate_directions(
	*,
	z_volume_path: str,
	y_volume_path: str | None = None,
	x_volume_path: str | None = None,
	output_path: str,
	batch_size: int = 32,
	pred_dt_path: str | None = None,
) -> None:
	"""Integrate dir channels from axis-y and axis-x volumes into the axis-z reference volume.

	The z-volume shape is the reference. For each source (y/x) volume:
	  - Z indices are mapped via ratio = z_step_eff / source_downscale (= raw step).
	  - The 2D YX plane is resized with bilinear interpolation to match z-volume dims.
	  - Only the sparse dimension actually changes size; the other is already identical.

	Output channels: [cos, grad_mag, dir0_z, dir1_z, valid, dir0_y, dir1_y, dir0_x, dir1_x]
	(y/x pairs only present if the corresponding volume is provided)
	"""
	import torch.nn.functional as F

	z_vol = zarr.open(str(z_volume_path), mode="r")
	z_shape = tuple(int(v) for v in z_vol.shape)
	if len(z_shape) != 4 or z_shape[0] != 5:
		raise ValueError(f"z-volume must have shape (5, Z, Y, X), got {z_shape}")
	_, ref_z, ref_y, ref_x = z_shape

	z_params = dict(z_vol.attrs.get("preprocess_params", {}))
	z_step_eff = int(z_params.get("z_step_eff", 1))

	# (axis_label, zarr handle, z-index ratio)
	sources: list[tuple[str, zarr.Array, float]] = []
	channel_names: list[str] = ["cos", "grad_mag", "dir0", "dir1", "valid"]

	if y_volume_path:
		y_vol = zarr.open(str(y_volume_path), mode="r")
		y_params = dict(y_vol.attrs.get("preprocess_params", {}))
		# Y-volume's Z dim uses downscale_xy (Z is a plane dim for axis=y)
		y_ds = int(y_params.get("downscale_xy", 1))
		sources.append(("y", y_vol, float(z_step_eff) / float(max(1, y_ds))))
		channel_names.extend(["dir0_y", "dir1_y"])

	if x_volume_path:
		x_vol = zarr.open(str(x_volume_path), mode="r")
		x_params = dict(x_vol.attrs.get("preprocess_params", {}))
		x_ds = int(x_params.get("downscale_xy", 1))
		sources.append(("x", x_vol, float(z_step_eff) / float(max(1, x_ds))))
		channel_names.extend(["dir0_x", "dir1_x"])

	# Validate pred-dt early, before any heavy processing
	if pred_dt_path:
		_pred_check = zarr.open(str(pred_dt_path), mode="r")
		if not hasattr(_pred_check, "shape"):
			if hasattr(_pred_check, "__getitem__") and "0" in _pred_check:
				_pred_check = _pred_check["0"]
			else:
				raise ValueError(f"pred-dt must point to a zarr array or OME-Zarr group, got: {pred_dt_path}")
		_pred_shape = tuple(int(v) for v in _pred_check.shape)
		if len(_pred_shape) != 3:
			raise ValueError(f"pred-dt array must be 3D (Z,Y,X), got shape {_pred_shape}")
		print(f"[integrate_directions] pred-dt={pred_dt_path} shape={_pred_shape}", flush=True)
		del _pred_check, _pred_shape
		channel_names.append("pred_dt")

	if not sources:
		raise ValueError("at least one of y_volume_path or x_volume_path must be provided")

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
		dimension_separator="/",
	)
	out_params = dict(z_params)
	out_params["channels"] = channel_names
	out.attrs["preprocess_params"] = out_params

	print(f"[integrate_directions] z_volume={z_volume_path} shape={z_shape}")
	for ax_label, vol, z_ratio in sources:
		print(f"[integrate_directions] {ax_label}_volume shape={tuple(int(v) for v in vol.shape)} z_ratio={z_ratio:.1f}")
	print(f"[integrate_directions] -> {output_path} shape=({n_out_ch}, {ref_z}, {ref_y}, {ref_x})")

	t0 = time.time()
	for zi_start in range(0, ref_z, batch_size):
		zi_end = min(ref_z, zi_start + batch_size)

		# Copy all 5 z-volume channels for this batch
		out[:5, zi_start:zi_end] = np.asarray(z_vol[:5, zi_start:zi_end])

		ch_off = 5
		for ax_label, vol, z_ratio in sources:
			vol_nz = int(vol.shape[1])
			vol_ny = int(vol.shape[2])
			vol_nx = int(vol.shape[3])

			# Map z-vol Z indices to source Z indices
			src_indices = [min(int(round(zi * z_ratio)), vol_nz - 1) for zi in range(zi_start, zi_end)]

			# Read dir0 (ch 2) and dir1 (ch 3) for each source Z index
			slices = []
			for sj in src_indices:
				slices.append(np.asarray(vol[2:4, sj]))  # (2, Yv, Xv)
			combined = np.stack(slices).astype(np.float32)  # (batch, 2, Yv, Xv)

			# Resize to (ref_y, ref_x) via bilinear interpolation
			if vol_ny != ref_y or vol_nx != ref_x:
				combined_t = torch.from_numpy(combined)
				resized_t = F.interpolate(combined_t, size=(ref_y, ref_x), mode="bilinear", align_corners=False)
				combined = resized_t.numpy()

			out[ch_off, zi_start:zi_end] = np.clip(combined[:, 0], 0, 255).astype(np.uint8)
			out[ch_off + 1, zi_start:zi_end] = np.clip(combined[:, 1], 0, 255).astype(np.uint8)
			ch_off += 2

		elapsed = max(1e-6, time.time() - t0)
		per = elapsed / max(1, zi_end)
		eta = per * max(0, ref_z - zi_end)
		bar_w = 30
		fill = int(round(float(zi_end) / float(max(1, ref_z)) * bar_w))
		bar = "#" * fill + "-" * (bar_w - fill)
		print(
			f"\r[integrate_directions] [{bar}] {zi_end}/{ref_z} "
			f"({100.0 * zi_end / max(1, ref_z):.1f}%) "
			f"eta {int(eta // 60):02d}:{int(eta % 60):02d}",
			end="", flush=True,
		)

	print("", flush=True)
	print(f"[integrate_directions] done in {time.time() - t0:.1f}s")

	if pred_dt_path:
		pred_dt_ch = channel_names.index("pred_dt")
		crop_param = z_params.get("crop_xyzwhd", None)
		_compute_pred_dt_channel(
			pred_path=pred_dt_path,
			output_arr=out,
			channel_idx=pred_dt_ch,
			ref_z=ref_z, ref_y=ref_y, ref_x=ref_x,
			downscale_xy=int(z_params.get("downscale_xy", 1)),
			z_step_eff=z_step_eff,
			crop_xyzwhd=crop_param,
		)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Run tiled UNet cos inference on an OME-Zarr volume and write 8-bit OME-Zarr."
	)
	p.add_argument("--input", required=True, help="Input OME-Zarr array path (must be Z,Y,X array).")
	p.add_argument("--output", required=True, help="Output OME-Zarr group path.")
	p.add_argument("--unet-checkpoint", required=True, help="UNet checkpoint path (.pt).")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu".')
	p.add_argument("--axis", choices=["z", "y", "x"], default="z",
		help="Dimension to slice along (default: z). The 2D plane perpendicular to this axis is processed by the UNet.")
	p.add_argument("--crop-xyzwhd", "--crop", dest="crop_xyzwhd", type=int, nargs=6, default=None, help="Crop in absolute input coordinates: x y z w h d.")
	p.add_argument("--z-step", "--step", dest="z_step", type=int, default=5,
		help="Step along the slice axis before downscale (default: 5).")
	p.add_argument("--tile-size", type=int, default=2048, help="Tile size.")
	p.add_argument("--overlap", type=int, default=128, help="Tile overlap.")
	p.add_argument("--border", type=int, default=32, help="Tile border discard width.")
	p.add_argument("--downscale-xy", "--downscale", dest="downscale_xy", type=int, default=4,
		help="Downscale factor for the 2D plane dimensions (default: 4).")
	p.add_argument("--grad-mag-blur-sigma", type=float, default=4.0, help="Gaussian blur sigma on downscaled grad-mag.")
	p.add_argument("--dir-blur-sigma", type=float, default=2.0, help="Gaussian blur sigma on downscaled dir0/dir1.")
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
		z_step=int(args.z_step),
		tile_size=int(args.tile_size),
		overlap=int(args.overlap),
		border=int(args.border),
		downscale_xy=int(args.downscale_xy),
		grad_mag_blur_sigma=float(args.grad_mag_blur_sigma),
		dir_blur_sigma=float(args.dir_blur_sigma),
		chunk_z=int(args.chunk_z),
		chunk_yx=int(args.chunk_yx),
		measure_cuda_timings=bool(args.measure_cuda_timings),
	)
	return 0


def main_integrate(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Integrate direction channels from axis-y / axis-x preprocessed volumes into the axis-z reference volume."
	)
	p.add_argument("--z-volume", required=True, help="Axis-z preprocessed zarr (reference shape).")
	p.add_argument("--y-volume", default=None, help="Axis-y preprocessed zarr (optional).")
	p.add_argument("--x-volume", default=None, help="Axis-x preprocessed zarr (optional).")
	p.add_argument("--output", required=True, help="Output zarr path.")
	p.add_argument("--batch-size", type=int, default=32, help="Z-slices per batch for resize (default: 32).")
	p.add_argument("--pred-dt", default=None, help="Surface prediction zarr for distance-to-skeleton channel.")
	args = p.parse_args(argv)

	if args.y_volume is None and args.x_volume is None:
		p.error("at least one of --y-volume or --x-volume must be provided")

	run_integrate_directions(
		z_volume_path=str(args.z_volume),
		y_volume_path=str(args.y_volume) if args.y_volume else None,
		x_volume_path=str(args.x_volume) if args.x_volume else None,
		output_path=str(args.output),
		batch_size=int(args.batch_size),
		pred_dt_path=str(args.pred_dt) if args.pred_dt else None,
	)
	return 0


if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == "integrate":
		raise SystemExit(main_integrate(sys.argv[2:]))
	raise SystemExit(main())
