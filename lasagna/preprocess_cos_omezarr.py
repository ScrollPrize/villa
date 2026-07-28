from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
del _os

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
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
import zarr

try:
	from tiled_predict3d import (
		ModelAdapter,
		OmeZarrOutputAdapter,
		OutputAdapter,
		OutputChannelSpec,
		OutputProductSpec,
		PYRAMID_POLICY_CUSTOM,
		PYRAMID_POLICY_DIRECTION,
		PYRAMID_POLICY_NONE,
		PYRAMID_POLICY_SCALAR,
		VALID_PYRAMID_POLICIES,
		_RollingZBand,
		_atomic_zarr_write,
		build_product_omezarr_pyramids,
		_build_tile_positions,
		_build_omezarr_pyramid,
		_canonical_local_tile_positions,
		_canonical_tile_positions_for_output_region,
		_cleanup_predict3d_temp_files,
		create_product_omezarr_groups,
		_create_omezarr,
		_downscaled_tile_clip,
		_find_resume_z,
		_format_eta,
		_get_input_meta,
		_get_libc,
		_infer_tiled_products_3d,
		_infer_tiled_3d,
		_invalidate_pyramid_chunks,
		_input_has_chunks,
		_iter_chunk_origins_for_region,
		_omezarr_chunk_exists,
		_omezarr_chunk_group_complete,
		_omezarr_dim_sep,
		_omezarr_level_shape,
		_open_or_create_omezarr,
		_predict3d_finalized_status,
		_predict3d_overall_eta,
		_predict3d_progress_line,
		_predict3d_temp_pid,
		_pid_is_running,
		_pyrdown3d,
		_read_tile_zarr,
		_release_memmap_pages,
		_remove_path_quiet,
		_rolling_band_has_range,
		_zarr_chunk_path,
		write_lasagna_product_manifest,
	)
except ImportError:
	from lasagna.tiled_predict3d import (
		ModelAdapter,
		OmeZarrOutputAdapter,
		OutputAdapter,
		OutputChannelSpec,
		OutputProductSpec,
		PYRAMID_POLICY_CUSTOM,
		PYRAMID_POLICY_DIRECTION,
		PYRAMID_POLICY_NONE,
		PYRAMID_POLICY_SCALAR,
		VALID_PYRAMID_POLICIES,
		_RollingZBand,
		_atomic_zarr_write,
		build_product_omezarr_pyramids,
		_build_tile_positions,
		_build_omezarr_pyramid,
		_canonical_local_tile_positions,
		_canonical_tile_positions_for_output_region,
		_cleanup_predict3d_temp_files,
		create_product_omezarr_groups,
		_create_omezarr,
		_downscaled_tile_clip,
		_find_resume_z,
		_format_eta,
		_get_input_meta,
		_get_libc,
		_infer_tiled_products_3d,
		_infer_tiled_3d,
		_invalidate_pyramid_chunks,
		_input_has_chunks,
		_iter_chunk_origins_for_region,
		_omezarr_chunk_exists,
		_omezarr_chunk_group_complete,
		_omezarr_dim_sep,
		_omezarr_level_shape,
		_open_or_create_omezarr,
		_predict3d_finalized_status,
		_predict3d_overall_eta,
		_predict3d_progress_line,
		_predict3d_temp_pid,
		_pid_is_running,
		_pyrdown3d,
		_read_tile_zarr,
		_release_memmap_pages,
		_remove_path_quiet,
		_rolling_band_has_range,
		_zarr_chunk_path,
		write_lasagna_product_manifest,
	)

try:
	from normal_encoding import encode_normal_nxny_u8, estimate_normal as _estimate_normal
except ImportError:
	from lasagna.normal_encoding import (
		encode_normal_nxny_u8,
		estimate_normal as _estimate_normal,
	)

try:
	from common import load_unet, unet_infer_tiled
except ImportError:
	from lasagna.common import load_unet, unet_infer_tiled


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


def _round_up_to_multiple(v: int, f: int) -> int:
	f = max(1, int(f))
	return ((max(0, int(v)) + f - 1) // f) * f


def _grad_mag_factor_from_input_sd(input_sd: int) -> float:
	"""Scale grad_mag from input-voxel density units to base-voxel density units."""
	return 1.0 / float(max(1, int(input_sd)))


class LasagnaCosPredict3DAdapter:
	"""Lasagna cos/normal model adapter for shared tiled predict3d mechanics."""

	COS_PRODUCT = "cos"
	NORMAL_PRODUCT = "lasagna_normal"
	PRED_DT_PRODUCT = "pred_dt"

	def __init__(
		self,
		*,
		checkpoint: str,
		tile_size: int,
		device_name: str,
		cos_product: OutputProductSpec,
		normal_product: OutputProductSpec,
		pred_dt_product: OutputProductSpec | None = None,
	) -> None:
		self.checkpoint = str(checkpoint)
		self.tile_size = int(tile_size)
		self.device_name = str(device_name)
		self.cos_product = cos_product
		self.normal_product = normal_product
		self.pred_dt_product = pred_dt_product
		self.norm_type = None
		self.upsample_mode = None
		self.output_sigmoid = True

	@property
	def model_output_products(self) -> tuple[OutputProductSpec, ...]:
		return (self.cos_product, self.normal_product)

	@property
	def derived_output_products(self) -> tuple[OutputProductSpec, ...]:
		return () if self.pred_dt_product is None else (self.pred_dt_product,)

	@property
	def output_products(self) -> tuple[OutputProductSpec, ...]:
		return self.model_output_products + self.derived_output_products

	def product_by_name(self, name: str) -> OutputProductSpec:
		for product in self.output_products:
			if product.name == name:
				return product
		raise KeyError(name)

	def load_model(self, *, device: torch.device):
		from train_unet_3d import build_model as build_model_3d

		model, self.norm_type, self.upsample_mode, output_sigmoid = build_model_3d(
			self.tile_size,
			str(device),
			weights=self.checkpoint,
			strict=True,
		)
		self.output_sigmoid = bool(output_sigmoid)
		model.eval()
		return model

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		_ = device
		return model(tile)

	def accumulate_tile_output(
		self,
		raw_output,
		*,
		tile_origin_zyx,
		tile_weight,
		accumulators,
	) -> None:
		# The current Lasagna rolling loop owns accumulation until the shared
		# runner is fully generalized; this method satisfies the shared boundary.
		_ = tile_origin_zyx, tile_weight
		accumulators["raw_output"] = raw_output

	def product_tensors_from_output(self, raw_output) -> dict[str, torch.Tensor]:
		output = raw_output.to(dtype=torch.float32)
		if output.ndim != 5:
			raise ValueError("Lasagna predict3d model output must have shape B,C,D,H,W")
		if int(output.shape[1]) < 8:
			raise ValueError(
				"Lasagna predict3d model output must contain at least 8 channels: "
				"cos, grad_mag, and six direction channels"
			)
		return {
			self.cos_product.name: output[:, 0:1],
			self.normal_product.name: output[:, 1:8],
		}

	def finalize_product_slab(
		self,
		product: OutputProductSpec,
		raw_slab: np.ndarray,
	) -> dict[str, np.ndarray]:
		slab = np.asarray(raw_slab, dtype=np.float32)
		if product.name == self.cos_product.name:
			if slab.ndim != 4 or int(slab.shape[0]) != 1:
				raise ValueError(f"cos slab must have shape 1,D,H,W; got {slab.shape}")
			return {
				"cos": np.clip(slab[0] * 255.0, 0.0, 255.0).astype(np.uint8),
			}
		if product.name == self.normal_product.name:
			if slab.ndim != 4 or int(slab.shape[0]) != 7:
				raise ValueError(
					f"normal slab must have shape 7,D,H,W; got {slab.shape}"
				)
			gm_u8 = np.clip(slab[0] * 1000.0, 0.0, 255.0).astype(np.uint8)
			nx_u8, ny_u8 = encode_normal_nxny_u8(
				slab[1],
				slab[2],
				slab[3],
				slab[4],
				slab[5],
				slab[6],
			)
			return {"grad_mag": gm_u8, "nx": nx_u8, "ny": ny_u8}
		raise KeyError(product.name)


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


def _edt_reader_proc(pred_path, work_list, overlap, pZ, pY, pX,
					  shm_in_names, shm_shapes, free_q, ready_q, work_q, stop_evt):
	"""Reader process: grab chunk indices from work_q, free slots from free_q,
	read zarr into shared memory, signal ready_q.  Module-level for pickling."""
	import tensorstore as ts
	import multiprocessing.shared_memory as _shm2
	ctx = ts.Context({
		'data_copy_concurrency': {'limit': 1},
		'file_io_concurrency': {'limit': 4},
	})
	spec = {'driver': 'zarr', 'kvstore': {'driver': 'file',
			'path': os.path.normpath(str(pred_path))}}
	pred = ts.open(spec, read=True, open=True, context=ctx).result()
	_n = 0; _t_wq = 0.0; _t_fq = 0.0; _t_read = 0.0; _t_copy = 0.0
	while not stop_evt.is_set():
		_t0 = time.monotonic()
		try:
			chunk_idx = work_q.get(timeout=0.1)
		except Exception:
			if work_q.empty():
				break
			continue
		_t1 = time.monotonic()
		slot = free_q.get()
		_t2 = time.monotonic()

		cz0, cz1, cy0, cy1, cx0, cx1 = work_list[chunk_idx]
		rz0 = max(0, cz0 - overlap)
		rz1 = min(pZ, cz1 + overlap)
		ry0 = max(0, cy0 - overlap)
		ry1 = min(pY, cy1 + overlap)
		rx0 = max(0, cx0 - overlap)
		rx1 = min(pX, cx1 + overlap)

		raw = pred[rz0:rz1, ry0:ry1, rx0:rx1].read().result()
		binary = (np.asarray(raw) > 0).astype(np.uint8)
		del raw
		_t3 = time.monotonic()

		sm = _shm2.SharedMemory(name=shm_in_names[slot])
		buf = np.ndarray(shm_shapes[slot], dtype=np.uint8, buffer=sm.buf)
		ashp = binary.shape
		buf[:ashp[0], :ashp[1], :ashp[2]] = binary
		sm.close()
		_t4 = time.monotonic()

		ready_q.put((slot, chunk_idx, ashp))

		_n += 1; _t_wq += _t1-_t0; _t_fq += _t2-_t1; _t_read += _t3-_t2; _t_copy += _t4-_t3
		if _n % 20 == 0:
			print(f"  [reader pid={os.getpid()}] n={_n} "
				  f"work_q={1000*_t_wq/_n:.0f} free_q={1000*_t_fq/_n:.0f} "
				  f"read={1000*_t_read/_n:.0f} copy={1000*_t_copy/_n:.0f}ms/chunk",
				  flush=True)


def _edt_writer_proc(out_path, out_level, work_list, overlap, pZ, pY, pX,
					  pred_z0, pred_y0, pred_x0, out_z0, out_y0, out_x0,
					  scaledown, shm_out_names, shm_out_shapes,
					  write_q, free_q, stop_evt, n_levels=0):
	"""Writer process: grab results from write_q, downsample + write zarr from
	shared memory, release slot to free_q.

	Writes are atomic: each chunk is written to a temp zarr location first,
	then the chunk file is renamed into the real output. A killed process
	can never leave a half-written chunk in the output.
	"""
	import tensorstore as ts
	import multiprocessing.shared_memory as _shm2
	import json as _json
	import shutil as _shutil
	import uuid as _uuid
	ctx = ts.Context({
		'data_copy_concurrency': {'limit': 1},
		'file_io_concurrency': {'limit': 4},
	})
	# Open the real output for reading shape info
	level_path = os.path.normpath(os.path.join(str(out_path), str(out_level)))
	spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': level_path}}
	out_arr = ts.open(spec, read=True, open=True, context=ctx).result()
	out_vol_shape = tuple(out_arr.shape)

	# Create a per-worker temp zarr dir for atomic writes (outside OME-Zarr)
	_out_dir = os.path.dirname(os.path.normpath(str(out_path)))
	_zarr_name = os.path.basename(os.path.normpath(str(out_path)))
	tmp_level_path = os.path.join(
		_out_dir,
		f".tmp.{_zarr_name}.{out_level}.{os.getpid()}.{_uuid.uuid4().hex}",
	)
	os.makedirs(tmp_level_path, exist_ok=True)
	# Copy .zarray metadata so tensorstore can open it
	zarray_src = os.path.join(level_path, ".zarray")
	zarray_dst = os.path.join(tmp_level_path, ".zarray")
	if os.path.isfile(zarray_src) and not os.path.isfile(zarray_dst):
		import shutil
		shutil.copy2(zarray_src, zarray_dst)
	tmp_spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': tmp_level_path}}
	tmp_arr = ts.open(tmp_spec, read=True, write=True, open=True, context=ctx).result()

	# Read dimension separator for chunk file naming
	dim_sep = "/"
	if os.path.isfile(zarray_src):
		with open(zarray_src) as f:
			dim_sep = _json.load(f).get("dimension_separator", ".")

	_n = 0; _t_wq = 0.0; _t_shm = 0.0; _t_ds = 0.0; _t_wr = 0.0
	while not stop_evt.is_set():
		_tw0 = time.monotonic()
		try:
			item = write_q.get(timeout=0.1)
		except Exception:
			continue
		if item is None:
			break
		_tw1 = time.monotonic()
		slot, chunk_idx, ashp = item

		cz0, cz1, cy0, cy1, cx0, cx1 = work_list[chunk_idx]
		rz0 = max(0, cz0 - overlap)
		ry0 = max(0, cy0 - overlap)
		rx0 = max(0, cx0 - overlap)

		sm = _shm2.SharedMemory(name=shm_out_names[slot])
		buf = np.ndarray(shm_out_shapes[slot], dtype=np.uint8, buffer=sm.buf)
		dt_u8 = buf[:ashp[0], :ashp[1], :ashp[2]].copy()
		sm.close()
		_tw2 = time.monotonic()

		# Downsample all z-slices, assemble into a single 3D output block
		sz, sy, sx = cz1 - cz0, cy1 - cy0, cx1 - cx0
		center = dt_u8[cz0 - rz0:cz0 - rz0 + sz,
					   cy0 - ry0:cy0 - ry0 + sy,
					   cx0 - rx0:cx0 - rx0 + sx]
		oyi_base = out_y0 + (cy0 - pred_y0) // scaledown
		oxi_base = out_x0 + (cx0 - pred_x0) // scaledown
		out_slices = []
		for zl in range(0, sz, scaledown):
			slc = center[zl]
			if scaledown > 1:
				oh = max(1, slc.shape[0] // scaledown)
				ow = max(1, slc.shape[1] // scaledown)
				slc = cv2.resize(slc, (ow, oh), interpolation=cv2.INTER_AREA)
			out_slices.append(slc)
		_tw3 = time.monotonic()

		# Write to temp zarr, then rename chunk files into real output (atomic)
		if out_slices:
			block = np.stack(out_slices, axis=0)
			ozi_start = out_z0 + (cz0 - pred_z0) // scaledown
			oy_end = min(out_vol_shape[1], oyi_base + block.shape[1])
			ox_end = min(out_vol_shape[2], oxi_base + block.shape[2])
			wy = oy_end - oyi_base
			wx = ox_end - oxi_base
			oz_end = min(out_vol_shape[0], ozi_start + block.shape[0])
			wz = oz_end - ozi_start
			if wz > 0 and wy > 0 and wx > 0:
				# Write to temp
				tmp_arr[ozi_start:oz_end, oyi_base:oy_end, oxi_base:ox_end].write(
					block[:wz, :wy, :wx]).result()

				# Rename each chunk file from temp to real output
				chunks = tmp_arr.chunk_layout.read_chunk.shape
				if chunks is not None:
					cs = [int(c) for c in chunks]
				else:
					cs = [32, 32, 32]  # fallback
				for cz in range(ozi_start, oz_end, cs[0]):
					for cy in range(oyi_base, oy_end, cs[1]):
						for cx in range(oxi_base, ox_end, cs[2]):
							iz, iy, ix = cz // cs[0], cy // cs[1], cx // cs[2]
							if dim_sep == "/":
								rel = os.path.join(str(iz), str(iy), str(ix))
							else:
								rel = f"{iz}{dim_sep}{iy}{dim_sep}{ix}"
							src_file = os.path.join(tmp_level_path, rel)
							dst_file = os.path.join(level_path, rel)
							if os.path.isfile(src_file):
								if os.path.isfile(dst_file):
									continue
								os.makedirs(os.path.dirname(dst_file), exist_ok=True)
								os.replace(src_file, dst_file)  # atomic on Linux
								if n_levels > 0:
									_invalidate_pyramid_chunks(
										os.path.normpath(str(out_path)), int(out_level),
										n_levels, iz, iy, ix)
		_tw4 = time.monotonic()

		free_q.put(slot)

		_n += 1
		_t_wq += _tw1 - _tw0; _t_shm += _tw2 - _tw1
		_t_ds += _tw3 - _tw2; _t_wr += _tw4 - _tw3
		if _n % 20 == 0:
			print(f"  [writer pid={os.getpid()}] n={_n} "
				  f"wq={1000*_t_wq/_n:.0f} shm={1000*_t_shm/_n:.0f} "
				  f"ds={1000*_t_ds/_n:.0f} write={1000*_t_wr/_n:.0f}ms/chunk",
				  flush=True)
	try:
		_shutil.rmtree(tmp_level_path)
	except FileNotFoundError:
		pass
	except OSError:
		pass




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
	n_levels: int = 0,
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

	# EDT chunk = power-of-two multiple of out_chunk_in_pred that fits GPU memory.
	# out_chunk_in_pred = ome_chunk * scaledown: pred voxels per output chunk.
	# Peak GPU per chunk: ~12 bytes/voxel (bool + 2×float32 + uint8).
	# Batch holds multiple chunks on GPU: inputs + results + one EDT working set.
	BATCH = 8
	out_chunk_in_pred = max(1, scaledown) * max(1, ome_chunk)
	try:
		_gpu_free = cp.cuda.Device().mem_info[0]
	except Exception:
		_gpu_free = 4 * 2**30
	k = 1
	while True:
		next_k = k * 2
		side = next_k * out_chunk_in_pred + 2 * overlap
		# Memory: BATCH inputs (uint8) + BATCH results (uint8) + 1 EDT working set (12 bytes/vox)
		mem = BATCH * side**3 * 2 + side**3 * 12
		if mem > _gpu_free * 0.5:
			break
		k = next_k
	edt_chunk = k * out_chunk_in_pred
	_padded = edt_chunk + 2 * overlap
	_peak_mb = (BATCH * _padded**3 * 2 + _padded**3 * 12) / 2**20
	print(f"[pred_dt] edt_chunk={edt_chunk} ({k}x{out_chunk_in_pred}, "
		  f"gpu_free={_gpu_free / 2**30:.1f}GiB, peak={_peak_mb:.0f}MB)", flush=True)
	# Validate alignment — caller must pass chunk-aligned coordinates
	ocp = out_chunk_in_pred
	for name, val, limit in [("pred_z0", pred_z0, pZ), ("pred_y0", pred_y0, pY), ("pred_x0", pred_x0, pX),
							 ("pred_z1", pred_z1, pZ), ("pred_y1", pred_y1, pY), ("pred_x1", pred_x1, pX)]:
		if val != limit and val % ocp != 0:
			raise ValueError(
				f"_compute_pred_dt_slab: {name}={val} not aligned to {ocp}. "
				"Caller must pass chunk-aligned coordinates."
			)

	def _out_region_complete(cz0, cz1, cy0, cy1, cx0, cx1):
		"""Check all pred-dt output chunks covered by this pred-src region."""
		ozi0 = out_z0 + (cz0 - pred_z0) // scaledown
		oyi0 = out_y0 + (cy0 - pred_y0) // scaledown
		oxi0 = out_x0 + (cx0 - pred_x0) // scaledown
		ozi1 = out_z0 + (cz1 - pred_z0 + scaledown - 1) // scaledown
		oyi1 = out_y0 + (cy1 - pred_y0 + scaledown - 1) // scaledown
		oxi1 = out_x0 + (cx1 - pred_x0 + scaledown - 1) // scaledown
		shape = tuple(int(v) for v in output_level_arr.shape)
		for oz, oy, ox in _iter_chunk_origins_for_region(
			ozi0, ozi1, oyi0, oyi1, oxi0, oxi1,
			ome_chunk,
			(shape[0], shape[1], shape[2]),
		):
			if not _omezarr_chunk_exists(
				output_omezarr_path, int(output_level_key), oz, oy, ox, ome_chunk,
			):
				return False
		return True

	# Build work list, skipping chunks whose output already exists
	work: list[tuple[int, int, int, int, int, int]] = []
	skipped = 0
	for cz0 in range(pred_z0, pred_z1, edt_chunk):
		cz1 = min(pred_z1, cz0 + edt_chunk)
		for cy0 in range(pred_y0, pred_y1, edt_chunk):
			cy1 = min(pred_y1, cy0 + edt_chunk)
			for cx0 in range(pred_x0, pred_x1, edt_chunk):
				cx1 = min(pred_x1, cx0 + edt_chunk)
				if _out_region_complete(cz0, cz1, cy0, cy1, cx0, cx1):
					skipped += 1
					continue
				# Skip if pred zarr has no input data in this region
				if not _input_has_chunks(pred_path, cz0, cz1, cy0, cy1, cx0, cx1):
					skipped += 1
					continue
				work.append((cz0, cz1, cy0, cy1, cx0, cx1))

	n_chunks = len(work)
	if progress is not None:
		done_before = int(progress.get("edt_done", 0))
		total_est = max(int(progress.get("edt_total_est", 0)), done_before + skipped + n_chunks)
		progress["edt_total_est"] = total_est
		progress["edt_done"] = done_before + skipped
		progress["edt_skipped"] = int(progress.get("edt_skipped", 0)) + skipped
		progress["edt_remaining_est"] = max(0, total_est - int(progress["edt_done"]))
	if skipped > 0:
		print(f"[pred_dt] skipped {skipped} already-processed chunks", flush=True)
	if n_chunks == 0:
		return

	t0 = time.time()
	n_workers = min(read_workers if read_workers > 0 else max(1, (os.cpu_count() or 4) // 4),
				   n_chunks)

	print(f"[pred_dt] {n_chunks} chunks to process, {n_workers} workers, edt_chunk={edt_chunk}",
		  flush=True)

	# --- Shared-memory ring buffer pipeline ---
	# No array pickling through IPC — only slot indices through queues.
	# Readers write into shared memory, GPU reads from it, writers read from it.
	import multiprocessing.shared_memory as _shm
	from multiprocessing import Process, Queue, Event

	N_SLOTS = 32  # ring buffer depth — enough to keep GPU fed
	padded_side = edt_chunk + 2 * overlap  # max buffer dim per axis
	buf_shape = (padded_side, padded_side, padded_side)
	buf_bytes = int(np.prod(buf_shape))  # uint8

	# Allocate shared memory slots (input bool + output uint8)
	shm_in: list[_shm.SharedMemory] = []
	shm_out: list[_shm.SharedMemory] = []
	shm_in_names: list[str] = []
	shm_out_names: list[str] = []
	for i in range(N_SLOTS):
		si = _shm.SharedMemory(create=True, size=buf_bytes)
		so = _shm.SharedMemory(create=True, size=buf_bytes)
		shm_in.append(si)
		shm_out.append(so)
		shm_in_names.append(si.name)
		shm_out_names.append(so.name)

	shm_shapes = [buf_shape] * N_SLOTS
	shm_out_shapes = [buf_shape] * N_SLOTS

	# Queues: free slots, read-ready, write-ready
	free_q: Queue = Queue()
	ready_q: Queue = Queue()
	write_q: Queue = Queue()
	stop_evt = Event()

	# Seed free queue with (slot, chunk_idx) pairs
	for chunk_idx in range(n_chunks):
		# Assign to slots round-robin; readers will block on free_q when full
		pass
	# Actually: seed with empty slots, then feed chunk indices separately
	work_q: Queue = Queue()  # chunk indices to process
	for ci in range(n_chunks):
		work_q.put(ci)
	for s in range(N_SLOTS):
		free_q.put(s)

	# Launch reader and writer processes
	n_readers = min(n_workers, n_chunks)
	n_writers = min(n_workers, n_chunks)
	readers = []
	for _ in range(n_readers):
		p = Process(target=_edt_reader_proc,
					args=(pred_path, work, overlap, pZ, pY, pX,
						  shm_in_names, shm_shapes, free_q, ready_q, work_q, stop_evt))
		p.daemon = True
		p.start()
		readers.append(p)

	writers = []
	for _ in range(n_writers):
		p = Process(target=_edt_writer_proc,
					args=(output_omezarr_path, output_level_key, work, overlap, pZ, pY, pX,
						  pred_z0, pred_y0, pred_x0, out_z0, out_y0, out_x0,
						  scaledown, shm_out_names, shm_out_shapes,
						  write_q, free_q, stop_evt, n_levels))
		p.daemon = True
		p.start()
		writers.append(p)

	# Main process: pipelined GPU loop with 3 CUDA streams.
	# stream_h2d uploads chunk N+1 while stream_compute runs EDT on chunk N
	# and stream_d2h downloads chunk N-1's result.  All three overlap.
	gpu_done = 0
	_t_loop_sum = 0.0


	BATCH = 8  # process this many chunks per GPU round-trip
	stream_h2d = cp.cuda.Stream(non_blocking=True)
	stream_d2h = cp.cuda.Stream(non_blocking=True)
	_t_wait_sum = 0.0; _t_h2d_sum = 0.0; _t_edt_sum = 0.0; _t_d2h_sum = 0.0

	while gpu_done < n_chunks:
		# Collect a batch of ready chunks (up to BATCH, at least 1)
		_tw0 = time.monotonic()
		batch_items = []
		batch_items.append(ready_q.get())  # block for at least one
		while len(batch_items) < BATCH and not ready_q.empty():
			try:
				batch_items.append(ready_q.get_nowait())
			except Exception:
				break
		_tw1 = time.monotonic()
		B = len(batch_items)

		# Batch H2D: async upload all chunks to GPU
		gpu_binaries = []
		with stream_h2d:
			for slot, chunk_idx, ashp in batch_items:
				sm = _shm.SharedMemory(name=shm_in_names[slot])
				buf = np.ndarray(shm_shapes[slot], dtype=np.uint8, buffer=sm.buf)
				gpu_binaries.append(cp.asarray(buf[:ashp[0], :ashp[1], :ashp[2]]))
				sm.close()
		h2d_evt = stream_h2d.record()
		_th2d = time.monotonic()

		# Batch EDT + encode on GPU (wait for H2D, then sequential per chunk)
		gpu_results = []
		cp.cuda.get_current_stream().wait_event(h2d_evt)
		for m_gpu in gpu_binaries:
			outer_dt = cnd.distance_transform_edt(~m_gpu)
			inner_dt = cnd.distance_transform_edt(m_gpu)
			outer_enc = (128 - cp.clip(cp.round(outer_dt), 1, _MAX_DT)).astype(cp.uint8)
			inner_enc = (127 + cp.clip(cp.round(inner_dt), 1, _MAX_DT)).astype(cp.uint8)
			gpu_results.append(cp.where(m_gpu, inner_enc, outer_enc))
			del m_gpu, outer_dt, inner_dt, outer_enc, inner_enc
		del gpu_binaries
		compute_evt = cp.cuda.get_current_stream().record()
		_tedt = time.monotonic()

		# Batch D2H: async download all results to shared memory
		with stream_d2h:
			stream_d2h.wait_event(compute_evt)
			for i, (slot, chunk_idx, ashp) in enumerate(batch_items):
				sm_out = _shm.SharedMemory(name=shm_out_names[slot])
				obuf = np.ndarray(shm_out_shapes[slot], dtype=np.uint8, buffer=sm_out.buf)
				if ashp == shm_out_shapes[slot]:
					gpu_results[i].get(out=obuf)
				else:
					obuf[:ashp[0], :ashp[1], :ashp[2]] = cp.asnumpy(gpu_results[i])
				sm_out.close()
		stream_d2h.synchronize()
		for i, (slot, chunk_idx, ashp) in enumerate(batch_items):
			write_q.put((slot, chunk_idx, ashp))
			gpu_done += 1
		del gpu_results
		cp.get_default_memory_pool().free_all_blocks()
		_td2h = time.monotonic()

		_t_wait_sum += _tw1 - _tw0
		_t_h2d_sum += _th2d - _tw1
		_t_edt_sum += _tedt - _th2d
		_t_d2h_sum += _td2h - _tedt

		if progress is not None:
			progress["edt_done"] = int(progress.get("edt_done", 0)) + B
			progress["edt_processed"] = int(progress.get("edt_processed", 0)) + B
			progress["edt_time_sum"] = float(progress.get("edt_time_sum", 0.0)) + (_td2h - _tw0)
			progress["edt_remaining_est"] = max(
				0,
				int(progress.get("edt_total_est", 0)) - int(progress.get("edt_done", 0)),
			)
		if gpu_done == 0:
			continue
		elapsed = max(1e-6, time.time() - t0)
		eta = elapsed / gpu_done * (n_chunks - gpu_done)
		overall = _predict3d_overall_eta(progress)
		finalized = _predict3d_finalized_status(progress)
		vox_per_chunk = edt_chunk ** 3
		mvox_s = gpu_done * vox_per_chunk / elapsed / 1e6
		print(f"\r[pred_dt] gpu {gpu_done}/{n_chunks} "
			  f"({100.0 * gpu_done / n_chunks:.0f}%) "
			  f"eta {_format_eta(eta)} "
			  f"{mvox_s:.0f}Mvox/s "
			  f"wait={1000*_t_wait_sum/gpu_done:.0f} "
			  f"h2d={1000*_t_h2d_sum/gpu_done:.0f} "
			  f"edt={1000*_t_edt_sum/gpu_done:.0f} "
			  f"d2h={1000*_t_d2h_sum/gpu_done:.0f}ms "
			  f"B={B}"
			  f"{overall}{finalized}  ",
			  end="", flush=True)

	# Signal writers to stop
	for _ in writers:
		write_q.put(None)
	stop_evt.set()
	for p in readers:
		p.join(timeout=5)
	for p in writers:
		p.join(timeout=10)

	# Cleanup shared memory
	for si in shm_in:
		si.close()
		si.unlink()
	for so in shm_out:
		so.close()
		so.unlink()

	cp.get_default_memory_pool().free_all_blocks()
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

	# Auto-detect: find the zarr group root and read actual level 0 shape.
	try:
		import json as _json
		inp = Path(str(input_path).rstrip("/"))
		group_path = inp.parent if inp.name.isdigit() else inp

		# Try 1: read level 0 .zarray directly (most accurate)
		level0_zarray = group_path / "0" / ".zarray"
		if level0_zarray.is_file():
			with open(level0_zarray) as f:
				meta = _json.load(f)
			sh = tuple(int(v) for v in meta["shape"])
			if len(sh) == 3:
				print(f"[predict3d] base shape from level 0 .zarray: {sh}", flush=True)
				return sh

		# Try 2: read from .zattrs multiscales metadata
		zattrs_path = group_path / ".zattrs"
		if zattrs_path.is_file():
			with open(zattrs_path) as f:
				zattrs = _json.load(f)
			ms = zattrs.get("multiscales", [])
			if ms:
				datasets = ms[0].get("datasets", [])
				for ds in datasets:
					if ds.get("path") == "0":
						# Level 0 exists in metadata — try reading its .zarray
						break
				# Get level 0 shape from the group if accessible
				grp = zarr.open_group(str(group_path), mode="r")
				if "0" in [str(k) for k in grp.keys()]:
					arr = grp["0"]
					sh = tuple(int(v) for v in arr.shape)
					if len(sh) == 3:
						print(f"[predict3d] base shape from level 0 array: {sh}", flush=True)
						return sh

		# Try 3: fall back to finest available level × 2^level (approximate!)
		grp = zarr.open_group(str(group_path), mode="r")
		level_keys = sorted(int(k) for k in grp.keys() if k.isdigit())
		if level_keys:
			finest_lv = level_keys[0]
			arr = grp[str(finest_lv)]
			sh = tuple(int(v) for v in arr.shape)
			if len(sh) == 3:
				f = 2 ** finest_lv
				base = (sh[0] * f, sh[1] * f, sh[2] * f)
				print(f"[predict3d] WARNING: base shape estimated from level {finest_lv} "
					  f"shape={sh} × {f} → {base} (may be off by a few voxels)", flush=True)
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
	if tile_size is None:
		_ckpt_meta = torch.load(unet3d_checkpoint, map_location="cpu", weights_only=False)
		if not isinstance(_ckpt_meta, dict) or _ckpt_meta.get("patch_size") is None:
			raise ValueError(
				"checkpoint does not store patch_size; pass --tile-size explicitly"
			)
		tile_size = int(_ckpt_meta["patch_size"])
		print(f"[predict3d] tile_size={tile_size} from checkpoint metadata", flush=True)

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
	_removed_tmp = _cleanup_predict3d_temp_files(out_dir, prefix)
	if _removed_tmp > 0:
		print(f"[predict3d] removed {_removed_tmp} stale temp path(s) from {out_dir}", flush=True)

	# Output dimensions for each resolution
	cos_sd = max(1, int(cos_scaledown))
	other_sd = max(1, int(scaledown))

	full_cos_z = _ds_size(sh[0], cos_sd)
	full_cos_y = _ds_size(sh[1], cos_sd)
	full_cos_x = _ds_size(sh[2], cos_sd)

	full_other_z = _ds_size(sh[0], other_sd)
	full_other_y = _ds_size(sh[1], other_sd)
	full_other_x = _ds_size(sh[2], other_sd)

	# OME-Zarr output chunk size (needed for crop rounding)
	oc = max(1, int(ome_chunk))

	# Crop offsets at each output resolution, rounded to ome_chunk boundaries
	# so every write fills complete chunks (enables per-chunk skip on resume).
	cos_oz0 = (_ds_index(z0, cos_sd) // oc) * oc
	cos_oy0 = (_ds_index(y0, cos_sd) // oc) * oc
	cos_ox0 = (_ds_index(x0, cos_sd) // oc) * oc
	cos_oz1 = min(full_cos_z, ((_ds_index(z0, cos_sd) + _ds_size(nz, cos_sd) + oc - 1) // oc) * oc)
	cos_oy1 = min(full_cos_y, ((_ds_index(y0, cos_sd) + _ds_size(ny, cos_sd) + oc - 1) // oc) * oc)
	cos_ox1 = min(full_cos_x, ((_ds_index(x0, cos_sd) + _ds_size(nx_dim, cos_sd) + oc - 1) // oc) * oc)

	other_oz0 = (_ds_index(z0, other_sd) // oc) * oc
	other_oy0 = (_ds_index(y0, other_sd) // oc) * oc
	other_ox0 = (_ds_index(x0, other_sd) // oc) * oc
	other_oz1 = min(full_other_z, ((_ds_index(z0, other_sd) + _ds_size(nz, other_sd) + oc - 1) // oc) * oc)
	other_oy1 = min(full_other_y, ((_ds_index(y0, other_sd) + _ds_size(ny, other_sd) + oc - 1) // oc) * oc)
	other_ox1 = min(full_other_x, ((_ds_index(x0, other_sd) + _ds_size(nx_dim, other_sd) + oc - 1) // oc) * oc)

	# Expand input crop to cover the rounded output chunk boundaries.
	# The accumulator must produce data for every output chunk.
	z0 = min(z0, cos_oz0 * cos_sd, other_oz0 * other_sd)
	y0 = min(y0, cos_oy0 * cos_sd, other_oy0 * other_sd)
	x0 = min(x0, cos_ox0 * cos_sd, other_ox0 * other_sd)
	z1 = max(z1, min(sh[0], cos_oz1 * cos_sd), min(sh[0], other_oz1 * other_sd))
	y1 = max(y1, min(sh[1], cos_oy1 * cos_sd), min(sh[1], other_oy1 * other_sd))
	x1 = max(x1, min(sh[2], cos_ox1 * cos_sd), min(sh[2], other_ox1 * other_sd))
	z0 = max(0, z0); y0 = max(0, y0); x0 = max(0, x0)
	nz = z1 - z0; ny = y1 - y0; nx_dim = x1 - x0

	# Recompute ALL output offsets from expanded crop
	cos_oz0 = (_ds_index(z0, cos_sd) // oc) * oc
	cos_oy0 = (_ds_index(y0, cos_sd) // oc) * oc
	cos_ox0 = (_ds_index(x0, cos_sd) // oc) * oc
	cos_oz1 = min(full_cos_z, ((_ds_index(z0, cos_sd) + _ds_size(nz, cos_sd) + oc - 1) // oc) * oc)
	cos_oy1 = min(full_cos_y, ((_ds_index(y0, cos_sd) + _ds_size(ny, cos_sd) + oc - 1) // oc) * oc)
	cos_ox1 = min(full_cos_x, ((_ds_index(x0, cos_sd) + _ds_size(nx_dim, cos_sd) + oc - 1) // oc) * oc)
	other_oz0 = (_ds_index(z0, other_sd) // oc) * oc
	other_oy0 = (_ds_index(y0, other_sd) // oc) * oc
	other_ox0 = (_ds_index(x0, other_sd) // oc) * oc
	other_oz1 = min(full_other_z, ((_ds_index(z0, other_sd) + _ds_size(nz, other_sd) + oc - 1) // oc) * oc)
	other_oy1 = min(full_other_y, ((_ds_index(y0, other_sd) + _ds_size(ny, other_sd) + oc - 1) // oc) * oc)
	other_ox1 = min(full_other_x, ((_ds_index(x0, other_sd) + _ds_size(nx_dim, other_sd) + oc - 1) // oc) * oc)

	# Recompute output region sizes with expanded crop
	cos_wz = cos_oz1 - cos_oz0
	cos_wy = cos_oy1 - cos_oy0
	cos_wx = cos_ox1 - cos_ox0
	other_wz = other_oz1 - other_oz0
	other_wy = other_oy1 - other_oy0
	other_wx = other_ox1 - other_ox0

	# Effective scaledowns from base and OME-Zarr level numbers
	effective_cos_sd = input_sd * cos_sd
	effective_other_sd = input_sd * other_sd
	cos_level = round(_math.log2(effective_cos_sd)) if effective_cos_sd > 1 else 0
	other_level = round(_math.log2(effective_other_sd)) if effective_other_sd > 1 else 0
	# Ensure enough levels
	n_levels = max(n_levels, cos_level + 2, other_level + 2)
	print(f"[predict3d] input_sd={input_sd} cos_level={cos_level} other_level={other_level} "
		  f"n_levels={n_levels}", flush=True)

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
		# Validate base_shape is consistent with pred zarr
		for dim, bs, ps in zip("ZYX", base_shape_zyx, _pred_shape):
			expected = _omezarr_level_shape(base_shape_zyx, round(_math.log2(pred_sd)) if pred_sd > 1 else 0)
			break  # just need the tuple
		pred_level = round(_math.log2(pred_sd)) if pred_sd > 1 else 0
		expected_pred_shape = _omezarr_level_shape(base_shape_zyx, pred_level)
		for dim, exp, actual in zip("ZYX", expected_pred_shape, _pred_shape):
			if abs(exp - actual) > 1:
				raise ValueError(
					f"base_shape {base_shape_zyx} implies pred level {pred_level} shape "
					f"{expected_pred_shape}, but actual pred shape is {_pred_shape}. "
					f"Dimension {dim}: expected {exp}, got {actual}. "
					f"Check --base-ref or ensure the input zarr group has level 0."
				)

	# --- Create OME-Zarr outputs ---
	cos_omezarr_path = os.path.join(out_dir, f"{prefix}cos.ome.zarr")
	gm_omezarr_path = os.path.join(out_dir, f"{prefix}grad_mag.ome.zarr")
	nx_omezarr_path = os.path.join(out_dir, f"{prefix}nx.ome.zarr")
	ny_omezarr_path = os.path.join(out_dir, f"{prefix}ny.ome.zarr")
	dt_omezarr_path = os.path.join(out_dir, f"{prefix}pred_dt.ome.zarr") if pred_dt_path else None

	predict_adapter = LasagnaCosPredict3DAdapter(
		checkpoint=unet3d_checkpoint,
		tile_size=tile_size,
		device_name=str(torch_device),
		cos_product=OutputProductSpec(
			name=LasagnaCosPredict3DAdapter.COS_PRODUCT,
			level=cos_level,
			scaledown=effective_cos_sd,
			channels=(OutputChannelSpec("cos", relative_path=cos_omezarr_path),),
			chunk_size=oc,
			pyramid_policy=PYRAMID_POLICY_SCALAR,
		),
		normal_product=OutputProductSpec(
			name=LasagnaCosPredict3DAdapter.NORMAL_PRODUCT,
			level=other_level,
			scaledown=effective_other_sd,
			channels=(
				OutputChannelSpec("grad_mag", relative_path=gm_omezarr_path),
				OutputChannelSpec("nx", relative_path=nx_omezarr_path),
				OutputChannelSpec("ny", relative_path=ny_omezarr_path),
			),
			chunk_size=oc,
			pyramid_policy=PYRAMID_POLICY_CUSTOM,
		),
		pred_dt_product=(
			OutputProductSpec(
				name=LasagnaCosPredict3DAdapter.PRED_DT_PRODUCT,
				level=cos_level,
				scaledown=effective_cos_sd,
				channels=(OutputChannelSpec("pred_dt", relative_path=dt_omezarr_path),),
				chunk_size=oc,
				pyramid_policy=PYRAMID_POLICY_SCALAR,
			)
			if dt_omezarr_path is not None
			else None
		),
	)
	output_adapter = OmeZarrOutputAdapter(
		products=predict_adapter.output_products,
		n_levels=n_levels,
	)
	cos_product = predict_adapter.cos_product
	normal_product = predict_adapter.normal_product

	output_groups = create_product_omezarr_groups(
		products=predict_adapter.output_products,
		base_shape_zyx=base_shape_zyx,
		n_levels=n_levels,
		ome_chunk=oc,
	)
	write_lasagna_product_manifest(
		output_path=output_path,
		products=predict_adapter.output_products,
		base_shape_zyx=base_shape_zyx,
		crop_xyzwhd_base=crop_xyzwhd,
		source_to_base=source_to_base,
		grad_mag_factor=_grad_mag_factor_from_input_sd(input_sd),
	)

	# Level arrays (3D) for writing
	cos_lv_arr = output_groups["cos"][str(cos_level)]
	gm_lv_arr = output_groups["grad_mag"][str(other_level)]
	nx_lv_arr = output_groups["nx"][str(other_level)]
	ny_lv_arr = output_groups["ny"][str(other_level)]
	dt_lv_arr = output_groups["pred_dt"][str(cos_level)] if dt_omezarr_path else None

	# Resume is handled per-chunk: _is_tile_done skips tiles whose output
	# chunks exist, and the flush skips writing existing chunks.
	sd_fine = cos_sd
	sd_coarse = other_sd
	pad0 = max(0, int(border))
	stride = max(1, tile_size - overlap)
	sd_max = max(sd_fine, sd_coarse)
	Zp_acc = _round_up_to_multiple(nz + 2 * pad0, sd_max)
	Yp_acc = _round_up_to_multiple(ny + 2 * pad0, sd_max)
	Xp_acc = _round_up_to_multiple(nx_dim + 2 * pad0, sd_max)
	Zo_f_acc, Yo_f_acc, Xo_f_acc = Zp_acc // sd_fine, Yp_acc // sd_fine, Xp_acc // sd_fine
	Zo_c_acc, Yo_c_acc, Xo_c_acc = Zp_acc // sd_coarse, Yp_acc // sd_coarse, Xp_acc // sd_coarse

	_t_total_start = time.time()

	# --- Pause training if running (free GPU) ---
	from gpu_pause import gpu_pause_context
	_gpu_ctx = gpu_pause_context()
	if _gpu_ctx is not None:
		_gpu_ctx.__enter__()

	# --- Build model ---
	model = predict_adapter.load_model(device=torch_device)
	_output_sigmoid = predict_adapter.output_sigmoid

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
	n_other = 7

	# Accumulator-to-output coordinate helpers
	b_f = pad0 // sd_fine   # fine accumulator index where output starts
	b_c = pad0 // sd_coarse
	out_end_f = b_f + cos_wz   # fine accumulator index where output ends
	out_end_c = b_c + other_wz

	# Input zarr path for chunk existence checks (resolve level path)
	_input_zarr_dir = str(Path(str(input_path).rstrip("/")).resolve())

	def _output_chunk_has_input_support(cz, cy, cx, sd):
		z_end = min(_ds_size(sh[0], sd), int(cz) + oc)
		y_end = min(_ds_size(sh[1], sd), int(cy) + oc)
		x_end = min(_ds_size(sh[2], sd), int(cx) + oc)
		z_pos = _canonical_tile_positions_for_output_region(
			volume_size=sh[0], output_start=int(cz), output_end=z_end,
			scaledown=sd, tile_size=tile_size, stride=stride,
			border=pad0, scaledown_multiple=sd_max,
		)
		y_pos = _canonical_tile_positions_for_output_region(
			volume_size=sh[1], output_start=int(cy), output_end=y_end,
			scaledown=sd, tile_size=tile_size, stride=stride,
			border=pad0, scaledown_multiple=sd_max,
		)
		x_pos = _canonical_tile_positions_for_output_region(
			volume_size=sh[2], output_start=int(cx), output_end=x_end,
			scaledown=sd, tile_size=tile_size, stride=stride,
			border=pad0, scaledown_multiple=sd_max,
		)
		for pz in z_pos:
			src_z0 = max(0, int(pz) - pad0)
			src_z1 = min(sh[0], int(pz) - pad0 + tile_size)
			for py in y_pos:
				src_y0 = max(0, int(py) - pad0)
				src_y1 = min(sh[1], int(py) - pad0 + tile_size)
				for px in x_pos:
					src_x0 = max(0, int(px) - pad0)
					src_x1 = min(sh[2], int(px) - pad0 + tile_size)
					if _input_has_chunks(_input_zarr_dir, src_z0, src_z1, src_y0, src_y1, src_x0, src_x1):
						return True
		return False

	def _is_tile_done(tz, ty, tx):
		"""Check if all output chunks exist OR no input chunks in tile region."""
		ts = tile_size
		# Check if input has any data in this tile's region
		# Tile reads from input at [tz + z0 - border, tz + z0 + ts - border]
		in_z0 = max(0, tz + z0 - pad0)
		in_z1 = min(sh[0], tz + z0 - pad0 + ts)
		in_y0 = max(0, ty + y0 - pad0)
		in_y1 = min(sh[1], ty + y0 - pad0 + ts)
		in_x0 = max(0, tx + x0 - pad0)
		in_x1 = min(sh[2], tx + x0 - pad0 + ts)
		if not _input_has_chunks(_input_zarr_dir, in_z0, in_z1, in_y0, in_y1, in_x0, in_x1):
			return True  # no input data → skip tile
		# Fine (cos) output chunks touched by this tile.
		ts_f = ts // sd_fine
		az0, az1, _, _ = _downscaled_tile_clip(tz, sd_fine, ts_f, Zo_f_acc)
		ay0, ay1, _, _ = _downscaled_tile_clip(ty, sd_fine, ts_f, Yo_f_acc)
		ax0, ax1, _, _ = _downscaled_tile_clip(tx, sd_fine, ts_f, Xo_f_acc)
		fz0 = max(cos_oz0, cos_oz0 + az0 - b_f)
		fz1 = min(cos_oz1, cos_oz0 + az1 - b_f)
		fy0 = max(cos_oy0, cos_oy0 + ay0 - b_f)
		fy1 = min(cos_oy1, cos_oy0 + ay1 - b_f)
		fx0 = max(cos_ox0, cos_ox0 + ax0 - b_f)
		fx1 = min(cos_ox1, cos_ox0 + ax1 - b_f)
		for z, y, x in _iter_chunk_origins_for_region(
			fz0, fz1, fy0, fy1, fx0, fx1, oc, (full_cos_z, full_cos_y, full_cos_x),
		):
			if not output_adapter.product_chunk_complete(
				cos_product,
				chunk_origin_zyx=(z, y, x),
			):
				return False

		# Coarse (prediction) output chunks touched by this tile.
		ts_c = ts // sd_coarse
		az0c, az1c, _, _ = _downscaled_tile_clip(tz, sd_coarse, ts_c, Zo_c_acc)
		ay0c, ay1c, _, _ = _downscaled_tile_clip(ty, sd_coarse, ts_c, Yo_c_acc)
		ax0c, ax1c, _, _ = _downscaled_tile_clip(tx, sd_coarse, ts_c, Xo_c_acc)
		cz0 = max(other_oz0, other_oz0 + az0c - b_c)
		cz1 = min(other_oz1, other_oz0 + az1c - b_c)
		cy0 = max(other_oy0, other_oy0 + ay0c - b_c)
		cy1 = min(other_oy1, other_oy0 + ay1c - b_c)
		cx0 = max(other_ox0, other_ox0 + ax0c - b_c)
		cx1 = min(other_ox1, other_ox0 + ax1c - b_c)
		for z, y, x in _iter_chunk_origins_for_region(
			cz0, cz1, cy0, cy1, cx0, cx1, oc, (full_other_z, full_other_y, full_other_x),
		):
			if not output_adapter.product_chunk_complete(
				normal_product,
				chunk_origin_zyx=(z, y, x),
			):
				return False
		return True

	def _update_finalized_z(
		*,
		cos_base_z: int | None = None,
		other_base_z: int | None = None,
	) -> None:
		prev_cos = int(_progress.get("finalized_cos_base_z", _finalized_start_base_z))
		prev_other = int(_progress.get("finalized_other_base_z", _finalized_start_base_z))
		changed = False
		if cos_base_z is not None:
			next_cos = min(int(base_shape_zyx[0]), max(prev_cos, int(cos_base_z)))
			_progress["finalized_cos_base_z"] = next_cos
			changed = changed or next_cos != prev_cos
		if other_base_z is not None:
			next_other = min(int(base_shape_zyx[0]), max(prev_other, int(other_base_z)))
			_progress["finalized_other_base_z"] = next_other
			changed = changed or next_other != prev_other
		cos_done = int(_progress.get("finalized_cos_base_z", _finalized_start_base_z))
		other_done = int(_progress.get("finalized_other_base_z", _finalized_start_base_z))
		_progress["finalized_base_z"] = min(cos_done, other_done)
		if changed and int(_progress.get("tiles_total", 0)) > 0:
			print(f"\r{_predict3d_progress_line(_progress)}  ", end="", flush=True)

	def _on_z_complete(acc_fine, wsum_fine, acc_coarse, wsum_coarse,
					   complete_z_padded, pad0_inner):
		"""Flush completed z-bands to OME-Zarr, compute pred_dt, release pages."""
		# --- Flush fine (cos + pred_dt) ---
		complete_z_f = complete_z_padded // sd_fine
		flush_from_f = max(_prev_flush_fine[0], b_f)
		# Round flush point down to oc-aligned output z (complete chunks only).
		# If complete_z_f reaches or exceeds out_end_f, flush everything (last band).
		if complete_z_f >= out_end_f:
			flush_to_f = out_end_f
		else:
			complete_out_z = complete_z_f - b_f  # in output coords
			aligned_out_z = (complete_out_z // oc) * oc
			flush_to_f = b_f + aligned_out_z

		if flush_to_f > flush_from_f:
			out_zs = flush_from_f - b_f   # output z-start
			out_ze = flush_to_f - b_f     # output z-end
			if out_ze > out_zs:
				have_fine_acc = _rolling_band_has_range(acc_fine, flush_from_f, flush_to_f)
				if have_fine_acc:
					# Normalize accumulator in-place for this band
					acc_band = acc_fine.view(0, flush_from_f, flush_to_f)
					ws_band = wsum_fine.view(0, flush_from_f, flush_to_f)
					acc_band /= np.maximum(ws_band, 1e-7)

				eff_out_zs = out_zs
				eff_out_ze = out_ze

				if have_fine_acc and eff_out_ze > eff_out_zs:
					local_from = 0
					local_to = flush_to_f - flush_from_f
					# Trim to crop region (Y, X)
					yf = pad0_inner // sd_fine
					xf = pad0_inner // sd_fine
					oz = cos_oz0 + eff_out_zs
					# Process per output chunk — skip any that already exist
					cos_slab = None  # lazy: only compute if needed
					n_skip_cos = 0
					n_write_cos = 0
					for dz in range(0, eff_out_ze - eff_out_zs, oc):
						for dy in range(0, cos_wy, oc):
							for dx in range(0, cos_wx, oc):
								cz = oz + dz
								cy = cos_oy0 + dy
								cx = cos_ox0 + dx
								if output_adapter.product_chunk_complete(
									cos_product,
									chunk_origin_zyx=(cz, cy, cx),
								):
									n_skip_cos += 1
									continue
								if not _output_chunk_has_input_support(cz, cy, cx, cos_sd):
									n_skip_cos += 1
									continue
								# Compute slab lazily on first needed chunk
								if cos_slab is None:
									cos_slab = np.ascontiguousarray(
										acc_band[local_from:local_to, yf:yf + cos_wy, xf:xf + cos_wx])
									cos_slab = np.clip(cos_slab * 255.0, 0.0, 255.0).astype(np.uint8)
								# Write just this chunk's region
								cze = min(eff_out_ze - eff_out_zs, dz + oc)
								cye = min(cos_wy, dy + oc)
								cxe = min(cos_wx, dx + oc)
								wz = cze - dz; wy = cye - dy; wx = cxe - dx
								if wz > 0 and wy > 0 and wx > 0:
									output_adapter.write_product_chunk(
										cos_product,
										chunk_origin_zyx=(cz, cy, cx),
										data={"cos": cos_slab[dz:cze, dy:cye, dx:cxe]},
									)
								n_write_cos += 1

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
					_t_edt0 = time.time()
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
						n_levels=n_levels,
						progress=_progress,
					)
					_t_edt_total[0] += time.time() - _t_edt0

			fine_final_base_z = (cos_oz0 + max(0, flush_to_f - b_f)) * effective_cos_sd
			_update_finalized_z(cos_base_z=fine_final_base_z)
			acc_fine.discard_before(flush_to_f)
			wsum_fine.discard_before(flush_to_f)

		_prev_flush_fine[0] = max(_prev_flush_fine[0], flush_to_f)

		# --- Flush coarse (grad_mag, nx, ny) ---
		complete_z_c = complete_z_padded // sd_coarse
		flush_from_c = max(_prev_flush_coarse[0], b_c)
		if complete_z_c >= out_end_c:
			flush_to_c = out_end_c
		else:
			complete_out_zc = complete_z_c - b_c
			aligned_out_zc = (complete_out_zc // oc) * oc
			flush_to_c = b_c + aligned_out_zc

		if flush_to_c > flush_from_c:
			out_zs_c = flush_from_c - b_c
			out_ze_c = flush_to_c - b_c
			if out_ze_c > out_zs_c:
				have_coarse_acc = _rolling_band_has_range(acc_coarse, flush_from_c, flush_to_c)
				if have_coarse_acc:
					# Normalize
					acc_band_c = [acc_coarse.view(ch, flush_from_c, flush_to_c) for ch in range(n_other)]
					ws_band_c = wsum_coarse.view(0, flush_from_c, flush_to_c)
					for band_ch in acc_band_c:
						band_ch /= np.maximum(ws_band_c, 1e-7)

				eff_out_zs_c = out_zs_c
				eff_out_ze_c = out_ze_c

				if have_coarse_acc and eff_out_ze_c > eff_out_zs_c:
					oz_c = other_oz0 + eff_out_zs_c
					# Process per output chunk — skip existing
					slab = None  # lazy compute
					n_skip_c = 0
					n_write_c = 0
					for dz in range(0, eff_out_ze_c - eff_out_zs_c, oc):
						for dy in range(0, other_wy, oc):
							for dx in range(0, other_wx, oc):
								cz = oz_c + dz
								cy = other_oy0 + dy
								cx = other_ox0 + dx
								chunk_origin = (cz, cy, cx)
								if output_adapter.product_chunk_complete(
									normal_product,
									chunk_origin_zyx=chunk_origin,
								):
									n_skip_c += 1
									continue
								gm_exists = output_adapter.channel_chunk_exists(
									normal_product, "grad_mag", chunk_origin_zyx=chunk_origin)
								nx_exists = output_adapter.channel_chunk_exists(
									normal_product, "nx", chunk_origin_zyx=chunk_origin)
								ny_exists = output_adapter.channel_chunk_exists(
									normal_product, "ny", chunk_origin_zyx=chunk_origin)
								if not _output_chunk_has_input_support(cz, cy, cx, other_sd):
									n_skip_c += 1
									continue
								if slab is None:
									local_from_c = 0
									local_to_c = flush_to_c - flush_from_c
									yc = pad0_inner // sd_coarse
									xc = pad0_inner // sd_coarse
									slab = np.ascontiguousarray(np.stack([
										band_ch[local_from_c:local_to_c, yc:yc + other_wy, xc:xc + other_wx]
										for band_ch in acc_band_c
									], axis=0))
								cze = min(eff_out_ze_c - eff_out_zs_c, dz + oc)
								cye = min(other_wy, dy + oc)
								cxe = min(other_wx, dx + oc)
								s = slab[:, dz:cze, dy:cye, dx:cxe]

								# grad_mag
								gm_u8 = np.clip(s[0] * 1000.0, 0.0, 255.0).astype(np.uint8)
								wz = gm_u8.shape[0]; wy = gm_u8.shape[1]; wx = gm_u8.shape[2]
								chunk_data = {}
								if not gm_exists:
									chunk_data["grad_mag"] = gm_u8

								# Normals
								if not (nx_exists and ny_exists):
									nx_u8, ny_u8 = encode_normal_nxny_u8(
										s[1], s[2], s[3], s[4], s[5], s[6])
									if not nx_exists:
										chunk_data["nx"] = nx_u8
									if not ny_exists:
										chunk_data["ny"] = ny_u8
								if chunk_data:
									output_adapter.write_product_chunk(
										normal_product,
										chunk_origin_zyx=chunk_origin,
										data=chunk_data,
									)
								n_write_c += 1

			other_final_base_z = (other_oz0 + max(0, flush_to_c - b_c)) * effective_other_sd
			_update_finalized_z(other_base_z=other_final_base_z)
			acc_coarse.discard_before(flush_to_c)
			wsum_coarse.discard_before(flush_to_c)

		_prev_flush_coarse[0] = max(_prev_flush_coarse[0], flush_to_c)

	_t_inference_start = time.time()
	# --- Streaming inference + flush ---
	# Shared progress tracker for unified ETA across tiles + EDT
	_t_edt_total = [0.0]  # accumulated wall time in EDT calls
	_finalized_start_base_z = min(
		cos_oz0 * effective_cos_sd,
		other_oz0 * effective_other_sd,
	)
	_finalized_end_base_z = min(
		int(base_shape_zyx[0]),
		cos_oz1 * effective_cos_sd,
		other_oz1 * effective_other_sd,
	)
	_progress = {
		"t0": time.time(),
		"tiles_done": 0,
		"tiles_total": 0,  # set by _infer_tiled_3d
		"tiles_skipped": 0,
		"tiles_processed": 0,
		"tile_time_sum": 0.0,
		"tiles_remaining_est": 0,
		"edt_done": 0,
		"edt_skipped": 0,
		"edt_processed": 0,
		"edt_time_sum": 0.0,
		"edt_total_est": 0,  # updated below
		"edt_remaining_est": 0,
		"finalized_base_z": int(_finalized_start_base_z),
		"finalized_cos_base_z": int(_finalized_start_base_z),
		"finalized_other_base_z": int(_finalized_start_base_z),
		"finalized_base_z_total": int(_finalized_end_base_z),
	}
	# Initial EDT estimate; each slab tightens this with its actual skip/work list.
	if pred_dt_zarr is not None:
		_psh = tuple(int(v) for v in pred_dt_zarr.shape)
		import math as _m2
		_edt_nz = max(1, _m2.ceil((pred_z1_total := (cos_oz1 - cos_oz0) * effective_cos_sd // max(1, pred_sd)) / 448))
		_edt_ny = max(1, _m2.ceil((_psh[1]) / 448))
		_edt_nx = max(1, _m2.ceil((_psh[2]) / 448))
		_progress["edt_total_est"] = _edt_nz * _edt_ny * _edt_nx
		_progress["edt_remaining_est"] = _progress["edt_total_est"]

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
		skip_z_positions=0,
		progress=_progress,
		is_tile_done=_is_tile_done,
		temp_prefix=prefix,
		model_adapter=predict_adapter,
	)
	del model
	torch.cuda.empty_cache()

	# --- Build pyramids ---
	_t_inference_end = time.time()
	_t_pyramid_start = time.time()
	print("[predict3d] building OME-Zarr pyramids ...", flush=True)
	cos_crop_zyx = (cos_oz0, cos_oy0, cos_ox0, cos_oz1, cos_oy1, cos_ox1)
	other_crop_zyx = (other_oz0, other_oy0, other_ox0, other_oz1, other_oy1, other_ox1)
	crop_by_product = {
		cos_product.name: cos_crop_zyx,
		normal_product.name: other_crop_zyx,
	}
	if predict_adapter.pred_dt_product is not None:
		crop_by_product[predict_adapter.pred_dt_product.name] = cos_crop_zyx
	build_product_omezarr_pyramids(
		products=predict_adapter.output_products,
		n_levels=n_levels,
		ome_chunk=oc,
		crop_zyx_by_product=crop_by_product,
	)

	# --- Resume training ---
	if _gpu_ctx is not None:
		_gpu_ctx.__exit__(None, None, None)

	_removed_tmp_finish = _cleanup_predict3d_temp_files(
		out_dir, prefix, remove_current_process=True,
	)
	if _removed_tmp_finish > 0:
		print(f"[predict3d] removed {_removed_tmp_finish} temp path(s) on finish", flush=True)
	_t_total_end = time.time()
	_t_inf_edt = _t_inference_end - _t_inference_start
	_t_edt = _t_edt_total[0]
	_t_inf = _t_inf_edt - _t_edt  # inference = total streaming phase minus EDT
	_t_pyr = _t_total_end - _t_pyramid_start
	_t_setup = _t_inference_start - _t_total_start
	_t_total = _t_total_end - _t_total_start
	print(f"[predict3d] done. manifest: {output_path}", flush=True)
	print(f"[predict3d] timing: total={_t_total:.1f}s "
		  f"setup={_t_setup:.1f}s ({100*_t_setup/max(1e-9,_t_total):.0f}%) "
		  f"inference={_t_inf:.1f}s ({100*_t_inf/max(1e-9,_t_total):.0f}%) "
		  f"edt={_t_edt:.1f}s ({100*_t_edt/max(1e-9,_t_total):.0f}%) "
		  f"pyramid={_t_pyr:.1f}s ({100*_t_pyr/max(1e-9,_t_total):.0f}%)",
		  flush=True)


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
	p.add_argument("--tile-size", type=int, default=None,
		help="Inference tile size (default: checkpoint patch_size).")
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
		tile_size=int(args.tile_size) if args.tile_size is not None else None,
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


def cli_main(argv: list[str] | None = None) -> int:
	"""Dispatch the installed ``lasagna-preprocess`` command."""
	import sys
	args = list(sys.argv[1:] if argv is None else argv)
	if args and args[0] == "integrate":
		return main_integrate(args[1:])
	if args and args[0] == "predict3d":
		return main_predict3d(args[1:])
	return main(args)


if __name__ == "__main__":
	raise SystemExit(cli_main())
