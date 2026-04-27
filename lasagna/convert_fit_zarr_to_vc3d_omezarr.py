from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import zarr


def _down2(a: np.ndarray) -> np.ndarray:
	return a[::2, ::2, ::2]


def _shape_div2(shape: tuple[int, int, int], n: int) -> tuple[int, int, int]:
	z, y, x = (int(v) for v in shape)
	for _ in range(max(0, int(n))):
		z = max(1, (z + 1) // 2)
		y = max(1, (y + 1) // 2)
		x = max(1, (x + 1) // 2)
	return z, y, x


def _shape_mul2(shape: tuple[int, int, int], n: int) -> tuple[int, int, int]:
	z, y, x = (int(v) for v in shape)
	f = 2 ** max(0, int(n))
	return max(1, z * f), max(1, y * f), max(1, x * f)


def _ceil_div(a: int, b: int) -> int:
	return (int(a) + int(b) - 1) // int(b)


def _print_progress(*, prefix: str, done: int, total: int, t0: float) -> None:
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
		f"\r{prefix} [{bar}] {d}/{t} ({(100.0 * d / float(t)):.1f}%) eta {eta_m:02d}:{eta_s:02d}",
		end="",
		flush=True,
	)


def _load_channel_chunked(*, src: zarr.Array, ci: int,
                          zs0: int, zs1: int, ys0: int, ys1: int, xs0: int, xs1: int,
                          chunk: int, label: str, workers: int = 16) -> np.ndarray:
	z = zs1 - zs0
	y = ys1 - ys0
	x = xs1 - xs0
	if z <= 0:
		return np.zeros((0, y, x), dtype=np.uint8)
	cz = max(1, min(int(chunk), z))
	n_chunks = (z + cz - 1) // cz
	sz_gb = z * y * x / 1e9
	print(f"{label}: {z}×{y}×{x} uint8 ({sz_gb:.1f} Gvox, {n_chunks} chunks, {workers} workers)", flush=True)

	# Pre-allocate output and fill in parallel
	out = np.zeros((z, y, x), dtype=np.uint8)
	done = [0]
	lock = threading.Lock()
	t0 = time.time()
	_print_progress(prefix=label, done=0, total=z, t0=t0)

	def _read_chunk(zoff):
		z_a = zs0 + zoff
		z_b = min(zs0 + zoff + cz, zs1)
		data = np.asarray(src[ci, z_a:z_b, ys0:ys1, xs0:xs1]).astype(np.uint8, copy=False)
		out[zoff:zoff + (z_b - z_a)] = data
		with lock:
			done[0] += (z_b - z_a)
			_print_progress(prefix=label, done=done[0], total=z, t0=t0)

	from concurrent.futures import ThreadPoolExecutor, as_completed
	with ThreadPoolExecutor(max_workers=workers) as pool:
		futs = [pool.submit(_read_chunk, zoff) for zoff in range(0, z, cz)]
		for f in as_completed(futs):
			f.result()

	_print_progress(prefix=label, done=z, total=z, t0=t0)
	print("", flush=True)
	return out


def _write_z_chunks(*, dst: zarr.Array, data: np.ndarray, chunk: int, label: str, workers: int = 16) -> None:
	z = int(data.shape[0])
	cz = max(1, int(chunk))
	done = [0]
	lock = threading.Lock()
	t0 = time.time()
	_print_progress(prefix=label, done=0, total=z, t0=t0)

	def _write(z0):
		z1 = min(z, z0 + cz)
		dst[z0:z1, :, :] = data[z0:z1, :, :]
		with lock:
			done[0] += (z1 - z0)
			_print_progress(prefix=label, done=done[0], total=z, t0=t0)

	from concurrent.futures import ThreadPoolExecutor, as_completed
	with ThreadPoolExecutor(max_workers=workers) as pool:
		futs = [pool.submit(_write, z0) for z0 in range(0, z, cz)]
		for f in as_completed(futs):
			f.result()
	print("", flush=True)


def _write_z_chunks_at(*, dst: zarr.Array, data: np.ndarray, z0: int, y0: int, x0: int, chunk: int, label: str, workers: int = 16) -> None:
	z = int(data.shape[0])
	y = int(data.shape[1])
	x = int(data.shape[2])
	cz = max(1, int(chunk))
	done = [0]
	lock = threading.Lock()
	t0 = time.time()
	_print_progress(prefix=label, done=0, total=z, t0=t0)

	def _write(za):
		zb = min(z, za + cz)
		dst[int(z0) + za : int(z0) + zb, int(y0) : int(y0) + y, int(x0) : int(x0) + x] = data[za:zb, :, :]
		with lock:
			done[0] += (zb - za)
			_print_progress(prefix=label, done=done[0], total=z, t0=t0)

	from concurrent.futures import ThreadPoolExecutor, as_completed
	with ThreadPoolExecutor(max_workers=workers) as pool:
		futs = [pool.submit(_write, za) for za in range(0, z, cz)]
		for f in as_completed(futs):
			f.result()
	print("", flush=True)


def _channels_from_src(*, c_in: int, params: dict) -> list[str]:
	ch = params.get("channels", None)
	if isinstance(ch, list) and all(isinstance(v, str) for v in ch) and len(ch) == int(c_in):
		return [str(v) for v in ch]
	return [f"ch{i}" for i in range(int(c_in))]


def _first_filled_level_from_downscale(scaledown: int) -> int:
	d = max(1, int(scaledown))
	lv = 0
	while d > 1:
		d //= 2
		lv += 1
	return lv


def _process_slab_worker(args_tuple):
	"""Multiprocessing worker: read one Z-slab from source, write base level only."""
	(input_path_str, out_path_str, ci,
	 src_z0, src_z1, ys0, ys1, xs0, xs1,
	 out_oz, out_oy, out_ox,
	 first_filled_level) = args_tuple

	src = zarr.open(input_path_str, mode="r")
	slab = np.asarray(src[ci, src_z0:src_z1, ys0:ys1, xs0:xs1]).astype(np.uint8, copy=False)

	dst_g = zarr.open_group(out_path_str, mode="r+")
	dst_g[str(first_filled_level)][out_oz:out_oz + slab.shape[0],
								   out_oy:out_oy + slab.shape[1],
								   out_ox:out_ox + slab.shape[2]] = slab


def _downsample_chunk_worker(args_tuple):
	"""Multiprocessing worker: read one chunk-aligned region from level N,
	downsample 2x, write to level N+1.  Atomic: writes to a temp dir
	then os.replace() each chunk file into the real output.
	Uses tensorstore for I/O (no zarr-python async overhead)."""
	import json, shutil, tensorstore as ts
	(out_path_str, src_level, dst_level,
	 z0, z1, y0, y1, x0, x1) = args_tuple

	# Read via tensorstore (data_copy=1, file_io=4)
	ctx = ts.Context({
		'data_copy_concurrency': {'limit': 1},
		'file_io_concurrency': {'limit': 4},
	})
	src_path = os.path.normpath(os.path.join(out_path_str, str(src_level)))
	src_spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': src_path}}
	src_arr = ts.open(src_spec, read=True, open=True, context=ctx).result()
	slab = src_arr[z0:z1, y0:y1, x0:x1].read().result()
	down = np.asarray(slab)[::2, ::2, ::2]
	if down.size == 0:
		return
	dz0, dy0, dx0 = z0 // 2, y0 // 2, x0 // 2

	# Read chunk size and dimension separator from .zarray
	dst_level_path = os.path.normpath(os.path.join(out_path_str, str(dst_level)))
	zarray_path = os.path.join(dst_level_path, ".zarray")
	with open(zarray_path) as f:
		meta = json.load(f)
	chunk_size = meta["chunks"][0]
	sep = meta.get("dimension_separator", ".")

	# Write to temp level dir via tensorstore, then rename atomically
	# Temp dir outside the OME-Zarr, in the parent output directory
	_conv_out_dir = os.path.dirname(os.path.normpath(out_path_str))
	_conv_zarr_name = os.path.basename(os.path.normpath(out_path_str))
	tmp_path = os.path.join(_conv_out_dir, f".tmp.{_conv_zarr_name}.{dst_level}.{os.getpid()}")
	os.makedirs(tmp_path, exist_ok=True)
	tmp_zarray = os.path.join(tmp_path, ".zarray")
	if not os.path.isfile(tmp_zarray):
		shutil.copy2(zarray_path, tmp_zarray)
	tmp_spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': tmp_path}}
	tmp_arr = ts.open(tmp_spec, read=True, write=True, open=True, context=ctx).result()
	tmp_arr[dz0:dz0 + down.shape[0],
			dy0:dy0 + down.shape[1],
			dx0:dx0 + down.shape[2]].write(down).result()

	# Rename each chunk file atomically
	for cz in range(dz0, dz0 + down.shape[0], chunk_size):
		for cy in range(dy0, dy0 + down.shape[1], chunk_size):
			for cx in range(dx0, dx0 + down.shape[2], chunk_size):
				iz, iy, ix = cz // chunk_size, cy // chunk_size, cx // chunk_size
				if sep == "/":
					rel = os.path.join(str(iz), str(iy), str(ix))
				else:
					rel = f"{iz}{sep}{iy}{sep}{ix}"
				src_file = os.path.join(tmp_path, rel)
				dst_file = os.path.join(dst_level_path, rel)
				if os.path.isfile(src_file):
					os.makedirs(os.path.dirname(dst_file), exist_ok=True)
					os.replace(src_file, dst_file)


def _upsample_chunk_worker(args_tuple):
	"""Multiprocessing worker: read one chunk-aligned region from level N,
	upsample 2x (nearest-neighbor), write to level N-1."""
	(out_path_str, src_level, dst_level,
	 z0, z1, y0, y1, x0, x1) = args_tuple

	g = zarr.open_group(out_path_str, mode="r+")
	slab = np.asarray(g[str(src_level)][z0:z1, y0:y1, x0:x1])
	if slab.size == 0:
		return
	up = slab.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
	dz0, dy0, dx0 = z0 * 2, y0 * 2, x0 * 2
	dst = g[str(dst_level)]
	# Clip to dest shape (in case source wasn't evenly divisible)
	uz = min(up.shape[0], dst.shape[0] - dz0)
	uy = min(up.shape[1], dst.shape[1] - dy0)
	ux = min(up.shape[2], dst.shape[2] - dx0)
	dst[dz0:dz0 + uz, dy0:dy0 + uy, dx0:dx0 + ux] = up[:uz, :uy, :ux]


def run(
	*,
	input_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
	workers: int = 0,
	base_shape: tuple[int, int, int] | None = None,
) -> None:
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())

	a = zarr.open(str(input_path), mode="r")
	if len(tuple(int(v) for v in a.shape)) != 4:
		raise ValueError(f"input must be (C,Z,Y,X), got shape={tuple(a.shape)}")
	if str(a.dtype) != "uint8":
		raise ValueError(f"input dtype must be uint8, got {a.dtype}")

	params = dict(getattr(a, "attrs", {}).get("preprocess_params", {}) or {})
	scaledown = int(params.get("scaledown", 1) or 1)
	first_filled_level = _first_filled_level_from_downscale(scaledown)

	c_in, z_full, y_full, x_full = (int(v) for v in a.shape)
	zs0, ys0, xs0 = 0, 0, 0
	zs1, ys1, xs1 = z_full, y_full, x_full
	crop_xyzwhd = params.get("crop_xyzwhd", None)
	if isinstance(crop_xyzwhd, (list, tuple)) and len(crop_xyzwhd) == 6:
		x0f, y0f, z0f, wf, hf, df = (int(v) for v in crop_xyzwhd)
		if wf > 0 and hf > 0 and df > 0:
			xs0 = max(0, x0f // scaledown)
			ys0 = max(0, y0f // scaledown)
			zs0 = max(0, z0f // scaledown)
			xs1 = min(x_full, _ceil_div(x0f + wf, scaledown))
			ys1 = min(y_full, _ceil_div(y0f + hf, scaledown))
			zs1 = min(z_full, _ceil_div(z0f + df, scaledown))
	if zs1 <= zs0 or ys1 <= ys0 or xs1 <= xs0:
		raise ValueError(
			f"empty crop selection for conversion: z=[{zs0},{zs1}) y=[{ys0},{ys1}) x=[{xs0},{xs1})"
		)

	channels = _channels_from_src(c_in=c_in, params=params)
	ch_n = len(channels)

	# Base shape: explicit override, from zarr attrs, or from input zarr
	if base_shape is not None:
		base_full_shape = base_shape
	else:
		bs = params.get("base_shape_zyx")
		if bs is not None and len(bs) == 3:
			base_full_shape = tuple(int(v) for v in bs)
		else:
			base_full_shape = (z_full, y_full, x_full)
	crop_z = zs1 - zs0

	TAG = "[convert_fit_zarr_to_vc3d_omezarr]"
	t0_all = time.time()

	for ci, ch in enumerate(channels):
		out_path = Path(f"{output_prefix}_{ch}.ome.zarr")
		g = zarr.open_group(str(out_path), mode="w", zarr_format=2)

		# Create all pyramid level arrays upfront.
		# base_full_shape is level 0 (full res). Each level halves.
		arrs_shapes: dict[int, tuple] = {}
		for lv in range(levels):
			sh = _shape_div2(base_full_shape, lv)
			arrs_shapes[lv] = sh
			g.create_array(
				str(lv),
				shape=sh,
				chunks=(
					min(sh[0], chunk),
					min(sh[1], chunk),
					min(sh[2], chunk),
				),
				dtype=np.uint8,
				overwrite=True,
				fill_value=0,
			)

		# --- Phase 1: Write base level in parallel (chunk-aligned, no contention) ---
		cz = chunk
		slab_ranges = []
		for z_off in range(0, crop_z, cz):
			z_end = min(crop_z, z_off + cz)
			slab_ranges.append((z_off, z_end))

		n_slabs = len(slab_ranges)
		sz_gb = crop_z * (ys1 - ys0) * (xs1 - xs0) / 1e9
		print(f"{TAG} {ch}: {crop_z}×{ys1-ys0}×{xs1-xs0} ({sz_gb:.1f} Gvox), "
			  f"{n_slabs} slabs, {workers} workers, {levels} levels", flush=True)

		work_items = []
		for z_off, z_end in slab_ranges:
			work_items.append((
				str(input_path), str(out_path), ci,
				zs0 + z_off, zs0 + z_end, ys0, ys1, xs0, xs1,
				zs0 + z_off, ys0, xs0,
				first_filled_level,
			))

		t0 = time.time()
		done_count = [0]
		lock = threading.Lock()

		def _progress_thread():
			while not _stop.is_set():
				with lock:
					d = done_count[0]
				_print_progress(prefix=f"{TAG} {ch} L{first_filled_level}", done=d, total=n_slabs, t0=t0)
				_stop.wait(0.5)

		_stop = threading.Event()
		prog = threading.Thread(target=_progress_thread, daemon=True)
		prog.start()

		with multiprocessing.Pool(processes=workers) as pool:
			for _ in pool.imap_unordered(_process_slab_worker, work_items):
				with lock:
					done_count[0] += 1

		_stop.set()
		prog.join(timeout=2)
		_print_progress(prefix=f"{TAG} {ch} L{first_filled_level}", done=n_slabs, total=n_slabs, t0=t0)
		print("", flush=True)

		# --- Phase 2: Build coarser pyramid levels, one level at a time ---
		# Each level reads from the previous level and writes chunk-aligned
		# regions. No contention: each worker writes to a unique chunk.
		for lv in range(first_filled_level + 1, levels):
			src_lv = lv - 1
			src_shape = arrs_shapes[src_lv]
			dst_shape = arrs_shapes[lv]
			# Chunk-aligned work items at the SOURCE level (2× chunk so
			# each worker produces exactly one chunk at the dest level).
			cz2 = chunk * 2
			ds_work = []
			for z0 in range(0, src_shape[0], cz2):
				z1 = min(src_shape[0], z0 + cz2)
				for y0 in range(0, src_shape[1], cz2):
					y1 = min(src_shape[1], y0 + cz2)
					for x0 in range(0, src_shape[2], cz2):
						x1 = min(src_shape[2], x0 + cz2)
						ds_work.append((
							str(out_path), src_lv, lv,
							z0, z1, y0, y1, x0, x1,
						))
			n_ds = len(ds_work)
			t_lv = time.time()
			done_count[0] = 0

			_stop2 = threading.Event()
			def _prog2():
				while not _stop2.is_set():
					with lock:
						d = done_count[0]
					_print_progress(prefix=f"{TAG} {ch} L{lv}", done=d, total=n_ds, t0=t_lv)
					_stop2.wait(0.5)
			prog2 = threading.Thread(target=_prog2, daemon=True)
			prog2.start()

			with multiprocessing.Pool(processes=workers) as pool:
				for _ in pool.imap_unordered(_downsample_chunk_worker, ds_work):
					with lock:
						done_count[0] += 1

			_stop2.set()
			prog2.join(timeout=2)
			_print_progress(prefix=f"{TAG} {ch} L{lv}", done=n_ds, total=n_ds, t0=t_lv)
			print("", flush=True)

		# --- Phase 3: Upsample to fill levels below first_filled_level ---
		for lv in range(first_filled_level - 1, -1, -1):
			src_lv = lv + 1
			src_shape = arrs_shapes[src_lv]
			cz2 = chunk  # one source chunk → one dest 2×chunk region
			us_work = []
			for z0 in range(0, src_shape[0], cz2):
				z1 = min(src_shape[0], z0 + cz2)
				for y0 in range(0, src_shape[1], cz2):
					y1 = min(src_shape[1], y0 + cz2)
					for x0 in range(0, src_shape[2], cz2):
						x1 = min(src_shape[2], x0 + cz2)
						us_work.append((
							str(out_path), src_lv, lv,
							z0, z1, y0, y1, x0, x1,
						))
			if us_work:
				n_us = len(us_work)
				t_lv = time.time()
				done_count[0] = 0

				_stop3 = threading.Event()
				def _prog3():
					while not _stop3.is_set():
						with lock:
							d = done_count[0]
						_print_progress(prefix=f"{TAG} {ch} L{lv} (upsample)", done=d, total=n_us, t0=t_lv)
						_stop3.wait(0.5)
				prog3 = threading.Thread(target=_prog3, daemon=True)
				prog3.start()

				with multiprocessing.Pool(processes=workers) as pool:
					for _ in pool.imap_unordered(_upsample_chunk_worker, us_work):
						with lock:
							done_count[0] += 1

				_stop3.set()
				prog3.join(timeout=2)
				_print_progress(prefix=f"{TAG} {ch} L{lv} (upsample)", done=n_us, total=n_us, t0=t_lv)
				print("", flush=True)

		# Write metadata
		datasets = []
		for lv in range(levels):
			scale_fac = 2 ** lv
			datasets.append({
				"path": str(lv),
				"coordinateTransformations": [{
					"type": "scale",
					"scale": [float(scale_fac)] * 3,
				}],
			})
		g.attrs["multiscales"] = [{
			"version": "0.4",
			"name": str(ch),
			"axes": [
				{"name": "z", "type": "space", "unit": "pixel"},
				{"name": "y", "type": "space", "unit": "pixel"},
				{"name": "x", "type": "space", "unit": "pixel"},
			],
			"datasets": datasets,
		}]
		g.attrs["source_preprocess_params"] = params
		g.attrs["vc3d_convert"] = {
			"channel": str(ch),
			"channel_index": ci,
			"scaledown": scaledown,
			"crop_scaled_zyx_start": [zs0, ys0, xs0],
			"crop_scaled_zyx_stop": [zs1, ys1, xs1],
			"crop_offset_zyx": [zs0, ys0, xs0],
			"base_full_shape": list(base_full_shape),
			"levels": levels,
			"first_filled_level": first_filled_level,
		}

		elapsed_ch = time.time() - t0
		print(f"{TAG} {ch} done in {elapsed_ch:.1f}s", flush=True)

		done = ci + 1
		elapsed = max(1e-6, time.time() - t0_all)
		per = elapsed / done
		eta = max(0.0, per * (ch_n - done))
		eta_m = int(eta // 60.0)
		eta_s = int(eta % 60.0)
		bar_w = 30
		fill = int(round((done / max(1, ch_n)) * bar_w))
		bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
		print(
			f"\r[convert_fit_zarr_to_vc3d_omezarr] [{bar}] {done}/{ch_n} ({(100.0 * done / max(1, ch_n)):.1f}%) "
			f"eta {eta_m:02d}:{eta_s:02d}",
			end="",
			flush=True,
		)

	print("", flush=True)


def run_from_manifest(
	*,
	manifest_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
	workers: int = 0,
	channels: list[str] | None = None,
	base_shape: tuple[int, int, int] | None = None,
) -> None:
	"""Convert channels from a .lasagna.json manifest to VC3D OME-Zarr pyramids.

	Default channels: cos + pred_dt (if present).
	base_shape from manifest's base_shape_zyx, or derived from finest zarr.
	"""
	from lasagna_volume import LasagnaVolume

	vol = LasagnaVolume.load(manifest_path)
	TAG = "[convert_fit_zarr_to_vc3d_omezarr]"

	# Resolve base shape: explicit > manifest > derive from finest zarr
	if base_shape is None:
		base_shape = vol.base_shape_zyx
	if base_shape is None:
		# Derive from finest-resolution zarr × scaledown
		min_sd = min(g.sd_fac for g in vol.groups.values())
		for g in vol.groups.values():
			if g.sd_fac == min_sd:
				zarr_path = str(vol.path.parent / g.zarr_path)
				a = zarr.open(zarr_path, mode="r")
				sh = tuple(int(v) for v in a.shape)
				if len(sh) == 4:
					sh = sh[1:]
				base_shape = (sh[0] * min_sd, sh[1] * min_sd, sh[2] * min_sd)
				print(f"{TAG} derived base_shape={base_shape} from {g.zarr_path} "
					  f"(shape={sh}, sd_fac={min_sd})", flush=True)
				break
	if base_shape is None:
		raise ValueError("Cannot determine base_shape: pass --base-shape or set base_shape_zyx in manifest")

	# Select channels to convert
	all_ch = vol.all_channels()
	if channels is None:
		channels = []
		for name in ["cos", "pred_dt"]:
			if name in all_ch:
				channels.append(name)
		if not channels:
			channels = all_ch
	print(f"{TAG} manifest: {manifest_path}", flush=True)
	print(f"{TAG} base_shape={base_shape}, channels={channels}", flush=True)

	# Convert each selected channel
	for ch_name in channels:
		group, ch_idx = vol.channel_group(ch_name)
		zarr_path = str(vol.path.parent / group.zarr_path)

		# Per-channel output: output_prefix_<channel>.ome.zarr
		out_prefix = f"{output_prefix}_{ch_name}"

		# Build per-channel preprocess_params so run() can read scaledown + crop
		a = zarr.open(zarr_path, mode="r")
		params = dict(getattr(a, "attrs", {}).get("preprocess_params", {}))
		params.setdefault("scaledown", group.sd_fac)
		if vol.crops:
			# Use the first crop for legacy converter compat
			params.setdefault("crop_xyzwhd", list(vol.crops[0]))
		# Channel names for this group
		params["channels"] = group.channels

		# Temporarily set attrs so run() can read them
		# (run() reads from zarr attrs — if missing, write them)
		if "preprocess_params" not in dict(getattr(a, "attrs", {})):
			try:
				aw = zarr.open(zarr_path, mode="r+")
				aw.attrs["preprocess_params"] = params
			except Exception:
				pass  # read-only zarr, run() will use defaults

		print(f"{TAG} converting {ch_name} (group={[k for k,v in vol.groups.items() if v is group][0]}, "
			  f"ch_idx={ch_idx}, sd_fac={group.sd_fac})", flush=True)

		run(
			input_path=zarr_path,
			output_prefix=out_prefix,
			levels=levels,
			chunk=chunk,
			workers=workers,
			base_shape=base_shape,
		)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Convert fit zarr (C,Z,Y,X uint8) to per-channel VC3D OME-Zarr pyramids."
	)
	# Input: either --input (per-zarr) or --manifest (.lasagna.json)
	inp = p.add_mutually_exclusive_group(required=True)
	inp.add_argument("--input", help="Input zarr array path (C,Z,Y,X uint8).")
	inp.add_argument("--manifest", help="Input .lasagna.json manifest.")
	p.add_argument("--output-prefix", required=True,
		help="Output prefix; files become <prefix>_<channel>.ome.zarr")
	p.add_argument("--levels", type=int, default=5, help="Number of pyramid levels to create.")
	p.add_argument("--chunk", type=int, default=32, help="Chunk size in x/y/z.")
	p.add_argument("--workers", type=int, default=0,
				   help="Number of parallel workers (default: cpu_count).")
	p.add_argument("--base-shape", type=int, nargs=3, default=None,
		metavar=("Z", "Y", "X"),
		help="Explicit base (full-res) shape. Overrides manifest/zarr-derived shapes.")
	p.add_argument("--channels", nargs="+", default=None,
		help="Channel names to convert (manifest mode only). Default: cos + pred_dt.")
	args = p.parse_args(argv)

	bs = tuple(args.base_shape) if args.base_shape else None

	if args.manifest:
		run_from_manifest(
			manifest_path=str(args.manifest),
			output_prefix=str(args.output_prefix),
			levels=int(args.levels),
			chunk=int(args.chunk),
			workers=int(args.workers),
			channels=args.channels,
			base_shape=bs,
		)
	else:
		run(
			input_path=str(args.input),
			output_prefix=str(args.output_prefix),
			levels=int(args.levels),
			chunk=int(args.chunk),
			workers=int(args.workers),
			base_shape=bs,
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
