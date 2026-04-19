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
	"""Multiprocessing worker: read one Z-slab, downsample, write all pyramid levels."""
	(input_path_str, out_path_str, ci,
	 src_z0, src_z1, ys0, ys1, xs0, xs1,
	 out_oz, out_oy, out_ox,
	 first_filled_level, n_levels) = args_tuple

	src = zarr.open(input_path_str, mode="r")
	slab = np.asarray(src[ci, src_z0:src_z1, ys0:ys1, xs0:xs1]).astype(np.uint8, copy=False)

	dst_g = zarr.open_group(out_path_str, mode="r+")

	cur = slab
	oz, oy, ox = out_oz, out_oy, out_ox
	dst_g[str(first_filled_level)][oz:oz + cur.shape[0], oy:oy + cur.shape[1], ox:ox + cur.shape[2]] = cur

	for lv in range(first_filled_level + 1, n_levels):
		sz, sy, sx = oz & 1, oy & 1, ox & 1
		cur = cur[sz::2, sy::2, sx::2]
		oz, oy, ox = oz // 2, oy // 2, ox // 2
		if cur.size == 0:
			break
		dst_g[str(lv)][oz:oz + cur.shape[0], oy:oy + cur.shape[1], ox:ox + cur.shape[2]] = cur


def run(
	*,
	input_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
	workers: int = 0,
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
	base_full_shape = (z_full, y_full, x_full)
	crop_z = zs1 - zs0

	TAG = "[convert_fit_zarr_to_vc3d_omezarr]"
	t0_all = time.time()

	for ci, ch in enumerate(channels):
		out_path = Path(f"{output_prefix}_{ch}.ome.zarr")
		g = zarr.open_group(str(out_path), mode="w", zarr_format=2)

		# Create all pyramid level arrays upfront
		arrs_shapes: dict[int, tuple] = {}
		for lv in range(levels):
			if lv < first_filled_level:
				sh = _shape_mul2(base_full_shape, first_filled_level - lv)
			else:
				sh = _shape_div2(base_full_shape, lv - first_filled_level)
			arrs_shapes[lv] = sh
			if lv >= first_filled_level:
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

		# Align slabs to output chunk boundaries for zero write contention.
		# Each slab is chunk-aligned in Z at the base level.
		cz = chunk
		slab_ranges = []
		for z_off in range(0, crop_z, cz):
			z_end = min(crop_z, z_off + cz)
			slab_ranges.append((z_off, z_end))

		n_slabs = len(slab_ranges)
		sz_gb = crop_z * (ys1 - ys0) * (xs1 - xs0) / 1e9
		print(f"{TAG} {ch}: {crop_z}×{ys1-ys0}×{xs1-xs0} ({sz_gb:.1f} Gvox), "
			  f"{n_slabs} slabs, {workers} workers, {levels} levels", flush=True)

		# Build work items for multiprocessing
		work_items = []
		for z_off, z_end in slab_ranges:
			work_items.append((
				str(input_path), str(out_path), ci,
				zs0 + z_off, zs0 + z_end, ys0, ys1, xs0, xs1,
				zs0 + z_off, ys0, xs0,  # output offsets = source offsets
				first_filled_level, levels,
			))

		t0 = time.time()
		done_count = [0]
		lock = threading.Lock()

		def _progress_thread():
			while not _stop.is_set():
				with lock:
					d = done_count[0]
				_print_progress(prefix=f"{TAG} {ch}", done=d, total=n_slabs, t0=t0)
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
		_print_progress(prefix=f"{TAG} {ch}", done=n_slabs, total=n_slabs, t0=t0)
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


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Convert fit zarr (C,Z,Y,X uint8) to per-channel VC3D OME-Zarr pyramids."
	)
	p.add_argument("--input", required=True, help="Input zarr array path (C,Z,Y,X uint8).")
	p.add_argument("--output-prefix", required=True, help="Output prefix; files become <prefix>_<channel>.ome.zarr")
	p.add_argument("--levels", type=int, default=5, help="Number of pyramid levels to create.")
	p.add_argument("--chunk", type=int, default=32, help="Chunk size in x/y/z.")
	p.add_argument("--workers", type=int, default=0,
				   help="Number of parallel workers (default: cpu_count).")
	args = p.parse_args(argv)

	run(
		input_path=str(args.input),
		output_prefix=str(args.output_prefix),
		levels=int(args.levels),
		chunk=int(args.chunk),
		workers=int(args.workers),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
