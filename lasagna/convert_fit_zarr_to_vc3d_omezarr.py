from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import time

import numpy as np
from vc_reader.reader import ZarrWriter, create_group, open_zarr

try:
	from omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		print_progress,
		shape_div2,
	)
except ImportError:
	from lasagna.omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		print_progress,
		shape_div2,
	)


def _ceil_div(a: int, b: int) -> int:
	return (int(a) + int(b) - 1) // int(b)


def _run_with_progress(work, worker, *, workers: int, prefix: str) -> None:
	t0 = time.time()
	print_progress(prefix=prefix, done=0, total=len(work), t0=t0)
	with ThreadPoolExecutor(max_workers=workers) as pool:
		for done, _ in enumerate(pool.map(worker, work), 1):
			print_progress(prefix=prefix, done=done, total=len(work), t0=t0)
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


def _upsample_level(*, src: ZarrWriter, dst: ZarrWriter, chunk: int, workers: int, prefix: str) -> None:
	"""Fill a level from the one above by nearest-neighbor 2x upsampling (one source chunk at a time)."""
	starts = [
		(z0, y0, x0)
		for z0 in range(0, src.shape[0], chunk)
		for y0 in range(0, src.shape[1], chunk)
		for x0 in range(0, src.shape[2], chunk)
	]

	def upsample(start):
		z0, y0, x0 = start
		slab = src[z0:z0 + chunk, y0:y0 + chunk, x0:x0 + chunk]
		up = slab.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
		dz0, dy0, dx0 = z0 * 2, y0 * 2, x0 * 2
		# Clip to dest shape (in case source wasn't evenly divisible)
		uz = min(up.shape[0], dst.shape[0] - dz0)
		uy = min(up.shape[1], dst.shape[1] - dy0)
		ux = min(up.shape[2], dst.shape[2] - dx0)
		dst[dz0:dz0 + uz, dy0:dy0 + uy, dx0:dx0 + ux] = up[:uz, :uy, :ux]

	_run_with_progress(starts, upsample, workers=workers, prefix=prefix)


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

	a = open_zarr(input_path, threads=1)
	if len(tuple(int(v) for v in a.shape)) != 4:
		raise ValueError(f"input must be (C,Z,Y,X), got shape={tuple(a.shape)}")
	if str(a.dtype) != "uint8":
		raise ValueError(f"input dtype must be uint8, got {a.dtype}")

	params = dict(a.attrs.get("preprocess_params", {}) or {})
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
	normal_pair_channels = {"nx", "ny"}.issubset(set(channels))
	normal_pair_paths: dict[str, Path] = {}

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
		if normal_pair_channels and ch in {"nx", "ny"}:
			normal_pair_paths[ch] = out_path
		g = create_group(out_path, overwrite=True)

		# Create all pyramid level arrays upfront.
		# base_full_shape is level 0 (full res). Each level halves.
		level_arrays: dict[int, ZarrWriter] = {}
		for lv in range(levels):
			sh = shape_div2(base_full_shape, lv)
			level_arrays[lv] = g.create_array(
				str(lv),
				shape=sh,
				chunks=(min(sh[0], chunk), min(sh[1], chunk), min(sh[2], chunk)),
				dtype=np.uint8,
				skip_empty=True,
			)

		# --- Phase 1: Write base level in parallel (one Z-slab per chunk depth) ---
		slab_ranges = [(z_off, min(crop_z, z_off + chunk)) for z_off in range(0, crop_z, chunk)]
		sz_gb = crop_z * (ys1 - ys0) * (xs1 - xs0) / 1e9
		print(f"{TAG} {ch}: {crop_z}×{ys1-ys0}×{xs1-xs0} ({sz_gb:.1f} Gvox), "
			  f"{len(slab_ranges)} slabs, {workers} workers, {levels} levels", flush=True)
		t0 = time.time()
		base = level_arrays[first_filled_level]

		def _write_slab(slab_range, ci=ci, base=base):
			z_off, z_end = slab_range
			base[zs0 + z_off:zs0 + z_end, ys0:ys1, xs0:xs1] = a[ci, zs0 + z_off:zs0 + z_end, ys0:ys1, xs0:xs1]

		_run_with_progress(slab_ranges, _write_slab, workers=workers, prefix=f"{TAG} {ch} L{first_filled_level}")

		# --- Phase 2: Build coarser pyramid levels, one level at a time ---
		# nx/ny are delayed until both base channels exist so their pyramid can
		# be built through the second-order normal moment representation.
		if normal_pair_channels and ch in {"nx", "ny"}:
			print(f"{TAG} {ch}: delaying coarser levels for paired normal pyramid", flush=True)
		else:
			build_scalar_omezarr_pyramid(out_path, first_filled_level, levels, workers=workers, label=ch,
										 zero_overrides=ch == "grad_mag")

		# --- Phase 3: Upsample to fill levels below first_filled_level ---
		for lv in range(first_filled_level - 1, -1, -1):
			_upsample_level(src=level_arrays[lv + 1], dst=level_arrays[lv], chunk=chunk, workers=workers,
							prefix=f"{TAG} {ch} L{lv} (upsample)")

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
	if normal_pair_channels and {"nx", "ny"}.issubset(normal_pair_paths):
		print(f"{TAG} building paired nx/ny normal pyramid", flush=True)
		build_normal_omezarr_pyramid(
			normal_pair_paths["nx"],
			normal_pair_paths["ny"],
			first_filled_level,
			levels,
			workers=workers,
			label="normal",
		)


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
				a = open_zarr(zarr_path)
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
		a = open_zarr(zarr_path)
		params = dict(a.attrs.get("preprocess_params", {}))
		params.setdefault("scaledown", group.sd_fac)
		if vol.crops:
			# Use the first crop for legacy converter compat
			params.setdefault("crop_xyzwhd", list(vol.crops[0]))
		# Channel names for this group
		params["channels"] = group.channels

		# Temporarily set attrs so run() can read them
		# (run() reads from zarr attrs — if missing, write them)
		if "preprocess_params" not in a.attrs:
			try:
				ZarrWriter.open(zarr_path).attrs["preprocess_params"] = params
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
