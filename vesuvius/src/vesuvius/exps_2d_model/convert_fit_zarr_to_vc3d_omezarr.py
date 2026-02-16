from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import zarr


def _down2(a: np.ndarray) -> np.ndarray:
	return a[::2, ::2, ::2]


def _expand_z_repeat(a: np.ndarray, *, z_repeat: int) -> np.ndarray:
	r = max(1, int(z_repeat))
	if r <= 1:
		return a
	return np.repeat(a, repeats=r, axis=0)


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


def _load_channel_chunked(*, src: zarr.Array, chunk: int, label: str) -> np.ndarray:
	z = int(src.shape[0])
	if z <= 0:
		return np.zeros((0, int(src.shape[1]), int(src.shape[2])), dtype=np.uint8)
	t0 = time.time()
	parts: list[np.ndarray] = []
	for z0 in range(0, z, max(1, int(chunk))):
		z1 = min(z, z0 + max(1, int(chunk)))
		parts.append(np.asarray(src[z0:z1, :, :]).astype(np.uint8, copy=False))
		_print_progress(prefix=label, done=int(z1), total=int(z), t0=float(t0))
	print("", flush=True)
	return np.concatenate(parts, axis=0)


def _write_z_chunks(*, dst: zarr.Array, data: np.ndarray, chunk: int, label: str) -> None:
	z = int(data.shape[0])
	t0 = time.time()
	for z0 in range(0, z, max(1, int(chunk))):
		z1 = min(z, z0 + max(1, int(chunk)))
		dst[z0:z1, :, :] = data[z0:z1, :, :]
		_print_progress(prefix=label, done=int(z1), total=int(z), t0=float(t0))
	print("", flush=True)


def _channels_from_src(*, c_in: int, params: dict) -> list[str]:
	ch = params.get("channels", None)
	if isinstance(ch, list) and all(isinstance(v, str) for v in ch) and len(ch) == int(c_in):
		return [str(v) for v in ch]
	return [f"ch{i}" for i in range(int(c_in))]


def _first_filled_level_from_downscale(*, downscale_xy: int) -> int:
	d = max(1, int(downscale_xy))
	lv = 0
	while d > 1:
		d //= 2
		lv += 1
	return lv


def run(
	*,
	input_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
) -> None:
	a = zarr.open(str(input_path), mode="r")
	if len(tuple(int(v) for v in a.shape)) != 4:
		raise ValueError(f"input must be (C,Z,Y,X), got shape={tuple(a.shape)}")
	if str(a.dtype) != "uint8":
		raise ValueError(f"input dtype must be uint8, got {a.dtype}")

	params = dict(getattr(a, "attrs", {}).get("preprocess_params", {}) or {})
	z_repeat = int(params.get("z_step", 1) or 1)
	downscale_xy = int(params.get("downscale_xy", 1) or 1)
	first_filled_level = _first_filled_level_from_downscale(downscale_xy=downscale_xy)

	c_in, z0, y0, x0 = (int(v) for v in a.shape)
	channels = _channels_from_src(c_in=int(c_in), params=params)
	ch_n = int(len(channels))
	t0 = time.time()

	for ci, ch in enumerate(channels):
		out_path = Path(f"{output_prefix}_{ch}.ome.zarr")
		g = zarr.open_group(str(out_path), mode="w")

		print(f"[convert_fit_zarr_to_vc3d_omezarr] loading channel={ch}", flush=True)
		base = _load_channel_chunked(
			src=a[ci],
			chunk=max(1, int(chunk)),
			label=f"[convert_fit_zarr_to_vc3d_omezarr] load {ch}",
		)
		base = _expand_z_repeat(base, z_repeat=z_repeat)
		base_shape = tuple(int(v) for v in base.shape)

		arrs: dict[int, zarr.Array] = {}
		for lv in range(max(1, int(levels))):
			if lv < int(first_filled_level):
				sh = _shape_mul2(base_shape, int(first_filled_level) - int(lv))
			else:
				sh = _shape_div2(base_shape, int(lv) - int(first_filled_level))
			arrs[lv] = g.create_dataset(
				str(lv),
				shape=sh,
				chunks=(
					min(int(sh[0]), max(1, int(chunk))),
					min(int(sh[1]), max(1, int(chunk))),
					min(int(sh[2]), max(1, int(chunk))),
				),
				dtype=np.uint8,
				overwrite=True,
				fill_value=0,
				dimension_separator="/",
			)

		# Fill only lower levels (>= first_filled_level).
		cur = base
		_write_z_chunks(
			dst=arrs[int(first_filled_level)],
			data=cur,
			chunk=max(1, int(chunk)),
			label=f"[convert_fit_zarr_to_vc3d_omezarr] write {ch}/L{int(first_filled_level)}",
		)
		for lv in range(int(first_filled_level) + 1, max(1, int(levels))):
			cur = _down2(cur)
			_write_z_chunks(
				dst=arrs[lv],
				data=cur,
				chunk=max(1, int(chunk)),
				label=f"[convert_fit_zarr_to_vc3d_omezarr] write {ch}/L{int(lv)}",
			)

		datasets = []
		for lv in range(max(1, int(levels))):
			scale_fac = 2 ** int(lv)
			datasets.append(
				{
					"path": str(lv),
					"coordinateTransformations": [
						{
							"type": "scale",
							"scale": [
								float(scale_fac),
								float(scale_fac),
								float(scale_fac),
							],
						}
					],
				}
			)

		g.attrs["multiscales"] = [
			{
				"version": "0.4",
				"name": str(ch),
				"axes": [
					{"name": "z", "type": "space", "unit": "pixel"},
					{"name": "y", "type": "space", "unit": "pixel"},
					{"name": "x", "type": "space", "unit": "pixel"},
				],
				"datasets": datasets,
			}
		]
		g.attrs["source_preprocess_params"] = params
		g.attrs["vc3d_convert"] = {
			"channel": str(ch),
			"channel_index": int(ci),
			"z_repeat": int(z_repeat),
			"downscale_xy": int(downscale_xy),
			"levels": int(levels),
			"first_filled_level": int(first_filled_level),
		}

		print(
			f"[convert_fit_zarr_to_vc3d_omezarr] channel={ch} "
			f"input=(Z,Y,X)={(z0, y0, x0)} repeated_z={int(base_shape[0])} "
			f"out={str(out_path)} filled_levels={int(first_filled_level)}..{int(levels)-1}"
		)

		done = int(ci) + 1
		elapsed = max(1e-6, float(time.time() - t0))
		per = elapsed / float(done)
		eta = max(0.0, per * float(ch_n - done))
		eta_m = int(eta // 60.0)
		eta_s = int(eta % 60.0)
		bar_w = 30
		fill = int(round((float(done) / float(max(1, ch_n))) * float(bar_w)))
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
	args = p.parse_args(argv)

	run(
		input_path=str(args.input),
		output_prefix=str(args.output_prefix),
		levels=int(args.levels),
		chunk=int(args.chunk),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
