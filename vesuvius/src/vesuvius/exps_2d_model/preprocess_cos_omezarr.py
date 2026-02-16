from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import zarr

from common import load_unet, unet_infer_tiled
from fit_data import _gaussian_blur_nchw
from tiled_infer import _load_omezarr_z_uint8_norm


def _z_range(*, z_size: int, z_start: int | None, z_end: int | None) -> tuple[int, int]:
	za = 0 if z_start is None else int(z_start)
	zb = int(z_size) if z_end is None else int(z_end)
	za = max(0, min(za, int(z_size)))
	zb = max(za, min(zb, int(z_size)))
	return za, zb


def _ds_size(v: int, f: int) -> int:
	# Match interpolate(scale_factor=1/f) floor behavior.
	return max(1, int(v) // int(f))


def _pyrdown2d(arr: np.ndarray, *, factor: int) -> np.ndarray:
	"""Gaussian pyramid downscale using repeated cv2.pyrDown for power-of-2 factors."""
	f = int(factor)
	if f <= 1:
		return arr
	if (f & (f - 1)) != 0:
		raise ValueError("downscale_xy must be a power of 2 for pyramid scaling")
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
	z_start: int | None,
	z_end: int | None,
	tile_size: int,
	overlap: int,
	border: int,
	downscale_xy: int,
	grad_mag_blur_sigma: float,
	dir_blur_sigma: float,
	chunk_z: int,
	chunk_yx: int,
) -> None:
	a_in = zarr.open(str(input_path), mode="r")
	if not hasattr(a_in, "shape"):
		raise ValueError(f"input must point to an OME-Zarr array, got non-array: {input_path}")
	sh = tuple(int(v) for v in a_in.shape)
	if len(sh) != 3:
		raise ValueError(f"input array must be (Z,Y,X), got {sh}")

	z0, z1 = _z_range(z_size=sh[0], z_start=z_start, z_end=z_end)
	nz = z1 - z0
	if nz <= 0:
		raise ValueError(f"empty z-range: [{z0}, {z1}) for input z-size {sh[0]}")

	if downscale_xy <= 0:
		raise ValueError("downscale_xy must be >= 1")

	out_z = nz
	out_y = _ds_size(sh[1], downscale_xy)
	out_x = _ds_size(sh[2], downscale_xy)

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

	out_store = zarr.open_group(str(output_path), mode="w")
	arr = out_store.create_dataset(
		"2",
		shape=(int(out_z), int(out_y), int(out_x)),
		chunks=(max(1, int(chunk_z)), min(int(out_y), max(1, int(chunk_yx))), min(int(out_x), max(1, int(chunk_yx)))),
		dtype=np.uint8,
		overwrite=True,
	)
	out_store.attrs["multiscales"] = [
		{
			"version": "0.4",
			"name": "cos_preprocessed",
			"axes": [
				{"name": "z", "type": "space", "unit": "pixel"},
				{"name": "y", "type": "space", "unit": "pixel"},
				{"name": "x", "type": "space", "unit": "pixel"},
			],
			"datasets": [
				{
					"path": "2",
					"coordinateTransformations": [
						{"type": "scale", "scale": [1.0, float(downscale_xy), float(downscale_xy)]}
					],
				}
			],
		}
	]

	print(
		f"[preprocess_cos_omezarr] input={input_path} z=[{z0},{z1}) in_shape={sh} "
		f"-> out={output_path}/2 out_shape={(out_z, out_y, out_x)} dtype=uint8"
	)
	t0 = time.time()

	for i, zz in enumerate(range(z0, z1)):
		raw = _load_omezarr_z_uint8_norm(path=str(input_path), z=int(zz), crop=None, device=torch_device)
		with torch.no_grad():
			pred = unet_infer_tiled(
				model,
				raw,
				tile_size=int(tile_size),
				overlap=int(overlap),
				border=int(border),
			)
		cos = pred[:, 0:1]
		grad_mag = pred[:, 1:2] if int(pred.shape[1]) > 1 else pred[:, 0:1]
		dir0 = pred[:, 2:3] if int(pred.shape[1]) > 2 else pred[:, 0:1]
		dir1 = pred[:, 3:4] if int(pred.shape[1]) > 3 else pred[:, 0:1]
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
		arr[i, :, :] = cos_u8

		done = int(i) + 1
		elapsed = max(1e-6, float(time.time() - t0))
		per = elapsed / float(done)
		eta = max(0.0, per * float(nz - done))
		eta_m = int(eta // 60.0)
		eta_s = int(eta % 60.0)
		bar_w = 30
		fill = int(round((float(done) / float(max(1, nz))) * float(bar_w)))
		bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
		print(
			f"\r[preprocess_cos_omezarr] [{bar}] {done}/{nz} ({(100.0 * done / max(1, nz)):.1f}%) "
			f"eta {eta_m:02d}:{eta_s:02d} (src z={zz})",
			end="",
			flush=True,
		)
	print("", flush=True)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Run tiled UNet cos inference on an OME-Zarr volume and write 8-bit OME-Zarr at 1/4 XY scale."
	)
	p.add_argument("--input", required=True, help="Input OME-Zarr array path (must be Z,Y,X array).")
	p.add_argument("--output", required=True, help="Output OME-Zarr group path.")
	p.add_argument("--unet-checkpoint", required=True, help="UNet checkpoint path (.pt).")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu".')
	p.add_argument("--z-start", type=int, default=None, help="Inclusive z start. Default: 0.")
	p.add_argument("--z-end", type=int, default=None, help="Exclusive z end. Default: input z size.")
	p.add_argument("--tile-size", type=int, default=2048, help="Tile size.")
	p.add_argument("--overlap", type=int, default=128, help="Tile overlap.")
	p.add_argument("--border", type=int, default=32, help="Tile border discard width.")
	p.add_argument("--downscale-xy", type=int, default=4, help="XY downscale factor for output (default: 4).")
	p.add_argument("--grad-mag-blur-sigma", type=float, default=4.0, help="Gaussian blur sigma on downscaled grad-mag.")
	p.add_argument("--dir-blur-sigma", type=float, default=2.0, help="Gaussian blur sigma on downscaled dir0/dir1.")
	p.add_argument("--chunk-z", type=int, default=32, help="Output z chunk size.")
	p.add_argument("--chunk-yx", type=int, default=32, help="Output y/x chunk size.")
	args = p.parse_args(argv)

	run_preprocess(
		input_path=str(args.input),
		output_path=str(args.output),
		unet_checkpoint=str(args.unet_checkpoint),
		device=args.device,
		z_start=args.z_start,
		z_end=args.z_end,
		tile_size=int(args.tile_size),
		overlap=int(args.overlap),
		border=int(args.border),
		downscale_xy=int(args.downscale_xy),
		grad_mag_blur_sigma=float(args.grad_mag_blur_sigma),
		dir_blur_sigma=float(args.dir_blur_sigma),
		chunk_z=int(args.chunk_z),
		chunk_yx=int(args.chunk_yx),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
