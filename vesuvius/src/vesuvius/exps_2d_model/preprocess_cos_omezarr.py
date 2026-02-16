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
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	z_step: int,
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

	z0, z1, y0, y1, x0, x1 = _crop_xyzwhd_bounds(shape_zyx=sh, crop_xyzwhd=crop_xyzwhd)
	nz = z1 - z0
	ny = y1 - y0
	nx = x1 - x0
	if nz <= 0 or ny <= 0 or nx <= 0:
		raise ValueError(f"empty crop: x=[{x0},{x1}) y=[{y0},{y1}) z=[{z0},{z1}) in shape={sh}")

	if downscale_xy <= 0:
		raise ValueError("downscale_xy must be >= 1")
	z_step_raw = max(1, int(z_step))
	z_step_eff = int(z_step_raw) * int(max(1, int(downscale_xy)))
	z_sel = list(range(int(z0), int(z1), int(z_step_eff)))
	if len(z_sel) <= 0:
		raise ValueError(f"empty z selection after downscale: z=[{z0},{z1}) step={z_step_eff}")
	proc_z = int(len(z_sel))

	out_z = _ds_size(int(sh[0]), int(z_step_eff))
	out_y = _ds_size(int(sh[1]), downscale_xy)
	out_x = _ds_size(int(sh[2]), downscale_xy)
	out_z0 = _ds_index(int(z0), int(z_step_eff))
	out_y0 = _ds_index(int(y0), int(downscale_xy))
	out_x0 = _ds_index(int(x0), int(downscale_xy))

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

	arr = zarr.open(
		str(output_path),
		mode="w",
		shape=(5, int(out_z), int(out_y), int(out_x)),
		chunks=(1, max(1, int(chunk_z)), min(int(out_y), max(1, int(chunk_yx))), min(int(out_x), max(1, int(chunk_yx)))),
		dtype=np.uint8,
		fill_value=0,
		dimension_separator="/",
	)
	arr.attrs["preprocess_params"] = {
		"downscale_xy": int(downscale_xy),
		"z_step": int(z_step_raw),
		"z_step_eff": int(z_step_eff),
		"processed_z_slices": int(proc_z),
		"crop_xyzwhd": [int(x0), int(y0), int(z0), int(nx), int(ny), int(nz)],
		"output_full_scaled": True,
		"channels": ["cos", "grad_mag", "dir0", "dir1", "valid"],
	}

	print(
		f"[preprocess_cos_omezarr] input={input_path} crop_xyzwhd=({x0},{y0},{z0},{nx},{ny},{nz}) "
		f"z_step={z_step_raw} z_step_eff={z_step_eff} proc_z={proc_z} out_shape_full={(out_z, out_y, out_x)} in_shape={sh} "
		f"-> out={output_path} out_shape={(5, out_z, out_y, out_x)} dtype=uint8"
	)
	t0 = time.time()
	t_read_sum = 0.0
	t_infer_sum = 0.0
	t_write_sum = 0.0
	cz = max(1, int(chunk_z))
	raw_chunks = getattr(a_in, "chunks", None)
	read_chunk_z = int(raw_chunks[0]) if isinstance(raw_chunks, tuple) and len(raw_chunks) >= 1 else cz
	read_chunk_z = max(1, int(read_chunk_z))

	read0 = (int(z0) // int(read_chunk_z)) * int(read_chunk_z)
	done = 0
	for zr0 in range(read0, int(z1), int(read_chunk_z)):
		zr1 = min(int(z1), int(zr0) + int(read_chunk_z))
		if zr1 <= int(z0):
			continue
		zlo = max(int(z0), int(zr0))
		zhi = int(zr1)
		z_keep = [zz for zz in range(zlo, zhi) if ((zz - int(z0)) % int(z_step_eff)) == 0]
		if len(z_keep) <= 0:
			continue
		idx_keep = np.asarray([int(zz - int(zr0)) for zz in z_keep], dtype=np.int64)
		t_read0 = time.time()
		raw_chunk_np = np.asarray(a_in[int(zr0):int(zr1), y0:y1, x0:x1])
		raw_blk_np = raw_chunk_np[idx_keep, :, :]
		if raw_blk_np.dtype == np.uint16:
			raw_blk_np = (raw_blk_np // 257).astype(np.uint8)
		raw_blk = torch.from_numpy(raw_blk_np.astype(np.float32)).to(device=torch_device)
		if raw_blk.numel() > 0:
			mx = raw_blk.amax(dim=(1, 2), keepdim=True)
			raw_blk = torch.where(mx > 0.0, raw_blk / mx, raw_blk)
		raw_blk = raw_blk[:, None, :, :]
		t_read_sum += float(time.time() - t_read0)

		for bi, zz in enumerate(z_keep):
			raw_i = raw_blk[bi : bi + 1]
			if torch_device.type == "cuda":
				torch.cuda.synchronize(torch_device)
			t_inf0 = time.time()
			with torch.no_grad():
				pred_i = unet_infer_tiled(
					model,
					raw_i,
					tile_size=int(tile_size),
					overlap=int(overlap),
					border=int(border),
				)
			if torch_device.type == "cuda":
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
			grad_mag_u8 = np.clip(grad_mag[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
			dir0_u8 = np.clip(dir0[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
			dir1_u8 = np.clip(dir1[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
			t_wr0 = time.time()
			oi = int(zz) // int(z_step_eff)
			y1w = min(int(out_y), int(out_y0) + int(cos_u8.shape[0]))
			x1w = min(int(out_x), int(out_x0) + int(cos_u8.shape[1]))
			if oi >= 0 and oi < int(out_z) and y1w > int(out_y0) and x1w > int(out_x0):
				ys = int(out_y0)
				xs = int(out_x0)
				yh = int(y1w - ys)
				xw = int(x1w - xs)
				arr[0, oi, ys:y1w, xs:x1w] = cos_u8[:yh, :xw]
				arr[1, oi, ys:y1w, xs:x1w] = grad_mag_u8[:yh, :xw]
				arr[2, oi, ys:y1w, xs:x1w] = dir0_u8[:yh, :xw]
				arr[3, oi, ys:y1w, xs:x1w] = dir1_u8[:yh, :xw]
				arr[4, oi, ys:y1w, xs:x1w] = 1
			t_write_sum += float(time.time() - t_wr0)

			done += 1
			elapsed = max(1e-6, float(time.time() - t0))
			per = elapsed / float(done)
			eta = max(0.0, per * float(proc_z - done))
			eta_m = int(eta // 60.0)
			eta_s = int(eta % 60.0)
			bar_w = 30
			fill = int(round((float(done) / float(max(1, proc_z))) * float(bar_w)))
			bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
			print(
				f"\r[preprocess_cos_omezarr] [{bar}] {done}/{proc_z} ({(100.0 * done / max(1, proc_z)):.1f}%) "
				f"eta {eta_m:02d}:{eta_s:02d} read_avg={((1000.0 * t_read_sum) / max(1, done)):.1f}ms "
				f"infer_avg={((1000.0 * t_infer_sum) / max(1, done)):.1f}ms "
				f"write_avg={((1000.0 * t_write_sum) / max(1, done)):.1f}ms (src z={zz})",
				end="",
				flush=True,
			)
	print("", flush=True)
	print(
		f"[preprocess_cos_omezarr] profile: processed_slices={proc_z} output_depth={out_z} "
		f"read_avg={((1000.0 * t_read_sum) / max(1, proc_z)):.2f}ms "
		f"infer_avg={((1000.0 * t_infer_sum) / max(1, proc_z)):.2f}ms "
		f"write_avg={((1000.0 * t_write_sum) / max(1, proc_z)):.2f}ms"
	)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Run tiled UNet cos inference on an OME-Zarr volume and write 8-bit OME-Zarr at 1/4 XY scale."
	)
	p.add_argument("--input", required=True, help="Input OME-Zarr array path (must be Z,Y,X array).")
	p.add_argument("--output", required=True, help="Output OME-Zarr group path.")
	p.add_argument("--unet-checkpoint", required=True, help="UNet checkpoint path (.pt).")
	p.add_argument("--device", default=None, help='Device, e.g. "cuda" or "cpu".')
	p.add_argument("--crop-xyzwhd", "--crop", dest="crop_xyzwhd", type=int, nargs=6, default=None, help="Crop in input coordinates: x y z w h d.")
	p.add_argument("--z-step", type=int, default=10, help="Additional z-step before downscale (default: 10).")
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
		crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd is not None else None,
		z_step=int(args.z_step),
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
