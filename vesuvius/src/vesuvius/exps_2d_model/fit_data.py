from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F

import tiled_infer
from common import load_unet, unet_infer_tiled


@dataclass(frozen=True)
class FitData:
	cos: torch.Tensor
	grad_mag: torch.Tensor
	dir0: torch.Tensor
	dir1: torch.Tensor
	downscale: float = 1.0

	def grid_sample_px(self, *, xy_px: torch.Tensor) -> "FitData":
		"""Sample using pixel xy positions.

		- `xy_px`: (N,H,W,2) with x in [0,W-1], y in [0,H-1].
		- Internally converts to normalized coords for `grid_sample`.
		"""
		if xy_px.ndim != 4 or int(xy_px.shape[-1]) != 2:
			raise ValueError("xy_px must be (N,H,W,2)")
		n, h, w, _c2 = (int(v) for v in xy_px.shape)
		h_img, w_img = self.size
		hd = float(max(1, int(h_img) - 1))
		wd = float(max(1, int(w_img) - 1))

		grid = xy_px.clone()
		grid[..., 0] = (xy_px[..., 0] / wd) * 2.0 - 1.0
		grid[..., 1] = (xy_px[..., 1] / hd) * 2.0 - 1.0

		cos_t = F.grid_sample(self.cos, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		mag_t = F.grid_sample(self.grad_mag, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		dir0_t = F.grid_sample(self.dir0, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		dir1_t = F.grid_sample(self.dir1, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		return FitData(cos=cos_t, grad_mag=mag_t, dir0=dir0_t, dir1=dir1_t, downscale=float(self.downscale))

	@property
	def size(self) -> tuple[int, int]:
		if self.cos.ndim != 4:
			raise ValueError("FitData.cos must be (N,C,H,W)")
		_, _, h, w = self.cos.shape
		return int(h), int(w)


def _to_nchw(img: object) -> torch.Tensor:
	img_t = torch.as_tensor(img)
	if img_t.ndim == 2:
		img_t = img_t[None, None, :, :]
	elif img_t.ndim == 3:
		img_t = img_t[:, None, :, :]
	else:
		raise ValueError(f"unsupported image shape: {tuple(img_t.shape)}")
	return img_t


def _read_tif_float(path: Path, device: torch.device) -> torch.Tensor:
	a = tifffile.imread(str(path))
	t = _to_nchw(a).to(dtype=torch.float32)
	return t.to(device=device)


def load(
	path: str,
	device: torch.device,
	downscale: float = 4.0,
	crop: tuple[int, int, int, int] | None = None,
	unet_checkpoint: str | None = None,
	unet_layer: int | None = None,
	unet_z: int | None = None,
	unet_tile_size: int = 512,
	unet_overlap: int = 128,
	unet_border: int = 0,
	unet_group: str | None = None,
	unet_out_dir_base: str | None = None,
) -> FitData:
	p = Path(path)
	s = str(p)
	is_omezarr = (
		s.endswith(".zarr")
		or s.endswith(".ome.zarr")
		or (".zarr/" in s)
		or (".ome.zarr/" in s)
	)

	if p.is_dir() and not is_omezarr:
		cos_files = sorted(p.glob("*_cos.tif"))
		if len(cos_files) != 1:
			raise ValueError(f"expected exactly one '*_cos.tif' in {p}, found {len(cos_files)}")
		cos_path = cos_files[0]

		base_stem = cos_path.stem
		if base_stem.endswith("_cos"):
			base_stem = base_stem[:-4]

		mag_path = cos_path.with_name(f"{base_stem}_mag.tif")
		dir0_path = cos_path.with_name(f"{base_stem}_dir0.tif")
		dir1_path = cos_path.with_name(f"{base_stem}_dir1.tif")
		missing = [pp.name for pp in (mag_path, dir0_path, dir1_path) if not pp.is_file()]
		if missing:
			raise FileNotFoundError(f"missing required tif(s) in {p}: {', '.join(missing)}")

		cos_t = _read_tif_float(cos_path, device=device)
		mag_t = _read_tif_float(mag_path, device=device)
		dir0_t = _read_tif_float(dir0_path, device=device)
		dir1_t = _read_tif_float(dir1_path, device=device)
	else:
		# Non-directory input: run UNet inference and return predictions as FitData.
		if unet_checkpoint is None:
			raise ValueError("non-directory input requires --unet-checkpoint")

		xywh = crop
		if xywh is None:
			raise ValueError("non-directory input requires --crop x y w h")
		x, y, w_c, h_c = (int(v) for v in xywh)
		ov = max(0, int(unet_overlap))
		b = max(0, int(unet_border))
		pad = ov + b
		# Expanded crop for tiling/blending, then trimmed back to target crop.
		# Include `border` in the padding so the final crop is unaffected by
		# tile-edge discard regions.
		xe = x - pad
		ye = y - pad
		we = w_c + 2 * pad
		he = h_c + 2 * pad
		expanded_crop = (xe, ye, we, he)

		# Load raw 2D slice as (1,1,H,W).
		if is_omezarr:
			raw = tiled_infer._load_omezarr_z_uint8_norm(
				path=str(p),
				z=unet_z,
				crop=expanded_crop,
				device=device,
			)
		else:
			raw = tiled_infer._load_tiff_layer_uint8_norm(
				Path(path),
				layer=unet_layer,
				device=device,
			)
		# Note: TIFF path currently ignores crop in loader; crop via slicing.
		if not is_omezarr:
			_, _, hh, ww = raw.shape
			x0 = max(0, min(xe, ww))
			y0 = max(0, min(ye, hh))
			x1 = max(x0, min(xe + we, ww))
			y1 = max(y0, min(ye + he, hh))
			raw = raw[:, :, y0:y1, x0:x1]

		mdl = load_unet(
			device=device,
			weights=unet_checkpoint,
			strict=True,
			in_channels=1,
			out_channels=4,
			base_channels=32,
			num_levels=6,
			max_channels=1024,
		)
		mdl.eval()
		with torch.no_grad():
			pred = unet_infer_tiled(
				mdl,
				raw,
				tile_size=int(unet_tile_size),
				overlap=int(unet_overlap),
				border=int(unet_border),
			)

		if unet_out_dir_base is not None:
			out_p = Path(unet_out_dir_base) / "unet_pred"
			out_p.mkdir(parents=True, exist_ok=True)
			# Save the extracted slice/crop fed into the UNet.
			raw_np = raw[0, 0].detach().cpu().numpy().astype("float32")
			tifffile.imwrite(str(out_p / "raw.tif"), raw_np, compression="lzw")
			prefix = out_p / "unet"
			pred_np = pred[0].detach().cpu().numpy().astype("float32")
			tifffile.imwrite(str(prefix) + "_cos.tif", pred_np[0], compression="lzw")
			tifffile.imwrite(str(prefix) + "_mag.tif", pred_np[1], compression="lzw")
			tifffile.imwrite(str(prefix) + "_dir0.tif", pred_np[2], compression="lzw")
			tifffile.imwrite(str(prefix) + "_dir1.tif", pred_np[3], compression="lzw")

		# Trim expanded crop back to requested target crop.
		_, _, ph, pw = pred.shape
		x0t = max(0, min(pad, pw))
		y0t = max(0, min(pad, ph))
		x1t = max(x0t, min(x0t + w_c, pw))
		y1t = max(y0t, min(y0t + h_c, ph))
		pred = pred[:, :, y0t:y1t, x0t:x1t]

		cos_t = pred[:, 0:1]
		mag_t = pred[:, 1:2]
		dir0_t = pred[:, 2:3]
		dir1_t = pred[:, 3:4]

		# For UNet input, interpret downscale as post-prediction downscale.
		crop = None

	if crop is not None:
		x, y, w_c, h_c = (int(v) for v in crop)
		_, _, h0, w0 = cos_t.shape
		x0 = max(0, min(x, w0))
		y0 = max(0, min(y, h0))
		x1 = max(x0, min(x + w_c, w0))
		y1 = max(y0, min(y + h_c, h0))
		cos_t = cos_t[:, :, y0:y1, x0:x1]
		mag_t = mag_t[:, :, y0:y1, x0:x1]
		dir0_t = dir0_t[:, :, y0:y1, x0:x1]
		dir1_t = dir1_t[:, :, y0:y1, x0:x1]

	if downscale is not None and float(downscale) > 1.0:
		scale = 1.0 / float(downscale)
		cos_t = F.interpolate(cos_t, scale_factor=scale, mode="bilinear", align_corners=True)
		mag_t = F.interpolate(mag_t, scale_factor=scale, mode="bilinear", align_corners=True)
		dir0_t = F.interpolate(dir0_t, scale_factor=scale, mode="bilinear", align_corners=True)
		dir1_t = F.interpolate(dir1_t, scale_factor=scale, mode="bilinear", align_corners=True)

	return FitData(cos=cos_t, grad_mag=mag_t, dir0=dir0_t, dir1=dir1_t, downscale=float(downscale) if downscale is not None else 1.0)
