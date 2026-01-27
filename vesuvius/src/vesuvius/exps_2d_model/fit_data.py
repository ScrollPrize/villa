from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FitData:
	cos: torch.Tensor
	grad_mag: torch.Tensor
	dir0: torch.Tensor
	dir1: torch.Tensor

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
		return FitData(cos=cos_t, grad_mag=mag_t, dir0=dir0_t, dir1=dir1_t)

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
		img_t = img_t[None, :, :, :]
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
) -> FitData:
	p = Path(path)
	if not p.is_dir():
		raise ValueError("FitData currently requires a directory input")

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

	return FitData(
		cos=cos_t,
		grad_mag=mag_t,
		dir0=dir0_t,
		dir1=dir1_t,
	)
