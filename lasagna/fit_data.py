from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from lasagna_volume import LasagnaVolume


@dataclass(frozen=True)
class CorrPoints3D:
	points_xyz_winda: torch.Tensor  # (K, 4) — x, y, z, winda in fullres (winda = depth index from d.tif)
	collection_idx: torch.Tensor    # (K,) — integer collection ID per point
	point_ids: torch.Tensor         # (K,) — integer point ID per point


@dataclass(frozen=True)
class FitData3D:
	cos: torch.Tensor              # (1, 1, Z, Y, X) uint8 on GPU
	grad_mag: torch.Tensor         # (1, 1, Z, Y, X) uint8 on GPU
	nx: torch.Tensor               # (1, 1, Z, Y, X) uint8 on GPU — hemisphere-encoded normal x
	ny: torch.Tensor               # (1, 1, Z, Y, X) uint8 on GPU — hemisphere-encoded normal y
	pred_dt: torch.Tensor | None
	corr_points: CorrPoints3D | None
	winding_volume: torch.Tensor | None  # (1, 1, Z, Y, X) float32 on GPU
	origin_fullres: tuple[float, float, float]  # (x0, y0, z0) in fullres voxels
	spacing: tuple[float, float, float]          # (sx, sy, sz) voxel size in fullres units (cos channel)
	channel_spacing: dict[str, tuple[float, float, float]] | None = None  # per-channel override
	source_to_base: float = 1.0                  # source-to-base factor for tifxyz coord conversion
	winding_min: float | None = None     # min valid winding value (from zarr metadata)
	winding_max: float | None = None     # max valid winding value (from zarr metadata)
	grad_mag_scale: float = 255.0                # encoding scale for grad_mag channel
	cuda_gridsample: bool = True                  # use custom CUDA uint8 kernel vs PyTorch F.grid_sample

	@property
	def size(self) -> tuple[int, int, int]:
		"""(Z, Y, X) spatial dimensions."""
		if self.cos.ndim != 5:
			raise ValueError("FitData3D.cos must be (1,1,Z,Y,X)")
		_, _, z, y, x = self.cos.shape
		return int(z), int(y), int(x)

	def _spacing_for(self, channel: str) -> tuple[float, float, float]:
		"""Return spacing for a specific channel."""
		if self.channel_spacing and channel in self.channel_spacing:
			return self.channel_spacing[channel]
		return self.spacing

	def _size_of(self, t: torch.Tensor | None) -> tuple[int, int, int]:
		"""Return (Z, Y, X) from a (1,1,Z,Y,X) tensor."""
		if t is None:
			return self.size
		return int(t.shape[2]), int(t.shape[3]), int(t.shape[4])

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor) -> "FitData3D":
		"""Sample at fullres positions.

		xyz_fullres: (D, H, W, 3) where last dim is (x, y, z) in fullres coords.
		Returns FitData3D with float32 decoded values in (1, 1, D, H, W).
		Uses custom CUDA uint8 kernel when cuda_gridsample=True, else PyTorch F.grid_sample.
		"""
		if self.cuda_gridsample:
			return self._grid_sample_cuda(xyz_fullres)
		return self._grid_sample_torch(xyz_fullres)

	def _grid_sample_cuda(self, xyz_fullres: torch.Tensor) -> "FitData3D":
		import importlib.util, os
		_spec = importlib.util.spec_from_file_location(
			"grid_sample_3d_u8",
			os.path.join(os.path.dirname(__file__), "grid_sample_3d_u8.py"),
		)
		_mod = importlib.util.module_from_spec(_spec)
		_spec.loader.exec_module(_mod)
		grid_sample_3d_u8 = _mod.grid_sample_3d_u8

		dev = xyz_fullres.device
		offset = torch.tensor(self.origin_fullres, dtype=torch.float32, device=dev)

		def _gs(t: torch.Tensor | None, decode, channel: str) -> torch.Tensor | None:
			if t is None:
				return None
			sp = self._spacing_for(channel)
			inv_scale = torch.tensor([1.0 / s for s in sp], dtype=torch.float32, device=dev)
			vol = t.squeeze(0)  # (1, Z, Y, X) — C=1
			out_u8 = grid_sample_3d_u8(vol, xyz_fullres, offset, inv_scale)  # (1, D, H, W) u8
			return decode(out_u8.float()).unsqueeze(0)  # (1, 1, D, H, W) float32

		return FitData3D(
			cos=_gs(self.cos, lambda t: t / 255.0, "cos"),
			grad_mag=_gs(self.grad_mag, lambda t: t / self.grad_mag_scale, "grad_mag"),
			nx=_gs(self.nx, lambda t: (t - 128.0) / 127.0, "nx"),
			ny=_gs(self.ny, lambda t: (t - 128.0) / 127.0, "ny"),
			pred_dt=_gs(self.pred_dt, lambda t: t.sqrt(), "pred_dt"),
			corr_points=self.corr_points,
			winding_volume=self.winding_volume,
			origin_fullres=self.origin_fullres,
			spacing=self.spacing,
			channel_spacing=self.channel_spacing,
			source_to_base=self.source_to_base,
			winding_min=self.winding_min,
			winding_max=self.winding_max,
			grad_mag_scale=self.grad_mag_scale,
			cuda_gridsample=self.cuda_gridsample,
		)

	def _grid_sample_torch(self, xyz_fullres: torch.Tensor) -> "FitData3D":
		def _make_grid(channel: str, t: torch.Tensor | None) -> torch.Tensor:
			sp = self._spacing_for(channel)
			Z, Y, X = self._size_of(t)
			g = xyz_fullres.clone()
			g[..., 0] = (g[..., 0] - self.origin_fullres[0]) / sp[0]
			g[..., 1] = (g[..., 1] - self.origin_fullres[1]) / sp[1]
			g[..., 2] = (g[..., 2] - self.origin_fullres[2]) / sp[2]
			g[..., 0] = g[..., 0] / max(1, X - 1) * 2 - 1
			g[..., 1] = g[..., 1] / max(1, Y - 1) * 2 - 1
			g[..., 2] = g[..., 2] / max(1, Z - 1) * 2 - 1
			return g

		def _gs(t: torch.Tensor | None, decode, channel: str) -> torch.Tensor | None:
			if t is None:
				return None
			grid_5d = _make_grid(channel, t).unsqueeze(0)
			t_f = decode(t.float())
			return F.grid_sample(t_f, grid_5d, mode="bilinear", padding_mode="zeros", align_corners=True)

		return FitData3D(
			cos=_gs(self.cos, lambda t: t / 255.0, "cos"),
			grad_mag=_gs(self.grad_mag, lambda t: t / self.grad_mag_scale, "grad_mag"),
			nx=_gs(self.nx, lambda t: (t - 128.0) / 127.0, "nx"),
			ny=_gs(self.ny, lambda t: (t - 128.0) / 127.0, "ny"),
			pred_dt=_gs(self.pred_dt, lambda t: t.sqrt(), "pred_dt"),
			corr_points=self.corr_points,
			winding_volume=self.winding_volume,
			origin_fullres=self.origin_fullres,
			spacing=self.spacing,
			channel_spacing=self.channel_spacing,
			source_to_base=self.source_to_base,
			winding_min=self.winding_min,
			winding_max=self.winding_max,
			grad_mag_scale=self.grad_mag_scale,
			cuda_gridsample=self.cuda_gridsample,
		)


def load_winding_volume(
	*,
	path: str,
	device: torch.device,
	crop: tuple[int, int, int, int, int, int] | None,
	downscale: float,
) -> tuple[torch.Tensor, float, float]:
	"""Load winding volume zarr, apply crop, return ((1,1,Z,Y,X) tensor, min_winding, max_winding)."""
	p = Path(path)
	zsrc = zarr.open(str(p), mode="r")
	wv_scaledown = int(zsrc.attrs.get("scaledown", 1))
	wv_min = float(zsrc.attrs.get("min_winding", 1.0))
	wv_max = float(zsrc.attrs.get("max_winding", 1.0))
	ds_i = max(1, int(round(downscale)))

	Z_all, Y_all, X_all = (int(v) for v in zsrc.shape)

	if crop is not None:
		x0, y0, z0, cw, ch, cd = (int(v) for v in crop)
		x0v = max(0, x0 // ds_i)
		y0v = max(0, y0 // ds_i)
		z0v = max(0, z0 // ds_i)
		x1v = min(X_all, (x0 + cw + ds_i - 1) // ds_i)
		y1v = min(Y_all, (y0 + ch + ds_i - 1) // ds_i)
		z1v = min(Z_all, (z0 + cd + ds_i - 1) // ds_i)
	else:
		x0v, y0v, z0v = 0, 0, 0
		x1v, y1v, z1v = X_all, Y_all, Z_all

	print(f"[fit_data] load_winding_volume: zarr shape=({Z_all},{Y_all},{X_all}) "
		  f"scaledown={wv_scaledown} winding=[{wv_min:.2f}, {wv_max:.2f}] "
		  f"reading z=[{z0v}:{z1v}] y=[{y0v}:{y1v}] x=[{x0v}:{x1v}]",
		  flush=True)

	arr = np.asarray(zsrc[z0v:z1v, y0v:y1v, x0v:x1v])
	print(f"[fit_data] load_winding_volume: loaded min={float(arr.min()):.3f} max={float(arr.max()):.3f}",
		  flush=True)
	t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
	return t.unsqueeze(0).unsqueeze(0), wv_min, wv_max  # (1, 1, Z, Y, X)


def auto_crop_for_mesh(
	mesh_bbox: tuple[float, float, float, float, float, float],
	volume_extent_fullres: tuple[int, int, int],
	margin: float = 3.0,
) -> tuple[int, int, int, int, int, int]:
	"""Compute crop = margin x mesh extent, centered on mesh, clamped to volume.

	Returns (x0, y0, z0, w, h, d) in fullres voxels.
	"""
	x_min, y_min, z_min, x_max, y_max, z_max = mesh_bbox
	vol_x, vol_y, vol_z = volume_extent_fullres

	cx = (x_min + x_max) / 2.0
	cy = (y_min + y_max) / 2.0
	cz = (z_min + z_max) / 2.0

	ex = max((x_max - x_min) * margin / 2.0, 100.0)
	ey = max((y_max - y_min) * margin / 2.0, 100.0)
	ez = max((z_max - z_min) * margin / 2.0, 100.0)

	x0 = max(0, int(cx - ex))
	y0 = max(0, int(cy - ey))
	z0 = max(0, int(cz - ez))
	x1 = min(vol_x, int(cx + ex))
	y1 = min(vol_y, int(cy + ey))
	z1 = min(vol_z, int(cz + ez))

	return (x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


def erode_grad_mag(data: FitData3D, radius: int) -> None:
	"""Erode grad_mag validity mask in-place by dilating the invalid (==0) region."""
	if radius <= 0:
		return
	t_gm = data.grad_mag
	if t_gm is None:
		return
	ks = 2 * radius + 1
	invalid_u8 = (t_gm == 0).byte()
	Z = t_gm.shape[2]
	for zi in range(0, Z, 16):
		ze = min(zi + 16, Z)
		z0 = max(zi - radius, 0)
		z1 = min(ze + radius, Z)
		chunk_f = invalid_u8[:, :, z0:z1].float()
		# Pad volume boundaries with 1.0 (=invalid) so erosion works from edges
		pad_z_before = radius - (zi - z0)
		pad_z_after = radius - (z1 - ze)
		chunk_f = F.pad(chunk_f, (radius, radius, radius, radius, pad_z_before, pad_z_after), value=1.0)
		dilated = F.max_pool3d(chunk_f, kernel_size=ks, stride=1, padding=0)
		t_gm[:, :, zi:ze][dilated > 0.5] = 0
	del invalid_u8


def load_3d_for_model(
	*, path: str, device: torch.device, model: object,
	blur_sigma: float = 0.0,
	erode_valid_mask: int = 0,
	cuda_gridsample: bool = True,
) -> FitData3D:
	"""Load 3D data auto-cropped around model mesh bbox. Optionally blurs."""
	with torch.no_grad():
		xyz = model._grid_xyz()
		mesh_bbox = (xyz[..., 0].min().item(), xyz[..., 1].min().item(), xyz[..., 2].min().item(),
					 xyz[..., 0].max().item(), xyz[..., 1].max().item(), xyz[..., 2].max().item())
	print(f"[fit_data] mesh bbox: "
		  f"min=({mesh_bbox[0]:.0f},{mesh_bbox[1]:.0f},{mesh_bbox[2]:.0f}) "
		  f"max=({mesh_bbox[3]:.0f},{mesh_bbox[4]:.0f},{mesh_bbox[5]:.0f})", flush=True)
	prep = get_preprocessed_params(path)
	crop = auto_crop_for_mesh(mesh_bbox, prep["volume_extent_fullres"])
	print(f"[fit_data] auto-crop: x={crop[0]} y={crop[1]} z={crop[2]} "
		  f"w={crop[3]} h={crop[4]} d={crop[5]}", flush=True)
	data = load_3d(path=path, device=device, crop=crop, cuda_gridsample=cuda_gridsample)
	if blur_sigma > 0:
		blur_3d(data, sigma=blur_sigma)
		print(f"[fit_data] blurred data sigma={blur_sigma}", flush=True)
	if erode_valid_mask > 0:
		erode_grad_mag(data, radius=erode_valid_mask)
		print(f"[fit_data] eroded valid mask by {erode_valid_mask} voxels", flush=True)
	return data


def _blur_normals_tensor(data: FitData3D, _blur_separable) -> None:
	"""Smooth nx/ny in-place using sign-invariant outer-product tensor representation."""
	nx_t = data.nx
	ny_t = data.ny
	if nx_t is None or ny_t is None:
		return

	Z, Y, X = nx_t.shape[2], nx_t.shape[3], nx_t.shape[4]
	dev = nx_t.device

	def _decode(t, zi, ze):
		return (t[:, :, zi:ze].float() - 128.0) / 127.0

	def _blur_product_to_cpu(chunk_fn):
		"""Build product volume from uint8 chunks, blur on GPU, return float16 on CPU."""
		vol = torch.empty(1, 1, Z, Y, X, dtype=torch.float32, device=dev)
		for zi in range(0, Z, 16):
			ze = min(zi + 16, Z)
			vol[:, :, zi:ze] = chunk_fn(zi, ze)
		_blur_separable(vol)
		result = vol.half().cpu()
		del vol
		return result

	def _nz(zi, ze):
		nx_c = _decode(nx_t, zi, ze)
		ny_c = _decode(ny_t, zi, ze)
		return torch.sqrt((1.0 - nx_c * nx_c - ny_c * ny_c).clamp(min=0.0))

	# Compute and blur 6 outer-product tensor components one at a time.
	# Products are built from uint8 in chunks (no full float32 decoded volumes).
	# Each blurred result is stored as float16 on CPU to free GPU memory.
	xx = _blur_product_to_cpu(lambda zi, ze: _decode(nx_t, zi, ze) ** 2)
	xy = _blur_product_to_cpu(lambda zi, ze: _decode(nx_t, zi, ze) * _decode(ny_t, zi, ze))
	yy = _blur_product_to_cpu(lambda zi, ze: _decode(ny_t, zi, ze) ** 2)
	xz = _blur_product_to_cpu(lambda zi, ze: _decode(nx_t, zi, ze) * _nz(zi, ze))
	yz = _blur_product_to_cpu(lambda zi, ze: _decode(ny_t, zi, ze) * _nz(zi, ze))
	zz = _blur_product_to_cpu(lambda zi, ze: _nz(zi, ze) ** 2)

	# Extract dominant eigenvector per Z-slice via power iteration.
	# Avoids cuSOLVER eigh which has huge workspace / internal errors for large batches.
	for zi in range(Z):
		# Load tensor components from CPU to GPU as float32
		c0 = torch.stack([xx[0, 0, zi].reshape(-1), xy[0, 0, zi].reshape(-1), xz[0, 0, zi].reshape(-1)], dim=-1).to(dev, dtype=torch.float32)
		c1 = torch.stack([xy[0, 0, zi].reshape(-1), yy[0, 0, zi].reshape(-1), yz[0, 0, zi].reshape(-1)], dim=-1).to(dev, dtype=torch.float32)
		c2 = torch.stack([xz[0, 0, zi].reshape(-1), yz[0, 0, zi].reshape(-1), zz[0, 0, zi].reshape(-1)], dim=-1).to(dev, dtype=torch.float32)
		# Power iteration: v ← M v / |M v|, 8 iters converges for 3x3 with separated eigenvalues
		v = c0  # (N, 3) — initialize with first column
		for _ in range(8):
			Mv = v[:, 0:1] * c0 + v[:, 1:2] * c1 + v[:, 2:3] * c2  # (N, 3)
			v = Mv / (Mv.norm(dim=-1, keepdim=True) + 1e-12)
		# Hemisphere convention: flip if nz < 0
		flip = (v[:, 2] < 0).unsqueeze(-1)
		v = torch.where(flip, -v, v)
		# Re-encode to uint8 directly into data tensors
		nx_t[0, 0, zi] = (v[:, 0].view(Y, X) * 127.0 + 128.0).round().clamp(0, 255).to(torch.uint8)
		ny_t[0, 0, zi] = (v[:, 1].view(Y, X) * 127.0 + 128.0).round().clamp(0, 255).to(torch.uint8)


def blur_3d(data: FitData3D, sigma: float) -> None:
	"""Apply separable 3D Gaussian blur in-place to all uint8 data channels."""
	radius = int(math.ceil(2 * sigma))
	ks = 2 * radius + 1
	# 1D Gaussian kernel
	x = torch.arange(ks, dtype=torch.float32, device=data.cos.device) - radius
	k1d = torch.exp(-0.5 * (x / sigma) ** 2)
	k1d = k1d / k1d.sum()

	def _blur_separable(t: torch.Tensor) -> torch.Tensor:
		"""Separable 3D Gaussian blur with chunked conv3d to limit peak memory."""
		v = t  # (1, 1, Z, Y, X) float32 — modified in-place
		Z, Y, X = v.shape[2], v.shape[3], v.shape[4]
		kz = k1d.view(1, 1, ks, 1, 1)
		ky = k1d.view(1, 1, 1, ks, 1)
		kx = k1d.view(1, 1, 1, 1, ks)
		# Z-blur: iterate over Y-chunks (each Y-row is independent along Z)
		for yi in range(0, Y, 64):
			ye = min(yi + 64, Y)
			s = v[:, :, :, yi:ye, :].contiguous()
			s = F.conv3d(F.pad(s, (0, 0, 0, 0, radius, radius), mode='reflect'), kz)
			v[:, :, :, yi:ye, :] = s
		# Y+X blur: iterate over Z-chunks (Y and X kernels don't cross Z)
		for zi in range(0, Z, 16):
			ze = min(zi + 16, Z)
			s = v[:, :, zi:ze, :, :].contiguous()
			s = F.conv3d(F.pad(s, (0, 0, radius, radius, 0, 0), mode='reflect'), ky)
			s = F.conv3d(F.pad(s, (radius, radius, 0, 0, 0, 0), mode='reflect'), kx)
			v[:, :, zi:ze, :, :] = s
		return v

	def _blur_to_u8(src_u8: torch.Tensor) -> None:
		"""Blur float copy of src_u8 in-place and write back as uint8."""
		blurred = _blur_separable(src_u8.float())
		blurred.round_().clamp_(0, 255)
		Z = blurred.shape[2]
		for zi in range(0, Z, 16):
			ze = min(zi + 16, Z)
			src_u8[:, :, zi:ze] = blurred[:, :, zi:ze].byte()
		del blurred

	# Blur scalar channels
	t_cos = data.cos
	if t_cos is not None:
		_blur_to_u8(t_cos)

	t_gm = data.grad_mag
	if t_gm is not None:
		# grad_mag == 0 encodes invalid voxels; save mask before blur (uint8 to save memory)
		invalid_u8 = (t_gm == 0).byte()
		_blur_to_u8(t_gm)
		# Dilate invalid region by blur radius, chunked along Z to limit memory
		Z = t_gm.shape[2]
		for zi in range(0, Z, 16):
			ze = min(zi + 16, Z)
			z0 = max(zi - radius, 0)
			z1 = min(ze + radius, Z)
			chunk_f = invalid_u8[:, :, z0:z1].float()
			dilated = F.max_pool3d(chunk_f, kernel_size=ks, stride=1, padding=radius)
			out_start = zi - z0
			out_end = out_start + (ze - zi)
			t_gm[:, :, zi:ze][dilated[:, :, out_start:out_end] > 0.5] = 0
		del invalid_u8

	# Blur normals with sign-invariant tensor method
	_blur_normals_tensor(data, _blur_separable)


def get_preprocessed_params(path: str) -> dict:
	"""Probe .lasagna.json metadata for scaledown and volume extent.

	Returns dict with keys 'scaledown', 'volume_extent_fullres', 'source_to_base'.
	volume_extent_fullres is in base (VC3D) coordinates.
	scaledown is the finest channel scaledown relative to source volume.
	"""
	vol = LasagnaVolume.load(path)
	s2b = vol.source_to_base
	# Use finest-resolution group to determine volume extent
	min_sd = min(g.scaledown for g in vol.groups.values())
	# Find a group at finest resolution, open its zarr to get spatial dims
	for g in vol.groups.values():
		if g.scaledown == min_sd:
			zarr_path = str(vol.path.parent / g.zarr_path)
			zsrc = zarr.open(zarr_path, mode="r")
			if not isinstance(zsrc, zarr.Array):
				raise ValueError(f"expected zarr.Array at {zarr_path}, got {type(zsrc)}")
			C, Z, Y, X = (int(v) for v in zsrc.shape)
			# Extent in source coords, then scale to base (VC3D) coords
			volume_extent_fullres = (
				int(X * min_sd * s2b),
				int(Y * min_sd * s2b),
				int(Z * min_sd * s2b),
			)
			return {
				"scaledown": float(min_sd),
				"volume_extent_fullres": volume_extent_fullres,
				"source_to_base": s2b,
			}
	raise ValueError(f"no groups in {path}")


def load_3d(
	*,
	path: str,
	device: torch.device,
	crop: tuple[int, int, int, int, int, int] | None = None,
	cuda_gridsample: bool = True,
) -> FitData3D:
	"""Load 3D sub-volume from .lasagna.json manifest.

	crop: (x0, y0, z0, w, h, d) in source-volume voxel coords, or None for full volume.
	"""
	vol = LasagnaVolume.load(path)
	gmag_enc = vol.grad_mag_encode_scale
	s2b = vol.source_to_base

	# Resolve which channels are available
	all_ch = vol.all_channels()
	req = ["cos", "grad_mag", "nx", "ny"]
	miss = [k for k in req if k not in all_ch]
	if miss:
		raise ValueError(f"lasagna volume missing required channels: {miss}; available={all_ch}")

	def _read_channel(name: str) -> torch.Tensor:
		"""Read a single channel from its group zarr."""
		group, ch_idx = vol.channel_group(name)
		zarr_path = str(vol.path.parent / group.zarr_path)
		zsrc = zarr.open(zarr_path, mode="r")
		if not isinstance(zsrc, zarr.Array):
			raise ValueError(f"expected zarr.Array at {zarr_path}, got {type(zsrc)}")
		shape = tuple(int(v) for v in zsrc.shape)
		if len(shape) != 4:
			raise ValueError(f"expected 4D CZYX zarr at {zarr_path}, got shape={shape}")

		# Full base-to-zarr factor: crop is in base coords
		ds_i = int(round(group.scaledown * s2b))
		C, Z_all, Y_all, X_all = shape

		if crop is not None:
			x0, y0, z0, cw, ch, cd = (int(v) for v in crop)
			x0v = max(0, x0 // ds_i)
			y0v = max(0, y0 // ds_i)
			z0v = max(0, z0 // ds_i)
			x1v = min(X_all, (x0 + cw + ds_i - 1) // ds_i)
			y1v = min(Y_all, (y0 + ch + ds_i - 1) // ds_i)
			z1v = min(Z_all, (z0 + cd + ds_i - 1) // ds_i)
		else:
			x0v, y0v, z0v = 0, 0, 0
			x1v, y1v, z1v = X_all, Y_all, Z_all

		print(f"[fit_data] read {name} from {group.zarr_path} ch_idx={ch_idx} "
			  f"ds_base={ds_i} (sd={group.scaledown}*s2b={s2b}) shape={shape} "
			  f"z=[{z0v}:{z1v}] y=[{y0v}:{y1v}] x=[{x0v}:{x1v}]", flush=True)

		a = np.asarray(zsrc[ch_idx, z0v:z1v, y0v:y1v, x0v:x1v])
		t = torch.from_numpy(a).to(device=device, dtype=torch.uint8)
		return t.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

	cos_t = _read_channel("cos")
	mag_t = _read_channel("grad_mag")
	nx_t = _read_channel("nx")
	ny_t = _read_channel("ny")
	pred_dt_t = _read_channel("pred_dt") if "pred_dt" in all_ch else None

	# Build per-channel spacing in base (VC3D) coordinates.
	# spacing = channel_scaledown * source_to_base  (base voxels per zarr voxel)
	cos_group, _ = vol.channel_group("cos")
	primary_sd = float(cos_group.scaledown) * s2b
	primary_spacing = (primary_sd, primary_sd, primary_sd)

	channel_spacing: dict[str, tuple[float, float, float]] = {}
	for name in ["cos", "grad_mag", "nx", "ny"] + (["pred_dt"] if pred_dt_t is not None else []):
		g, _ = vol.channel_group(name)
		sd = float(g.scaledown) * s2b
		channel_spacing[name] = (sd, sd, sd)

	# Origin in base (VC3D) coords
	if crop is not None:
		x0, y0, z0 = (int(v) for v in crop[:3])
		# Align to finest resolution grid (in base coords)
		finest_sd = min(g.scaledown for g in vol.groups.values())
		base_step = finest_sd * s2b
		origin_fullres = (
			float(int(x0 / base_step) * base_step),
			float(int(y0 / base_step) * base_step),
			float(int(z0 / base_step) * base_step),
		)
	else:
		origin_fullres = (0.0, 0.0, 0.0)

	print(f"[fit_data] load_3d: origin={origin_fullres} primary_spacing={primary_spacing} "
		  f"source_to_base={s2b}", flush=True)

	return FitData3D(
		cos=cos_t,
		grad_mag=mag_t,
		nx=nx_t,
		ny=ny_t,
		pred_dt=pred_dt_t,
		corr_points=None,
		winding_volume=None,
		origin_fullres=origin_fullres,
		spacing=primary_spacing,
		channel_spacing=channel_spacing,
		source_to_base=vol.source_to_base,
		grad_mag_scale=gmag_enc,
		cuda_gridsample=cuda_gridsample,
	)
