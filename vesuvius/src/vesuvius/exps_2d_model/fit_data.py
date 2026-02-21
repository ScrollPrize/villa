from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import zarr

import tiled_infer
from common import load_unet, unet_infer_tiled
import cli_data


@dataclass(frozen=True)
class PointConstraintsData:
	points_xyz_winda: torch.Tensor
	collection_idx: torch.Tensor
	idx_left: torch.Tensor
	valid_left: torch.Tensor
	idx_right: torch.Tensor
	valid_right: torch.Tensor


@dataclass(frozen=True)
class ConstraintsData:
	points: PointConstraintsData | None = None


@dataclass(frozen=True)
class FitData:
	cos: torch.Tensor
	grad_mag: torch.Tensor
	dir0: torch.Tensor
	dir1: torch.Tensor
	valid: torch.Tensor | None = None
	dir0_y: torch.Tensor | None = None
	dir1_y: torch.Tensor | None = None
	dir0_x: torch.Tensor | None = None
	dir1_x: torch.Tensor | None = None
	downscale: float = 1.0
	constraints: ConstraintsData | None = None

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
		def _gs_opt(t):
			return None if t is None else F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		return FitData(
			cos=cos_t,
			grad_mag=mag_t,
			dir0=dir0_t,
			dir1=dir1_t,
			valid=_gs_opt(self.valid),
			dir0_y=_gs_opt(self.dir0_y),
			dir1_y=_gs_opt(self.dir1_y),
			dir0_x=_gs_opt(self.dir0_x),
			dir1_x=_gs_opt(self.dir1_x),
			downscale=float(self.downscale),
			constraints=self.constraints,
		)

	@property
	def size(self) -> tuple[int, int]:
		if self.cos.ndim != 4:
			raise ValueError("FitData.cos must be (N,C,H,W)")
		_, _, h, w = self.cos.shape
		return int(h), int(w)


def grow_z_from_omezarr_unet(
	*,
	data: FitData,
	cfg: "cli_data.DataConfig",
	unet_z0: int,
	new_z_size: int,
	insert_z: int,
	out_dir_base: str | None,
) -> tuple[FitData, int]:
	"""Expand `data` along Z by loading the newly required slice(s).

	Contract:
	- Only supports growing by 1 slice.
	- Only supports prepend (insert_z==0) or append (insert_z==old_N).
	- Requires OME-Zarr input.
	- For preprocessed zarr input: reloads the full target stack via [`load()`](fit_data.py:201).
	- For raw OME-Zarr input: infers only the missing slice via UNet.
	- Returns (expanded_data, new_unet_z0).
	"""
	old_n = int(data.cos.shape[0])
	new_n = int(new_z_size)
	if new_n != old_n + 1:
		raise ValueError("grow_z: new_z_size must be old_N+1")
	ins = int(insert_z)
	if ins not in (0, old_n):
		raise ValueError("grow_z: only prepend/append supported")

	# Preprocessed zarr path: reload full target stack with the same loader mapping.
	try:
		p_in = Path(str(cfg.input))
		s_in = str(p_in)
		is_omezarr_in = (
			s_in.endswith(".zarr")
			or s_in.endswith(".ome.zarr")
			or (".zarr/" in s_in)
			or (".ome.zarr/" in s_in)
		)
		if is_omezarr_in:
			zsrc_in = zarr.open(str(p_in), mode="r")
			if isinstance(zsrc_in, zarr.Array) and ("preprocess_params" in dict(getattr(zsrc_in, "attrs", {}))):
				z0_new = int(unet_z0)
				if ins == 0:
					z0_new = int(unet_z0) - int(max(1, int(cfg.z_step)))
				d_reload = load(
					path=str(cfg.input),
					device=data.cos.device,
					downscale=float(cfg.downscale),
					crop=cfg.crop,
					unet_checkpoint=str(cfg.unet_checkpoint) if cfg.unet_checkpoint is not None else None,
					unet_layer=cfg.unet_layer,
					unet_z=int(z0_new),
					z_size=int(new_n),
					z_step=int(cfg.z_step),
					unet_tile_size=int(cfg.unet_tile_size),
					unet_overlap=int(cfg.unet_overlap),
					unet_border=int(cfg.unet_border),
					unet_group=cfg.unet_group,
					unet_out_dir_base=out_dir_base,
					grad_mag_blur_sigma=float(cfg.grad_mag_blur_sigma),
					dir_blur_sigma=float(cfg.dir_blur_sigma),
				)
				if int(d_reload.cos.shape[0]) != int(new_n):
					raise RuntimeError(f"grow_z: preprocessed reload returned N={int(d_reload.cos.shape[0])}, expected {int(new_n)}")
				return d_reload, int(z0_new)
	except Exception:
		pass

	if ins == 0:
		unet_z0 = int(unet_z0) - int(max(1, int(cfg.z_step)))
		z_inf = int(unet_z0)
	else:
		z_inf = int(unet_z0) + int(old_n) * int(max(1, int(cfg.z_step)))

	d_new = load(
		path=str(cfg.input),
		device=data.cos.device,
		downscale=float(cfg.downscale),
		crop=cfg.crop,
		unet_checkpoint=str(cfg.unet_checkpoint),
		unet_layer=cfg.unet_layer,
		unet_z=int(z_inf),
		z_size=1,
		z_step=1,
		unet_tile_size=int(cfg.unet_tile_size),
		unet_overlap=int(cfg.unet_overlap),
		unet_border=int(cfg.unet_border),
		unet_group=cfg.unet_group,
		unet_out_dir_base=out_dir_base,
	)
	if int(d_new.cos.shape[0]) != 1:
		raise RuntimeError("grow_z: expected 1 inferred slice")
	def _cat_opt(a, b, dim=0):
		if a is None or b is None:
			return None
		return torch.cat([a, b], dim=dim)
	if ins == 0:
		return (
			FitData(
				cos=torch.cat([d_new.cos, data.cos], dim=0),
				grad_mag=torch.cat([d_new.grad_mag, data.grad_mag], dim=0),
				dir0=torch.cat([d_new.dir0, data.dir0], dim=0),
				dir1=torch.cat([d_new.dir1, data.dir1], dim=0),
				valid=_cat_opt(d_new.valid, data.valid),
				dir0_y=_cat_opt(d_new.dir0_y, data.dir0_y),
				dir1_y=_cat_opt(d_new.dir1_y, data.dir1_y),
				dir0_x=_cat_opt(d_new.dir0_x, data.dir0_x),
				dir1_x=_cat_opt(d_new.dir1_x, data.dir1_x),
				downscale=float(data.downscale),
				constraints=data.constraints,
			),
			int(unet_z0),
		)
	return (
		FitData(
			cos=torch.cat([data.cos, d_new.cos], dim=0),
			grad_mag=torch.cat([data.grad_mag, d_new.grad_mag], dim=0),
			dir0=torch.cat([data.dir0, d_new.dir0], dim=0),
			dir1=torch.cat([data.dir1, d_new.dir1], dim=0),
			valid=_cat_opt(data.valid, d_new.valid),
			dir0_y=_cat_opt(data.dir0_y, d_new.dir0_y),
			dir1_y=_cat_opt(data.dir1_y, d_new.dir1_y),
			dir0_x=_cat_opt(data.dir0_x, d_new.dir0_x),
			dir1_x=_cat_opt(data.dir1_x, d_new.dir1_x),
			downscale=float(data.downscale),
			constraints=data.constraints,
		),
		int(unet_z0),
	)


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


def _gaussian_blur_nchw(*, x: torch.Tensor, sigma: float, kernel_size: int = 21) -> torch.Tensor:
	if x.ndim != 4:
		raise ValueError("gaussian_blur: x must be (N,C,H,W)")
	if float(sigma) <= 0.0:
		return x
	ks = int(kernel_size)
	if ks <= 1:
		return x
	if (ks % 2) == 0:
		ks += 1
	device = x.device
	dtype = x.dtype
	r = ks // 2
	idx = torch.arange(-r, r + 1, device=device, dtype=dtype)
	k = torch.exp(-(idx * idx) / (2.0 * float(sigma) * float(sigma)))
	k = k / (k.sum() + 1e-12)
	kx = k.view(1, 1, 1, ks)
	ky = k.view(1, 1, ks, 1)
	n, c, _h, _w = (int(v) for v in x.shape)
	pad = (r, r, 0, 0)
	y = F.pad(x, pad, mode="reflect")
	y = F.conv2d(y, kx.expand(c, 1, 1, ks), groups=c)
	pad = (0, 0, r, r)
	y = F.pad(y, pad, mode="reflect")
	y = F.conv2d(y, ky.expand(c, 1, ks, 1), groups=c)
	return y


def load(
	path: str,
	device: torch.device,
	downscale: float = 4.0,
	crop: tuple[int, int, int, int] | None = None,
	unet_checkpoint: str | None = None,
	unet_layer: int | None = None,
	unet_z: int | None = None,
	z_size: int = 1,
	z_step: int = 1,
	unet_tile_size: int = 512,
	unet_overlap: int = 128,
	unet_border: int = 0,
	unet_group: str | None = None,
	unet_out_dir_base: str | None = None,
	grad_mag_blur_sigma: float = 0.0,
	dir_blur_sigma: float = 0.0,
) -> FitData:
	p = Path(path)
	s = str(p)
	skip_postprocess = False
	dir0_y_t: torch.Tensor | None = None
	dir1_y_t: torch.Tensor | None = None
	dir0_x_t: torch.Tensor | None = None
	dir1_x_t: torch.Tensor | None = None
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
		valid_t = None
	else:
		# Zarr input can be either raw OME-Zarr (run UNet) or preprocessed fit zarr.
		if is_omezarr:
			zsrc = zarr.open(str(p), mode="r")
		else:
			zsrc = None
		if isinstance(zsrc, zarr.Array) and int(len(zsrc.shape)) == 4 and int(zsrc.shape[0]) >= 4 and ("preprocess_params" in dict(getattr(zsrc, "attrs", {}))):
			params = dict(getattr(zsrc, "attrs", {}).get("preprocess_params", {}) or {})
			channels = [str(v) for v in (params.get("channels", []) or [])]
			if len(channels) <= 0:
				channels = ["cos", "grad_mag", "dir0", "dir1"]

			req_keys = ["downscale_xy", "z_step", "grad_mag_blur_sigma", "dir_blur_sigma", "grad_mag_encode_scale"]
			missing = [k for k in req_keys if k not in params]
			if missing:
				raise ValueError(f"preprocessed zarr missing preprocess_params keys for fit-arg consistency: {missing}")

			ds_meta = float(params.get("downscale_xy", 1.0))
			zs_meta = int(params.get("z_step", 1))
			gmb_meta = float(params.get("grad_mag_blur_sigma", 0.0))
			db_meta = float(params.get("dir_blur_sigma", 0.0))
			gmag_enc = float(params.get("grad_mag_encode_scale"))
			if gmag_enc <= 0.0:
				raise ValueError(f"invalid preprocess_params.grad_mag_encode_scale: {gmag_enc}")
			ds_cli = float(downscale)
			zs_cli = max(1, int(z_step))
			gmb_cli = float(grad_mag_blur_sigma)
			db_cli = float(dir_blur_sigma)
			tol = 1e-6
			bad: list[str] = []
			# CLI defaults mean "not explicitly set" for preprocessed volumes.
			zs_cli_is_default = int(zs_cli) == 1
			gmb_cli_is_default = abs(float(gmb_cli) - 0.0) <= tol
			db_cli_is_default = abs(float(db_cli) - 0.0) <= tol
			if abs(ds_meta - ds_cli) > tol:
				bad.append(f"downscale(meta={ds_meta}, cli={ds_cli})")
			if (not zs_cli_is_default) and int(zs_meta) != int(zs_cli):
				bad.append(f"z_step(meta={zs_meta}, cli={zs_cli})")
			if (not gmb_cli_is_default) and abs(gmb_meta - gmb_cli) > tol:
				bad.append(f"grad_mag_blur_sigma(meta={gmb_meta}, cli={gmb_cli})")
			if (not db_cli_is_default) and abs(db_meta - db_cli) > tol:
				bad.append(f"dir_blur_sigma(meta={db_meta}, cli={db_cli})")
			if bad:
				raise ValueError("preprocessed zarr preprocess_params mismatch vs fit args: " + ", ".join(bad))

			ci = {name: i for i, name in enumerate(channels)}
			need = ["cos", "grad_mag", "dir0", "dir1"]
			miss_need = [k for k in need if k not in ci]
			if miss_need:
				raise ValueError(f"preprocessed zarr missing required channels: {miss_need}; available={channels}")

			shape_czyx = tuple(int(v) for v in zsrc.shape)
			if len(shape_czyx) != 4:
				raise ValueError(f"preprocessed zarr must be CZYX (4D), got shape={shape_czyx}")
			_, z_all, h_all, w_all = shape_czyx
			output_full_scaled = bool(params.get("output_full_scaled", False))
			z_step_eff_meta = int(params.get("z_step_eff", int(zs_meta) * int(max(1, int(ds_meta)))))
			if z_step_eff_meta <= 0:
				raise ValueError(f"invalid preprocess_params.z_step_eff: {z_step_eff_meta}")
			if not output_full_scaled:
				raise ValueError("preprocessed zarr requires preprocess_params.output_full_scaled=true")

			def _ceil_div(a: int, b: int) -> int:
				return (int(a) + int(b) - 1) // int(b)

			x0 = 0
			y0 = 0
			cw = int(w_all)
			ch = int(h_all)
			if crop is not None:
				x0i, y0i, wi, hi = (int(v) for v in crop)
				ds_i = max(1, int(round(float(ds_meta))))
				x0s = max(0, int(x0i))
				y0s = max(0, int(y0i))
				x1s = max(x0s, int(x0i) + max(0, int(wi)))
				y1s = max(y0s, int(y0i) + max(0, int(hi)))
				x0 = int(x0s) // int(ds_i)
				y0 = int(y0s) // int(ds_i)
				x1m = _ceil_div(int(x1s), int(ds_i))
				y1m = _ceil_div(int(y1s), int(ds_i))
				cw = max(1, min(int(x1m), int(w_all)) - int(x0))
				ch = max(1, min(int(y1m), int(h_all)) - int(y0))
			x1 = int(x0) + int(cw)
			y1 = int(y0) + int(ch)

			z0_raw = int(unet_z) if unet_z is not None else 0
			zs = max(1, int(z_size))
			# CLI coords are always full-res; preprocessed spacing along Z is z_step_eff.
			z_req_raw = [int(z0_raw) + int(i) * int(z_step_eff_meta) for i in range(int(zs))]
			z_idx = [int(zz) // int(z_step_eff_meta) for zz in z_req_raw]
			if any((zi < 0 or zi >= int(z_all)) for zi in z_idx):
				raise ValueError(
					"requested z range out of bounds for preprocessed zarr: "
					f"z_req_raw={z_req_raw}, z_idx_loaded={z_idx}, z_max={int(z_all) - 1}, "
					f"z_step_eff_meta={int(z_step_eff_meta)}"
				)

			print(
				"[fit_data] preprocessed zarr load "
				f"shape_czyx={shape_czyx} "
				f"z_req_raw={z_req_raw} "
				f"z_idx_loaded={z_idx} "
				f"z_step_eff_meta={int(z_step_eff_meta)} "
				f"output_full_scaled={int(output_full_scaled)} "
				f"crop_xywh=({int(x0)},{int(y0)},{int(cw)},{int(ch)})",
				flush=True,
			)

			def _read_ch(name: str) -> np.ndarray:
				ci0 = int(ci[name])
				if len(z_idx) <= 1 or all((int(z_idx[i]) == int(z_idx[0]) + i) for i in range(len(z_idx))):
					z_a = int(z_idx[0])
					z_b = int(z_idx[-1]) + 1
					return np.asarray(zsrc[ci0, z_a:z_b, y0:y1, x0:x1])
				return np.stack([np.asarray(zsrc[ci0, int(zi), y0:y1, x0:x1]) for zi in z_idx], axis=0)

			cos_np = _read_ch("cos")
			mag_np = _read_ch("grad_mag")
			dir0_np = _read_ch("dir0")
			dir1_np = _read_ch("dir1")
			valid_np = _read_ch("valid") if "valid" in ci else None

			def _u8_to_t(a: np.ndarray) -> torch.Tensor:
				t = torch.from_numpy(a.astype(np.float32) / 255.0).to(device=device, dtype=torch.float32)
				return t.unsqueeze(1)

			def _u8_to_t_scaled(a: np.ndarray, *, scale: float) -> torch.Tensor:
				s = float(scale)
				if s <= 0.0:
					raise ValueError(f"invalid grad_mag decode scale: {s}")
				t = torch.from_numpy(a.astype(np.float32) / s).to(device=device, dtype=torch.float32)
				return t.unsqueeze(1)

			def _u8_valid_to_t(a: np.ndarray) -> torch.Tensor:
				t = torch.from_numpy((a > 0).astype(np.float32)).to(device=device, dtype=torch.float32)
				return t.unsqueeze(1)

			cos_t = _u8_to_t(cos_np)
			mag_t = _u8_to_t_scaled(mag_np, scale=float(gmag_enc))
			dir0_t = _u8_to_t(dir0_np)
			dir1_t = _u8_to_t(dir1_np)
			valid_t = None if valid_np is None else _u8_valid_to_t(valid_np)
			dir0_y_t = _u8_to_t(_read_ch("dir0_y")) if "dir0_y" in ci else None
			dir1_y_t = _u8_to_t(_read_ch("dir1_y")) if "dir1_y" in ci else None
			dir0_x_t = _u8_to_t(_read_ch("dir0_x")) if "dir0_x" in ci else None
			dir1_x_t = _u8_to_t(_read_ch("dir1_x")) if "dir1_x" in ci else None
			crop = None
			downscale = float(ds_meta)
			skip_postprocess = True
		else:
			# Raw input path: run UNet inference and return predictions as FitData.
			if unet_checkpoint is None:
				raise ValueError("non-directory input requires --unet-checkpoint")

			xywh = crop
			if xywh is None:
				raise ValueError("non-directory input requires --crop x y w h")
			x, y, w_c, h_c = (int(v) for v in xywh)
			crop = (x, y, w_c, h_c)

			z0 = unet_z
			if z0 is None:
				raise ValueError("OME-Zarr inference requires --unet-z")
			zs = max(1, int(z_size))
			zst = max(1, int(z_step))
			# Load raw 2D slice(s) as (N,1,H,W).
			raws: list[torch.Tensor] = []
			show_z_progress = int(zs) > 1
			if show_z_progress:
				print(f"[tiled_infer] loading z slices: 0/{int(zs)}", end="", flush=True)
			for zi in range(zs):
				zv = int(z0) + int(zi) * int(zst)
				if is_omezarr:
					raw_i = tiled_infer._load_omezarr_z_uint8_norm(
						path=str(p),
						z=zv,
						crop=crop,
						device=device,
					)
				else:
					# Non-OME-Zarr paths are single-slice only.
					raise ValueError("z_size>1 requires OME-Zarr input")
				raws.append(raw_i)
				if show_z_progress:
					print(f"\r[tiled_infer] loading z slices: {int(zi) + 1}/{int(zs)}", end="", flush=True)
			if show_z_progress:
				print("", flush=True)
			raw = torch.cat(raws, dim=0)

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
			print(
				f"[tiled_infer] device={str(device)} cuda_available={bool(torch.cuda.is_available())} "
				f"raw_device={str(raw.device)} model_device={str(next(mdl.parameters()).device)}"
			)
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

			cos_t = pred[:, 0:1]
			mag_t = pred[:, 1:2]
			dir0_t = pred[:, 2:3]
			dir1_t = pred[:, 3:4]
			valid_t = None

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
		if valid_t is not None:
			valid_t = valid_t[:, :, y0:y1, x0:x1]

	if (not skip_postprocess) and downscale is not None and float(downscale) > 1.0:
		scale = 1.0 / float(downscale)
		cos_t = F.interpolate(cos_t, scale_factor=scale, mode="bilinear", align_corners=True)
		mag_t = F.interpolate(mag_t, scale_factor=scale, mode="bilinear", align_corners=True)
		dir0_t = F.interpolate(dir0_t, scale_factor=scale, mode="bilinear", align_corners=True)
		dir1_t = F.interpolate(dir1_t, scale_factor=scale, mode="bilinear", align_corners=True)
		if valid_t is not None:
			valid_t = F.interpolate(valid_t, scale_factor=scale, mode="area")

	if (not skip_postprocess) and float(grad_mag_blur_sigma) > 0.0:
		mag_t = _gaussian_blur_nchw(x=mag_t, sigma=float(grad_mag_blur_sigma))
	if (not skip_postprocess) and float(dir_blur_sigma) > 0.0:
		dir0_t = _gaussian_blur_nchw(x=dir0_t, sigma=float(dir_blur_sigma))
		dir1_t = _gaussian_blur_nchw(x=dir1_t, sigma=float(dir_blur_sigma))
	if valid_t is not None:
		valid_t = (valid_t > 0.5).to(dtype=torch.float32)

	if unet_out_dir_base is not None:
		dbg_dir = Path(unet_out_dir_base) / "dbg_cos_z"
		dbg_dir.mkdir(parents=True, exist_ok=True)
		cos_np_all = cos_t[:, 0].detach().cpu().numpy().astype("float32")
		for zi in range(int(cos_np_all.shape[0])):
			tifffile.imwrite(str(dbg_dir / f"z{zi:04d}.tif"), cos_np_all[zi], compression="lzw")

	return FitData(
		cos=cos_t,
		grad_mag=mag_t,
		dir0=dir0_t,
		dir1=dir1_t,
		valid=valid_t,
		dir0_y=dir0_y_t,
		dir1_y=dir1_y_t,
		dir0_x=dir0_x_t,
		dir1_x=dir1_x_t,
		downscale=float(downscale) if downscale is not None else 1.0,
		constraints=None,
	)
