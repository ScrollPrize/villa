from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

import tifffile

import fit_data


@dataclass(frozen=True)
class ModelInit:
	init_size_frac: float
	init_size_frac_h: float | None
	init_size_frac_v: float | None
	mesh_step_px: int
	winding_step_px: int
	mesh_h: int
	mesh_w: int


@dataclass(frozen=True)
class ModelParams:
	mesh_step_px: int
	winding_step_px: int
	subsample_mesh: int
	subsample_winding: int
	z_step_vx: int
	# Fit-time scale-down between full-res voxel/pixel space and model pixel space.
	scaledown: float
	# 3D crop in full-res voxel/pixel space: (x, y, w, h, z0, d).
	crop_fullres_xyzwhd: tuple[int, int, int, int, int, int] | None = None
	# Margin (in model pixels) from expanded data origin to original crop origin.
	# Used by _build_base_grid to center mesh on the original crop within expanded data.
	data_margin_modelpx: tuple[float, float] = (0.0, 0.0)
	# Data size in model pixels (h, w). Used for validity masking and mask scheduling
	# when data has expanded margins. (0,0) = not set, fall back to crop dims.
	data_size_modelpx: tuple[int, int] = (0, 0)

	@property
	def crop_xyzwhd(self) -> tuple[int, int, int, int, int, int] | None:
		"""3D crop in model pixel space (fullres / scaledown), rounded to int."""
		c = self.crop_fullres_xyzwhd
		if c is None:
			return None
		ds = float(self.scaledown)
		if ds <= 0.0:
			ds = 1.0
		x, y, w, h, z0, d = (int(v) for v in c)
		return (
			int(round(float(x) / ds)),
			int(round(float(y) / ds)),
			max(1, int(round(float(w) / ds))),
			max(1, int(round(float(h) / ds))),
			int(z0),
			int(d),
		)


def xy_img_validity_mask(*, params: ModelParams, xy: torch.Tensor) -> torch.Tensor:
	"""Return a binary mask for pixel xy image positions.

	`xy` must encode (x,y) in the last dimension (..,2) in model pixel coords.
	Output has shape `xy.shape[:-1]`.

	When data_size_modelpx is set (expanded data with margins), uses data extent
	as the valid region. Otherwise falls back to crop bounds.
	"""
	if xy.ndim < 1 or int(xy.shape[-1]) != 2:
		raise ValueError("xy must have last dim == 2")
	dh, dw = params.data_size_modelpx
	if dh > 0 and dw > 0:
		h = float(max(1, int(dh) - 1))
		w = float(max(1, int(dw) - 1))
	else:
		if params.crop_xyzwhd is None:
			raise ValueError("xy_img_validity_mask requires params.crop_xyzwhd or data_size_modelpx")
		_cx, _cy, cw, ch, _z0, _d = params.crop_xyzwhd
		h = float(max(1, int(ch) - 1))
		w = float(max(1, int(cw) - 1))
	flat = xy.reshape(-1, 2)
	x = flat[:, 0]
	y = flat[:, 1]
	inside = (x >= 0.0) & (x <= w) & (y >= 0.0) & (y <= h)
	return inside.to(dtype=torch.float32).reshape(xy.shape[:-1])


@torch.no_grad()
def xy_img_mask(*, res: "FitResult", xy: torch.Tensor, loss_name: str) -> torch.Tensor:
	"""Return image-space mask at `xy` for `loss_name`.

	Mask is the base xy validity mask multiplied by any stage-scheduled image masks
	configured to apply to `loss_name`.

	- `xy`: (...,2) in model pixel coords
	- returns: xy.shape[:-1] float mask in [0,1]
	"""
	base = xy_img_validity_mask(params=res.params, xy=xy)
	masks = res.stage_img_masks
	losses = res.stage_img_masks_losses
	if not masks or not losses:
		return base
	name = str(loss_name)
	if xy.ndim < 1 or int(xy.shape[-1]) != 2:
		raise ValueError("xy must have last dim == 2")
	if res.h_img <= 0 or res.w_img <= 0:
		raise ValueError("invalid image size")

	out = base
	for lbl, m_img in masks.items():
		ls = losses.get(lbl, [])
		if name not in ls:
			continue
		if m_img.ndim != 4 or int(m_img.shape[1]) != 1:
			raise ValueError("stage img mask must be (N,1,H,W)")
		if int(m_img.shape[0]) != int(xy.shape[0]):
			raise ValueError("stage img mask batch must match xy batch")
		grid = xy.detach().to(dtype=torch.float32).clone()
		grid = grid.reshape(int(grid.shape[0]), -1, 1, 2)
		hd = float(max(1, int(res.h_img) - 1))
		wd = float(max(1, int(res.w_img) - 1))
		grid[..., 0] = (grid[..., 0] / wd) * 2.0 - 1.0
		grid[..., 1] = (grid[..., 1] / hd) * 2.0 - 1.0
		s = F.grid_sample(m_img.to(dtype=torch.float32), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
		s = s.reshape(int(xy.shape[0]), *[int(v) for v in xy.shape[1:-1]])
		out = out * s
	return out


@dataclass(frozen=True)
class FitResult:
	_xy_lr: torch.Tensor
	_xy_hr: torch.Tensor
	_xy_conn: torch.Tensor
	_data: fit_data.FitData
	_data_s: fit_data.FitData
	_target_plain: torch.Tensor
	_target_mod: torch.Tensor
	_amp_lr: torch.Tensor
	_bias_lr: torch.Tensor
	_mask_hr: torch.Tensor
	_mask_lr: torch.Tensor
	_mask_conn: torch.Tensor
	_h_img: int
	_w_img: int
	_params: ModelParams
	_stage_img_masks: dict[str, torch.Tensor] | None = None
	_stage_img_masks_losses: dict[str, list[str]] | None = None
	_vis_loss_maps: dict[str, torch.Tensor] | None = None

	@property
	def xy_lr(self) -> torch.Tensor:
		return self._xy_lr

	@property
	def xy_hr(self) -> torch.Tensor:
		return self._xy_hr

	@property
	def xy_conn(self) -> torch.Tensor:
		return self._xy_conn

	@property
	def stage_img_masks(self) -> dict[str, torch.Tensor] | None:
		return self._stage_img_masks

	@property
	def stage_img_masks_losses(self) -> dict[str, list[str]] | None:
		return self._stage_img_masks_losses

	@property
	def vis_loss_maps(self) -> dict[str, torch.Tensor] | None:
		return self._vis_loss_maps

	@property
	def data(self) -> fit_data.FitData:
		return self._data

	@property
	def data_s(self) -> fit_data.FitData:
		return self._data_s

	@property
	def target_plain(self) -> torch.Tensor:
		return self._target_plain

	@property
	def target_mod(self) -> torch.Tensor:
		return self._target_mod

	@property
	def amp_lr(self) -> torch.Tensor:
		return self._amp_lr

	@property
	def bias_lr(self) -> torch.Tensor:
		return self._bias_lr

	@property
	def mask_hr(self) -> torch.Tensor:
		return self._mask_hr

	@property
	def mask_lr(self) -> torch.Tensor:
		return self._mask_lr

	@property
	def mask_conn(self) -> torch.Tensor:
		return self._mask_conn

	@property
	def h_img(self) -> int:
		return int(self._h_img)

	@property
	def w_img(self) -> int:
		return int(self._w_img)

	@property
	def params(self) -> ModelParams:
		return self._params

	@property
	def mesh_step_px(self) -> int:
		return int(self._params.mesh_step_px)

	@property
	def winding_step_px(self) -> int:
		return int(self._params.winding_step_px)

	@property
	def subsample_mesh(self) -> int:
		return int(self._params.subsample_mesh)

	@property
	def subsample_winding(self) -> int:
		return int(self._params.subsample_winding)

	@property
	def z_step_vx(self) -> int:
		return int(self._params.z_step_vx)



class Model2D(nn.Module):
	def __init__(
		self,
		init: ModelInit,
		device: torch.device,
		*,
		z_size: int = 1,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
		z_step_vx: int,
		scaledown: float,
		crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None,
		data_margin_modelpx: tuple[float, float] | None = None,
		data_size_modelpx: tuple[int, int] | None = None,
	) -> None:
		super().__init__()
		self.init = init
		self.device = device
		self.z_size = max(1, int(z_size))
		self.global_transform_enabled = True
		# FIXME need better init ...
		self.theta = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32)-0.5)
		self.winding_scale = nn.Parameter(torch.ones((), device=device, dtype=torch.float32))
		self.params = ModelParams(
			mesh_step_px=int(self.init.mesh_step_px),
			winding_step_px=int(self.init.winding_step_px),
			subsample_mesh=int(subsample_mesh),
			subsample_winding=int(subsample_winding),
			z_step_vx=max(1, int(z_step_vx)),
			scaledown=float(scaledown),
			crop_fullres_xyzwhd=None if crop_xyzwhd is None else tuple(int(v) for v in crop_xyzwhd),
			data_margin_modelpx=tuple(float(v) for v in data_margin_modelpx) if data_margin_modelpx is not None else (0.0, 0.0),
			data_size_modelpx=tuple(int(v) for v in data_size_modelpx) if data_size_modelpx is not None else (0, 0),
		)

		self.mesh_h = max(2, int(init.mesh_h))
		self.mesh_w = max(2, int(init.mesh_w))
		offset_scales = 5
		self.mesh_ms = nn.ParameterList(
			self._build_mesh_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.conn_offset_ms = nn.ParameterList(
			self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		amp_init = torch.full((int(self.z_size), 1, int(self.mesh_h), int(self.mesh_w)), 1.0, device=device, dtype=torch.float32)
		bias_init = torch.full((int(self.z_size), 1, int(self.mesh_h), int(self.mesh_w)), 0.5, device=device, dtype=torch.float32)
		self.amp = nn.Parameter(amp_init)
		self.bias = nn.Parameter(bias_init)
		self.const_mask_lr: torch.Tensor | None = None
		self._last_grow_insert_lr: tuple[int, int, int, int] | None = None
		self._last_grow_insert_z: int | None = None
		self.subsample_mesh = int(self.params.subsample_mesh)
		self.subsample_winding = int(self.params.subsample_winding)
		self._ema_decay = 0.99
		self.register_buffer("xy_ema", torch.zeros((0, 0, 0, 0), device=device, dtype=torch.float32))
		self.register_buffer("xy_conn_ema", torch.zeros((0, 0, 0, 0, 0), device=device, dtype=torch.float32))

	def bake_global_transform_into_mesh(self) -> None:
		"""Bake current global transform into mesh & disable it."""
		with torch.no_grad():
			uv = self.mesh_coarse()
			u = uv[:, 0:1]
			v = uv[:, 1:2]
			x, y = self._apply_global_transform(u, v)
			flat = torch.cat([x, y], dim=1)
			self.mesh_ms = self._construct_param_pyramid_from_flat(flat, n_scales=len(self.mesh_ms))
			self.theta.data.zero_()
			self.winding_scale.data.fill_(1.0)
			self.global_transform_enabled = False

	def xy_img_validity_mask(self, *, xy: torch.Tensor) -> torch.Tensor:
		return xy_img_validity_mask(params=self.params, xy=xy)

	@torch.no_grad()
	def update_ema(self, *, xy_lr: torch.Tensor, xy_conn: torch.Tensor) -> None:
		"""Update LR EMA tensors used by downstream mask scheduling.

		Inputs are expected to match `FitResult.xy_lr` and `FitResult.xy_conn`.
		"""
		d = float(self._ema_decay)
		x = xy_lr.detach().to(dtype=torch.float32)
		xc = xy_conn.detach().to(dtype=torch.float32)
		if self.xy_ema.numel() == 0 or tuple(self.xy_ema.shape) != tuple(x.shape):
			self.xy_ema = x.clone()
		else:
			self.xy_ema.mul_(d).add_(x, alpha=(1.0 - d))
		if self.xy_conn_ema.numel() == 0 or tuple(self.xy_conn_ema.shape) != tuple(xc.shape):
			self.xy_conn_ema = xc.clone()
		else:
			self.xy_conn_ema.mul_(d).add_(xc, alpha=(1.0 - d))

	def forward(self, data: fit_data.FitData) -> FitResult:
		if int(data.cos.shape[0]) != int(self.z_size):
			raise ValueError(f"data batch (z) mismatch: data.N={int(data.cos.shape[0])} model.z_size={int(self.z_size)}")
		xy_lr = self._grid_xy()
		xy_hr = self._grid_xy_subsampled_from_lr(xy_lr=xy_lr)
		xy_conn = self._xy_conn_px(xy_lr=xy_lr)
		data_s = data.grid_sample_px(xy_px=xy_hr)
		# Targets are defined purely in mesh/winding coordinates (no image-size dependence).
		h, w = int(xy_hr.shape[1]), int(xy_hr.shape[2])
		periods = max(1, int(self.mesh_w) - 1)
		xs = torch.linspace(0.0, float(periods), w, device=self.device, dtype=torch.float32)
		phase = (2.0 * torch.pi) * xs.view(1, 1, 1, w)
		target_plain = 0.5 + 0.5 * torch.cos(phase).expand(int(self.z_size), 1, h, w)

		amp_lr = self.amp
		bias_lr = self.bias
		amp_lr = amp_lr.clamp(0.1, 1.0)
		bias_lr = bias_lr.clamp(0.0, 0.45)
		amp_hr = F.interpolate(amp_lr, size=(h, w), mode="bilinear", align_corners=True)
		bias_hr = F.interpolate(bias_lr, size=(h, w), mode="bilinear", align_corners=True)
		target_mod = bias_hr + amp_hr * (target_plain - 0.5)
		target_mod = target_mod.clamp(0.0, 1.0)
		# Masking: when valid channel is available (preprocessed zarr with expanded data),
		# use it directly — model pixel coords = data pixel coords, no offset.
		# Otherwise fall back to crop-bounds validity mask.
		if data.valid is not None:
			def _sample_valid(xy: torch.Tensor) -> torch.Tensor:
				h_data, w_data = data.size
				hd = float(max(1, int(h_data) - 1))
				wd = float(max(1, int(w_data) - 1))
				grid = xy.clone()
				grid[..., 0] = (xy[..., 0] / wd) * 2.0 - 1.0
				grid[..., 1] = (xy[..., 1] / hd) * 2.0 - 1.0
				sv = F.grid_sample(data.valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
				return (sv > 0.5).to(dtype=torch.float32)
			mask_hr = _sample_valid(xy_hr)
			mask_lr = _sample_valid(xy_lr)
			# xy_conn has extra dim: (N, Hm, Wm, 3, 2) — sample each of the 3 conn points
			n, hm, wm, _3, _2 = (int(v) for v in xy_conn.shape)
			conn_flat = xy_conn.reshape(n, hm, wm * 3, 2)
			valid_conn = _sample_valid(conn_flat).reshape(n, 1, hm, wm, 3)
			mask_conn = valid_conn
		else:
			mask_hr = self.xy_img_validity_mask(xy=xy_hr).unsqueeze(1)
			mask_lr = self.xy_img_validity_mask(xy=xy_lr).unsqueeze(1)
			mask_conn = self.xy_img_validity_mask(xy=xy_conn).unsqueeze(1)
		# Edge conn points are synthetic (copied from the nearest column) and should not be used.
		if mask_conn.shape[3] >= 1:
			mask_conn[:, :, :, 0, 0] = 0.0
			mask_conn[:, :, :, -1, 2] = 0.0
		h_data, w_data = data.size
		return FitResult(
			_xy_lr=xy_lr,
			_xy_hr=xy_hr,
			_xy_conn=xy_conn,
			_data=data,
			_data_s=data_s,
			_target_plain=target_plain,
			_target_mod=target_mod,
			_amp_lr=amp_lr,
			_bias_lr=bias_lr,
			_mask_hr=mask_hr,
			_mask_lr=mask_lr,
			_mask_conn=mask_conn,
			_h_img=int(h_data),
			_w_img=int(w_data),
			_params=self.params,
		)

	def _apply_global_transform(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if not bool(self.global_transform_enabled):
			return u, v
		u = self.winding_scale * u
		c = torch.cos(self.theta)
		s = torch.sin(self.theta)
		# Rotate/scale around the current mesh center in pixel space.
		# This avoids coupling the global transform to external image/crop scaling.
		min_u = torch.amin(u)
		max_u = torch.amax(u)
		min_v = torch.amin(v)
		max_v = torch.amax(v)
		xc = 0.5 * (min_u + max_u)
		yc = 0.5 * (min_v + max_v)
		x = xc + c * (u - xc) - s * (v - yc)
		y = yc + s * (u - xc) + c * (v - yc)
		return x, y

	def opt_params(self) -> dict[str, list[nn.Parameter]]:
		amp = [self.amp]
		bias = [self.bias]
		out = {
			"mesh_ms": list(self.mesh_ms),
			"conn_offset_ms": list(self.conn_offset_ms),
			"amp": amp,
			"bias": bias,
			"amp_ms": amp,
			"bias_ms": bias,
		}
		if bool(self.global_transform_enabled):
			out["theta"] = [self.theta]
			out["winding_scale"] = [self.winding_scale]
		return out

	def load_state_dict_compat(self, state_dict: dict, *, strict: bool = False) -> tuple[list[str], list[str]]:
		st = dict(state_dict)
		st.pop("_model_params_", None)
		# EMA buffers are runtime-only; allow loading checkpoints with/without them.
		st.pop("xy_ema", None)
		st.pop("xy_conn_ema", None)
		# Back-compat renames.
		for k in list(st.keys()):
			if k.startswith("mesh_offset_ms."):
				st["conn_offset_ms." + k[len("mesh_offset_ms."):]] = st.pop(k)
			elif k == "amp_ms.0" or k == "amp_coarse":
				st["amp"] = st.pop(k)
			elif k == "bias_ms.0" or k == "bias_coarse":
				st["bias"] = st.pop(k)
			elif k == "phase":
				st.pop(k)

		# Fill missing tensors (common when loading older checkpoints).
		for k, p in self.state_dict().items():
			if k in st:
				continue
			if k.startswith("conn_offset_ms."):
				st[k] = torch.zeros_like(p)
			elif k == "amp":
				st[k] = torch.ones_like(p)
			elif k == "bias":
				st[k] = torch.full_like(p, 0.5)

		incompat = super().load_state_dict(st, strict=bool(strict))
		return list(incompat.missing_keys), list(incompat.unexpected_keys)

	def grow(self, *, directions: list[str], steps: int) -> None:
		steps = max(0, int(steps))
		if steps <= 0:
			return

		dirs = {str(d).strip().lower() for d in directions}
		if not dirs:
			dirs = {"left", "right", "up", "down"}
		bad = dirs - {"left", "right", "up", "down", "fw", "bw"}
		if bad:
			raise ValueError(f"invalid grow direction(s): {sorted(bad)}")
		if ("fw" in dirs or "bw" in dirs) and (dirs - {"fw", "bw"}):
			raise ValueError("grow: fw/bw may only be used alone")
		if "fw" in dirs and "bw" in dirs:
			raise ValueError("grow: cannot combine fw and bw")

		self._last_grow_insert_z = None
		if "fw" in dirs or "bw" in dirs:
			side = +1 if "fw" in dirs else -1
			z0 = int(self.z_size)
			self.mesh_ms = nn.ParameterList(
				[nn.Parameter(self._expand_copy_edge(src=p, dim=0, side=side)) for p in list(self.mesh_ms)]
			)
			self.conn_offset_ms = nn.ParameterList(
				[nn.Parameter(self._expand_copy_edge(src=p, dim=0, side=side)) for p in list(self.conn_offset_ms)]
			)
			self.amp = nn.Parameter(self._expand_copy_edge(src=self.amp, dim=0, side=side))
			self.bias = nn.Parameter(self._expand_copy_edge(src=self.bias, dim=0, side=side))
			self.z_size = int(self.z_size) + 1
			self._last_grow_insert_z = z0 if side > 0 else 0
			self._last_grow_insert_lr = None
			self.const_mask_lr = None
			return

		h0 = int(self.mesh_h)
		w0 = int(self.mesh_w)
		py0 = 0
		px0 = 0
		ho = h0
		wo = w0
		order = ["up", "down", "left", "right"]
		dirs_list = [d for d in order if d in dirs]
		grow_specs: dict[str, tuple[int, int, int, int, int, int]] = {
			# (dim, side, d_mesh_h, d_mesh_w, d_py0, d_px0)
			"up": (2, -1, +1, 0, +1, 0),
			"down": (2, +1, +1, 0, 0, 0),
			"left": (3, -1, 0, +1, 0, +1),
			"right": (3, +1, 0, +1, 0, 0),
		}
		for _ in range(int(steps)):
			for d in dirs_list:
				dim, side, dh, dw, dpy, dpx = grow_specs[d]
				self.mesh_ms = self._grow_param_pyramid_flat_edit(
					src=self.mesh_ms,
					dim=dim,
					side=side,
					editor=self._expand_linear,
				)
				self.conn_offset_ms = self._grow_param_pyramid_flat_edit(
					src=self.conn_offset_ms,
					dim=dim,
					side=side,
					editor=self._expand_copy_edge,
				)
				amp2 = self._expand_copy_edge(src=self.amp, dim=dim, side=side)
				bias2 = self._expand_copy_edge(src=self.bias, dim=dim, side=side)
				self.amp = nn.Parameter(amp2)
				self.bias = nn.Parameter(bias2)
				self.mesh_h = int(self.mesh_h) + dh
				self.mesh_w = int(self.mesh_w) + dw
				py0 += dpy
				px0 += dpx

		self._last_grow_insert_lr = (int(py0), int(px0), int(ho), int(wo))
		self.const_mask_lr = None

	@staticmethod
	def _expand_linear(*, src: torch.Tensor, dim: int, side: int) -> torch.Tensor:
		"""Expand `src` by 1 along `dim` using a 2-point linear extrapolation at the chosen border."""
		if src.ndim <= dim:
			raise ValueError("grow: dim out of range")
		dim = int(dim)
		side = int(side)
		if side not in (-1, +1):
			raise ValueError("grow: side must be -1 or +1")
		if int(src.shape[dim]) < 2:
			raise ValueError("grow: need at least 2 samples for linear extrapolation")
		if side < 0:
			edge = src.select(dim, 0)
			nextv = src.select(dim, 1)
			new = (2.0 * edge - nextv).unsqueeze(dim)
			return torch.cat([new, src], dim=dim)
		edge = src.select(dim, int(src.shape[dim]) - 1)
		nextv = src.select(dim, int(src.shape[dim]) - 2)
		new = (2.0 * edge - nextv).unsqueeze(dim)
		return torch.cat([src, new], dim=dim)

	@staticmethod
	def _expand_copy_edge(*, src: torch.Tensor, dim: int, side: int) -> torch.Tensor:
		"""Expand `src` by 1 along `dim` by copying the last slice at the grown border."""
		if src.ndim <= dim:
			raise ValueError("grow: dim out of range")
		dim = int(dim)
		side = int(side)
		if side not in (-1, +1):
			raise ValueError("grow: side must be -1 or +1")
		if side < 0:
			edge = src.select(dim, 0).unsqueeze(dim)
			return torch.cat([edge, src], dim=dim)
		edge = src.select(dim, int(src.shape[dim]) - 1).unsqueeze(dim)
		return torch.cat([src, edge], dim=dim)

	def _grow_param_pyramid_flat_edit(
		self,
		*,
		src: nn.ParameterList,
		dim: int,
		side: int,
		editor,
	) -> nn.ParameterList:
		flat = self._integrate_param_pyramid(src)
		flat2 = editor(src=flat, dim=int(dim), side=int(side))
		return self._construct_param_pyramid_from_flat(flat2, n_scales=len(src))

	def _integrate_param_pyramid(self, src: nn.ParameterList) -> torch.Tensor:
		v = src[-1]
		for d in reversed(src[:-1]):
			v = self._upsample2_crop(src=v, h_t=int(d.shape[2]), w_t=int(d.shape[3])) + d
		return v

	def _construct_param_pyramid_from_flat(self, flat: torch.Tensor, n_scales: int) -> nn.ParameterList:
		shapes: list[tuple[int, int]] = [(int(flat.shape[2]), int(flat.shape[3]))]
		for _ in range(1, max(1, int(n_scales))):
			gh_prev, gw_prev = shapes[-1]
			gh_i = max(2, (gh_prev + 1) // 2)
			gw_i = max(2, (gw_prev + 1) // 2)
			shapes.append((gh_i, gw_i))
		targets: list[torch.Tensor] = [flat]
		for gh_i, gw_i in shapes[1:]:
			t = F.interpolate(targets[-1], size=(int(gh_i), int(gw_i)), mode="bilinear", align_corners=True)
			targets.append(t)
		residuals: list[torch.Tensor] = [torch.empty(0)] * len(targets)
		recon = targets[-1]
		residuals[-1] = targets[-1]
		for i in range(len(targets) - 2, -1, -1):
			up = self._upsample2_crop(src=recon, h_t=int(targets[i].shape[2]), w_t=int(targets[i].shape[3]))
			d = targets[i] - up
			residuals[i] = d
			recon = up + d
		out = nn.ParameterList()
		for r in residuals:
			out.append(nn.Parameter(r))
		return out

	def save_tiff(self, *, data: fit_data.FitData, path: str) -> None:
		"""Save raw model tensors as tiff stacks.

	Writes multiple files:
	- `<path>_xy_lr.tif`: (2,Hm,Wm)
	- `<path>_xy_hr.tif`: (2,He,We)
	- `<path>_xy_conn.tif`: (6,Hm,Wm) in order [l.x,l.y,p.x,p.y,r.x,r.y]
	- `<path>_mask_lr.tif`: (1,Hm,Wm)
		- `<path>_mask_hr.tif`: (1,He,We)
		- `<path>_mask_conn.tif`: (3,Hm,Wm)
		- `<path>_mesh.tif`: (2,Hm,Wm)
		- `<path>_conn_offset.tif`: (2,Hm,Wm)
	"""
		p = Path(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		stem = p.with_suffix("")
		with torch.no_grad():
			res = self(data)
			xy_lr = res.xy_lr.detach().cpu().to(dtype=torch.float32).numpy().transpose(0, 3, 1, 2)
			xy_hr = res.xy_hr.detach().cpu().to(dtype=torch.float32).numpy().transpose(0, 3, 1, 2)
			xy_conn = res.xy_conn.detach().cpu().to(dtype=torch.float32).numpy().transpose(0, 3, 4, 1, 2).reshape(int(xy_lr.shape[0]), 6, int(xy_lr.shape[2]), int(xy_lr.shape[3]))
			mask_lr = res.mask_lr.detach().cpu().to(dtype=torch.float32).numpy()
			mask_hr = res.mask_hr.detach().cpu().to(dtype=torch.float32).numpy()
			mask_conn = res.mask_conn[:, 0].detach().cpu().to(dtype=torch.float32).numpy().transpose(0, 3, 1, 2)
			mesh = self.mesh_coarse().detach().cpu().to(dtype=torch.float32).numpy()
			conn_off = self.conn_offset_coarse().detach().cpu().to(dtype=torch.float32).numpy()

			tifffile.imwrite(str(stem) + "_xy_lr.tif", xy_lr, compression="lzw")
			tifffile.imwrite(str(stem) + "_xy_hr.tif", xy_hr, compression="lzw")
			tifffile.imwrite(str(stem) + "_xy_conn.tif", xy_conn, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_lr.tif", mask_lr, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_hr.tif", mask_hr, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_conn.tif", mask_conn, compression="lzw")
			tifffile.imwrite(str(stem) + "_mesh.tif", mesh, compression="lzw")
			tifffile.imwrite(str(stem) + "_conn_offset.tif", conn_off, compression="lzw")

	def conn_offset_coarse(self) -> torch.Tensor:
		off = self.conn_offset_ms[-1]
		for d in reversed(self.conn_offset_ms[:-1]):
			off = self._upsample2_crop(src=off, h_t=int(d.shape[2]), w_t=int(d.shape[3])) + d
		return off

	def _xy_conn_px(self, *, xy_lr: torch.Tensor) -> torch.Tensor:
		"""Return per-mesh connection positions in pixel coordinates.

		For each base-mesh point, returns 3 pixel positions:
		- left-connection interpolation result (to previous column)
		- the point itself
		- right-connection interpolation result (to next column)

		Shape: (N,Hm,Wm,3,2)
		"""
		if xy_lr.ndim != 4 or int(xy_lr.shape[-1]) != 2:
			raise ValueError("xy_lr must be (N,H,W,2)")
		n, hm, wm, _c2 = (int(v) for v in xy_lr.shape)
		if hm <= 0 or wm <= 0:
			raise ValueError("invalid xy_lr shape")
		mesh_off = self.conn_offset_coarse()
		if mesh_off.shape[-2:] != (hm, wm):
			raise RuntimeError("mesh_offset must be defined on the base mesh")
		xy_px = xy_lr.permute(0, 3, 1, 2).contiguous()

		left_src = torch.cat([xy_px[:, :, :, 0:1], xy_px[:, :, :, :-1]], dim=3)
		right_src = torch.cat([xy_px[:, :, :, 1:], xy_px[:, :, :, -1:]], dim=3)
		off_l = mesh_off[:, 0]
		off_r = mesh_off[:, 1]
		base_i = torch.arange(hm, device=xy_lr.device, dtype=xy_lr.dtype).view(1, hm, 1)
		left_conn = self._interp_col(src=left_src, y=base_i + off_l)
		right_conn = self._interp_col(src=right_src, y=base_i + off_r)
		# Edge fallback for undefined horizontal neighbors:
		# mirror the nearest inside conn vector (visual/search-only backup).
		with torch.no_grad():
			if wm > 1:
				v_in_l = right_conn[:, :, :, 1] - xy_px[:, :, :, 1]
				v_in_r = left_conn[:, :, :, -2] - xy_px[:, :, :, -2]
				left_conn[:, :, :, 0] = (xy_px[:, :, :, 0] - v_in_l).detach()
				right_conn[:, :, :, -1] = (xy_px[:, :, :, -1] - v_in_r).detach()
		conn = torch.stack([left_conn, xy_px, right_conn], dim=1)
		return conn.permute(0, 3, 4, 1, 2).contiguous()

	@staticmethod
	def _interp_col(*, src: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""Interpolate `src` along height dimension at (possibly fractional) indices `y`.

		src: (N,2,H,W)
		y: (N,H,W)
		"""
		n, c, h, w = (int(v) for v in src.shape)
		if y.shape != (n, h, w):
			raise ValueError("y must be (N,H,W)")
		# Allow extrapolation beyond the mesh by using the nearest valid segment.
		# This makes weights outside [0,1] possible.
		yf = y.to(dtype=src.dtype)
		y0f = torch.floor(yf)
		t = (yf - y0f)
		low = yf < 0.0
		high = yf > float(h - 1)
		y0f = torch.where(low, torch.zeros_like(y0f), y0f)
		y0f = torch.where(high, torch.full_like(y0f, float(h - 2)), y0f)
		y0f = y0f.clamp(0.0, float(h - 2))
		y1f = y0f + 1.0
		# Recompute t based on the chosen segment start so it extrapolates correctly.
		t = yf - y0f
		y0i = y0f.to(dtype=torch.int64)
		y1i = y1f.to(dtype=torch.int64)
		idx0 = y0i.view(n, 1, h, w).expand(n, c, h, w)
		idx1 = y1i.view(n, 1, h, w).expand(n, c, h, w)
		v0 = torch.take_along_dim(src, idx0, dim=2)
		v1 = torch.take_along_dim(src, idx1, dim=2)
		return v0 * (1.0 - t).view(n, 1, h, w) + v1 * t.view(n, 1, h, w)

	def grid_uv(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return base (u,v) mesh coordinates in pixel units."""
		uv = self.mesh_coarse()
		return uv[:, 0:1], uv[:, 1:2]

	def mesh_coarse(self) -> torch.Tensor:
		m = self.mesh_ms[-1]
		for d in reversed(self.mesh_ms[:-1]):
			m = self._upsample2_crop(src=m, h_t=int(d.shape[2]), w_t=int(d.shape[3])) + d
		return m

	@staticmethod
	def _upsample2_crop(*, src: torch.Tensor, h_t: int, w_t: int) -> torch.Tensor:
		up = F.interpolate(src, scale_factor=2.0, mode="bilinear", align_corners=True)
		return up[:, :, :h_t, :w_t]

	def _build_offset_ms(self, *, offset_scales: int, device: torch.device, gh0: int, gw0: int) -> list[nn.Parameter]:
		gh0 = int(gh0)
		gw0 = int(gw0)
		n_scales = max(1, int(offset_scales))
		shapes: list[tuple[int, int]] = [(gh0, gw0)]
		for _ in range(1, n_scales):
			gh_prev, gw_prev = shapes[-1]
			gh_i = max(2, (gh_prev + 1) // 2)
			gw_i = max(2, (gw_prev + 1) // 2)
			shapes.append((gh_i, gw_i))
		n = int(self.z_size)
		return [nn.Parameter(torch.zeros(n, 2, gh_i, gw_i, device=device, dtype=torch.float32)) for (gh_i, gw_i) in shapes]

	def _build_mesh_ms(self, *, offset_scales: int, device: torch.device, gh0: int, gw0: int) -> list[nn.Parameter]:
		"""Build residual pyramid whose reconstruction is the mesh coords in pixel units."""
		base = self._build_base_grid(gh0=int(gh0), gw0=int(gw0)).to(device=device)
		ms = self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=int(gh0), gw0=int(gw0))
		ms[0] = nn.Parameter(base)
		return ms

	def _grid_xy(self) -> torch.Tensor:
		"""Return globally transformed mesh coordinates in pixel xy (N,Hm,Wm,2)."""
		u, v = self.grid_uv()
		x, y = self._apply_global_transform(u, v)
		return torch.stack([x[:, 0], y[:, 0]], dim=-1)

	def _grid_xy_subsampled_from_lr(self, *, xy_lr: torch.Tensor) -> torch.Tensor:
		h = max(2, (self.mesh_h - 1) * int(self.params.subsample_mesh) + 1)
		w = max(2, (self.mesh_w - 1) * int(self.params.subsample_winding) + 1)
		xy_nchw = xy_lr.permute(0, 3, 1, 2).contiguous()
		xy_nchw = F.interpolate(xy_nchw, size=(h, w), mode="bilinear", align_corners=True)
		return xy_nchw.permute(0, 2, 3, 1).contiguous()

	@classmethod
	def from_fit_data(
		cls,
		data: "fit_data.FitData",
		mesh_step_px: int,
		winding_step_px: int,
		init_size_frac: float,
		device: torch.device,
		init_size_frac_h: float | None = None,
		init_size_frac_v: float | None = None,
		z_size: int = 1,
		z_step_vx: int = 1,
		scaledown: float = 1.0,
		crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None,
		data_margin_modelpx: tuple[float, float] | None = None,
		data_size_modelpx: tuple[int, int] | None = None,
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> "Model2D":
		# Use crop dimensions (not data.size) for mesh sizing — data may have
		# expanded margins for valid-mask coverage.
		if crop_xyzwhd is not None:
			_cx, _cy, _cw, _ch, _z0, _d = crop_xyzwhd
			ds = max(1e-6, float(scaledown))
			h_img = max(1, int(round(float(_ch) / ds)))
			w_img = max(1, int(round(float(_cw) / ds)))
		else:
			h_img, w_img = data.size
		step_h = int(mesh_step_px)
		step_w = int(winding_step_px)
		fh = float(init_size_frac if init_size_frac_h is None else init_size_frac_h)
		fw = float(init_size_frac if init_size_frac_v is None else init_size_frac_v)
		h = max(2, int(round(float(fh) * float(int(h_img)))))
		w = max(2, int(round(float(fw) * float(int(w_img)))))
		mesh_h = max(2, (h + step_h - 1) // step_h + 1)
		mesh_w = max(2, (w + step_w - 1) // step_w + 1)
		init = ModelInit(
			init_size_frac=float(init_size_frac),
			init_size_frac_h=None if init_size_frac_h is None else float(init_size_frac_h),
			init_size_frac_v=None if init_size_frac_v is None else float(init_size_frac_v),
			mesh_step_px=int(mesh_step_px),
			winding_step_px=int(winding_step_px),
			mesh_h=int(mesh_h),
			mesh_w=int(mesh_w),
		)
		return cls(
			init=init,
			device=device,
			z_size=max(1, int(z_size)),
			subsample_mesh=subsample_mesh,
			subsample_winding=subsample_winding,
			z_step_vx=max(1, int(z_step_vx)),
			scaledown=float(scaledown),
			crop_xyzwhd=crop_xyzwhd,
			data_margin_modelpx=data_margin_modelpx,
			data_size_modelpx=data_size_modelpx,
		)

	def _build_base_grid(self, *, gh0: int, gw0: int) -> torch.Tensor:
		# Model domain is initialized to a configurable fraction of the image extent.
		# Internal coordinates are stored in pixel units.
		if self.params.crop_xyzwhd is None:
			raise ValueError("_build_base_grid requires params.crop_xyzwhd")
		_cx, _cy, cw, ch, _z0, _d = self.params.crop_xyzwhd
		w = float(max(1, int(cw) - 1))
		h = float(max(1, int(ch) - 1))
		# Center mesh on the original crop within (possibly expanded) data.
		mx, my = self.params.data_margin_modelpx
		xc = float(mx) + 0.5 * w
		yc = float(my) + 0.5 * h
		fh = float(self.init.init_size_frac if self.init.init_size_frac_h is None else self.init.init_size_frac_h)
		fw = float(self.init.init_size_frac if self.init.init_size_frac_v is None else self.init.init_size_frac_v)
		u0 = xc - 0.5 * fw * w
		u1 = xc + 0.5 * fw * w
		v0 = yc - 0.5 * fh * h
		v1 = yc + 0.5 * fh * h
		u = torch.linspace(u0, u1, int(gw0), dtype=torch.float32)
		v = torch.linspace(v0, v1, int(gh0), dtype=torch.float32)
		vv, uu = torch.meshgrid(v, u, indexing="ij")
		grid = torch.stack([uu, vv], dim=0).unsqueeze(0)
		return grid.expand(int(self.z_size), -1, -1, -1).contiguous()
