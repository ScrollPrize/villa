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
	h_img: int
	w_img: int
	init_size_frac: float
	init_size_frac_h: float | None
	init_size_frac_v: float | None
	mesh_step_px: int
	winding_step_px: int


@dataclass(frozen=True)
class FitResult:
	_xy_lr: torch.Tensor
	_xy_hr: torch.Tensor
	_xy_conn: torch.Tensor
	_data_s: fit_data.FitData
	_mask_hr: torch.Tensor
	_mask_lr: torch.Tensor
	_mask_conn: torch.Tensor
	_h_img: int
	_w_img: int
	_mesh_step_px: int
	_winding_step_px: int

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
	def data_s(self) -> fit_data.FitData:
		return self._data_s

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
	def mesh_step_px(self) -> int:
		return int(self._mesh_step_px)

	@property
	def winding_step_px(self) -> int:
		return int(self._winding_step_px)



class Model2D(nn.Module):
	def __init__(
		self,
		init: ModelInit,
		device: torch.device,
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> None:
		super().__init__()
		self.init = init
		self.device = device
		# FIXME need better init ...
		self.theta = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32)-0.5)
		self.phase = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
		self.winding_scale = nn.Parameter(torch.ones((), device=device, dtype=torch.float32))

		fh = float(init.init_size_frac if init.init_size_frac_h is None else init.init_size_frac_h) * float(int(init.h_img))
		fw = float(init.init_size_frac if init.init_size_frac_v is None else init.init_size_frac_v) * float(int(init.w_img))
		h = max(2, int(round(fh)))
		w = max(2, int(round(fw)))
		self.mesh_h = max(2, (h + int(init.mesh_step_px) - 1) // int(init.mesh_step_px) + 1)
		self.mesh_w = max(2, (w + int(init.winding_step_px) - 1) // int(init.winding_step_px) + 1)
		offset_scales = 5
		self.mesh_ms = nn.ParameterList(
			self._build_mesh_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.conn_offset_ms = nn.ParameterList(
			self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.const_mask_lr: torch.Tensor | None = None
		self._last_grow_insert_lr: tuple[int, int, int, int] | None = None
		self.subsample_winding = int(subsample_winding)
		self.subsample_mesh = int(subsample_mesh)

	def xy_img_validity_mask(self, *, xy: torch.Tensor) -> torch.Tensor:
		"""Return a binary mask for pixel xy image positions.

	`xy` must encode (x,y) in the last dimension (..,2) in pixel coords.
	Output has shape `xy.shape[:-1]`.
	"""
		if xy.ndim < 1 or int(xy.shape[-1]) != 2:
			raise ValueError("xy must have last dim == 2")
		h = float(max(1, int(self.init.h_img) - 1))
		w = float(max(1, int(self.init.w_img) - 1))
		flat = xy.reshape(-1, 2)
		x = flat[:, 0]
		y = flat[:, 1]
		inside = (x >= 0.0) & (x <= w) & (y >= 0.0) & (y <= h)
		return inside.to(dtype=torch.float32).reshape(xy.shape[:-1])

	def forward(self, data: fit_data.FitData) -> FitResult:
		xy_lr = self._grid_xy()
		xy_hr = self._grid_xy_subsampled_from_lr(xy_lr=xy_lr)
		xy_conn = self._xy_conn_px(xy_lr=xy_lr)
		data_s = data.grid_sample_px(xy_px=xy_hr)
		mask_hr = self.xy_img_validity_mask(xy=xy_hr).unsqueeze(1)
		mask_lr = self.xy_img_validity_mask(xy=xy_lr).unsqueeze(1)
		mask_conn = self.xy_img_validity_mask(xy=xy_conn).unsqueeze(1)
		return FitResult(
			_xy_lr=xy_lr,
			_xy_hr=xy_hr,
			_xy_conn=xy_conn,
			_data_s=data_s,
			_mask_hr=mask_hr,
			_mask_lr=mask_lr,
			_mask_conn=mask_conn,
			_h_img=int(self.init.h_img),
			_w_img=int(self.init.w_img),
			_mesh_step_px=int(self.init.mesh_step_px),
			_winding_step_px=int(self.init.winding_step_px),
		)

	def _apply_global_transform(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		period = (2.0 * float(max(1, int(self.init.w_img) - 1))) / float(max(1, self.mesh_w - 1))
		phase = torch.remainder(self.phase + 0.5 * period, period) - 0.5 * period
		u = self.winding_scale * u + phase
		c = torch.cos(self.theta)
		s = torch.sin(self.theta)
		xc = 0.5 * float(max(1, int(self.init.w_img) - 1))
		yc = 0.5 * float(max(1, int(self.init.h_img) - 1))
		x = xc + c * (u - xc) - s * (v - yc)
		y = yc + s * (u - xc) + c * (v - yc)
		return x, y

	def opt_params(self) -> dict[str, list[nn.Parameter]]:
		return {
			"theta": [self.theta],
			"phase": [self.phase],
			"winding_scale": [self.winding_scale],
			"mesh_ms": list(self.mesh_ms),
			"conn_offset_ms": list(self.conn_offset_ms),
		}

	def grow(self, *, directions: list[str], steps: int) -> None:
		steps = max(0, int(steps))
		if steps <= 0:
			return

		dirs = {str(d).strip().lower() for d in directions}
		if not dirs:
			dirs = {"left", "right", "up", "down"}
		bad = dirs - {"left", "right", "up", "down"}
		if bad:
			raise ValueError(f"invalid grow direction(s): {sorted(bad)}")

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
			xy_lr = res.xy_lr[0].detach().cpu().to(dtype=torch.float32).numpy().transpose(2, 0, 1)
			xy_hr = res.xy_hr[0].detach().cpu().to(dtype=torch.float32).numpy().transpose(2, 0, 1)
			xy_conn = res.xy_conn[0].detach().cpu().to(dtype=torch.float32).numpy().reshape(xy_lr.shape[1], xy_lr.shape[2], 6).transpose(2, 0, 1)
			mask_lr = res.mask_lr[0].detach().cpu().to(dtype=torch.float32).numpy()
			mask_hr = res.mask_hr[0].detach().cpu().to(dtype=torch.float32).numpy()
			mask_conn = res.mask_conn[0, 0].detach().cpu().to(dtype=torch.float32).numpy().transpose(2, 0, 1)
			mesh = self.mesh_coarse()[0].detach().cpu().to(dtype=torch.float32).numpy()
			conn_off = self.conn_offset_coarse()[0].detach().cpu().to(dtype=torch.float32).numpy()

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
		return [nn.Parameter(torch.zeros(1, 2, gh_i, gw_i, device=device, dtype=torch.float32)) for (gh_i, gw_i) in shapes]

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
		h = max(2, (self.mesh_h - 1) * self.subsample_mesh + 1)
		w = max(2, (self.mesh_w - 1) * self.subsample_winding + 1)
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
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> "Model2D":
		h_img, w_img = data.size
		init = ModelInit(
			h_img=int(h_img),
			w_img=int(w_img),
			init_size_frac=float(init_size_frac),
			init_size_frac_h=None if init_size_frac_h is None else float(init_size_frac_h),
			init_size_frac_v=None if init_size_frac_v is None else float(init_size_frac_v),
			mesh_step_px=int(mesh_step_px),
			winding_step_px=int(winding_step_px),
		)
		return cls(
			init=init,
			device=device,
			subsample_mesh=subsample_mesh,
			subsample_winding=subsample_winding,
		)

	def _build_base_grid(self, *, gh0: int, gw0: int) -> torch.Tensor:
		# Model domain is initialized to a configurable fraction of the image extent.
		# Internal coordinates are stored in pixel units.
		w = float(max(1, int(self.init.w_img) - 1))
		h = float(max(1, int(self.init.h_img) - 1))
		fh = float(self.init.init_size_frac if self.init.init_size_frac_h is None else self.init.init_size_frac_h)
		fw = float(self.init.init_size_frac if self.init.init_size_frac_v is None else self.init.init_size_frac_v)
		u0 = 0.5 * (1.0 - fw) * w
		u1 = 0.5 * (1.0 + fw) * w
		v0 = 0.5 * (1.0 - fh) * h
		v1 = 0.5 * (1.0 + fh) * h
		u = torch.linspace(u0, u1, int(gw0), dtype=torch.float32)
		v = torch.linspace(v0, v1, int(gh0), dtype=torch.float32)
		vv, uu = torch.meshgrid(v, u, indexing="ij")
		return torch.stack([uu, vv], dim=0).unsqueeze(0)

	def target_cos(
		self,
		*,
		cosine_periods: float | None = None,
		subsample_winding: int | None = None,
		subsample_mesh: int | None = None,
	) -> torch.Tensor:
		if cosine_periods is None:
			periods = max(1, self.mesh_w - 1)
		else:
			periods = float(cosine_periods)

		sub_w = int(self.subsample_winding if subsample_winding is None else subsample_winding)
		sub_m = int(self.subsample_mesh if subsample_mesh is None else subsample_mesh)
		sub_w = max(1, sub_w)
		sub_m = max(1, sub_m)
		h = max(2, (self.mesh_h - 1) * sub_m + 1)
		w = max(2, (self.mesh_w - 1) * sub_w + 1)
		xd = float(max(1, int(self.init.w_img) - 1))
		yd = float(max(1, int(self.init.h_img) - 1))
		us = torch.linspace(0.0, xd, w, device=self.device, dtype=torch.float32)
		vs = torch.linspace(0.0, yd, h, device=self.device, dtype=torch.float32)
		vv, uu = torch.meshgrid(vs, us, indexing="ij")
		u = uu.view(1, 1, h, w)

		phase = torch.pi * (u / xd) * periods
		return 0.5 + 0.5 * torch.cos(phase)
