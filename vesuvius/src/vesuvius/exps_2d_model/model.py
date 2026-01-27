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
		self.theta = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
		self.phase = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
		self.winding_scale = nn.Parameter(torch.ones((), device=device, dtype=torch.float32))

		fh = float(init.init_size_frac) * float(int(init.h_img))
		fw = float(init.init_size_frac) * float(int(init.w_img))
		h = max(2, int(round(fh)))
		w = max(2, int(round(fw)))
		self.mesh_h = max(2, (h + int(init.mesh_step_px) - 1) // int(init.mesh_step_px) + 1)
		self.mesh_w = max(2, (w + int(init.winding_step_px) - 1) // int(init.winding_step_px) + 1)

		self.base_grid = self._build_base_grid().to(device=device)
		offset_scales = 5
		self.offset_ms = nn.ParameterList(
			self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.mesh_offset_ms = nn.ParameterList(
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
		if self.const_mask_lr is None:
			xy_lr = self._grid_xy()
		else:
			m = self.const_mask_lr
			if m.ndim != 4 or int(m.shape[1]) != 1:
				raise ValueError("const_mask_lr must be (N,1,Hm,Wm)")
			if m.shape[-2:] != (int(self.mesh_h), int(self.mesh_w)):
				raise ValueError("const_mask_lr must match current mesh size")
			if int(m.shape[0]) == 1 and int(data.cos.shape[0]) > 1:
				m = m.expand(int(data.cos.shape[0]), 1, int(self.mesh_h), int(self.mesh_w))
			m = m.detach().to(device=self.device)
			with torch.no_grad():
				xy_lr_ng = self._grid_xy()
			xy_lr_g = self._grid_xy()
			m4 = m.to(dtype=xy_lr_g.dtype).permute(0, 2, 3, 1).contiguous()
			xy_lr = m4 * xy_lr_ng + (1.0 - m4) * xy_lr_g
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
			"offset_ms": list(self.offset_ms),
			"mesh_offset_ms": list(self.mesh_offset_ms),
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

		add_up = steps if "up" in dirs else 0
		add_dn = steps if "down" in dirs else 0
		add_l = steps if "left" in dirs else 0
		add_r = steps if "right" in dirs else 0

		h0 = int(self.mesh_h)
		w0 = int(self.mesh_w)
		h1 = max(2, h0 + add_up + add_dn)
		w1 = max(2, w0 + add_l + add_r)
		if (h1, w1) == (h0, w0):
			return

		self.mesh_h = int(h1)
		self.mesh_w = int(w1)
		self.base_grid = self._build_base_grid().to(device=self.device)
		self.offset_ms, ins = self._grow_param_pyramid(src=self.offset_ms, dirs=dirs, h0=h0, w0=w0)
		self.mesh_offset_ms, _ins2 = self._grow_param_pyramid(src=self.mesh_offset_ms, dirs=dirs, h0=h0, w0=w0)
		self._last_grow_insert_lr = ins
		self.const_mask_lr = None

	def _grow_param_pyramid(
		self,
		*,
		src: nn.ParameterList,
		dirs: set[str],
		h0: int,
		w0: int,
	) -> tuple[nn.ParameterList, tuple[int, int, int, int]]:
		n_scales = len(src)
		if n_scales <= 0:
			return nn.ParameterList([]), (0, 0, 0, 0)

		def _build_shapes(gh: int, gw: int) -> list[tuple[int, int]]:
			shapes: list[tuple[int, int]] = [(int(gh), int(gw))]
			for _ in range(1, n_scales):
				gh_prev, gw_prev = shapes[-1]
				gh_i = max(2, gh_prev // 2 + 1)
				gw_i = max(2, gw_prev // 2 + 1)
				shapes.append((gh_i, gw_i))
			return shapes

		old_shapes = _build_shapes(h0, w0)
		new_shapes = _build_shapes(int(self.mesh_h), int(self.mesh_w))
		out = nn.ParameterList()
		insert0: tuple[int, int, int, int] | None = None
		for i, (hn, wn) in enumerate(new_shapes):
			o = src[i]
			_, c, ho, wo = (int(v) for v in o.shape)
			if (ho, wo) != old_shapes[i]:
				raise RuntimeError("grow: unexpected pyramid shape mismatch")
			n0 = torch.zeros(1, c, hn, wn, device=o.device, dtype=o.dtype)
			dy = max(0, hn - ho)
			dx = max(0, wn - wo)
			if ("up" in dirs) and ("down" in dirs):
				py0 = dy // 2
			else:
				py0 = dy if "up" in dirs else 0
			if ("left" in dirs) and ("right" in dirs):
				px0 = dx // 2
			else:
				px0 = dx if "left" in dirs else 0
			n0[:, :, py0:py0 + ho, px0:px0 + wo] = o.data
			if i == 0:
				insert0 = (py0, px0, ho, wo)
			out.append(nn.Parameter(n0))
		if insert0 is None:
			raise RuntimeError("grow: failed to compute insertion")
		return out, insert0

	def save_tiff(self, *, data: fit_data.FitData, path: str) -> None:
		"""Save raw model tensors as tiff stacks.

	Writes multiple files:
	- `<path>_xy_lr.tif`: (2,Hm,Wm)
	- `<path>_xy_hr.tif`: (2,He,We)
	- `<path>_xy_conn.tif`: (6,Hm,Wm) in order [l.x,l.y,p.x,p.y,r.x,r.y]
	- `<path>_mask_lr.tif`: (1,Hm,Wm)
	- `<path>_mask_hr.tif`: (1,He,We)
	- `<path>_mask_conn.tif`: (3,Hm,Wm)
	- `<path>_base_grid.tif`: (2,Hm,Wm)
	- `<path>_offset.tif`: (2,Hm,Wm)
	- `<path>_mesh_offset.tif`: (2,Hm,Wm)
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
			base_grid = self.base_grid[0].detach().cpu().to(dtype=torch.float32).numpy()
			off = self.offset_coarse()[0].detach().cpu().to(dtype=torch.float32).numpy()
			mesh_off = self.mesh_offset_coarse()[0].detach().cpu().to(dtype=torch.float32).numpy()

			tifffile.imwrite(str(stem) + "_xy_lr.tif", xy_lr, compression="lzw")
			tifffile.imwrite(str(stem) + "_xy_hr.tif", xy_hr, compression="lzw")
			tifffile.imwrite(str(stem) + "_xy_conn.tif", xy_conn, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_lr.tif", mask_lr, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_hr.tif", mask_hr, compression="lzw")
			tifffile.imwrite(str(stem) + "_mask_conn.tif", mask_conn, compression="lzw")
			tifffile.imwrite(str(stem) + "_base_grid.tif", base_grid, compression="lzw")
			tifffile.imwrite(str(stem) + "_offset.tif", off, compression="lzw")
			tifffile.imwrite(str(stem) + "_mesh_offset.tif", mesh_off, compression="lzw")

	def mesh_offset_coarse(self) -> torch.Tensor:
		off = self.mesh_offset_ms[-1]
		for d in reversed(self.mesh_offset_ms[:-1]):
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
		mesh_off = self.mesh_offset_coarse()
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
		y = y.clamp(0.0, float(h - 1))
		y0 = torch.floor(y)
		y1 = y0 + 1.0
		t = (y - y0).clamp(0.0, 1.0)
		y0i = y0.clamp(0.0, float(h - 1)).to(dtype=torch.int64)
		y1i = y1.clamp(0.0, float(h - 1)).to(dtype=torch.int64)
		idx0 = y0i.view(n, 1, h, w).expand(n, c, h, w)
		idx1 = y1i.view(n, 1, h, w).expand(n, c, h, w)
		v0 = torch.take_along_dim(src, idx0, dim=2)
		v1 = torch.take_along_dim(src, idx1, dim=2)
		return v0 * (1.0 - t).view(n, 1, h, w) + v1 * t.view(n, 1, h, w)

	def grid_uv(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return base (u,v) mesh coordinates in pixel units."""
		uv = self.base_grid + self.offset_coarse()
		return uv[:, 0:1], uv[:, 1:2]

	def offset_coarse(self) -> torch.Tensor:
		off = self.offset_ms[-1]
		for d in reversed(self.offset_ms[:-1]):
			off = self._upsample2_crop(src=off, h_t=int(d.shape[2]), w_t=int(d.shape[3])) + d
		return off

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
			gh_i = max(2, gh_prev // 2 + 1)
			gw_i = max(2, gw_prev // 2 + 1)
			shapes.append((gh_i, gw_i))
		return [nn.Parameter(torch.zeros(1, 2, gh_i, gw_i, device=device, dtype=torch.float32)) for (gh_i, gw_i) in shapes]

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
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> "Model2D":
		h_img, w_img = data.size
		init = ModelInit(
			h_img=int(h_img),
			w_img=int(w_img),
			init_size_frac=float(init_size_frac),
			mesh_step_px=int(mesh_step_px),
			winding_step_px=int(winding_step_px),
		)
		return cls(
			init=init,
			device=device,
			subsample_mesh=subsample_mesh,
			subsample_winding=subsample_winding,
		)

	def _build_base_grid(self) -> torch.Tensor:
		# Model domain is initialized to a configurable fraction of the image extent.
		# Internal coordinates are stored in pixel units.
		w = float(max(1, int(self.init.w_img) - 1))
		h = float(max(1, int(self.init.h_img) - 1))
		f = float(self.init.init_size_frac)
		u0 = 0.5 * (1.0 - f) * w
		u1 = 0.5 * (1.0 + f) * w
		v0 = 0.5 * (1.0 - f) * h
		v1 = 0.5 * (1.0 + f) * h
		u = torch.linspace(u0, u1, self.mesh_w, dtype=torch.float32)
		v = torch.linspace(v0, v1, self.mesh_h, dtype=torch.float32)
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
