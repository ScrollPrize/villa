from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import fit_data


@dataclass(frozen=True)
class ModelInit:
	h_img: int
	w_img: int
	mesh_step_px: int
	winding_step_px: int


@dataclass(frozen=True)
class FitResult:
	_xy_lr: torch.Tensor
	_xy_hr: torch.Tensor
	_xy_conn: torch.Tensor
	_data_s: fit_data.FitData
	_mask: torch.Tensor
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
	def mask(self) -> torch.Tensor:
		return self._mask

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

		h2 = 2 * int(init.h_img)
		w2 = 2 * int(init.w_img)
		self.mesh_h = max(2, (h2 + int(init.mesh_step_px) - 1) // int(init.mesh_step_px) + 1)
		self.mesh_w = max(2, (w2 + int(init.winding_step_px) - 1) // int(init.winding_step_px) + 1)

		self.base_grid = self._build_base_grid().to(device=device)
		offset_scales = 5
		self.offset_ms = nn.ParameterList(
			self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.mesh_offset_ms = nn.ParameterList(
			self._build_offset_ms(offset_scales=offset_scales, device=device, gh0=self.mesh_h, gw0=self.mesh_w)
		)
		self.subsample_winding = int(subsample_winding)
		self.subsample_mesh = int(subsample_mesh)

	def forward(self, data: fit_data.FitData) -> FitResult:
		xy_lr = torch.cat(self._grid_xy(), dim=1)
		xy_hr = torch.cat(self._grid_xy_subsampled(), dim=1)
		xy_conn = self._xy_conn_px(xy_lr=xy_lr)
		data_s, mask = data.grid_sample_xy(xy=xy_hr)
		return FitResult(
			_xy_lr=xy_lr,
			_xy_hr=xy_hr,
			_xy_conn=xy_conn,
			_data_s=data_s,
			_mask=mask,
			_h_img=int(self.init.h_img),
			_w_img=int(self.init.w_img),
			_mesh_step_px=int(self.init.mesh_step_px),
			_winding_step_px=int(self.init.winding_step_px),
		)

	def _apply_global_transform(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		period = 4.0 / float(max(1, self.mesh_w - 1))
		phase = torch.remainder(self.phase + 0.5 * period, period) - 0.5 * period
		u = self.winding_scale * u + phase
		c = torch.cos(self.theta)
		s = torch.sin(self.theta)
		x = c * u - s * v
		y = s * u + c * v
		return x, y

	def opt_params(self) -> dict[str, list[nn.Parameter]]:
		return {
			"theta": [self.theta],
			"phase": [self.phase],
			"winding_scale": [self.winding_scale],
			"offset_ms": list(self.offset_ms),
			"mesh_offset_ms": list(self.mesh_offset_ms),
		}

	def mesh_offset_coarse(self) -> torch.Tensor:
		off = self.mesh_offset_ms[-1]
		for d in reversed(self.mesh_offset_ms[:-1]):
			off = F.interpolate(off, size=(int(d.shape[2]), int(d.shape[3])), mode="bilinear", align_corners=True) + d
		return off

	def _xy_conn_px(self, *, xy_lr: torch.Tensor) -> torch.Tensor:
		"""Return per-mesh connection positions in pixel coordinates.

		For each base-mesh point, returns 3 pixel positions:
		- left-connection interpolation result (to previous column)
		- the point itself
		- right-connection interpolation result (to next column)

		Shape: (N,3,2,Hm,Wm)
		"""
		if xy_lr.ndim != 4 or int(xy_lr.shape[1]) != 2:
			raise ValueError("xy_lr must be (N,2,H,W)")
		n, _c, hm, wm = (int(v) for v in xy_lr.shape)
		if hm <= 0 or wm <= 0:
			raise ValueError("invalid xy_lr shape")
		mesh_off = self.mesh_offset_coarse()
		if mesh_off.shape[-2:] != (hm, wm):
			raise RuntimeError("mesh_offset must be defined on the base mesh")

		w = float(max(1, int(self.init.w_img) - 1))
		h = float(max(1, int(self.init.h_img) - 1))
		x_px = (xy_lr[:, 0:1] + 1.0) * 0.5 * w
		y_px = (xy_lr[:, 1:2] + 1.0) * 0.5 * h
		xy_px = torch.cat([x_px, y_px], dim=1)

		left_src = torch.cat([xy_px[:, :, :, 0:1], xy_px[:, :, :, :-1]], dim=3)
		right_src = torch.cat([xy_px[:, :, :, 1:], xy_px[:, :, :, -1:]], dim=3)
		off_l = mesh_off[:, 0]
		off_r = mesh_off[:, 1]
		base_i = torch.arange(hm, device=xy_lr.device, dtype=xy_lr.dtype).view(1, hm, 1)
		left_conn = self._interp_col(src=left_src, y=base_i + off_l)
		right_conn = self._interp_col(src=right_src, y=base_i + off_r)
		return torch.stack([left_conn, xy_px, right_conn], dim=1)

	@staticmethod
	def _interp_col(*, src: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""Interpolate `src` along height dimension at (possibly fractional) indices `y`.

		src: (N,2,H,W)
		y: (N,H,W)
		"""
		n, c, h, w = (int(v) for v in src.shape)
		if y.shape != (n, h, w):
			raise ValueError("y must be (N,H,W)")
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
		"""Return base (u,v) mesh coordinates."""
		uv = self.base_grid + self.offset_coarse()
		return uv[:, 0:1], uv[:, 1:2]

	def offset_coarse(self) -> torch.Tensor:
		off = self.offset_ms[-1]
		for d in reversed(self.offset_ms[:-1]):
			off = F.interpolate(off, size=(int(d.shape[2]), int(d.shape[3])), mode="bilinear", align_corners=True) + d
		return off

	def _build_offset_ms(self, *, offset_scales: int, device: torch.device, gh0: int, gw0: int) -> list[nn.Parameter]:
		gh0 = int(gh0)
		gw0 = int(gw0)
		n_scales = max(1, int(offset_scales))
		shapes: list[tuple[int, int]] = [(gh0, gw0)]
		for _ in range(1, n_scales):
			gh_prev, gw_prev = shapes[-1]
			gh_i = max(2, (gh_prev - 1) // 2 + 1)
			gw_i = max(2, (gw_prev - 1) // 2 + 1)
			shapes.append((gh_i, gw_i))
		return [nn.Parameter(torch.zeros(1, 2, gh_i, gw_i, device=device, dtype=torch.float32)) for (gh_i, gw_i) in shapes]

	def _grid_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return globally transformed mesh coordinates."""
		u, v = self.grid_uv()
		return self._apply_global_transform(u, v)

	def _grid_xy_subsampled(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return globally transformed mesh coordinates, bilinear-upsampled to the subsampled eval grid."""
		x, y = self._grid_xy()
		h = max(2, (self.mesh_h - 1) * self.subsample_mesh + 1)
		w = max(2, (self.mesh_w - 1) * self.subsample_winding + 1)
		x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
		y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
		return x, y

	@classmethod
	def from_fit_data(
		cls,
		data: "fit_data.FitData",
		mesh_step_px: int,
		winding_step_px: int,
		device: torch.device,
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> "Model2D":
		h_img, w_img = data.size
		init = ModelInit(
			h_img=int(h_img),
			w_img=int(w_img),
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
		# Model domain is initialized to ~2x image extent in each dimension, so the
		# canonical winding-space coordinates span [-2,2] (image spans [-1,1]).
		u = torch.linspace(-2.0, 2.0, self.mesh_w, dtype=torch.float32)
		v = torch.linspace(-2.0, 2.0, self.mesh_h, dtype=torch.float32)
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
		us = torch.linspace(-1.0, 1.0, w, device=self.device, dtype=torch.float32)
		vs = torch.linspace(-1.0, 1.0, h, device=self.device, dtype=torch.float32)
		vv, uu = torch.meshgrid(vs, us, indexing="ij")
		u = uu.view(1, 1, h, w)

		phase = 0.5 * torch.pi * (u + 1.0) * periods
		return 0.5 + 0.5 * torch.cos(phase)
