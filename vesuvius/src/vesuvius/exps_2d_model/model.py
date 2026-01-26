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
		self.theta = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
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
		self.subsample_winding = int(subsample_winding)
		self.subsample_mesh = int(subsample_mesh)

	def direction_encoding(self, *, shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return (dir0, dir1) in the same encoding as the UNet outputs."""
		n, _c, h, w = (int(v) for v in shape)
		if n <= 0 or h <= 0 or w <= 0:
			raise ValueError(f"invalid shape: {shape}")

		x, y = self.grid_xy()

		# Direction is derived only from *vertical* mesh connections (v-edges).
		dvx = x.new_zeros(x.shape)
		dvy = y.new_zeros(y.shape)
		dvx[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
		dvy[:, :, :-1, :] = y[:, :, 1:, :] - y[:, :, :-1, :]
		if x.shape[2] >= 2:
			dvx[:, :, -1, :] = dvx[:, :, -2, :]
			dvy[:, :, -1, :] = dvy[:, :, -2, :]

		gx = -dvy
		gy = dvx
		eps = 1e-8
		r2 = gx * gx + gy * gy + eps
		cos2 = (gx * gx - gy * gy) / r2
		sin2 = (2.0 * gx * gy) / r2
		inv_sqrt2 = 1.0 / (2.0 ** 0.5)
		dir0 = 0.5 + 0.5 * cos2
		dir1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)

		if dir0.shape[-2:] != (h, w):
			dir0 = F.interpolate(dir0, size=(h, w), mode="bilinear", align_corners=True)
			dir1 = F.interpolate(dir1, size=(h, w), mode="bilinear", align_corners=True)

		if n != 1:
			dir0 = dir0.expand(n, 1, h, w)
			dir1 = dir1.expand(n, 1, h, w)
		return dir0, dir1

	def _apply_global_transform(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		u = self.winding_scale * u + self.phase
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
		}

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

	def grid_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return globally transformed mesh coordinates."""
		u, v = self.grid_uv()
		return self._apply_global_transform(u, v)

	def grid_xy_subsampled(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return globally transformed mesh coordinates, bilinear-upsampled to the subsampled eval grid."""
		x, y = self.grid_xy()
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
