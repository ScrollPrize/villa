from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

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

		h2 = 2 * int(init.h_img)
		w2 = 2 * int(init.w_img)
		self.mesh_h = max(2, (h2 + int(init.mesh_step_px) - 1) // int(init.mesh_step_px) + 1)
		self.mesh_w = max(2, (w2 + int(init.winding_step_px) - 1) // int(init.winding_step_px) + 1)

		self.base_grid = self._build_base_grid().to(device=device)
		self.subsample_winding = int(subsample_winding)
		self.subsample_mesh = int(subsample_mesh)

	def direction_encoding(self, *, shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return (dir0, dir1) in the same encoding as the UNet outputs."""
		n, c, h, w = (int(v) for v in shape)
		if n <= 0 or h <= 0 or w <= 0:
			raise ValueError(f"invalid shape: {shape}")

		t = self.theta.to(dtype=torch.float32)
		cos2 = torch.cos(2.0 * t)
		sin2 = torch.sin(2.0 * t)
		inv_sqrt2 = 1.0 / (2.0 ** 0.5)
		dir0 = 0.5 + 0.5 * cos2
		dir1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
		dir0_t = dir0.view(1, 1, 1, 1).expand(n, 1, h, w)
		dir1_t = dir1.view(1, 1, 1, 1).expand(n, 1, h, w)
		return dir0_t, dir1_t

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
