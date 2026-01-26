from __future__ import annotations

from dataclasses import dataclass

import torch

import fit_data


@dataclass(frozen=True)
class ModelInit:
	h_img: int
	w_img: int
	mesh_step_px: int
	winding_step_px: int


class Model2D:
	def __init__(
		self,
		init: ModelInit,
		device: torch.device,
		*,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
	) -> None:
		self.init = init
		self.device = device

		h2 = 2 * int(init.h_img)
		w2 = 2 * int(init.w_img)
		self.mesh_h = max(2, (h2 + int(init.mesh_step_px) - 1) // int(init.mesh_step_px) + 1)
		self.mesh_w = max(2, (w2 + int(init.winding_step_px) - 1) // int(init.winding_step_px) + 1)

		self.base_grid = self._build_base_grid().to(device=device)
		self.subsample_winding = int(subsample_winding)
		self.subsample_mesh = int(subsample_mesh)

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
		u = torch.linspace(-1.0, 1.0, self.mesh_w, dtype=torch.float32)
		v = torch.linspace(-1.0, 1.0, self.mesh_h, dtype=torch.float32)
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
