"""Small spawn-safe adapters used by shared tiled inference tests."""

from __future__ import annotations

import time
import os

import torch


class _ScaleIdentity(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.gain = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

	def forward(self, value: torch.Tensor) -> torch.Tensor:
		return value * self.gain


class SpawnIdentityAdapter:
	def __init__(self, product, *, delay_zero_origin: bool = False):
		self._products = (product,)
		self.delay_zero_origin = bool(delay_zero_origin)

	@property
	def output_products(self):
		return self._products

	def load_model(self, *, device: torch.device):
		return _ScaleIdentity().to(device)

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		if self.delay_zero_origin and float(tile[0, 0, 0, 0, 0]) == 0.0:
			time.sleep(0.25)
		return model(tile)

	def product_tensors_from_output(self, raw_output):
		return {self._products[0].name: raw_output}


class SpawnHardExitAdapter(SpawnIdentityAdapter):
	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		os._exit(23)
