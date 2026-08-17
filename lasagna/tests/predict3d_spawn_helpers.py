"""Small spawn-safe adapters used by shared tiled inference tests."""

from __future__ import annotations

import time
import os
from pathlib import Path

import numpy as np

import torch


class _ScaleIdentity(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.gain = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

	def forward(self, value: torch.Tensor) -> torch.Tensor:
		return value * self.gain


class SpawnIdentityAdapter:
	def __init__(self, product, *, delay_zero_origin: bool = False, marker_on_second_call: str | None = None):
		self._products = (product,)
		self.delay_zero_origin = bool(delay_zero_origin)
		self.marker_on_second_call = marker_on_second_call
		self.calls = 0

	@property
	def output_products(self):
		return self._products

	def load_model(self, *, device: torch.device):
		return _ScaleIdentity().to(device)

	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		self.calls += 1
		if self.calls == 2 and self.marker_on_second_call is not None:
			Path(self.marker_on_second_call).touch()
		if self.delay_zero_origin and float(tile[0, 0, 0, 0, 0]) == 0.0:
			time.sleep(0.25)
		return model(tile)

	def product_tensors_from_output(self, raw_output):
		return {self._products[0].name: raw_output}


class SpawnHardExitAdapter(SpawnIdentityAdapter):
	def run_tile_inference(self, model, tile: torch.Tensor, *, device: torch.device):
		os._exit(23)


class SpawnFileOutputAdapter:
	"""Spawn-safe output adapter whose writes are observable by the parent."""

	def __init__(self, root: str, *, wait_for_marker: str | None = None, fail: bool = False, hard_exit: bool = False, delay_s: float = 0.0):
		self.root = str(root)
		self.wait_for_marker = wait_for_marker
		self.fail = bool(fail)
		self.hard_exit = bool(hard_exit)
		self.delay_s = float(delay_s)

	def _path(self, origin) -> Path:
		return Path(self.root) / ("chunk_" + "_".join(str(int(v)) for v in origin) + ".npy")

	def product_chunk_complete(self, product, *, chunk_origin_zyx):
		return self._path(chunk_origin_zyx).exists()

	def write_product_chunk(self, product, *, chunk_origin_zyx, data):
		if self.hard_exit:
			os._exit(24)
		if self.fail:
			raise OSError("forced process flush failure")
		if self.delay_s > 0.0:
			time.sleep(self.delay_s)
		if self.wait_for_marker is not None and int(chunk_origin_zyx[0]) == 0:
			deadline = time.monotonic() + 5.0
			while not Path(self.wait_for_marker).exists():
				if time.monotonic() >= deadline:
					raise TimeoutError("next inference band did not start during flush")
				time.sleep(0.01)
		Path(self.root).mkdir(parents=True, exist_ok=True)
		payload = np.asarray(data[next(iter(data))])
		np.save(self._path(chunk_origin_zyx), payload)
		self._path(chunk_origin_zyx).with_suffix(".pid").write_text(str(os.getpid()))
