"""Loss terms whose backward cannot run on a device are rejected before optimization."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import optimizer


def _stages(winding_vol: float, *, wrap: bool = False) -> list[optimizer.Stage]:
	cfg = {
		"base": {"normal": 1.0, "step": 10.0, "winding_vol": winding_vol},
		"stages": [{"name": "s1", "steps": 10, "lr": 0.1, "params": ["mesh_ms"], "min_scaledown": 0}],
	}
	stages = optimizer.load_stages_cfg(cfg)
	if wrap:
		return [optimizer.Stage(name="expand-z", global_opt=None, children=tuple(stages))]
	return stages


@pytest.fixture
def mps_backward(monkeypatch):
	"""Set whether this PyTorch has the MPS 5D grid_sample backward, without touching a GPU."""
	def set_available(available: bool) -> None:
		monkeypatch.setattr(optimizer, "_mps_grid_sample_3d_backward", lambda: available)
	return set_available


@pytest.mark.parametrize("wrap", [False, True])
def test_winding_vol_is_rejected_on_mps_without_the_backward(mps_backward, wrap):
	mps_backward(False)
	with pytest.raises(ValueError, match=r"stage 's1' weights the winding_vol loss.*--device cpu or cuda"):
		optimizer.require_device_support(_stages(1.0, wrap=wrap), torch.device("mps"))


def test_winding_vol_runs_on_mps_with_the_backward(mps_backward):
	mps_backward(True)
	optimizer.require_device_support(_stages(1.0), torch.device("mps"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_winding_vol_is_allowed_off_mps(mps_backward, device):
	mps_backward(False)
	optimizer.require_device_support(_stages(1.0), torch.device(device))


def test_mps_fit_without_winding_vol_is_allowed(mps_backward):
	mps_backward(False)
	optimizer.require_device_support(_stages(0.0), torch.device("mps"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_probe_matches_the_winding_vol_sample_on_mps():
	# the probe must agree with what the loss's own 5D sample does on this PyTorch
	vol = torch.rand(1, 1, 8, 8, 8, device="mps")
	grid = (torch.rand(1, 2, 3, 4, 3, device="mps") * 2 - 1).requires_grad_(True)
	try:
		F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True).sum().backward()
		runs = True
	except NotImplementedError:
		runs = False
	optimizer._mps_grid_sample_3d_backward.cache_clear()
	assert optimizer._mps_grid_sample_3d_backward() == runs
