"""Parity of the fused Metal mesh losses against their PyTorch code in float64 on the CPU.

Step, bend and smooth: forward map and gradient (for a random upstream gradient, not only the mean's) on
jittered meshes with short, long and collapsed edges and a sharp fold, so every branch runs: the step penalty's
directional terms and its fallback for edges whose direction is undefined, the 2-tap direction averages of meshes
two vertices wide, and both sides of the bend threshold. Variants with a wrong constant must fail the same bars.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import mesh_loss_metal
import opt_loss_bend
import opt_loss_smooth
import opt_loss_step

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="needs an Apple GPU")

FWD_REL, FWD_ABS = 1e-4, 1e-6
BWD_REL = 1e-4  # of the largest gradient component


def _step_map(res):
	return opt_loss_step.step_loss_maps(res=res)[0]


def _bend_map(res):
	return opt_loss_bend.bend_loss(res=res)[1][0]


def _smooth_map(res):
	return opt_loss_smooth.smooth_loss(res=res)[1][0]


# name -> (loss map, smallest H and W the Metal path takes, size of the map relative to the mesh)
LOSSES = {"step": (_step_map, 2, 1), "bend": (_bend_map, 3, 2), "smooth": (_smooth_map, 3, 2)}


def _mesh(D: int, H: int, W: int, step: float, seed: int) -> torch.Tensor:
	gen = torch.Generator().manual_seed(seed)
	ii, jj = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
	base = torch.stack([jj * step, ii * step, torch.zeros_like(ii)], dim=-1)
	xyz = base.unsqueeze(0).repeat(D, 1, 1, 1)
	# a wide jitter so edges land on both sides of the target and the short term's relu is active
	xyz = xyz + torch.randn(xyz.shape, generator=gen, dtype=torch.float64) * (0.35 * step)
	if H >= 3 and W >= 3:
		# a collapsed anti-diagonal (quad 1, 1): direction undefined, the fallback step penalty applies
		xyz[0, 2, 1] = xyz[0, 1, 2]
		# a collapsed H edge: one invalid tap inside a direction average
		xyz[0, 2, 0] = xyz[0, 1, 0]
	if H >= 5 and W >= 5:
		# a sharp fold: the vertex at (3, 3) pulled back over its neighbour, so the bend penalty is active there
		xyz[-1, 3, 3] = xyz[-1, 2, 3] + torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64) * step
	return xyz


def _res(x: torch.Tensor, step: float):
	return SimpleNamespace(xyz_lr=x, params=SimpleNamespace(mesh_step=step))


def _reference(loss: str, xyz64: torch.Tensor, step: float, grad_out: torch.Tensor):
	x = xyz64.clone().requires_grad_(True)
	lm = LOSSES[loss][0](_res(x, step))
	(lm * grad_out).sum().backward()
	return lm.detach(), x.grad


def _metal(loss: str, xyz64: torch.Tensor, step: float, grad_out: torch.Tensor):
	x = xyz64.to(torch.float32).to("mps").requires_grad_(True)
	assert mesh_loss_metal.enabled(x, LOSSES[loss][1])
	lm = LOSSES[loss][0](_res(x, step))
	assert lm.shape == grad_out.shape
	(lm * grad_out.to(torch.float32).to("mps")).sum().backward()
	return lm.detach().cpu().double(), x.grad.cpu().double()


def _check(lm, grad, lm_ref, grad_ref) -> None:
	assert lm.shape == lm_ref.shape and grad.shape == grad_ref.shape
	assert torch.isfinite(lm).all() and torch.isfinite(grad).all()
	torch.testing.assert_close(lm, lm_ref, rtol=FWD_REL, atol=FWD_ABS)
	scale = float(grad_ref.abs().max())
	assert scale > 0.0
	torch.testing.assert_close(grad, grad_ref, rtol=0.0, atol=BWD_REL * scale)


def _grad_out(loss: str, D: int, H: int, W: int, seed: int) -> torch.Tensor:
	k = LOSSES[loss][2]
	gen = torch.Generator().manual_seed(100 + seed)
	return torch.rand(D, 1, H - k, W - k, generator=gen, dtype=torch.float64)


CASES = [
	(2, 7, 9, 20.0, 0),
	(1, 12, 5, 4.0, 1),
	(1, 2, 2, 20.0, 2),  # one quad: 2-tap directions in both axes
	(1, 2, 6, 20.0, 3),
	(3, 5, 3, 8.0, 4),
	(1, 3, 3, 20.0, 5),  # one interior vertex
]


@needs_mps
@pytest.mark.parametrize("loss", sorted(LOSSES))
@pytest.mark.parametrize("D,H,W,step,seed", CASES)
def test_metal_loss_matches_reference(loss, D, H, W, step, seed):
	if min(H, W) < LOSSES[loss][1]:
		pytest.skip("mesh below the Metal path's size; the PyTorch path runs")
	xyz = _mesh(D, H, W, step, seed)
	grad_out = _grad_out(loss, D, H, W, seed)
	lm_ref, grad_ref = _reference(loss, xyz, step, grad_out)
	lm, grad = _metal(loss, xyz, step, grad_out)
	_check(lm, grad, lm_ref, grad_ref)


@needs_mps
@pytest.mark.parametrize("loss", sorted(LOSSES))
def test_metal_loss_mean_backward(loss):
	# the fit's reduction: lm.mean() hands backward an expanded (stride 0) gradient
	xyz = _mesh(1, 6, 6, 20.0, 7)
	x = xyz.to(torch.float32).to("mps").requires_grad_(True)
	lm = LOSSES[loss][0](_res(x, 20.0))
	lm.mean().backward()
	lm_ref, grad_ref = _reference(loss, xyz, 20.0, torch.full(lm.shape, 1.0 / lm.numel(), dtype=torch.float64))
	_check(lm.detach().cpu().double(), x.grad.cpu().double(), lm_ref, grad_ref)


def test_branches_are_exercised():
	# CPU only: the meshes above reach the branches the parity cases claim to cover
	xyz = _mesh(1, 6, 6, 20.0, 7)
	# the collapsed anti-diagonal's fallback is ((0 - t) / t)^2 = 1, a quarter of the quad's average
	assert float(_step_map(_res(xyz, 20.0))[0, 0, 1, 1]) >= 0.25
	bend = _bend_map(_res(_mesh(2, 7, 9, 20.0, 0), 20.0))
	assert int((bend > 0).sum()) >= 1 and int((bend == 0).sum()) >= 1


@needs_mps
@pytest.mark.parametrize("loss", sorted(LOSSES))
def test_wrong_constant_fails_the_bars(loss, monkeypatch):
	xyz = _mesh(2, 7, 9, 20.0, 0)
	grad_out = _grad_out(loss, 2, 7, 9, 0)
	lm_ref, grad_ref = _reference(loss, xyz, 20.0, grad_out)
	if loss == "step":
		def wrong(xyz_t, mesh_step):
			t, tsq = mesh_loss_metal._rounded_target(mesh_step)
			# diagonals held to the axis target instead of mesh_step * sqrt(2)
			return mesh_loss_metal._StepLossMap.apply(xyz_t, [t, tsq, t, tsq])

		monkeypatch.setattr(mesh_loss_metal, "step_loss_map", wrong)
	else:
		# the bend threshold or smooth's step^2 off by 1%
		monkeypatch.setattr(mesh_loss_metal, "_f32", lambda v: float(v) * 1.01)
	lm, grad = _metal(loss, xyz, 20.0, grad_out)
	with pytest.raises(AssertionError):
		_check(lm, grad, lm_ref, grad_ref)


@needs_mps
def test_metal_disabled_uses_pytorch(monkeypatch):
	monkeypatch.setenv("LASAGNA_METAL", "0")
	assert not mesh_loss_metal.enabled(torch.zeros(1, 3, 3, 3, device="mps"), 2)
	monkeypatch.delenv("LASAGNA_METAL")
	assert mesh_loss_metal.enabled(torch.zeros(1, 3, 3, 3, device="mps"), 3)
	assert not mesh_loss_metal.enabled(torch.zeros(1, 2, 3, 3, device="mps"), 3)
	assert not mesh_loss_metal.enabled(torch.zeros(1, 3, 3, 3), 2)
