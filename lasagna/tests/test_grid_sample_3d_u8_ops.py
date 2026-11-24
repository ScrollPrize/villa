"""Dense uint8 grid_sample in torch ops (MPS without Metal) and the MPS dense-sampler choice (LASAGNA_METAL).

Reference: F.grid_sample (align_corners=True, zeros padding) on the CPU, where it has a 3D backward. Volume
sizes are 2^k + 1, so its coordinate normalisation is a power-of-two scaling; tolerance 2e-3 on values in
[0, 255] (float32 fractions), gradients compared off voxel boundaries (F.grid_sample's normalisation can round a
boundary point into the other cell).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(ROOT) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ops = _load("grid_sample_3d_u8_ops")


def _case(seed=0, C=2, Z=17, Y=33, X=65, D=3, H=11, W=13):
    gen = torch.Generator().manual_seed(seed)
    volume = torch.randint(0, 256, (C, Z, Y, X), generator=gen, dtype=torch.uint8)
    offset = torch.tensor([5.0, -3.0, 7.0])
    inv_scale = torch.tensor([0.5, 0.25, 2.0])
    size = torch.tensor([X, Y, Z], dtype=torch.float32)
    local = torch.rand(D, H, W, 3, generator=gen) * (size + 2.0) - 1.0  # about one voxel past every face
    return volume, local / inv_scale + offset, offset, inv_scale


def _reference(volume, grid, offset, inv_scale):
    C, Z, Y, X = volume.shape
    g = (grid - offset) * inv_scale / torch.tensor([X - 1, Y - 1, Z - 1], dtype=torch.float32) * 2 - 1
    return F.grid_sample(volume.float().unsqueeze(0), g.unsqueeze(0), mode="bilinear",
                         padding_mode="zeros", align_corners=True).squeeze(0)


@pytest.mark.parametrize("device", DEVICES)
def test_values_and_grad_match_f_grid_sample(device):
    volume, grid, offset, inv_scale = _case()
    g_ref = grid.clone().requires_grad_(True)
    ref = _reference(volume, g_ref, offset, inv_scale)
    go = torch.randn(ref.shape, generator=torch.Generator().manual_seed(1))
    (ref * go).sum().backward()
    args = [t.to(device) for t in (volume, grid, offset, inv_scale)]
    got = ops.sample_f32(*args).cpu()
    torch.testing.assert_close(got, ref.detach(), rtol=0, atol=2e-3)
    grad = ops.sample_grad(*args, go.to(device)).cpu()
    p = ((grid - offset) * inv_scale).reshape(-1, 3)
    off_boundary = ((p - p.round()).abs() > 1e-3).all(dim=1).view(*grid.shape[:3], 1)
    scale = float(inv_scale.max()) * float(go.abs().max()) * volume.shape[0]
    torch.testing.assert_close(torch.where(off_boundary, grad, 0.0), torch.where(off_boundary, g_ref.grad, 0.0),
                               rtol=1e-4, atol=2e-3 * scale)
    u8 = ops.sample_u8(*args).cpu().float()
    f = got.double()
    assert torch.equal(u8, torch.floor(f + 0.5).clamp(0, 255).float())


@pytest.mark.parametrize("device", DEVICES)
def test_half_rounds_away_and_non_finite_reads_zero(device):
    volume = torch.zeros(1, 2, 2, 2, dtype=torch.uint8)
    volume[0, :, :, 0] = 10
    volume[0, :, :, 1] = 11
    grid = torch.tensor([[[[0.5, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.5, float("inf"), 0.0]]]])
    args = [t.to(device) for t in (volume, grid, torch.zeros(3), torch.ones(3))]
    assert ops.sample_u8(*args).cpu().flatten().tolist() == [11, 0, 0]
    assert ops.sample_f32(*args).cpu().flatten().tolist() == [10.5, 0.0, 0.0]
    g = ops.sample_grad(*args, torch.ones(1, 1, 1, 3, device=device)).cpu()
    assert g[0, 0, 1].tolist() == [0.0, 0.0, 0.0] and g[0, 0, 2].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_torch_ops_agree_with_metal_and_cpu():
    metal = _load("grid_sample_3d_u8_metal")
    volume, grid, offset, inv_scale = _case(seed=3)
    go = torch.randn(volume.shape[0], *grid.shape[:3], generator=torch.Generator().manual_seed(4))
    m = [t.to("mps") for t in (volume, grid, offset, inv_scale)]
    torch.testing.assert_close(ops.sample_f32(*m).cpu(), ops.sample_f32(volume, grid, offset, inv_scale),
                               rtol=0, atol=0)
    torch.testing.assert_close(ops.sample_f32(*m).cpu(), metal.grid_sample_3d_u8_diff(*m).detach().cpu(),
                               rtol=0, atol=1e-4)
    g_metal = m[1].clone().requires_grad_(True)
    (metal.grid_sample_3d_u8_diff(m[0], g_metal, m[2], m[3]) * go.to("mps")).sum().backward()
    torch.testing.assert_close(ops.sample_grad(*m, go.to("mps")).cpu(), g_metal.grad.cpu(), rtol=1e-5, atol=1e-3)


def _wrappers():
    return _load("grid_sample_3d_u8"), _load("grid_sample_3d_u8_diff")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_metal_off_routes_mps_wrappers_to_torch_ops_with_a_backward(monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "0")
    u8, diff = _wrappers()
    volume, grid, offset, inv_scale = _case(seed=5)
    m = [t.to("mps") for t in (volume, grid, offset, inv_scale)]
    assert diff._ops().mps_dense_backend()[1] == "torch (LASAGNA_METAL=0)"
    g = m[1].clone().requires_grad_(True)
    out = diff.grid_sample_3d_u8_diff(m[0], g, m[2], m[3])
    out.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    torch.testing.assert_close(out.detach().cpu(), ops.sample_f32(volume, grid, offset, inv_scale), rtol=0, atol=0)
    assert torch.equal(u8.grid_sample_3d_u8(*m).cpu(), ops.sample_u8(volume, grid, offset, inv_scale))


def test_wrappers_share_one_ops_module_when_executed_afresh():
    # fit_data executes the wrapper files afresh on every call. The backend choice and the compiled Metal library
    # live in the one imported grid_sample_3d_u8_ops, so they are made once per process.
    modules = {id(w._ops()) for _ in range(2) for w in _wrappers()}
    assert modules == {id(sys.modules["grid_sample_3d_u8_ops"])}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_dense_metal_library_compiles_once_per_process(monkeypatch, capsys):
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    compiled = []
    compile_shader = torch.mps.compile_shader
    monkeypatch.setattr(torch.mps, "compile_shader", lambda src: compiled.append(1) or compile_shader(src))
    volume, grid, offset, inv_scale = _case(seed=6)
    m = [t.to("mps") for t in (volume, grid, offset, inv_scale)]
    for _ in range(3):
        u8, diff = _wrappers()  # as fit_data loads them on every call
        u8.grid_sample_3d_u8(*m)
        diff.grid_sample_3d_u8_diff(*m)
        assert diff._ops().mps_dense_backend()[1] == "metal"
    torch.mps.synchronize()
    assert len(compiled) <= 1  # 0 when an earlier test of this process compiled it
    assert capsys.readouterr().out.count("MPS dense sampler: metal") <= 1


def test_metal_failure_falls_back_unless_required(monkeypatch):
    monkeypatch.setattr(ops, "_metal_dense", None)
    monkeypatch.setattr(ops, "_metal_dense_error", "RuntimeError: shader failed")
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    impl, label = ops.mps_dense_backend()
    assert impl is ops._TorchDense and label == "torch (metal unavailable: RuntimeError: shader failed)"
    monkeypatch.setenv("LASAGNA_METAL", "1")
    with pytest.raises(RuntimeError, match="dense Metal sampler is unavailable"):
        ops.mps_dense_backend()
