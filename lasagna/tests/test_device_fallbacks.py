"""Paths that had no implementation off CUDA: the dense uint8 sampler on CPU, and the
corr-point ROI normals (which loaded the whole volume densely off CUDA)."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from vc_reader.reader import ZarrWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(ROOT) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gs_u8 = _load("grid_sample_3d_u8")


def _reference(volume, grid, offset, inv_scale):
    C, Z, Y, X = volume.shape
    p = (grid - offset) * inv_scale
    g = p / torch.tensor([X - 1, Y - 1, Z - 1], dtype=torch.float32) * 2 - 1
    return F.grid_sample(volume.float().unsqueeze(0), g.unsqueeze(0), mode="bilinear",
                         padding_mode="zeros", align_corners=True).squeeze(0)


def test_dense_u8_on_cpu_rounds_like_the_kernel():
    gen = torch.Generator().manual_seed(0)
    volume = torch.randint(0, 256, (2, 17, 33, 65), generator=gen, dtype=torch.uint8)
    offset = torch.tensor([5.0, -3.0, 7.0])
    inv_scale = torch.tensor([0.5, 0.25, 2.0])
    size = torch.tensor([65.0, 33.0, 17.0])
    grid = (torch.rand(3, 11, 13, 3, generator=gen) * (size + 2.0) - 1.0) / inv_scale + offset
    got = gs_u8.grid_sample_3d_u8(volume, grid, offset, inv_scale)
    ref = _reference(volume, grid, offset, inv_scale)
    assert got.dtype == torch.uint8 and got.shape == ref.shape
    want = torch.floor(ref + 0.5).clamp(0, 255)
    near_half = ((ref - torch.floor(ref)) - 0.5).abs() < 1e-3
    assert torch.equal(got.float()[~near_half], want[~near_half])
    assert float((got.float() - want).abs().max()) <= 1.0


def test_dense_u8_on_cpu_half_away_from_zero_and_nan():
    volume = torch.zeros(1, 2, 2, 2, dtype=torch.uint8)
    volume[0, :, :, 0] = 10
    volume[0, :, :, 1] = 11
    grid = torch.tensor([[[[0.5, 0.0, 0.0], [0.5, 1.0, 1.0], [float("nan"), 0.0, 0.0]]]])
    got = gs_u8.grid_sample_3d_u8(volume, grid, torch.zeros(3), torch.ones(3))
    assert got.flatten().tolist() == [11, 11, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_dense_u8_one_voxel_axis_reads_zero_past_the_voxel(device):
    # As the kernel: on a one-voxel z axis, z = 0.5 has one valid corner (weight 0.5) and points past the
    # volume read 0. F.grid_sample's align_corners normalisation would map every z to the voxel (200 each).
    volume = torch.full((1, 1, 4, 4), 200, dtype=torch.uint8)
    grid = torch.tensor([[[[1.0, 1.0, z] for z in (0.0, 0.5, 5.0, -3.0)]]])
    args = [t.to(device) for t in (volume, grid, torch.zeros(3), torch.ones(3))]
    assert gs_u8.grid_sample_3d_u8(*args).cpu().flatten().tolist() == [200, 100, 0, 0]


def _write_normals_manifest(root: Path, values: np.ndarray) -> Path:
    ZarrWriter.create(root / "surface.zarr", shape=values.shape, chunks=(3, 16, 16, 16), dtype="u1")[...] = values
    path = root / "vol.lasagna.json"
    path.write_text(json.dumps({
        "version": 2, "source_to_base": 1.0, "base_shape_zyx": list(values.shape[1:]),
        "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 1.0,
        "groups": {"surface": {"zarr": "surface.zarr", "scaledown": 0, "channels": ["grad_mag", "nx", "ny"]}},
    }) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("device", DEVICES)
def test_corr_point_roi_normals_stream_on_every_device(tmp_path, device):
    import fit
    import fit_data

    rng = np.random.default_rng(1)
    values = rng.integers(0, 256, (3, 40, 50, 60), dtype=np.uint8)
    values[1:] = rng.integers(60, 196, (2, 40, 50, 60), dtype=np.uint8)  # |nx|, |ny| < 0.53: nz stays real
    manifest = _write_normals_manifest(tmp_path, values)
    dev = torch.device(device)
    with mock.patch.object(fit_data, "load_3d", side_effect=AssertionError("dense whole-volume load")):
        data = fit._load_corr_point_roi_normal_data(input_path=str(manifest), device=dev)
    assert data.sparse_caches and data.nx is None
    points = torch.tensor([[12.0, 30.0, 20.0, 0.0], [59.0, 49.0, 39.0, 1.0], [0.0, 0.0, 0.0, 2.0]])
    normals = fit._sample_corr_point_roi_normals(
        data=data, corr_points=SimpleNamespace(points_xyz_winda=points), device=dev).cpu()
    for i, (x, y, z, _) in enumerate(points.tolist()):
        nx = (float(values[1, int(z), int(y), int(x)]) - 128.0) / 127.0
        ny = (float(values[2, int(z), int(y), int(x)]) - 128.0) / 127.0
        n = np.array([nx, ny, np.sqrt(max(1.0 - nx * nx - ny * ny, 1e-8))])
        n /= np.linalg.norm(n)
        np.testing.assert_allclose(normals[i].numpy(), n, atol=1e-5)


def test_mps_request_without_an_apple_gpu_fails_clearly(monkeypatch):
    import fit

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS device was requested"):
        fit._require_torch_device_available(torch.device("mps"))
    fit._require_torch_device_available(torch.device("cpu"))


def test_sync_debug_synchronizes_mps_when_there_is_no_cuda(monkeypatch):
    import optimizer

    calls = []
    monkeypatch.setenv("LASAGNA_SYNC_DEBUG", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    optimizer._debug_cuda_sync("step")
    assert calls == ["mps"]

    def _fail():
        raise RuntimeError("command buffer error")
    monkeypatch.setattr(torch.mps, "synchronize", _fail)
    with pytest.raises(RuntimeError, match="MPS failure after step"):
        optimizer._debug_cuda_sync("step")


def test_cuda_request_on_a_mac_points_to_mps(monkeypatch):
    import fit

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="pass --device mps"):
        fit._require_torch_device_available(torch.device("cuda"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as info:
        fit._require_torch_device_available(torch.device("cuda"))
    assert "--device mps" not in str(info.value)
