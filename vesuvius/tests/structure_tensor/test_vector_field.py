import torch
import pytest

from vesuvius.structure_tensor.smooth_vf import VectorFieldModule


def test_vector_field_module_preserves_constants_cpu():
    # With zero-padding conv, a normalized Gaussian preserves constants on the
    # *interior* (far from borders). Border voxels attenuate due to zeros.
    xi = 1.5  # radius = int(3*xi) = 4
    m = VectorFieldModule(xi=xi, device=torch.device("cpu"))
    Dz, Dy, Dx = 21, 21, 21  # large enough so interior exists
    x = torch.ones((1, 3, Dz, Dy, Dx), dtype=torch.float32)
    with torch.no_grad():
        y = m.smooth(x)
    assert y.shape == x.shape
    radius = int(3 * xi)
    # validate only the "valid" interior region
    interior = (
        slice(None), slice(None),
        slice(radius, -radius),
        slice(radius, -radius),
        slice(radius, -radius),
    )
    assert torch.allclose(y[interior], x[interior], atol=1e-6)


def test_vector_field_module_channel_separation():
    # Grouped conv should not mix channels.
    xi = 1.0
    m = VectorFieldModule(xi=xi, device=torch.device("cpu"))
    Dz, Dy, Dx = 7, 7, 7
    x = torch.zeros((1, 3, Dz, Dy, Dx), dtype=torch.float32)
    # Impulse in channel 0 only
    x[0, 0, Dz // 2, Dy // 2, Dx // 2] = 1.0
    with torch.no_grad():
        y = m.smooth(x)
    # Other channels must stay zero
    assert torch.allclose(y[0, 1], torch.zeros_like(y[0, 1]))
    assert torch.allclose(y[0, 2], torch.zeros_like(y[0, 2]))
    # Channel 0 should be a normalized blob (sum ~ 1)
    assert torch.isclose(y[0, 0].sum(), torch.tensor(1.0), atol=1e-5)


def test_compute_vf_cli_default_device_runs_without_cuda(tmp_path, monkeypatch):
    # main passed torch.device("cuda") by default, so a CPU-only machine failed at the first block.
    import numpy as np
    from vc_reader.reader import ZarrWriter, create_group, open_zarr
    from vesuvius.structure_tensor import compute_vf

    Z = Y = X = 8
    ZarrWriter.create(str(tmp_path / "labels.zarr"), shape=(Z, Y, X), chunks=(Z, Y, X), dtype=np.uint8)[...] = 1
    ev = np.zeros((9, Z, Y, X), dtype=np.float32)
    ev[0] = ev[4] = ev[8] = 1.0  # u = x, v = y, n = z
    create_group(str(tmp_path / "eig.zarr")).create_array(
        "eigenvectors", shape=ev.shape, chunks=(1, Z, Y, X), dtype=ev.dtype)[...] = ev
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("sys.argv", [
        "compute_vf", "--input-zarr", str(tmp_path / "labels.zarr"), "--eigen", f"1:{tmp_path / 'eig.zarr'}",
        "--output-zarr", str(tmp_path / "vf.zarr"), "--chunk-size", "8,8,8", "--xi", "1.0"])
    compute_vf.main()
    v = open_zarr(str(tmp_path / "vf.zarr/V"))[...]
    assert v.shape == (3, Z, Y, X)
    assert np.all(v[1] > 0) and np.all(v[0] == 0) and np.all(v[2] == 0)
