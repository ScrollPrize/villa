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
