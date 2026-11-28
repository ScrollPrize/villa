import pytest
import torch

import vesuvius.neural_tracing.fiber_trace_2d.train as train_module
from vesuvius.neural_tracing.fiber_trace_2d.train import _normalize_image_tensor, _prepare_images

_DEVICES = (
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an MPS device"),
    ),
)


def _strips():
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(3, 1, 16, 24, generator=generator) * 4000.0 + 1000.0
    valid = torch.rand(3, 16, 24, generator=generator) > 0.25
    return images, valid


@pytest.mark.parametrize("device", _DEVICES)
def test_normalize_image_tensor_on_device_matches_cpu(device):
    images, valid = _strips()
    expected = _normalize_image_tensor(images, valid)
    for sample in range(images.shape[0]):
        inside = expected[sample, 0][valid[sample]]
        assert abs(inside.mean().item()) < 1e-4
        assert abs(inside.std(unbiased=False).item() - 1.0) < 1e-3
    assert torch.all(expected[~valid.unsqueeze(1)] == 0)

    output = _normalize_image_tensor(images.to(device), valid.to(device))

    assert output.device.type == device
    assert output.dtype == images.dtype
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("device", _DEVICES)
def test_prepare_images_on_device_matches_cpu(device):
    images, valid = _strips()
    expected = _prepare_images(images.numpy(), valid.numpy(), device=torch.device("cpu"))

    output = _prepare_images(images.numpy(), valid.numpy(), device=torch.device(device))

    assert output.device.type == device
    assert torch.equal(output.cpu(), expected)


def test_normalize_image_tensor_cuda_without_native_backend_matches_cpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(train_module.vc_cuda, "is_available", lambda: False)
    images, valid = _strips()
    expected = _normalize_image_tensor(images, valid)

    output = _normalize_image_tensor(images.cuda(), valid.cuda())

    assert output.device.type == "cuda"
    assert torch.equal(output.cpu(), expected)
