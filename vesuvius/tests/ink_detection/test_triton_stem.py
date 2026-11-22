"""The opt-in Triton stem convolution: torch fallback everywhere, parity on ROCm."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.models.triton_conv3d import TritonConv3d, triton

from .helpers import config_mapping


def _paired_convs(channels=16, **kwargs):
    torch.manual_seed(3)
    reference = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False, **kwargs)
    conv = TritonConv3d(channels, channels, kernel_size=3, padding=1, bias=False, **kwargs)
    conv.load_state_dict(reference.state_dict())
    return reference, conv


@pytest.mark.parametrize("channels", [16, 2])
def test_triton_conv_falls_back_to_torch_on_cpu(channels):
    reference, conv = _paired_convs(channels)
    x = torch.randn(2, channels, 5, 8, 8)

    assert not conv._uses_triton(x)
    torch.testing.assert_close(conv(x), reference(x), rtol=0, atol=0)
    assert conv.state_dict().keys() == reference.state_dict().keys()


def test_stem_conv_backend_defaults_to_torch_and_rejects_unknown_names():
    authored = config_mapping("vesuvius_unet_3d_stem_2d", depth=3)
    model = make_model(InkConfig.from_mapping(authored))
    assert type(model.depth_fusion.features[3]) is nn.Conv3d

    authored["model_config"]["stem_conv_backend"] = "cudnn"
    with pytest.raises(ValueError, match="stem_conv_backend"):
        InkConfig.from_mapping(authored)


def test_triton_stem_backend_builds_and_forwards_on_cpu():
    authored = config_mapping("vesuvius_unet_3d_stem_2d", depth=3)
    authored["model_config"]["stem_conv_backend"] = "triton"
    model = make_model(InkConfig.from_mapping(authored))
    assert isinstance(model.depth_fusion.features[3], TritonConv3d)

    output = model(torch.randn(1, 1, 3, 16, 16))

    assert tuple(output["ink"].shape) == (1, 1, 16, 16)
    assert torch.isfinite(output["ink"]).all()


@pytest.mark.skipif(
    triton is None or not torch.version.hip or not torch.cuda.is_available(),
    reason="the Triton stem convolution runs only on ROCm",
)
def test_triton_conv_matches_torch_forward_and_backward_on_rocm():
    reference, conv = _paired_convs()
    reference, conv = reference.cuda(), conv.cuda()
    x = torch.randn(2, 16, 17, 32, 32, device="cuda", dtype=torch.float16)
    x_reference = x.clone().requires_grad_(True)
    x = x.requires_grad_(True)
    grad = torch.randn(2, 16, 17, 32, 32, device="cuda", dtype=torch.float16)

    assert conv._uses_triton(x)
    with torch.autocast("cuda", dtype=torch.float16):
        y = conv(x)
        y_reference = reference(x_reference)
    y.backward(grad)
    y_reference.backward(grad)

    torch.testing.assert_close(y, y_reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=2e-3, atol=2e-3)
    cosine = nn.functional.cosine_similarity(
        conv.weight.grad.flatten(), reference.weight.grad.flatten(), dim=0
    )
    assert cosine > 0.9999
