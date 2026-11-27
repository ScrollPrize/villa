"""3D convolutions run as sums of 2D convolutions give conv3d's result."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.inference.inference_runtime import (
    prepare_model_for_inference,
)
from vesuvius.ink_detection.models.conv3d_as_2d import (
    Conv3dAsConv2d,
    resolve_conv3d_as_2d,
    split_conv3d_for_inference,
    split_conv3d_layers,
)
from vesuvius.ink_detection.models.hybrid_3d2d import LocalDepthFusionStem


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel, stride, padding, groups, bias",
    [
        (1, 16, 3, 1, 1, 1, False),  # depth-fusion stem, first layer
        (16, 16, 3, 1, 1, 1, False),  # depth-fusion stem, second layer
        (16, 1, 1, 1, 0, 1, True),  # attention logits
        (4, 6, (3, 5, 1), (2, 1, 2), (1, 2, 0), 2, True),
        (3, 3, (2, 3, 3), (1, 2, 1), (0, 1, 1), 3, False),
    ],
)
def test_conv3d_as_conv2d_matches_conv3d(
    in_channels, out_channels, kernel, stride, padding, groups, bias
):
    torch.manual_seed(0)
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
    ).double()
    image = torch.randn(2, in_channels, 9, 13, 11, dtype=torch.float64)

    torch.testing.assert_close(
        Conv3dAsConv2d.from_conv3d(conv)(image), conv(image), rtol=1e-12, atol=1e-12
    )


def test_split_keeps_weights_names_and_stem_output():
    torch.manual_seed(1)
    stem = LocalDepthFusionStem(channels=16).eval()
    with torch.no_grad():
        # Non-zero attention so the softmax over depth is not a plain mean.
        stem.attention_logits.weight.normal_()
        stem.attention_logits.bias.normal_()
    image = torch.randn(2, 1, 17, 32, 32)
    with torch.no_grad():
        expected = stem(image)
    keys = list(stem.state_dict())
    parameters = list(stem.parameters())

    assert split_conv3d_layers(stem) == 3
    assert all(
        type(module) is not nn.Conv3d for module in stem.modules()
    ), "every plain Conv3d is swapped"
    assert list(stem.state_dict()) == keys
    assert all(a is b for a, b in zip(stem.parameters(), parameters))
    with torch.no_grad():
        torch.testing.assert_close(stem(image), expected, rtol=1e-5, atol=1e-5)


def test_split_leaves_layers_it_cannot_rewrite():
    layers = nn.Sequential(
        nn.Conv3d(1, 1, 3, padding=1, padding_mode="replicate"),
        nn.Conv3d(1, 1, 3, padding="same"),
        nn.Conv3d(1, 1, 3, dilation=2),
    )
    assert split_conv3d_layers(layers) == 0
    assert all(type(layer) is nn.Conv3d for layer in layers)


def test_auto_runs_as_2d_only_on_rocm_and_mps(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert resolve_conv3d_as_2d("auto", torch.device("cuda")) is False
    assert resolve_conv3d_as_2d("auto", torch.device("mps")) is True
    assert resolve_conv3d_as_2d("auto", torch.device("cpu")) is False
    monkeypatch.setattr(torch.version, "hip", "7.2")
    assert resolve_conv3d_as_2d("auto", torch.device("cuda")) is True
    assert resolve_conv3d_as_2d("auto", torch.device("cpu")) is False
    assert resolve_conv3d_as_2d("on", torch.device("cpu")) is True
    assert resolve_conv3d_as_2d("off", torch.device("cuda")) is False
    with pytest.raises(ValueError, match="conv3d_as_2d"):
        resolve_conv3d_as_2d("yes", torch.device("cuda"))


def _stem_and_outside_conv3d() -> nn.Module:
    model = nn.Module()
    model.depth_fusion = LocalDepthFusionStem(channels=4)
    model.body = nn.Conv3d(8, 8, 3, padding=1)
    return model


def _swapped(model: nn.Module) -> dict[str, bool]:
    return {
        name: isinstance(module, Conv3dAsConv2d)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv3d)
    }


def test_auto_swaps_only_the_depth_fusion_stem_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    model = _stem_and_outside_conv3d()

    assert split_conv3d_for_inference(model, "auto", torch.device("cuda")) == 3
    assert _swapped(model) == {
        "depth_fusion.features.0": True,
        "depth_fusion.features.3": True,
        "depth_fusion.attention_logits": True,
        "body": False,
    }


def test_auto_swaps_only_the_depth_fusion_stem_on_mps(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    model = _stem_and_outside_conv3d()

    assert split_conv3d_for_inference(model, "auto", torch.device("mps")) == 3
    assert _swapped(model)["body"] is False


def test_a_conv_registered_under_two_names_gets_one_replacement():
    # ConvDropoutNormReLU keeps its convolution as both `conv` and `all_modules[0]`.
    block = nn.Module()
    block.conv = nn.Conv3d(2, 3, 3, padding=1)
    block.all_modules = nn.Sequential(block.conv, nn.ReLU())
    original_weight = block.conv.weight

    assert split_conv3d_layers(block) == 1
    assert isinstance(block.conv, Conv3dAsConv2d)
    assert block.all_modules[0] is block.conv
    assert block.conv.weight is original_weight
    assert len([m for m in block.modules() if isinstance(m, nn.Conv3d)]) == 1


def test_on_swaps_every_conv3d_and_auto_leaves_cuda_alone(monkeypatch):
    model = _stem_and_outside_conv3d()
    monkeypatch.setattr(torch.version, "hip", None)
    assert split_conv3d_for_inference(model, "auto", torch.device("cuda")) == 0
    assert not any(_swapped(model).values())

    assert split_conv3d_for_inference(model, "on", torch.device("cuda")) == 4
    assert all(_swapped(model).values())


@pytest.mark.parametrize("mode, swapped", [("on", True), ("off", False), ("auto", False)])
def test_prepare_model_for_inference_swaps_on_request(mode, swapped):
    model = nn.Sequential(nn.Conv3d(1, 2, 3, padding=1)).eval()

    prepared, _ = prepare_model_for_inference(
        model,
        gpu_ids=(),
        compile_model=False,
        compile_mode="default",
        device_type="cpu",
        conv3d_as_2d=mode,
    )

    assert isinstance(prepared[0], Conv3dAsConv2d) is swapped
