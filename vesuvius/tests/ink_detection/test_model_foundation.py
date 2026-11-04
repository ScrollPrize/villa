"""CPU witnesses for model construction, depth fusion, and input padding."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import vesuvius.models.build.build_network_from_config as network_builder
from vesuvius.ink_detection.config import InkConfig
import vesuvius.ink_detection.models.model as model_module
from vesuvius.ink_detection.models.hybrid_3d2d import (
    Local3DStem2DUNet,
    LocalDepthFusionStem,
)
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.ink_detection.models.model import SliceChannel2DModel, make_model

from .helpers import config_mapping


@pytest.mark.parametrize(
    ("model_type", "depth", "expected_shape", "wrapper_type"),
    [
        ("vesuvius_unet", 4, (1, 1, 4, 16, 16), None),
        ("vesuvius_unet_2p5d", 3, (1, 1, 16, 16), SliceChannel2DModel),
        (
            "vesuvius_unet_3d_stem_2d",
            3,
            (1, 1, 16, 16),
            Local3DStem2DUNet,
        ),
    ],
)
def test_real_model_families_build_and_forward_on_cpu(
    model_type,
    depth,
    expected_shape,
    wrapper_type,
):
    torch.manual_seed(7)
    model = make_model(InkConfig.from_mapping(config_mapping(model_type, depth=depth)))
    if wrapper_type is not None:
        assert isinstance(model, wrapper_type)

    output = model(torch.randn(1, 1, depth, 16, 16))

    assert tuple(output["ink"].shape) == expected_shape
    assert torch.isfinite(output["ink"]).all()


@pytest.mark.parametrize(
    "model_type",
    ["unet", "unet_2p5d", "unet_3d_stem_2d"],
)
def test_unet_aliases_dispatch_to_the_same_model_families(model_type):
    depth = 4 if model_type == "unet" else 3
    model = make_model(InkConfig.from_mapping(config_mapping(model_type, depth=depth)))

    if model_type == "unet":
        assert model.op_dims == 3
    elif model_type == "unet_2p5d":
        assert isinstance(model, SliceChannel2DModel)
    else:
        assert isinstance(model, Local3DStem2DUNet)


def test_dinov2_backbone_configuration_reaches_the_shared_model_builder(
    monkeypatch,
):
    authored = config_mapping("vesuvius_unet", depth=4)
    authored["model_config"].update(
        {
            "pretrained_backbone": "dinov2_ps8",
            "pretrained_decoder_type": "primus_patch_decode",
        }
    )
    observed = {}
    sentinel = nn.Identity()

    def record_build(config_dict, *, op_dims):
        observed.update(config_dict)
        observed["op_dims"] = op_dims
        return sentinel

    monkeypatch.setattr(model_module, "_build_network", record_build)

    model = make_model(InkConfig.from_mapping(authored))

    assert model is sentinel
    assert observed["op_dims"] == 3
    assert observed["model_config"]["pretrained_backbone"] == "dinov2_ps8"
    assert (
        observed["model_config"]["pretrained_decoder_type"]
        == "primus_patch_decode"
    )


@pytest.mark.parametrize(
    ("model_type", "expected_spacing"),
    [
        ("vesuvius_unet", (2.0, 3.0, 4.0)),
        ("vesuvius_unet_2p5d", (3.0, 4.0)),
        ("vesuvius_unet_3d_stem_2d", (3.0, 4.0)),
    ],
)
def test_typed_spacing_is_forwarded_in_model_axis_order(
    model_type,
    expected_spacing,
    monkeypatch,
):
    authored = config_mapping(model_type, depth=3)
    authored["model_config"]["spacing"] = [2, 3, 4]
    observed = []
    original = network_builder.get_pool_and_conv_props

    def record_spacing(*, spacing, **kwargs):
        observed.append(tuple(spacing))
        return original(spacing=spacing, **kwargs)

    monkeypatch.setattr(
        network_builder,
        "get_pool_and_conv_props",
        record_spacing,
    )

    config = InkConfig.from_mapping(authored)
    make_model(config)

    assert config.model.spacing == (2.0, 3.0, 4.0)
    assert observed == [expected_spacing]


def test_depth_fusion_is_uniform_attention_weighted_sum_followed_by_maximum():
    stem = LocalDepthFusionStem(channels=1)
    stem.features = nn.Identity()
    image_BCZYX = torch.tensor([1.0, 3.0, 8.0]).reshape(1, 1, 3, 1, 1)

    fused_BCYX = stem(image_BCZYX)

    torch.testing.assert_close(
        fused_BCYX,
        torch.tensor([4.0, 8.0]).reshape(1, 2, 1, 1),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.softmax(stem.attention_logits(image_BCZYX).float(), dim=2),
        torch.full((1, 1, 3, 1, 1), 1.0 / 3.0),
    )


def test_learned_attention_head_uses_nonuniform_fp32_weighted_sum(monkeypatch):
    stem = LocalDepthFusionStem(channels=1).double()
    stem.features = nn.Identity()
    with torch.no_grad():
        stem.attention_logits.weight.fill_(1.0)
        stem.attention_logits.bias.zero_()
    image_BCZYX = torch.tensor(
        [0.0, math.log(2.0), math.log(4.0)],
        dtype=torch.float64,
    ).reshape(1, 1, 3, 1, 1)
    original_softmax = torch.softmax
    softmax_calls = []

    def record_softmax_dtype(tensor, dim):
        softmax_calls.append((tensor.dtype, dim))
        return original_softmax(tensor, dim=dim)

    monkeypatch.setattr(torch, "softmax", record_softmax_dtype)

    fused_BCYX = stem(image_BCZYX)

    expected_attention = 10.0 * math.log(2.0) / 7.0
    torch.testing.assert_close(
        fused_BCYX,
        torch.tensor(
            [expected_attention, math.log(4.0)],
            dtype=image_BCZYX.dtype,
        ).reshape(1, 2, 1, 1),
        rtol=1e-6,
        atol=1e-7,
    )
    assert expected_attention != pytest.approx(image_BCZYX.mean().item())
    assert softmax_calls == [(torch.float32, 2)]


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 1, 2, 4, 4), "input depth mismatch"),
        ((1, 2, 3, 4, 4), "one source image channel"),
        ((1, 3, 4, 4), "must have shape"),
    ],
)
def test_hybrid_rejects_depth_channel_and_rank_errors(shape, message):
    model = Local3DStem2DUNet(nn.Identity(), input_depth=3, stem_channels=1)

    with pytest.raises(ValueError, match=message):
        model(torch.zeros(shape))


def test_hybrid_checkpoint_restores_predictions(tmp_path):
    from vesuvius.ink_detection.models.checkpoint import load_checkpoint, load_model_state

    config = InkConfig.from_mapping(config_mapping("vesuvius_unet_3d_stem_2d", depth=3))
    # Keep this test's initialization deterministic without changing other tests' RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        original = make_model(config).eval()
        restored = make_model(config).eval()
    image = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 1, 3, 16, 16)
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model": original.state_dict()}, checkpoint_path)

    with torch.inference_mode():
        expected = original(image)["ink"]
        # Ensure a no-op restore would fail the output comparison.
        assert not torch.allclose(restored(image)["ink"], expected)
        load_model_state(restored, load_checkpoint(checkpoint_path)["model"])
        torch.testing.assert_close(restored(image)["ink"], expected, rtol=0, atol=0)


def test_depth_padding_places_even_and_odd_deficits_on_literal_sides():
    image_BCZYX = torch.tensor([2.0, 3.0]).reshape(1, 1, 2, 1, 1)

    even_BCZYX = center_pad_input_depth(image_BCZYX, 4)
    odd_BCZYX = center_pad_input_depth(image_BCZYX, 5)

    assert even_BCZYX.flatten().tolist() == [0.0, 2.0, 3.0, 0.0]
    assert odd_BCZYX.flatten().tolist() == [0.0, 2.0, 3.0, 0.0, 0.0]
    torch.testing.assert_close(
        center_pad_input_depth(image_BCZYX, 2),
        image_BCZYX,
        rtol=0.0,
        atol=0.0,
    )


def test_depth_padding_rejects_rank_and_implicit_crop():
    with pytest.raises(ValueError, match="expects"):
        center_pad_input_depth(torch.zeros(1, 2, 3, 4), 5)
    with pytest.raises(ValueError, match="Cannot pad input depth 4 down to 3"):
        center_pad_input_depth(torch.zeros(1, 1, 4, 2, 2), 3)


def test_config_rejects_nonpositive_padding_depth():
    mapping = config_mapping()
    mapping["model_config"]["input_pad_depth_to"] = 0

    with pytest.raises(ValueError, match="input_pad_depth_to must be positive"):
        InkConfig.from_mapping(mapping)


@pytest.mark.parametrize("spacing", [[1, 1], [1, 1, 1, 1], "1,1,1"])
def test_config_rejects_spacing_without_three_axes(spacing):
    mapping = config_mapping()
    mapping["model_config"]["spacing"] = spacing

    with pytest.raises(ValueError, match="spacing.*three axes"):
        InkConfig.from_mapping(mapping)
