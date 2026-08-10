"""CPU witnesses for model construction, depth fusion, and input padding."""

from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch
from torch import nn

import vesuvius.models.build.build_network_from_config as network_builder
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.hybrid_3d2d import (
    Local3DStem2DUNet,
    LocalDepthFusionStem,
)
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.ink_detection.models.model import SliceChannel2DModel, make_model


def _config_mapping(
    model_type: str = "vesuvius_unet",
    *,
    depth: int = 4,
    side: int = 16,
    stem_channels: int = 2,
) -> dict:
    is_base = model_type in {"vesuvius_unet", "unet"}
    dimensions = 3 if is_base else 2
    kernel = [3] * dimensions
    first_stride = [1] * dimensions
    second_stride = [2] * dimensions
    model_config = {
        "autoconfigure": True,
        "basic_encoder_block": "BasicBlockD",
        "basic_decoder_block": "ConvBlock",
        "features_per_stage": [2, 4],
        "n_blocks_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "kernel_sizes": [kernel, kernel],
        "strides": [first_stride, second_stride],
        "pool_op_kernel_sizes": [first_stride, second_stride],
        "spacing": [1, 1, 1],
    }
    if "3d_stem_2d" in model_type:
        model_config["stem_channels"] = stem_channels
    return {
        "model_type": model_type,
        "mode": "flat",
        "patch_size": [depth, side, side],
        "patch_overlap": 0.25,
        "patch_min_labeled_coverage": 0.0,
        "batch_size": 1,
        "in_channels": depth if "2p5d" in model_type else 1,
        "model_config": model_config,
        "targets": {
            "ink": {
                "out_channels": 1,
                "activation": "none",
                "z_projection_mode": "none",
            }
        },
        "datasets": [{"segments_path": "/tmp/ink", "volume_scale": 0}],
    }


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
    model = make_model(InkConfig.from_mapping(_config_mapping(model_type, depth=depth)))
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
    model = make_model(InkConfig.from_mapping(_config_mapping(model_type, depth=depth)))

    if model_type == "unet":
        assert model.op_dims == 3
    elif model_type == "unet_2p5d":
        assert isinstance(model, SliceChannel2DModel)
    else:
        assert isinstance(model, Local3DStem2DUNet)


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
    authored = _config_mapping(model_type, depth=3)
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


def test_tiny_hybrid_state_manifest_is_ordered_and_stable():
    mapping = _config_mapping(
        "vesuvius_unet_3d_stem_2d",
        depth=17,
        side=32,
        stem_channels=16,
    )
    mapping["batch_size"] = 2
    mapping["model_config"]["features_per_stage"] = [4, 8]
    model = make_model(InkConfig.from_mapping(mapping))
    manifest = [
        [key, list(tensor.shape), str(tensor.dtype)]
        for key, tensor in model.state_dict().items()
    ]
    digest = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":")).encode()
    ).hexdigest()

    assert len(manifest) == 66
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_298
    assert manifest[0][0].startswith("network.")
    assert manifest[-1][0] == "depth_fusion.attention_logits.bias"
    assert digest == "c3ef958b71090231c673d8022a352f5c622dbfa86427064aabebbadd4fbf27a9"


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
    mapping = _config_mapping()
    mapping["model_config"]["input_pad_depth_to"] = 0

    with pytest.raises(ValueError, match="input_pad_depth_to must be positive"):
        InkConfig.from_mapping(mapping)


@pytest.mark.parametrize("spacing", [[1, 1], [1, 1, 1, 1], "1,1,1"])
def test_config_rejects_spacing_without_three_axes(spacing):
    mapping = _config_mapping()
    mapping["model_config"]["spacing"] = spacing

    with pytest.raises(ValueError, match="spacing.*three axes"):
        InkConfig.from_mapping(mapping)
