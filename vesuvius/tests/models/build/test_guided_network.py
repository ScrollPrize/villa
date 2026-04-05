from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.build.guidance import TokenBook3D
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.build.pretrained_backbones.dinov2 import (
    PixelShuffleConvHeadBigDinov2Decoder,
    PixelShuffleConvDinov2Decoder,
    build_dinov2_backbone,
    build_dinov2_decoder,
)
from vesuvius.models.build.simple_conv_blocks import ConvDropoutNormReLU
from vesuvius.models.build.transformers.patch_encode_decode import PixelShuffle3D
from vesuvius.models.utils import InitWeights_He


def _tiny_dinovol_model_config() -> dict:
    return {
        "model_type": "v2",
        "input_channels": 1,
        "global_crops_size": [16, 16, 16],
        "local_crops_size": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "num_reg_tokens": 2,
        "mlp_ratio": 2.0,
        "drop_path_rate": 0.0,
        "qkv_fused": True,
    }


def _write_local_guide_checkpoint(path: Path) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    checkpoint = {
        "config": {
            "model": _tiny_dinovol_model_config(),
            "dataset": {
                "global_crop_size": [16, 16, 16],
                "local_crop_size": [16, 16, 16],
            },
        },
        "teacher": teacher_state,
    }
    torch.save(checkpoint, path)


def _make_mgr(
    checkpoint_path: Path,
    *,
    basic_encoder_block: str,
    guide_tokenbook_tokens: int | None = None,
    guide_compile_policy: str | None = None,
    guide_fusion_stage: str | None = None,
    guide_tokenbook_prototype_weighting: str | None = None,
    guide_feature_gate_alpha: float | None = None,
    guide_skip_concat_projector_channels: list[int] | None = None,
    target_out_channels: int = 1,
) -> SimpleNamespace:
    guided_config = {}
    if guide_tokenbook_tokens is not None:
        guided_config["guide_tokenbook_tokens"] = int(guide_tokenbook_tokens)
    if guide_compile_policy is not None:
        guided_config["guide_compile_policy"] = str(guide_compile_policy)
    if guide_fusion_stage is not None:
        guided_config["guide_fusion_stage"] = str(guide_fusion_stage)
    if guide_tokenbook_prototype_weighting is not None:
        guided_config["guide_tokenbook_prototype_weighting"] = str(guide_tokenbook_prototype_weighting)
    if guide_feature_gate_alpha is not None:
        guided_config["guide_feature_gate_alpha"] = float(guide_feature_gate_alpha)
    if guide_skip_concat_projector_channels is not None:
        guided_config["guide_skip_concat_projector_channels"] = [int(v) for v in guide_skip_concat_projector_channels]
    return SimpleNamespace(
        targets={"ink": {"out_channels": int(target_out_channels), "activation": "none"}},
        train_patch_size=(16, 16, 16),
        train_batch_size=2,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "features_per_stage": [8, 16, 32],
            "n_stages": 3,
            "n_blocks_per_stage": [1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1],
            "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": basic_encoder_block,
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
            "guide_backbone": str(checkpoint_path),
            "guide_freeze": True,
            "guide_tokenbook_sample_rate": 1.0,
            "input_shape": [16, 16, 16],
            **guided_config,
        },
    )


def _make_pretrained_backbone_mgr(
    checkpoint_path: Path,
    *,
    freeze_encoder: bool,
    decoder_type: str = "primus_patch_decode",
) -> SimpleNamespace:
    return SimpleNamespace(
        targets={"surface": {"out_channels": 2, "activation": "none"}},
        train_patch_size=(16, 16, 16),
        train_batch_size=2,
        in_channels=1,
        autoconfigure=False,
        model_name="pretrained_backbone_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "pretrained_backbone": str(checkpoint_path),
            "pretrained_decoder_type": str(decoder_type),
            "freeze_encoder": bool(freeze_encoder),
            "input_shape": [16, 16, 16],
        },
    )


def test_tokenbook3d_returns_unit_interval_mask():
    module = TokenBook3D(n_tokens=8, embed_dim=16)
    x = torch.randn(2, 16, 2, 2, 2)

    guide = module(x)

    assert guide.shape == (2, 1, 2, 2, 2)
    assert float(guide.min().detach()) >= 0.0
    assert float(guide.max().detach()) <= 1.0


def test_tokenbook3d_token_mlp_returns_unit_interval_mask():
    module = TokenBook3D(
        n_tokens=8,
        embed_dim=16,
        prototype_weighting="token_mlp",
        weight_mlp_hidden=12,
    )
    x = torch.randn(2, 16, 2, 2, 2)

    guide = module(x)

    assert guide.shape == (2, 1, 2, 2, 2)
    assert module.prototype_weight_mlp is not None
    assert float(guide.min().detach()) >= 0.0
    assert float(guide.max().detach()) <= 1.0


def test_pixelshuffle3d_upsamples_tuple_factor_shape():
    module = PixelShuffle3D((2, 2, 2))
    x = torch.randn(2, 32, 3, 4, 5)

    y = module(x)

    assert y.shape == (2, 4, 6, 8, 10)


@pytest.mark.parametrize("basic_encoder_block", ["ConvBlock", "BasicBlockD"])
def test_guided_network_forward_shapes_and_aux_outputs(tmp_path: Path, basic_encoder_block: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(checkpoint_path, basic_encoder_block=basic_encoder_block)
    model = NetworkFromConfig(mgr)
    x = torch.randn(2, 1, 16, 16, 16)

    outputs = model(x)
    outputs_with_aux, aux = model(x, return_aux=True)

    assert set(outputs.keys()) == {"ink"}
    assert outputs["ink"].shape == (2, 1, 16, 16, 16)
    assert set(outputs_with_aux.keys()) == {"ink"}
    assert aux["guide_mask"].shape == (2, 1, 2, 2, 2)


@pytest.mark.parametrize("basic_encoder_block", ["ConvBlock", "BasicBlockD"])
def test_feature_encoder_guidance_returns_encoder_stage_aux_outputs(tmp_path: Path, basic_encoder_block: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block=basic_encoder_block,
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)

    outputs, aux = model(torch.randn(2, 1, 16, 16, 16), return_aux=True)

    assert outputs["ink"].shape == (2, 1, 16, 16, 16)
    assert list(aux.keys()) == ["enc_0", "enc_1", "enc_2"]
    assert aux["enc_0"].shape == (2, 1, 16, 16, 16)
    assert aux["enc_1"].shape == (2, 1, 8, 8, 8)
    assert aux["enc_2"].shape == (2, 1, 4, 4, 4)
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_stage_keys"] == ["enc_0", "enc_1", "enc_2"]
    assert model.final_config["guide_feature_gate_alpha"] == pytest.approx(1.0)
    assert len(model.guide_stage_tokenbooks) == 3


@pytest.mark.parametrize("basic_encoder_block", ["ConvBlock", "BasicBlockD"])
def test_feature_skip_concat_forward_shapes_and_empty_aux_outputs(tmp_path: Path, basic_encoder_block: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block=basic_encoder_block,
        guide_fusion_stage="feature_skip_concat",
    )
    model = NetworkFromConfig(mgr)

    outputs, aux = model(torch.randn(2, 1, 16, 16, 16), return_aux=True)

    assert outputs["ink"].shape == (2, 1, 16, 16, 16)
    assert aux == {}
    assert list(model.guide_skip_projectors.keys()) == ["enc_0", "enc_1", "enc_2"]
    assert all(isinstance(projector, ConvDropoutNormReLU) for projector in model.guide_skip_projectors.values())
    assert all(projector.conv.bias is None for projector in model.guide_skip_projectors.values())
    assert all(isinstance(projector.norm, model.norm_op) for projector in model.guide_skip_projectors.values())
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32]
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None
    assert model.final_config["guide_tokenbook_weight_mlp_hidden"] is None


def test_direct_segmentation_returns_two_channel_logits_and_low_res_guide_mask(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="direct_segmentation",
        guide_tokenbook_prototype_weighting="token_mlp",
        target_out_channels=2,
    )
    model = NetworkFromConfig(mgr)

    outputs, aux = model(torch.randn(2, 1, 16, 16, 16), return_aux=True)
    probabilities = torch.softmax(outputs["ink"], dim=1)[:, 1:2]
    expected = F.interpolate(
        aux["guide_mask"],
        size=probabilities.shape[2:],
        mode="trilinear",
        align_corners=False,
    )

    assert model.shared_encoder is None
    assert model.final_config["guide_fusion_stage"] == "direct_segmentation"
    assert model.final_config["guide_direct_output_mode"] == "two_channel_logits"
    assert outputs["ink"].shape == (2, 2, 16, 16, 16)
    assert aux["guide_mask"].shape == (2, 1, 2, 2, 2)
    assert torch.allclose(probabilities, expected, atol=1e-5, rtol=1e-5)


def test_pretrained_backbone_pixelshuffle_conv_decoder_returns_input_resolution(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_pretrained_backbone_mgr(
        checkpoint_path,
        freeze_encoder=True,
        decoder_type="pixelshuffle_conv",
    )
    model = NetworkFromConfig(mgr)

    outputs = model(torch.randn(2, 1, 16, 16, 16))

    assert outputs["surface"].shape == (2, 2, 16, 16, 16)
    assert model.final_config["pretrained_decoder_type"] == "pixelshuffle_conv"


def test_pretrained_backbone_pixelshuffle_convhead_big_returns_input_resolution(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_pretrained_backbone_mgr(
        checkpoint_path,
        freeze_encoder=True,
        decoder_type="pixelshuffle_convhead_big",
    )
    model = NetworkFromConfig(mgr)

    outputs = model(torch.randn(2, 1, 16, 16, 16))

    assert outputs["surface"].shape == (2, 2, 16, 16, 16)
    assert model.final_config["pretrained_decoder_type"] == "pixelshuffle_convhead_big"


def test_encoder_skip_only_feature_gating_uses_residual_alpha_formula():
    class _TestStage(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            return torch.full_like(x, self.value)

    from vesuvius.models.build.encoder import Encoder

    test_encoder = Encoder.__new__(Encoder)
    torch.nn.Module.__init__(test_encoder)
    stage0 = _TestStage(2.0)
    stage1 = _TestStage(3.0)
    test_encoder.stem = None
    test_encoder.stages = torch.nn.Sequential(stage0, stage1)
    test_encoder.output_channels = [1, 1]
    test_encoder.strides = [[1, 1, 1], [1, 1, 1]]
    test_encoder.return_skips = True
    test_encoder.conv_op = torch.nn.Conv3d
    test_encoder.norm_op = None
    test_encoder.norm_op_kwargs = {}
    test_encoder.nonlin = None
    test_encoder.nonlin_kwargs = {}
    test_encoder.dropout_op = None
    test_encoder.dropout_op_kwargs = {}
    test_encoder.conv_bias = True
    test_encoder.kernel_sizes = [[3, 3, 3], [3, 3, 3]]

    x = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    feature_gates = {
        "enc_0": torch.full((1, 1, 2, 2, 2), 0.5, dtype=torch.float32),
        "enc_1": torch.full((1, 1, 2, 2, 2), 0.25, dtype=torch.float32),
    }

    skips = test_encoder(x, feature_gates=feature_gates, feature_gate_alpha=0.9)

    expected_stage0_skip = 2.0 * (0.1 + 0.9 * 0.5)
    expected_stage1_skip = 3.0 * (0.1 + 0.9 * 0.25)
    assert torch.allclose(skips[0], torch.full_like(skips[0], expected_stage0_skip))
    assert torch.allclose(skips[1], torch.full_like(skips[1], expected_stage1_skip))
    assert torch.allclose(stage1.last_input, torch.full_like(stage1.last_input, 2.0))


def test_feature_skip_concat_decoder_contract_uses_augmented_skip_channels(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
    )
    model = NetworkFromConfig(mgr)
    decoder = model.task_decoders["ink"]

    assert decoder.skip_channels == [16, 32, 64]
    assert decoder.bottleneck_input_channels == 64
    assert decoder.decoder_stage_input_channels == [16 + 32, 8 + 16]
    assert decoder.decoder_stage_output_channels == [16, 8]
    assert model.guide_stage_keys == ["enc_0", "enc_1", "enc_2"]
    assert all(
        projector.output_channels == native_width
        for projector, native_width in zip(model.guide_skip_projectors.values(), [8, 16, 32])
    )


def test_feature_skip_concat_decoder_contract_uses_configured_projector_widths(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
        guide_skip_concat_projector_channels=[2, 4, 8],
    )
    model = NetworkFromConfig(mgr)
    decoder = model.task_decoders["ink"]

    assert decoder.skip_channels == [10, 20, 40]
    assert decoder.bottleneck_input_channels == 40
    assert decoder.decoder_stage_input_channels == [16 + 20, 8 + 10]
    assert decoder.decoder_stage_output_channels == [16, 8]
    assert model.final_config["guide_skip_concat_projector_channels"] == [2, 4, 8]
    assert [projector.output_channels for projector in model.guide_skip_projectors.values()] == [2, 4, 8]
    assert all(projector.conv.bias is None for projector in model.guide_skip_projectors.values())
    assert all(isinstance(projector.norm, model.norm_op) for projector in model.guide_skip_projectors.values())


def test_feature_skip_concat_projects_low_res_guide_features_before_upsampling(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
    )
    model = NetworkFromConfig(mgr)

    class _RecordingProjector(torch.nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.out_channels = out_channels
            self.last_input_shape = None

        def forward(self, x):
            self.last_input_shape = tuple(x.shape)
            return torch.ones(
                x.shape[0],
                self.out_channels,
                x.shape[2],
                x.shape[3],
                x.shape[4],
                device=x.device,
                dtype=x.dtype,
            )

    model.guide_skip_projectors = torch.nn.ModuleDict(
        {
            "enc_0": _RecordingProjector(8),
            "enc_1": _RecordingProjector(16),
            "enc_2": _RecordingProjector(32),
        }
    )
    low_res_guide = torch.randn(2, 48, 2, 2, 2)
    model._compute_guide_features = lambda x: low_res_guide

    encoder_features = [
        torch.randn(2, 8, 16, 16, 16),
        torch.randn(2, 16, 8, 8, 8),
        torch.randn(2, 32, 4, 4, 4),
    ]

    augmented_features, aux = model._build_skip_concat_features(
        torch.randn(2, 1, 16, 16, 16),
        encoder_features,
    )

    assert aux == {}
    assert [feature.shape for feature in augmented_features] == [
        (2, 16, 16, 16, 16),
        (2, 32, 8, 8, 8),
        (2, 64, 4, 4, 4),
    ]
    for projector in model.guide_skip_projectors.values():
        assert projector.last_input_shape == (2, 48, 2, 2, 2)


def test_guided_network_backprop_updates_tokenbook_but_not_frozen_guide_backbone(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(checkpoint_path, basic_encoder_block="ConvBlock")
    model = NetworkFromConfig(mgr)
    model.train()

    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16)
    outputs, aux = model(x, return_aux=True)
    loss = torch.nn.functional.mse_loss(outputs["ink"], target) + aux["guide_mask"].mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.guide_tokenbook.parameters())
    assert all(parameter.grad is None for parameter in model.guide_backbone.parameters())


def test_direct_segmentation_backprop_updates_tokenbook_but_not_frozen_guide_backbone(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="direct_segmentation",
        target_out_channels=2,
    )
    model = NetworkFromConfig(mgr)
    model.train()

    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16, 16))
    outputs, aux = model(x, return_aux=True)
    loss = torch.nn.functional.cross_entropy(outputs["ink"], target) + aux["guide_mask"].mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.guide_tokenbook.parameters())
    assert all(parameter.grad is None for parameter in model.guide_backbone.parameters())


def test_init_weights_he_preserves_frozen_guide_backbone_weights(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="direct_segmentation",
        target_out_channels=2,
    )
    model = NetworkFromConfig(mgr)
    before = {
        key: value.detach().clone()
        for key, value in model.guide_backbone.state_dict().items()
    }

    model.apply(InitWeights_He(neg_slope=0.2))

    after = model.guide_backbone.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_feature_skip_concat_requires_projector_channels_to_match_num_stages(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
        guide_skip_concat_projector_channels=[8, 16],
    )

    with pytest.raises(ValueError, match="guide_skip_concat_projector_channels must have as many entries"):
        NetworkFromConfig(mgr)


def test_direct_segmentation_requires_single_binary_target(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = SimpleNamespace(
        targets={
            "ink": {"out_channels": 2, "activation": "none"},
            "surface": {"out_channels": 2, "activation": "none"},
        },
        train_patch_size=(16, 16, 16),
        train_batch_size=2,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "guide_backbone": str(checkpoint_path),
            "guide_freeze": True,
            "guide_fusion_stage": "direct_segmentation",
            "input_shape": [16, 16, 16],
        },
    )

    with pytest.raises(ValueError, match="exactly one non-auxiliary target"):
        NetworkFromConfig(mgr)


def test_pretrained_backbone_freeze_encoder_disables_grads_and_preserves_weights_under_init(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_pretrained_backbone_mgr(checkpoint_path, freeze_encoder=True)
    model = NetworkFromConfig(mgr)

    assert model.final_config["freeze_encoder"] is True
    assert all(not parameter.requires_grad for parameter in model.shared_encoder.parameters())

    before = {
        key: value.detach().clone()
        for key, value in model.shared_encoder.state_dict().items()
    }
    model.apply(InitWeights_He(neg_slope=0.2))
    after = model.shared_encoder.state_dict()

    assert all(torch.equal(before[key], after[key]) for key in before)
    model.train()
    assert model.shared_encoder.training is False


def test_pretrained_backbone_without_freeze_encoder_still_reinitializes_conv_stem(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_pretrained_backbone_mgr(checkpoint_path, freeze_encoder=False)
    model = NetworkFromConfig(mgr)

    before = {
        key: value.detach().clone()
        for key, value in model.shared_encoder.state_dict().items()
    }
    model.apply(InitWeights_He(neg_slope=0.2))
    after = model.shared_encoder.state_dict()

    changed = [key for key in before if not torch.equal(before[key], after[key])]
    assert "backbone.down_projection.proj.weight" in changed


def test_build_dinov2_decoder_accepts_pixelshuffle_conv(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    encoder = build_dinov2_backbone(str(checkpoint_path), input_channels=1, input_shape=(16, 16, 16))
    decoder = build_dinov2_decoder("pixelshuffle_conv", encoder, num_classes=2)

    features = torch.randn(1, encoder.embed_dim, 2, 2, 2)
    output = decoder(features)

    assert isinstance(decoder, PixelShuffleConvDinov2Decoder)
    assert output.shape == (1, 2, 16, 16, 16)
    assert all(isinstance(stage[3], torch.nn.GroupNorm) for stage in decoder.decode)
    assert all(isinstance(stage[4], torch.nn.GELU) for stage in decoder.decode)
    assert all(isinstance(stage[2], torch.nn.Conv3d) for stage in decoder.decode)
    assert all(stage[2].kernel_size == (3, 3, 3) for stage in decoder.decode)
    assert all(stage[2].padding == (1, 1, 1) for stage in decoder.decode)
    assert all(stage[2].bias is None for stage in decoder.decode)
    assert decoder.decode[-1][2].out_channels > 2
    assert decoder.final_refine[0].in_channels == decoder.decode[-1][2].out_channels
    assert decoder.final_refine[0].bias is None
    assert isinstance(decoder.final_refine[1], torch.nn.GroupNorm)
    assert isinstance(decoder.final_refine[2], torch.nn.GELU)
    assert len(decoder.final_refine) == 5
    assert decoder.final_refine[-1].kernel_size == (1, 1, 1)
    assert decoder.final_refine[0].kernel_size == (3, 3, 3)
    assert decoder.final_refine[3].kernel_size == (3, 3, 3)


def test_build_dinov2_decoder_accepts_pixelshuffle_convhead_big(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    encoder = build_dinov2_backbone(str(checkpoint_path), input_channels=1, input_shape=(16, 16, 16))
    decoder = build_dinov2_decoder("pixelshuffle_convhead_big", encoder, num_classes=2)

    features = torch.randn(1, encoder.embed_dim, 2, 2, 2)
    output = decoder(features)

    assert isinstance(decoder, PixelShuffleConvHeadBigDinov2Decoder)
    assert output.shape == (1, 2, 16, 16, 16)
    assert not hasattr(decoder, "pre_refine")
    assert all(isinstance(stage[3], torch.nn.GroupNorm) for stage in decoder.decode)
    assert all(isinstance(stage[4], torch.nn.GELU) for stage in decoder.decode)
    assert all(isinstance(stage[2], torch.nn.Conv3d) for stage in decoder.decode)
    assert all(stage[2].kernel_size == (3, 3, 3) for stage in decoder.decode)
    assert all(stage[2].padding == (1, 1, 1) for stage in decoder.decode)
    assert all(stage[2].bias is None for stage in decoder.decode)
    assert decoder.decode[-1][2].out_channels > 2
    assert decoder.final_refine[0].in_channels == decoder.decode[-1][2].out_channels
    assert decoder.final_refine[0].bias is None
    assert isinstance(decoder.final_refine[1], torch.nn.GroupNorm)
    assert isinstance(decoder.final_refine[2], torch.nn.GELU)
    assert len(decoder.final_refine) == 5
    assert decoder.final_refine[-1].kernel_size == (1, 1, 1)
    assert decoder.final_refine[0].kernel_size == (7, 7, 7)
    assert decoder.final_refine[0].padding == (3, 3, 3)
    assert decoder.final_refine[3].kernel_size == (7, 7, 7)
    assert decoder.final_refine[3].padding == (3, 3, 3)


def test_feature_encoder_guidance_backprop_updates_all_stage_tokenbooks(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)
    model.train()

    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16)
    outputs, aux = model(x, return_aux=True)
    loss = torch.nn.functional.mse_loss(outputs["ink"], target)
    loss = loss + sum(stage_aux.mean() for stage_aux in aux.values())
    loss.backward()

    assert set(aux.keys()) == {"enc_0", "enc_1", "enc_2"}
    assert all(
        any(parameter.grad is not None for parameter in tokenbook.parameters())
        for tokenbook in model.guide_stage_tokenbooks.values()
    )
    assert all(parameter.grad is None for parameter in model.guide_backbone.parameters())


def test_feature_encoder_token_mlp_backprop_updates_stage_weight_mlps(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    model = NetworkFromConfig(mgr)
    model.train()

    outputs, aux = model(torch.randn(2, 1, 16, 16, 16), return_aux=True)
    loss = outputs["ink"].square().mean() + sum(stage_aux.mean() for stage_aux in aux.values())
    loss.backward()

    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"
    assert all(
        tokenbook.prototype_weight_mlp is not None
        for tokenbook in model.guide_stage_tokenbooks.values()
    )
    assert all(
        any(parameter.grad is not None for parameter in tokenbook.prototype_weight_mlp.parameters())
        for tokenbook in model.guide_stage_tokenbooks.values()
    )


def test_guided_network_supports_reduced_tokenbook_prototype_count(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_tokenbook_tokens=3,
    )
    model = NetworkFromConfig(mgr)
    outputs, aux = model(torch.randn(1, 1, 16, 16, 16), return_aux=True)

    assert model.guide_tokenbook.book.shape[0] == 3
    assert model.final_config["guide_tokenbook_tokens"] == 3
    assert outputs["ink"].shape == (1, 1, 16, 16, 16)
    assert aux["guide_mask"].shape == (1, 1, 2, 2, 2)


def test_feature_encoder_guidance_supports_reduced_tokenbook_prototype_count(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_tokenbook_tokens=3,
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)
    outputs, aux = model(torch.randn(1, 1, 16, 16, 16), return_aux=True)

    assert model.final_config["guide_tokenbook_tokens"] == 3
    assert all(tokenbook.book.shape[0] == 3 for tokenbook in model.guide_stage_tokenbooks.values())
    assert outputs["ink"].shape == (1, 1, 16, 16, 16)
    assert set(aux.keys()) == {"enc_0", "enc_1", "enc_2"}


@pytest.mark.parametrize("guide_compile_policy", ["off", "backbone_only", "tokenbook_only", "all_guidance"])
def test_guided_network_records_guide_compile_policy(tmp_path: Path, guide_compile_policy: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy=guide_compile_policy,
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_compile_policy == guide_compile_policy
    assert model.final_config["guide_compile_policy"] == guide_compile_policy


def test_guided_network_all_guidance_compile_policy_compiles_backbone_and_tokenbook(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy="all_guidance",
    )
    model = NetworkFromConfig(mgr)
    compiled_targets: list[str] = []

    def _record_compile(module):
        compiled_targets.append(type(module).__name__)
        return module

    monkeypatch.setattr(model, "_compile_module_in_place", _record_compile)

    compiled_modules = model._compile_guidance_submodules(device_type="cuda")

    assert compiled_modules == ["guide_backbone", "guide_tokenbook"]
    assert compiled_targets == ["Dinov2Backbone", "TokenBook3D"]


def test_feature_encoder_all_guidance_compile_policy_compiles_backbone_and_stage_tokenbooks(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy="all_guidance",
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)
    compiled_targets: list[str] = []

    def _record_compile(module):
        compiled_targets.append(type(module).__name__)
        return module

    monkeypatch.setattr(model, "_compile_module_in_place", _record_compile)

    compiled_modules = model._compile_guidance_submodules(device_type="cuda")

    assert compiled_modules == [
        "guide_backbone",
        "guide_tokenbook:enc_0",
        "guide_tokenbook:enc_1",
        "guide_tokenbook:enc_2",
    ]
    assert compiled_targets == ["Dinov2Backbone", "TokenBook3D", "TokenBook3D", "TokenBook3D"]


def test_feature_encoder_guidance_records_exact_fusion_stage(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_fusion_stage == "feature_encoder"
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.guide_feature_gate_alpha == pytest.approx(1.0)


def test_feature_skip_concat_records_exact_fusion_stage_and_no_alpha(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_fusion_stage == "feature_skip_concat"
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None
    assert model.final_config["guide_tokenbook_weight_mlp_hidden"] is None


def test_feature_skip_concat_ignores_stale_tokenbook_and_alpha_keys(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_skip_concat",
        guide_tokenbook_tokens=17,
        guide_tokenbook_prototype_weighting="token_mlp",
        guide_feature_gate_alpha=0.9,
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_fusion_stage == "feature_skip_concat"
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None
    assert model.final_config["guide_tokenbook_weight_mlp_hidden"] is None
    assert model.final_config["guide_feature_gate_alpha"] is None


def test_feature_encoder_guidance_records_configured_gate_alpha(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
        guide_feature_gate_alpha=0.9,
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_feature_gate_alpha == pytest.approx(0.9)
    assert model.final_config["guide_feature_gate_alpha"] == pytest.approx(0.9)


def test_feature_encoder_guidance_rejects_invalid_gate_alpha(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
        guide_feature_gate_alpha=1.5,
    )

    with pytest.raises(ValueError, match="guide_feature_gate_alpha"):
        NetworkFromConfig(mgr)


def test_feature_encoder_guidance_records_token_mlp_weighting(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_tokenbook_prototype_weighting == "token_mlp"
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_guided_network_rejects_invalid_guide_compile_policy(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy="bad_policy",
    )

    with pytest.raises(ValueError, match="guide_compile_policy"):
        NetworkFromConfig(mgr)


def test_feature_skip_concat_all_guidance_compile_policy_compiles_backbone_and_projectors(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy="all_guidance",
        guide_fusion_stage="feature_skip_concat",
    )
    model = NetworkFromConfig(mgr)
    compiled_targets: list[str] = []

    def _record_compile(module):
        compiled_targets.append(type(module).__name__)
        return module

    monkeypatch.setattr(model, "_compile_module_in_place", _record_compile)

    compiled_modules = model._compile_guidance_submodules(device_type="cuda")

    assert compiled_modules == ["guide_backbone", "guide_projector:enc_0", "guide_projector:enc_1", "guide_projector:enc_2"]
    assert compiled_targets == ["Dinov2Backbone", "ConvDropoutNormReLU", "ConvDropoutNormReLU", "ConvDropoutNormReLU"]


def test_guided_network_rejects_primus_architecture(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)

    mgr = SimpleNamespace(
        targets={"ink": {"out_channels": 1, "activation": "none"}},
        train_patch_size=(16, 16, 16),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "architecture_type": "primus_s",
            "input_shape": [16, 16, 16],
            "patch_embed_size": [8, 8, 8],
            "guide_backbone": str(checkpoint_path),
        },
    )

    with pytest.raises(ValueError, match="not supported with primus"):
        NetworkFromConfig(mgr)
