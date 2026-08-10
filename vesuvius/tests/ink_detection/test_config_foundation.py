"""Boundary witnesses for canonical model, target, loss, and data configuration."""

from __future__ import annotations

from copy import deepcopy
import pickle

import pytest

from vesuvius.ink_detection.config import InkConfig

from .test_model_foundation import _config_mapping


def test_non_dinov2_mapping_round_trips_without_inserted_defaults():
    authored = _config_mapping("vesuvius_unet_3d_stem_2d", depth=3)
    authored["description"] = "kept"
    authored["custom"] = {"ordered": [3, 1], "enabled": False}
    expected = deepcopy(authored)

    config = InkConfig.from_mapping(authored)
    first = config.to_mapping()
    first["custom"]["ordered"].append(9)

    assert config.to_mapping() == expected
    assert list(config.to_mapping()) == list(expected)
    assert "crop_size" not in config.to_mapping()
    assert "ema" not in config.to_mapping()


def test_full_typed_config_is_pickle_safe_and_keeps_mapping_order():
    authored = _config_mapping("vesuvius_unet_3d_stem_2d", depth=3)
    authored["custom"] = {"values": [2, 1]}

    restored = pickle.loads(pickle.dumps(InkConfig.from_mapping(authored)))

    assert restored.to_mapping() == authored
    assert list(restored.to_mapping()) == list(authored)


def test_recursive_frozen_mappings_reject_internal_storage_assignment():
    authored = _config_mapping("vesuvius_unet_3d_stem_2d", depth=3)
    authored["custom"] = {"nested": {"ordered": [3, 1]}}
    config = InkConfig.from_mapping(authored)
    frozen_mappings = (
        config._canonical,
        config._canonical["custom"],
        config._canonical["custom"]["nested"],
    )

    for frozen in frozen_mappings:
        with pytest.raises(AttributeError, match="immutable"):
            frozen._items = (("replaced", True),)
        with pytest.raises(AttributeError, match="immutable"):
            del frozen._items

    copied = deepcopy(config)
    assert copied is not config
    assert copied.to_mapping() == authored
    assert config.to_mapping() == authored
    assert list(config.to_mapping()) == list(authored)


def test_dinov2_rewrite_preserves_keys_and_setdefault_values():
    authored = _config_mapping()
    authored["model_type"] = "dinov2"
    authored["pretrained_backbone"] = "/top/backbone.pth"
    authored["pretrained_decoder_type"] = "linear"
    authored["model_config"]["pretrained_backbone"] = "/nested/backbone.pth"

    canonical = InkConfig.from_mapping(authored).to_mapping()

    assert canonical["model_type"] == "vesuvius_unet"
    assert canonical["pretrained_backbone"] == "/top/backbone.pth"
    assert canonical["pretrained_decoder_type"] == "linear"
    assert canonical["model_config"]["pretrained_backbone"] == "/nested/backbone.pth"
    assert canonical["model_config"]["pretrained_decoder_type"] == "linear"


def test_dinov2_requires_a_pretrained_backbone():
    authored = _config_mapping()
    authored["model_type"] = "dinov2"

    with pytest.raises(ValueError, match="dinov2.*pretrained_backbone"):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("guide_backbone", "dinov2", "guide_backbone"),
        ("architecture_type", "mednext_v1", "architecture_type"),
        ("upsample_mode", "pixelshuffle", "upsample_mode"),
        (
            "target_z_projection",
            {"ink": {"mode": "max"}},
            "target_z_projection",
        ),
        (
            "pretrained_backbone_config_path",
            "/different/backbone.json",
            "pretrained_backbone_config_path",
        ),
    ],
)
def test_unsupported_model_construction_paths_fail_at_config_boundary(
    key,
    value,
    message,
):
    authored = _config_mapping()
    authored["model_config"][key] = value
    authored["model_config"]["guide_fusion_stage"] = "direct_segmentation"
    authored["model_config"]["mednext_model_id"] = "S"
    if key == "pretrained_backbone_config_path":
        authored["model_config"]["pretrained_backbone"] = "dinov2"

    with pytest.raises(ValueError, match=message):
        InkConfig.from_mapping(authored)


def test_accepted_unknown_model_settings_remain_preserved():
    authored = _config_mapping()
    authored["model_config"]["architecture_type"] = "primus_s"
    authored["model_config"]["freeze_encoder"] = True
    authored["model_config"]["guide_fusion_stage"] = "feature_encoder"
    authored["model_config"]["mednext_model_id"] = "S"
    authored["model_config"]["upsample_mode"] = "transpconv"
    authored["model_config"]["target_z_projection"] = {}
    authored["model_config"]["pretrained_backbone_config_path"] = (
        "/inactive/backbone.json"
    )
    authored["model_config"]["future_training_metadata"] = {
        "ordered": [2, 1]
    }

    canonical = InkConfig.from_mapping(authored).to_mapping()

    assert canonical == authored
    assert list(canonical["model_config"]) == list(authored["model_config"])


@pytest.mark.parametrize("missing", ["patch_overlap", "patch_min_labeled_coverage"])
def test_full_config_keeps_required_patch_values_required(missing):
    authored = _config_mapping()
    authored.pop(missing)

    with pytest.raises(KeyError, match=missing):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize("model_type", ["resnet3d", "resnet3d-50", "transformer"])
def test_unsupported_model_values_fail_factually(model_type):
    authored = _config_mapping()
    authored["model_type"] = model_type

    with pytest.raises(ValueError, match="model_type"):
        InkConfig.from_mapping(authored)


def test_normal_pooled_mode_fails_factually():
    authored = _config_mapping()
    authored["mode"] = "normal_pooled_3d"

    with pytest.raises(ValueError, match="mode.*full_3d"):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize("target_name", ["papyrus", "mask"])
def test_unknown_target_names_fail_factually(target_name):
    authored = _config_mapping()
    authored["targets"] = {
        target_name: {
            "out_channels": 1,
            "activation": "none",
            "z_projection_mode": "none",
        }
    }

    with pytest.raises(ValueError, match="Unsupported target"):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("out_channels", 2, "out_channels"),
        ("activation", "sigmoid", "activation"),
        ("z_projection_mode", "mystery", "z_projection"),
    ],
)
def test_unknown_target_settings_fail_factually(key, value, message):
    authored = _config_mapping()
    authored["targets"]["ink"][key] = value

    with pytest.raises(ValueError, match=message):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize("location", ["nested_target", "model_config"])
def test_projection_modes_are_validated_without_storing_a_second_view(location):
    authored = _config_mapping()
    if location == "nested_target":
        authored["targets"]["ink"]["z_projection"] = {"mode": "mystery"}
    else:
        authored["model_config"]["z_projection_mode"] = "mystery"

    with pytest.raises(ValueError, match="z_projection"):
        InkConfig.from_mapping(authored)


@pytest.mark.parametrize("name", ["BettiMatchingLoss", "NearbyDiceLoss"])
def test_unsupported_loss_names_fail_factually(name):
    authored = _config_mapping()
    authored["loss"] = {"terms": [{"name": name}]}

    with pytest.raises(ValueError, match="Unsupported loss term"):
        InkConfig.from_mapping(authored)
