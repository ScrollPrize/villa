from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vc_reader.reader import ZarrWriter
from vesuvius.data.vc_dataset import VCDataset
from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.run import inference
from vesuvius.models.run.inference import (
    Inferer,
    _checkpoint_normalization_scheme,
    _normalize_train_py_model_config,
    _select_evenly_spaced_middle_indices,
    _select_train_py_state_dict,
)
from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization


def _tiny_model_config() -> dict:
    return {
        "basic_encoder_block": "BasicBlockD",
        "basic_decoder_block": "ConvBlock",
        "bottleneck_block": "BasicBlockD",
        "features_per_stage": [4, 8],
        "n_stages": 2,
        "n_blocks_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "kernel_sizes": [[3, 3, 3], [3, 3, 3]],
        "strides": [[1, 1, 1], [2, 2, 2]],
        "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
        "separate_decoders": False,
        "autoconfigure": False,
    }


def _build_network_state(model_config: dict) -> dict:
    mgr = SimpleNamespace(
        model_config=model_config,
        targets=model_config["targets"],
        train_patch_size=model_config["train_patch_size"],
        train_batch_size=model_config["train_batch_size"],
        in_channels=model_config["in_channels"],
        autoconfigure=model_config.get("autoconfigure", False),
        enable_deep_supervision=False,
        model_name=model_config["model_name"],
        spacing=[1, 1, 1],
    )
    model = NetworkFromConfig(mgr)
    return model.state_dict()


def _constant_like_state(state: dict, value: float) -> dict:
    return {
        key: torch.full_like(tensor, value) if torch.is_floating_point(tensor) else tensor.clone()
        for key, tensor in state.items()
    }


def test_legacy_checkpoint_config_and_ema_selection() -> None:
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema_model": {"weight": torch.tensor([2.0])},
        "config": {
            "model_config": {"autoconfigure": True},
            "patch_size": [256, 256, 256],
            "batch_size": 2,
            "in_channels": 1,
            "targets": {"ink": {"out_channels": 1, "activation": "none"}},
            "wandb_run_name": "ps256_legacy_ink",
            "image_normalization": "percentile_minmax",
            "ema": {"validate": True},
        },
    }

    model_config = _normalize_train_py_model_config(checkpoint)
    state_dict, source = _select_train_py_state_dict(checkpoint)

    assert model_config["patch_size"] == (256, 256, 256)
    assert model_config["train_patch_size"] == (256, 256, 256)
    assert model_config["batch_size"] == 2
    assert model_config["train_batch_size"] == 2
    assert model_config["in_channels"] == 1
    assert model_config["targets"] == {"ink": {"out_channels": 1, "activation": "none"}}
    assert model_config["model_name"] == "ps256_legacy_ink"
    assert model_config["enable_deep_supervision"] is False
    assert _checkpoint_normalization_scheme(checkpoint) == "percentile_minmax"
    assert source == "ema_model"
    torch.testing.assert_close(state_dict["weight"], torch.tensor([2.0]))


def test_legacy_checkpoint_raw_weights_when_ema_validation_disabled() -> None:
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema_model": {"weight": torch.tensor([2.0])},
        "config": {"ema": {"validate": False}},
    }

    state_dict, source = _select_train_py_state_dict(checkpoint)

    assert source == "model"
    torch.testing.assert_close(state_dict["weight"], torch.tensor([1.0]))


def test_select_evenly_spaced_middle_indices_prefers_middle_for_one_patch() -> None:
    assert _select_evenly_spaced_middle_indices([10, 20, 30, 40, 50], 1) == [30]


def test_max_patches_prefers_non_empty_candidates() -> None:
    inferer = Inferer.__new__(Inferer)
    inferer.max_patches = 2
    inferer.verbose = False
    inferer.part_id = 0
    inferer.num_parts = 1
    inferer.dataset = SimpleNamespace(
        all_positions=[(idx, 0, 0) for idx in range(6)],
        non_empty_mask=np.array([False, True, False, True, True, False]),
    )

    inferer._limit_dataset_patches()

    assert inferer.dataset.all_positions == [(1, 0, 0), (4, 0, 0)]
    np.testing.assert_array_equal(inferer.dataset.non_empty_mask, np.array([True, True]))


def test_max_patches_keeps_empty_fallback_when_no_non_empty_candidates() -> None:
    inferer = Inferer.__new__(Inferer)
    inferer.max_patches = 1
    inferer.verbose = False
    inferer.part_id = 0
    inferer.num_parts = 1
    inferer.dataset = SimpleNamespace(
        all_positions=[(idx, 0, 0) for idx in range(5)],
        non_empty_mask=np.array([False, False, False, False, False]),
    )

    inferer._limit_dataset_patches()

    assert inferer.dataset.all_positions == [(2, 0, 0)]
    np.testing.assert_array_equal(inferer.dataset.non_empty_mask, np.array([False]))


def test_load_train_py_model_supports_legacy_checkpoint_with_ema(tmp_path: Path) -> None:
    legacy_model_config = _tiny_model_config()
    legacy_config = {
        "model_config": legacy_model_config,
        "patch_size": [8, 8, 8],
        "batch_size": 1,
        "in_channels": 1,
        "targets": {"ink": {"out_channels": 1, "activation": "none"}},
        "wandb_run_name": "tiny_legacy",
        "image_normalization": "percentile_minmax",
        "ema": {"validate": True},
    }
    normalized_config = _normalize_train_py_model_config({"config": legacy_config})
    base_state = _build_network_state(normalized_config)
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(
        {
            "model": _constant_like_state(base_state, 1.0),
            "ema_model": _constant_like_state(base_state, 2.0),
            "config": legacy_config,
        },
        checkpoint_path,
    )

    inferer = Inferer.__new__(Inferer)
    inferer.device = torch.device("cpu")
    inferer.verbose = False
    inferer.model_normalization_scheme = None
    inferer.model_intensity_properties = None
    inferer.is_multi_task = False
    inferer.target_info = None

    model_info = inferer._load_train_py_model(checkpoint_path)
    loaded_state = model_info["network"].state_dict()
    floating_key = next(key for key, tensor in loaded_state.items() if torch.is_floating_point(tensor))

    assert model_info["patch_size"] == (8, 8, 8)
    assert model_info["num_input_channels"] == 1
    assert model_info["num_seg_heads"] == 1
    assert inferer.model_normalization_scheme == "percentile_minmax"
    torch.testing.assert_close(loaded_state[floating_key], torch.full_like(loaded_state[floating_key], 2.0))


def _normalized(path: Path, scheme: str) -> np.ndarray:
    dataset = VCDataset(input_path=str(path), patch_size=(1, 1, 1), normalization_scheme=scheme,
                        return_as_type="np.float32", skip_empty_patches=False)
    return dataset._read_patch(*(slice(None),) * 3).numpy()


def test_percentile_minmax_normalization(tmp_path: Path) -> None:
    path = tmp_path / "input.zarr"
    source = np.array([[[0, 10, 20, 30, 255]]], dtype=np.uint8)
    ZarrWriter.create(path, shape=(1, 1, 5), chunks=(1, 1, 5), dtype="uint8")[:] = source

    normalized = _normalized(path, "percentile_minmax")
    lower, upper = np.percentile(source.astype(np.float32), (1.0, 99.0))
    expected = (np.clip(source.astype(np.float32), lower, upper) - lower) / (upper - lower)

    np.testing.assert_allclose(normalized, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_percentile_minmax_constant_input_returns_zero(tmp_path: Path) -> None:
    path = tmp_path / "constant.zarr"
    ZarrWriter.create(path, shape=(2, 2, 2), chunks=(2, 2, 2), dtype="uint8")[:] = 7

    np.testing.assert_array_equal(_normalized(path, "percentile_minmax"), np.zeros((2, 2, 2), dtype=np.float32))


def test_vcdataset_passes_anon_to_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    path = tmp_path / "input.zarr"
    ZarrWriter.create(path, shape=(8, 8, 8), chunks=(8, 8, 8), dtype="uint8")
    from vc_reader.reader import open_node as real_open_node

    def open_node(location, **kwargs):
        captured["auth"] = kwargs["auth"]
        return real_open_node(location)

    monkeypatch.setattr("vesuvius.data.vc_dataset.open_node", open_node)

    dataset = VCDataset(input_path=str(path), patch_size=(8, 8, 8), anon=True, skip_empty_patches=False)

    assert captured["auth"] is None
    assert len(dataset) == 1


def test_main_accepts_input_anon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    logits_path = tmp_path / "logits_part_0.zarr"
    coords_path = tmp_path / "coordinates_part_0.zarr"
    logits_path.mkdir()
    coords_path.mkdir()

    class FakeInferer:
        skip_empty_patches = False
        dataset = None

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def infer(self):
            return str(logits_path), str(coords_path)

    monkeypatch.setattr(inference, "Inferer", FakeInferer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesuvius.predict",
            "--model_path",
            str(tmp_path / "model.pth"),
            "--input_dir",
            "s3://example-bucket/input.zarr",
            "--output_dir",
            str(tmp_path / "out"),
            "--input_anon",
            "--max_patches",
            "1",
            "--device",
            "cpu",
        ],
    )

    assert inference.main() == 0
    assert captured["input_anon"] is True
    assert captured["max_patches"] == 1


def test_finalization_handles_single_channel_multitask_probabilities() -> None:
    logits = np.array([[[[-2.0, 0.0, 2.0]]]], dtype=np.float32)
    config = FinalizeConfig(
        mode="binary",
        threshold=None,
        is_multi_task=True,
        target_info={"ink": {"start_channel": 0, "end_channel": 1, "out_channels": 1}},
    )

    output, is_empty = apply_finalization(logits, num_classes=1, config=config)
    expected_probs = 1.0 / (1.0 + np.exp(-logits))
    expected = np.clip(expected_probs * 255.0, 0, 255).astype(np.uint8)

    assert is_empty is False
    np.testing.assert_array_equal(output, expected)


def test_finalization_handles_single_channel_multitask_threshold() -> None:
    logits = np.array([[[[-1.0, 0.0, 1.0]]]], dtype=np.float32)
    config = FinalizeConfig(
        mode="binary",
        threshold=0.5,
        is_multi_task=True,
        target_info={"ink": {"start_channel": 0, "end_channel": 1, "out_channels": 1}},
    )

    output, is_empty = apply_finalization(logits, num_classes=1, config=config)

    assert is_empty is False
    np.testing.assert_array_equal(output, np.array([[[[0, 0, 255]]]], dtype=np.uint8))
