"""Pooled-to-native scan-type augmentation: config parsing and batch behaviour."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.training.scan_transfer import ScanTransfer, load_kernel

from .helpers import config_mapping


def _training_mapping(**extra) -> dict:
    authored = config_mapping(depth=3, side=8)
    authored.update({"mode": "flat", "num_iterations": 6, "out_dir": "/tmp/ink-output", "seed": 17})
    authored.update(extra)
    return authored


def _parse(authored: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


def _delta(kz=5, kxy=9) -> np.ndarray:
    k = np.zeros((kz, kxy, kxy), np.float32)
    k[kz // 2, kxy // 2, kxy // 2] = 1.0
    return k


def test_scan_transfer_is_off_unless_enabled():
    assert _parse(_training_mapping()).scan_transfer is None
    assert _parse(_training_mapping(scan_transfer={"enabled": False, "kernel": "x.json"})).scan_transfer is None


def test_scan_transfer_config_round_trip():
    config = _parse(
        _training_mapping(
            scan_transfer={"enabled": True, "kernel": "builtin:pooled_to_native_pherc0814", "probability": 0.25}
        )
    ).scan_transfer
    assert config.kernel == "builtin:pooled_to_native_pherc0814"
    assert config.probability == 0.25
    assert config.noise_amplitude == 0.5


@pytest.mark.parametrize(
    "value, message",
    [
        ({"enabled": True}, "kernel is required"),
        ({"enabled": True, "kernel": "k.json", "probability": 1.5}, "probability"),
        ({"enabled": True, "kernel": "k.json", "noise_amplitude": -1}, "noise_amplitude"),
        ({"enabled": True, "kernel": "k.json", "noise_sigma_xy": 0}, "sigmas"),
    ],
)
def test_scan_transfer_config_rejects_bad_values(value, message):
    with pytest.raises(ValueError, match=message):
        _parse(_training_mapping(scan_transfer=value))


def test_scan_transfer_is_flat_only():
    authored = _training_mapping(scan_transfer={"enabled": True, "kernel": "k.json"})
    authored["mode"] = "full_3d"
    authored.pop("in_channels", None)
    authored["targets"]["ink"].pop("out_channels", None)
    authored["targets"]["ink"].pop("activation", None)
    with pytest.raises(ValueError, match="only in mode='flat'"):
        _parse(authored)


def test_builtin_kernel_loads():
    kernel = load_kernel("builtin:pooled_to_native_pherc0814")
    assert kernel.shape == (5, 9, 9)
    assert 0.8 < float(kernel.sum()) < 1.2


def test_kernel_from_json_file(tmp_path):
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"kernel": _delta().tolist()}), encoding="utf-8")
    assert np.array_equal(load_kernel(str(path)), _delta())
    path.write_text(json.dumps({"kernel": np.zeros((4, 9, 9)).tolist()}), encoding="utf-8")
    with pytest.raises(ValueError, match="odd sizes"):
        load_kernel(str(path))


def test_probability_zero_leaves_the_batch_untouched():
    image = torch.randn(4, 1, 7, 32, 32)
    out = ScanTransfer(_delta(), probability=0.0)(image)
    assert torch.equal(out, image)


def test_delta_kernel_without_noise_is_identity():
    image = torch.randn(4, 1, 7, 32, 32)
    out = ScanTransfer(_delta(), probability=1.0, noise_amplitude=0.0)(image)
    assert torch.allclose(out, image, atol=1e-4)


def test_blur_keeps_each_samples_mean_and_std():
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(3, 1, 9, 48, 48, generator=generator)
    out = ScanTransfer(load_kernel("builtin:pooled_to_native_pherc0814"), probability=1.0, noise_amplitude=0.0)(image)
    assert not torch.allclose(out, image)
    assert torch.allclose(out.flatten(1).mean(1), image.flatten(1).mean(1), atol=1e-3)
    assert torch.allclose(out.flatten(1).std(1), image.flatten(1).std(1), atol=1e-3)


def test_only_selected_samples_change():
    generator = torch.Generator().manual_seed(3)
    image = torch.randn(16, 1, 5, 24, 24)
    out = ScanTransfer(_delta(), probability=0.5, noise_amplitude=0.5, generator=generator)(image)
    changed = (out - image).flatten(1).abs().amax(1) > 1e-6
    assert 0 < int(changed.sum()) < 16


def test_rejects_wrong_layout():
    with pytest.raises(ValueError, match="expects"):
        ScanTransfer(_delta(), probability=1.0)(torch.randn(2, 5, 24, 24))
