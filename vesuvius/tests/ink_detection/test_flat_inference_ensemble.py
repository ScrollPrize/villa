from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch
from torch import nn
import zarr

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference.infer import (
    ProbabilityMeanEnsemble,
    main,
    normalize_inference_paths,
    parse_args,
    resolve_checkpoint_paths,
)
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


class _ConstantLogits(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, image_BCZYX: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = image_BCZYX.shape
        return torch.full((batch, 1, height, width), self.value)


def _save_checkpoint(path, *, depth: int = 3, side: int = 16, seed: int = 0):
    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=depth, side=side)
    config_mapping["image_normalization"] = "robust_mad"
    torch.manual_seed(seed)
    model = make_model(InkConfig.from_mapping(config_mapping))
    torch.save({"config": config_mapping, "model": model.state_dict()}, path)
    return path


def _write_surface(path, *, depth: int = 3, side: int = 16):
    array = zarr.open(
        path,
        mode="w",
        shape=(depth, side, side),
        chunks=(depth, side, side),
        dtype="u1",
        zarr_format=2,
    )
    array[:] = np.random.default_rng(0).integers(1, 255, size=(depth, side, side))
    return path


def _run(input_path, checkpoint, output, *extra):
    argv = [
        str(input_path),
        str(checkpoint),
        str(output),
        "--workers",
        "0",
        "--no-compile",
        "--blend-mode",
        "constant",
        *(str(value) for value in extra),
    ]
    assert main(argv) == 0
    return tifffile.imread(output)


def test_probability_mean_ensemble_averages_probabilities_not_logits():
    ensemble = ProbabilityMeanEnsemble([_ConstantLogits(-4.0), _ConstantLogits(2.0)])
    images = torch.zeros((2, 1, 3, 8, 8))

    probabilities = ensemble(images).sigmoid()

    expected = (torch.sigmoid(torch.tensor(-4.0)) + torch.sigmoid(torch.tensor(2.0))) / 2
    assert probabilities.shape == (2, 1, 8, 8)
    assert torch.allclose(probabilities, expected.expand_as(probabilities), atol=1e-6)
    # A logit mean would give sigmoid(-1) = 0.269; the probability mean is 0.449.
    assert not torch.allclose(probabilities, torch.sigmoid(torch.tensor(-1.0)))


def test_probability_mean_ensemble_saturated_members_stay_finite():
    ensemble = ProbabilityMeanEnsemble([_ConstantLogits(60.0), _ConstantLogits(60.0)])

    logits = ensemble(torch.zeros((1, 1, 3, 4, 4)))

    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.sigmoid(), torch.ones_like(logits), atol=1e-5)


def test_probability_mean_ensemble_requires_two_members():
    with pytest.raises(ValueError, match="at least two"):
        ProbabilityMeanEnsemble([_ConstantLogits(0.0)])


def test_ensemble_checkpoint_cli_is_repeatable_and_optional(tmp_path):
    single = normalize_inference_paths(
        parse_args(["in.zarr", "a.pth", "out.tif"])
    )
    assert resolve_checkpoint_paths(single) == [single.checkpoint]

    args = normalize_inference_paths(
        parse_args(
            [
                "in.zarr",
                "a.pth",
                "out.tif",
                "--ensemble-checkpoint",
                "b.pth",
                "--ensemble_checkpoint",
                "c.pth",
            ]
        )
    )
    assert [path.name for path in resolve_checkpoint_paths(args)] == [
        "a.pth",
        "b.pth",
        "c.pth",
    ]


def test_cpu_ensemble_tiff_is_mean_of_member_probabilities(tmp_path):
    first = _save_checkpoint(tmp_path / "seed0.pth", seed=0)
    second = _save_checkpoint(tmp_path / "seed1.pth", seed=1)
    surface = _write_surface(tmp_path / "surface.zarr")

    single_first = _run(surface, first, tmp_path / "first.tif").astype(np.int16)
    single_second = _run(surface, second, tmp_path / "second.tif").astype(np.int16)
    ensemble = _run(
        surface, first, tmp_path / "ensemble.tif", "--ensemble-checkpoint", second
    ).astype(np.int16)

    assert not np.array_equal(single_first, single_second)
    # Each TIFF truncates probability * 255, so the member mean is within 1 LSB.
    assert np.abs(ensemble * 2 - (single_first + single_second)).max() <= 2
    # Re-running the primary checkpoint alone is unchanged by the new option.
    assert np.array_equal(
        _run(surface, first, tmp_path / "again.tif").astype(np.int16), single_first
    )


def test_ensemble_rejects_incompatible_checkpoint_contract(tmp_path):
    first = _save_checkpoint(tmp_path / "depth3.pth", depth=3)
    deeper = _save_checkpoint(tmp_path / "depth5.pth", depth=5)
    surface = _write_surface(tmp_path / "surface.zarr", depth=5)

    with pytest.raises(ValueError, match="input_depth=5 \\(expected 3\\)"):
        _run(
            surface, first, tmp_path / "out.tif", "--ensemble-checkpoint", deeper
        )
