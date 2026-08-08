from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.winding_models import winding_targets
from vesuvius.neural_tracing.winding_models.volume_plane_extractor import (
    VolumePlaneExtractor,
)
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel
from vesuvius.neural_tracing.winding_models.winding_model_dataset import (
    WindingModelDataset,
)

CONFIG_PATH = (
    Path(__file__).parents[2] / "src/vesuvius/neural_tracing/winding_models/config.json"
)

CROSSING_SIGMA_WV = 1.0


@pytest.fixture(scope="module")
def real_samples() -> list[dict]:
    cfg = json.loads(CONFIG_PATH.read_text())
    volume_cfg = cfg["datasets"][0]
    volume_path = VolumePlaneExtractor.scaled_volume_path(
        Path(volume_cfg["volume_path"]), int(volume_cfg["volume_scale"])
    )
    if not volume_path.exists():
        pytest.skip(f"configured winding volume is unavailable: {volume_path}")
    cfg["datasets"] = [volume_cfg]
    cfg["num_samples"] = 4
    dataset = WindingModelDataset(cfg)
    torch.manual_seed(0)
    return [dataset[index] for index in range(2)]


def _canonical_indices(sample: dict) -> np.ndarray:
    indices = np.asarray(sample["winding_indices"], dtype=np.float64)
    return -indices if indices[-1] < indices[0] else indices


def _spacing(sample: dict) -> float:
    return float(sample["ray_extent"]) / (int(sample["ray_length"]) - 1)


def test_phase_target_is_piecewise_linear_through_crossings(real_samples) -> None:
    for sample in real_samples:
        targets = winding_targets.render_targets(
            sample, crossing_sigma_wv=CROSSING_SIGMA_WV
        )

        spacing = _spacing(sample)
        ts = np.arange(int(sample["ray_length"])) * spacing
        crossing_t = np.asarray(sample["crossing_t"], dtype=np.float64)
        indices = _canonical_indices(sample)

        phase = targets["phase_target"].numpy()
        np.testing.assert_allclose(
            phase, np.interp(ts, crossing_t, indices), rtol=1e-5, atol=1e-5
        )
        assert (np.diff(phase) >= 0).all()
        np.testing.assert_array_equal(
            targets["phase_valid"].numpy(), np.asarray(sample["winding_valid"])
        )

        heatmap = targets["crossing_target"].numpy()
        nearest = np.rint(crossing_t / spacing).astype(int)
        np.testing.assert_array_equal(heatmap[nearest], 1.0)
        far = np.abs(ts[:, None] - crossing_t[None, :]).min(axis=1) > (
            3.0 * CROSSING_SIGMA_WV
        )
        assert (heatmap[far] < 0.05).all()

        crossing_valid = targets["crossing_valid"].numpy()
        assert (crossing_valid | ~np.asarray(sample["winding_valid"])).all()
        assert crossing_valid[nearest].all()
        assert not crossing_valid[far & ~np.asarray(sample["winding_valid"])].any()


def test_decreasing_winding_indices_are_canonicalized_to_increase(
    real_samples,
) -> None:
    sample = real_samples[0]
    flipped = dict(sample)
    flipped["winding_indices"] = -sample["winding_indices"]

    original = winding_targets.render_targets(
        sample, crossing_sigma_wv=CROSSING_SIGMA_WV
    )
    mirrored = winding_targets.render_targets(
        flipped, crossing_sigma_wv=CROSSING_SIGMA_WV
    )

    np.testing.assert_array_equal(
        original["phase_target"].numpy(), mirrored["phase_target"].numpy()
    )


def test_phase_loss_is_shift_invariant_and_masked(real_samples) -> None:
    targets = winding_targets.render_targets(
        real_samples[0], crossing_sigma_wv=CROSSING_SIGMA_WV
    )
    target = targets["phase_target"][None]
    valid = targets["phase_valid"][None]

    assert winding_targets.phase_loss(target + 7.0, target, valid).item() < 1e-9

    torch.manual_seed(0)
    pred = target + 0.3 * torch.randn_like(target)
    np.testing.assert_allclose(
        winding_targets.phase_loss(pred, target, valid).item(),
        winding_targets.phase_loss(pred + 3.0, target, valid).item(),
        rtol=1e-5,
    )

    corrupted = pred.clone()
    corrupted[~valid] = 1e4
    np.testing.assert_allclose(
        winding_targets.phase_loss(corrupted, target, valid).item(),
        winding_targets.phase_loss(pred, target, valid).item(),
        rtol=1e-5,
    )


def test_crossing_loss_ignores_masked_samples(real_samples) -> None:
    targets = winding_targets.render_targets(
        real_samples[0], crossing_sigma_wv=CROSSING_SIGMA_WV
    )
    target = targets["crossing_target"][None]
    valid = targets["crossing_valid"][None]
    if valid.all():
        pytest.skip("drawn ray has no unsupervised span")

    torch.manual_seed(0)
    logits = torch.randn_like(target)
    baseline = winding_targets.crossing_loss(logits, target, valid)
    corrupted = logits.clone()
    corrupted[~valid] = 30.0

    assert winding_targets.crossing_loss(corrupted, target, valid).item() == (
        baseline.item()
    )
    assert baseline.item() > 0.0


def test_collate_stacks_real_samples_and_pads_crossings(real_samples) -> None:
    collated = winding_targets.collate_winding_batch(
        real_samples, crossing_sigma_wv=CROSSING_SIGMA_WV
    )

    ray_length = int(real_samples[0]["ray_length"])
    counts = [len(sample["crossing_t"]) for sample in real_samples]
    assert collated["plane_images"].shape == (
        len(real_samples),
        *real_samples[0]["plane_images"].shape,
    )
    assert collated["phase_target"].shape == (len(real_samples), ray_length)
    assert collated["crossing_t"].shape == (len(real_samples), max(counts))
    np.testing.assert_array_equal(collated["num_crossings"].numpy(), counts)
    for row, count in enumerate(counts):
        assert torch.isnan(collated["crossing_t"][row, count:]).all()
        np.testing.assert_array_equal(
            collated["crossing_t"][row, :count].numpy(),
            np.asarray(real_samples[row]["crossing_t"]),
        )


def test_peak_decoding_recovers_labeled_crossings(real_samples) -> None:
    for sample in real_samples:
        targets = winding_targets.render_targets(
            sample, crossing_sigma_wv=CROSSING_SIGMA_WV
        )
        spacing = _spacing(sample)

        peaks = winding_targets.extract_peaks(
            targets["crossing_target"].numpy(), threshold=0.3, min_distance=2
        )
        tp, fp, fn = winding_targets.match_crossings(
            peaks.astype(np.float64),
            np.asarray(sample["crossing_t"]) / spacing,
            tolerance=2.0 / spacing,
        )

        assert (tp, fp, fn) == (len(sample["crossing_t"]), 0, 0)


def test_model_outputs_full_resolution_monotone_phase(real_samples) -> None:
    collated = winding_targets.collate_winding_batch(
        real_samples, crossing_sigma_wv=CROSSING_SIGMA_WV
    )
    torch.manual_seed(0)
    model = WindingModel(
        {
            "encoder_channels": [8, 16],
            "transformer_dim": 32,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "max_relative_distance": 16,
        }
    )

    with torch.no_grad():
        output = model(collated["plane_images"], collated["plane_valid"])

    batch, _, _, ray_length = collated["plane_images"].shape
    assert output["phase"].shape == (batch, ray_length)
    assert output["crossing_logits"].shape == (batch, ray_length)
    assert (output["phase"].diff(dim=-1) >= 0).all()
