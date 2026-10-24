"""Unit tests for the comparison in verify_windows_ink.py (no GPU, no network).

They pin down that the metric (a) recovers r ~ 1 and zero shift when the
"figure" is made from the prediction exactly like the tutorial figure
(display rescale + downsample to the figure size), (b) finds a known shift,
and (c) does not pass an unrelated prediction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import verify_windows_ink as v  # noqa: E402

pytestmark = pytest.mark.slow  # ~20 s: builds a full-segment-sized synthetic map

FIG_W, FIG_H = 1441, 1600  # size of the published tutorial figures


def _figure_from(full_pred: np.ndarray) -> np.ndarray:
    display = np.clip((full_pred - 0.25) / 0.5, 0, 1)
    img = Image.fromarray((display * 255).astype(np.uint8)).resize((FIG_W, FIG_H), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


@pytest.fixture(scope="module")
def full_pred():
    rng = np.random.default_rng(0)
    from scipy import ndimage

    h, w = v.VOLUME_SHAPE[1:]
    noise = ndimage.gaussian_filter(rng.random((h, w), dtype=np.float32), 6)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (0.2 + 0.6 * noise).astype(np.float32)


def _crop(full, dy=0, dx=0):
    y0, x0, h, w = v.CROP
    return full[y0 + dy : y0 + dy + h, x0 + dx : x0 + dx + w]


def test_same_prediction_matches_its_figure(full_pred):
    result = v.compare(_crop(full_pred), _figure_from(full_pred))
    assert result["pearson_r"] > 0.98
    assert result["best_shift_volume_px"] == [0, 0] or max(map(abs, result["best_shift_volume_px"])) <= 1


def test_known_offset_is_found(full_pred):
    result = v.compare(_crop(full_pred, dy=4, dx=-3), _figure_from(full_pred))
    dy, dx = result["best_shift_volume_px"]
    assert abs(dy + 4) <= 1 and abs(dx - 3) <= 1, result
    assert result["pearson_r"] > 0.98


def test_unrelated_prediction_does_not_pass(full_pred):
    other = np.ascontiguousarray(full_pred[::-1, ::-1])
    result = v.compare(_crop(other), _figure_from(full_pred))
    assert result["pearson_r"] < v.R_MIN - v.MARGIN


def test_parity_metric_tolerates_rounding_but_not_real_differences(full_pred):
    crop = _crop(full_pred)
    rounded = np.round(crop * 255) / 255  # uint8 TIFF round trip
    ok = v.compare_parity(crop, rounded)
    assert ok["pearson_r"] >= v.PARITY_R_MIN and ok["mae"] <= v.PARITY_MAE_MAX
    shifted = v.compare_parity(crop, _crop(full_pred, dy=8))
    assert not (shifted["pearson_r"] >= v.PARITY_R_MIN and shifted["mae"] <= v.PARITY_MAE_MAX)
