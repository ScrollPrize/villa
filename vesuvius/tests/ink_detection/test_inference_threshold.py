from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.inference import threshold


_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _write_label_zarr(path: Path, plane: np.ndarray, depth: int = 65) -> None:
    volume = np.zeros((depth, *plane.shape), dtype=np.uint8)
    volume[depth // 2] = plane
    root = zarr.open_group(path, mode="w")
    if _ZARR_V3:
        root.create_array("0", data=volume, chunks=(depth, *plane.shape))
    else:
        root.create_dataset("0", data=volume, chunks=(depth, *plane.shape))


def _segment(tmp_path: Path, name: str, best: int, seed: int) -> tuple[str, str]:
    """A prediction whose ink scores sit just above ``best`` and background just below."""
    rng = np.random.default_rng(seed)
    ink = np.zeros((40, 50), dtype=bool)
    ink[10:30, 5:45] = True
    prediction = np.where(ink, rng.integers(best, best + 40, ink.shape), rng.integers(0, best, ink.shape))
    pred_path = tmp_path / f"{name}.tif"
    tifffile.imwrite(pred_path, prediction.astype(np.uint8))
    label_path = tmp_path / f"{name}_inklabels.zarr"
    _write_label_zarr(label_path, ink.astype(np.uint8) * 255)
    return str(pred_path), str(label_path)


def test_f1_curve_matches_direct_count():
    rng = np.random.default_rng(0)
    prediction = rng.integers(0, 256, (30, 30)).astype(np.uint8)
    ink = rng.random((30, 30)) < 0.3
    scored = rng.random((30, 30)) < 0.8
    curve = threshold.f1_curve(prediction, ink, scored)
    for t in (0, 1, 64, 128, 200, 255):
        marked = prediction >= t
        tp = np.sum(marked & ink & scored)
        fp = np.sum(marked & ~ink & scored)
        fn = np.sum(~marked & ink & scored)
        expected = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        assert curve[t] == pytest.approx(expected)


def test_calibrate_takes_median_and_scores_each_scroll_at_the_others(tmp_path):
    cells = []
    for scroll, name, best, seed in (("A", "a1", 90, 1), ("A", "a2", 92, 2), ("B", "b1", 120, 3)):
        cells.append([scroll, *_segment(tmp_path, name, best, seed)])
    out = tmp_path / "calibration.json"

    result = threshold.calibrate(cells, out, checkpoint="test")

    assert [cell["best_threshold"] for cell in result["cells"]] == [90, 92, 120]
    assert all(cell["f1_best"] == pytest.approx(1.0) for cell in result["cells"])
    assert result["threshold"] == 92
    check = result["check_against_other_scrolls"]
    assert check["A"]["threshold_from_other_scrolls"] == 120
    assert check["B"]["threshold_from_other_scrolls"] == 91
    assert check["A"]["mean_f1_loss_value"] > 0
    assert check["B"]["mean_f1_loss_128"] > 0
    assert json.loads(out.read_text(encoding="utf-8")) == result


def test_calibrate_mask_limits_scoring(tmp_path):
    pred_path, label_path = _segment(tmp_path, "a1", 90, 1)
    prediction = tifffile.imread(pred_path)
    prediction[:5] = 255  # false positives outside the mask
    tifffile.imwrite(pred_path, prediction)
    mask = np.ones(prediction.shape, dtype=np.uint8)
    mask[:5] = 0
    mask_path = tmp_path / "a1_supervision_mask.tif"
    tifffile.imwrite(mask_path, mask)

    unmasked = threshold.calibrate([["A", pred_path, label_path]], tmp_path / "u.json")
    masked = threshold.calibrate([["A", pred_path, label_path, str(mask_path)]], tmp_path / "m.json")

    assert unmasked["cells"][0]["f1_best"] < 1.0
    assert masked["cells"][0]["f1_best"] == pytest.approx(1.0)
    assert masked["check_against_other_scrolls"] is None


def test_quantile_rule_follows_each_predictions_own_scale(tmp_path):
    # Two scrolls whose optima differ by 30 grey levels but mark the same share of the sheet.
    cells = [["A", *_segment(tmp_path, "a1", 90, 1)], ["B", *_segment(tmp_path, "b1", 120, 2)]]
    result = threshold.calibrate(cells, tmp_path / "c.json")

    fractions = [cell["sheet_fraction_at_best"] for cell in result["cells"]]
    assert fractions[0] == pytest.approx(fractions[1], abs=0.02)
    check = result["check_against_other_scrolls"]
    assert check["A"]["mean_f1_loss_value"] > 0.1
    assert check["A"]["mean_f1_loss_quantile"] < 0.05
    assert check["B"]["mean_f1_loss_quantile"] < 0.05


def test_apply_quantile_marks_the_calibrated_fraction(tmp_path):
    prediction = np.arange(1, 257, dtype=np.int32).clip(1, 255).astype(np.uint8).reshape(16, 16)
    pred_path = tmp_path / "unlabelled.tif"
    tifffile.imwrite(pred_path, prediction)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"threshold": 100, "sheet_fraction": 0.25}), encoding="utf-8")

    report = threshold.apply(pred_path, calibration, tmp_path / "mask.tif", rule="quantile")

    assert report["rule"] == "quantile"
    assert report["fraction_of_sheet_marked"] <= 0.25
    assert report["fraction_of_sheet_marked"] > 0.24
    mask = tifffile.imread(tmp_path / "mask.tif")
    assert np.array_equal(mask, np.where(prediction >= report["threshold"], 255, 0))


def test_apply_binarizes_at_the_calibrated_threshold(tmp_path):
    prediction = np.arange(256, dtype=np.uint8).reshape(16, 16)
    pred_path = tmp_path / "unlabelled.tif"
    tifffile.imwrite(pred_path, prediction)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"threshold": 100}), encoding="utf-8")
    out = tmp_path / "mask.tif"

    report = threshold.apply(pred_path, calibration, out)

    mask = tifffile.imread(out)
    assert mask.dtype == np.uint8
    assert np.array_equal(mask, np.where(prediction >= 100, 255, 0))
    assert report["threshold"] == 100


def test_rejects_mismatched_shapes_and_non_uint8(tmp_path):
    pred_path, _ = _segment(tmp_path, "a1", 90, 1)
    small_label = tmp_path / "small.tif"
    tifffile.imwrite(small_label, np.ones((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        threshold.calibrate([["A", pred_path, str(small_label)]], tmp_path / "x.json")

    float_pred = tmp_path / "float.tif"
    tifffile.imwrite(float_pred, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        threshold.read_prediction(float_pred)


def test_cli_round_trip(tmp_path):
    pred_a, label_a = _segment(tmp_path, "a1", 90, 1)
    pred_b, label_b = _segment(tmp_path, "b1", 110, 2)
    out = tmp_path / "calibration.json"
    assert threshold.main(
        ["calibrate", "--cell", "A", pred_a, label_a, "--cell", "B", pred_b, label_b, "--out", str(out)]
    ) == 0
    assert threshold.main(["apply", pred_a, "--calibration", str(out), "--out", str(tmp_path / "m.tif")]) == 0
    assert threshold.main(
        ["apply", pred_a, "--calibration", str(out), "--rule", "quantile", "--out", str(tmp_path / "q.tif")]
    ) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["threshold"] == 100
