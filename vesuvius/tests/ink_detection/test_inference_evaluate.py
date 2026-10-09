from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.inference.evaluate import (
    auroc_from_histograms,
    evaluate_segment,
    find_label_assets,
    main,
    read_label_plane,
    threshold_scores,
)


ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _write_label(path: Path, plane: np.ndarray, *, depth: int = 21, center: int | None = None) -> None:
    kwargs = {"mode": "w"}
    if ZARR_V3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    volume = np.zeros((depth, *plane.shape), dtype=np.uint8)
    volume[depth // 2 if center is None else center] = plane.astype(np.uint8) * 255
    if center is not None:
        group.attrs["annotation_center_channel"] = center
    if ZARR_V3:
        group.create_array("0", data=volume, chunks=(depth, 16, 16))
    else:
        group.create_dataset("0", data=volume, chunks=(depth, 16, 16))


def _segment(tmp_path: Path, *, validation: bool = True) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    shape = (64, 96)
    ink = rng.random(shape) < 0.2
    supervision = np.zeros(shape, bool)
    supervision[:, :48] = True
    folder = tmp_path / "pherc0000-w001"
    folder.mkdir()
    _write_label(folder / "pherc0000-w001_inklabels.zarr", ink)
    _write_label(folder / "pherc0000-w001_supervision_mask.zarr", supervision)
    if validation:
        region = np.zeros(shape, bool)
        region[:, 48:] = True
        _write_label(folder / "pherc0000-w001_validation_mask.zarr", region)
    return folder, ink, supervision


def _pairwise_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = scores[labels].astype(float), scores[~labels].astype(float)
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def test_histogram_auroc_matches_pairwise_count_with_ties():
    rng = np.random.default_rng(0)
    labels = rng.random(500) < 0.3
    scores = np.clip(rng.normal(100 + 40 * labels, 30), 0, 255).astype(np.uint8) // 8 * 8
    positive = np.bincount(scores[labels], minlength=256).astype(float)
    negative = np.bincount(scores[~labels], minlength=256).astype(float)
    assert auroc_from_histograms(positive, negative) == pytest.approx(_pairwise_auroc(scores, labels))


def test_threshold_scores_report_f1_at_128_and_the_best_cut():
    positive = np.zeros(256)
    negative = np.zeros(256)
    positive[100] = 10
    negative[50] = 10
    scores = threshold_scores(positive, negative)
    assert scores["f1_at_128"] == 0.0
    assert scores["best_f1"] == 1.0
    assert 51 <= scores["best_threshold"] <= 100


def test_label_plane_uses_annotation_center_attribute(tmp_path):
    plane = np.eye(8, dtype=bool)
    _write_label(tmp_path / "a_inklabels.zarr", plane, depth=21, center=3)
    _write_label(tmp_path / "b_inklabels.zarr", plane, depth=65)
    assert np.array_equal(read_label_plane(tmp_path / "a_inklabels.zarr"), plane)
    assert np.array_equal(read_label_plane(tmp_path / "b_inklabels.zarr"), plane)


def test_find_label_assets_picks_the_greatest_version(tmp_path):
    folder, _, _ = _segment(tmp_path)
    _write_label(folder / "pherc0000-w001_inklabels_v2.zarr", np.zeros((64, 96), bool))
    (folder / "other_inklabels.zarr").mkdir()
    assets = find_label_assets(folder)
    assert assets["inklabels"].name == "pherc0000-w001_inklabels_v2.zarr"
    assert assets["validation_mask"].name == "pherc0000-w001_validation_mask.zarr"


def test_validation_and_training_regions_are_scored_separately(tmp_path):
    folder, ink, supervision = _segment(tmp_path)
    rng = np.random.default_rng(2)
    good = np.where(ink, 200, 60).astype(np.uint8)
    noisy = np.clip(good + rng.normal(0, 60, ink.shape), 0, 255).astype(np.uint8)
    tifffile.imwrite(tmp_path / "good.tif", good)
    tifffile.imwrite(tmp_path / "noisy.tif", noisy)

    result = evaluate_segment(
        folder,
        {"noisy": tmp_path / "noisy.tif", "good": tmp_path / "good.tif"},
        bootstraps=200,
    )

    assert set(result["regions"]) == {"validation", "supervision_minus_validation"}
    validation = result["regions"]["validation"]
    assert validation["pixels"] == int((~supervision).sum())
    assert validation["predictions"]["good"]["auroc"] == 1.0
    assert validation["predictions"]["noisy"]["auroc"] == pytest.approx(
        _pairwise_auroc(noisy[:, 48:].ravel(), ink[:, 48:].ravel())
    )
    delta = validation["vs_reference"]["good"]
    assert delta["auroc_delta"] > 0
    assert delta["ci95"][0] > 0
    assert delta["fraction_not_better"] == 0.0


def test_identical_predictions_have_zero_bootstrap_difference(tmp_path):
    folder, ink, _ = _segment(tmp_path)
    tifffile.imwrite(tmp_path / "a.tif", np.where(ink, 150, 100).astype(np.uint8))
    result = evaluate_segment(
        folder, {"a": tmp_path / "a.tif", "b": tmp_path / "a.tif"}, bootstraps=50
    )
    for region in result["regions"].values():
        assert region["vs_reference"]["b"]["ci95"] == [0.0, 0.0]


def test_segment_without_validation_mask_scores_supervision_only(tmp_path, caplog):
    folder, ink, _ = _segment(tmp_path, validation=False)
    tifffile.imwrite(tmp_path / "a.tif", np.where(ink, 150, 100).astype(np.uint8))
    with caplog.at_level("WARNING"):
        result = evaluate_segment(folder, {"a": tmp_path / "a.tif"}, bootstraps=20)
    assert list(result["regions"]) == ["supervision"]
    assert "has seen those labels" in caplog.text


def test_prediction_on_another_grid_is_rejected(tmp_path):
    folder, _, _ = _segment(tmp_path)
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((64, 95), np.uint8))
    with pytest.raises(ValueError, match="run inference on the surface volume"):
        evaluate_segment(folder, {"a": tmp_path / "a.tif"})


def test_cli_prints_a_table_and_writes_json(tmp_path, capsys):
    folder, ink, _ = _segment(tmp_path)
    tifffile.imwrite(tmp_path / "s42.tif", np.where(ink, 150, 100).astype(np.uint8))
    tifffile.imwrite(tmp_path / "s43.tif", np.where(ink, 140, 110).astype(np.uint8))
    out = tmp_path / "metrics.json"
    assert main([
        str(folder), f"seed42={tmp_path / 's42.tif'}", str(tmp_path / "s43.tif"),
        "--reference", "s43", "--bootstraps", "20", "--json-out", str(out),
    ]) == 0
    printed = capsys.readouterr().out
    assert "validation:" in printed and "seed42 - s43" in printed
    assert json.loads(out.read_text())["reference"] == "s43"
