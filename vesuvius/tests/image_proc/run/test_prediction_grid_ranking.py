"""Tests for _collect_aggregated_model_stats' cross-model fairness guarantee.

Covers a real bug: means were previously computed per model over however
many images that model happened to have a stats.json for, with no
guarantee different models were compared on the same images. A model
evaluated on only an easier partial subset of the test set (a realistic
situation -- a crashed/partial eval run, or evaluated before harder images
were added to the test set) could show artificially better numbers than a
model honestly evaluated on the full, harder set, with nothing in the
tool's output revealing the comparison wasn't fair.
"""
import json

import pytest

from vesuvius.image_proc.run.prediction_grid import (
    _choose_best_model,
    _collect_aggregated_model_stats,
)


def _write_stats(model_dir, stem, *, cc, bp, hd, pr, rc):
    d = model_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.json").write_text(json.dumps({
        "connected_components_difference_class_1": cc,
        "branch_points_absdiff_class_1": bp,
        "hausdorff_distance_95_class_1": hd,
        "precision_class_1": pr,
        "recall_class_1": rc,
    }))


def _make_images(images_dir, n):
    images_dir.mkdir(exist_ok=True)
    for i in range(n):
        (images_dir / f"img{i:02d}.tif").touch()


def test_equal_coverage_is_unaffected(tmp_path):
    """No-regression check: when every model has stats for every image,
    behaviour is unchanged from a plain per-model mean."""
    eval_root = tmp_path / "eval"
    images_dir = tmp_path / "images"
    eval_root.mkdir()
    _make_images(images_dir, 10)

    model_a = eval_root / "model_A"
    model_b = eval_root / "model_B"
    for i in range(10):
        _write_stats(model_a, f"img{i:02d}", cc=1, bp=1, hd=2, pr=0.95, rc=0.95)
        _write_stats(model_b, f"img{i:02d}", cc=5, bp=5, hd=10, pr=0.70, rc=0.65)

    models_stats, num_images = _collect_aggregated_model_stats(eval_root, images_dir)
    assert num_images == 10

    means = dict(models_stats)
    assert means["model_A"]["connected_components_difference_class_1"] == pytest.approx(1.0)
    assert means["model_B"]["connected_components_difference_class_1"] == pytest.approx(5.0)
    assert _choose_best_model(models_stats) == "model_A"


def test_partial_coverage_does_not_unfairly_favor_the_less_evaluated_model(tmp_path, capsys):
    """Reproduces the real bug directly: model_A is only evaluated on the
    easier half of the test set (all great scores); model_B is evaluated
    on the full set, matching model_A exactly on the easy half and scoring
    worse (but reasonably) on the harder half it actually attempted.

    Both models' means must be computed over the SAME (shared) image set,
    so a model's apparent superiority can never come purely from having
    skipped the harder images -- on the shared 10-image subset here, both
    models are in fact numerically identical, so neither has genuine cause
    to be preferred and both should therefore win or lose only via a
    stable tie-break, never via unequal coverage.
    """
    eval_root = tmp_path / "eval"
    images_dir = tmp_path / "images"
    eval_root.mkdir()
    _make_images(images_dir, 20)

    model_a = eval_root / "model_A_partial_eval"
    model_b = eval_root / "model_B_full_eval"
    for i in range(10):
        _write_stats(model_a, f"img{i:02d}", cc=1, bp=1, hd=2, pr=0.95, rc=0.95)
        _write_stats(model_b, f"img{i:02d}", cc=1, bp=1, hd=2, pr=0.95, rc=0.95)
    for i in range(10, 20):
        # model_A never attempted these; model_B did, and did fine but not
        # perfectly. Under the old per-model-coverage aggregation, this drags
        # model_B's mean down while model_A's stays at its easy-subset value.
        _write_stats(model_b, f"img{i:02d}", cc=3, bp=3, hd=6, pr=0.85, rc=0.85)

    models_stats, num_images = _collect_aggregated_model_stats(eval_root, images_dir)
    means = dict(models_stats)

    assert num_images == 10, "must be restricted to the 10 images both models share"
    assert means["model_A_partial_eval"] == means["model_B_full_eval"], (
        "on the shared 10 images the two models are numerically identical -- "
        "model_A's mean must not differ just because it has no data at all "
        "for the other 10 images"
    )

    captured = capsys.readouterr()
    assert "model_B_full_eval" in captured.out
    assert "unequal stats coverage" in captured.out.lower() or "unfair" in captured.out.lower()


def test_genuinely_better_full_coverage_model_still_wins(tmp_path):
    """If the fully-evaluated model is ALSO genuinely better on the shared
    subset (not just equal), it must still win -- this fix restricts the
    comparison to fair ground, it must not penalize full evaluation."""
    eval_root = tmp_path / "eval"
    images_dir = tmp_path / "images"
    eval_root.mkdir()
    _make_images(images_dir, 20)

    model_a = eval_root / "model_A_partial_eval"
    model_b = eval_root / "model_B_full_eval"
    for i in range(10):
        _write_stats(model_a, f"img{i:02d}", cc=5, bp=5, hd=10, pr=0.70, rc=0.65)
        _write_stats(model_b, f"img{i:02d}", cc=1, bp=1, hd=2, pr=0.95, rc=0.95)
    for i in range(10, 20):
        _write_stats(model_b, f"img{i:02d}", cc=3, bp=3, hd=6, pr=0.85, rc=0.85)

    models_stats, _ = _collect_aggregated_model_stats(eval_root, images_dir)
    assert _choose_best_model(models_stats) == "model_B_full_eval"


def test_no_shared_images_leaves_metric_absent_for_all_models(tmp_path):
    """If two models share zero images for a metric, no model can be fairly
    scored on it -- it must be dropped for everyone rather than silently
    falling back to per-model averages."""
    eval_root = tmp_path / "eval"
    images_dir = tmp_path / "images"
    eval_root.mkdir()
    _make_images(images_dir, 4)

    model_a = eval_root / "model_A"
    model_b = eval_root / "model_B"
    _write_stats(model_a, "img00", cc=1, bp=1, hd=2, pr=0.9, rc=0.9)
    _write_stats(model_a, "img01", cc=1, bp=1, hd=2, pr=0.9, rc=0.9)
    _write_stats(model_b, "img02", cc=1, bp=1, hd=2, pr=0.9, rc=0.9)
    _write_stats(model_b, "img03", cc=1, bp=1, hd=2, pr=0.9, rc=0.9)

    models_stats, num_images = _collect_aggregated_model_stats(eval_root, images_dir)
    means = dict(models_stats)
    assert num_images == 0
    assert means["model_A"] == {}
    assert means["model_B"] == {}
