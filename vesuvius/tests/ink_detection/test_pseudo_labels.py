"""Confidence-thresholded pseudo-label rules for flat self-training."""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.ink_detection.pseudo_labels import pseudo_label_masks


def _field(shape=(64, 64)):
    papyrus = np.ones(shape, bool)
    mean = np.full(shape, 0.26, np.float32)          # confident background everywhere ...
    mean[20:30, 20:30] = 0.60                         # ... one confident ink block
    mean[40:44, 40:44] = 0.42                         # ... one weak ink candidate
    top = mean.copy()
    return mean, top, papyrus


def test_mean_rule_labels_confident_pixels_and_ignores_the_middle():
    mean, top, papyrus = _field()
    ink, sup = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=0)
    assert ink[25, 25] and sup[25, 25]
    assert not ink[5, 5] and sup[5, 5]
    assert not sup[42, 42]                            # 0.42 is between lo and hi: ignored


def test_max_rule_keeps_a_margin_around_weak_candidates():
    mean, top, papyrus = _field()
    _, sup_mean = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=0)
    ink, sup = pseudo_label_masks(mean, top, papyrus, background="max", margin=8, erode=0)
    assert sup_mean[40, 36] and not sup[40, 36]       # 4 px from the weak candidate: no longer background
    assert sup[5, 5]                                  # far away: still background
    assert ink[25, 25]


def test_max_rule_drops_background_where_any_teacher_disagrees():
    mean, top, papyrus = _field()
    top[5, 5] = 0.35                                  # one teacher above lo
    _, sup = pseudo_label_masks(mean, top, papyrus, background="max", margin=0, erode=0)
    assert not sup[5, 5]


def test_outside_papyrus_and_eroded_edges_are_ignored():
    mean, top, papyrus = _field()
    papyrus[:, :10] = False
    _, sup = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=4)
    assert not sup[:, :14].any()
    assert sup[5, 30]


def test_rejects_bad_thresholds_and_rules():
    mean, top, papyrus = _field()
    with pytest.raises(ValueError):
        pseudo_label_masks(mean, top, papyrus, hi=0.3, lo=0.5)
    with pytest.raises(ValueError):
        pseudo_label_masks(mean, top, papyrus, background="median")
