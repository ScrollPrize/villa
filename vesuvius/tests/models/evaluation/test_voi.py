"""Exact-value tests for the VOI metric.

The fixtures are one-voxel-wide bars in a 3x3x9 volume, small enough that the
conditional entropies can be computed by hand. skimage's
variation_of_information(gt, pred) returns (H(PR | GT), H(GT | PR)) in bits,
so a prediction that splits a ground-truth component must charge voi_split
only, and a prediction that merges two ground-truth components must charge
voi_merge only. These tests pin both directions to closed-form values.
"""
import math

import numpy as np
import pytest
import torch

from vesuvius.models.evaluation.voi import VOIMetric, compute_voi

# A 9-voxel bar scored against the same bar broken into two 4-voxel pieces
# yields labels with counts {4, 4, 1} over the 9-voxel union (the gap voxel is
# background on the broken side). The conditional entropy of that partition is
#   H = -(4/9)*log2(4/9) - (4/9)*log2(4/9) - (1/9)*log2(1/9) = log2(9) - 16/9
_BAR_ENTROPY = math.log2(9.0) - 16.0 / 9.0

_FULL_BAR = (slice(0, 9),)  # one 26-connected component of 9 voxels
_BROKEN_BAR = (slice(0, 4), slice(5, 9))  # two components of 4, gap at z=4


def _volume(spans):
    """Return a (1, 1, 3, 3, 9) float tensor with a bar at [1, 1, span]."""
    vol = np.zeros((3, 3, 9), dtype=np.float32)
    for span in spans:
        vol[1, 1, span] = 1.0
    return torch.from_numpy(vol)[None, None]


def test_identical_labellings_are_zero():
    """Perfect prediction: both conditional entropies are exactly zero."""
    gt = _volume(_BROKEN_BAR)
    result = compute_voi(pred=gt.clone(), gt=gt)
    assert result["voi_split"] == 0.0
    assert result["voi_merge"] == 0.0
    assert result["voi_total"] == 0.0
    assert result["voi_score"] == 1.0


def test_split_prediction_charges_only_voi_split():
    """Breaking one GT component in two is over-segmentation: H(PR | GT)."""
    gt = _volume(_FULL_BAR)
    pred = _volume(_BROKEN_BAR)
    result = compute_voi(pred=pred, gt=gt)
    assert result["voi_split"] == pytest.approx(_BAR_ENTROPY, abs=1e-12)
    assert result["voi_merge"] == 0.0
    assert result["voi_total"] == pytest.approx(_BAR_ENTROPY, abs=1e-12)
    assert result["voi_score"] == pytest.approx(1.0 / (1.0 + _BAR_ENTROPY), abs=1e-12)


def test_merge_prediction_charges_only_voi_merge():
    """Fusing two GT components is under-segmentation: H(GT | PR)."""
    gt = _volume(_BROKEN_BAR)
    pred = _volume(_FULL_BAR)
    result = compute_voi(pred=pred, gt=gt)
    assert result["voi_split"] == 0.0
    assert result["voi_merge"] == pytest.approx(_BAR_ENTROPY, abs=1e-12)
    assert result["voi_total"] == pytest.approx(_BAR_ENTROPY, abs=1e-12)
    assert result["voi_score"] == pytest.approx(1.0 / (1.0 + _BAR_ENTROPY), abs=1e-12)


def test_voi_metric_class_matches_compute_voi():
    """VOIMetric.compute goes through the same path as compute_voi."""
    gt = _volume(_BROKEN_BAR)
    pred = _volume(_FULL_BAR)
    metric = VOIMetric()
    assert metric.compute(pred, gt) == compute_voi(pred=pred, gt=gt)
