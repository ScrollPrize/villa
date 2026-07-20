"""Tests for the gaussian blend normalization (#1173 review).

The blend step accumulates sum(logit * w) and sum(w) per voxel, then
normalizes. Dividing by (sum(w) + epsilon) instead of sum(w) pulled
logits toward 0 wherever the accumulated weight was not far above the
epsilon: at a voxel covered by a single outer-corner Gaussian
contribution (weight ~exp(-24)), a saturated -20 logit came out as
~-0.075 (sigmoid ~0.48), above low probability thresholds - and genuine
model logits were flattened the same way. The normalization must
preserve the weighted average at every covered voxel and leave
uncovered voxels untouched; the `where=weights > 0` guard already
protects the division, so no epsilon is needed.
"""

import math

import numpy as np

from vesuvius.models.run.blending import generate_gaussian_map, normalize_blended_logits

# Probability threshold used by the affected surface-prediction runs;
# finalize_outputs converts it to a logit cutoff: sigmoid(x) > T <=> x > log(T/(1-T)).
THRESHOLD = 0.2
LOGIT_CUTOFF = math.log(THRESHOLD / (1.0 - THRESHOLD))

LEGACY_EPSILON = 1e-8
SATURATION = 20.0


def _corner_weight():
    gmap = generate_gaussian_map((128, 128, 128))[0]
    return float(gmap[0, 0, 0])


def test_corner_weight_is_in_the_legacy_pathological_regime():
    """Documents the #1173 review finding against the legacy formula."""
    w = _corner_weight()
    # The outer corner weighs ~exp(-24), far below the legacy epsilon.
    assert 0.0 < w < LEGACY_EPSILON

    # sum(logit*w) / (sum(w) + eps) diluted -20 above the threshold cutoff,
    # i.e. a masked-background voxel read as foreground at threshold 0.2.
    diluted = (-SATURATION * w) / (w + LEGACY_EPSILON)
    assert diluted > LOGIT_CUTOFF
    assert 0.4 < 1.0 / (1.0 + math.exp(-diluted)) < 0.5

    # Two-channel convention: +20/-20 diluted to ~+0.075/-0.075, so the
    # foreground softmax read ~0.462 - also above the 0.2 threshold.
    fg = (-SATURATION * w) / (w + LEGACY_EPSILON)
    bg = (+SATURATION * w) / (w + LEGACY_EPSILON)
    assert (fg - bg) > LOGIT_CUTOFF
    assert 0.4 < 1.0 / (1.0 + math.exp(-(fg - bg))) < 0.5


def test_normalization_preserves_saturation_at_corner_weight():
    w = _corner_weight()
    logits = np.full((1, 1, 1, 1), -SATURATION * w, dtype=np.float32)
    weights = np.full((1, 1, 1), w, dtype=np.float32)

    normalize_blended_logits(logits, weights)

    assert np.isclose(logits[0, 0, 0, 0], -SATURATION, rtol=1e-5)
    assert logits[0, 0, 0, 0] < LOGIT_CUTOFF


def test_normalization_preserves_softmax_pair_at_corner_weight():
    """The two-channel +20/-20 background/foreground pair also survives at the
    corner weight: the foreground softmax stays decisively below threshold."""
    w = _corner_weight()
    logits = np.zeros((2, 1, 1, 1), dtype=np.float32)
    logits[0, 0, 0, 0] = +SATURATION * w
    logits[1, 0, 0, 0] = -SATURATION * w
    weights = np.full((1, 1, 1), w, dtype=np.float32)

    normalize_blended_logits(logits, weights)

    assert np.isclose(logits[0, 0, 0, 0], +SATURATION, rtol=1e-5)
    assert np.isclose(logits[1, 0, 0, 0], -SATURATION, rtol=1e-5)
    # p_fg > T <=> logit_fg - logit_bg > cutoff; -40 is decisively below.
    assert (logits[1, 0, 0, 0] - logits[0, 0, 0, 0]) < LOGIT_CUTOFF


def test_normalization_is_exact_for_multiple_contributions():
    """A weighted average of identical logits is that logit, at any weight."""
    w = _corner_weight()
    total_w = np.float32(w) + np.float32(2.0 * w) + np.float32(0.5 * w)
    logits = np.full((2, 1, 1, 1), -SATURATION * total_w, dtype=np.float32)
    weights = np.full((1, 1, 1), total_w, dtype=np.float32)

    normalize_blended_logits(logits, weights)

    assert np.allclose(logits, -SATURATION, rtol=1e-5)


def test_normalization_leaves_uncovered_voxels_untouched():
    logits = np.zeros((2, 2, 2, 2), dtype=np.float32)
    weights = np.zeros((2, 2, 2), dtype=np.float32)
    weights[0, 0, 0] = 0.5
    logits[:, 0, 0, 0] = 3.0 * 0.5

    normalize_blended_logits(logits, weights)

    # The covered voxel is the weighted average; uncovered voxels stay 0.
    assert np.allclose(logits[:, 0, 0, 0], 3.0)
    uncovered = np.ones((2, 2, 2), dtype=bool)
    uncovered[0, 0, 0] = False
    assert (logits[:, uncovered] == 0.0).all()


def test_normalization_matches_legacy_for_typical_weights():
    """For ordinary interior weights the epsilon removal is a ~1e-8
    relative change: results agree with the legacy formula to float32
    precision."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.5, 4.0, size=(3, 3, 3)).astype(np.float32)
    raw = rng.normal(0.0, 5.0, size=(2, 3, 3, 3)).astype(np.float32)
    logits = raw * weights[np.newaxis]
    legacy = logits / (weights[np.newaxis] + LEGACY_EPSILON)

    normalize_blended_logits(logits, weights)

    assert np.allclose(logits, raw, rtol=1e-5)
    assert np.allclose(logits, legacy, rtol=1e-5)
