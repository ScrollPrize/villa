"""Safety checks for the offline material-mask experiment."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.scripts.compare_ct_background import (
    coarse_mean, display, make_mask, material_stats,
)


BACKGROUND = {'center': 50., 'noise': 4.}


def test_coarse_evidence_caps_outliers_without_mutating_input():
    raw = np.full((4, 4, 4), 50, np.uint8)
    raw[0, 0, 0] = 255
    before = raw.copy()
    coarse = coarse_mean(raw, 82.)
    assert coarse[0, 0, 0] == 54.
    assert coarse[-1, -1, -1] == 50.
    np.testing.assert_array_equal(raw, before)


def test_spatial_support_rejects_scattered_spikes_and_preserves_material_core():
    rng = np.random.default_rng(92)
    raw = np.clip(rng.normal(50, 4, (48, 48, 48)), 0, 255).astype(np.uint8)
    raw.ravel()[rng.choice(raw.size, raw.size//100, replace=False)] = 255
    assert make_mask(raw, BACKGROUND, 'pixel').any()
    assert not make_mask(raw, BACKGROUND, 'spatial1').any()
    raw[12:36, 12:36, 12:36] = 90
    strict = make_mask(raw, BACKGROUND, 'spatial2')
    loose = make_mask(raw, BACKGROUND, 'spatial1')
    assert strict[16:32, 16:32, 16:32].all()
    assert np.all(~strict | loose)
    assert not loose[:4].any()


def test_empty_and_constant_crops_produce_finite_foreground_only_display():
    for value in (0, 50, 90):
        raw = np.full((24, 24, 24), value, np.uint8)
        mask = make_mask(raw, BACKGROUND, 'spatial1')
        stats = material_stats(raw, mask, BACKGROUND)
        result = display(raw, mask, stats)
        assert np.isfinite(result).all()
        assert np.all(result[~mask] == 0)
        assert bool(mask.all()) == (value == 90)


def test_background_pixels_do_not_change_foreground_statistics():
    raw = np.full((24, 24, 24), 50, np.uint8)
    mask = np.zeros(raw.shape, bool)
    mask[8:16, 8:16, 8:16] = True
    raw[mask] = 100
    before = material_stats(raw, mask, BACKGROUND)
    raw[~mask] = 255
    assert material_stats(raw, mask, BACKGROUND) == before
