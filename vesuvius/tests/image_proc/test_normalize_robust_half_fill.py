import numpy as np

from vesuvius.image_proc.intensity.normalization import normalize_robust


def test_half_fill_window_stays_bounded():
    rng = np.random.default_rng(0)
    window = rng.integers(50, 60000, size=(17, 128, 128)).astype(np.float32) / 65280.0
    window[:, :, :64] = 0.0  # left half is fill, the MAD is a few grey levels
    for last_value in (None, 50 / 65280.0):  # exactly half fill, and one fill voxel less
        if last_value is not None:
            window[0, 0, 0] = last_value
        out = normalize_robust(window, percentile_lower=1.0, percentile_upper=99.0)
        assert np.isfinite(out).all() and np.abs(out).max() < 20


def test_ordinary_window_is_unchanged():
    x = np.random.default_rng(1).random((8, 32, 32)).astype(np.float32)
    c = np.clip(x, *np.percentile(x, [1.0, 99.0]))
    expected = (c - np.median(c)) / (1.4826 * np.median(np.abs(c - np.median(c))))  # the 1e-6 rule
    assert np.allclose(normalize_robust(x, percentile_lower=1.0, percentile_upper=99.0), expected, atol=1e-4)
