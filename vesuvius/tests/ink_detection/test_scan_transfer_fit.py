"""The transfer-kernel fit recovers a known blur from aligned synthetic stacks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from vesuvius.ink_detection.scan_transfer_fit import fit_kernel


def test_fit_recovers_a_known_kernel():
    rng = np.random.default_rng(1)
    pooled = ndimage.gaussian_filter(rng.standard_normal((21, 160, 160)), (0, 1.0, 1.0)).astype(np.float32)
    truth = np.zeros((3, 5, 5), np.float32)
    truth[0, 2, 2], truth[1, 2, 2], truth[2, 2, 2] = 0.25, 0.5, 0.25
    truth[1, 1:4, 1:4] += 0.05
    native = ndimage.correlate(pooled, truth, mode="nearest")
    native = native[3:24] if native.shape[0] > 21 else native
    mask = np.zeros(pooled.shape[1:], bool)
    mask[10:150, 10:150] = True
    fit = fit_kernel(pooled, native, mask, 0, kz=3, kxy=5, samples=20000)
    assert np.allclose(fit["kernel"], truth, atol=0.02)
    assert fit["heldout_r2"] > 0.99
