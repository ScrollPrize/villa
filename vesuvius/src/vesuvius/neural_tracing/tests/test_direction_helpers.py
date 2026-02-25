import numpy as np

from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_triplet_direction_priors_from_conditioning_surface,
)


def test_build_triplet_direction_priors_from_conditioning_surface_cond_mask():
    crop_size = (3, 3, 3)
    cond_mask = np.zeros(crop_size, dtype=bool)
    cond_mask[1, 1, 1] = True
    cond_mask[1, 2, 1] = True

    surface = np.zeros((3, 3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            surface[r, c] = np.array([1.0, float(r), float(c)], dtype=np.float32)

    priors = build_triplet_direction_priors_from_conditioning_surface(
        crop_size=crop_size,
        cond_mask=cond_mask,
        cond_surface_local=surface,
        mask_mode="cond",
    )
    assert priors is not None
    assert priors.shape == (6, *crop_size)

    assert np.allclose(priors[0:3, 1, 1, 1], -priors[3:6, 1, 1, 1], atol=1e-6, rtol=0.0)
    assert np.isclose(np.linalg.norm(priors[0:3, 1, 1, 1]), 1.0, atol=1e-6, rtol=0.0)
    assert np.allclose(priors[:, 0, 0, 0], 0.0, atol=1e-6, rtol=0.0)


def test_build_triplet_direction_priors_from_conditioning_surface_degenerate_returns_none():
    crop_size = (2, 2, 2)
    cond_mask = np.ones(crop_size, dtype=bool)
    surface = np.zeros((2, 2, 3), dtype=np.float32)

    priors = build_triplet_direction_priors_from_conditioning_surface(
        crop_size=crop_size,
        cond_mask=cond_mask,
        cond_surface_local=surface,
        mask_mode="cond",
    )
    assert priors is None
