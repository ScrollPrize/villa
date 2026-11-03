"""`peak_check`: whether a measured shift is where two blocks match best."""

import numpy as np
from scipy import ndimage

from volreg.cores import dense_correlation as dense

TRUE_SHIFT = np.array([1.3, -0.7, 0.4])


def textured(rng, shape=(dense.BLOCK,) * 3):
    return ndimage.gaussian_filter(rng.normal(size=shape), 2.0).astype(np.float32)


def pair(rng, coarse):
    """A fine block that `TRUE_SHIFT` brings onto `coarse`, under noise of its own."""
    fine = ndimage.shift(coarse, -TRUE_SHIFT, order=3, mode="nearest")
    return fine + 0.05 * float(coarse.std()) * rng.normal(size=coarse.shape).astype(np.float32)


def test_a_right_shift_is_its_own_optimum():
    rng = np.random.default_rng(0)
    coarse = textured(rng)
    fine = pair(rng, coarse)
    assert dense.aligned_match(coarse, fine, TRUE_SHIFT) > 0.9, "the convention under test"
    check = dense.peak_check(coarse, fine, TRUE_SHIFT)
    assert check["offset"] < 0.2
    assert check["peak_match"] > 0.9


def test_a_wrong_shift_is_found_out_and_the_right_one_named():
    rng = np.random.default_rng(1)
    coarse = textured(rng)
    fine = pair(rng, coarse)
    error = np.array([1.6, 0.0, -0.9])
    check = dense.peak_check(coarse, fine, TRUE_SHIFT + error)
    assert abs(check["offset"] - np.linalg.norm(error)) < 0.25
    assert np.abs(np.asarray(check["peak_zyx"]) - TRUE_SHIFT).max() < 0.25
    assert check["peak_match"] > dense.aligned_match(coarse, fine, TRUE_SHIFT + error)


def test_sheets_are_pinned_across_and_loose_along():
    rng = np.random.default_rng(2)
    # Layers stacked along z, each the same everywhere in y and x, plus faint texture.
    layers = ndimage.gaussian_filter1d(rng.normal(size=dense.BLOCK), 2.0).astype(np.float32)
    coarse = layers[:, None, None] * np.ones((1, dense.BLOCK, dense.BLOCK), np.float32)
    coarse = coarse + 0.02 * textured(rng) / float(textured(rng).std())
    fine = pair(rng, coarse)
    weakest, middle, strongest = dense.peak_check(coarse, fine, TRUE_SHIFT)["sharpness"]
    assert strongest > 10 * middle
    assert weakest <= middle


def test_octants_follow_a_block_that_did_not_move_as_a_whole():
    rng = np.random.default_rng(3)
    coarse = 40.0 * textured(rng) / float(textured(rng).std()) + 100.0
    lower, upper = TRUE_SHIFT, TRUE_SHIFT + np.array([0.0, 1.5, -1.0])
    fine = pair(rng, coarse)
    # The upper half in z moved further: a block deformed inside, not a bad block.
    half = dense.BLOCK // 2
    fine[half:] = ndimage.shift(coarse, -upper, order=3, mode="nearest")[half:]
    octants = dense.measure_octants(np.zeros(3), coarse, fine, np.eye(4), 1)
    assert len(octants) == 8
    for octant in octants:
        assert "rejected" not in octant
        expected = upper if octant["corner_zyx"][0] == half else lower
        # The phase correlation is biased towards zero on a half-size block by up to about
        # half a voxel here; the NCC optimum the check finds is not.
        assert np.abs(np.asarray(octant["shift_zyx"]) - expected).max() < 0.5
        assert np.abs(np.asarray(octant["check"]["peak_zyx"]) - expected).max() < 0.05
        # The point is where the octant is, and the fine point is that moved by its shift.
        assert all(corner <= at < corner + half
                   for corner, at in zip(octant["corner_zyx"], octant["centre_zyx"]))
        np.testing.assert_allclose(octant["fine_zyx"],
                                   np.asarray(octant["centre_zyx"]) + octant["shift_zyx"])
