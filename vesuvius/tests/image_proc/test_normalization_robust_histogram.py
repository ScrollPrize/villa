"""The histogram fast path must be bit-identical to `normalize_robust`, including all delegation cases."""
from __future__ import annotations

import numpy as np
import pytest

from vesuvius.image_proc.intensity.normalization import (
    _histogram_robust_statistics,
    _median_weighted,
    _percentile_from_histogram,
    normalize_robust,
    normalize_robust_histogram,
)

FLAT_PATCH_SHAPE = (17, 128, 128)
_REFERENCE_SHAPE = (3, 48, 48)


def _bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        a.view(np.uint8), b.view(np.uint8)
    )


def _assert_fast_matches_reference(patch: np.ndarray) -> None:
    expected = normalize_robust(patch)
    actual = normalize_robust_histogram(patch)
    assert _bit_identical(actual, expected), (
        f"fast path differs from reference for dtype={patch.dtype} shape={patch.shape}; "
        f"max|diff|={np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max():.3e}"
    )


def _render_like_patch(shape, seed):
    """Render-like uint8 patch: mid-grey papyrus, a dark air band, bright specks."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=118.0, scale=11.0, size=shape)
    base[base < 0] = 0.0
    base[base > 255] = 255.0
    base[..., : shape[-1] // 8] = 0.0
    base[rng.random(shape) < 0.01] = 255.0
    return np.rint(base).astype(np.uint8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bit_identical_on_render_like_uint8(seed):
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed).astype(np.float32)
    _assert_fast_matches_reference(patch)


@pytest.mark.parametrize(
    "shape",
    [(17, 128, 128), (17, 128, 129), (1, 1, 1), (2, 3, 5), (5, 64, 64), (34, 33, 127)],
)
def test_bit_identical_across_shapes(shape):
    _assert_fast_matches_reference(_render_like_patch(shape, seed=abs(hash(shape)) % 1000).astype(np.float32))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.uint16, np.int16])
def test_bit_identical_across_dtypes(dtype):
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=7)
    _assert_fast_matches_reference(patch.astype(dtype))


@pytest.mark.parametrize(
    "name,make",
    [
        ("all zero", lambda s: np.zeros(s, np.float32)),
        ("constant 100", lambda s: np.full(s, 100.0, np.float32)),
        ("two-level 30/31", lambda s: (30 + (np.indices(s).sum(0) % 2)).astype(np.float32)),
        ("checker 0/200", lambda s: (np.indices(s).sum(0) % 2 * 200).astype(np.float32)),
        ("ramp mod 256", lambda s: (np.indices(s).sum(0) % 256).astype(np.float32)),
        ("single bright voxel", lambda s: np.pad(np.full((1, 1, 1), 255.0, np.float32),
                                                 [(0, s[0] - 1), (0, s[1] - 1), (0, s[2] - 1)])),
        ("all 255", lambda s: np.full(s, 255.0, np.float32)),
    ],
)
def test_bit_identical_on_edge_cases(name, make):
    _assert_fast_matches_reference(make(_REFERENCE_SHAPE))


def test_input_is_not_mutated():
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=11).astype(np.float32)
    before = patch.copy()
    normalize_robust_histogram(patch)
    assert np.array_equal(patch, before)


def test_output_is_c_contiguous_float32():
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=12).astype(np.float32)
    out = normalize_robust_histogram(patch)
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]


def test_non_contiguous_input_matches_reference():
    patch = _render_like_patch((34, 128, 128), seed=13).astype(np.float32)[::2]
    _assert_fast_matches_reference(patch)


def test_delegates_for_non_integer_values():
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=14).astype(np.float32) + 0.25
    _assert_fast_matches_reference(patch)


def test_delegates_for_out_of_range_values():
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=15).astype(np.float32) * 4.0
    _assert_fast_matches_reference(patch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentile_lower": 0.5},
        {"percentile_upper": 99.5},
        {"clip_values": False},
        {"target_dtype": np.float64},
    ],
)
def test_delegates_for_non_default_arguments(kwargs):
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=16).astype(np.float32)
    assert _bit_identical(
        normalize_robust_histogram(patch, **kwargs),
        normalize_robust(patch, **kwargs),
    )


def test_delegates_when_masked():
    patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=17).astype(np.float32)
    mask = np.ones(FLAT_PATCH_SHAPE, bool)
    mask[..., :20] = False
    assert _bit_identical(
        normalize_robust_histogram(patch, mask=mask, use_mask=True),
        normalize_robust(patch, mask=mask, use_mask=True),
    )


def test_normalize_flat_patch_is_unchanged():
    infer = pytest.importorskip("vesuvius.ink_detection.inference.infer")
    for seed in (0, 1, 2):
        patch = _render_like_patch(FLAT_PATCH_SHAPE, seed=seed).astype(np.float32)
        assert _bit_identical(
            infer.normalize_flat_patch(patch, "tifxyz_robust"),
            np.ascontiguousarray(normalize_robust(patch)),
        )


def test_percentile_matches_numpy_on_random_multisets():
    rng = np.random.default_rng(21)
    for _ in range(150):
        values = rng.integers(0, 256, size=int(rng.integers(1, 60000))).astype(np.float32)
        histogram = np.bincount(values.astype(np.int64), minlength=256)
        for percentile in (1.0, 99.0):
            got = _percentile_from_histogram(histogram, percentile)
            expected = np.float32(np.percentile(values, percentile))
            assert np.float32(got).tobytes() == expected.tobytes(), (
                f"percentile {percentile} disagreed on n={values.size}: {got!r} vs {expected!r}"
            )


def test_median_matches_numpy_on_random_multisets():
    rng = np.random.default_rng(22)
    for _ in range(150):
        values = rng.integers(0, 256, size=int(rng.integers(1, 400))).astype(np.float32)
        counts = np.bincount(values.astype(np.int64), minlength=256).astype(np.float64)
        present = np.nonzero(counts)[0]
        got = _median_weighted(present.astype(np.float64), counts[present])
        expected = np.float32(np.median(values))
        assert np.float32(got).tobytes() == expected.tobytes(), (
            f"median disagreed on n={values.size}: {got!r} vs {expected!r}"
        )


def test_percentile_between_two_different_order_statistics():
    """Both percentiles land between two different levels, so the interpolation weight matters."""
    n = FLAT_PATCH_SHAPE[0] * FLAT_PATCH_SHAPE[1] * FLAT_PATCH_SHAPE[2]
    blocks = [(40, 2786), (50, 136214), (51, 136742), (52, 2786)]
    assert sum(count for _, count in blocks) == n
    flat = np.concatenate([np.full(count, level, np.float32) for level, count in blocks])
    patch = flat.reshape(FLAT_PATCH_SHAPE)

    assert float(np.percentile(patch, 1.0)) == pytest.approx(40.0 + 0.27 * (50.0 - 40.0), rel=1e-6)
    assert float(np.percentile(patch, 99.0)) == pytest.approx(51.0 + 0.73 * (52.0 - 51.0), rel=1e-6)
    _, _, _, _, degenerate = _histogram_robust_statistics(np.rint(patch).astype(np.int64).ravel())
    assert degenerate is False
    _assert_fast_matches_reference(patch)


def _adversarial_patches():
    rng = np.random.default_rng(20261027)
    yield "render-like float32", np.rint(rng.normal(118.0, 11.0, (17, 128, 128))).clip(0, 255).astype(np.float32)
    yield "three levels", rng.choice(np.array([0.0, 3.0, 255.0], np.float32), size=(24, 24))
    yield "constant + one spike", np.concatenate([np.full(899, 7.0, np.float32), [250.0]]).reshape(30, 30)
    yield "two levels (MAD collapses)", rng.choice(np.array([10.0, 20.0], np.float32), size=(16, 16))
    yield "0/255 only", rng.choice(np.array([0.0, 255.0], np.float32), size=(20, 20))
    yield "uint8 dtype", rng.integers(0, 256, (40, 40)).astype(np.uint8)


@pytest.mark.parametrize("name,patch", list(_adversarial_patches()))
def test_adversarial_patches_are_bit_identical(name: str, patch: np.ndarray) -> None:
    _assert_fast_matches_reference(patch)


def test_delegation_kwargs_are_bit_identical() -> None:
    patch = np.rint(np.random.default_rng(7).normal(118.0, 11.0, (17, 64, 64))).clip(0, 255).astype(np.float32)
    for kwargs in (
        dict(percentile_lower=0.5, percentile_upper=99.5),
        dict(clip_values=False),
        dict(target_dtype=np.float32),
    ):
        actual = normalize_robust_histogram(patch, **kwargs)
        expected = normalize_robust(patch, **kwargs)
        assert _bit_identical(actual, expected), kwargs
