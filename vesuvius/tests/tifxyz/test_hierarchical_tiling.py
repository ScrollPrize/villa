"""Tests for _binary_search_min_size's correctness guarantee.

Covers a real bug: the function used an actual binary search, which
assumes test_fn is monotonic (once True, always True for larger inputs).
That assumption does not hold for its real use in find_valid_child_size:
whether a tile size produces enough "fully valid" children depends on
exact grid alignment against a segment's holes/torn edges, which is not
monotonic in tile size. On a realistic irregular mask, the previous
implementation's own short-circuit check on the largest size failed
(coverage had already dropped to zero there) and it returned 0 -- "no
valid size exists" -- while a genuinely valid, much smaller size was
available and never tested.
"""
import numpy as np
import pytest

from vesuvius.tifxyz.hierarchical_tiling import _binary_search_min_size


def test_monotonic_case_still_works():
    """No-regression check: an ordinary monotonic test_fn is still found
    correctly."""
    def test_fn(size):
        return size >= 50

    assert _binary_search_min_size(15, 89, test_fn) == 50


def test_genuinely_impossible_case_returns_zero():
    def always_false(size):
        return False

    assert _binary_search_min_size(15, 89, always_false) == 0


def test_trivial_range_checks_test_fn():
    """When lo >= hi, the single candidate must actually be tested, not
    assumed valid. Reproduces a smaller version of the same bug: the
    original code returned `hi` unconditionally in this branch, without
    ever calling test_fn."""
    def always_false(size):
        return False

    assert _binary_search_min_size(50, 50, always_false) == 0

    def always_true(size):
        return True

    assert _binary_search_min_size(50, 50, always_true) == 50


def test_small_valid_window_with_larger_invalid_region_is_not_missed():
    """A test_fn that is True only in a small window near `lo`, then False
    for the remainder of the range up to `hi` -- exactly the shape that
    defeats binary search's monotonicity assumption."""
    def test_fn(size):
        return 15 <= size <= 20

    assert _binary_search_min_size(15, 89, test_fn) == 15


def _count_fully_valid_children(valid_mask, parent_bbox, child_h, child_w):
    """Mirrors create_child_tiles' grid-placement and strict
    child_valid.all() filter -- the actual mechanism driving
    _test_child_size_validity's ratio in the real codebase."""
    r_min, r_max, c_min, c_max = parent_bbox
    total, valid = 0, 0
    for r0 in range(r_min, r_max - child_h + 2, child_h):
        for c0 in range(c_min, c_max - child_w + 2, child_w):
            r1 = min(r0 + child_h - 1, r_max)
            c1 = min(c0 + child_w - 1, c_max)
            if (r1 - r0 + 1) < child_h or (c1 - c0 + 1) < child_w:
                continue
            total += 1
            if valid_mask[r0:r1 + 1, c0:c1 + 1].all():
                valid += 1
    return total, valid


def test_realistic_irregular_mask_finds_true_minimum(tmp_path):
    """Direct reproduction against a realistic scenario: a region with
    scattered small holes, modeling a real segment's torn/incomplete
    edges. The true minimum valid tile size must be found, not missed in
    favour of reporting complete failure.
    """
    rng = np.random.default_rng(3)
    h, w = 200, 200
    mask = np.ones((h, w), dtype=bool)
    for _ in range(40):
        hr, hc = rng.integers(10, h - 10), rng.integers(10, w - 10)
        rad = rng.integers(2, 6)
        mask[max(0, hr - rad):hr + rad, max(0, hc - rad):hc + rad] = False

    parent_bbox = (0, h - 1, 0, w - 1)
    min_valid_fraction = 0.5

    def test_fn(size):
        total, valid = _count_fully_valid_children(mask, parent_bbox, size, size)
        return total > 0 and valid / total >= min_valid_fraction

    true_minimum = next(
        (s for s in range(15, 90) if test_fn(s)), None
    )
    assert true_minimum is not None, "test setup should have a valid size"

    result = _binary_search_min_size(15, 89, test_fn)
    assert result == true_minimum, (
        f"expected the true minimum valid size ({true_minimum}), got {result} -- "
        f"a non-monotonic test_fn must not cause the search to report failure "
        f"or return a size larger than the true minimum"
    )
