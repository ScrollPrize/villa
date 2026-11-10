import itertools

import numpy as np
import pytest

import losses


@pytest.mark.parametrize('num_items', [2, 3, 4, 7, 16, 61])
def test_unrank_pairs_matches_itertools_order(num_items):
    expected = list(itertools.combinations(range(num_items), 2))
    first, second = losses._unrank_pairs(num_items, np.arange(len(expected)))
    assert list(zip(first.tolist(), second.tolist())) == expected


def test_small_sets_use_the_legacy_draws():
    np.random.seed(7)
    legacy = np.random.choice(10 * 9 // 2, 4, replace=False)
    expected = [list(itertools.combinations(range(10), 2))[r] for r in legacy]
    np.random.seed(7)
    assert losses._sample_patch_pairs(10, 4, adjacent_only=False) == expected


def test_adjacent_only_small_matches_legacy_zip():
    ids = list(range(9))
    np.random.seed(3)
    legacy = np.random.choice(8, 4, replace=False)
    expected = [list(zip(ids, ids[1:]))[r] for r in legacy]
    np.random.seed(3)
    assert losses._sample_patch_pairs(9, 4, adjacent_only=False if False else True) == expected


@pytest.mark.parametrize('adjacent_only', [False, True])
def test_large_sets_give_distinct_valid_pairs(adjacent_only):
    np.random.seed(11)
    num = 5000  # 12.5M unordered pairs: enumerating them is what made whole-scroll steps crawl
    pairs = losses._sample_patch_pairs(num, 4, adjacent_only=adjacent_only)
    assert len(pairs) == 4 and len(set(pairs)) == 4
    for a, b in pairs:
        assert 0 <= a < b < num
        if adjacent_only:
            assert b == a + 1


def test_pair_count_is_capped_by_the_candidate_count():
    assert sorted(losses._sample_patch_pairs(3, 10, adjacent_only=False)) == [(0, 1), (0, 2), (1, 2)]
    assert sorted(losses._sample_patch_pairs(3, 10, adjacent_only=True)) == [(0, 1), (1, 2)]
    assert losses._sample_patch_pairs(1, 4, adjacent_only=False) == []
    assert losses._sample_patch_pairs(0, 4, adjacent_only=True) == []


def test_large_path_is_uniform_over_pairs():
    np.random.seed(5)
    num = 100  # 4950 candidates > the enumerate limit, so the rejection + unrank path runs
    counts = np.zeros((num, num), dtype=np.int64)
    for _ in range(20000):
        for a, b in losses._sample_patch_pairs(num, 4, adjacent_only=False):
            counts[a, b] += 1
    upper = counts[np.triu_indices(num, 1)]
    assert counts[np.tril_indices(num)].sum() == 0
    expected = 20000 * 4 / len(upper)
    # chi-square against uniform: mean dof, sd ~ sqrt(2 dof); 5 sd keeps Poisson tails from flaking
    chi2 = ((upper - expected) ** 2 / expected).sum()
    dof = len(upper) - 1
    assert chi2 < dof + 5 * np.sqrt(2 * dof)
