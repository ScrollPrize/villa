import numpy as np

from losses import _random_patch_pairs


def test_random_patch_pairs_are_distinct_ordered_pairs():
    np.random.seed(0)
    ids = [f'p{i}' for i in range(5000)]
    pairs = _random_patch_pairs(ids, 64)
    assert len(set(pairs)) == 64
    assert all(ids.index(a) < ids.index(b) for a, b in pairs)
    # Asking for more than exist returns every pair once.
    assert sorted(_random_patch_pairs(ids[:4], 99)) == [
        ('p0', 'p1'), ('p0', 'p2'), ('p0', 'p3'),
        ('p1', 'p2'), ('p1', 'p3'), ('p2', 'p3')]
    assert _random_patch_pairs(ids[:1], 8) == []
