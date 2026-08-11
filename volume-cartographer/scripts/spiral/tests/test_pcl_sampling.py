import numpy as np
import pytest

import losses


def _cfg(*, stratified=True, weights=None):
    return {
        'pcl_stratified_pcl_sampling': stratified,
        'pcl_sampling_weights': weights,
    }


def test_legacy_stratification_draws_equally_from_each_group():
    cfg = _cfg()
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b', 'c', 'c'], cfg)
    np.random.seed(1)
    chosen = losses._choose_pcl_indices(strata, 3, cfg)
    assert sorted(index // 2 for index in chosen) == [0, 1, 2]


def test_explicit_weights_take_precedence_and_can_disable_groups():
    cfg = _cfg(stratified=False, weights={'a': 0, 'b': 1})
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b'], cfg)
    assert strata['groups'] == ['b']
    assert np.array_equal(strata['all'], np.array([2, 3]))


def test_explicit_weights_require_every_group():
    cfg = _cfg(weights={'a': 1})
    with pytest.raises(KeyError, match='sampling group.*b'):
        losses.build_pcl_sampling_strata(['a', 'b'], cfg)
