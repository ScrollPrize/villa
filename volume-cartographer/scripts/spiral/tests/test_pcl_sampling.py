import numpy as np
import pytest

import losses


def _configure(monkeypatch, *, stratified=True, weights=None):
    monkeypatch.setattr(losses, 'cfg', {
        'stratified_pcl_sampling': stratified,
        'pcl_sampling_weights': weights,
    })


def test_legacy_stratification_draws_equally_from_each_group(monkeypatch):
    _configure(monkeypatch)
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b', 'c', 'c'])
    np.random.seed(1)
    chosen = losses._choose_pcl_indices(strata, 3)
    assert sorted(index // 2 for index in chosen) == [0, 1, 2]


def test_explicit_weights_take_precedence_and_can_disable_groups(monkeypatch):
    _configure(monkeypatch, stratified=False, weights={'a': 0, 'b': 1})
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b'])
    assert strata['groups'] == ['b']
    assert np.array_equal(strata['all'], np.array([2, 3]))


def test_explicit_weights_require_every_group(monkeypatch):
    _configure(monkeypatch, weights={'a': 1})
    with pytest.raises(KeyError, match='sampling group.*b'):
        losses.build_pcl_sampling_strata(['a', 'b'])
