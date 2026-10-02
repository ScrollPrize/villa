from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser
from vesuvius.neural_tracing.fiber_follow.shared.collect import fiber_visit_order


def dataset(lengths, **kwargs):
    return FollowDataset([SimpleNamespace(length=float(v)) for v in lengths], None, None, None, **kwargs)


def test_default_power_keeps_uniform_arclength_weights():
    lengths = np.array([300., 375., 619., 1030., 2435.])
    np.testing.assert_array_equal(dataset(lengths).weights, lengths/lengths.sum())
    np.testing.assert_array_equal(dataset(lengths, length_power=1.).weights, lengths/lengths.sum())


def test_power_favors_long_fibers():
    lengths = np.array([300., 600., 1200.])
    weights = dataset(lengths, length_power=3.).weights
    np.testing.assert_allclose(weights, lengths**3/np.sum(lengths**3))
    assert weights[2]/weights[0] == pytest.approx(64.)


@pytest.mark.parametrize('power', [-1., float('nan'), float('inf')])
def test_invalid_power_is_rejected(power):
    with pytest.raises(ValueError, match='length power'):
        dataset([100., 200.], length_power=power)


def test_cli_default_and_value():
    assert build_parser().parse_args(['--name', 'run']).afv_length_power == 1.
    assert build_parser().parse_args(['--name', 'run', '--afv-length-power', '3']).afv_length_power == 3.


def test_collector_order_is_length_weighted_without_replacement():
    fibers = [SimpleNamespace(length=v) for v in np.r_[np.full(990, 350.), np.full(10, 1500.)]]
    rng = np.random.default_rng(0)
    firsts = []
    for _ in range(200):
        order = fiber_visit_order(fibers, 3., rng)
        assert sorted(order.tolist()) == list(range(len(fibers)))
        firsts.append(order[0] >= 990)
    expected = 10*1500.**3/(10*1500.**3+990*350.**3)
    assert abs(np.mean(firsts)-expected) < .1


def test_collector_default_order_is_the_original_permutation():
    fibers = [SimpleNamespace(length=v) for v in (300., 900., 400.)]
    np.testing.assert_array_equal(fiber_visit_order(fibers, 1., np.random.default_rng(5)),
                                  np.random.default_rng(5).permutation(3))
