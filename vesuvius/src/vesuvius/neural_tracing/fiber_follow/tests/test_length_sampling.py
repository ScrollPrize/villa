from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser
from vesuvius.neural_tracing.fiber_follow.shared.collect import CoverageCursor


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


def test_collection_covers_fibers_uniformly_unseen_first_and_alternates_directions():
    fibers = [SimpleNamespace(name=f'f{i}', source_hash='', points=np.zeros((2, 3)), endpoint_stop=(False, False),
                              length=v) for i, v in enumerate(np.r_[np.full(9, 350.), [5000.]])]
    cursor = CoverageCursor(fibers, seed=3)
    rng = np.random.default_rng(0)
    first = [cursor.take(rng) for _ in range(10)]
    # Every fiber once per epoch, regardless of length; no fiber repeats before all are seen.
    assert sorted(fi for fi, _ in first) == list(range(10))
    second = [cursor.take(rng) for _ in range(10)]
    signs = {fi: [s] for fi, s in first}
    for fi, s in second:
        signs[fi].append(s)
    assert all(a == -b for a, b in signs.values())
    # Excluded fibers (already in this collection) are deferred, not dropped.
    cursor = CoverageCursor(fibers, seed=3, state=cursor.state)
    taken = cursor.take(rng, exclude={cursor.state['order'][cursor.state['position'] % 10]} if cursor.state['position'] < 10 else ())
    assert taken is not None
