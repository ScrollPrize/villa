"""The fiber CC loss sampler: fibers weighted by item count ** exponent."""

import numpy as np
import pytest
import torch

from fiber_cc_loss import _Table

# Items per fiber and each fiber's CC: CC 0 holds an empty fiber, CC 2 a
# single fiber, CC 3 an empty fiber only, CC 5 nothing at all.
COUNTS = np.array([5, 0, 1, 20, 3, 0, 7, 0, 12, 2, 1, 40])
CCS = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4])
NUM_CC = 6
START = np.cumsum(COUNTS) - COUNTS


def _table():
    return _Table(START, COUNTS, CCS, NUM_CC, torch.device('cpu'))


def _fiber_of(item):
    """The fiber holding each item (only fibers with items own any)."""
    return np.searchsorted(START + COUNTS, item, side='right')


def _old_sample(table, draws, per_draw, generator):
    """The uniform-fiber draw as it was before the exponent existed."""
    cc = torch.multinomial(table.cc_weight, draws, replacement=True, generator=generator)
    cc = cc[:, None].expand(draws, per_draw).reshape(-1)
    u = torch.rand((2, cc.numel()), device=cc.device, generator=generator)
    count = table.cc_count[cc]
    rank = torch.minimum((u[0] * count).long(), count - 1)
    fiber = table.fiber_order[table.cc_start[cc] + rank]
    count = table.item_count[fiber]
    item = table.item_start[fiber] + torch.minimum((u[1] * count).long(), count - 1)
    return cc, item


@pytest.mark.parametrize('exponent', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('per_draw', [1, 4])
def test_fiber_frequencies_follow_length_to_the_exponent(exponent, per_draw):
    generator = torch.Generator().manual_seed(1234)
    draws = 200_000
    cc, item = _table().sample(draws, per_draw, generator, exponent=exponent)
    cc, item = cc.numpy(), item.numpy()
    fiber = _fiber_of(item)

    # Every item lies inside a fiber with items, of the CC drawn.
    assert ((item >= 0) & (item < COUNTS.sum())).all()
    assert (COUNTS[fiber] > 0).all()
    np.testing.assert_array_equal(CCS[fiber], cc)
    # One CC per draw, repeated per_draw times.
    assert (cc.reshape(draws, per_draw) == cc.reshape(draws, per_draw)[:, :1]).all()

    weight = np.where(COUNTS > 0, COUNTS.astype(float) ** exponent, 0.0)
    expected = weight / weight.sum()
    observed = np.bincount(fiber, minlength=len(COUNTS)) / fiber.size
    # Draws within one CC are not independent across per_draw; size the
    # tolerance by the number of CC draws.
    sigma = np.sqrt(expected * (1 - expected) / draws)
    assert (np.abs(observed - expected) <= 6 * sigma + 1e-12).all(), (observed, expected)

    # Items uniformly within each fiber.
    for f in np.flatnonzero(COUNTS > 1):
        local = item[fiber == f] - START[f]
        share = np.bincount(local, minlength=COUNTS[f]) / local.size
        bound = 6 * np.sqrt((1 / COUNTS[f]) / local.size)
        assert np.abs(share - 1 / COUNTS[f]).max() <= bound


def test_exponent_zero_reproduces_the_uniform_fiber_draw():
    table = _table()
    new = table.sample(5000, 3, torch.Generator().manual_seed(7), exponent=0.0)
    old = _old_sample(table, 5000, 3, torch.Generator().manual_seed(7))
    for a, b in zip(new, old):
        assert torch.equal(a, b)
    # The default is the uniform-fiber draw too (the health metrics use it).
    default = table.sample(5000, 3, torch.Generator().manual_seed(7))
    for a, b in zip(default, old):
        assert torch.equal(a, b)


def test_weighted_tables_follow_a_changed_exponent():
    table = _table()
    generator = torch.Generator().manual_seed(3)
    table.sample(10, 1, generator, exponent=1.0)
    table.sample(10, 1, generator, exponent=0.5)
    cc_weight = table._weighted(0.5)[3].numpy()
    weight = np.where(COUNTS > 0, COUNTS.astype(float) ** 0.5, 0.0)
    np.testing.assert_allclose(
        cc_weight, np.bincount(CCS, weights=weight, minlength=NUM_CC), rtol=1e-6)

