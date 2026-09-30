"""Source allocation before identity sampling and replay rejection/fallback."""
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentitySampling


@pytest.mark.parametrize('fresh', [0., .7, 1.])
def test_bank_budget_is_independent_of_fresh_and_preserves_pair_slots(fresh):
    builder = SimpleNamespace(sampling=IdentitySampling(decision_fraction=.3, bank_following_probability=.2))
    ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None,
                       chunk=8, batch_builder=builder, fresh_fraction=fresh)
    rng = np.random.default_rng(29)
    counts = np.zeros(3)
    for _ in range(10000):
        pairs, bank = ds.endpoint_requests(rng)
        assert 0 <= 2*pairs+bank <= ds.chunk
        counts += [2*pairs, bank, ds.chunk-2*pairs-bank]
    np.testing.assert_allclose(counts/counts.sum(), [.3, .2, .5], atol=.01)


@pytest.mark.parametrize('decisions,following', [(1., 0.), (0., 1.), (.8, .2), (0., 0.)])
def test_endpoint_budget_boundaries(decisions, following):
    builder = SimpleNamespace(sampling=IdentitySampling(
        decision_fraction=decisions, bank_following_probability=following))
    ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None, batch_builder=builder)
    rng = np.random.default_rng(7)
    for _ in range(20):
        pairs, bank = ds.endpoint_requests(rng)
        assert 2*pairs+bank == (ds.chunk if decisions+following else 0)
    with pytest.raises(ValueError, match='endpoint fractions'):
        IdentitySampling(decision_fraction=.81, bank_following_probability=.2)
    # Covered annotation locations no longer compete with the following budget.
    IdentitySampling(bank_following_probability=1., bank_coverage_probability=.6, memory_switch_probability=.4)


@pytest.mark.parametrize('bank_status', ['valid', 'missing', 'unsafe'])
def test_loader_bank_budget_and_fallback_do_not_replace_fresh(monkeypatch, bank_status):
    import vesuvius.neural_tracing.fiber_follow.shared.data as module
    monkeypatch.setattr(module, 'FiberVolume', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'crop_local_grid', lambda crop: np.zeros((1, 3)))
    monkeypatch.setattr(module, 'make_sample', lambda *a, **kw: dict(allowed=True))

    class Builder:
        sampling = IdentitySampling(decision_fraction=.3, bank_following_probability=.2)
        bank_calls = 0

        def decision_pair(self, cfg, rng):
            return [dict(source=5, allowed=True), dict(source=5, allowed=True)]

        def bank_following(self, cfg, rng):
            self.bank_calls += 1
            return None if bank_status == 'missing' else dict(source=4, allowed=bank_status == 'valid')

        def __call__(self, items, vol):
            return dict(items=items)

    builder = Builder()
    ds = FollowDataset([SimpleNamespace(length=100.)], None,
        SimpleNamespace(crop=None, future_s=[16.]), None, chunk=8,
        batch_builder=builder, fresh_fraction=1.)
    monkeypatch.setattr(ds, 'state_allowed', lambda item: item['allowed'])
    batches = iter(ds)
    counts = np.zeros(6)
    for _ in range(1000):
        items = next(batches)['items']
        assert len(items) == 8
        for item in items:
            counts[item['source']] += 1
    assert builder.bank_calls/8000 == pytest.approx(.2, abs=.02)
    assert counts[5]/8000 == pytest.approx(.3, abs=.02)
    assert counts[4] == (builder.bank_calls if bank_status == 'valid' else 0)
    assert counts[0]/8000 == pytest.approx(.5 if bank_status == 'valid' else .7, abs=.02)


class SamplingRatiosTest(unittest.TestCase):
    def dataset(self, **kwargs):
        ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None, **kwargs)
        # Populate every band so this measures allocation without empty-bank fallback.
        ds.recent_pools = [{0: [('recent', np.array([0]))]} for _ in range(5)]
        return ds

    def test_custom_mix_and_departure_reservation(self):
        ds = self.dataset(fresh_fraction=.7)
        rng = np.random.default_rng(19)
        counts = np.zeros(3, dtype=int)
        departures = 0
        for _ in range(20000):
            draw = ds.draw_replay(rng)
            source = 0 if draw is None else draw[0]
            counts[source] += 1
            if draw is not None:
                self.assertEqual(source, 2)
                self.assertEqual(draw[2], 'recent')
                departures += draw[1] == 4
        np.testing.assert_allclose(counts / counts.sum(), [.7, 0., .3], atol=.015)
        self.assertAlmostEqual(departures / counts[1:].sum(), .1, delta=.015)

    def test_default_preserves_seeded_draws(self):
        default, explicit = self.dataset(), self.dataset(fresh_fraction=.7)
        a, b = np.random.default_rng(8), np.random.default_rng(8)
        self.assertEqual([default.draw_replay(a) for _ in range(100)],
                         [explicit.draw_replay(b) for _ in range(100)])

    def test_endpoints_fallback_and_validation(self):
        rng = np.random.default_rng(5)
        ds = self.dataset(fresh_fraction=1.)
        self.assertTrue(all(ds.draw_replay(rng) is None for _ in range(100)))
        ds = self.dataset(fresh_fraction=0.)
        self.assertTrue(all(ds.draw_replay(rng) is not None for _ in range(100)))
        ds.recent_pools = [{} for _ in range(5)]
        self.assertIsNone(ds.draw_replay(rng))
        for value in (-.01, 1.01, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.dataset(fresh_fraction=value)


if __name__ == '__main__':
    unittest.main()
