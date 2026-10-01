"""Source allocation before identity sampling and replay rejection/fallback."""
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentitySampling


@pytest.mark.parametrize('available', [True, False])
@pytest.mark.parametrize('light_probability', [0., .25])
def test_clean_budget_is_unconditional_and_hard_fallback_stays_clean(monkeypatch, available, light_probability):
    import vesuvius.neural_tracing.fiber_follow.shared.data as module
    monkeypatch.setattr(module, 'FiberVolume', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'crop_local_grid', lambda crop: np.zeros((1, 3)))

    def sample(*args, perturb=True, light_perturbation=None):
        assert not perturb
        assert light_perturbation in (None, (.5, 2.))
        return dict(allowed=True, clean=light_perturbation is None)

    monkeypatch.setattr(module, 'make_sample', sample)

    class Builder:
        sampling = IdentitySampling(decision_fraction=.3, memory_switch_probability=.3,
                                    bank_following_probability=0.)

        def decision_pair(self, cfg, rng):
            return [dict(source=5, allowed=True)]*2 if available else None

        def memory_switch(self, cfg, rng):
            return dict(source=3, allowed=True) if available else None

        def replace_fresh(self, *args):
            raise AssertionError('Reserved clean GT must not be replaced')

        def __call__(self, items, vol):
            return dict(items=items)

    ds = FollowDataset([SimpleNamespace(length=100.)], None,
        SimpleNamespace(crop=None, future_s=[16.]), None, chunk=12,
        batch_builder=Builder(), fresh_fraction=.9, clean_fraction=.8,
        gt_perturb_probability=light_probability)
    monkeypatch.setattr(ds, 'state_allowed', lambda item: item['allowed'])
    monkeypatch.setattr(ds, 'draw_replay', lambda rng, force=False: 'replay' if available else None)
    monkeypatch.setattr(ds, 'replay_item', lambda *args: dict(source=2, allowed=True))
    batches = iter(ds)
    counts = np.zeros(6)
    light = 0
    for _ in range(2000):
        items = next(batches)['items']
        assert len(items) == 12
        assert sum(i['source'] == 5 for i in items) % 2 == 0
        for item in items:
            if item['source'] == 0:
                assert item['clean'] == (not item['gt_perturbed']) == item['gt_unperturbed']
                light += item['gt_perturbed']
            counts[item['source']] += 1
    probabilities = ds.sampling_probabilities()
    expected = ([.8, 0., probabilities['recent'], probabilities['memory_switch'], 0., probabilities['decision']]
                if available else [1., 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(counts/counts.sum(), expected, atol=.015)
    assert light/counts[0] == pytest.approx(light_probability, abs=.015)


def test_70_10_10_10_source_budget():
    ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None,
        clean_fraction=.7, fresh_fraction=.75, gt_perturb_probability=.25,
        batch_builder=SimpleNamespace(sampling=IdentitySampling(
            decision_fraction=.2, bank_following_probability=0., memory_switch_probability=1/3)))
    assert ds.sampling_probabilities() == pytest.approx(
        dict(clean=.7, decision=.1, bank_following=0., memory_switch=.1, recent=.1))
    assert ds.clean_fraction*(1-ds.gt_perturb_probability) > .5


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_invalid_clean_fraction(value):
    with pytest.raises(ValueError, match='Clean fraction'):
        FollowDataset([SimpleNamespace(length=100.)], None, None, None, clean_fraction=value)


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


@pytest.mark.parametrize('available', [True, False])
def test_light_gt_slots_prefer_correct_replay_and_preserve_clean_gt(monkeypatch, available):
    import vesuvius.neural_tracing.fiber_follow.shared.data as module
    monkeypatch.setattr(module, 'FiberVolume', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'crop_local_grid', lambda crop: np.zeros((1, 3)))
    monkeypatch.setattr(module, 'make_sample', lambda *a, **kw: dict(allowed=True))
    builder = SimpleNamespace(sampling=IdentitySampling(decision_fraction=0., bank_following_probability=0.),
                              __call__=lambda items, vol: dict(items=items))
    class Builder:
        sampling = builder.sampling
        def __call__(self, items, vol):
            return dict(items=items)
    ds = FollowDataset([SimpleNamespace(length=100.)], None,
        SimpleNamespace(crop=None, future_s=[16.]), None, chunk=10, batch_builder=Builder(),
        clean_fraction=1., gt_perturb_probability=.25, prefer_replay_for_light_gt=True)
    monkeypatch.setattr(ds, 'state_allowed', lambda item: True)
    monkeypatch.setattr(ds, 'correct_continuation_item', lambda rng:
        dict(source=2, replay_correct_continuation=True, light_gt_replay=True) if available else None)
    counts = dict(clean=0, light=0, replay=0)
    batches = iter(ds)
    for _ in range(1000):
        for item in next(batches)['items']:
            if item['source'] == 2:
                assert item['replay_correct_continuation'] and item['light_gt_replay']
                counts['replay'] += 1
            else:
                counts['light' if item['gt_perturbed'] else 'clean'] += 1
    assert counts['clean']/10000 == pytest.approx(.75, abs=.02)
    assert counts['replay' if available else 'light']/10000 == pytest.approx(.25, abs=.02)
    assert counts['light' if available else 'replay'] == 0
