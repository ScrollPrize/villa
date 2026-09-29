"""Source allocation before identity sampling and replay rejection/fallback."""
import unittest
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset


class SamplingRatiosTest(unittest.TestCase):
    def dataset(self, **kwargs):
        ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None, **kwargs)
        # Populate every band so this measures allocation without empty-bank fallback.
        ds.fixed_pools = [{0: [('fixed', np.array([0]))]} for _ in range(5)]
        ds.recent_pools = [{0: [('recent', np.array([0]))]} for _ in range(5)]
        return ds

    def test_custom_mix_and_departure_reservation(self):
        ds = self.dataset(fresh_fraction=.6)
        rng = np.random.default_rng(19)
        counts = np.zeros(3, dtype=int)
        departures = 0
        for _ in range(20000):
            draw = ds.draw_replay(rng)
            source = 0 if draw is None else draw[0]
            counts[source] += 1
            if draw is not None:
                self.assertEqual(draw[2], ('fixed', 'recent')[source-1])
                departures += draw[1] == 4
        np.testing.assert_allclose(counts / counts.sum(), [.6, .2, .2], atol=.015)
        self.assertAlmostEqual(departures / counts[1:].sum(), .1, delta=.015)

    def test_default_preserves_seeded_draws(self):
        default, explicit = self.dataset(), self.dataset(fresh_fraction=.5)
        a, b = np.random.default_rng(8), np.random.default_rng(8)
        self.assertEqual([default.draw_replay(a) for _ in range(100)],
                         [explicit.draw_replay(b) for _ in range(100)])

    def test_endpoints_fallback_and_validation(self):
        rng = np.random.default_rng(5)
        ds = self.dataset(fresh_fraction=1.)
        self.assertTrue(all(ds.draw_replay(rng) is None for _ in range(100)))
        ds = self.dataset(fresh_fraction=0.)
        self.assertTrue(all(ds.draw_replay(rng) is not None for _ in range(100)))
        ds.fixed_pools = ds.recent_pools = [{} for _ in range(5)]
        self.assertIsNone(ds.draw_replay(rng))
        for value in (-.01, 1.01, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.dataset(fresh_fraction=value)


if __name__ == '__main__':
    unittest.main()
