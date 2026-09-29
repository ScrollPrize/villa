"""Analytic geometry checks; runnable with unittest without extra dependencies."""
import unittest
import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
from vesuvius.neural_tracing.fiber_follow.shared.recovery_scoring import PolylineProjector, recovery_score


def fiber(points=((0, 0, 0), (100, 0, 0)), known=False):
    p = np.asarray(points, dtype=float)
    return TracedFiber('synthetic', p, arclength(p), '', endpoint_stop=(known, known))


class RecoveryScoringTests(unittest.TestCase):
    def test_straight_and_sparse_annotation_segments(self):
        r = recovery_score([[10, 0, 0], [70, 0, 0]], fiber(), 10, 1)
        self.assertAlmostEqual(r['local_precision'], 1)
        self.assertAlmostEqual(r['recovered_coverage_length'], 60)
        self.assertEqual(r['excursion_count'], 0)
        self.assertAlmostEqual(r['distance_max'], 0)

    def test_recovery_credits_later_path_without_filling_gap(self):
        p = [[0,0,0], [10,0,0], [10,10,0], [30,10,0], [30,0,0], [50,0,0]]
        r = recovery_score(p, fiber(), 0, 1)
        self.assertAlmostEqual(r['local_correct_length'], 36)
        self.assertAlmostEqual(r['local_wrong_length'], 34)
        self.assertAlmostEqual(r['recovered_coverage_length'], 30)
        self.assertEqual(r['excursion_count'], 1)
        self.assertEqual(r['recovery_count'], 1)
        self.assertAlmostEqual(r['longest_excursion_length'], 34)
        self.assertAlmostEqual(r['first_sustained_departure_length'], 13)

    def test_retracing_does_not_inflate_coverage(self):
        r = recovery_score([[0,0,0], [40,0,0], [10,0,0], [50,0,0]], fiber(), 0, 1)
        self.assertAlmostEqual(r['recovered_coverage_length'], 50)
        self.assertAlmostEqual(r['backtrack_length'], 30)
        self.assertAlmostEqual(r['local_precision'], 1)

    def test_reverse_direction(self):
        r = recovery_score([[90,0,0], [30,0,0]], fiber(), 90, -1)
        self.assertAlmostEqual(r['recovered_coverage_length'], 60)
        self.assertAlmostEqual(r['recovered_coverage'], 60/90)

    def test_unknown_endpoint_after_recovery_is_censored(self):
        p = [[0,0,0], [10,0,0], [10,10,0], [30,10,0], [30,0,0], [120,0,0]]
        r = recovery_score(p, fiber(), 0, 1)
        self.assertAlmostEqual(r['local_unknown_length'], 20)
        self.assertAlmostEqual(r['recovered_coverage_length'], 80)
        self.assertEqual(r['recovery_count'], 1)
        self.assertAlmostEqual(r['local_scored_length'], 120)

    def test_known_endpoint_overrun_is_wrong(self):
        r = recovery_score([[0,0,0], [120,0,0]], fiber(known=True), 0, 1)
        self.assertAlmostEqual(r['local_correct_length'], 100)
        self.assertAlmostEqual(r['local_wrong_length'], 20)
        self.assertAlmostEqual(r['local_endpoint_overrun_length'], 20)
        self.assertAlmostEqual(r['local_unknown_length'], 0)

    def test_short_deviation_affects_local_score_without_excursion(self):
        p = [[0,0,0], [10,3.1,0], [10.1,0,0], [30,0,0]]
        r = recovery_score(p, fiber(), 0, 1)
        self.assertLess(r['local_precision'], 1)
        self.assertEqual(r['excursion_count'], 0)

    def test_empty_continuation(self):
        r = recovery_score([[0,0,0]], fiber(), 0, 1)
        self.assertIsNone(r['local_precision'])
        self.assertEqual(r['recovered_coverage_length'], 0)

    def test_close_remote_winding_cannot_credit_arc_jump(self):
        f = fiber([[0,0,0], [100,0,0], [100,2,0], [0,2,0]])
        r = recovery_score([[0,0,0], [10,0,0], [10,2,0], [0,2,0]], f, 0, 1)
        self.assertLessEqual(r['recovered_coverage_length'], 11)
        self.assertLess(r['local_precision'], 1)

    def test_projection_matches_bruteforce_with_unequal_segments(self):
        rng = np.random.default_rng(4)
        points = np.cumsum(rng.normal(size=(30,3)) * rng.uniform(.1,20,(30,1)), axis=0)
        f = fiber(points); projector = PolylineProjector(f)
        q = rng.normal(size=(100,3))*40
        d, s = projector.project(q)
        a, v = points[:-1], np.diff(points,axis=0)
        u = np.clip(np.sum((q[:,None]-a)*v,axis=2)/np.sum(v*v,axis=1),0,1)
        dd = np.linalg.norm(q[:,None]-a-u[...,None]*v,axis=2)
        np.testing.assert_allclose(d,dd.min(axis=1),atol=1e-10)


if __name__ == '__main__':
    unittest.main()
