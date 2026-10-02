"""Batched broad-phase queries retain scalar projection and tie ordering."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import PolylineIndex, exact_nearest


def scalar_nearest(points, index):
    def project(point, ids):
        delta, a = index.delta[ids], index.a[ids]
        u = np.clip(((point-a)*delta).sum(-1)/np.maximum(index.length2[ids], 1e-20), 0, 1)
        q = a+u[:, None]*delta
        distances = np.linalg.norm(point-q, axis=-1)
        j = distances.argmin()
        return distances[j], q[j], ids[j], u[j]
    rows = []
    for point in points:
        first = index.tree.query(point)[1]
        best = project(point, np.array([first]))[0]
        ids = np.asarray(index.tree.query_ball_point(point, best+index.half+1e-8))
        rows.append(project(point, ids))
    return tuple(np.asarray(values) for values in zip(*rows))


def test_batched_nearest_is_bitwise_equal_including_ties():
    rng = np.random.default_rng(72)
    for vertices in (2, 2, 250, 250):
        target = rng.normal(size=(vertices, 3)).cumsum(0)
        target[1] = target[0]  # repeated vertices and zero-length segments
        points = np.concatenate([target, rng.normal(size=(100, 3))*15])
        index = PolylineIndex(target)
        for actual, expected in zip(exact_nearest(points, target, index), scalar_nearest(points, index)):
            np.testing.assert_array_equal(actual, expected)
