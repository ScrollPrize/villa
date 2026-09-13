import numpy as np
import torch

from lasagna.demand_predict3d import DemandGeometry
from vesuvius.neural_tracing.fiber_trace_3d.embedding_consistency import (
    analyze_window, corners, interpolate, sample_line,
)


def test_linear_interpolation_and_line_spacing():
    z, y, x = np.indices((5, 5, 5))
    p = np.array([[.25, 1.5, 2.75], [1, 2, 3]])
    np.testing.assert_allclose(interpolate((z+2*y+3*x)[None], p)[:, 0], p @ [1, 2, 3])
    np.testing.assert_allclose(corners(p)[1].sum(1), 1)
    line = sample_line(np.array([[0., 0, 0], [0, 0, 20]]), 2, 12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(line, axis=0), axis=1), 2)


class Adapter:
    def preprocess_tile(self, tile, valid):
        return tile


class Model:
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, tile, return_embeddings):
        e = torch.zeros((1, 2, *tile.shape[-3:]))
        e[:, 0] = -1 if self.flip and tile[0, 0, 0, 0, 0] > 0 else 1
        return dict(embeddings=e, prediction=torch.ones((1, 7, *tile.shape[-3:]))*.5)


def measure(flip):
    geometry = DemandGeometry((40, 40, 40), 16, 8, 2, 1, 4)
    points = np.array([[14., 14., x] for x in np.arange(10, 26, .5)])
    return analyze_window(Model(flip), Adapter(),
        lambda tile: np.full((16,)*3, int(tile[2] >= 16), np.uint8),
        geometry, points, points, None, torch.device('cpu'), 1)


def test_identical_tiles_have_no_blend_or_storage_seams():
    r = measure(False)
    assert r['metrics']['same_point_cross_tile']['min'] == 1
    assert r['metrics']['cancellation']['min'] > .99999
    for row in r['metrics']['by_lag'].values():
        for v in row.values():
            if v['n']:
                assert v['min'] > .99999


def test_antipodal_tiles_are_not_mistaken_for_consistency():
    r = measure(True)
    assert r['metrics']['same_point_cross_tile']['min'] == -1
    assert r['metrics']['cancellation']['min'] < .5
