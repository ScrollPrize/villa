import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('alternatives', Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/density_alternatives.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('power', [.5, 1., 2.])
def test_centered_preserves_weighted_budget(power):
    weights = np.array([1., 3., 8.])
    values = module.centered([1., .7, .5], weights, power)
    assert np.average(values, weights=weights) == pytest.approx(1.)
    assert values[0] > 1 > values[-1]


def test_counts_ignore_inactive_include_hard(tmp_path):
    p = tmp_path / 'f.csv'
    p.write_text('piece_a,piece_b,effective_parallel_winding_weight,effective_perpendicular_winding_weight,effective_parallel_sign_penalty,effective_perpendicular_sign_penalty,hard_parallel_sign,hard_perpendicular_sign\n0,1,0,0,0,0,0,0\n1,2,0,0,0,0,0,1\n0,2,1,0,0,0,0,0\n')
    assert list(module.measurement_counts(p, 3)) == [1, 1, 2]


def test_sustained_cuts_and_relocated_lengths():
    # Three consecutive 32-vx intervals; only the last two have high density.
    rows = np.array([[0, 64, 0, 0, 1], [64, 96, 64, 0, 1], [96, 128, 64, 0, 1], [128, 256, 0, 0, 1]], dtype=float)
    cuts = module.sustained_cuts(rows, 1., 64.)
    assert cuts == [64., 128.]
    assert module.sustained_cuts(rows, 1., 128.) == []
    moved = module.relocated_cuts(cuts, 256., np.random.default_rng(3))
    assert sorted(np.diff([0., *cuts, 256.])) == sorted(np.diff([0., *moved, 256.]))
