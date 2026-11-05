"""Evaluation scoring: length past an untagged annotation end is unknown, also after a departure."""
import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.fibers.evaluation.scoring import score_trace
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.geometry import arclength


@pytest.mark.parametrize('offset, tagged, beyond_unknown', [(5., False, True), (15., False, False), (5., True, False)])
def test_departed_path_past_annotation_end(offset, tagged, beyond_unknown):
    p = np.c_[np.arange(101.), np.zeros(101), np.zeros(101)]
    fiber = TracedFiber('test.json', p, arclength(p), 'H', endpoint_stop=(False, tagged))
    # On the fiber from x=50 to 80, then `offset` voxels to the side until x=200 (the annotation ends at x=100).
    path = np.r_[np.c_[np.arange(50., 81), np.zeros(31), np.zeros(31)],
                 np.c_[np.arange(81., 201), np.full(120, offset), np.zeros(120)]]
    m = score_trace(path, fiber, 50., 1.)
    assert m['diverged'] and m['correct'] == pytest.approx(30.)
    total = m['length']
    if beyond_unknown:
        assert m['unknown'] == pytest.approx(100.) and m['offtrack'] == pytest.approx(total - 130.)
    else:
        assert m['unknown'] == 0. and m['offtrack'] == pytest.approx(total - 30.)
