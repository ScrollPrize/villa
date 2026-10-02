import json

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared import data as D
from vesuvius.neural_tracing.fiber_follow.shared.annotation_repair import foldbacks, repair_kinks, turning, unit_curve
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength

from test_data_and_scoring import write_fiber


def line(n=81):
    return np.c_[np.arange(n), np.zeros(n), np.zeros(n)].astype(float)


def repaired(points):
    out, runs = repair_kinks(points, arclength(points))
    return out, runs


def test_v_kink_is_bridged_and_nothing_else_moves():
    p = line()
    p[38:43, 1] = [1., 2., 3., 2., 1.]  # a 3-voxel V over 6 voxels
    out, runs = repaired(p)
    assert len(runs) == 1 and len(out) == len(p)
    assert np.abs(out[:, 1]).max() < .5
    start, end = runs[0]
    outside = (p[:, 0] < start) | (p[:, 0] > end)
    np.testing.assert_array_equal(out[outside], p[outside])


def test_repair_is_idempotent_on_survey_shapes():
    p = line(121)
    p[38:43, 1] = [1., 2., 3., 2., 1.]
    p[80:, 1] = 2.
    p[60, 2] = .8  # a one-vertex tick out of plane
    once, runs = repaired(p)
    assert len(runs) == 3
    twice, again = repaired(once)
    assert again == []
    np.testing.assert_array_equal(twice, once)


def test_step_kink_becomes_smooth():
    p = line()
    p[40:, 1] = 2.  # the annotation jumps sideways and continues parallel
    out, runs = repaired(p)
    assert len(runs) == 1
    _, curve = unit_curve(out, arclength(out))
    assert turning(curve).max() < 15


def test_real_geometry_is_untouched():
    t = np.arange(0., 120.)
    smooth = np.c_[t, 2*np.sin(2*np.pi*t/30), np.zeros_like(t)]
    corner = np.r_[line(41), np.c_[np.full(40, 40.), np.arange(1., 41), np.zeros(40)]]  # a 90 degree corner
    angle = np.linspace(0, np.pi, 12)
    hairpin = np.r_[np.c_[np.arange(-30., 0), np.full(30, 3.), np.zeros(30)],
                    np.c_[3*np.sin(angle), 3*np.cos(angle), np.zeros(12)],
                    np.c_[np.arange(-1., -31, -1), np.full(30, -3.), np.zeros(30)]]
    for points in (smooth, corner, hairpin):
        out, runs = repaired(points)
        assert runs == []
        np.testing.assert_array_equal(out, points)


def test_foldback_is_found_and_kept_out_of_training_only():
    out_leg = line(41)
    back = out_leg[-2:-21:-1]+[0., .5, 0.]
    away = np.c_[np.full(30, 21.), .5+np.arange(1., 31), np.zeros(30)]
    p = np.r_[out_leg, back, away]
    assert len(foldbacks(p, arclength(p))) == 1
    assert foldbacks(line(), arclength(line())) == ()
    folded = D.TracedFiber('folded.json', p, arclength(p), 'H', foldbacks=foldbacks(p, arclength(p)))
    clean = D.TracedFiber('clean.json', line(), arclength(line()), 'H')
    train, val = D.split_fibers([folded, clean], D.ZBand(100, 200))
    assert train == [clean] and val == []
    train, val = D.split_fibers([folded], D.ZBand(-1, 1))
    assert train == [] and val == [folded]


def test_loader_repairs_kink_and_keeps_spans_on_vertices(tmp_path):
    raw = write_fiber(tmp_path/'f.json')
    raw['line_points'][20] = [20., 1.5, 0.]
    (tmp_path/'f.json').write_text(json.dumps(raw))
    f, = D.load_fibers(str(tmp_path), grid_scale=1)
    assert f.kink_repairs == 1 and f.foldbacks == ()
    assert np.abs(f.points[:, 1]).max() < .5
    np.testing.assert_array_equal(f.points[[0, -1]], [[5, 0, 0], [33, 0, 0]])
    assert f.spans[0].start == 0 and f.spans[-1].end == f.length
    assert f.spans[0].end == f.spans[1].start
    # The control point at x=17 stays a vertex and its span boundary follows the repaired arclength.
    k = int(np.argmin(np.linalg.norm(f.points-[17, 0, 0], axis=1)))
    assert f.spans[0].end == f.s[k]
