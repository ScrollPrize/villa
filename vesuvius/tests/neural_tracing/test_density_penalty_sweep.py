import math
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('density_sweep', Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/density_penalty_sweep.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mapping, coarse_grid, refine, resumable, best_result = (getattr(module, name) for name in ('mapping', 'coarse_grid', 'refine', 'resumable', 'best_result'))


def test_mapping():
    assert mapping(.5, .5, .1) == .5
    assert mapping(100, .5, .1) == .1
    assert mapping(0, .5, .1) == 1
    for args in [(-1, 1, .1), (1, 0, .1), (1, 1, 0), (1, 1, 2), (math.nan, 1, .1)]:
        with pytest.raises(ValueError):
            mapping(*args)


def test_grid_refinement():
    grid = coarse_grid()
    assert len(set(grid)) == 9
    points = refine(.5, .1, grid, 0)
    assert len(points) == 4
    assert points[0][0] < .5 < points[1][0]
    assert points[2][1] < .1 < points[3][1]
    assert all(0 < mapping(.3, s, f) <= 1 for s, f in points)


def test_resume():
    record = dict(returncode=0, command=['cmd'], inputs={'x': 'a'}, references={}, binary_sha256='b', shared_libraries={})
    args = (['cmd'], {'x': 'a'}, {}, 'b', {}, b'patch', b'patch')
    assert resumable(record, *args)
    assert not resumable(dict(record, returncode=1), *args)
    assert not resumable(dict(record, inputs={'x': 'changed'}), *args)
    assert not resumable(record, *args[:-1], b'changed')


def test_ties():
    rows = [dict(name='b', constraint_score_percent=-10, mean_multiplier=.5), dict(name='a', constraint_score_percent=-10, mean_multiplier=1)]
    assert best_result(rows)['name'] == 'a'


def test_resume_patch_excludes_reports():
    native = b'diff --git a/volume-cartographer/core/a.cpp b/volume-cartographer/core/a.cpp\nchange\n'
    report = b'diff --git a/volume-cartographer/docs/report.md b/volume-cartographer/docs/report.md\ntext\n'
    assert module.native_patch(native + report) == native


def test_initial_only_metrics_validate_reference_coverage():
    header = 'independent_reference_summary\nreference,calibrated\n'
    candidates = 'independent_reference_candidates\nreference,winding,rank,hard_sign_errors\n'
    end = 'independent_reference_end\n'
    good = header+'0,1\n0.5,1\n'+candidates+'0,0,1,0\n0.5,0.5,1,0\n'+end
    assert module.independent_metrics(good)['independent_percent'] == 100
    missing = header+'0,1\n0.5,1\n'+candidates+'0,0,1,0\n'+end
    assert module.independent_metrics(missing)['independent_missing'] == 1
    assert module.independent_metrics(missing)['independent_percent'] == 50
    for bad in (header+candidates+end, good.replace('0.5,0.5,1,0', '2,2,1,0'),
                good.replace('0.5,0.5,1,0', '0.5,nan,1,0')):
        with pytest.raises(ValueError):
            module.independent_metrics(bad)
