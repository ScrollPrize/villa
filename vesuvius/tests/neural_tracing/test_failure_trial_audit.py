import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.failure_trial_audit import (
    audit_trace_batch, bounded_map)
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import (
    audit_trial_artifacts, recover_completed_trial_batches)


def fixture(tmp_path, count=3):
    result = tmp_path/'batch/attempt_1_inference/full'
    result.mkdir(parents=True)
    (result/'summary.json').write_text(json.dumps(dict(completed=True, cases=count+1,
        evaluated_cases=count, cancelled_cases=1)))
    expected = {}
    for i in range(count+1):
        key = (str(i), 'forward', 100.)
        expected[key] = ({}, 'forward', 100, tmp_path/'trials'/f'{i}.json', 0, 200)
        if i == count:
            continue
        path = result/'cases'/str(i)/'trace.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(provenance=dict(candidate_id=str(i), direction='forward',
            target_distance_base=100, native_reproduced=False, coordinate_contract='original-fiber-arclength-v2',
            expected_start_base_xyz=[0,0,0], expected_end_base_xyz=[200,0,0]),
            reference_points_base_xyz=[[0,0,0], [200,0,0]], replay=dict(failures=[]))))
    return result, expected


def test_parallel_audit_and_accounting(tmp_path):
    result, expected = fixture(tmp_path)
    assert audit_trial_artifacts(result, expected, grace=80, global_worklist=True, workers=2) == 3
    for i in range(3):
        doc = json.loads((tmp_path/'trials'/f'{i}.json').read_text())
        assert doc['valid'] is False and doc['original_arcs_base'] == [0, 200]
    assert not (tmp_path/'trials/3.json').exists()  # Pruning is not a negative trial.


@pytest.mark.parametrize('error', ['duplicate', 'missing', 'native', 'count'])
def test_audit_rejects_invalid_batch(tmp_path, error):
    result, expected = fixture(tmp_path)
    path = result/'cases/0/trace.json'
    if error == 'duplicate':
        other = result/'cases/extra/trace.json'; other.parent.mkdir()
        other.write_bytes(path.read_bytes())
    elif error == 'missing':
        path.unlink()
    elif error == 'native':
        doc = json.loads(path.read_text()); doc['provenance']['native_reproduced'] = True
        path.write_text(json.dumps(doc))
    else:
        doc = json.loads((result/'summary.json').read_text()); doc['cancelled_cases'] = 2
        (result/'summary.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        audit_trial_artifacts(result, expected, grace=80, global_worklist=True, workers=2)


def test_completed_recovery_uses_inference_attempt(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import short_failure_benchmark as m
    batch = tmp_path/'batches/old_identity'
    result = batch/'attempt_1_inference/full'; result.mkdir(parents=True)
    (result/'summary.json').write_text(json.dumps(dict(completed=True)))
    (batch/'attempt_2/full').mkdir(parents=True)  # Interrupted attempt ignored.
    candidate = dict(id='a', piece_key=['fiber.json'], source_id='s',
                     target=400, run=dict(original_arcs_base=[0, 1000]))
    trial = dict(candidate_id='a', direction='forward', distance=100,
                 coordinate_contract='original-fiber-arclength-v2',
                 fiber_path='fiber.json', source_id='s', original_fiber_arc_start_base=300,
                 original_fiber_arc_end_base=500,
                 original_arcs_base=[300, 500])
    (batch/'evaluation.json').write_text(json.dumps(dict(failure_search_trials=[trial])))
    observed = []
    def audit(root, expected, **kwargs):
        observed.append((root, expected)); return 1
    monkeypatch.setattr(m, 'audit_trial_artifacts', audit)
    assert recover_completed_trial_batches(tmp_path, [candidate], 8) == 1
    assert observed[0][0] == result
    assert observed[0][1][('a', 'forward', 100.)][3] == tmp_path/'trials/a_forward_100.json'
    trial['original_fiber_arc_start_base'] = 301
    (batch/'evaluation.json').write_text(json.dumps(dict(failure_search_trials=[trial])))
    with pytest.raises(ValueError, match='geometry'):
        recover_completed_trial_batches(tmp_path, [candidate], 8)


def test_audit_endpoint_guard(tmp_path):
    path = tmp_path/'trace.json'
    doc = dict(provenance=dict(coordinate_contract='original-fiber-arclength-v2',
        candidate_id='x', direction='reverse', target_distance_base=100,
        expected_start_base_xyz=[400, 5, 6], expected_end_base_xyz=[200, 5, 6]),
        reference_points_base_xyz=[[400.1, 5, 6], [200, 5, 6]], replay=dict(failures=[]))
    path.write_text(json.dumps(doc))
    assert audit_trace_batch(([str(path)], 80, True, {}))[0][2]['valid'] is False
    doc['reference_points_base_xyz'][0][0] -= 200
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='endpoint'):
        audit_trace_batch(([str(path)], 80, True, {}))


def test_bounded_map_order_and_errors():
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(bounded_map(pool, abs, range(-20, 0), pending=3)) == list(range(20, 0, -1))
        with pytest.raises(ZeroDivisionError):
            list(bounded_map(pool, lambda x: 1/x, [1, 0, 2], pending=2))


def test_vectorized_corridor_matches_serial_and_benchmark(monkeypatch):
    import itertools
    import time
    import statistics
    import numpy as np
    from vesuvius.neural_tracing.fiber_trace_3d import short_failure_benchmark as m
    shape = np.array([100, 200, 300]); chunks = np.array([8, 16, 32]); scale = np.array([8., 4., 2.])
    monkeypatch.setattr(m, '_zarr_chunk_geometry', lambda root: (shape, chunks, scale))
    points = np.cumsum(np.random.default_rng(123).normal(size=(50000, 3)), axis=0) + [100, 200, 300]
    points[0] = [-1000, -2000, -3000]
    radius = 64
    def serial():
        sampled = points[::4]
        if not np.array_equal(sampled[-1], points[-1]):
            sampled = np.concatenate((sampled, points[-1:]), axis=0)
        result = set()
        for point in sampled:
            centre = point[::-1]/scale
            lo = np.maximum(np.floor(centre-radius/scale).astype(int), 0)
            hi = np.minimum(np.floor(centre+radius/scale).astype(int), shape-1)
            if np.any(hi < lo): continue
            result.update(itertools.product(*(range(lo[k]//chunks[k], hi[k]//chunks[k]+1) for k in range(3))))
        return result
    timings = [[], []]
    for _ in range(3):
        start = time.perf_counter(); expected = serial(); timings[0].append(time.perf_counter()-start)
        start = time.perf_counter(); actual = m._corridor_chunk_keys(points, 'unused', radius)
        timings[1].append(time.perf_counter()-start)
        assert actual == expected
    print('corridor 50000 points, 3 iterations:',
          [(statistics.mean(t), statistics.median(t), min(t), max(t)) for t in timings])
