import numpy as np
import sqlite3

from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_crop import export, run, random_groups
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFV_SCHEMA
from vesuvius.neural_tracing.tracers.fibers.evaluation.monitor.microbatch import clipped_polyline


def test_crop_crossings_and_independent_obj_indices(tmp_path):
    points = np.array([[-2., 0, 0], [2, 0, 0], [2, 2, 0], [-2, 2, 0]])
    clipped = clipped_polyline(points, [-1, -1, -1], [1, 3, 1]).reshape(-1, 3, 3)[:, :2]
    assert len(clipped) == 2
    records = [(10, 1, 0, clipped), (20, 0, 1, clipped), (30, 0, 2, clipped)]
    root = tmp_path / 'crop'
    result = export(root, records, [-1, -1, -1], [1, 3, 1])
    for name, group in result['groups'].items():
        assert group['fibers'] == 1
        assert group['vertices'] == 4 and group['lines'] == 2
        text = (root / f'{name}.obj').read_text()
        assert 'l 1 2\n' in text and 'l 3 4\n' in text
        assert '\nf ' not in text
    assert result['groups']['h_that_were_v']['fiber_ids'] == [10]
    assert result['groups']['v_that_were_h']['fiber_ids'] == [20]


def test_auto_crop_from_afv(tmp_path):
    afv = tmp_path / 'input.afv'
    with sqlite3.connect(afv) as db:
        for statement in AFV_SCHEMA:
            db.execute(statement)
        for fid, family in ((10, 'V'), (20, 'H'), (30, 'H')):
            points = np.array([[1., 1, 1], [3, 1, 1]], dtype='<f8')
            db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (fid, str(fid), family, 2, 2., 1., 3., 1., 1., 1., 1., '{}'))
            db.execute('INSERT INTO blocks VALUES(?,?,?,?)', (fid, fid, 0, points.tobytes()))
            db.execute('INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)', (fid, 1., 3., 1., 1., 1., 1.))
    comparison = tmp_path / 'comparison.npz'
    np.savez(comparison, ids=[10, 20, 30], original_labels=[1, 0, 0], labels=[0, 1, 2])
    result = run(dict(afv=str(afv), comparison=str(comparison), output_root=str(tmp_path / 'out'),
                      size_base_voxels=10, minimum_each_change=1))
    assert result['bounds_xyz'] == [[0, 0, 0], [10, 10, 10]]
    assert all(group['fibers'] == 1 for group in result['groups'].values())
    sampled = run(dict(afv=str(afv), comparison=str(comparison), output_root=str(tmp_path / 'full'),
                       sample_per_group=100, seed=42))
    assert sampled['full_fibers']
    assert all(group['fibers'] == 1 and group['lines'] == 1 for group in sampled['groups'].values())
    assert sampled['bounds_xyz'] == [[1, 1, 1], [3, 1, 1]]


def test_random_groups_reproducible_and_bounded():
    ids = np.arange(1000)
    original = ids % 2
    labels = ids % 3
    a = random_groups(ids, original, labels, 100, 42)
    assert a == random_groups(ids, original, labels, 100, 42)
    assert a != random_groups(ids, original, labels, 100, 43)
    assert all(len(group) == len(set(group)) == 100 for group in a.values())
    assert all(labels[a[name]].tolist() == [label] * 100 for name, label in (('h', 0), ('v', 1), ('broken', 2)))
    assert all(original[a['h_that_were_v']] == 1)
    assert all(original[a['v_that_were_h']] == 0)
