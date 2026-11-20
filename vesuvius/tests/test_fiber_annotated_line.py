import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.shared.annotations.annotated_line import (
    annotated_line, annotated_line_range, document_control_points, packed_annotated_ranges)

LINE = np.column_stack([np.arange(10.), np.zeros(10), np.zeros(10)])


def test_tails_on_both_sides_are_dropped():
    controls = LINE[[2, 5, 7]]
    assert annotated_line_range(controls, LINE) == (2, 8)
    np.testing.assert_array_equal(annotated_line(controls, LINE), LINE[2:8])


def test_nearest_vertex_and_outermost_tie():
    controls = np.array([[2.4, 1, 0], [6.6, -1, 0]])
    assert annotated_line_range(controls, LINE) == (2, 8)
    repeated = np.repeat(LINE, 2, axis=0)  # duplicate vertices tie: first takes lowest, last highest
    assert annotated_line_range(LINE[[1, 3]], repeated) == (2, 8)


@pytest.mark.parametrize('controls', [np.empty((0, 3)), LINE[[4]]])
def test_fewer_than_two_controls_keep_the_whole_line(controls):
    assert annotated_line_range(controls, LINE) == (0, 10)


def test_empty_line_and_control_order():
    assert annotated_line_range(LINE[[1, 2]], np.empty((0, 3))) == (0, 0)
    assert annotated_line_range(LINE[[6, 2]], LINE) == (2, 7)
    assert annotated_line_range(np.r_[LINE[[6]], [[np.inf, 0, 0]], LINE[[2]]], LINE) == (2, 7)


def test_packed_matches_single():
    lines = [LINE, LINE[:4], np.empty((0, 3)), LINE[::-1]]
    controls = [LINE[[2, 7]], np.empty((0, 3)), LINE[[1, 2]], LINE[[8, 3, 1]]]
    line_offsets = np.r_[0, np.cumsum([len(x) for x in lines])]
    control_offsets = np.r_[0, np.cumsum([len(c) for c in controls])]
    begin, end = packed_annotated_ranges(control_offsets, np.concatenate(controls), line_offsets, np.concatenate(lines))
    assert list(zip(begin, end)) == [annotated_line_range(c, x) for c, x in zip(controls, lines)]


def test_document_control_points_both_versions():
    np.testing.assert_array_equal(document_control_points({'control_points': [[1, 2, 3], [4, 5, 6]]}),
                                  [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(document_control_points({'control_points': [{'position': [1, 2, 3]}]}), [[1, 2, 3]])
    assert document_control_points({}).shape == (0, 3)


FIBERS = [(dict(id='00000000-0000-4000-8000-000000000001', name='m', family='H', control_points=LINE[[2, 5, 7]]), LINE),
          (dict(id='00000000-0000-4000-8000-000000000002', name='t', family='V'), LINE+[0, 1, 0])]


def test_store_readers_drop_tails_by_default(tmp_path):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
        fiber_catalog, read_polylines, write_initial_fibers)
    write_initial_fibers(tmp_path/'f.vcobj', FIBERS, volume='vol', packed=True)
    (row, xyz), (_, whole) = read_polylines(tmp_path/'f.vcobj')
    np.testing.assert_array_equal(xyz, LINE[2:8])
    np.testing.assert_array_equal(whole, LINE+[0, 1, 0])
    assert row['line_range'] == (2, 8) and row['point_count'] == 6 and row['length'] == 5.
    assert row['bbox'] == [2., 0., 0., 7., 0., 0.]
    np.testing.assert_array_equal(read_polylines(tmp_path/'f.vcobj', [row['id']], raw=True)[0][1], LINE)
    assert [r['length'] for r in fiber_catalog(tmp_path/'f.vcobj')] == [5., 9.]


def test_training_store_drops_tails(tmp_path):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import FiberStore, write_fibers
    write_fibers(tmp_path/'f.vcobj', FIBERS, volume='vol')
    manual, trace = (row['id'] for row, _ in FIBERS)
    store = FiberStore(tmp_path/'f.vcobj', 'vol', split='all')
    points = store.geometry(store.id_to_index[manual]).points
    assert points[:, 0].min() == 2. and points[:, 0].max() == 7.
    assert store.nearby_fiber_ids(np.array([0.5, 0.5, 0]), .6, None) == [trace]
    blocks = list(store.nearby_blocks(np.array([[0, -1, -1], [9, .5, 1]]), trace, records=True))
    assert len(blocks) == 1 and blocks[0]['index'] == 2
    np.testing.assert_array_equal(blocks[0]['points'], LINE[2:8])
