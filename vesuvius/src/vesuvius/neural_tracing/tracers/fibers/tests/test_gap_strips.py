"""Piece cutting, chain mirroring, strip splitting and fit patches for store fibers with gap/damaged spans."""
import json

import numpy as np
import pytest
import tifffile

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips import fit_patches, mirror, pieces, retry, spans, split
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    FiberStore, fiber_volume, read_polylines, write_fibers)

SOURCE = '00000000-0000-4000-8000-000000000001'


def tags(*segments):
    return [frozenset(s) for s in (*segments, ())]


def fiber(segment_tags):
    """A straight fiber along x: 6 control points 10 apart, a line point every unit, 5 tail points at both ends."""
    return dict(id=SOURCE, name='f', family='H', reference=None, tags=segment_tags,
                line=np.array([[float(x), 0., 0.] for x in range(-5, 56)]),
                controls=np.array([[10.*k, 0., 0.] for k in range(6)]))


def test_segment_tags_of_annotations():
    annotations = [dict(segment_to_next=dict(tags=['gap'])), None, dict(segment_to_next={}), dict()]
    assert spans.segment_tags(annotations, 4) == [frozenset({'gap'}), frozenset(), frozenset(), frozenset()]
    assert spans.segment_tags(None, 2) == [frozenset(), frozenset()]


def test_control_runs_split_at_excluded_spans():
    assert spans.control_runs(tags((), (), (), (), ())) == [(0, 5)]
    assert spans.control_runs(tags((), ('gap',), (), (), ())) == [(0, 1), (2, 5)]
    assert spans.control_runs(tags(('damaged',), (), ('gap',), ('gap',), ())) == [(1, 2), (4, 5)]
    # a lone control point between two excluded spans is no piece
    assert spans.control_runs(tags((), ('gap',), ('gap',), (), ())) == [(0, 1), (3, 5)]


def test_pieces_have_no_tail_and_no_gap_points():
    first, second = spans.pieces_of(fiber(tags((), ('gap',), (), (), ())), spacing=None)
    assert first['points'][:, 0].tolist() == list(range(0, 11))
    assert second['points'][:, 0].tolist() == list(range(20, 51))
    assert (first['name'], first['controls'], first['pieces']) == ('f#piece0', [0, 1], 2)
    assert first['id'] == spans.derived_id(SOURCE, 'piece0') != second['id']
    dense = spans.pieces_of(fiber(tags((), ('gap',), (), (), ())), spacing=.5)[0]['points']
    assert len(dense) == 21 and dense[[0, -1], 0].tolist() == [0., 10.]


def test_stores_round_trip_and_are_traceable(tmp_path):
    source = tmp_path/'source.vcobj'
    line = np.array([[float(x), 0., 0.] for x in range(0, 101, 5)])
    write_fibers(source, [(dict(name='a', family='V'), line)], volume='volume')
    volume, fibers = spans.read_annotated(source, ['a'])
    assert volume == 'volume' and len(fibers) == 1 and np.allclose(fibers[0]['line'], line)
    assert spans.control_runs(fibers[0]['tags']) == [(0, 1)]
    with pytest.raises(ValueError):
        spans.read_annotated(source, ['missing'])
    mirror.main([str(source), str(tmp_path/'pieces.vcobj'), str(tmp_path/'chains.vcobj'), '--ext', '5'])
    (piece, piece_points), = read_polylines(tmp_path/'pieces.vcobj')
    assert piece['name'] == 'a#piece0' and piece['id'] == spans.derived_id(fibers[0]['id'], 'piece0')
    assert piece_points[[0, -1], 0].tolist() == [0., 100.]
    (chain, chain_points), = read_polylines(tmp_path/'chains.vcobj')
    assert chain_points[[0, -1], 0].tolist() == [-20., 120.]  # 5 patch voxels of 4 volume voxels mirrored out
    entry, = json.loads((tmp_path/'chains.vcobj.joined.json').read_text())
    assert entry['id'] == chain['id'] and entry['pieces'][0]['id'] == piece['id']
    assert entry['ext'] == [[0, 11], [73, 84]] and (entry['pieces'][0]['start'], entry['pieces'][0]['end']) == (12, 72)
    traced = FiberStore(tmp_path/'chains.vcobj', fiber_volume(tmp_path/'chains.vcobj'), 4., split='all')
    assert traced[0].length == pytest.approx(35.)
    retry.main([str(tmp_path/'chains.vcobj'), str(tmp_path/'retry.vcobj'), chain['id']])
    (again, again_points), = read_polylines(tmp_path/'retry.vcobj')
    assert again['id'] == chain['id'] and again_points[[0, -1], 0].tolist() == [0., 120.]
    shifted, = json.loads((tmp_path/'retry.vcobj.joined.json').read_text())
    assert shifted['ext'] == [[61, 72]] and shifted['pieces'][0]['start'] == 0
    pieces.main([str(source), str(tmp_path/'only_pieces.vcobj')])
    assert [row['id'] for row, _ in read_polylines(tmp_path/'only_pieces.vcobj')] == [piece['id']]


def test_extension_mirrors_through_the_end_point():
    line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]])
    ext = mirror.extension(line, 2.)
    assert np.allclose(ext[0], 2*line[-1]-line[1]) and np.allclose(ext[1], 2*line[-1]-line[0])
    # a line shorter than the extension is reflected again and keeps moving outward
    short = np.array([[0., 0., 0.], [1., 0., 0.]])
    assert np.allclose(mirror.extension(short, 3.), [[2., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    assert mirror.extension(short[:1], 3.).shape == (0, 3)


def test_chains_join_close_pieces_only():
    piece = lambda k, x0: dict(id=str(k), piece=k, points=np.array([[x0, 0., 0.], [x0+10., 0., 0.]]))
    chains = mirror.chains_of([piece(0, 0.), piece(1, 15.), piece(2, 100.)], join=5.)
    assert [[p['piece'] for p in chain] for chain in chains] == [[0, 1], [2]]
    points, piece_spans, ext, length = mirror.extended_chain(chains[0], ext=4.)
    assert length == pytest.approx(25.)
    assert [(s['id'], s['start'], s['end']) for s in piece_spans] == [('0', 1, 2), ('1', 3, 4)]
    assert ext == [[0, 0], [5, 5]] and np.allclose(points[[0, -1], 0], [-10., 35.])


def test_split_labels_columns_by_piece_and_drops_joints():
    # chain along x: mirrored copy 0..1, piece A 2..4, piece B 5..7 (after a joint step of 10), mirrored copy 8..9
    points = np.array([[x, 0., 0.] for x in (-2., -1., 0., 1., 2., 12., 13., 14., 15., 16.)])
    entry = dict(pieces=[dict(id='a', piece=0, start=2, end=4), dict(id='b', piece=1, start=5, end=7)],
                 ext=[[0, 1], [8, 9]])
    label_points, labels = split.chain_labels(points, entry)
    xs = np.array([-1.6, 0.4, 1.6, 7., 12.6, 13.9, 15.8])
    x = np.repeat(xs[None], 3, 0)
    y = np.repeat(np.array([[-1.], [0.], [1.]]), len(xs), 1)
    z = np.zeros_like(x)
    z[:, 1] = -1  # a column without valid vertices
    assert split.column_labels(x, y, z, label_points, labels).tolist() == [split.JOINT, -2, 0, split.JOINT, 1, 1,
                                                                         split.JOINT]


def write_strip(path, radius, shape=(30, 40)):
    path.mkdir(parents=True)
    rows, cols = np.indices(shape, dtype=np.float32)
    for name, values in (('x.tif', radius+cols), ('y.tif', np.zeros(shape, np.float32)), ('z.tif', 100+rows)):
        tifffile.imwrite(path/name, values.astype(np.float32))
    (path/'meta.json').write_text(json.dumps(dict(scale=[0.25, 0.2], uuid=path.name)))


def test_split_cuts_a_chain_strip_into_piece_strips(tmp_path):
    line = np.array([[float(x), 0., 400.] for x in range(0, 401, 4)])  # volume voxels: 0..100 patch voxels at z 100
    gap = [None, dict(segment_to_next=dict(tags=['gap'])), None, None]
    source = dict(id=SOURCE, name='f', family='H', reference=None, line=line, tags=spans.segment_tags(gap, 4),
                  controls=np.array([[0., 0., 400.], [160., 0., 400.], [240., 0., 400.], [400., 0., 400.]]))
    first, second = spans.pieces_of(source, spacing=2.)
    (points, piece_spans, ext, _), = [mirror.extended_chain(c, 40.) for c in mirror.chains_of([first, second], 400.)]
    chain_id = spans.derived_id(SOURCE, 'chain0')
    write_fibers(tmp_path/'chains.vcobj', [(dict(id=chain_id, name='f#chain0', family='H'), points)], volume='volume')
    (tmp_path/'chains.vcobj.joined.json').write_text(json.dumps(
        [dict(id=chain_id, name='f#chain0', pieces=piece_spans, ext=ext)]))
    (tmp_path/'pieces.vcobj.pieces.json').write_text(json.dumps(
        [dict(id=p['id'], name=p['name']) for p in (first, second)]))
    run = tmp_path/'run'
    strip = run/'strips'/chain_id[:2]/f'fiber_{chain_id}'
    strip.mkdir(parents=True)
    xs = np.arange(-8., 109., 2.)  # a column every 2 patch voxels, from inside the leading copy to the trailing one
    shape = (5, len(xs))
    for name, values in (('x.tif', np.broadcast_to(xs, shape)), ('y.tif', np.indices(shape)[0]-2.),
                         ('z.tif', np.full(shape, 100.))):
        tifffile.imwrite(strip/name, values.astype(np.float32))
    (strip/'meta.json').write_text(json.dumps(dict(uuid='s', area_vx2=100., area_cm2=1., bbox=None)))
    (strip/'trace.json').write_text(json.dumps(dict(fiber_along='columns', family='H')))
    (run/'run.json').write_text(json.dumps(dict(fibers='chains.vcobj')))
    split.main([str(run), str(tmp_path/'chains.vcobj'), str(tmp_path/'pieces.vcobj'), str(tmp_path/'out')])
    got = {}
    for piece in (first, second):
        folder = tmp_path/'out'/'strips'/piece['id'][:2]/f"fiber_{piece['id']}"
        x = tifffile.imread(folder/'x.tif')
        got[piece['piece']] = (float(x.min()), float(x.max()))
        assert json.loads((folder/'trace.json').read_text())['split']['chain'] == chain_id
    assert got == {0: (0., 40.), 1: (60., 100.)}
    assert json.loads((tmp_path/'out'/'run.json').read_text())['fibers'].endswith('pieces.vcobj')


def test_fit_patches_weights_and_stride(tmp_path):
    run = tmp_path/'run'
    write_strip(run/'strips'/'00'/'fiber_1', radius=1000.)
    write_strip(run/'strips'/'00'/'fiber_2', radius=50.)
    write_strip(run/'strips'/'01'/'fiber_3', radius=1000., shape=(10, 10))  # under --min-nodes
    umbilicus = tmp_path/'umbilicus.json'
    umbilicus.write_text(json.dumps(dict(control_points=[dict(x=0., y=0., z=0.), dict(x=0., y=0., z=500.)])))
    out = tmp_path/'patches'
    fit_patches.main([str(run), str(out), '--umbilicus', str(umbilicus)])
    assert sorted(p.name for p in out.iterdir()) == ['strip-net-1', 'strip-net-2']
    far, core = (json.loads((out/f'strip-net-{k}'/'meta.json').read_text()) for k in (1, 2))
    assert far['loss_weight_scale'] == 1.0 and 'inner_core_radius' not in far
    assert core['loss_weight_scale'] == 0.2 and core['inner_core_radius'] == pytest.approx(69.5)
    assert far['scale'] == [0.125, 0.1] and far['grid_stride'] == 2
    assert tifffile.imread(out/'strip-net-1'/'z.tif').shape == (15, 20)
    with pytest.raises(SystemExit):
        fit_patches.main([str(run), str(out)])
