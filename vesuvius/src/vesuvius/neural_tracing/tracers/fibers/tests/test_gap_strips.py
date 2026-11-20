"""Piece cutting, chain mirroring, strip splitting and fit patches for store fibers with gap/damaged spans."""
import json
import struct

import numpy as np
import pytest
from vc_object_store import Fiber, ObjectStore, object_reference
from vc_reader import images, tifxyz

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips import fit_patches, mirror, pieces, retry, spans, split
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    FiberStore, fiber_catalog, fiber_volume, read_polylines, write_fibers)
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface

SOURCE = '00000000-0000-4000-8000-000000000001'
GAP = dict(segment_to_next=dict(tags=['gap']))


def tags(*segments):
    return [frozenset(s) for s in (*segments, ())]


def fiber(segment_tags, reverse_line=False):
    """A straight fiber along x: 6 control points 10 apart, a line point every unit, 5 tail points at both ends."""
    line = np.array([[float(x), 0., 0.] for x in range(-5, 56)])
    return dict(id=SOURCE, name='f', family='H', reference=None, tags=segment_tags,
                line=line[::-1] if reverse_line else line, controls=np.array([[10.*k, 0., 0.] for k in range(6)]))


def source_store(path, name='f', uuid=SOURCE):
    """A store with one H fiber along x at z 400 (volume voxels): control points at x 0, 160, 240, 400, the span
    160..240 tagged gap, a line point every 4 with tails to -20 and 420."""
    controls = np.array([[x, 0., 400.] for x in (0., 160., 240., 400.)])
    line = np.array([[float(x), 0., 400.] for x in range(-20, 421, 4)])
    record = dict(uuid=uuid, kind='fiber', name=name, format_version=4,
                  fiber=dict(family='H', control_point_count=3, line_point_count=len(line), length=440., bbox=None),
                  meta=dict(coordinates=dict(volume_uuid='volume', scale_factor=1)))
    with ObjectStore.create(str(path), author='test', reader_pool_size=1) as store:
        with store.write(expected_versions={uuid: None}) as tx:
            tx.upsert_object(record)
            tx.put_geometry(uuid, Fiber.from_arrays(np.ascontiguousarray(controls[:3]), line))
        reference = object_reference(store.object(uuid))
        with store.write(expected_versions={uuid: reference}) as tx:  # the last control point, with the annotations
            tx.edit_geometry_sequence(uuid, dict(
                array='control_points', start=3, erase_count=0,
                parts=[dict(count=1, ids=[3], data=struct.pack('<3d', *controls[3]))],
                transformed_annotations=[None, GAP, None, None]), reference['version'])
    return path


def test_segment_tags_of_annotations():
    annotations = [GAP, None, dict(segment_to_next={}), dict()]
    assert spans.segment_tags(annotations, 4) == [frozenset({'gap'}), frozenset(), frozenset(), frozenset()]
    assert spans.segment_tags(None, 2) == [frozenset(), frozenset()]


def test_control_runs_split_at_excluded_spans():
    assert spans.control_runs(tags((), (), (), (), ())) == [(0, 5)]
    assert spans.control_runs(tags((), ('gap',), (), (), ())) == [(0, 1), (2, 5)]
    assert spans.control_runs(tags(('damaged',), (), ('gap',), ('gap',), ())) == [(1, 2), (4, 5)]
    # a lone control point between two excluded spans is no piece
    assert spans.control_runs(tags((), ('gap',), ('gap',), (), ())) == [(0, 1), (3, 5)]


@pytest.mark.parametrize('reverse_line', [False, True])
def test_pieces_have_no_tail_and_no_gap_points_and_run_in_control_order(reverse_line):
    first, second = spans.pieces_of(fiber(tags((), ('gap',), (), (), ()), reverse_line), spacing=None)
    assert first['points'][:, 0].tolist() == list(range(0, 11))
    assert second['points'][:, 0].tolist() == list(range(20, 51))
    assert (first['name'], first['controls'], first['pieces']) == ('f#piece0', [0, 1], 2)
    assert first['id'] == spans.derived_id(SOURCE, 'piece0') != second['id']
    dense = spans.pieces_of(fiber(tags((), ('gap',), (), (), ()), reverse_line), spacing=.5)[0]['points']
    assert len(dense) == 21 and dense[[0, -1], 0].tolist() == [0., 10.]


def test_control_points_out_of_line_order_are_refused():
    crossed = fiber(tags((), (), (), (), ()))
    crossed['controls'] = crossed['controls'][[0, 2, 1, 3, 4, 5]]
    with pytest.raises(ValueError, match='not in the order'):
        spans.pieces_of(crossed, spacing=None)


def test_selection_prefers_uuids_and_refuses_ambiguous_names():
    other = '00000000-0000-4000-8000-000000000002'
    records = [dict(uuid=SOURCE, name='a'), dict(uuid=other, name=SOURCE), dict(uuid=other[:-1]+'3', name='twin'),
               dict(uuid=other[:-1]+'4', name='twin')]
    assert [r['uuid'] for r in spans.select_records(records, [SOURCE])] == [SOURCE]  # not the fiber named like it
    assert [r['uuid'] for r in spans.select_records(records, ['a', SOURCE.upper()])] == [SOURCE]
    with pytest.raises(ValueError, match='Several fibers'):
        spans.select_records(records, ['twin'])
    with pytest.raises(ValueError, match='Not in'):
        spans.select_records(records, ['missing'])


def test_tagged_store_to_pieces_chains_and_retry(tmp_path):
    source = source_store(tmp_path/'source.vcobj')
    volume, (read,) = spans.read_annotated(source, ['f'])
    assert volume == 'volume' and spans.control_runs(read['tags']) == [(0, 1), (2, 3)]
    (tmp_path/'names.txt').write_text('missing\n')
    with pytest.raises(SystemExit, match='Not in'):
        pieces.main([str(source), str(tmp_path/'none.vcobj'), '--only', str(tmp_path/'names.txt')])
    mirror.main([str(source), str(tmp_path/'pieces.vcobj'), str(tmp_path/'chains.vcobj'), '--ext', '10'])
    made = spans.read_derived(tmp_path/'pieces.vcobj', 'piece')
    first, second = (made[spans.derived_id(SOURCE, f'piece{k}')] for k in (0, 1))
    assert first[0]['name'] == 'f#piece0' and first[1][[0, -1], 0].tolist() == [0., 160.]
    assert second[1][[0, -1], 0].tolist() == [240., 400.]
    assert first[0]['gap_strips'] == dict(role='piece', source=SOURCE, source_reference=read['reference'], piece=0,
                                          controls=[0, 1], pieces=2)
    (chain, chain_points), = spans.read_derived(tmp_path/'chains.vcobj', 'chain').values()
    derived = chain['gap_strips']
    assert chain['id'] == spans.derived_id(SOURCE, 'chain0') and derived['level_scale'] == 4.
    assert chain_points[[0, -1], 0].tolist() == [-40., 440.]  # 10 patch voxels of 4 volume voxels mirrored out
    assert [(s['id'], s['reference'], s['start'], s['end']) for s in derived['pieces']] == [
        (first[0]['id'], first[0]['reference'], 20, 100), (second[0]['id'], second[0]['reference'], 101, 181)]
    assert derived['ext'] == [[0, 19], [182, 201]]
    assert chain_points[[100, 101], 0].tolist() == [160., 240.]  # the gap's line points are cut out
    with pytest.raises(ValueError, match='not a gap_strips chain'):
        spans.read_derived(tmp_path/'pieces.vcobj', 'chain')
    traced = FiberStore(tmp_path/'chains.vcobj', fiber_volume(tmp_path/'chains.vcobj'), 4., split='all')
    assert traced[0].length == pytest.approx(120.)
    retry.main([str(tmp_path/'chains.vcobj'), str(tmp_path/'retry.vcobj'), chain['id']])
    (again, again_points), = spans.read_derived(tmp_path/'retry.vcobj', 'chain').values()
    assert again['id'] == chain['id'] and again_points[[0, -1], 0].tolist() == [0., 440.]
    assert again['gap_strips']['ext'] == [[162, 181]] and again['gap_strips']['leading_extension_removed'] == 20
    assert [(s['start'], s['end']) for s in again['gap_strips']['pieces']] == [(0, 80), (81, 161)]
    pieces.main([str(source), str(tmp_path/'only_pieces.vcobj')])
    assert sorted(row['id'] for row, _ in read_polylines(tmp_path/'only_pieces.vcobj')) == sorted(made)


def test_a_selection_without_pieces_is_refused(tmp_path):
    # a fiber without length has no piece
    write_fibers(tmp_path/'source.vcobj', [(dict(name='a', family='V'), np.array([[0., 0., 0.], [1e-300, 0., 0.]]))],
                 volume='volume')
    with pytest.raises(SystemExit, match='No pieces'):
        pieces.main([str(tmp_path/'source.vcobj'), str(tmp_path/'pieces.vcobj'), '--spacing', '2'])
    assert not (tmp_path/'pieces.vcobj').exists()


def test_extension_mirrors_through_the_end_point_to_the_exact_length():
    line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]])
    ext = mirror.extension(line, 2.)
    assert np.allclose(ext[0], 2*line[-1]-line[1]) and mirror.arclen(np.vstack([line[-1:], ext])) == pytest.approx(2.)
    assert np.allclose(mirror.extension(np.array([[0., 0., 0.], [10., 0., 0.]]), 4.), [[14., 0., 0.]])
    # a line shorter than the extension is reflected again and keeps moving outward
    short = np.array([[0., 0., 0.], [1., 0., 0.]])
    assert np.allclose(mirror.extension(short, 3.), [[2., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    assert mirror.extension(short[:1], 3.).shape == (0, 3) and mirror.extension(short, 0.).shape == (0, 3)
    with pytest.raises(ValueError):
        mirror.extension(short, -1.)


def test_chains_join_close_pieces_only():
    piece = lambda k, x0: dict(id=str(k), piece=k, points=np.array([[x0, 0., 0.], [x0+10., 0., 0.]]))
    chains = mirror.chains_of([piece(0, 0.), piece(1, 15.), piece(2, 100.)], join=5.)
    assert [[p['piece'] for p in chain] for chain in chains] == [[0, 1], [2]]
    points, piece_spans, ext, length = mirror.extended_chain(chains[0], ext=4.)
    assert length == pytest.approx(25.)
    assert [(s['id'], s['start'], s['end']) for s in piece_spans] == [('0', 1, 2), ('1', 3, 4)]
    assert ext == [[0, 0], [5, 5]] and points[:, 0].tolist() == [-4., 0., 10., 15., 25., 29.]


def test_columns_beside_a_piece_end_are_dropped():
    # chain along x: mirrored point, piece A 0..1, joint step, piece B 2..3, mirrored point
    points = np.array([[x, 0., 0.] for x in (-.5, 0., 1., 2., 3., 3.5)])
    chain_pieces = [dict(start=1, end=2), dict(start=3, end=4)]
    xs = np.array([-.1, 0., .5, 1., 1.1, 1.9, 2., 3., 3.1, 9.])
    xyz = np.stack([np.repeat(xs[None], 3, 0), np.repeat(np.array([[-1.], [0.], [1.]]), len(xs), 1),
                    np.zeros((3, len(xs)))], -1)
    valid = np.ones((3, len(xs)), bool)
    valid[:, -1] = False  # a column without valid vertices
    labels, offset = split.column_labels(xyz, valid, points, chain_pieces)
    J = split.JOINT
    assert labels.tolist() == [J, 0, 0, 0, J, J, 1, 1, J, split.EMPTY] and offset == 0.
    position, distance = split.chain_positions(points, np.array([[.5, 3., 0.], [2.5, -4., 0.]]))
    assert position.tolist() == [1.5, 3.5] and distance.tolist() == [3., 4.]


def test_a_column_on_a_gap_is_dropped_when_a_later_piece_passes_close_by():
    # the chain leaves along x, turns back and ends 0.05 above the middle of its second segment, the excluded one
    points = np.array([[x, y, 100.] for x, y in ((-2, 0), (-1, 0), (1, 0), (2, 0), (2, 1), (1, 1), (.125, 1),
                                                 (.125, .05), (-.125, .05))], float)
    position, distance = split.chain_positions(points, np.array([[0., 0., 100.]]))
    assert position.tolist() == [1.5] and distance.tolist() == [0.]
    labels, _ = split.column_labels(np.array([[[0., 0., 100.]]]), np.ones((1, 1), bool), points,
                                    [dict(start=0, end=1), dict(start=2, end=8)])
    assert labels.tolist() == [split.JOINT]


def traced_chain_strip(tmp_path, fiber_along):
    """source_store's fiber as stores and a synthetic infer_all run of its chain: a flat strip at z 100 (patch
    voxels) with a column every 2 patch voxels from x -8 to 108 and 5 nodes across."""
    source = source_store(tmp_path/'source.vcobj')
    mirror.main([str(source), str(tmp_path/'pieces.vcobj'), str(tmp_path/'chains.vcobj'), '--ext', '10'])
    chain, = fiber_catalog(tmp_path/'chains.vcobj')
    xs = np.arange(-8., 109., 2.)
    rows = np.stack(np.broadcast_arrays(xs[None], np.arange(-2., 3.)[:, None], 100.), -1)  # (5, C, 3): fiber on columns
    steps = (1., 2.)
    if fiber_along == 'rows':
        rows, steps = np.swapaxes(rows, 0, 1), steps[::-1]
    strip = tmp_path/'run'/'strips'/chain['id'][:2]/f"fiber_{chain['id']}"
    write_surface(strip, rows, np.ones(rows.shape[:2], bool), *steps,
                  dict(fiber=chain['id'], family='H', fiber_along=fiber_along, source_reference=chain['reference']),
                  voxel_size_um=9.6, mmap_layout=True)
    (tmp_path/'run'/'run.json').write_text(json.dumps(dict(fibers=str(tmp_path/'chains.vcobj'))))
    return chain, strip


@pytest.mark.parametrize('fiber_along', ['columns', 'rows'])
def test_split_cuts_a_chain_strip_into_piece_strips(tmp_path, fiber_along):
    chain, strip = traced_chain_strip(tmp_path, fiber_along)
    split.main([str(tmp_path/'run'), str(tmp_path/'chains.vcobj'), str(tmp_path/'pieces.vcobj'), str(tmp_path/'out')])
    made = {row['gap_strips']['piece']: row for row, _ in spans.read_derived(tmp_path/'pieces.vcobj', 'piece').values()}
    for number, (lo, hi) in {0: (0., 40.), 1: (60., 100.)}.items():
        piece = made[number]
        folder = tmp_path/'out'/'strips'/piece['id'][:2]/f"fiber_{piece['id']}"
        surface = tifxyz.read(folder)
        x = np.asarray(surface.xyz)[..., 0][np.asarray(surface.valid)]
        assert (x.min(), x.max()) == (lo, hi) and surface.shape == ((5, 21) if fiber_along == 'columns' else (21, 5))
        assert tuple(surface.scale) == tuple(tifxyz.read(strip).scale)
        area = lambda path: json.loads((path/'meta.json').read_text())['area_vx2']
        assert area(folder) == pytest.approx(area(strip)*40/116)  # measured on the piece: 40 of 116 patch voxels
        trace = json.loads((folder/'trace.json').read_text())
        assert trace['fiber'] == piece['id'] and trace['source_reference'] == piece['reference']
        assert trace['split']['chain'] == chain['id'] and trace['split']['chain_reference'] == chain['reference']
    assert json.loads((tmp_path/'out'/'run.json').read_text())['fibers'].endswith('pieces.vcobj')
    index = [json.loads(line) for line in (tmp_path/'out'/'index'/'shard_0.jsonl').read_text().splitlines()]
    assert sorted(entry['fiber'] for entry in index) == sorted(row['id'] for row in made.values())


def test_split_refuses_a_strip_of_another_chain_version_or_scale(tmp_path):
    chain, strip = traced_chain_strip(tmp_path, 'columns')
    arguments = [str(tmp_path/'run'), str(tmp_path/'chains.vcobj'), str(tmp_path/'pieces.vcobj')]
    trace = json.loads((strip/'trace.json').read_text())
    (strip/'trace.json').write_text(json.dumps(dict(trace, source_reference=dict(chain['reference'], version=99))))
    with pytest.raises(SystemExit, match='not traced on the version'):
        split.main([*arguments, str(tmp_path/'stale')])
    # the same strip against chains made for tracing at 8 volume voxels per patch voxel
    mirror.main([str(tmp_path/'source.vcobj'), str(tmp_path/'pieces8.vcobj'), str(tmp_path/'chains8.vcobj'),
                 '--level-scale', '8'])
    other, = fiber_catalog(tmp_path/'chains8.vcobj')
    (strip/'trace.json').write_text(json.dumps(dict(trace, source_reference=other['reference'])))
    with pytest.raises(SystemExit, match='patch voxels from chain'):
        split.main([arguments[0], str(tmp_path/'chains8.vcobj'), str(tmp_path/'pieces8.vcobj'), str(tmp_path/'scale')])
    with pytest.raises(SystemExit, match='not an infer_all run'):
        split.main([str(tmp_path/'missing'), *arguments[1:], str(tmp_path/'none')])
    # pieces made again are other versions than the chain names
    (strip/'trace.json').write_text(json.dumps(trace))
    pieces.main([str(tmp_path/'source.vcobj'), str(tmp_path/'pieces_again.vcobj')])
    with pytest.raises(SystemExit, match='does not hold piece'):
        split.main([*arguments[:2], str(tmp_path/'pieces_again.vcobj'), str(tmp_path/'again')])
    images.write_tiff(strip/'expected_error.tif', np.zeros((1, 1), np.float32))
    with pytest.raises(SystemExit, match='not on the strip grid'):
        split.main([*arguments, str(tmp_path/'error')])
    assert not any((tmp_path/name).exists() for name in ('stale', 'scale', 'none', 'again', 'error'))


def write_strip(path, shape=(30, 40)):
    rows, cols = np.indices(shape, dtype=np.float64)
    write_surface(path, np.stack([cols*4., rows*5., np.full(shape, 100.)], -1), np.ones(shape, bool), 5., 4.,
                  dict(fiber=path.name), voxel_size_um=9.6, mmap_layout=True)


def test_fit_patches_thin_strips(tmp_path):
    run = tmp_path/'run'
    write_strip(run/'strips'/'00'/'fiber_1')
    write_strip(run/'strips'/'01'/'fiber_2', shape=(10, 10))  # under --min-nodes
    out = tmp_path/'patches'
    fit_patches.main([str(run), str(out)])
    assert [p.name for p in out.iterdir()] == ['strip-net-1']
    patch, strip = tifxyz.read(out/'strip-net-1'), tifxyz.read(run/'strips'/'00'/'fiber_1')
    assert patch.shape == (15, 20) and tuple(patch.scale) == (strip.scale[0]/2, strip.scale[1]/2)
    assert np.array_equal(np.asarray(patch.xyz), np.asarray(strip.xyz)[::2, ::2])
    meta = json.loads((out/'strip-net-1'/'meta.json').read_text())
    assert meta['area_vx2'] == pytest.approx(14*10.*19*8.) and 'loss_weight_scale' not in meta
    assert json.loads((out/'strip-net-1'/'trace.json').read_text())['patch']['stride'] == 2
    with pytest.raises(SystemExit, match='exists'):
        fit_patches.main([str(run), str(out)])
    with pytest.raises(SystemExit, match='no strips'):
        fit_patches.main([str(tmp_path/'missing'), str(tmp_path/'other')])
    images.write_tiff(run/'strips'/'00'/'fiber_1'/'expected_error.tif', np.zeros((1, 1), np.float32))
    with pytest.raises(SystemExit, match='not on the strip grid'):
        fit_patches.main([str(run), str(tmp_path/'other')])
    (run/'strips'/'00'/'fiber_1'/'expected_error.tif').unlink()
    (run/'strips'/'01'/'fiber_2'/'trace.json').write_text('{')
    with pytest.raises(SystemExit, match='no readable trace.json'):
        fit_patches.main([str(run), str(tmp_path/'other')])
    (run/'strips'/'01'/'fiber_2'/'y.tif').unlink()
    with pytest.raises(SystemExit, match='not a readable tifxyz'):
        fit_patches.main([str(run), str(tmp_path/'other')])
    assert not (tmp_path/'other').exists()
