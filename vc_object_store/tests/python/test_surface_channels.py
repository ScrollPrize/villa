"""Native surface channels, bounded access and ordinary revision history."""
import json
import uuid

import numpy as np
import pytest
from vc_object_store import ObjectStore
from object_helpers import edge_choice, patch, sqlite_rows


def array(name, values):
    values = np.asarray(values)
    return dict(name=name, dtype=values.dtype.str, shape=list(values.shape), data=values.tobytes())


def manifest(db, revision):
    return json.loads(sqlite_rows(db.path, "SELECT manifest FROM geometry_revisions WHERE lower(hex(uuid))=?",
                                  (revision.replace('-', ''),))[0][0])


def read(db, revision, name):
    descriptor = next(a for a in db.geometry_revision(revision)['arrays'] if a['name'] == name)
    return np.frombuffer(db.read_geometry_array(revision, name), dtype=descriptor['dtype']).reshape(descriptor['shape'])


def source(tmp_path, channels=(), overlap=None, **tiff_options):
    tifffile = pytest.importorskip('tifffile')
    folder = tmp_path / str(uuid.uuid4())
    folder.mkdir()
    uid = str(uuid.uuid4())
    (folder / 'meta.json').write_text(json.dumps(dict(uuid=uid, name='surface', type='patch', scale=[1, 1])))
    for name in 'xyz':
        tifffile.imwrite(folder / f'{name}.tif', np.ones((35, 37), dtype=np.float32), metadata=None)
    for name, values in channels:
        tifffile.imwrite(folder / f'{name}.tif', values, metadata=None, **tiff_options)
    if overlap is not None:
        (folder / 'overlapping.json').write_text(json.dumps(overlap))
    return uid, folder


@pytest.mark.parametrize('components', [1, 3])
@pytest.mark.parametrize('tiled', [False, True])
def test_import_native_channels_and_overlap(db, tmp_path, components, tiled):
    shape = (35, 37) if components == 1 else (35, 37, components)
    approval = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    generations = np.arange(35*37, dtype=np.uint16).reshape(35, 37) + 64000
    winding = np.linspace(-3, 3, 35*37, dtype=np.float32).reshape(35, 37)
    # Preserve exact NaN payload, negative zero, positive zero, infinities.
    winding.view(np.uint32).flat[:4] = [0x7fc00042, 0x80000000, 0, 0x7f800000]
    tifffile = pytest.importorskip('tifffile')
    uid, folder = source(tmp_path)
    values = {'approval': approval, 'generations': generations, 'winding': winding, 'custom': generations.copy()}
    for name, value in values.items():
        tifffile.imwrite(folder/f'{name}.tif', value, metadata=None,
                         photometric='rgb' if value.ndim == 3 else 'minisblack', tile=(16, 16) if tiled else None)
    overlap = {'overlapping': ['missing-source-id', 'another-id'], 'extra': {'directional': True}}
    (folder/'overlapping.json').write_text(json.dumps(overlap))
    report = db.ingest_geometry_source('patch', folder)
    assert not report['problems'], report
    revision = db.current_geometry(uid)
    assert db.object(uid)['meta']['source']['overlapping.json'] == overlap
    assert db.edges_of(uid) == []
    for name, expected in values.items():
        descriptor = next(a for a in revision['arrays'] if a['name'] == name)
        assert descriptor['dtype'] == expected.dtype.str
        assert descriptor['shape'] == list(expected.shape)
        assert descriptor['components'] == (expected.shape[2] if expected.ndim == 3 else 1)
        assert read(db, revision['uuid'], name).tobytes() == expected.tobytes()
        assert db.read_geometry_rectangle(revision['uuid'], name, 31, 32, 4, 5) == expected[31:35, 32:37].tobytes()
    old = db.current_object_revision(uid)['uuid']
    db.set_meta(uid, 'source', {'overlapping.json': {'overlapping': []}})
    assert db.object_revision(old)['record']['meta']['source']['overlapping.json'] == overlap


@pytest.mark.parametrize('overlap', [None, [], {'overlapping': []}, {'overlapping': ['unresolved']}])
def test_overlap_presence_and_absent_channels(db, tmp_path, overlap):
    uid, folder = source(tmp_path, overlap=overlap)
    assert not db.ingest_geometry_source('patch', folder)['problems']
    assert [a['name'] for a in db.current_geometry(uid)['arrays']] == ['points']
    meta = db.object(uid)['meta']['source']
    assert ('overlapping.json' in meta) == (overlap is not None)
    if overlap is not None:
        assert meta['overlapping.json'] == overlap


@pytest.mark.parametrize('failure', ['shape', 'dtype', 'pages', 'orientation', 'overlap', 'duplicate'])
def test_import_rejects_malformed_channels_without_publication(db, tmp_path, failure):
    tifffile = pytest.importorskip('tifffile')
    uid, folder = source(tmp_path)
    values = np.zeros((35, 37), np.uint16)
    options = dict(metadata=None, photometric='minisblack')
    if failure == 'shape': values = values[:-1]
    if failure == 'dtype': values = values.astype(np.int16)
    if failure == 'pages': values = np.stack([values, values])
    if failure == 'orientation': options['extratags'] = [(274, 'H', 1, 3, False)]
    if failure == 'overlap': (folder/'overlapping.json').write_text('{"overlapping": [4]}')
    tifffile.imwrite(folder/'generations.tif', values, **options)
    if failure == 'duplicate': tifffile.imwrite(folder/'generations.tiff', values, **options)
    report = db.ingest_geometry_source('patch', folder)
    assert report['problems']
    assert db.object(uid) is None


@pytest.mark.parametrize('tile', [0, 32, 64, 128])
@pytest.mark.parametrize('dtype,components', [('|u1', 3), ('<u2', 1), ('<f4', 1), ('<f8', 4)])
def test_channel_lifecycle_rectangles_sparse_history_reopen(db, tile, dtype, components):
    uid = db.upsert_object(dict(kind='patch'))
    initial = db.put_geometry(uid, patch(70, 73), patch_tile_size=tile)
    shape = (70, 73) if components == 1 else (70, 73, components)
    values = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
    added = db.update_surface_channels(uid, [array('review', values)], [], initial)
    assert db.geometry_revision(initial)['arrays'][0]['name'] == 'points'
    assert read(db, added, 'review').tobytes() == values.tobytes()
    assert db.update_surface_channels(uid, [array('review', values)], [], added) == added
    expected = values.copy()
    region = np.full(expected[31:35, 32:37].shape, 17, dtype=dtype)
    edited = db.edit_geometry_rectangle(uid, 'review', 31, 32, 4, 5, region.tobytes(), added)
    expected[31:35, 32:37] = region
    assert read(db, edited, 'review').tobytes() == expected.tobytes()
    assert db.read_geometry_arrays([added, edited], 'review').values.tobytes() == values.tobytes() + expected.tobytes()
    assert db.read_geometry_rectangle(edited, 'review', 31, 32, 4, 5) == region.tobytes()
    assert db.read_geometry_rectangles(edited, 'review', [[31, 32, 4, 5], [0, 0, 0, 0]]) == [region.tobytes(), b'']
    raw = expected.tobytes()
    assert db.read_geometry_array(edited, 'review', 13, 100) == raw[13:113]
    assert db.read_geometry_ranges(edited, 'review', [[13, 100], [4090, 16]]) == [raw[13:113], raw[4090:4106]]
    # Addition following a delta must retain that exact delta-backed channel.
    next_revision = db.update_surface_channels(uid, [array('second', np.zeros((70, 73), np.uint16))], [], edited)
    assert read(db, next_revision, 'review').tobytes() == raw
    replacement = np.zeros((70, 73, 2), np.uint8)
    replaced = db.update_surface_channels(uid, [array('review', replacement)], [], next_revision)
    removed = db.update_surface_channels(uid, [], ['review'], replaced)
    with ObjectStore.open(db.path, read_only=True) as reopened:
        assert read(reopened, added, 'review').tobytes() == values.tobytes()
        assert read(reopened, edited, 'review').tobytes() == raw
        assert read(reopened, next_revision, 'review').tobytes() == raw
        assert read(reopened, replaced, 'review').tobytes() == replacement.tobytes()
        with pytest.raises(ValueError, match='array not found'):
            reopened.read_geometry_array(removed, 'review')
        assert reopened.read_geometry_array(removed, 'points') == patch(70, 73)['arrays'][0]['data']
        packed = reopened.read_geometry_arrays([initial, added, edited, removed], 'points')
        assert packed.values.tobytes() == patch(70, 73)['arrays'][0]['data'] * 4


def test_channel_storage_reuse_and_checked_atomic_relationships(db):
    a = db.upsert_object(dict(kind='patch')); b = db.upsert_object(dict(kind='patch'))
    first = db.put_geometry(a, patch(35, 37)); db.put_geometry(b, patch(2, 2))
    anchor = dict(points=[dict(array='points', index=[0, 0])])
    edge = db.add_edge(dict(kind='connected', a=a, b=b, a_anchor=anchor, b_anchor=anchor))
    added = db.update_surface_channels(a, [array('approval', np.zeros((35, 37), np.uint8))], [], first)
    assert db.relationships_at({a: None, b: None})[0]['applicable']
    refs = sqlite_rows(db.path, 'SELECT array_name FROM geometry_revision_chunks WHERE lower(hex(revision_uuid))=?', [added.replace('-', '')])
    assert refs and all(row == ['approval'] for row in refs)
    assert manifest(db, added)['arrays']['points']['source_revision'] == first
    dependent = db.add_edge(dict(kind='review', a=a, b=b,
                                 a_anchor=dict(**anchor, channels=['approval']), b_anchor=anchor))
    before = {u: db.current_object_revision(u)['uuid'] for u in (a, b)}
    with pytest.raises(RuntimeError, match='explicit retain/update/remove'):
        with db.write(expected_revisions=before) as tx:
            tx.set_meta(a, 'aborted', True)
            tx.edit_geometry_rectangle(a, 'approval', 0, 0, 1, 1, b'\x01', added)
    assert db.current_geometry(a)['uuid'] == added
    assert 'aborted' not in db.object(a)['meta']
    with db.write(expected_revisions=before) as tx:
        edited = tx.edit_geometry_rectangle(a, 'approval', 0, 0, 1, 1, b'\x01', added,
                                            edge_edits=[edge_choice(db, dependent, 'retain')])
        tx.set_meta(a, 'reviewed', True)
    assert all(e['applicable'] for e in db.relationships_at({a: None, b: None}))
    assert db.current_object_revision(a)['geometry_revision'] == edited
    assert [c['array'] for c in manifest(db, edited)['delta']['chunks']] == ['approval']
    assert sqlite_rows(db.path, 'SELECT DISTINCT array_name FROM geometry_current_chunks') == [['approval']]
    with pytest.raises(RuntimeError, match='revision'):
        with db.write(expected_revisions=before) as tx:
            tx.update_surface_channels(a, [], ['approval'], edited)
    assert db.current_geometry(a)['uuid'] == edited
    removed = db.update_surface_channels(a, [], ['approval'], edited,
                                          edge_edits=[edge_choice(db, dependent, 'remove')])
    assert db.edge(edge) is not None
    assert db.relationships_at({a: None, b: None})[0]['applicable']
    assert sqlite_rows(db.path, 'SELECT count(*) FROM geometry_revision_chunks WHERE lower(hex(revision_uuid))=?', [removed.replace('-', '')]) == [[0]]


def test_channel_sync_then_delta(db, tmp_path):
    uid = db.upsert_object(dict(kind='patch'))
    first = db.put_geometry(uid, patch(35, 37))
    added = db.update_surface_channels(uid, [array('gen', np.zeros((35, 37), np.uint16))], [], first)
    changed = db.edit_geometry_rectangle(uid, 'gen', 1, 2, 1, 1, b'\xff\xff', added)
    kept = db.update_surface_channels(uid, [array('other', np.ones((35, 37), np.uint8))], [], changed)
    final = db.edit_geometry_rectangle(uid, 'gen', 3, 4, 1, 1, b'\x01\x00', kept)
    with ObjectStore.create(str(tmp_path/'peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        for rev in [first, added, changed, kept, final]:
            assert peer.read_geometry_array(rev, 'points') == db.read_geometry_array(rev, 'points')
        assert peer.read_geometry_array(final, 'gen') == db.read_geometry_array(final, 'gen')
        assert read(peer, final, 'gen')[1, 2] == 65535
        next_rev = peer.edit_geometry_rectangle(uid, 'gen', 0, 0, 1, 1, b'\x02\x00', final)
        peer.synchronize(db)
        assert read(db, next_rev, 'gen')[0, 0] == 2


def test_channel_layout_errors_and_caught_geometry_conflicts_are_atomic(db):
    uid = db.upsert_object(dict(kind='patch'))
    initial = db.put_geometry(uid, patch(3, 4))
    channel = array('approval', np.zeros((3, 4), np.uint8))
    added = db.update_surface_channels(uid, [channel], [], initial)
    for channels, removed in [([array('wrong', np.zeros((4, 3), np.uint8))], []),
                               ([array('wrong', np.zeros(12, np.uint8))], []),
                               ([channel, channel], []), ([channel], ['approval']),
                               ([], ['points']), ([], ['missing']),
                               ([dict(channel, name='points')], [])]:
        with pytest.raises(ValueError):
            db.update_surface_channels(uid, channels, removed, added)
        assert db.current_geometry(uid)['uuid'] == added
    for coordinates, data in [((0, 0, 1, 1), b''), ((-1, 0, 1, 1), b'\x01'), ((2, 3, 2, 2), b'1234')]:
        with pytest.raises(ValueError):
            db.edit_geometry_rectangle(uid, 'approval', *coordinates, data, added)
    before = db.current_object_revision(uid)['uuid']
    with pytest.raises(RuntimeError, match='conflict|revision'):
        with db.write(expected_revisions={uid: before}) as tx:
            tx.set_meta(uid, 'aborted', True)
            with pytest.raises(RuntimeError, match='revision'):
                tx.update_surface_channels(uid, [], ['approval'], initial)
    assert db.current_object_revision(uid)['uuid'] == before
    assert 'aborted' not in db.object(uid)['meta']


@pytest.mark.parametrize('tiled', [False, True])
def test_separate_tiff_planes_remain_in_sample_order(db, tmp_path, tiled):
    tifffile = pytest.importorskip('tifffile')
    uid, folder = source(tmp_path)
    planes = np.stack([np.full((35, 37), v, np.uint8) for v in [17, 99, 201]])
    tifffile.imwrite(folder/'approval.tif', planes, photometric='rgb', planarconfig='separate',
                     metadata=None, tile=(16, 16) if tiled else None)
    assert not db.ingest_geometry_source('patch', folder)['problems']
    rev = db.current_geometry(uid)['uuid']
    assert read(db, rev, 'approval').tobytes() == planes.transpose(1, 2, 0).tobytes()
