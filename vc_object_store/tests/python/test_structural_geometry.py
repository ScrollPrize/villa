"""Item 5: identity-aware sequence splices and coordinated surface resizing."""
import json
import numpy as np
import pytest
from vc_object_store import Fiber, ObjectStore, PointCollection
from object_helpers import edge_choice, sqlite_rows
from test_surface_channels import array, read, manifest


def tree_leaves(db, root):
    if root is None:
        return []
    node = json.loads(sqlite_rows(db.path, 'SELECT CAST(data AS TEXT) FROM geometry_chunks WHERE hash=?', (root['hash'],))[0][0])
    assert node['summary'] == {k: v for k, v in root.items() if k != 'hash'}
    if 'value' in node:
        return [node['value']]
    assert abs(node['left']['height'] - node['right']['height']) <= 1
    assert root['height'] == 1 + max(node['left']['height'], node['right']['height'])
    return tree_leaves(db, node['left']) + tree_leaves(db, node['right'])


def heads(db, *objects):
    return {uid: db.current_object_revision(uid)['uuid'] for uid in objects}


def sequence(db, uid, start, erase_count, parts=(), array='points', **kwargs):
    old = db.current_geometry(uid)['uuid']
    with db.write(expected_revisions=heads(db, uid)) as tx:
        return tx.edit_geometry_sequence(uid, dict(array=array, start=start, erase_count=erase_count,
                                                 parts=list(parts), **kwargs), old)


def point(id, x, **annotations):
    return dict(count=1, ids=[id], data=np.array([[x, 2, 3]], dtype='<f8').tobytes(),
                annotations=[annotations])


def collection(db):
    uid = db.upsert_object(dict(kind='point_collection', meta={'source': {
        'point_attributes': [{'label': 'A'}, {'label': 'B'}]}}))
    old = db.put_geometry(uid, PointCollection([10, 20], [[1, 2, 3], [4, 5, 6]]))
    return uid, old


def test_sequence_splices_identity_annotations_export_and_history(db, tmp_path):
    uid, original = collection(db)
    appended = sequence(db, uid, 2, 0, [point(30, 7, label='C')])
    inserted = sequence(db, uid, 1, 0, [point(40, 8, label='D')])
    deleted = sequence(db, uid, 2, 1)
    assert read(db, deleted, 'point_ids').tolist() == [10, 40, 30]
    before = db.current_object_revision(uid)
    with pytest.raises(RuntimeError, match='already used'):
        sequence(db, uid, 2, 0, [point(20, 99)])
    assert db.current_object_revision(uid) == before
    # Explicit historical restoration keeps the original ID and annotation.
    restored = sequence(db, uid, 1, 1, [dict(revision=original, start=1, count=1)])
    reversed_ = sequence(db, uid, 0, 3, [dict(revision=restored, start=0, count=3, reverse=True)])
    assert read(db, reversed_, 'point_ids').tolist() == [30, 20, 10]
    assert db.geometry_point_annotations(reversed_, 'points') == [{'label': 'C'}, {'label': 'B'}, {'label': 'A'}]
    reordered = sequence(db, uid, 0, 3, [dict(revision=reversed_, start=i, count=1) for i in (2, 0, 1)])
    assert read(db, reordered, 'point_ids').tolist() == [10, 30, 20]
    destination = tmp_path/'out.json'
    db.export_geometry(uid, destination)
    exported = next(iter(json.loads(destination.read_text())['collections'].values()))['points']
    assert {id: p['label'] for id, p in exported.items()} == {'10': 'A', '20': 'B', '30': 'C'}
    assert read(db, original, 'point_ids').tolist() == [10, 20]
    assert read(db, appended, 'point_ids').tolist() == [10, 20, 30]
    assert read(db, inserted, 'point_ids').tolist() == [10, 40, 20, 30]
    with ObjectStore.open(db.path, author='reader') as reopened:
        assert reopened.geometry_point_annotations(reversed_, 'points')[2] == {'label': 'A'}
        assert read(reopened, reordered, 'point_ids').tolist() == [10, 30, 20]


def test_fiber_ids_and_explicit_span_direction_transform(db):
    uid = db.upsert_object(dict(kind='fiber', meta={'source': {'control_points_attributes': [
        {'label': 'A', 'segment_to_next': {'mode': 'traced'}, 'direction': [1, 0, 0]}, {'label': 'B'}]}}))
    old = db.put_geometry(uid, Fiber([[1, 2, 3], [4, 5, 6]]))
    before = db.current_object_revision(uid)
    with pytest.raises(RuntimeError, match='transformed span/direction'):
        sequence(db, uid, 0, 2, [dict(revision=old, count=2, reverse=True)], array='control_points')
    assert db.current_object_revision(uid) == before
    transformed = [{'label': 'B', 'segment_to_next': {'mode': 'traced'}, 'direction': [-1, 0, 0]}, {'label': 'A'}]
    new = sequence(db, uid, 0, 2, [dict(revision=old, count=2, reverse=True)], array='control_points', transformed_annotations=transformed)
    assert read(db, new, 'control_point_ids').tolist() == [1, 0]
    assert db.geometry_point_annotations(new, 'control_points') == transformed
    with db.geometry_snapshot() as snapshot:
        assert snapshot.fiber(uid).load().control_point_ids == [1, 0]
    # Adjacency changes also require an application decision.
    with pytest.raises(RuntimeError, match='transformed span/direction'):
        sequence(db, uid, 1, 0, [point(2, 7)], array='control_points')
    sequence(db, uid, 0, 0, [dict(count=3, data=np.arange(9, dtype='<f8').tobytes())], array='line_points')


def surface(db, tile=32):
    points = np.indices((65, 67)).transpose(1, 2, 0).astype('<f4')
    points = np.concatenate([points, np.ones((65, 67, 1), '<f4')], axis=2)
    channels = {'points': points, 'generations': np.arange(65*67, dtype='<u2').reshape(65, 67),
                'approval': np.arange(65*67*3, dtype='u1').reshape(65, 67, 3)}
    uid = db.upsert_object(dict(kind='patch'))
    old = db.put_geometry(uid, dict(kind='patch', attributes={'scale': [1, 1]},
                                   arrays=[array(n, a) for n, a in channels.items()]), patch_tile_size=tile)
    fills = {n: np.full((1, 1) + a.shape[2:], 7 if n != 'points' else -1, dtype=a.dtype) for n, a in channels.items()}
    return uid, old, channels, fills


def resized(values, row, col, rows, cols, fill):
    result = np.empty((rows, cols) + values.shape[2:], values.dtype)
    result[...] = fill
    for r in range(rows):
        for c in range(cols):
            if 0 <= r+row < values.shape[0] and 0 <= c+col < values.shape[1]:
                result[r, c] = values[r+row, c+col]
    return result


@pytest.mark.parametrize('tile', [0, 32, 128])
def test_growth_crop_all_sides_channels_ranges_and_sparse_followup(db, tile):
    uid, original, values, fills = surface(db, tile)
    old = original
    for row, col, rows, cols in [(-5, -3, 75, 80), (2, 1, 70, 76), (0, 0, 82, 91), (7, 6, 62, 65)]:
        with db.write(expected_revisions=heads(db, uid)) as tx:
            new = tx.resize_surface(uid, dict(row=row, column=col, rows=rows, columns=cols,
                fill=[array(n, a) for n, a in fills.items()]), old)
        for name in values:
            values[name] = resized(values[name], row, col, rows, cols, fills[name])
            assert read(db, new, name).tobytes() == values[name].tobytes()
            assert db.read_geometry_rectangle(new, name, 30, 31, 3, 4) == values[name][30:33, 31:35].tobytes()
        old = new
    # Existing rectangle and channel APIs continue to operate on resized grids.
    new = db.edit_geometry_rectangle(uid, 'generations', 31, 31, 2, 3,
                                    np.full((2, 3), 65535, '<u2').tobytes(), old)
    values['generations'][31:33, 31:34] = 65535
    assert read(db, new, 'generations').tobytes() == values['generations'].tobytes()
    new = db.update_surface_channels(uid, [array('extra', np.zeros((62, 65), 'u1'))], ['approval'], new)
    assert read(db, new, 'points').tobytes() == values['points'].tobytes()
    with ObjectStore.open(db.path, author='reader') as other:
        assert read(other, original, 'generations').shape == (65, 67)
        assert read(other, new, 'extra').shape == (62, 65)


def test_surface_interior_reuse_and_fill_rollback(db):
    uid, old, values, fills = surface(db)
    original = manifest(db, old)['arrays']['points']['chunks']
    before = db.current_object_revision(uid)
    with pytest.raises(RuntimeError, match='fill'):
        with db.write(expected_revisions=heads(db, uid)) as tx:
            tx.resize_surface(uid, dict(row=0, column=0, rows=66, columns=68, fill=[]), old)
    assert db.current_object_revision(uid) == before
    with db.write(expected_revisions=heads(db, uid)) as tx:
        new = tx.resize_surface(uid, dict(row=0, column=0, rows=66, columns=68,
            fill=[array(n, a) for n, a in fills.items()]), old)
    tiles = tree_leaves(db, manifest(db, new)['arrays']['points']['tree'])
    assert {t['hash'] for t in tiles if t['rows'] == t['columns'] == 32} == set(original[i] for i in [0, 1, 3, 4])


def test_split_merge_retirement_and_missing_source_checks(db):
    source, old = collection(db)
    a = db.upsert_object(dict(kind='point_collection'))
    b = db.upsert_object(dict(kind='point_collection'))
    empty_a = db.put_geometry(a, PointCollection([], []))
    empty_b = db.put_geometry(b, PointCollection([], []))
    edit = dict(array='points', start=0, erase_count=0, parts=[dict(revision=old, start=0, count=1)])
    with pytest.raises(RuntimeError, match='expected validation endpoint'):
        with db.write(expected_revisions=heads(db, a)) as tx:
            tx.edit_geometry_sequence(a, edit, empty_a)
    assert db.current_geometry(a)['uuid'] == empty_a
    with db.write(expected_revisions=heads(db, source, a, b)) as tx:
        ra = tx.edit_geometry_sequence(a, edit, empty_a)
        edit['parts'][0]['start'] = 1
        rb = tx.edit_geometry_sequence(b, edit, empty_b)
        tx.retire_object(source)
    assert db.object(source)['retired']
    assert read(db, ra, 'point_ids').tolist() == [10]
    assert read(db, rb, 'point_ids').tolist() == [20]
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        merged = tx.edit_geometry_sequence(a, dict(array='points', start=1, erase_count=0,
            parts=[dict(revision=rb, count=1)]), ra)
        tx.retire_object(b)
    assert read(db, merged, 'point_ids').tolist() == [10, 20]
    assert db.geometry_point_annotations(merged, 'points') == [{'label': 'A'}, {'label': 'B'}]


def test_relationship_decisions_and_checked_whole_action_rollback(db):
    uid, old = collection(db)
    peer = db.upsert_object(dict(kind='note'))
    edge = db.add_edge(dict(kind='connected', a=uid, b=peer,
        a_anchor={'points': [{'array': 'point_ids', 'index': [20]}]}, b_anchor={'whole_geometry': True}))
    before = heads(db, uid, peer)
    edit = dict(array='points', start=1, erase_count=1)
    with pytest.raises(RuntimeError, match='explicit'):
        with db.write(expected_revisions=before) as tx:
            tx.set_meta(peer, 'must_rollback', True)
            tx.edit_geometry_sequence(uid, edit, old)
    assert heads(db, uid, peer) == before
    decision = edge_choice(db, edge, 'keep_recorded')
    with db.write(expected_revisions=before) as tx:
        new = tx.edit_geometry_sequence(uid, edit, old, [decision])
    assert read(db, new, 'point_ids').tolist() == [10]
    assert db.recorded_relationships(before[uid])[0]['record']['uuid'] == edge
    assert db.read_geometry_array(old, 'points')
    with pytest.raises(RuntimeError, match='expected revision conflict'):
        with db.write(expected_revisions=before):
            pass


def test_long_history_compaction_and_sync(db, tmp_path):
    uid, first = collection(db)
    expected = [[1, 2, 3], [4, 5, 6]]
    for i in range(100):
        sequence(db, uid, 1, 0, [point(100+i, i)])
        expected.insert(1, [i, 2, 3])
    last = db.current_geometry(uid)['uuid']
    assert read(db, last, 'points').tolist() == expected
    assert len(tree_leaves(db, manifest(db, last)['arrays']['points']['tree'])) <= 2
    assert db.read_geometry_array(last, 'points', 24, 48) == np.array(expected[1:3], '<f8').tobytes()
    with ObjectStore.create(str(tmp_path/'peer.vcobj'), author='peer') as peer:
        assert db.synchronize(peer)['applied']
        assert read(peer, last, 'points').tolist() == expected
        assert read(peer, first, 'point_ids').tolist() == [10, 20]
        assert peer.geometry_point_annotations(last, 'points') == db.geometry_point_annotations(last, 'points')


def test_annotation_swap_reproduction_is_rejected(db):
    uid, old = collection(db)
    with pytest.raises(ValueError, match='structural'):
        db.edit_geometry(uid, [dict(array='point_ids', element_offset=0,
                                   data=np.array([20, 10], '<i8').tobytes())], old)
    with pytest.raises(ValueError, match='structural'):
        db.put_geometry(uid, PointCollection([20, 10], [[4, 5, 6], [1, 2, 3]]), old)
    assert db.current_geometry(uid)['uuid'] == old


def test_surface_anchor_mapping_requires_decision_and_updates_atomically(db):
    uid, old, values, fills = surface(db)
    peer = db.upsert_object(dict(kind='note'))
    edge = db.add_edge(dict(kind='connected', a=uid, b=peer,
        a_anchor={'points': [{'array': 'points', 'index': [10, 11]}]}, b_anchor={'whole_geometry': True}))
    before = heads(db, uid, peer)
    resize = dict(row=-3, column=-5, rows=68, columns=72, fill=[array(n, a) for n, a in fills.items()])
    with pytest.raises(RuntimeError, match='explicit'):
        with db.write(expected_revisions=before) as tx:
            tx.resize_surface(uid, resize, old)
    edge_record = db.edge(edge)
    anchors = {side+'_anchor': edge_record[side+'_anchor'] for side in ('a', 'b')}
    side = 'a' if edge_record['a'] == uid else 'b'
    anchors[side+'_anchor'] = {'points': [{'array': 'points', 'index': [13, 16]}]}
    decision = edge_choice(db, edge, 'update', **anchors)
    with db.write(expected_revisions=before) as tx:
        new = tx.resize_surface(uid, resize, old, [decision])
    assert db.geometry_revision(new)['grid_origin'] == [-3, -5]
    assert db.read_geometry_rectangle(new, 'points', 13, 16, 1, 1) == values['points'][10:11, 11:12].tobytes()
    assert db.relationships_at(heads(db, uid, peer))[0]['applicable']
    assert db.relationships_at(before)[0]['applicable']


def test_caught_invalid_edit_poisoning_and_restore_id_ledger(db):
    uid, old = collection(db)
    original = db.current_object_revision(uid)['uuid']
    sequence(db, uid, 2, 0, [point(30, 9)])
    sequence(db, uid, 2, 1)
    with db.write(expected_revisions=heads(db, uid)) as tx:
        tx.restore_object(uid, source_revision=original)
    assert db.current_geometry(uid)['uuid'] == old
    before = db.current_object_revision(uid)
    with pytest.raises(RuntimeError, match='conflict'):
        with db.write(expected_revisions=heads(db, uid)) as tx:
            tx.set_meta(uid, 'rollback', True)
            with pytest.raises(RuntimeError, match='already used'):
                tx.edit_geometry_sequence(uid, dict(array='points', start=2, erase_count=0, parts=[point(30, 99)]), old)
    assert db.current_object_revision(uid) == before


def test_empty_grids_and_sequences_regrow_with_explicit_new_values(db):
    uid, old, values, fills = surface(db)
    with db.write(expected_revisions=heads(db, uid)) as tx:
        empty = tx.resize_surface(uid, dict(row=2, column=3, rows=0, columns=0,
            fill=[array(n, a) for n, a in fills.items()]), old)
    with db.write(expected_revisions=heads(db, uid)) as tx:
        new = tx.resize_surface(uid, dict(row=-2, column=-3, rows=2, columns=2,
            fill=[array(n, a) for n, a in fills.items()]), empty)
    for name, fill in fills.items():
        expected = np.empty((2, 2) + fill.shape[2:], dtype=fill.dtype)
        expected[...] = fill
        assert read(db, new, name).tobytes() == expected.tobytes()
    uid, old = collection(db)
    empty = sequence(db, uid, 0, 2)
    assert read(db, empty, 'point_ids').size == 0
    new = sequence(db, uid, 0, 0, [point(50, 1)])
    assert read(db, new, 'point_ids').tolist() == [50]
    assert db.edit_geometry(uid, [dict(array='points', element_offset=0, data=np.array([1.], '<f8').tobytes())], new) == new


def test_fiber_stable_vs_positional_identity_with_equal_coordinates(db):
    uid = db.upsert_object(dict(kind='fiber'))
    db.put_geometry(uid, Fiber([[1, 2, 3], [1, 2, 3]]))
    old = sequence(db, uid, 0, 0, array='control_points')
    stable_peer, positional_peer = [db.upsert_object(dict(kind='note')) for _ in range(2)]
    stable = db.add_edge(dict(kind='connected', a=uid, b=stable_peer,
        a_anchor={'points': [{'array': 'control_point_ids', 'index': [0]}]}, b_anchor={'whole_geometry': True}))
    positional = db.add_edge(dict(kind='connected', a=uid, b=positional_peer,
        a_anchor={'points': [{'array': 'control_points', 'index': [0]}]}, b_anchor={'whole_geometry': True}))
    edit = dict(array='control_points', start=0, erase_count=2, parts=[dict(revision=old, count=2, reverse=True)])
    before = heads(db, uid, stable_peer, positional_peer)
    with pytest.raises(RuntimeError, match='explicit'):
        with db.write(expected_revisions=before) as tx:
            tx.edit_geometry_sequence(uid, edit, old)
    with db.write(expected_revisions=before) as tx:
        new = tx.edit_geometry_sequence(uid, edit, old, [edge_choice(db, positional, 'keep_recorded')])
    applicability = {r['record']['uuid']: r['applicable'] for r in db.relationships_at(heads(db, uid, stable_peer, positional_peer))}
    assert applicability[stable]
    assert not applicability[positional]
    assert read(db, new, 'control_point_ids').tolist() == [1, 0]


def test_indexed_large_annotations_and_sample_attributes_survive_sync(db, tmp_path):
    # Larger than a payload chunk: annotation size must not inflate ID buckets.
    label = 'annotation-' * 100_000
    uid, old = collection(db)
    revision = sequence(db, uid, 2, 0, [point(30, 7, label=label)])
    assert db.geometry_point_annotations(revision, 'points')[-1] == {'label': label}
    fiber = db.upsert_object(dict(kind='fiber'))
    db.put_geometry(fiber, Fiber([[0, 0, 0]], line_points=[[1, 2, 3]]))
    samples = sequence(db, fiber, 1, 0, [dict(count=1, data=np.array([[4, 5, 6]], '<f8').tobytes(),
        annotations=[{'label': label}])], array='line_points')
    with ObjectStore.create(str(tmp_path/'annotation-peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        assert peer.geometry_point_annotations(revision, 'points')[-1] == {'label': label}
        assert peer.geometry_point_annotations(samples, 'line_points')[-1] == {'label': label}


def test_indexed_random_splices_and_sparse_id_dependency(db):
    rng = np.random.default_rng(15)
    count = 5000
    ids = list(range(count))
    xyz = np.column_stack((np.arange(count), np.ones(count), np.ones(count))).astype('<f8')
    uid = db.upsert_object(dict(kind='point_collection'))
    original = db.put_geometry(uid, PointCollection(ids, xyz.tolist()))
    revision = sequence(db, uid, 0, 0)
    retained = {}
    for new_id in range(count, count+120):
        start = int(rng.integers(0, len(ids)+1))
        erase = min(int(rng.integers(0, 5)), len(ids)-start)
        revision = sequence(db, uid, start, erase, [point(new_id, -new_id, label=str(new_id))])
        ids[start:start+erase] = [new_id]
        xyz = np.concatenate((xyz[:start], np.array([[-new_id, 2, 3]], '<f8'), xyz[start+erase:]))
        if new_id % 30 == 0:
            retained[revision] = xyz[19:21].tobytes()
    for historical, expected_bytes in retained.items():
        assert db.read_geometry_array(historical, 'points', 19*24, 2*24) == expected_bytes
    assert read(db, revision, 'point_ids').tolist() == ids
    assert read(db, revision, 'points').tobytes() == xyz.tobytes()
    leaves = tree_leaves(db, manifest(db, revision)['arrays']['points']['tree'])
    assert len(leaves) <= 2 * (xyz.nbytes // 4096 + 1)
    assert db.read_geometry_array(original, 'point_ids', 3999*8, 8) == np.array([3999], '<i8').tobytes()
    peer = db.upsert_object(dict(kind='note'))
    edge = db.add_edge(dict(kind='connected', a=uid, b=peer,
        a_anchor={'points': [{'array': 'point_ids', 'index': [ids[0]]}]}, b_anchor={'whole_geometry': True}))
    preview = db.affected_geometry_edit_edges(uid, [dict(array='points', element_offset=0,
        data=np.array([-999.], '<f8').tobytes())], revision)
    assert [e['uuid'] for e in preview] == [edge]
    with pytest.raises(RuntimeError, match='explicit'):
        with db.write(expected_revisions=heads(db, uid, peer)) as tx:
            tx.edit_geometry(uid, [dict(array='points', element_offset=0, data=np.array([-999.], '<f8').tobytes())], revision)
    with db.write(expected_revisions=heads(db, uid, peer)) as tx:
        changed = tx.edit_geometry(uid, [dict(array='points', element_offset=0, data=np.array([-999.], '<f8').tobytes())],
                                   revision, [edge_choice(db, edge, 'keep_recorded')])
    assert db.read_geometry_array(changed, 'points', 0, 8) == np.array([-999.], '<f8').tobytes()
    assert db.geometry_point_annotations(changed, 'points') == db.geometry_point_annotations(revision, 'points')


@pytest.mark.parametrize('cache', [0, 1024*1024])
def test_indexed_spatial_tree_covers_seams_and_paginates(db, cache):
    rows, cols = 64, 96
    y, x = np.indices((rows, cols))
    xyz = np.stack((x, y, np.ones_like(x)), axis=-1).astype('<f4')
    uid = db.upsert_object(dict(kind='patch'))
    old = db.put_geometry(uid, dict(kind='patch', attributes={'scale': [1, 1]}, arrays=[array('points', xyz)]), patch_tile_size=32)
    with db.write(expected_revisions=heads(db, uid)) as tx:
        current = tx.resize_surface(uid, dict(row=-3, column=-5, rows=rows+3, columns=cols+6,
            fill=[array('points', np.full((1, 1, 3), -1, '<f4'))]), old)
    with ObjectStore.open(db.path, read_only=True, spatial_cache_bytes=cache) as reader:
        box = [31.5, 31.5, 1, 31.5, 31.5, 1]
        regions = reader.spatial_regions(box, object_uuids=[uid])
        assert regions
        assert all(r['array_name'] == 'points' and r['revision_uuid'] == current for r in regions)
        assert any(r['start'][0] <= 34 and r['start'][0]+r['shape'][0] > 35 and
                   r['start'][1] <= 36 and r['start'][1]+r['shape'][1] > 37 for r in regions)
        assert reader.spatial_regions(box, object_uuids=[uid], limit=1, offset=1) == regions[1:2]
    # A line segment crossing two leaves must still be a candidate in the gap.
    from object_helpers import track
    line = db.upsert_object(dict(kind='track'))
    coords = [[0, 0, 1]]*5461 + [[100, 0, 1]]*5461
    db.put_geometry(line, track(coords))
    sequence(db, line, 0, 0)
    rs = db.spatial_regions([49, -1, 0, 51, 1, 2], object_uuids=[line])
    assert any(r['start'][0] <= 5460 and r['start'][0]+r['shape'][0] > 5461 for r in rs)


def test_v10_migration_builds_dependency_index_without_new_history(tmp_path):
    import sqlite3
    path = str(tmp_path/'v10.vcobj')
    with ObjectStore.create(path, author='before') as db:
        uid, old = collection(db)
        peer = db.upsert_object(dict(kind='note'))
        edge = db.add_edge(dict(kind='connected', a=uid, b=peer,
            a_anchor={'points': [{'array': 'point_ids', 'index': [10]}]}, b_anchor={'whole_geometry': True}))
        before = heads(db, uid, peer)
    with sqlite3.connect(path) as sql:
        sql.execute('DROP TRIGGER geometry_dependencies_insert')
        sql.execute('DROP TRIGGER geometry_dependencies_update')
        sql.execute('DROP TABLE geometry_edge_dependencies')
        sql.execute('DROP INDEX geometry_revision_generation')
        sql.execute('PRAGMA user_version=10')
    with ObjectStore.open(path, read_only=True) as reader:
        assert read(reader, old, 'point_ids').tolist() == [10, 20]
    with ObjectStore.open(path, author='after') as db:
        assert heads(db, uid, peer) == before
        with pytest.raises(RuntimeError, match='explicit'):
            sequence(db, uid, 0, 1)
        with db.write(expected_revisions=heads(db, uid, peer)) as tx:
            tx.edit_geometry_sequence(uid, dict(array='points', start=0, erase_count=1, parts=[]), old,
                                      [edge_choice(db, edge, 'keep_recorded')])
        assert not db.relationships_at(heads(db, uid, peer))[0]['applicable']


def test_sample_edit_preserves_explicit_control_ids_before_control_conversion(db):
    fiber = Fiber([[1, 2, 3], [4, 5, 6]], line_points=[[1, 2, 3]])
    fiber.control_point_ids = [40, 50]
    uid = db.upsert_object(dict(kind='fiber'))
    db.put_geometry(uid, fiber)
    peer = db.upsert_object(dict(kind='note'))
    edge = db.add_edge(dict(kind='connected', a=uid, b=peer,
        a_anchor={'points': [{'array': 'control_point_ids', 'index': [40]}]}, b_anchor={'whole_geometry': True}))
    revision = sequence(db, uid, 1, 0, [dict(count=1, data=np.array([[7, 8, 9]], '<f8').tobytes())], array='line_points')
    assert read(db, revision, 'control_point_ids').tolist() == [40, 50]
    assert not db.edge_geometry_state(edge)['stale']
    assert db.relationships_at(heads(db, uid, peer))[0]['applicable']
