"""Range, rectangle and packed reads preserve bytes, revisions and ownership."""
import gc
import hashlib
import json
import random
import sqlite3
import struct
import uuid

import numpy as np
import pytest
from vc.objects import ObjectStore
from object_helpers import edge_choice, patch, sqlite_rows, track


def crop(payload, cols, y, x, h, w):
    return b''.join(payload[(r*cols+x)*12:(r*cols+x+w)*12] for r in range(y, y+h))


def manifest(db, rev):
    return json.loads(sqlite_rows(db.path, 'SELECT manifest FROM geometry_revisions WHERE lower(hex(uuid))=?',
                                 (rev.replace('-', ''),))[0][0])


@pytest.mark.parametrize('tile', [0, 32, 64, 128])
def test_tiled_full_ranges_rectangles_and_special_float_bits(db, tile):
    uid = db.upsert_object(dict(kind='patch'))
    data = patch(129, 141)
    raw = bytearray(data['arrays'][0]['data'])
    # Preserve NaN payload, signed zero and infinity bits, without float conversions.
    raw[12:28] = struct.pack('<4I', 0x7fc12345, 0x80000000, 0x7f800000, 0xff800000)
    payload = data['arrays'][0]['data'] = bytes(raw)
    rev = db.put_geometry(uid, data, patch_tile_size=tile)
    assert db.read_geometry_array(rev, 'points') == payload
    rng = random.Random(41)
    ranges = [(0, 0), (len(payload), 0), (0, len(payload)), (31*141*12+383, 3), (63*141*12+767, 2345)]
    for _ in range(70):
        offset = rng.randrange(len(payload)+1)
        count = rng.randrange(min(11000, len(payload)-offset)+1)
        ranges.append((offset, count))
    for offset, count in ranges:
        assert db.read_geometry_array(rev, 'points', offset, count) == payload[offset:offset+count]
    assert db.read_geometry_ranges(rev, 'points', ranges) == [payload[o:o+c] for o, c in ranges]
    rectangles = [(0, 0, 129, 141), (31, 63, 33, 33), (128, 140, 1, 1), (129, 141, 0, 0), (0, 141, 129, 0)]
    for _ in range(70):
        y, x = rng.randrange(130), rng.randrange(142)
        rectangles.append((y, x, rng.randrange(130-y), rng.randrange(142-x)))
    for y, x, h, w in rectangles:
        assert db.read_geometry_rectangle(rev, 'points', y, x, h, w) == crop(payload, 141, y, x, h, w)
    assert db.read_geometry_rectangles(rev, 'points', rectangles) == [crop(payload, 141, *r) for r in rectangles]
    assert db.read_geometry_rectangles(rev, 'points', []) == []
    for args in [(-1, 0, 1, 1), (0, 0, -1, 1), (128, 0, 2, 1), (0, 141, 1, 1), (130, 0, 0, 0)]:
        with pytest.raises(IndexError, match='invalid rectangle'):
            db.read_geometry_rectangle(rev, 'points', *args)
        with pytest.raises(IndexError, match='invalid rectangle'):
            db.read_geometry_rectangles(rev, 'points', [(0, 0, 1, 1), args])
    with pytest.raises(ValueError, match='array not found'):
        db.read_geometry_rectangle(rev, 'missing', 0, 0, 1, 1)
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.read_geometry_rectangle(rev, 'points', 31, 63, 33, 33) == crop(payload, 141, 31, 63, 33, 33)


@pytest.mark.parametrize('budget', [0, 1, 64*1024, 16*1024*1024])
def test_rectangle_cache_budgets_and_concurrent_readers(tmp_path, budget):
    from concurrent.futures import ThreadPoolExecutor

    path = str(tmp_path / 'cache.vcobj')
    with ObjectStore.create(path, author='test', geometry_cache_bytes=budget) as db:
        revisions = []
        for n in range(6):
            data = patch(65, 129)
            raw = bytearray(data['arrays'][0]['data'])
            raw[:12] = struct.pack('<3f', n, 0, 1)
            data['arrays'][0]['data'] = bytes(raw)
            uid = db.upsert_object(dict(kind='patch'))
            revisions.append((db.put_geometry(uid, data), bytes(raw)))
        rectangles = [(0, 0, 33, 33), (32, 96, 33, 33), (0, 0, 1, 1)]

        def read(i):
            rev, raw = revisions[i % len(revisions)]
            assert db.read_geometry_rectangles(rev, 'points', rectangles) == [crop(raw, 129, *r) for r in rectangles]
        # Repeated reads across revisions, budget pressure and concurrent leases.
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(read, range(120)))
        # Another connection publishes; already cached old revisions remain exact.
        with ObjectStore.open(path, author='other', geometry_cache_bytes=budget) as other:
            uid = other.upsert_object(dict(kind='patch'))
            rev = other.put_geometry(uid, patch(2, 2))
        assert db.read_geometry_rectangle(rev, 'points', 0, 0, 2, 2) == patch(2, 2)['arrays'][0]['data']
        for i in range(len(revisions)):
            read(i)


@pytest.mark.parametrize('shape', [(0, 0), (0, 100), (100, 0), (1, 1), (1, 129), (129, 1), (64, 64)])
def test_empty_degenerate_and_exact_chunk_edges(db, shape):
    uid = db.upsert_object(dict(kind='patch'))
    data = patch(*shape)
    rev = db.put_geometry(uid, data)
    assert db.read_geometry_array(rev, 'points') == data['arrays'][0]['data']
    assert db.read_geometry_rectangle(rev, 'points', 0, 0, *shape) == data['arrays'][0]['data']


def test_legacy_upgrade_noop_and_changed_geometry_uses_new_layout(db, tmp_path):
    uid = db.upsert_object(dict(kind='patch'))
    data = patch(129, 141)
    old = db.put_geometry(uid, data, patch_tile_size=0)
    other = db.upsert_object(dict(kind='note'))
    edge = db.add_edge(dict(kind='linked', a=uid, b=other, a_anchor={'whole_geometry': True}, b_anchor={'whole_geometry': True}))
    before = db.export_json(), db.journal_since()
    assert db.affected_geometry_edges(uid, data) == []
    assert db.put_geometry(uid, data, expected_revision=old) == old
    assert (db.export_json(), db.journal_since()) == before
    assert 'storage' not in manifest(db, old)['arrays']['points']
    payload = data['arrays'][0]['data']
    data['arrays'][0]['data'] = struct.pack('<f', 12) + payload[4:]
    new = db.put_geometry(uid, data, expected_revision=old, edge_edits=[edge_choice(db, edge, 'retain')])
    assert manifest(db, new)['arrays']['points']['storage']['tile_size'] == 128
    assert db.read_geometry_array(old, 'points') == payload
    assert db.read_geometry_array(new, 'points') == data['arrays'][0]['data']
    assert db.put_geometry(uid, data, expected_revision=new, patch_tile_size=32) == new
    assert db.affected_geometry_edges(uid, data) == []
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        assert peer.read_geometry_array(old, 'points') == payload
        assert peer.read_geometry_array(new, 'points') == data['arrays'][0]['data']
        assert peer.spatial_regions([-100]*3+[1000]*3) == db.spatial_regions([-100]*3+[1000]*3)
        assert not peer.edge_geometry_state(edge)['stale']


def test_only_one_physical_tile_changes_and_rollback_is_atomic(db, tmp_path):
    uid = db.upsert_object(dict(kind='patch'))
    data = patch(257, 259)
    old = db.put_geometry(uid, data)
    before = manifest(db, old)['arrays']['points']['chunks']
    raw = data['arrays'][0]['data']
    offset = (70*259+70)*12+8
    data['arrays'][0]['data'] = raw[:offset]+struct.pack('<f', 99)+raw[offset+4:]
    with pytest.raises(RuntimeError, match='rollback'):
        with db.write() as tx:
            tx.put_geometry(uid, data, expected_revision=old)
            raise RuntimeError('rollback')
    assert db.current_geometry(uid)['uuid'] == old
    new = db.put_geometry(uid, data, expected_revision=old)
    after = manifest(db, new)['arrays']['points']['chunks']
    changed = [b for a, b in zip(before, after) if a != b]
    assert len(changed) == 1
    assert sqlite_rows(db.path, 'SELECT length(data) FROM geometry_chunks WHERE hash=?', (changed[0],))[0][0] == 128*128*12
    sqlite_rows(db.path, "CREATE TRIGGER fail_after_bounds BEFORE UPDATE ON objects BEGIN SELECT RAISE(ABORT,'injected'); END")
    data['arrays'][0]['data'] = raw
    with db.write() as tx:
        with pytest.raises(RuntimeError, match='injected'):
            tx.put_geometry(uid, data, expected_revision=new)
    assert db.current_geometry(uid)['uuid'] == new
    assert db.read_geometry_array(old, 'points') == raw


def test_corrupt_tiles_are_checked_on_each_operation_and_sync(db, tmp_path):
    uid = db.upsert_object(dict(kind='patch'))
    rev = db.put_geometry(uid, patch(129, 129))
    db.read_geometry_rectangle(rev, 'points', 0, 0, 2, 2)
    chunk = manifest(db, rev)['arrays']['points']['chunks'][0]
    sqlite_rows(db.path, 'DROP TRIGGER geometry_chunks_immutable_update')
    sqlite_rows(db.path, 'UPDATE geometry_chunks SET data=zeroblob(length(data)) WHERE hash=?', (chunk,))
    for read in [lambda: db.read_geometry_array(rev, 'points'), lambda: db.read_geometry_rectangle(rev, 'points', 0, 0, 2, 2)]:
        with pytest.raises(RuntimeError, match='corrupt payload'):
            read()
    # Unrelated tiles remain independently readable.
    assert db.read_geometry_rectangle(rev, 'points', 128, 128, 1, 1) == struct.pack('<3f', 128, 128, 1)
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        with pytest.raises(RuntimeError, match='corrupt geometry chunk'):
            db.synchronize(peer)
        assert peer.object(uid) is None


def test_unknown_layout_rejected_and_invalid_writer_options_leave_no_state(db):
    uid = db.upsert_object(dict(kind='patch'))
    for size in (-1, 1, 65, 256):
        with pytest.raises(ValueError, match='patch tile size'):
            db.put_geometry(uid, patch(2, 2), patch_tile_size=size)
        assert db.current_geometry(uid) is None
    rev = db.put_geometry(uid, patch(2, 2))
    db.read_geometry_rectangle(rev, 'points', 0, 0, 2, 2)
    m = manifest(db, rev)
    m['arrays']['points']['storage']['version'] = 999
    text = json.dumps(m, sort_keys=True, separators=(',', ':'))
    sqlite_rows(db.path, 'DROP TRIGGER geometry_revisions_immutable_update')
    sqlite_rows(db.path, 'UPDATE geometry_revisions SET manifest=?,content_hash=?', (text, hashlib.sha256(text.encode()).hexdigest()))
    with pytest.raises(RuntimeError, match='unsupported storage layout'):
        db.read_geometry_array(rev, 'points')
    with pytest.raises(RuntimeError, match='unsupported storage layout'):
        db.read_geometry_rectangle(rev, 'points', 0, 0, 2, 2)


def test_batched_geometry_ranges_preserve_integrity(db):
    uid = db.upsert_object(dict(kind="fiber"))
    payload = struct.pack("<3d", 1, 2, 3) * 100000
    revision = db.put_geometry(uid, dict(kind="fiber", arrays=[
        dict(name="control_points", dtype="<f8", shape=[0, 3], data=b""),
        dict(name="line_points", dtype="<f8", shape=[100000, 3], data=payload)]))
    ranges = [(1048570, 30), (0, 24), (24, 24), (len(payload), 0), (0, 24)]
    assert db.read_geometry_ranges(revision, "line_points", ranges) == [payload[o:o+n] for o, n in ranges]
    for ranges in ([(-1, 2)], [(0, -1)], [(len(payload), 1)]):
        with pytest.raises(IndexError):
            db.read_geometry_ranges(revision, "line_points", ranges)
    # A later operation must reverify bytes, even on the same pooled connection.
    with sqlite3.connect(db.path) as sql:
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    with pytest.raises(RuntimeError, match="corrupt payload"):
        db.read_geometry_ranges(revision, "line_points", [(0, 24)])


def test_packed_tracks_across_query_windows_and_array_ownership(db):
    revisions, payloads, lengths = [], [], []
    with db.write() as tx:
        for i in range(271):
            points = [[i, j, -j] for j in range(i % 7)]
            data = track(points)
            obj = tx.upsert_object(dict(kind='track'))
            revisions.append(tx.put_geometry(obj, data))
            payloads.append(data['arrays'][0]['data'])
            lengths.append(len(points))
    order = list(reversed(range(271))) + [0, 2, 270]
    packed = db.read_geometry_arrays([revisions[i] for i in order], 'points')
    expected = b''.join(payloads[i] for i in order)
    assert packed.dtype == '<i4'
    assert packed.shape == [sum(lengths[i] for i in order), 3]
    assert packed.data.tobytes() == expected
    assert packed.values.tobytes() == expected
    assert packed.values.dtype == np.dtype('<i4')
    offsets, values = packed.offsets, packed.values
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum([lengths[i] for i in order])])
    assert np.shares_memory(values, packed.data)
    assert not values.flags.writeable and not offsets.flags.writeable
    with pytest.raises(ValueError):
        values.setflags(write=True)
    del packed
    db.close()
    gc.collect()
    assert values.tobytes() == expected
    assert int(offsets[-1]) == values.shape[0]


def test_graph_packed_reads_pin_revisions_and_keep_requested_node_order(db):
    a, b = [db.upsert_object(dict(uuid=str(uuid.UUID(int=i+1)), kind='track')) for i in range(2)]
    ra = db.put_geometry(a, track([[1, 2, 3], [4, 5, 6]]))
    db.put_geometry(b, track([[7, 8, 9]]))
    with db.graph_read(object_kind='track') as graph:
        db.put_geometry(a, track([[99, 0, 0]]), expected_revision=ra)
        packed = graph.read_geometry_arrays('points', nodes=[1, 0, 1])
        np.testing.assert_array_equal(packed.values, [[7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(packed.offsets, [0, 1, 3, 4])
        all_nodes = graph.read_geometry_arrays('points')
        np.testing.assert_array_equal(all_nodes.values, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert graph.read_geometry_arrays('points', nodes=[]).values.shape == (0,)
        with pytest.raises(IndexError):
            graph.read_geometry_arrays('points', nodes=[2])
    # Resident point gathers need no reader transaction and retain exact dtype.
    values = packed.values
    del packed
    np.testing.assert_array_equal(values[[3, 0, 2]], [[7, 8, 9], [7, 8, 9], [4, 5, 6]])
    with pytest.raises(RuntimeError, match='closed'):
        graph.read_geometry_arrays('points')
    with db.graph_read(object_kind='track') as fresh:
        np.testing.assert_array_equal(fresh.read_geometry_arrays('points').values, [[99, 0, 0], [7, 8, 9]])


def test_packed_mixed_physical_layouts_and_special_float_bits(db):
    revisions, payloads = [], []
    for rows, tile in [(33, 32), (17, 0)]:
        data = patch(rows, 35)
        raw = bytearray(data['arrays'][0]['data'])
        raw[12:28] = np.array([0x7fc12345, 0x80000000, 0x7f800000, 0xff800000], dtype='<u4').tobytes()
        data['arrays'][0]['data'] = bytes(raw)
        obj = db.upsert_object(dict(kind='patch'))
        revisions.append(db.put_geometry(obj, data, patch_tile_size=tile))
        payloads.append(bytes(raw))
    result = db.read_geometry_arrays(revisions[::-1], 'points')
    assert result.shape == [50, 35, 3]
    np.testing.assert_array_equal(result.offsets, [0, 17, 50])
    assert result.values.dtype == np.dtype('<f4')
    assert result.values.tobytes() == b''.join(payloads[::-1])


def test_packed_multichunk_linear_array(db):
    points = np.arange(300000, dtype='<i4').reshape(-1, 3)
    obj = db.upsert_object(dict(kind='track'))
    revision = db.put_geometry(obj, dict(kind='track', arrays=[dict(
        name='points', dtype='<i4', shape=list(points.shape), data=points.tobytes())]))
    packed = db.read_geometry_arrays([revision, revision], 'points')
    assert packed.values.tobytes() == points.tobytes() * 2
    np.testing.assert_array_equal(packed.offsets, [0, 100000, 200000])


def test_packed_empty_missing_and_incompatible_arrays(db):
    empty = db.read_geometry_arrays([], 'points')
    assert empty.dtype == '|u1' and empty.values.shape == (0,)
    np.testing.assert_array_equal(empty.offsets, [0])
    obj = db.upsert_object(dict(kind='track'))
    revision = db.put_geometry(obj, track([]))
    packed = db.read_geometry_arrays([revision, revision], 'points')
    assert packed.values.shape == (0, 3) and packed.dtype == '<i4'
    np.testing.assert_array_equal(packed.offsets, [0, 0, 0])
    with pytest.raises(ValueError, match='revision not found'):
        db.read_geometry_arrays([str(uuid.uuid4())], 'points')
    with pytest.raises(ValueError, match='array not found'):
        db.read_geometry_arrays([revision], 'missing')
    other = db.upsert_object(dict(kind='patch'))
    r = db.put_geometry(other, patch(2, 2))
    with pytest.raises(ValueError, match='matching dtype'):
        db.read_geometry_arrays([revision, r], 'points')
    r2 = db.put_geometry(other, patch(2, 3), expected_revision=r)
    with pytest.raises(ValueError, match='trailing dimensions'):
        db.read_geometry_arrays([r, r2], 'points')
    db.upsert_object(dict(kind='track'))
    with db.graph_read(object_kind='track') as graph:
        with pytest.raises(ValueError, match='no stored geometry'):
            graph.read_geometry_arrays('points')


@pytest.mark.parametrize('damage', ['payload', 'manifest', 'missing_chunk', 'wrong_reference'])
def test_packed_read_rechecks_integrity_after_previous_success(db, damage):
    obj = db.upsert_object(dict(kind='track'))
    rev = db.put_geometry(obj, track([[1, 2, 3]]))
    expected = db.read_geometry_arrays([rev], 'points').values.tobytes()
    if damage == 'payload':
        sqlite_rows(db.path, 'DROP TRIGGER geometry_chunks_immutable_update')
        sqlite_rows(db.path, 'UPDATE geometry_chunks SET data=zeroblob(length(data))')
    elif damage == 'manifest':
        sqlite_rows(db.path, 'DROP TRIGGER geometry_revisions_immutable_update')
        sqlite_rows(db.path, "UPDATE geometry_revisions SET manifest='{}'")
    elif damage == 'wrong_reference':
        other = db.upsert_object(dict(kind='track'))
        db.put_geometry(other, track([[4, 5, 6]]))
        sqlite_rows(db.path, 'DROP TRIGGER geometry_revision_chunks_immutable_update')
        sqlite_rows(db.path, 'UPDATE geometry_revision_chunks SET hash=(SELECT max(hash) FROM geometry_chunks)')
        # At least one of the two revisions now disagrees with its manifest.
        revs = [r['uuid'] for o in (obj, other) for r in db.geometry_history(o)]
        with pytest.raises(RuntimeError, match='corrupt payload'):
            db.read_geometry_arrays(revs, 'points')
        return
    else:
        sqlite_rows(db.path, 'DROP TRIGGER geometry_revision_chunks_immutable_delete')
        sqlite_rows(db.path, 'DELETE FROM geometry_revision_chunks')
    with pytest.raises(RuntimeError, match='corrupt|missing payload'):
        db.read_geometry_arrays([rev], 'points')
    assert expected == np.array([[1, 2, 3]], dtype='<i4').tobytes()
