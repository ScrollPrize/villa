"""Physical patch layout must never change logical bytes or revision semantics."""
import hashlib
import json
import random
import struct

import pytest
from vc.objects import ObjectStore
from test_object_spatial_regions import patch
from object_helpers import edge_choice, sqlite_rows


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / 'tiles.vcobj'), author='test') as store:
        yield store


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
    storage = manifest(db, rev)['arrays']['points'].get('storage')
    assert storage == (dict(version=1, layout='tiles', tile_size=tile) if tile else None)
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
    for args in [(-1, 0, 1, 1), (0, 0, -1, 1), (128, 0, 2, 1), (0, 141, 1, 1), (130, 0, 0, 0)]:
        with pytest.raises(IndexError, match='invalid rectangle'):
            db.read_geometry_rectangle(rev, 'points', *args)
    with pytest.raises(ValueError, match='array not found'):
        db.read_geometry_rectangle(rev, 'missing', 0, 0, 1, 1)
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.read_geometry_rectangle(rev, 'points', 31, 63, 33, 33) == crop(payload, 141, 31, 63, 33, 33)


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


def test_v5_open_does_not_rewrite_history(tmp_path):
    path = str(tmp_path / 'legacy.vcobj')
    with ObjectStore.create(path, author='test') as db:
        uid = db.upsert_object(dict(kind='patch'))
        data = patch(70, 71)
        rev = db.put_geometry(uid, data, patch_tile_size=0)
        before = db.uuid, db.export_json(), db.journal_since(), manifest(db, rev)
    sqlite_rows(path, 'PRAGMA user_version=5')
    with ObjectStore.open(path, read_only=True) as db:
        assert db.read_geometry_rectangle(rev, 'points', 32, 32, 33, 33) == crop(data['arrays'][0]['data'], 71, 32, 32, 33, 33)
    with ObjectStore.open(path, author='test') as db:
        assert (db.uuid, db.export_json(), db.journal_since(), manifest(db, rev)) == before
        assert db.put_geometry(uid, data, expected_revision=rev) == rev
    assert sqlite_rows(path, 'PRAGMA user_version')[0][0] == 6


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
    m = manifest(db, rev)
    m['arrays']['points']['storage']['version'] = 999
    text = json.dumps(m, sort_keys=True, separators=(',', ':'))
    sqlite_rows(db.path, 'DROP TRIGGER geometry_revisions_immutable_update')
    sqlite_rows(db.path, 'UPDATE geometry_revisions SET manifest=?,content_hash=?', (text, hashlib.sha256(text.encode()).hexdigest()))
    with pytest.raises(RuntimeError, match='unsupported storage layout'):
        db.read_geometry_array(rev, 'points')
