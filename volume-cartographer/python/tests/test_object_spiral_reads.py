"""Packed fitting inputs retain exact bytes, revisions and graph selection."""
import gc
import uuid

import numpy as np
import pytest

from object_helpers import patch, sqlite_rows, track


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


def test_graph_compact_query_matches_edge_selector(db):
    nodes = [db.upsert_object(dict(kind='track', track={'bbox': [i, 0, 0, i+1, 1, 1]})) for i in range(7)]
    for i in range(7):
        edge = db.add_edge(dict(kind='crossing' if i % 2 else 'other', a=nodes[i], b=nodes[(i+1) % 7], tags=['chosen']))
        db.set_approval(edge, 'approved' if i % 3 else 'pending')
    for states in ([], ['approved'], ['pending', 'approved']):
        with db.graph_read(object_kind='track', bbox=[0, 0, 0, 4, 1, 1],
                           edge_kind='crossing', edge_tag='chosen', approval_states=states) as graph:
            expected = db.select_edges(endpoint_uuids=graph.object_uuids, both_endpoints=True,
                edge_kind='crossing', tag='chosen', approval_states=states, fields=[])
            assert graph.edge_uuids == [e['uuid'] for e in expected]
            for endpoints, edge in zip(graph.endpoints, expected):
                assert [graph.object_uuids[i] for i in endpoints] == [edge['a'], edge['b']]
