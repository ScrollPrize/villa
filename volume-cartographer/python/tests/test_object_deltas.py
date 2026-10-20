"""Sparse array editing, retained history, and current materialization contracts."""
import json
import shutil
import sqlite3
import struct
import uuid
from pathlib import Path

import pytest
from vc.objects import ObjectStore
from object_helpers import ALL, edge_choice, patch, sqlite_rows, track


def edit(offset, data, array='points'):
    return dict(array=array, element_offset=offset, data=data)


@pytest.mark.parametrize('tile', [0, 32, 64, 128])
def test_delta_ranges_exact_history_noops_and_structural_replacement(db, tile):
    uid = db.upsert_object(dict(kind='patch'))
    data = patch(130, 140)
    raw = bytearray(data['arrays'][0]['data'])
    with db.write() as tx:
        first = rev = tx.put_geometry(uid, data, patch_tile_size=tile)
    expected = [(rev, bytes(raw))]
    for offset, value in [(127*3+2, b'\x01\x00\xc0\x7f'), (140*128*3, struct.pack('<6I', 0x80000000, 0, 0x7f800000, 1, 2, 3)), (139*3, struct.pack('<6f', *range(6)))]:
        rev = db.edit_geometry(uid, [edit(offset, value)], expected_revision=rev)
        raw[offset*4:offset*4+len(value)] = value
        expected.append((rev, bytes(raw)))
    for r, payload in expected:
        assert db.read_geometry_array(r, 'points') == payload
        assert db.read_geometry_ranges(r, 'points', [(1520, 91), (0, 7)]) == [payload[1520:1611], payload[:7]]
        crop = b''.join(payload[(y*140+125)*12:(y*140+135)*12] for y in range(126,130))
        assert db.read_geometry_rectangle(r, 'points', 126,125,4,10) == crop
    packed = db.read_geometry_arrays([r for r,_ in expected], 'points')
    assert packed.data.tobytes() == b''.join(p for _,p in expected)
    journal = db.journal_since()
    assert db.edit_geometry(uid, [edit(0, b'\0'*4), edit(0, bytes(raw[:4]))], expected_revision=rev) == rev
    assert db.edit_geometry(uid, [], expected_revision=rev) == rev
    assert db.journal_since() == journal
    data['arrays'][0]['data'] = bytes(raw)
    assert db.put_geometry(uid, data, expected_revision=rev) == rev
    with pytest.raises(RuntimeError, match='current revision changed'):
        db.edit_geometry(uid, [], expected_revision=first)
    replaced = db.put_geometry(uid, patch(2,2), expected_revision=rev)
    after = db.edit_geometry(uid, [edit(2, struct.pack('<f', 99))], expected_revision=replaced)
    for r,payload in expected: assert db.read_geometry_array(r,'points') == payload
    assert struct.unpack('<f',db.read_geometry_array(after,'points',8,4))[0] == 99
    assert len(db.geometry_history(uid)) == 6


@pytest.mark.parametrize('kind,dtype,shape', [('track','<i4',[2,3]), ('point_collection','<f8',[2,3]), ('fiber','<f8',[2,3])])
def test_all_kinds_auxiliary_arrays_and_transaction_rollback(db,kind,dtype,shape):
    name = 'line_points' if kind == 'fiber' else 'points'
    scalar = 8 if dtype == '<f8' else 4
    arrays = [dict(name=name,dtype=dtype,shape=shape,data=b'\0'*(6*scalar)),
              dict(name='labels',dtype='|u1',shape=[2],data=b'ab')]
    if kind == 'fiber': arrays.append(dict(name='control_points',dtype=dtype,shape=[0,3],data=b''))
    if kind == 'point_collection': arrays.append(dict(name='point_ids',dtype='<i8',shape=[2],data=struct.pack('<2q',10,20)))
    uid = db.upsert_object(dict(kind=kind))
    initial = db.put_geometry(uid,dict(kind=kind,arrays=arrays))
    with pytest.raises(ValueError,match='abort'):
        with db.write() as tx:
            abandoned = tx.edit_geometry(uid,[edit(0,b'z','labels')],expected_revision=initial)
            raise ValueError('abort')
    assert db.geometry_revision(abandoned) is None
    assert db.current_geometry(uid)['uuid'] == initial
    with db.write() as tx:
        with pytest.raises((ValueError,IndexError)):
            tx.edit_geometry(uid,[edit(0,b'x','labels'),edit(6*scalar,b'!',name)],expected_revision=initial)
        final = tx.edit_geometry(uid,[edit(0,b'c','labels'),edit(0,b'd','labels')],expected_revision=initial)
    assert db.read_geometry_array(final,'labels') == b'db'
    assert db.read_geometry_array(initial,'labels') == b'ab'
    assert len(db.geometry_history(uid)) == 2
    with ObjectStore.open(db.path,read_only=True) as ro:
        assert ro.read_geometry_array(final,'labels') == b'db'


def test_delta_relationship_dependencies_and_approval(db):
    a = db.upsert_object(dict(kind='track'))
    b = db.upsert_object(dict(kind='track'))
    first = db.put_geometry(a,track([(1,2,3),(4,5,6)]))
    db.put_geometry(b,track([(1,2,3)]))
    edge = db.add_edge(dict(kind='linked',a=a,b=b,a_anchor=dict(points=[dict(array='points',index=[0])]),b_anchor=dict(whole_geometry=True)))
    db.set_approval(a,'approved'); db.set_approval(edge,'approved')
    unrelated = db.edit_geometry(a,[edit(3,struct.pack('<i',10))],expected_revision=first)
    assert db.object(a)['approval_state'] == 'pending'
    assert db.edge(edge)['approval_state'] == 'approved'
    changes = [edit(0,struct.pack('<i',11))]
    assert [e['uuid'] for e in db.affected_geometry_edit_edges(a,changes,expected_revision=unrelated)] == [edge]
    with db.write() as tx:
        with pytest.raises(RuntimeError,match='explicit retain/update/remove'):
            tx.edit_geometry(a,changes,expected_revision=unrelated)
    assert db.current_geometry(a)['uuid'] == unrelated
    final = db.edit_geometry(a,changes,expected_revision=unrelated,edge_edits=[edge_choice(db,edge,'retain')])
    assert db.edge(edge)['approval_state'] == 'pending'
    assert not db.edge_geometry_state(edge)['stale']
    assert db.read_geometry_array(first,'points') == track([(1,2,3),(4,5,6)])['arrays'][0]['data']
    assert final != unrelated


@pytest.mark.parametrize('kind', ['patch','track','point_collection'])
def test_incremental_bounds_equal_full_replacement(db,kind):
    if kind == 'patch': data = patch(65,65); offsets = [0,(32*65+32)*3,(64*65+64)*3]; fmt='<3f'
    elif kind == 'track': data=track([(0,0,0)]*514); offsets=[0,256*3,513*3]; fmt='<3i'
    else:
        data=dict(kind=kind,arrays=[dict(name='points',dtype='<f8',shape=[514,3],data=b'\0'*(514*24)),
                                   dict(name='point_ids',dtype='<i8',shape=[514],data=struct.pack('<514q',*range(514)))])
        offsets=[0,256*3,513*3]; fmt='<3d'
    a=db.upsert_object(dict(kind=kind)); b=db.upsert_object(dict(kind=kind))
    ar=db.put_geometry(a,data); br=db.put_geometry(b,data)
    raw=bytearray(data['arrays'][0]['data']); scalar=struct.calcsize(fmt)//3
    for i,offset in enumerate(offsets):
        value=struct.pack(fmt,1000+i,-50-i,100+i)
        ar=db.edit_geometry(a,[edit(offset,value)],expected_revision=ar)
        raw[offset*scalar:offset*scalar+len(value)]=value
        data['arrays'][0]['data']=bytes(raw)
        br=db.put_geometry(b,data,expected_revision=br)
        def regions(uid):
            return [dict(start=r['start'],shape=r['shape'],bbox=r['bbox']) for r in db.spatial_regions(ALL,object_uuids=[uid])]
        assert regions(a)==regions(b)


def test_sync_delta_histories_branches_and_followup_edit(tmp_path):
    with ObjectStore.create(str(tmp_path/'a.vcobj'),author='a') as a, ObjectStore.create(str(tmp_path/'b.vcobj'),author='b') as b:
        uid=a.upsert_object(dict(kind='track'))
        first=a.put_geometry(uid,track([(1,2,3)]))
        a.synchronize(b)
        ra=a.edit_geometry(uid,[edit(0,struct.pack('<i',10))],expected_revision=first)
        rb=b.edit_geometry(uid,[edit(1,struct.pack('<i',20))],expected_revision=first)
        plan=a.synchronize(b,dry_run=True)
        assert plan['conflicts']
        a.synchronize(b,resolutions={uid:'local'},expected_plan=plan['plan_token'])
        for store in (a,b):
            assert store.read_geometry_array(ra,'points') == struct.pack('<3i',10,2,3)
            assert store.read_geometry_array(rb,'points') == struct.pack('<3i',1,20,3)
            assert len(store.geometry_history(uid))==3
        rc=b.edit_geometry(uid,[edit(2,struct.pack('<i',30))],expected_revision=ra)
        b.synchronize(a)
        assert a.read_geometry_array(rc,'points') == struct.pack('<3i',10,2,30)
        assert a.current_geometry(uid)==b.current_geometry(uid)


def test_delta_payload_corruption_and_materialized_corruption(db):
    uid=db.upsert_object(dict(kind='track'))
    first=db.put_geometry(uid,track([(1,2,3)]))
    second=db.edit_geometry(uid,[edit(0,struct.pack('<i',4))],expected_revision=first)
    third=db.edit_geometry(uid,[edit(1,struct.pack('<i',5))],expected_revision=second)
    sqlite_rows(db.path,'DROP TRIGGER geometry_deltas_immutable_update')
    sqlite_rows(db.path,"UPDATE geometry_deltas SET data=x'00' WHERE lower(hex(revision_uuid))=?",(second.replace('-',''),))
    with pytest.raises(RuntimeError,match='corrupt payload'): db.read_geometry_array(second,'points')
    assert db.read_geometry_array(third,'points') == struct.pack('<3i',4,5,3)
    sqlite_rows(db.path,"UPDATE geometry_current_chunks SET data=x'00'")
    with pytest.raises(RuntimeError,match='corrupt payload chunk'): db.read_geometry_array(third,'points')


def test_v7_migration_keeps_snapshot_and_allows_delta(tmp_path):
    path=str(tmp_path/'old.vcobj')
    with ObjectStore.create(path,author='test') as store:
        uid=store.upsert_object(dict(kind='track')); first=store.put_geometry(uid,track([(1,2,3)]))
    with sqlite3.connect(path) as sql:
        from object_helpers import remove_history_schema
        remove_history_schema(sql)
        for table in ('geometry_deltas','geometry_current_chunks','geometry_materializations'): sql.execute(f'DROP TABLE {table}')
        sql.execute('PRAGMA user_version=7')
    with ObjectStore.open(path,read_only=True) as ro: assert ro.read_geometry_array(first,'points') == struct.pack('<3i',1,2,3)
    with ObjectStore.open(path,author='test') as store:
        second=store.edit_geometry(uid,[edit(0,struct.pack('<i',9))],expected_revision=first)
        assert store.read_geometry_array(second,'points') == struct.pack('<3i',9,2,3)
        assert store.read_geometry_array(first,'points') == struct.pack('<3i',1,2,3)


def test_real_scroll_patch_sparse_history_storage_and_reopen(tmp_path):
    """Use actual PHerc 0172 coordinates, not generated geometry, for storage evidence."""
    np=pytest.importorskip('numpy')
    fixture=Path(__file__).resolve().parents[2]/'core/test/data/segments/20241113070770'
    copied=tmp_path/'segment'; shutil.copytree(fixture,copied)
    metadata=json.loads((copied/'meta.json').read_text()); uid=str(uuid.uuid4()); metadata['uuid']=uid
    (copied/'meta.json').write_text(json.dumps(metadata))
    path=str(tmp_path/'history.vcobj')
    with ObjectStore.create(path,author='test') as store:
        assert store.ingest_geometry_source('patch',copied)['ingested']==1
        first=rev=store.current_geometry(uid)['uuid']
        initial_object = store.current_object_revision(uid)
        with store.write(expected_revisions={uid: initial_object['uuid']}) as tx:
            tx.set_meta(uid, 'history_validation', {'accepted': True})
        metadata_revision = store.current_object_revision(uid)
        assert metadata_revision['geometry_revision'] == first
        assert 'history_validation' not in store.object_revision(initial_object['uuid'])['record']['meta']
        assert metadata_revision['record']['meta']['history_validation'] == {'accepted': True}
        initial=store.read_geometry_array(first,'points'); pts=np.frombuffer(initial,dtype='<f4').reshape(-1,3)
        valid=np.flatnonzero(np.isfinite(pts).all(axis=1) & (pts[:,2]>0))[:100]
        assert len(valid)==100
        original_chunks=sqlite_rows(path,'SELECT count(*),sum(length(data)) FROM geometry_chunks')[0]
        expected=bytearray(initial)
        saved=[]
        for i in range(100):
            point=int(valid[i%len(valid)]); offset=point*3+2
            value=struct.pack('<f',float(pts[point,2])+0.25*(i+1))
            rev=store.edit_geometry(uid,[edit(offset,value)],expected_revision=rev)
            expected[offset*4:offset*4+4]=value
            if i in (0,49,99): saved.append((rev,bytes(expected)))
        assert sqlite_rows(path,'SELECT count(*),sum(length(data)) FROM geometry_chunks')[0]==original_chunks
        assert sqlite_rows(path,'SELECT count(*) FROM geometry_deltas')[0][0]==100
        assert sqlite_rows(path,'SELECT sum(length(data)) FROM geometry_deltas')[0][0]<100*512
        for r,raw in saved: assert store.read_geometry_array(r,'points')==raw
        store.checkpoint()
    with ObjectStore.open(path,read_only=True) as store:
        assert store.read_geometry_array(first,'points')==initial
        assert store.read_geometry_array(rev,'points')==bytes(expected)


def test_compressed_auxiliary_ranges_cross_linear_chunks_and_hashes(db):
    uid=db.upsert_object(dict(kind='track'))
    size=2*1024*1024+16
    data=track([(1,2,3)])
    data['arrays'].append(dict(name='labels',dtype='|u1',shape=[size],data=b'a'*size))
    first=db.put_geometry(uid,data)
    second=db.edit_geometry(uid,[edit(0,b'b'*size,'labels')],expected_revision=first)
    descriptor=json.loads(sqlite_rows(db.path,'SELECT manifest FROM geometry_revisions WHERE lower(hex(uuid))=?',(second.replace('-',''),))[0][0])
    assert descriptor['delta']['codec']=='zstd'
    third=db.edit_geometry(uid,[edit(size-1,b'c','labels')],expected_revision=second)
    assert db.read_geometry_array(second,'labels')==b'b'*size
    assert db.read_geometry_array(third,'labels')==b'b'*(size-1)+b'c'
    assert db.read_geometry_array(second,'labels',1024*1024-7,32)==b'b'*32
    reverted=db.edit_geometry(uid,[edit(0,b'a'*size,'labels')],expected_revision=third)
    assert db.geometry_revision(first)['content_hash']==db.geometry_revision(reverted)['content_hash']
    assert db.put_geometry(uid,data,expected_revision=reverted)==reverted


def test_failed_publication_rolls_back_inside_caller_transaction(db):
    uid=db.upsert_object(dict(kind='track'))
    first=db.put_geometry(uid,track([(1,2,3)]))
    sqlite_rows(db.path,"CREATE TRIGGER fail_delta BEFORE INSERT ON geometry_current_chunks BEGIN SELECT RAISE(ABORT,'injected storage failure'); END")
    with db.write() as tx:
        with pytest.raises(RuntimeError,match='injected storage failure'):
            tx.edit_geometry(uid,[edit(0,struct.pack('<i',9))],expected_revision=first)
        tx.set_meta(uid,'survived',True)
    assert db.current_geometry(uid)['uuid']==first
    assert len(db.geometry_history(uid))==1
    assert db.object(uid)['meta']['survived']
    assert sqlite_rows(db.path,'SELECT count(*) FROM geometry_deltas')[0][0]==0


def test_stable_ids_are_checked_against_edited_identity_array(db):
    uid=db.upsert_object(dict(kind='point_collection'))
    other=db.upsert_object(dict(kind='note'))
    data=dict(kind='point_collection',arrays=[dict(name='points',dtype='<f8',shape=[2,3],data=struct.pack('<6d',1,2,3,4,5,6)),
                                             dict(name='point_ids',dtype='<i8',shape=[2],data=struct.pack('<2q',10,20))])
    first=db.put_geometry(uid,data)
    edge=db.add_edge(dict(kind='linked',a=uid,b=other,a_anchor=dict(points=[dict(array='point_ids',index=[10])]),b_anchor=dict(whole_geometry=True)))
    changes=[edit(0,struct.pack('<q',30),'point_ids')]
    assert [e['uuid'] for e in db.affected_geometry_edit_edges(uid,changes,expected_revision=first)]==[edge]
    with pytest.raises(RuntimeError,match='missing or invalid point'):
        db.edit_geometry(uid,changes,expected_revision=first,edge_edits=[edge_choice(db,edge,'retain')])
    second=db.edit_geometry(uid,changes,expected_revision=first,edge_edits=[edge_choice(db,edge,'remove')])
    assert db.read_geometry_array(second,'point_ids')==struct.pack('<2q',30,20)


@pytest.mark.parametrize('field,value,match', [('raw_size',2**40,'invalid size'), ('codec','unknown','unsupported codec')])
def test_corrupt_delta_envelope_is_rejected_before_reconstruction(db,field,value,match):
    import hashlib
    uid=db.upsert_object(dict(kind='track'))
    first=db.put_geometry(uid,track([(1,2,3)]))
    second=db.edit_geometry(uid,[edit(0,struct.pack('<i',8))],expected_revision=first)
    # An explicit replacement removes the mutable shortcut to the old delta.
    db.put_geometry(uid,track([(10,20,30)]),expected_revision=second)
    text=sqlite_rows(db.path,'SELECT manifest FROM geometry_revisions WHERE lower(hex(uuid))=?',(second.replace('-',''),))[0][0]
    descriptor=json.loads(text); descriptor['delta'][field]=value
    text=json.dumps(descriptor,separators=(',',':'),sort_keys=True)
    sqlite_rows(db.path,'DROP TRIGGER geometry_revisions_immutable_update')
    sqlite_rows(db.path,'DROP TRIGGER geometry_deltas_immutable_update')
    sqlite_rows(db.path,'UPDATE geometry_revisions SET manifest=? WHERE lower(hex(uuid))=?',(text,second.replace('-','')))
    sqlite_rows(db.path,'UPDATE geometry_deltas SET manifest_hash=? WHERE lower(hex(revision_uuid))=?',(hashlib.sha256(text.encode()).hexdigest(),second.replace('-','')))
    with pytest.raises(RuntimeError,match=match): db.read_geometry_array(second,'points')


def test_near_head_history_can_reverse_without_replaying_old_payloads(db):
    uid=db.upsert_object(dict(kind='track'))
    rev=db.put_geometry(uid,track([(0,0,0)])); revisions=[]
    for i in range(1,7):
        rev=db.edit_geometry(uid,[edit(0,struct.pack('<i',i))],expected_revision=rev); revisions.append(rev)
    sqlite_rows(db.path,'DROP TRIGGER geometry_deltas_immutable_update')
    sqlite_rows(db.path,"UPDATE geometry_deltas SET data=x'00' WHERE lower(hex(revision_uuid))=?",(revisions[0].replace('-',''),))
    # Exact earlier values can still be reconstructed from the valid current
    # materialization and a valid inverse suffix, without decoding the prefix.
    assert db.read_geometry_array(revisions[-2],'points')==struct.pack('<3i',5,0,0)
    with pytest.raises(RuntimeError,match='corrupt payload'): db.read_geometry_array(revisions[0],'points')
