"""Native selected-reference hashing matches the existing complete-owner codec."""
import hashlib
import uuid

import pytest

from vc_object_store import ObjectStore, import_initial_records, object_reference, object_references_digest

DOMAIN='fibers_digest v3'


def expected(references,domain=DOMAIN):
    digest=hashlib.sha256((domain+'\n').encode())
    for reference in sorted(references,key=lambda row:row['object']):
        digest.update(f"{reference['store']} {reference['generation']} {reference['object']} {reference['version']}\n".encode())
    return digest.hexdigest()


@pytest.mark.parametrize('history',[False,True])
def test_selected_reference_digest_matches_python_without_geometry(tmp_path,history):
    path=tmp_path/'local.vcobj'
    with ObjectStore.create(str(path),author='test',history_enabled=history,spatial_index=False) as store:
        a=store.upsert_object(dict(uuid=str(uuid.UUID(int=20)),kind='fiber',name='first'))
        b=store.upsert_object(dict(uuid=str(uuid.UUID(int=10)),kind='fiber',name='second'))
        store.upsert_object(dict(kind='track'))
        refs=[object_reference(row) for row in store.select_objects(object_kind='fiber',fields=['name'])]
        initial=store.selected_reference_digest(object_kind='fiber')
        assert store.object_count('fiber')==len(refs)
        assert initial==expected(refs)==object_references_digest(refs[::-1])
        assert store.selected_reference_digest(object_kind='fiber',object_uuids=[a,b,a])==initial
        assert store.selected_reference_digest(object_kind='fiber',object_uuids=[])==expected([])
        assert object_references_digest([])==expected([])
        assert store.selected_reference_digest(object_kind='fiber',object_uuids=[b])==expected([refs[0]])
        with pytest.raises(ValueError,match='Unknown requested object'):
            store.selected_reference_digest(object_kind='fiber',object_uuids=[str(uuid.UUID(int=999))])
        store.set_meta(a,'changed',True)
        assert store.selected_reference_digest(object_kind='fiber')!=initial
        store.delete_object(b)
        current=[object_reference(row) for row in store.select_objects(object_kind='fiber',fields=['name'])]
        assert store.object_count('fiber')==len(current)
        assert store.selected_reference_digest(object_kind='fiber')==expected(current)


@pytest.mark.parametrize('packed',[False,True])
def test_selected_reference_digest_preserves_shard_and_overlay_ownership(tmp_path,packed):
    ids=[str(uuid.UUID(int=i)) for i in range(1,9)]
    base=tmp_path/'base.vcobj'
    base=import_initial_records(base,[dict(record=dict(uuid=uid,kind='fiber')) for uid in ids],workers=2,packed=packed)['path']
    with ObjectStore.open(str(base),read_only=True) as store:
        refs=[object_reference(row) for row in store.select_objects(object_kind='fiber',fields=['name'])]
        baseline=store.selected_reference_digest(object_kind='fiber')
        assert baseline==expected(refs)
        assert store.object_count('fiber')==len(refs)
        # Physical partitions are one logical store; shard placement must not
        # invent different object owners. The overlay below is an independent
        # store and must change ownership only for the object it replaces.
        assert len({ref['store'] for ref in refs})==1
    overlay=tmp_path/'overlay.vcobj'
    overlay=import_initial_records(overlay,[dict(record=dict(uuid=ids[0],kind='fiber',name='override'))],
                           dependencies=[base],workers=1,packed=packed)['path']
    with ObjectStore.open(str(overlay),read_only=True) as store:
        after=[object_reference(row) for row in store.select_objects(object_kind='fiber',fields=['name'])]
        assert store.selected_reference_digest(object_kind='fiber')==expected(after)
        assert store.selected_reference_digest(object_kind='fiber')!=baseline
        assert after[0]['version']==refs[0]['version']
        assert after[0]['store']!=refs[0]['store']
        assert after[1:]==refs[1:]
        assert store.object_count('fiber')==len(after)


def test_explicit_reference_digest_includes_owner_and_stable_duplicate_order():
    reference=dict(store=str(uuid.UUID(int=1)),generation=str(uuid.UUID(int=2)),object=str(uuid.UUID(int=3)),version=1)
    other=dict(reference,store=str(uuid.UUID(int=9)))
    assert object_references_digest([reference])!=object_references_digest([other])
    assert object_references_digest([reference,other])==expected([reference,other])
    assert object_references_digest([other,reference])==expected([other,reference])
