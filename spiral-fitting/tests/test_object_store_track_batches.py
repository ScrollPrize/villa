"""Tracks are read from a store in batches, each cut down to what the fit reads
of it, and the fit's inputs are those of one read of everything."""
import hashlib

import numpy as np
import pytest

pytest.importorskip('vc_object_store')

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    crossing_edge, put_patch, put_track)


pytestmark = pytest.mark.usefixtures('cpu')

# Not what the fit reads of a track, so the read drops it (every real track
# carries the whole store's signature here).
SOURCE = {'source_db_signature': [['tracks.dbm', 1, 2]], 'subset_note': 'x' * 100}


def _line(start, axis, length):
    points = np.tile(np.asarray(start, dtype=np.int64), (length + 1, 1))
    points[:, axis] += np.arange(length + 1)
    return points


def _store(db):
    """A patch and eight tracks of two datasets (one off the z window), a
    crossing between two of them and a link from a track to the patch;
    returns the track UUIDs."""
    patch = put_patch(db, np.stack(np.meshgrid(
        np.arange(36.0, 46.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    tracks = []
    for index, y in enumerate((60, 70, 80)):
        tracks.append(put_track(db, _line((40, y, 50), 2, 40), 2 * index, 'horizontal',
                                meta={'source': SOURCE}, tags=['first', 'second']))
    for index, x in enumerate((55, 68, 81)):
        tracks.append(put_track(db, _line((40, 55, x), 1, 32), 2 * index + 1, 'vertical',
                                meta={'source': SOURCE}))
    tracks.append(put_track(db, _line((40, 60, 52), 2, 10), 0, 'horizontal',
                            dataset='other.vctracks', meta={'source': SOURCE}))
    tracks.append(put_track(db, _line((200, 60, 52), 2, 10), 1, 'vertical',
                            dataset='other.vctracks', meta={'source': SOURCE}))
    crossing_edge(db, tracks[0], 10, 10.0, tracks[3], 5, 5.0, 3.0)
    db.add_edge({'kind': 'same-surface', 'a': tracks[0], 'b': patch,
                 'a_anchor': {'points': [{'array': 'points', 'index': [2]}]},
                 'b_anchor': {'points': [{'array': 'points', 'index': [2, 2]}]},
                 'meta': {'track_surface': {}}})
    return tracks


def _one_call(store, refs, tracks, position):
    """The read as it was: every object in one call."""
    if tracks:
        return store.selected_inputs(
            refs, position.provenance(),
            relationship_refs=[ref for ref in refs if ref['object'] not in tracks])
    return store.selected_inputs(refs, position.provenance())


def _read(db, batch=None, old=False, tracks=True):
    """What a fit takes from the store, however the tracks were read."""
    config = {**reference.fit_config(), 'input_use_tracks': tracks}
    with pytest.MonkeyPatch.context() as patch:
        if old:
            patch.setattr(osi, '_read_selected_inputs', _one_call)
        elif batch:
            patch.setattr(osi, 'TRACK_READ_BATCH', batch)
        snapshot = osi.read_snapshot(db, osi.request_from_config(config))
    if not tracks:
        return snapshot
    inputs = osi.materialize_from_config(db, snapshot, config)
    manifest = osi.input_manifest(snapshot, config, osi.NormalSource('none'),
                                  tracks=inputs.tracks)
    arrays = hashlib.sha256()
    collection = inputs.tracks.tracks
    for name in ('coordinates', 'offsets', 'source_ids', 'family_codes', 'arclengths',
                 'tortuosities'):
        arrays.update(np.ascontiguousarray(getattr(collection, name)).tobytes())
    for key in sorted(inputs.tracks.crossings):
        arrays.update(np.ascontiguousarray(inputs.tracks.crossings[key]).tobytes())
    return {'digest': osi.manifest_digest(manifest), 'manifest': osi._compared(manifest),
            'arrays': arrays.hexdigest(), 'counts': dict(inputs.counts),
            'exclusions': dict(snapshot.exclusions),
            'datasets': {uuid: obj.dataset for uuid, obj in snapshot.objects.items()},
            'pins': snapshot.pins().provenance()}, snapshot


@pytest.mark.parametrize('batch', [1, 2, 3, 4, 7, 8, 9, 20000])
def test_batches_give_the_inputs_of_one_read(db, batch):
    # Of eight tracks: 3 leaves a remainder, 4 and 2 are exact multiples, 1 is
    # a batch each, 7 leaves one, 8 is one full batch, 9 and 20000 hold all.
    _store(db)
    expected, _ = _read(db, old=True)
    found, snapshot = _read(db, batch=batch)
    assert found == expected
    assert found['exclusions']  # the patch's link to a track is counted, as before
    assert len(snapshot.objects_of('track')) == 8


def test_tracks_alone_and_no_tracks(db):
    put_patch(db, np.stack(np.meshgrid(
        np.arange(36.0, 46.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    for index in range(5):
        put_track(db, _line((40, 60 + index, 50), 2, 10), index, 'horizontal')
    expected, _ = _read(db, old=True)
    assert _read(db, batch=2)[0] == expected
    assert _read(db, tracks=False).objects_of('track') == []


class _Recording:
    def __init__(self, store):
        self._store = store
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._store, name)

    def selected_inputs(self, refs, position, relationship_refs=None):
        self.calls.append((len(refs), relationship_refs))
        return self._store.selected_inputs(refs, position, relationship_refs)


def test_everything_but_tracks_is_read_in_one_call_and_tracks_in_batches(db, monkeypatch):
    _store(db)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 3)
    store = _Recording(db)
    osi.read_snapshot(store, osi.request_from_config(
        {**reference.fit_config(), 'input_use_tracks': True}))
    # The patch, with its relationships; then batches of 3, 3 and 2 tracks,
    # which enumerate none.
    assert [(count, refs is not None and len(refs)) for count, refs in store.calls] == [
        (1, 1), (3, 0), (3, 0), (2, 0)]


def test_a_kept_track_holds_only_what_the_fit_reads(db):
    uuids = _store(db)
    _, snapshot = _read(db, batch=3)
    _, whole = _read(db, old=True)
    obj, original = snapshot.objects[uuids[0]], whole.objects[uuids[0]]
    # The saving cannot come back unseen: nothing but these keys is kept.
    assert set(obj.record) == {*osi._TRACK_RECORD_KEYS, 'meta', 'track'}
    assert set(obj.revision) == {'object_uuid', 'geometry_revision', 'record'}
    assert 'source' in original.record['meta']
    assert obj.record['meta'] == {'coordinates': original.record['meta']['coordinates']}
    assert set(original.record['track']) > set(osi._TRACK_FIELDS)
    assert obj.record['track'] == {key: original.record['track'][key] for key in osi._TRACK_FIELDS}
    assert obj.record['tags'] == original.record['tags']
    assert obj.tags == ['first', 'second']
    # What later code asks of a track is the same.
    asked = ('uuid', 'kind', 'name', 'version', 'retired', 'tags', 'reference',
             'geometry_revision', 'metadata_sha256')
    assert [getattr(obj, name) for name in asked] == [getattr(original, name) for name in asked]
    assert obj.provenance() == original.provenance()
    # Other kinds are read whole.
    patch = next(uuid for uuid, o in snapshot.objects.items() if o.kind == 'surface')
    assert snapshot.objects[patch].record == whole.objects[patch].record


def test_tracks_of_one_dataset_and_frame_hold_what_they_repeat_once(db):
    uuids = _store(db)
    _, snapshot = _read(db, batch=3)
    a, b = snapshot.objects[uuids[0]], snapshot.objects[uuids[1]]
    assert a.record['track']['dataset_ref'] is b.record['track']['dataset_ref']
    assert a.record['meta']['coordinates'] is b.record['meta']['coordinates']
    assert a.record['version_store'] is b.record['version_store']


def test_a_retired_track_stays_retired_when_cut():
    # A retired track cannot be selected, so the cut is tried on a record.
    revision = {'object_uuid': 'u', 'geometry_revision': 3, 'record': {
        'uuid': 'u', 'kind': 'track', 'name': 'n', 'version': 1, 'version_generation': 'g',
        'version_store': 's', 'retired': True, 'tags': ['first'], 'meta': {'source': SOURCE},
        'track': {'dataset_ref': 'd', 'external_key': '1', 'family': 'vertical', 'x': 1},
        'created_by': 'someone'}}
    record = osi._cut_track_revision(revision, {})['record']
    assert record['retired'] is True
    assert record['tags'] == ['first']
    assert record['meta'] == {} and 'created_by' not in record
