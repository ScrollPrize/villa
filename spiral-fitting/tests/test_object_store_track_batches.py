"""Tracks are read from a store in batches, each cut down to what the fit reads
of it, and the fit's inputs are those of one read of everything."""
import hashlib
import threading

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

    def selected_inputs(self, refs, position, relationship_refs=None, **options):
        self.calls.append((len(refs), relationship_refs, options))
        return self._store.selected_inputs(refs, position, relationship_refs, **options)


def test_everything_is_read_in_one_call_whose_tracks_arrive_in_batches(db, monkeypatch):
    uuids = _store(db)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 3)
    store = _Recording(db)
    osi.read_snapshot(store, osi.request_from_config(
        {**reference.fit_config(), 'input_use_tracks': True}))
    # One read of the patch and the eight tracks, as it was; only the patch's
    # relationships are enumerated; the objects are handed over 3 at a time.
    [(count, relationships, options)] = store.calls
    assert count == 9 and len(relationships) == 1 and relationships[0]['object'] not in uuids
    assert options['slice_size'] == 3 and callable(options['on_slice']) and set(options) == {'slice_size', 'on_slice'}


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


# --- the read is one capture of the position it accepted ---------------------
#
# A fit's read can overlap an editor's save. Whatever lands after the read
# began, it finishes at the position it accepted: reading the tracks in
# batches must not raise StoreBusy on an overlay that moved, or lose a track
# version a history-off store no longer retains, since the one read of
# everything did neither.

class _SavesAfterFirstRead:
    """A store handle that runs ``save`` once, while a capture is under way:
    after the first native selected_inputs response of a capture made in
    several calls (the batched read), or inside the first slice handed to
    ``on_slice`` of a capture made in one (the read is then open). ``save``
    writes through ANOTHER handle of the store (opened by the test), so it is
    an edit another session would make."""

    def __init__(self, store, save):
        self._store = store
        self._save = save
        self.saves = 0

    def __getattr__(self, name):
        return getattr(self._store, name)

    def _once(self):
        if not self.saves:
            self.saves += 1
            self._save()

    def selected_inputs(self, refs, position, relationship_refs=None, on_slice=None, **options):
        if on_slice is None:
            result = self._store.selected_inputs(refs, position, relationship_refs, **options)
            self._once()
            return result

        def hooked(part):
            self._once()
            return on_slice(part)
        return self._store.selected_inputs(refs, position, relationship_refs, on_slice=hooked, **options)


def _set_meta(store, uuid, key, value, request='7b9d0c2e-5a1f-4f6b-9c3d-2e1f0a9b8c7d'):
    from vc_object_store import object_reference
    status = store.status()
    return store.submit({'command_version': 3, 'request': request, 'store': status['store'],
                         'generation': status['generation'], 'author': 'editor',
                         'expected_objects': {uuid: object_reference(
                             store.current_object_revision(uuid)['record'])},
                         'expected_relationships': {}, 'read_dependencies': {},
                         'expected_commit_sequence': None,
                         'operations': [{'op': 'set_meta', 'uuid': uuid, 'key': key,
                                         'value': value}]})


def _tracks_only(**overrides):
    return {**reference.fit_config(), 'input_use_tracks': True, 'input_disable_patches': True,
            'object_store_refresh_links': False, **overrides}


def _writable_overlay(tmp_path):
    """An overlay of a base store with three tracks, open for writing. Python
    writes extension stores sealed, so the seal is lifted for the test."""
    import sqlite3
    from vc_object_store import ObjectStore, write_extension_store
    base = tmp_path / 'base.vcobj'
    with ObjectStore.create(str(base), author='ingest') as db:
        for index, y in enumerate((60, 70, 80)):
            put_track(db, _line((40, y, 50), 2, 40), index, 'horizontal')
        db.seal()
    overlay = tmp_path / 'overlay.vcobj'
    write_extension_store(str(overlay), [str(base)], [], {})
    with sqlite3.connect(str(overlay)) as connection:
        connection.execute('UPDATE store_manifest SET sealed=0')
    connection.close()
    return overlay


def _unused_patch(db):
    return put_patch(db, np.stack(np.meshgrid(
        np.arange(36.0, 46.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))


def _track_versions(snapshot):
    return {obj.uuid: obj.version for obj in snapshot.objects_of('track')}


def test_a_save_to_an_overlay_during_the_capture_does_not_break_it(tmp_path, monkeypatch):
    from vc_object_store import ObjectStore
    # The overlay owns a surface no selection reads, and inherits three tracks.
    overlay = _writable_overlay(tmp_path)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 1)
    config = _tracks_only()
    with ObjectStore.open(str(overlay), author='fitter') as db:
        unused = _unused_patch(db)
        request = osi.request_from_config(config)
        expected = osi.read_snapshot(db, request)
        before = db.status()['commit_sequence']
        with ObjectStore.open(str(overlay), author='editor') as other:
            store = _SavesAfterFirstRead(db, lambda: _set_meta(other, unused, 'review_note', 'seen'))
            found = osi.read_snapshot(store, request)
        assert db.status()['commit_sequence'] > before  # the save did land mid-capture
        assert store.saves == 1
    assert _track_versions(found) == _track_versions(expected) and len(_track_versions(found)) == 3
    assert found.pins().provenance() == expected.pins().provenance()


def _history_off_store(path):
    from vc_object_store import ObjectStore
    with ObjectStore.create(str(path), author='ingest', history_enabled=False) as db:
        return [put_track(db, _line((40, y, 50), 2, 30), index, 'horizontal')
                for index, y in enumerate((60, 70))]


def test_a_note_edit_during_the_capture_keeps_the_versions_it_began_with(tmp_path, monkeypatch):
    from vc_object_store import ObjectStore
    path = tmp_path / 'tracks.vcobj'
    tracks = _history_off_store(path)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 1)
    request = osi.request_from_config(_tracks_only())
    with ObjectStore.open(str(path), author='fitter') as db:
        expected = osi.read_snapshot(db, request)
        pins = expected.pins()
        # The edit lands during the capture, on the track read last.
        edited = pins.refs()[-1]['object']
        with ObjectStore.open(str(path), author='editor') as other:
            store = _SavesAfterFirstRead(db, lambda: _set_meta(other, edited, 'review_note', 'seen'))
            found = osi.read_snapshot(store, request)
        assert store.saves == 1
        assert db.current_version(edited) == _track_versions(expected)[edited] + 1
        assert _track_versions(found) == _track_versions(expected) and len(_track_versions(found)) == 2
        assert found.pins().provenance() == pins.provenance()
        # A pin already stale when a read begins is still refused.
        with pytest.raises(Exception, match='not retained|has no version'):
            osi.read_snapshot(db, __import__('dataclasses').replace(request, pins=pins))


def _within(seconds, call):
    """``call`` in a daemon thread, so a read that blocks cannot hang the suite:
    (finished, {'value': ...} or {'error': ...})."""
    box = {}

    def run():
        try:
            box['value'] = call()
        except BaseException as error:  # noqa: BLE001
            box['error'] = error
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    return not thread.is_alive(), box


def _one_reader_store(tmp_path):
    """A store with a pool of ONE reader connection, and what a capture of its
    tracks needs: (store, refs, tracks, position). A second read of it while
    one is open waits for that one forever."""
    from vc_object_store import ObjectStore
    path = tmp_path / 'tracks.vcobj'
    _history_off_store(path)
    db = ObjectStore.open(str(path), read_only=True, reader_pool_size=1)
    selected = osi.resolve_selection(db, osi.request_from_config(_tracks_only()))
    return db, selected.pins().refs(), frozenset(selected['track']), selected.position


def _reads_again(db, refs, position):
    done, box = _within(10, lambda: db.selected_inputs(refs, position.provenance()))
    return done and 'error' not in box


def _inputs_of(db, config=None):
    selected = osi.resolve_selection(db, osi.request_from_config(config or _tracks_only()))
    return selected.pins().refs(), selected.position.provenance()


def _by_object(result):
    return {**result, 'objects': sorted(result['objects'], key=lambda revision: revision['object_uuid'])}


@pytest.mark.parametrize('size', [1, 2, 3, 5, 100])
def test_a_sliced_read_gives_the_unsliced_result(db, size):
    uuids = _store(db)
    refs, position = _inputs_of(db, {**reference.fit_config(), 'input_use_tracks': True})
    related = [ref for ref in refs if ref['object'] not in uuids]
    assert related and len(related) < len(refs)
    whole = db.selected_inputs(refs, position, relationship_refs=related)
    seen = []

    def keep(part):
        seen.append(len(part))
        return list(part)
    sliced = db.selected_inputs(refs, position, relationship_refs=related, slice_size=size, on_slice=keep)
    assert _by_object(sliced) == _by_object(whole) and sliced['relationships'] and sliced['track_datasets']
    assert max(seen) <= size and sum(seen) == len(refs)
    assert [revision['object_uuid'] for revision in sliced['objects']][:len(related)] == [
        ref['object'] for ref in related]  # the objects whose relationships are read come first
    # What on_slice returns is what is kept.
    assert len(db.selected_inputs(refs, position, relationship_refs=related, slice_size=size,
                                  on_slice=lambda part: part[:1])['objects']) == len(seen)


def test_slicing_is_given_with_both_arguments_and_a_list_back(db):
    uuids = _store(db)
    refs, position = _inputs_of(db)
    with pytest.raises(ValueError, match='together'):
        db.selected_inputs(refs, position, relationship_refs=[], slice_size=2)
    with pytest.raises(ValueError, match='together'):
        db.selected_inputs(refs, position, relationship_refs=[], on_slice=list)
    with pytest.raises(ValueError, match='needs relationship_refs'):
        db.selected_inputs(refs, position, slice_size=2, on_slice=list)
    with pytest.raises(TypeError, match='callable'):
        db.selected_inputs(refs, position, relationship_refs=[], slice_size=2, on_slice=3)
    with pytest.raises(TypeError, match='must return a list'):
        db.selected_inputs(refs, position, relationship_refs=[], slice_size=2, on_slice=lambda part: None)
    assert uuids


def test_an_object_selected_twice_is_refused_across_slices(tmp_path):
    db, refs, _, position = _one_reader_store(tmp_path)
    with db:
        with pytest.raises(ValueError, match='selected twice'):
            db.selected_inputs(refs + refs[:1], position.provenance(), relationship_refs=[], slice_size=1,
                               on_slice=list)
        assert _reads_again(db, refs, position)


def test_a_pin_stale_before_the_call_is_refused_by_it(tmp_path):
    from vc_object_store import ObjectStore
    path = tmp_path / 'tracks.vcobj'
    _history_off_store(path)
    with ObjectStore.open(str(path), author='fitter', reader_pool_size=1) as db:
        refs, position = _inputs_of(db)
        _set_meta(db, refs[-1]['object'], 'review_note', 'seen')
        with pytest.raises(ValueError, match='not retained|has no version'):
            db.selected_inputs(refs, position, relationship_refs=[], slice_size=1, on_slice=list)
        assert _reads_again(db, refs[:1], osi.resolve_selection(
            db, osi.request_from_config(_tracks_only())).position)


def test_the_read_is_over_when_the_call_returns(tmp_path):
    db, refs, _, position = _one_reader_store(tmp_path)
    with db:
        result = db.selected_inputs(refs, position.provenance(), relationship_refs=[], slice_size=1, on_slice=list)
        assert len(result['objects']) == 2
        assert _reads_again(db, refs, position)


def test_an_error_in_on_slice_comes_out_unchanged_and_ends_the_read(tmp_path):
    db, refs, _, position = _one_reader_store(tmp_path)

    class Boom(Exception):
        pass
    calls = []

    def fails(part):
        calls.append(len(part))
        if len(calls) == 2:
            raise Boom('second slice')
        return list(part)
    with db:
        with pytest.raises(Boom, match='^second slice$') as failure:
            db.selected_inputs(refs, position.provenance(), relationship_refs=[], slice_size=1, on_slice=fails)
        assert failure.value.__traceback__ is not None  # held by the test: the read is over all the same
        assert calls == [1, 1]
        assert _reads_again(db, refs, position)


def test_a_second_slice_error_through_the_fit_reader_ends_the_read(tmp_path, monkeypatch):
    db, refs, tracks, position = _one_reader_store(tmp_path)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 1)
    real, cuts = osi._cut_track_revision, []

    def failing(revision, shared):
        cuts.append(1)
        if len(cuts) == 2:
            raise RuntimeError('cut failed')
        return real(revision, shared)
    monkeypatch.setattr(osi, '_cut_track_revision', failing)
    with db:
        with pytest.raises(RuntimeError, match='cut failed'):
            osi._read_selected_inputs(db, refs, tracks, position)
        assert _reads_again(db, refs, position)


def _readers_together(path, pool, sliced, plain):
    """Run in a child process (a hang must not take the suite with it): ``sliced`` threads each make three sliced
    reads, ``plain`` threads three hundred plain ones, on one store with ``pool`` reader connections. Every read must give
    the objects of a single read; returns '' or what differed."""
    import time
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), read_only=True, reader_pool_size=pool) as db:
        refs, position = _inputs_of(db)
        want = _by_object(db.selected_inputs(refs, position, relationship_refs=[]))['objects']

        def slow(part):
            time.sleep(0.05)  # the other threads run meanwhile
            return list(part)
        wrong = []

        def sliced_reads():
            for _ in range(3):
                got = db.selected_inputs(refs, position, relationship_refs=[], slice_size=1, on_slice=slow)
                if _by_object(got)['objects'] != want:
                    wrong.append('sliced')

        def plain_reads():
            for _ in range(300):
                if _by_object(db.selected_inputs(refs, position, relationship_refs=[]))['objects'] != want:
                    wrong.append('plain')
        threads = [threading.Thread(target=plain_reads) for _ in range(plain)]
        threads += [threading.Thread(target=sliced_reads) for _ in range(sliced)]
        for thread in threads:
            thread.start()
            time.sleep(0.002)  # the plain readers are under way when the sliced ones begin
        for thread in threads:
            thread.join()
        return ','.join(wrong)


def _together_in_a_child(path, pool, sliced, plain, seconds=30, work=None):
    import multiprocessing
    context = multiprocessing.get_context('fork')
    queue = context.Queue()

    def run():
        try:
            queue.put(('ok', work(path, pool, sliced) if work else _readers_together(path, pool, sliced, plain)))
        except BaseException as error:  # noqa: BLE001
            queue.put(('error', repr(error)))
    child = context.Process(target=run, daemon=True)
    child.start()
    child.join(seconds)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail(f'{sliced} sliced and {plain} plain readers on {pool} reader connections hung for {seconds} s')
    assert queue.get(timeout=10) == ('ok', '')


@pytest.mark.parametrize('pool, sliced, plain', [(1, 2, 0), (2, 3, 0), (1, 1, 1), (2, 1, 2), (1, 2, 1), (2, 3, 2), (4, 2, 0)])
def test_sliced_and_plain_reads_together_finish_with_the_same_objects(tmp_path, pool, sliced, plain):
    path = tmp_path / 'tracks.vcobj'
    _history_off_store(path)
    _together_in_a_child(path, pool, sliced, plain)


def _readers_gated(path, pool, sliced):
    """In a child: ``sliced`` threads make a sliced read each, and their first on_slice waits (GIL released) until
    a PLAIN read (status(), then a selected_inputs) on another thread has started, then pauses a moment: the plain read then waits for a reader
    connection while every one is held by a sliced read. Returns '' or what differed."""
    import time
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), read_only=True, reader_pool_size=pool) as db:
        refs, position = _inputs_of(db)
        want = _by_object(db.selected_inputs(refs, position, relationship_refs=[]))['objects']
        started = threading.Event()
        status = db.status()
        wrong = []

        def sliced_read():
            waited = []

            def on_slice(part):
                if not waited:
                    waited.append(1)
                    started.wait()
                    time.sleep(0.2)
                return list(part)
            got = db.selected_inputs(refs, position, relationship_refs=[], slice_size=1, on_slice=on_slice)
            if _by_object(got)['objects'] != want:
                wrong.append('sliced')

        def plain_read():
            started.set()
            # status() takes its reader with the GIL held, as most plain reads of the store do.
            if db.status() != status or _by_object(db.selected_inputs(refs, position, relationship_refs=[]))['objects'] != want:
                wrong.append('plain')
        threads = [threading.Thread(target=sliced_read) for _ in range(sliced)]
        for thread in threads:
            thread.start()
        time.sleep(0.2)  # every sliced read holds its connection before the plain one begins
        threads.append(threading.Thread(target=plain_read))
        threads[-1].start()
        for thread in threads:
            thread.join()
        return ','.join(wrong)


@pytest.mark.parametrize('pool, sliced', [(1, 1), (2, 2)])
def test_a_plain_read_waiting_for_a_reader_does_not_stop_the_sliced_reads(tmp_path, pool, sliced):
    path = tmp_path / 'tracks.vcobj'
    _history_off_store(path)
    _together_in_a_child(path, pool, sliced, 0, seconds=30, work=_readers_gated)


hosted = pytest.importorskip('test_object_store_hosted')
needs_server = pytest.mark.skipif(not hosted.SERVER, reason='set VC_OBJECT_SERVER to a vc_object_server binary')


@needs_server
def test_the_same_edits_during_a_served_capture(tmp_path, monkeypatch):
    # Over HTTP: a history-off store edited after the first response.
    path = tmp_path / 'tracks.vcobj'
    tracks = _history_off_store(path)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 1)
    config = _tracks_only(object_store_location=str(path), object_store_author='fitter')
    with hosted.serving(path, tmp_path) as endpoint:
        config = dict(config, object_store_location=endpoint)
        with osi.open_store(config, cache_directory=str(tmp_path / 'cache')) as handle:
            request = osi.request_from_config(config)
            expected = osi.read_snapshot(handle, request)
            edited = expected.pins().refs()[-1]['object']
            store = _SavesAfterFirstRead(handle, lambda: hosted.set_meta(handle, edited, 'review_note', 'seen'))
            found = osi.read_snapshot(store, request)
            assert handle.current_version(edited) == _track_versions(expected)[edited] + 1
            assert _track_versions(found) == _track_versions(expected) and len(_track_versions(found)) == 2


@needs_server
def test_a_save_to_a_served_overlay_during_the_capture(tmp_path, monkeypatch):
    from vc_object_store import ObjectStore
    overlay = _writable_overlay(tmp_path)
    with ObjectStore.open(str(overlay), author='fitter') as db:
        unused = _unused_patch(db)
    monkeypatch.setattr(osi, 'TRACK_READ_BATCH', 1)
    config = _tracks_only(object_store_author='fitter')
    with hosted.serving(overlay, tmp_path) as endpoint:
        config = dict(config, object_store_location=endpoint)
        with osi.open_store(config, cache_directory=str(tmp_path / 'cache')) as handle:
            request = osi.request_from_config(config)
            expected = osi.read_snapshot(handle, request)
            store = _SavesAfterFirstRead(handle, lambda: hosted.set_meta(handle, unused, 'review_note', 'seen'))
            found = osi.read_snapshot(store, request)
            assert _track_versions(found) == _track_versions(expected) and len(_track_versions(found)) == 3
