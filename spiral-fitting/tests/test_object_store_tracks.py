"""Tracks read from an object store."""
import dataclasses
import pathlib

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    array, crossing_edge, fit_config, frame, put_patch, put_track,
    write_reference_store, write_walk_store)


pytestmark = pytest.mark.usefixtures('cpu')


def _line(start, axis, length):
    """Integer ZYX voxels from ``start`` along one axis."""
    points = np.tile(np.asarray(start, dtype=np.int64), (length + 1, 1))
    points[:, axis] += np.arange(length + 1)
    return points


def _lattice(db, z=10, **options):
    """Horizontal tracks along x crossing vertical tracks along y at z;
    returns their (stored ZYX, family) by UUID."""
    tracks = {}
    for index, y in enumerate((60, 70, 80)):
        points = _line((z, y, 50), 2, 40)
        tracks[put_track(db, points, 2 * index, 'horizontal', **options)] = (points, 0)
    for index, x in enumerate((55, 68, 81)):
        points = _line((z, 55, x), 1, 32)
        tracks[put_track(db, points, 2 * index + 1, 'vertical', **options)] = (points, 1)
    return tracks


def _materialize_tracks(db, z_begin=0, z_end=64, **request):
    snapshot = osi.read_snapshot(db, osi.InputRequest(**request))
    return osi.materialize_tracks(db, snapshot, z_begin=z_begin, z_end=z_end)


def test_spellings_of_one_dataset_are_one_group(db):
    """The store resolves a track's dataset (its owning store's directory /
    its dataset_ref), as its track-track detector does: two spellings of one
    path are one dataset."""
    put_track(db, _line((10, 60, 50), 2, 40), 0, 'horizontal', dataset='tracks.vctracks')
    put_track(db, _line((10, 55, 55), 1, 32), 1, 'vertical', dataset='./tracks.vctracks')
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    [dataset] = {obj.dataset for obj in snapshot.objects_of('track')}
    assert dataset.endswith('/tracks.vctracks') and '/./' not in dataset
    inputs, counts = _materialize_tracks(db)
    assert counts['tracks'] == 2
    assert inputs.tracks.selected_source_ids.tolist() == [0, 1]


def test_a_track_placeholder_does_not_stop_a_selection(db):
    """A track registered without details or geometry is skipped by a
    default selection, as before; one a fit would consume is refused for
    what it lacks."""
    _lattice(db)
    placeholder = db.upsert_object({'kind': 'track', 'meta': {'coordinates': frame()}})
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    assert placeholder not in snapshot.objects and len(snapshot.objects_of('track')) == 6
    assert any('without geometry' in note for note in snapshot.notes)
    _, counts = _materialize_tracks(db)
    assert counts['tracks'] == 6
    db.put_geometry(placeholder, {'kind': 'track', 'attributes': {}, 'arrays': [
        array('points', np.asarray(_line((10, 60, 50), 2, 4))[:, ::-1], '<i4')]})
    with pytest.raises(osi.ObjectStoreInputError, match=f'track {placeholder} has no dataset_ref'):
        _materialize_tracks(db)


def test_one_file_name_in_two_source_stores_is_two_datasets(tmp_path):
    """Tracks an overlay inherits from two stores that each name
    ``tracks.vctracks`` beside themselves belong to two datasets, and the
    stores may reuse source IDs."""
    from vc_object_store import write_extension_store
    owners = {}
    for name, families in (('one', ('horizontal', 'vertical')), ('two', ('vertical', 'horizontal'))):
        (tmp_path / name).mkdir()
        with ObjectStore.create(str(tmp_path / name / 'base.vcobj'), author='ingest') as base:
            # Source ID 0 is the same line in both stores, in opposite
            # families: one dataset would refuse the repeated key.
            owners[put_track(base, _line((10, 60, 50), 2, 40), 0, families[0])] = name
            owners[put_track(base, _line((10, 55, 55), 1, 32), 1, families[1])] = name
            base.seal()
    (tmp_path / 'top').mkdir()
    overlay = tmp_path / 'top' / 'overlay.vcobj'
    write_extension_store(str(overlay), [str(tmp_path / name / 'base.vcobj') for name in ('one', 'two')],
                          [], {})
    with ObjectStore.open(str(overlay), read_only=True) as db:
        snapshot = osi.read_snapshot(db, osi.InputRequest())
        datasets = {obj.uuid: obj.dataset for obj in snapshot.objects_of('track')}
        assert {uuid: pathlib.Path(dataset).parent.name for uuid, dataset in datasets.items()} == owners
        inputs, counts = osi.materialize_tracks(db, snapshot, z_begin=0, z_end=64)
    assert counts['tracks'] == 4
    assert inputs.tracks.selected_source_ids.tolist() == [0, 1, 2, 3]


def test_tracks_keep_their_window_order_and_frame(db):
    # Stored voxels are twice the fit voxels.
    scaled = frame(2)
    first = put_track(db, _line((40, 120, 100), 2, 40), 10, 'horizontal',
                      coordinates=scaled)
    second = put_track(db, _line((40, 110, 130), 1, 20), 9, 'vertical',
                       coordinates=scaled)
    # Fit z 70..80: outside the [0, 64) window.
    put_track(db, _line((140, 120, 100), 0, 20), 2, 'vertical',
                        coordinates=scaled)
    # A crossing link of the store is not what the fit consumes.
    crossing_edge(db, first, 3, 3.0, second, 4, 4.0, 1.0)
    inputs, counts = _materialize_tracks(db)
    assert counts['tracks'] == 2 and counts['tracks outside the z window'] == 1
    collection = inputs.tracks
    # Rows follow the numeric source key: 2 (outside), 9, 10.
    assert collection.selected_source_ids.tolist() == [1, 2]
    assert collection.selected_family_codes.tolist() == [1, 0]
    np.testing.assert_array_equal(collection[0], _line((40, 110, 130), 1, 20) / 2)
    np.testing.assert_array_equal(collection[1], _line((40, 120, 100), 2, 40) / 2)
    assert collection.selected_arclengths.tolist() == [10.0, 20.0]
    assert collection.selected_tortuosities.tolist() == [1.0, 1.0]


def test_track_links_are_not_consumed(db):
    patch = put_patch(db, np.stack(np.meshgrid(
        np.arange(10.0, 20.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    track = put_track(db, _line((12, 60, 50), 2, 8), 0, 'horizontal')
    other = put_track(db, _line((12, 56, 54), 1, 8), 1, 'vertical')
    db.add_edge({'kind': 'same-surface', 'a': track, 'b': patch,
                 'a_anchor': {'points': [{'array': 'points', 'index': [2]}]},
                 'b_anchor': {'points': [{'array': 'points', 'index': [2, 2]}]},
                 'meta': {'track_surface': {}}})
    crossing_edge(db, track, 4, 4.0, other, 4, 4.0, 4.0)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    assert 'relationships' not in snapshot.manifest()
    assert not any('crossing' in line for line in osi.snapshot_report(snapshot))


class _Recording:
    """A store handle recording the object UUIDs every relationship read
    names."""

    def __init__(self, store):
        self._store = store
        self.endpoints = set()

    def __getattr__(self, name):
        return getattr(self._store, name)

    def selected_inputs(self, refs, position, relationship_refs=None):
        self.endpoints.update(
            ref['object'] for ref in (refs if relationship_refs is None else relationship_refs))
        return self._store.selected_inputs(refs, position, relationship_refs)

    def relationships_at_versions(self, refs):
        self.endpoints.update(ref['object'] for ref in refs)
        return self._store.relationships_at_versions(refs)

    def select_edges(self, *args, **kwargs):
        self.endpoints.update(kwargs.get('endpoint_uuids') or ())
        return self._store.select_edges(*args, **kwargs)

    def edges_of(self, uuid, *args, **kwargs):
        self.endpoints.add(uuid)
        return self._store.edges_of(uuid, *args, **kwargs)


def _manifest(store, config):
    snapshot = osi.read_snapshot(store, osi.request_from_config(config))
    inputs = osi.materialize_from_config(store, snapshot, config)
    return osi.input_manifest(snapshot, config), inputs


def test_crossing_links_of_a_store_change_nothing_and_are_never_read(db):
    put_patch(db, np.stack(np.meshgrid(
        np.arange(36.0, 46.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    _lattice(db, z=40)
    config = {**reference.fit_config(), 'input_use_tracks': True}
    before, without = _manifest(db, config)
    assert db.link_tracks_to_tracks()['edges_created'] == 9
    recording = _Recording(db)
    after, with_links = _manifest(recording, config)
    assert recording.endpoints == set()
    assert osi._compared(after) == osi._compared(before)
    assert osi.manifest_digest(after) == osi.manifest_digest(before)
    assert len(with_links.tracks.tracks) == len(without.tracks.tracks) == 6


def test_tracks_are_read_under_the_selected_inputs_guard(db):
    """Tracks are validated with the other inputs by the store's one
    selected read: retained versions, at a position the store still has."""
    from vc_object_store import StoreBusy
    tracks = _lattice(db)
    request = osi.InputRequest()
    selected = osi.resolve_selection(db, request)
    assert osi.read_snapshot(db, request, selected).objects.keys() == set(tracks)
    uuid = next(iter(tracks))
    gone = 'requested graph position is no longer available'
    for moved in (dataclasses.replace(selected.position,
                                      generation='0b0f3c1e-8d2a-4c57-9f10-6a5e4d3c2b1a'),
                  dataclasses.replace(selected.position,
                                      commit_sequence=selected.position.commit_sequence + 1)):
        with pytest.raises(StoreBusy, match=gone):
            osi.read_snapshot(db, request, dataclasses.replace(selected, position=moved))
    # A version the store does not retain is refused.
    missing = dataclasses.replace(selected, references={
        **selected.references, uuid: dict(selected.references[uuid], version=99)})
    with pytest.raises(ValueError, match=f'selected object is not retained: {uuid}'):
        osi.read_snapshot(db, request, missing)
    # A later head still holds the selected versions.
    db.set_meta(uuid, 'review_note', 'seen')
    assert db.status()['commit_sequence'] > selected.position.commit_sequence
    assert osi.read_snapshot(db, request, selected).objects.keys() == set(tracks)


# ---------------------------------------------------------------------------
# Input errors: nothing is dropped silently
# ---------------------------------------------------------------------------

def _track_pair(db, **second):
    put_track(db, _line((10, 60, 50), 2, 40), 0, 'horizontal')
    options = {'key': 1, 'family': 'vertical', **second}
    return put_track(db, _line((10, 55, 55), 1, 32), options.pop('key'), options.pop('family'),
                     **options)


@pytest.mark.parametrize('second, problem', [
    ({'family': 'diagonal'}, "invalid stored track family 'diagonal'"),
    ({'key': '07'}, 'must be a canonical decimal source ID: 07'),
    ({'key': 'seven'}, 'must be a canonical decimal source ID: seven'),
    ({'key': str(1 << 64)}, 'must be a canonical decimal source ID'),
])
def test_invalid_track_metadata_is_refused(db, second, problem):
    uuid = _track_pair(db, **second)
    with pytest.raises(osi.ObjectStoreInputError) as error:
        _materialize_tracks(db)
    assert problem in str(error.value) and uuid in str(error.value)


def test_a_source_key_is_one_track_of_its_dataset(db):
    _track_pair(db, key=0, dataset='other.vctracks')
    _, counts = _materialize_tracks(db)
    assert counts['tracks'] == 2
    # One store refuses the same key twice in a dataset; tracks of several
    # owners could still bring it.
    tracks = osi.read_snapshot(db, osi.InputRequest()).objects_of('track')
    assert len({obj.dataset for obj in tracks}) == 2
    for obj in tracks:
        obj.dataset = '/data/tracks.vctracks'
    with pytest.raises(osi.ObjectStoreInputError,
                       match='duplicate registered track source ID 0 in dataset /data/tracks.vctracks'):
        osi._ordered_tracks(tracks)


def test_track_points_of_another_shape_are_refused(db, monkeypatch):
    _track_pair(db)
    far = put_track(db, _line((10, 60, 90), 2, 6), 2, 'horizontal')
    read = osi.read_geometry

    def flat(store, revision, names, optional=()):
        arrays, attributes = read(store, revision, names, optional)
        return {'points': arrays['points'].reshape(-1)}, attributes
    monkeypatch.setattr(osi, 'read_geometry', flat)
    with pytest.raises(osi.ObjectStoreInputError, match=f'track {far} points have shape'):
        _materialize_tracks(db)


# ---------------------------------------------------------------------------
# Manifest and identity
# ---------------------------------------------------------------------------

def _track_store(path, lattice=True):
    """The walk store with a lattice of tracks at z=40 (or one track)."""
    write_walk_store(path)
    with ObjectStore.open(str(path), author='ingest') as db:
        if lattice:
            _lattice(db, z=40)
        else:
            put_track(db, _line((40, 60, 50), 2, 40), 0, 'horizontal')
    return str(path)


def _load(path, **overrides):
    return osi.load_fit_inputs(fit_config(path, input_use_tracks=True, **overrides),
                               log=lambda line: None)


def test_the_track_identity_of_a_fit(tmp_path):
    path = _track_store(tmp_path / 'fit.vcobj')
    inputs = _load(path)
    again = _load(path)
    assert osi.track_identity(again.manifest) == osi.track_identity(inputs.manifest)
    assert len(osi.track_identity(inputs.manifest)) == 6
    assert 'derived_inputs' not in inputs.manifest


def test_a_fit_prepares_store_tracks(db, tmp_path):
    """A fit reads the tracks and prepares them, with and without point
    exclusion."""
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    from tracks import PackedTrackCollection, prepare_main_phase_tracks

    write_reference_store(db)
    _lattice(db, z=40)
    config = reference.fit_config()
    # The tracks toggle selects them (by tag when given), and none when off.
    assert osi.request_from_config(config).selector('track').uuids == ()
    config.update(input_use_tracks=True)
    assert osi.request_from_config(config).selector('track').is_default
    assert osi.request_from_config({**config, 'object_store_track_tags': ['curated']}
                                   ).selector('track').tags == ('curated',)
    snapshot = osi.read_snapshot(db, osi.request_from_config(config))
    inputs = osi.materialize_from_config(db, snapshot, config)
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)
    context = FitContext(
        FitConfig(config), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), object_store_inputs=inputs)
    context.load_host_inputs()
    assert isinstance(context.tracks, PackedTrackCollection)
    assert len(context.tracks) == 6
    for radius in (0.0, 2.0):
        prepared = prepare_main_phase_tracks(
            context.tracks, None, radius, 'cpu',
            anchor_tree=context.trusted_geometry_tree,
            sampling_config=context.track_sampling_config)
        assert prepared['lengths'].numel() == 6

    # Switched off, the fit reads no tracks even when they were materialized.
    off = dataclasses.replace(inputs)
    config.update(input_use_tracks=False)
    context = FitContext(
        FitConfig(config), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), object_store_inputs=off)
    context.load_host_inputs()
    assert context.tracks is None
