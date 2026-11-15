"""Tracks read from an object store, and the crossings the fit computes from
them."""
import dataclasses
import json
import pathlib
import threading

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
import tracks as tracks_module  # noqa: E402
from object_store_fixture import (  # noqa: E402
    crossing_edge, fit_config, frame, put_patch, put_track, shared_edit,
    write_reference_store, write_walk_store)


pytestmark = pytest.mark.usefixtures('cpu')

CSR_ARRAYS = ('offsets', 'partners', 'self_local', 'partner_local', 'positions', 'clearances')


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


def _materialize_tracks(db, z_begin=0, z_end=64, cache_directory=None,
                        workers=None, **request):
    snapshot = osi.read_snapshot(db, osi.InputRequest(**request))
    return osi.materialize_tracks(db, snapshot, z_begin=z_begin, z_end=z_end,
                                  cache_directory=cache_directory,
                                  workers=workers)


def _move_last_point(db, uuid):
    """Another geometry revision of a track: its last point one voxel on."""
    revision = db.current_object_revision(uuid)['geometry_revision']
    points = osi.read_geometry(db, revision, ['points'])[0]['points']
    moved = points[-1:].copy()
    moved[0, 2] += 1
    edits = [{'array': 'points', 'element_offset': len(points) - 1, 'data': moved.tobytes()}]
    version = db.current_version(uuid)
    # Links anchored on the track (the store's own crossings) are kept.
    kept = []
    for affected in db.affected_geometry_edit_edges(uuid, edits, version):
        edge = db.edge(affected['uuid'])
        kept.append({'edge_uuid': edge['uuid'], 'expected_version': edge['version'],
                     'action': 'retain',
                     'a_revision': db.current_geometry(edge['a'])['uuid'],
                     'b_revision': db.current_geometry(edge['b'])['uuid']})
    db.edit_geometry(uuid, edits, version, edge_edits=kept)
    assert db.current_object_revision(uuid)['geometry_revision'] != revision


def _assert_same_crossings(found, expected):
    for name in CSR_ARRAYS:
        np.testing.assert_array_equal(found[name], expected[name], err_msg=name)


def _link_crossings(db, order):
    """The crossing CSR (stored voxels) of the store's crossing edges over
    the tracks ``order`` (UUIDs by row): what a fit read from the links the
    store's track-track detector published."""
    row = {uuid: index for index, uuid in enumerate(order)}
    records = []
    for edge in db.edges_by_kind('crossing'):
        meta = edge['meta']['track_crossing']
        ends = [(row[uuid], anchor['points'][0]['index'][0],
                 float(meta['endpoints'][uuid]['arclength_voxels']))
                for uuid, anchor in ((edge['a'], edge['a_anchor']), (edge['b'], edge['b_anchor']))]
        clearance = float(meta['clearance_voxels'])
        (a, a_local, a_position), (b, b_local, b_position) = ends
        records.append((a, b, a_local, b_local, a_position, clearance))
        records.append((b, a, b_local, a_local, b_position, clearance))
    table = np.array(records, dtype=np.float64).reshape(-1, 6)
    table = table[np.lexsort((table[:, 1], table[:, 0]))]
    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(np.bincount(table[:, 0].astype(np.int64), minlength=len(order)), out=offsets[1:])
    return {'offsets': offsets, 'partners': table[:, 1].astype(np.int32),
            'self_local': table[:, 2].astype(np.int32),
            'partner_local': table[:, 3].astype(np.int32),
            'positions': table[:, 4].copy(), 'clearances': table[:, 5].copy()}


def _varied_tracks(db):
    """Tracks of both families with a pair that crosses in two clusters, a
    repeated voxel at a crossing, a blank family, an empty track and a track
    leaving the [0, 64) z window; returns their UUIDs by row."""
    horizontal = _line((10, 60, 50), 2, 40)
    # Up along y at x=55, across at y=65, back down at x=75: two crossings
    # of the horizontal track, with the first crossing's voxel stored twice.
    up, down = _line((10, 55, 55), 1, 10), _line((10, 55, 75), 1, 10)[::-1]
    twice = np.concatenate([up[:6], up[5:], _line((10, 65, 56), 2, 18), down])
    return [
        put_track(db, horizontal, 0, 'horizontal'),
        put_track(db, twice, 1, 'vertical'),
        put_track(db, _line((10, 50, 62), 1, 20), 2, ''),
        put_track(db, np.empty((0, 3), dtype=np.int64), 3, 'vertical'),
        # Along z through the horizontal track's voxel (10, 60, 85), to z=100.
        put_track(db, _line((2, 60, 85), 0, 98), 4, 'vertical'),
        put_track(db, _line((10, 70, 50), 2, 40), 5, 'horizontal'),
        put_track(db, _line((10, 52, 80), 1, 30), 6, 'vertical'),
    ]


def test_computed_crossings_are_the_ones_the_stores_detector_links(db):
    """The crossings a fit computes from its tracks are the ones the store's
    track-track detector publishes as links for the same tracks."""
    order = _varied_tracks(db)
    computed, counts = _materialize_tracks(db)
    assert counts['tracks'] == 5 and counts['tracks outside the z window'] == 1
    report = db.link_tracks_to_tracks()
    assert report['edges_created'] > 0 and not report['problems']
    linked = _link_crossings(db, order)
    assert len(linked['partners']) // 2 == report['edges_created'] == counts['track crossings']
    _assert_same_crossings(computed.crossings, linked)
    # One representative for the pair crossing twice; the blank family and
    # the empty track have none; the z-excluded track keeps its row.
    partners = [computed.crossings['partners'][begin:end].tolist() for begin, end in zip(
        computed.crossings['offsets'][:-1], computed.crossings['offsets'][1:])]
    assert partners == [[1, 4, 6], [0], [], [], [0], [6], [0, 5]]
    assert computed.tracks.selected_source_ids.tolist() == [0, 1, 2, 5, 6]
    assert computed.derived['source'] == 'computed'
    assert computed.derived['count'] == counts['track crossings']
    # The store's crossing links are not read: the same crossings with them.
    again, _ = _materialize_tracks(db)
    _assert_same_crossings(again.crossings, computed.crossings)
    assert again.derived == computed.derived


def test_crossings_do_not_depend_on_the_worker_count(db):
    _varied_tracks(db)
    _lattice(db, dataset='lattice.vctracks')
    one, _ = _materialize_tracks(db, workers=1)
    many, _ = _materialize_tracks(db, workers=4)
    _assert_same_crossings(one.crossings, many.crossings)
    assert one.derived == many.derived and one.derived['count'] > 9


def test_tracks_pair_only_within_their_dataset(db):
    """Two datasets reusing the source keys, each with one crossing of its
    own at a voxel all four tracks share."""
    put_track(db, _line((10, 60, 50), 2, 40), 0, 'horizontal', dataset='a.vctracks')
    put_track(db, _line((10, 55, 55), 1, 32), 1, 'vertical', dataset='a.vctracks')
    put_track(db, _line((10, 60, 40), 2, 50), 0, 'horizontal', dataset='b.vctracks')
    put_track(db, _line((10, 50, 55), 1, 37), 1, 'vertical', dataset='b.vctracks')
    inputs, counts = _materialize_tracks(db)
    crossings = inputs.crossings
    assert counts['track crossings'] == 2
    # Global rows, partners remapped to them, each track's own point indices.
    assert crossings['source_ids'].tolist() == [0, 1, 2, 3]
    assert crossings['offsets'].tolist() == [0, 1, 2, 3, 4]
    assert crossings['partners'].tolist() == [1, 0, 3, 2]
    assert crossings['self_local'].tolist() == [5, 5, 15, 10]
    assert crossings['partner_local'].tolist() == [5, 5, 10, 15]
    assert crossings['positions'].tolist() == [5.0, 5.0, 15.0, 10.0]
    assert inputs.tracks.selected_source_ids.tolist() == [0, 1, 2, 3]


def test_spellings_of_one_dataset_are_one_group(db):
    """The store resolves a track's dataset (its owning store's directory /
    its dataset_ref), as its track-track detector does: two spellings of one
    path are one dataset, and their tracks cross."""
    put_track(db, _line((10, 60, 50), 2, 40), 0, 'horizontal', dataset='tracks.vctracks')
    put_track(db, _line((10, 55, 55), 1, 32), 1, 'vertical', dataset='./tracks.vctracks')
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    [dataset] = {obj.dataset for obj in snapshot.objects_of('track')}
    assert dataset.endswith('/tracks.vctracks') and '/./' not in dataset
    inputs, counts = _materialize_tracks(db)
    assert counts['track crossings'] == 1
    assert inputs.crossings['partners'].tolist() == [1, 0]
    # The store's own detector pairs them too.
    assert db.link_tracks_to_tracks()['edges_created'] == 1


def test_one_file_name_in_two_source_stores_is_two_datasets(tmp_path):
    """Tracks an overlay inherits from two stores that each name
    ``tracks.vctracks`` beside themselves belong to two datasets: they do
    not cross, and the stores may reuse source IDs."""
    from vc_object_store import write_extension_store
    owners = {}
    for name, families in (('one', ('horizontal', 'vertical')), ('two', ('vertical', 'horizontal'))):
        (tmp_path / name).mkdir()
        with ObjectStore.create(str(tmp_path / name / 'base.vcobj'), author='ingest') as base:
            # Source ID 0 is the same line in both stores, in opposite
            # families: one dataset would cross them at every voxel.
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
    assert counts['tracks'] == 4 and counts['track crossings'] == 2
    crossings = inputs.crossings
    # Each store's two tracks cross each other, and nothing else.
    assert crossings['offsets'].tolist() == [0, 1, 2, 3, 4]
    assert crossings['partners'].tolist() == [1, 0, 3, 2]


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
    # The CSR spans every selected track; positions and clearance are fit voxels.
    crossings = inputs.crossings
    assert crossings['offsets'].tolist() == [0, 0, 1, 2]
    assert crossings['partners'].tolist() == [2, 1]
    assert crossings['self_local'].tolist() == [10, 30]
    assert crossings['partner_local'].tolist() == [30, 10]
    assert crossings['positions'].tolist() == [5.0, 15.0]
    assert crossings['clearances'].tolist() == [5.0, 5.0]


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
    assert snapshot.relationships == []
    # The surface's link to a track is seen from the surface, whose other
    # endpoint the read does not hold; the crossing is not seen at all.
    assert snapshot.exclusions == {'inapplicable': 1}
    # Tracks without consumed relationships are no missing family.
    assert osi.empty_family_warnings(snapshot) == []
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
    return osi.input_manifest(snapshot, config, osi.NormalSource('none'), tracks=inputs.tracks), inputs


def test_crossing_links_of_a_store_change_nothing_and_are_never_read(db):
    patch = put_patch(db, np.stack(np.meshgrid(
        np.arange(36.0, 46.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    tracks = _lattice(db, z=40)
    config = {**reference.fit_config(), 'input_use_tracks': True}
    before, without = _manifest(db, config)
    assert db.link_tracks_to_tracks()['edges_created'] == 9
    recording = _Recording(db)
    after, with_links = _manifest(recording, config)
    assert recording.endpoints == {patch}
    assert not recording.endpoints & set(tracks)
    assert osi._compared(after) == osi._compared(before)
    assert osi.manifest_digest(after) == osi.manifest_digest(before)
    assert after['relationships'] == [] and after['derived_inputs'] == before['derived_inputs']
    assert after['derived_inputs']['track_crossing']['count'] == 9
    _assert_same_crossings(with_links.tracks.crossings, without.tracks.crossings)


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

def _crossing_pair(db, **second):
    put_track(db, _line((10, 60, 50), 2, 40), 0, 'horizontal')
    options = {'key': 1, 'family': 'vertical', **second}
    return put_track(db, _line((10, 55, 55), 1, 32), options.pop('key'), options.pop('family'),
                     **options)


@pytest.mark.parametrize('second, problem', [
    ({'family': 'diagonal'}, "invalid stored track family 'diagonal'"),
    ({'key': '07'}, 'must be a canonical decimal source ID: 07'),
    ({'key': 'seven'}, 'must be a canonical decimal source ID: seven'),
    ({'key': str(1 << 64)}, 'must be a canonical decimal source ID'),
    ({'coordinates': frame(2)}, "tracks.vctracks' are in different frames"),
])
def test_track_metadata_crossings_cannot_use_is_refused(db, second, problem):
    uuid = _crossing_pair(db, **second)
    with pytest.raises(osi.ObjectStoreInputError) as error:
        _materialize_tracks(db)
    assert problem in str(error.value) and (uuid in str(error.value) or 'frames' in problem)


def test_a_source_key_is_one_track_of_its_dataset(db):
    _crossing_pair(db, key=0, dataset='other.vctracks')
    _, counts = _materialize_tracks(db)
    assert counts['tracks'] == 2 and counts['track crossings'] == 0
    # One store refuses the same key twice in a dataset; tracks of several
    # owners could still bring it.
    tracks = osi.read_snapshot(db, osi.InputRequest()).objects_of('track')
    assert len({obj.dataset for obj in tracks}) == 2
    for obj in tracks:
        obj.dataset = '/data/tracks.vctracks'
    with pytest.raises(osi.ObjectStoreInputError,
                       match='duplicate registered track source ID 0 in dataset /data/tracks.vctracks'):
        osi._ordered_tracks(tracks)


def test_track_points_crossings_cannot_index_are_refused(db, monkeypatch):
    _crossing_pair(db)
    far = put_track(db, _line((10, 60, (1 << 20) - 3), 2, 6), 2, 'horizontal')
    with pytest.raises(osi.ObjectStoreInputError, match=f'track {far} has coordinates outside'):
        _materialize_tracks(db)
    read = osi.read_geometry

    def flat(store, revision, names, optional=()):
        arrays, attributes = read(store, revision, names, optional)
        return {'points': arrays['points'].reshape(-1)}, attributes
    monkeypatch.setattr(osi, 'read_geometry', flat)
    with pytest.raises(osi.ObjectStoreInputError, match=f'track {far} points have shape'):
        _materialize_tracks(db)


def test_store_tracks_need_the_native_crossing_kernel(db, monkeypatch):
    _crossing_pair(db)
    monkeypatch.setenv('VC_DISABLE_NATIVE_TRACK_CROSSINGS', '1')
    with pytest.raises(osi.ObjectStoreInputError, match='native kernel vc_spiral.track_crossings'):
        _materialize_tracks(db)


# ---------------------------------------------------------------------------
# The crossing cache
# ---------------------------------------------------------------------------

@pytest.fixture()
def computes(monkeypatch):
    """The number of crossing computations."""
    calls = []
    compute = osi._compute_track_crossings

    def counted(*args, **kwargs):
        calls.append(1)
        return compute(*args, **kwargs)
    monkeypatch.setattr(osi, '_compute_track_crossings', counted)
    return calls


def _cache_files(cache):
    directory = cache / osi.TRACK_CROSSING_CACHE_DIRECTORY
    return sorted(directory.glob('*.npz')) if directory.is_dir() else []


def test_crossings_are_cached_by_the_track_revisions(db, tmp_path, computes):
    cache = tmp_path / 'cache'
    tracks = _lattice(db)
    first, _ = _materialize_tracks(db, cache_directory=str(cache))
    [cached] = _cache_files(cache)
    assert cached.suffix == '.npz' and len(computes) == 1
    again, _ = _materialize_tracks(db, cache_directory=str(cache))
    assert len(computes) == 1 and again.derived == first.derived
    _assert_same_crossings(again.crossings, first.crossings)
    # A store commit that leaves the tracks' geometry alone is still a hit.
    uuid = next(iter(tracks))
    db.set_meta(uuid, 'review_note', 'seen')
    db.link_tracks_to_tracks()
    _materialize_tracks(db, cache_directory=str(cache))
    assert len(computes) == 1 and _cache_files(cache) == [cached]
    # Another geometry revision of one track is another key.
    _move_last_point(db, uuid)
    changed, _ = _materialize_tracks(db, cache_directory=str(cache))
    assert len(computes) == 2 and len(_cache_files(cache)) == 2
    assert changed.derived['count'] == first.derived['count']


def _damage(path, damage):
    data = path.read_bytes()
    if damage == 'garbage':
        path.write_bytes(b'not a zip archive')
    elif damage == 'empty':
        path.write_bytes(b'')
    elif damage == 'truncated':
        path.write_bytes(data[:len(data) // 2])
    elif damage == 'crc':
        # One byte of an array's data: the archive's CRC no longer matches.
        _, csr = tracks_module.read_crossing_csr_file(path)
        byte = data.index(csr['positions'].tobytes()) + 3
        path.write_bytes(data[:byte] + bytes([data[byte] ^ 0xff]) + data[byte + 1:])
    else:
        metadata, csr = tracks_module.read_crossing_csr_file(path)
        if damage == 'checksum':
            csr['positions'] = csr['positions'] + 1.0
        elif damage == 'dtype':
            csr['partners'] = csr['partners'].astype(np.int64)
            np.savez(path, metadata=np.asarray(json.dumps(metadata)), **csr)
            return
        elif damage == 'rows':
            csr = osi._empty_crossings(2)
            metadata['sha256'] = tracks_module.crossing_csr_sha256(csr)
        elif damage == 'key':
            metadata['key'] = '0' * 64
        tracks_module.write_crossing_csr_file(path, csr, metadata)


@pytest.mark.parametrize('damage', ['garbage', 'empty', 'truncated', 'crc', 'checksum', 'dtype',
                                    'rows', 'key'])
def test_a_damaged_cache_file_is_a_miss_and_is_rebuilt(db, tmp_path, computes, capsys, damage):
    cache = tmp_path / 'cache'
    _lattice(db)
    first, _ = _materialize_tracks(db, cache_directory=str(cache))
    [cached] = _cache_files(cache)
    intact = cached.read_bytes()
    _damage(cached, damage)
    assert cached.read_bytes() != intact
    again, _ = _materialize_tracks(db, cache_directory=str(cache))
    assert len(computes) == 2 and 'ignoring invalid track crossing cache' in capsys.readouterr().out
    assert again.derived == first.derived
    _assert_same_crossings(again.crossings, first.crossings)
    assert _cache_files(cache) == [cached] and cached.read_bytes() == intact


def test_a_process_that_waited_for_the_crossings_reads_them(db, tmp_path, computes, monkeypatch):
    """Ranks that miss one cache file together: one computes, the rest wait
    for its file."""
    from service_files import ExclusiveFileLock
    _lattice(db)
    first, _ = _materialize_tracks(db, cache_directory=str(tmp_path / 'first'))
    [written] = _cache_files(tmp_path / 'first')
    path = tmp_path / 'cache' / osi.TRACK_CROSSING_CACHE_DIRECTORY / written.name
    path.parent.mkdir(parents=True)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    result, waiting = [], threading.Event()

    def said(*args, **kwargs):
        if args and 'waiting for another process' in str(args[0]):
            waiting.set()
    monkeypatch.setattr(osi, 'print', said, raising=False)
    with ExclusiveFileLock(f'{path}.lock'):  # another process computes
        waiter = threading.Thread(target=lambda: result.append(osi.materialize_tracks(
            db, snapshot, z_begin=0, z_end=64, cache_directory=str(tmp_path / 'cache'))))
        waiter.start()
        assert waiting.wait(30) and not result  # it found the lock held
        path.write_bytes(written.read_bytes())  # ... and publishes its file
    waiter.join(30)
    assert not waiter.is_alive() and len(computes) == 1
    assert result[0][0].derived == first.derived


def test_concurrent_writers_leave_one_complete_cache_file(db, tmp_path):
    _lattice(db)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    inputs, _ = osi.materialize_tracks(db, snapshot, z_begin=0, z_end=64,
                                       cache_directory=str(tmp_path / 'first'))
    [written] = _cache_files(tmp_path / 'first')
    metadata, csr = tracks_module.read_crossing_csr_file(written)
    path = tmp_path / 'cache' / osi.TRACK_CROSSING_CACHE_DIRECTORY / written.name
    start = threading.Barrier(8)

    def write():
        start.wait()
        for _ in range(5):
            osi._store_track_crossings(str(path), metadata['key'], csr, metadata['sha256'])
    writers = [threading.Thread(target=write) for _ in range(8)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    assert _cache_files(tmp_path / 'cache') == [path]
    assert path.read_bytes() == written.read_bytes()
    loaded, sha256 = osi._load_track_crossings(str(path), metadata['key'], 6)
    assert sha256 == inputs.derived['sha256']


def test_a_warm_cache_serves_another_fit_level_and_frame_scale(db, tmp_path, computes):
    """The cache holds stored voxels: the same file gives another fit level,
    and tracks redeclared in another frame scale, their own fit voxels."""
    cache = str(tmp_path / 'cache')
    tracks = _lattice(db)
    full, _ = _materialize_tracks(db, cache_directory=cache)
    assert len(computes) == 1 and full.crossings['positions'].max() > 0
    half, _ = _materialize_tracks(db, cache_directory=cache, fit_volume_level=1)
    for uuid in tracks:
        db.set_meta(uuid, 'coordinates', frame(4))
    quarter, _ = _materialize_tracks(db, cache_directory=cache)
    assert len(computes) == 1
    for scaled, factor in ((half, 0.5), (quarter, 0.25)):
        assert scaled.derived == full.derived
        for name in ('offsets', 'partners', 'self_local', 'partner_local'):
            np.testing.assert_array_equal(scaled.crossings[name], full.crossings[name])
        for name in ('positions', 'clearances'):
            np.testing.assert_array_equal(scaled.crossings[name], full.crossings[name] * factor)


def test_crossings_do_not_depend_on_the_crossing_limits(db, tmp_path, computes):
    """The crossings are inputs of the fit, not of its run settings: with
    both crossing limits 0 they are computed and recorded all the same, so
    changing the limits later leaves the fit's inputs as they were."""
    cache = tmp_path / 'cache'
    _lattice(db, z=40)
    config = {**reference.fit_config(), 'input_use_tracks': True}
    snapshot = osi.read_snapshot(db, osi.request_from_config(config))

    def tracks(precompute, per_step):
        return osi.materialize_from_config(db, snapshot, {
            **config, 'track_crossing_precompute_max': precompute,
            'track_max_track_crossing_per_step': per_step}, str(cache))
    off, on = tracks(0, 0), tracks(8, 2)
    assert off.counts['track crossings'] == on.counts['track crossings'] == 9
    assert len(computes) == 1
    assert off.tracks.derived == on.tracks.derived and off.tracks.derived['source'] == 'computed'


# ---------------------------------------------------------------------------
# Manifest, identity and the link refresh
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


def test_the_manifest_records_the_derived_crossings(tmp_path):
    inputs = _load(_track_store(tmp_path / 'fit.vcobj'))
    derived = inputs.manifest['derived_inputs']['track_crossing']
    assert derived == inputs.tracks.derived
    assert derived['source'] == 'computed' and derived['count'] == 9 and len(derived['sha256']) == 64
    assert derived['algorithm'] == tracks_module.TRACK_CROSSING_ALGORITHM
    assert derived['parameters'] == tracks_module.TRACK_CROSSING_PARAMETERS
    assert not [rel for rel in inputs.manifest['relationships'] if rel['family'] == 'track_crossing']
    summary = osi.input_summary(inputs)
    assert summary['constraints']['track crossings'] == 9
    assert 'track_crossing' not in summary['relationships']
    assert 'track crossings 9' in '\n'.join(inputs.report())
    assert not any('track_crossing' in line for line in inputs.report())
    # Tracks without a crossing record a count of zero; no tracks, nothing.
    lone = _load(_track_store(tmp_path / 'lone.vcobj', lattice=False))
    assert lone.manifest['derived_inputs']['track_crossing']['count'] == 0
    assert len(lone.manifest['derived_inputs']['track_crossing']['sha256']) == 64
    none = osi.load_fit_inputs(fit_config(tmp_path / 'fit.vcobj'), log=lambda line: None)
    assert none.tracks is None and 'derived_inputs' not in none.manifest


def test_the_fit_caches_its_crossings_under_its_cache(tmp_path, computes):
    path = _track_store(tmp_path / 'fit.vcobj')
    cache = tmp_path / 'cache'
    first = osi.load_fit_inputs(fit_config(path, input_use_tracks=True), log=lambda line: None,
                                cache_directory=str(cache))
    [cached] = _cache_files(cache)
    again = osi.load_fit_inputs(fit_config(path, input_use_tracks=True), log=lambda line: None,
                                cache_directory=str(cache))
    assert len(computes) == 1 and again.digest == first.digest
    assert _cache_files(cache) == [cached]


def test_derived_crossings_identify_the_tracks_of_a_fit(tmp_path):
    path = _track_store(tmp_path / 'fit.vcobj')
    inputs = _load(path)
    again = _load(path, object_store_refresh_links=False)
    assert osi.track_identity(again.manifest) == osi.track_identity(inputs.manifest)
    assert osi.track_identity(inputs.manifest)['derived']['count'] == 9
    assert len(osi.track_identity(inputs.manifest)['objects']) == 6
    # Other derived crossings of the same tracks are other inputs.
    changed = {**inputs.manifest, 'derived_inputs': {'track_crossing': {
        **inputs.manifest['derived_inputs']['track_crossing'], 'sha256': '0' * 64}}}
    assert osi.track_identity(changed) != osi.track_identity(inputs.manifest)
    assert osi.manifest_digest(changed) != inputs.digest
    assert osi.manifest_differences(inputs.manifest, changed) == ['derived_inputs.track_crossing']
    limits = _load(path, track_crossing_precompute_max=0, track_max_track_crossing_per_step=0)
    assert osi.track_identity(limits.manifest) == osi.track_identity(inputs.manifest)


def test_a_checkpoint_with_crossing_relationships_reports_changed_inputs(tmp_path):
    """A checkpoint of a fit that read its crossings as relationships (no
    derived crossings, one track_crossing entry each) is of other inputs."""
    path = _track_store(tmp_path / 'fit.vcobj')
    inputs = _load(path)
    a, b = sorted(entry['uuid'] for entry in inputs.manifest['objects'] if entry['kind'] == 'track')[:2]
    recorded = {key: value for key, value in inputs.manifest.items() if key != 'derived_inputs'}
    recorded['relationships'] = [*recorded['relationships'], {
        'uuid': '5d0f6c1e-8d2a-4c57-9f10-6a5e4d3c2b1a', 'kind': 'crossing',
        'family': 'track_crossing', 'approval_state': 'pending', 'a': a, 'b': b,
        'a_geometry_revision': None, 'b_geometry_revision': None, 'content_sha256': '0' * 64,
        'version': 1, 'reference': {
            'store': inputs.manifest['store']['uuid'],
            'generation': inputs.manifest['store']['generation'],
            'relationship': '5d0f6c1e-8d2a-4c57-9f10-6a5e4d3c2b1a', 'version': 1}}]
    [problem] = osi.resume_problems({'object_store': recorded}, inputs)
    assert 'derived_inputs' in problem and 'relationships' in problem
    assert 'object_store_allow_changed_inputs' in problem
    assert osi.track_identity(recorded) != osi.track_identity(inputs.manifest)
    warnings = []
    assert osi.resume_problems({'object_store': recorded}, inputs, allow_changed=True,
                               log=warnings.append) == []
    assert 'derived_inputs' in warnings[0]


def _prepare(path, **options):
    """prepare_position; returns (prepared, the refresh's mode)."""
    config = fit_config(path, 'service', input_use_tracks=True)
    log = []
    with ObjectStore.open(path, author='service') as store:
        prepared = osi.prepare_position(store, config, osi.request_from_config(config),
                                        log=log.append, **options)
    [line] = [line for line in log if line.startswith('link refresh: ')
              and line.split(': ')[1] in ('skip', 'full', 'incremental')]
    return prepared, line.split(': ')[1]


def test_the_link_refresh_links_no_tracks(tmp_path):
    path = _track_store(tmp_path / 'fit.vcobj')
    config = fit_config(path, 'service', input_use_tracks=True)
    with ObjectStore.open(path, author='service') as store:
        snapshot = osi.read_snapshot(store, osi.request_from_config(config))
        assert len(snapshot.objects_of('track')) == 6
        planned = osi.link_plan(snapshot, config, osi.NormalSource('none'))
        assert osi.link_marker_selection(snapshot).keys() == {'surface', 'point_collection', 'fiber'}
    assert [outcome.detector for outcome in planned.outcomes] == [
        'pcl_patch', 'fiber_patch', 'patch_overlap']
    steps = planned.plan['detectors']
    assert [step['detector'] for step in steps] == ['point-collection-surface', 'surface-overlap']
    assert 'track' not in str(steps)
    prepared, mode = _prepare(path)
    assert mode == 'full'
    with ObjectStore.open(path, read_only=True) as db:
        marker = osi.read_link_marker(db)
        assert db.edges_by_kind('crossing') == []
    assert marker['version'] == osi.LINK_MARKER_VERSION == 3
    assert 'track' not in marker['selection']
    assert sorted(marker['key_parts']['detectors']) == [
        'fiber-surface', 'point-collection-surface', 'surface-overlap']
    assert [step['detector'] for step in marker['detectors']] == [
        'point-collection-surface', 'surface-overlap']
    assert len(prepared.pins.selection['track']) == 6


def _add_crossing(db):
    tracks = {record['track']['external_key']: record['uuid'] for record in db.objects_by_kind('track')}
    crossing_edge(db, tracks['0'], 5, 5.0, tracks['1'], 5, 5.0, 5.0)


def _approve_crossing(db):
    db.set_approval(db.edges_by_kind('crossing')[0]['uuid'], 'approved')


def _remove_crossing(db):
    db.remove_edge(db.edges_by_kind('crossing')[0]['uuid'])


def _edit_track(db):
    db.set_meta(db.objects_by_kind('track')[0]['uuid'], 'review_note', 'edited')


def _move_track(db):
    _move_last_point(db, db.objects_by_kind('track')[0]['uuid'])


def _link_tracks(db):
    db.link_tracks_to_tracks()


def _approve_link(db):
    db.set_approval(db.edges_by_kind('point-surface')[0]['uuid'], 'approved')


def test_tracks_and_crossing_links_do_not_stale_the_links(tmp_path):
    """Crossing links (the store's detector, or a review of one) and changed
    tracks leave the fit's links current; a deleted relationship, whose kind
    the store no longer tells, and a consumed kind still relink in full."""
    path = _track_store(tmp_path / 'fit.vcobj')
    assert _prepare(path)[1] == 'full'
    for change in (_add_crossing, _approve_crossing, _edit_track, _move_track, _link_tracks):
        shared_edit(path, change)
        assert _prepare(path)[1] == 'skip', change
    shared_edit(path, _remove_crossing)
    assert _prepare(path)[1] == 'full'
    assert _prepare(path)[1] == 'skip'
    shared_edit(path, _approve_link)
    assert _prepare(path)[1] == 'full'


def test_a_marker_of_the_old_format_is_replaced():
    position = osi.InputPosition('store', 'generation', 7)
    selection = {kind: {'count': 0, 'sha256': osi._uuids_sha256([]), 'uuids': []}
                 for kind in osi.LINKED_KINDS}
    key_parts = {'volume_uuid': 'volume'}
    marker = {'version': osi.LINK_MARKER_VERSION, 'store': 'store', 'generation': 'generation',
              'key': osi.content_sha256(key_parts), 'key_parts': key_parts, 'exact': True,
              'commit_sequence': 7, 'selection': selection}

    def plan(marker):
        return osi.plan_link_refresh(marker, position, 7, lambda after: [], key_parts, selection,
                                     set(), incremental=False)
    assert plan(marker).mode == 'skip'
    # A marker a fit with the track-track detector wrote: version 2, with a
    # track selection.
    old = {**marker, 'version': 2, 'selection': {**selection, 'track': {
        'count': 0, 'sha256': osi._uuids_sha256([]), 'selector': {'uuids': None, 'tags': None}}}}
    assert plan(old) == osi.LinkPlan('full', 'the link marker has another format')


class _Served:
    """A store file behind the reading interface of a served store: edges
    are selected by UUID in one request, never read one by one."""
    endpoint = 'http://store.invalid'

    def __init__(self, store):
        self._store = store
        self.selections = []

    def __getattr__(self, name):
        return getattr(self._store, name)

    def edge(self, uuid):
        raise AssertionError('a served store reads no edge by itself')

    def select_edges(self, edge_kind=None, edge_uuids=None, endpoint_uuids=None,
                     include_retired=False, fields=None):
        assert edge_kind is None and endpoint_uuids is None and include_retired
        self.selections.append((list(edge_uuids), list(fields)))
        found = [self._store.edge(uuid) for uuid in edge_uuids]
        return [{field: edge[field] for field in fields} for edge in found if edge is not None]


def test_changed_relationships_are_classified_in_one_projected_read(tmp_path):
    path = _track_store(tmp_path / 'fit.vcobj')
    prepared, _ = _prepare(path)
    shared_edit(path, _link_tracks)
    shared_edit(path, _approve_crossing)
    removed = []

    def remove(db):
        removed.append(db.edges_by_kind('crossing')[-1]['uuid'])
        db.remove_edge(removed[0])
    shared_edit(path, remove)
    with ObjectStore.open(path, read_only=True) as db:
        served = _Served(db)
        head = int(db.status()['commit_sequence'])
        changes = osi.changes_after(served, prepared.position, head)
        live = {edge['uuid'] for edge in db.edges_by_kind('crossing')}
    relationships = [change for change in changes if change.record_kind == 'relationship']
    [(uuids, fields)] = served.selections
    assert fields == ['uuid', 'kind'] and set(uuids) == live | set(removed) and len(live) == 8
    assert {change.kind for change in relationships if change.uuid in live} == {'crossing'}
    assert {change.kind for change in relationships if change.uuid in removed} == {None}
    tracks = [change for change in changes if change.record_kind == 'object']
    assert tracks and {change.kind for change in tracks} == {'track'}


def test_a_fit_prepares_store_tracks_with_their_crossings(db, tmp_path):
    """A fit reads the tracks, computes their crossings, builds its crossing
    graph and prepares the crossing index from it, after point exclusion as
    well."""
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    from tracks import PackedTrackCollection, prepare_main_phase_tracks

    write_reference_store(db, reference.load_reference(), fibers=False)
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
    assert context.track_graph.edge_count == 9
    for radius in (0.0, 2.0):
        prepared = prepare_main_phase_tracks(
            context.tracks, None, radius, 'cpu',
            anchor_tree=context.trusted_geometry_tree,
            sampling_config=context.track_sampling_config,
            track_families=context.track_families,
            track_source_ids=context.track_source_ids,
            track_graph=context.track_graph)
        assert int(prepared['crossing_index_stats']['directed_crossings']) == 18

    # Switched off, the fit reads no tracks even when they were materialized.
    off = dataclasses.replace(inputs)
    config.update(input_use_tracks=False)
    context = FitContext(
        FitConfig(config), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), object_store_inputs=off)
    context.load_host_inputs()
    assert context.tracks is None and context.track_graph is None
