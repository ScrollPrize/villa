"""Tracks and their crossings read from an object store."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('vc_object_store')

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    crossing_edge, frame, put_patch, put_track, write_reference_store)


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


def test_refreshed_crossings_are_the_ones_spiral_finds(db):
    """The store's track-crossing detector, run by the link refresh, gives
    the crossing CSR Spiral's own exact-crossing builder finds on the same
    tracks."""
    from tracks import _build_crossing_partner_csr
    tracks = _lattice(db)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    outcomes = osi.refresh_links(db, snapshot, reference.fit_config(),
                                 osi.NormalSource('none'))
    [crossing] = [outcome for outcome in outcomes if outcome.detector == 'track_crossing']
    assert crossing.report['edges_created'] == 9 and not crossing.failed
    inputs, counts = _materialize_tracks(db)
    assert counts['tracks'] == 6 and counts['track crossings'] == 9

    # Spiral's builder over the same tracks, in the same (source key) order.
    order = sorted(tracks, key=lambda uuid: int(db.object(uuid)['track']['external_key']))
    expected = _build_crossing_partner_csr(
        [tracks[uuid][0] for uuid in order], [tracks[uuid][1] for uuid in order])
    for name in ('offsets', 'partners', 'self_local', 'partner_local',
                 'positions', 'clearances'):
        np.testing.assert_array_equal(inputs.crossings[name], expected[name], err_msg=name)
    assert inputs.tracks.selected_source_ids.tolist() == list(range(6))
    assert inputs.tracks.selected_family_codes.tolist() == [0, 1, 0, 1, 0, 1]


def test_tracks_keep_their_window_order_and_frame(db):
    # Stored voxels are twice the fit voxels.
    scaled = frame(2)
    first = put_track(db, _line((40, 120, 100), 2, 30), 10, 'horizontal',
                      coordinates=scaled)
    second = put_track(db, _line((40, 110, 130), 1, 20), 9, 'vertical',
                       coordinates=scaled)
    # Fit z 70..80: outside the [0, 64) window.
    put_track(db, _line((140, 120, 100), 0, 20), 2, 'vertical',
                        coordinates=scaled)
    crossing_edge(db, first, 30, 30.0, second, 10, 10.0, 6.0)
    inputs, counts = _materialize_tracks(db)
    assert counts['tracks'] == 2 and counts['tracks outside the z window'] == 1
    collection = inputs.tracks
    # Rows follow the numeric source key: 2 (outside), 9, 10.
    assert collection.selected_source_ids.tolist() == [1, 2]
    assert collection.selected_family_codes.tolist() == [1, 0]
    np.testing.assert_array_equal(collection[0], _line((40, 110, 130), 1, 20) / 2)
    np.testing.assert_array_equal(collection[1], _line((40, 120, 100), 2, 30) / 2)
    assert collection.selected_arclengths.tolist() == [10.0, 15.0]
    assert collection.selected_tortuosities.tolist() == [1.0, 1.0]
    # The CSR spans every selected track; positions and clearance are fit voxels.
    crossings = inputs.crossings
    assert crossings['offsets'].tolist() == [0, 0, 1, 2]
    assert crossings['partners'].tolist() == [2, 1]
    assert crossings['self_local'].tolist() == [10, 30]
    assert crossings['partner_local'].tolist() == [30, 10]
    assert crossings['positions'].tolist() == [5.0, 15.0]
    assert crossings['clearances'].tolist() == [3.0, 3.0]


def test_track_surface_links_are_not_consumed(db):
    patch = put_patch(db, np.stack(np.meshgrid(
        np.arange(10.0, 20.0), [60.0], np.arange(50.0, 60.0), indexing='ij'),
        axis=-1).reshape(10, 10, 3), (1, 1))
    track = put_track(db, _line((12, 60, 50), 2, 8), 0, 'horizontal')
    db.add_edge({'kind': 'same-surface', 'a': track, 'b': patch,
                 'a_anchor': {'points': [{'array': 'points', 'index': [2]}]},
                 'b_anchor': {'points': [{'array': 'points', 'index': [2, 2]}]},
                 'meta': {'track_surface': {}}})
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    assert snapshot.relationships == []
    assert snapshot.exclusions == {'unused kind same-surface': 1}


def test_a_fit_prepares_store_tracks_with_their_crossings(db, tmp_path):
    """A fit reads the tracks and crossings, builds its crossing graph and
    prepares the crossing index from it, after point exclusion as well."""
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
    db.link_tracks_to_tracks()
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
