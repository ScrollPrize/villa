"""Selection, frames and relationship filtering of object-store fit inputs."""
import json

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
from object_store_fixture import (  # noqa: E402
    VOLUME, branch, frame, pcl_patch_edge, put_fiber, put_patch, put_pcl)


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / 'inputs.vcobj'), author='test',
                            hostname='host') as store:
        yield store


def _grid(z=20.0, size=6, scale=1):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([np.full(rows.shape, z), rows * 4.0 + 10, cols * 4.0 + 10],
                    axis=-1)


def _line(n=6, z=30.0):
    return [[z, 12.0, 10.0 + 4 * i] for i in range(n)]


def request(**selectors):
    volume = selectors.pop('volume', None)
    approved = selectors.pop('approved', False)
    return osi.InputRequest(
        {kind: osi.Selector(**value) for kind, value in selectors.items()},
        volume, approved)


def test_default_selection_reads_every_active_object_in_uuid_order(db):
    patches = sorted(put_patch(db, _grid(z)) for z in (10.0, 20.0, 30.0))
    pcl = put_pcl(db, {3: ([20.0, 14.0, 14.0], None)})
    fiber = put_fiber(db, _line(), [0, 5])
    db.upsert_object({'kind': 'patch', 'name': 'placeholder'})
    snapshot = osi.read_snapshot(db, request())
    assert [obj.uuid for obj in snapshot.objects_of('patch')] == patches
    assert [obj.uuid for obj in snapshot.objects_of('point_collection')] == [pcl]
    assert [obj.uuid for obj in snapshot.objects_of('fiber')] == [fiber]
    assert snapshot.volume_uuid == VOLUME
    assert snapshot.volume_source == 'inferred'
    assert snapshot.notes == ['skipped 1 patches without geometry']


def test_explicit_selectors_tags_and_empty_matches(db):
    a = put_patch(db, _grid(10.0), tags=['keep'])
    b = put_patch(db, _grid(20.0))
    put_patch(db, _grid(30.0))
    snapshot = osi.read_snapshot(db, request(
        patch={'uuids': (b,), 'tags': ('keep',)},
        fiber={'tags': ('absent',)}, volume=VOLUME))
    assert [obj.uuid for obj in snapshot.objects_of('patch')] == sorted([a, b])
    # A selector matching nothing selects nothing, never everything.
    assert snapshot.objects_of('fiber') == []
    assert snapshot.manifest()['selection']['fiber']['count'] == 0


def test_explicit_selection_problems_are_reported_together(db):
    missing = '00000000-0000-4000-8000-00000000dead'
    empty = db.upsert_object({'kind': 'patch', 'name': 'placeholder'})
    pcl = put_pcl(db, {0: ([20.0, 14.0, 14.0], None)})
    with pytest.raises(osi.ObjectStoreInputError) as error:
        osi.read_snapshot(db, request(patch={'uuids': (missing, pcl)}))
    assert len(error.value.problems) == 2
    with pytest.raises(osi.ObjectStoreInputError, match='has no geometry'):
        osi.read_snapshot(db, request(patch={'uuids': (empty,)}))


def test_volume_inference(db):
    put_patch(db, _grid())
    put_pcl(db, {0: ([20.0, 14.0, 14.0], None)})
    assert osi.read_snapshot(db, request()).volume_source == 'inferred'
    explicit = osi.read_snapshot(db, request(volume=VOLUME))
    assert (explicit.volume_uuid, explicit.volume_source) == (VOLUME, 'configured')
    with pytest.raises(osi.ObjectStoreInputError, match='does not match'):
        osi.read_snapshot(db, request(volume='other-volume'))


def test_conflicting_volumes_fail_even_when_one_is_configured(db):
    put_patch(db, _grid())
    put_patch(db, _grid(30.0), coordinates=frame(volume='other-volume'))
    for volume in (None, VOLUME):
        with pytest.raises(osi.ObjectStoreInputError, match='different volumes'):
            osi.read_snapshot(db, request(volume=volume))


def test_undeclared_volume_requires_configuration(db):
    with pytest.raises(osi.ObjectStoreInputError, match='configure fit_volume_uuid'):
        osi.read_snapshot(db, request())
    # An empty selection with a configured volume is valid.
    assert osi.read_snapshot(db, request(volume=VOLUME)).objects == {}
    put_patch(db, _grid(), coordinates=None)
    with pytest.raises(osi.ObjectStoreInputError, match='configure fit_volume_uuid'):
        osi.read_snapshot(db, request())


def test_every_object_without_a_frame_is_reported(db):
    put_patch(db, _grid())
    first = put_patch(db, _grid(30.0), coordinates=None, name='first')
    second = put_pcl(db, {0: ([20.0, 14.0, 14.0], None)},
                     coordinates={'volume_uuid': VOLUME}, name='second')
    with pytest.raises(osi.ObjectStoreInputError) as error:
        osi.read_snapshot(db, request())
    problems = '\n'.join(error.value.problems)
    assert first in problems and second in problems
    assert 'unknown geometry coordinate frame' in problems


@pytest.mark.parametrize('scale', [0.25, 1, 2])
def test_frame_conversion_round_trips_mixed_scales(db, scale):
    grid = _grid()
    grid[0, 0] = -1
    patch = put_patch(db, grid, scale=(0.25, 0.5), coordinates=frame(scale))
    put_patch(db, _grid(40.0), coordinates=frame(1))
    snapshot = osi.read_snapshot(db, request())
    obj = snapshot.objects[patch]
    assert obj.frame.scale_factor == scale
    arrays, attributes = osi.read_geometry(db, obj.geometry_revision, ['points'])
    points = arrays['points']
    invalid = np.all(points == -1, axis=-1)
    zyxs = obj.frame.xyz_to_fit_zyx(points, invalid)
    np.testing.assert_allclose(zyxs, grid, atol=1e-4)
    assert (zyxs[0, 0] == -1).all()
    np.testing.assert_allclose(
        obj.frame.grid_scale_to_fit(attributes['scale']), [0.25, 0.5])


def test_relationship_filtering_and_endpoint_order(db):
    patch = put_patch(db, _grid())
    other_patch = put_patch(db, _grid(40.0))
    pcl = put_pcl(db, {0: ([20.0, 10.0, 10.0], None), 1: ([20.0, 14.0, 14.0], None),
                       2: ([20.0, 18.0, 18.0], None)})
    a = put_fiber(db, _line(), [0, 5], [10, 20])
    b = put_fiber(db, _line(z=34.0), [0, 5], [30, 40])
    pending = pcl_patch_edge(db, pcl, 0, patch, (0.0, 0.0))
    approved = pcl_patch_edge(db, pcl, 1, patch, (1.0, 1.0))
    db.set_approval(approved, 'approved')
    rejected = pcl_patch_edge(db, pcl, 2, patch, (2.0, 2.0))
    db.set_approval(rejected, 'rejected')
    outside = pcl_patch_edge(db, pcl, 1, other_patch, (1.0, 1.0))
    ordinary = branch(db, b, 30, a, 20)
    branch(db, a, 10, b, 40, adjacent=True)

    snapshot = osi.read_snapshot(db, request(patch={'uuids': (patch,)}))
    consumed = {rel.uuid: rel for rel in snapshot.relationships}
    assert set(consumed) == {pending, approved, ordinary}
    assert outside not in consumed
    assert snapshot.exclusions == {
        'rejected': 1, 'endpoint outside selection': 1, 'adjacent fiber branch': 1}
    edge = consumed[pending]
    assert (edge.family, edge.a, edge.b) == ('pcl_patch', patch, pcl)
    assert edge.b_anchor['points'][0]['array'] == 'point_ids'
    fiber_edge = consumed[ordinary]
    assert (fiber_edge.a, fiber_edge.b) == tuple(sorted([a, b]))

    approved_only = osi.read_snapshot(db, request(
        patch={'uuids': (patch,)}, approved=True))
    assert [rel.uuid for rel in approved_only.relationships] == [approved]
    assert approved_only.exclusions['pending (approved-only)'] == 2


def test_manifest_records_frames_and_its_digest_follows_inputs(db, tmp_path):
    patch = put_patch(db, _grid(), coordinates=frame(2))
    pcl = put_pcl(db, {0: ([20.0, 14.0, 14.0], None)})
    edge = pcl_patch_edge(db, pcl, 0, patch, (1.0, 1.0))
    snapshot = osi.read_snapshot(db, request())
    manifest = snapshot.manifest()
    assert manifest['units'] == osi.FIT_UNITS
    assert {obj['uuid']: obj['scale_factor'] for obj in manifest['objects']} == {
        patch: 2, pcl: 1}
    assert manifest['relationships'][0]['uuid'] == edge
    assert snapshot.digest() == osi.read_snapshot(db, request()).digest()
    db.set_approval(edge, 'approved')
    assert snapshot.digest() != osi.read_snapshot(db, request()).digest()

    report = tmp_path / 'report.json'
    assert osi.main([db.path, '--json', str(report)]) == 0
    written = json.loads(report.read_text())
    assert written['manifest']['volume_uuid'] == VOLUME
    assert any('pcl_patch 1' in line for line in written['report'])
