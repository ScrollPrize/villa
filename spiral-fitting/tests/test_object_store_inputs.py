"""Selection and frames of object-store fit inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
import object_store_inputs as osi  # noqa: E402
from config import Config  # noqa: E402
from object_store_fixture import (  # noqa: E402
    VOLUME, array, frame, plane, put_patch, put_track)


def _grid(z=20.0, size=6, scale=1):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([np.full(rows.shape, z), rows * 4.0 + 10, cols * 4.0 + 10],
                    axis=-1)


def _track(db, **record):
    return put_track(db, [[30, 12, 10 + i] for i in range(6)], 0, 'horizontal', **record)


def request(**selectors):
    volume = selectors.pop('volume', None)
    return osi.InputRequest(
        {kind: osi.Selector(**value) for kind, value in selectors.items()},
        volume)


@pytest.mark.parametrize('value', [True, 1.5, '1', -1, 2**63])
def test_saved_positions_use_native_validation(db, value):
    position = dict(db.status(), commit_sequence=value)
    for decode in (osi.InputPosition.from_provenance, osi.InputPins.from_provenance):
        with pytest.raises((ValueError, RuntimeError), match='commit_sequence'):
            decode({'position': position})
    with pytest.raises((ValueError, RuntimeError), match='commit_sequence'):
        osi.InputPins.from_manifest({'position': position})
    assert osi.InputPosition.from_provenance(dict(db.status(), commit_sequence=0)).commit_sequence == 0


def test_default_selection_reads_every_active_object_in_uuid_order(db):
    patches = sorted(put_patch(db, _grid(z)) for z in (10.0, 20.0, 30.0))
    track = _track(db)
    db.upsert_object({'kind': 'surface', 'name': 'placeholder'})
    snapshot = osi.read_snapshot(db, request())
    assert [obj.uuid for obj in snapshot.objects_of('surface')] == patches
    assert [obj.uuid for obj in snapshot.objects_of('track')] == [track]
    assert snapshot.volume_uuid == VOLUME
    assert snapshot.volume_source == 'inferred'
    assert snapshot.notes == ['skipped 1 surfaces without geometry']


def test_unknown_selector_kind_is_refused():
    # Store kind "patch" became "surface"; a stale key must not select everything.
    with pytest.raises(ValueError, match='unknown object kinds'):
        request(patch={'uuids': ()})


def test_selection_preserves_inherited_source_identity_and_manifest_pins(db):
    """A collection attachment has its own position, not its inherited objects' identities.

    Delegate actual object/geometry/version reads to a real native store while exposing
    a distinct attachment identity. Native collection routing is covered by test_coherent_store.
    """
    import uuid
    import dataclasses
    from vc_object_store import object_reference

    patch = put_patch(db, _grid())
    source = object_reference(db.object(patch))
    class Attachment:
        def __getattr__(self, name):
            return getattr(db, name)

        def status(self):
            return {**db.status(), 'store': owner, 'generation': generation}

        def selected_inputs(self, refs, position, relationship_refs=None):
            # This fixture tests transport, not collection routing. Actual
            # inherited-position validation is native test_coherent_store.
            return db.selected_inputs(refs, dict(position, store=db.status()['store'], generation=db.status()['generation']),
                                      relationship_refs)

    owner, generation = str(uuid.uuid4()), str(uuid.uuid4())
    attachment = Attachment()
    snapshot = osi.read_snapshot(attachment, request())
    assert snapshot.pins().refs() == [source]
    assert snapshot.request.pins.refs() == [source]
    pins = osi.InputPins.from_manifest(snapshot.manifest())
    assert pins.refs() == [source]
    assert osi.InputPins.from_provenance(pins.provenance()).refs() == [source]
    again = osi.read_snapshot(attachment, dataclasses.replace(request(), pins=pins))
    assert again.digest() == snapshot.digest()
    with pytest.raises(ValueError, match='regenerate'):
        osi.InputPins.from_provenance(dict(position=osi.InputPosition.of_status(db.status()).provenance(),
                                          objects={patch: source['version']}))
    original = dict(source)
    with pytest.raises(ValueError, match='regenerate'):
        osi.InputPins.from_provenance(dict(position=dict(store=owner, generation=generation,
                                                       commit_sequence=0), objects={patch: 1}))
    assert source == original


def test_reference_only_pins_validate_boundary_and_carry_reference_unchanged(db):
    from vc_object_store import object_reference
    patch = put_patch(db, _grid())
    reference = object_reference(db.object(patch))
    value = dict(position=osi.InputPosition.of_status(db.status()).provenance(),
                 references={patch: reference}, selection={'surface': [patch]})
    pins = osi.InputPins.from_provenance(value)
    assert pins.refs()[0] is pins.references[patch]
    assert pins.provenance()['references'] is pins.references
    assert 'objects' not in pins.provenance()
    assert osi.InputPins.from_provenance(pins.provenance()).refs() == [reference]
    with pytest.raises(ValueError, match='inconsistent exact input reference'):
        osi.InputPins.from_provenance(dict(value, references={VOLUME: reference}))
    with pytest.raises(ValueError):
        osi.InputPins.from_provenance(dict(value, references={patch: dict(reference, version=0)}))


@pytest.mark.parametrize('bad', [[], None, '', 1, True])
def test_saved_pin_maps_are_not_coerced_to_empty(db, bad):
    base = dict(position=osi.InputPosition.of_status(db.status()).provenance())
    for field in ('references', 'selection'):
        with pytest.raises(ValueError, match='maps'):
            osi.InputPins.from_provenance(dict(base, **{field: bad}))


@pytest.mark.parametrize('bad', [{}, None, '', 1, True])
def test_saved_pin_selection_arrays_are_validated_once(db, bad):
    value = dict(position=osi.InputPosition.of_status(db.status()).provenance(),
                 references={}, selection={'surface': bad})
    with pytest.raises(ValueError, match='arrays'):
        osi.InputPins.from_provenance(value)
    assert osi.InputPins.from_provenance(dict(value, selection={'surface': []})).refs() == []


def test_explicit_selectors_tags_and_empty_matches(db):
    a = put_patch(db, _grid(10.0), tags=['keep'])
    b = put_patch(db, _grid(20.0))
    put_patch(db, _grid(30.0))
    snapshot = osi.read_snapshot(db, request(
        surface={'uuids': (b,), 'tags': ('keep',)},
        track={'tags': ('absent',)}, volume=VOLUME))
    assert [obj.uuid for obj in snapshot.objects_of('surface')] == sorted([a, b])
    # A selector matching nothing selects nothing, never everything.
    assert snapshot.objects_of('track') == []
    assert snapshot.manifest()['selection']['track']['count'] == 0


def test_explicit_selection_problems_are_reported_together(db):
    missing = '00000000-0000-4000-8000-00000000dead'
    empty = db.upsert_object({'kind': 'surface', 'name': 'placeholder'})
    track = _track(db)
    with pytest.raises(osi.ObjectStoreInputError) as error:
        osi.read_snapshot(db, request(surface={'uuids': (missing, track)}))
    assert len(error.value.problems) == 2
    with pytest.raises(osi.ObjectStoreInputError, match='has no geometry'):
        osi.read_snapshot(db, request(surface={'uuids': (empty,)}))


def test_the_volume_is_inferred_configured_or_refused(db):
    # Undeclared: it must be configured; an empty selection with one is valid.
    with pytest.raises(osi.ObjectStoreInputError, match='configure fit_volume_uuid'):
        osi.read_snapshot(db, request())
    assert osi.read_snapshot(db, request(volume=VOLUME)).objects == {}
    put_patch(db, _grid())
    _track(db)
    assert osi.read_snapshot(db, request()).volume_source == 'inferred'
    explicit = osi.read_snapshot(db, request(volume=VOLUME))
    assert (explicit.volume_uuid, explicit.volume_source) == (VOLUME, 'configured')
    with pytest.raises(osi.ObjectStoreInputError, match='does not match'):
        osi.read_snapshot(db, request(volume='other-volume'))
    # Conflicting volumes fail even when one is configured.
    put_patch(db, _grid(30.0), coordinates=frame(volume='other-volume'))
    for volume in (None, VOLUME):
        with pytest.raises(osi.ObjectStoreInputError, match='different volumes'):
            osi.read_snapshot(db, request(volume=volume))


def test_every_object_without_a_frame_is_reported(db):
    put_patch(db, _grid())
    first = put_patch(db, _grid(30.0), coordinates=None, name='first')
    second = _track(db, coordinates={'volume_uuid': VOLUME}, name='second')
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


def test_a_fit_level_converts_frames_to_its_voxels(db):
    patch = put_patch(db, _grid(), scale=(0.25, 0.5), coordinates=frame(2))
    config = Config().as_dict()
    config.update(fit_volume_level=2, input_use_outer_shell=False,
                  loss_weight_shell_outer=0, dense_spacing_mode='grad_mag')
    request = osi.request_from_config(config)
    assert request.fit_volume_level == 2 and request.fit_scale == 4
    obj = osi.read_snapshot(db, request).objects[patch]
    arrays, attributes = osi.read_geometry(db, obj.geometry_revision, ['points'])
    # Stored at 2x the volume's voxels; the fit's are 4 of the volume's.
    np.testing.assert_allclose(obj.frame.xyz_to_fit_zyx(arrays['points']),
                               _grid() / 4, atol=1e-4)
    # Grid steps per fit voxel: 4x those per volume voxel.
    np.testing.assert_allclose(obj.frame.grid_scale_to_fit(attributes['scale']),
                               [0.25 * 4, 0.5 * 4])


def test_manifest_records_frames_and_its_digest_follows_inputs(db, tmp_path):
    patch, other = (put_patch(db, plane(dx=dx), (0.25, 0.25)) for dx in (0.0, 8.0))
    track = _track(db, coordinates=frame(2))
    snapshot = osi.read_snapshot(db, request())
    manifest = snapshot.manifest()
    assert manifest['units'] == osi.FIT_UNITS
    assert {obj['uuid']: obj['scale_factor'] for obj in manifest['objects']} == {
        patch: 1, other: 1, track: 2}
    incomplete = json.loads(json.dumps(manifest))
    incomplete['objects'][0].pop('reference')
    for read in (osi.InputPins.from_manifest, osi.manifest_digest):
        with pytest.raises(ValueError, match='regenerate'):
            read(incomplete)
    assert osi.validated_manifest_references({'objects': []}) == {}
    original = manifest['objects'][0]
    conflicting = dict(original, version=original['version'] + 1,
                       reference=dict(original['reference'], version=original['version'] + 1))
    for duplicate in (original, conflicting):
        repeated = json.loads(json.dumps(manifest))
        repeated['objects'].append(duplicate)
        for read in (osi.InputPins.from_manifest, osi.manifest_digest):
            with pytest.raises(ValueError, match='regenerate'):
                read(repeated)
    repeated = json.loads(json.dumps(manifest))
    repeated['outer_shell'] = repeated['objects'][0]
    with pytest.raises(ValueError, match='duplicate object'):
        osi.validated_manifest_references(repeated)
    # A position-only request is current/position discovery, not partial pins.
    discovery = osi.InputPins.from_provenance({'position': snapshot.position.provenance()})
    assert not discovery.pinned
    assert snapshot.digest() == osi.read_snapshot(db, request()).digest()
    assert snapshot.digest() != osi.read_snapshot(db, request(surface={'uuids': (patch,)})).digest()

    report = tmp_path / 'report.json'
    assert osi.main([db.path, '--json', str(report)]) == 0
    written = json.loads(report.read_text())
    assert written['manifest']['volume_uuid'] == VOLUME


def test_manifest_validation_reuses_decoded_references(db, monkeypatch):
    import vc_object_store
    put_patch(db, _grid())
    snapshot = osi.read_snapshot(db, request())
    manifest = snapshot.manifest()
    decode = vc_object_store.manifest_references
    decoded = []
    def counted(value):
        result = decode(value)
        decoded.append(result['objects'])
        return result
    monkeypatch.setattr(vc_object_store, 'manifest_references', counted)
    pins = osi.InputPins.from_manifest(manifest)
    assert len(decoded) == 1 and pins.references is decoded[0]
    assert pins.references == {entry['uuid']: entry['reference'] for entry in manifest['objects']}
    decoded.clear()
    assert osi.resume_problems({'object_store': manifest}, SimpleNamespace(manifest=manifest)) == []
    assert len(decoded) == 2


def test_the_outer_shell_is_read_on_its_own_never_as_a_patch(db):
    patch = put_patch(db, _grid(10.0), tags=['keep'])
    shell = put_patch(db, _grid(30.0, size=8), tags=[osi.OUTER_SHELL_TAG, 'keep'])
    # Neither the default selection nor a tag it shares makes it a patch...
    for selected in (request(), request(surface={'tags': ('keep',)})):
        snapshot = osi.read_snapshot(db, selected)
        assert [obj.uuid for obj in snapshot.objects_of('surface')] == [patch]
        assert snapshot.outer_shell.uuid == shell
    # ...and naming it as one is refused.
    with pytest.raises(osi.ObjectStoreInputError, match='is the outer shell'):
        osi.read_snapshot(db, request(surface={'uuids': (shell,)}))
    # Its revision is part of the manifest and so of the digest.
    snapshot = osi.read_snapshot(db, request())
    assert snapshot.manifest()['outer_shell']['uuid'] == shell
    # Exact pins identify owners without changing the consumed-content digest.
    manifest = snapshot.manifest()
    legacy = json.loads(json.dumps(manifest))
    legacy['outer_shell'].pop('reference')
    for entry in legacy['objects']:
        entry.pop('reference')
    with pytest.raises(ValueError, match='regenerate'):
        osi._compared(legacy)
    assert any(line.startswith('outer shell: ') and shell in line
               for line in osi.snapshot_report(snapshot))
    moved = _grid(31.0, size=8)[..., ::-1].astype('<f4')
    db.put_geometry(shell, {'kind': 'surface', 'attributes': {'scale': [1, 1]},
                            'arrays': [array('points', moved, '<f4')]},
                    expected_version=db.current_version(shell))
    assert osi.read_snapshot(db, request()).digest() != snapshot.digest()
    # It materializes whole, in the fit frame, beside the patches.
    inputs = osi.materialize(db, osi.read_snapshot(db, request()), z_begin=0, z_end=64,
                             z_margin=0, erode_cells_default=0)
    assert list(inputs.patches) == [patch]
    assert inputs.outer_shell.zyxs.shape == (8, 8, 3)
    assert float(inputs.outer_shell.zyxs[0, 0, 0]) == 31.0


def test_a_committed_spiral_fit_is_never_a_fit_input(db):
    patch = put_patch(db, _grid(10.0), tags=['keep'])
    before = osi.read_snapshot(db, request())
    fit = put_patch(db, _grid(30.0, size=8), tags=[osi.SPIRAL_FIT_TAG, 'keep'])
    # Neither the default selection nor a tag it shares takes it in, so the
    # inputs a commit recorded still match the session's after the commit...
    for selected in (request(), request(surface={'tags': ('keep',)})):
        snapshot = osi.read_snapshot(db, selected)
        assert [obj.uuid for obj in snapshot.objects_of('surface')] == [patch]
    assert osi.read_snapshot(db, request()).digest() == before.digest()
    # ...and naming it is refused.
    with pytest.raises(osi.ObjectStoreInputError, match='is a committed Spiral fit'):
        osi.read_snapshot(db, request(surface={'uuids': (fit,)}))


def test_the_outer_shell_is_one_surface_and_required_when_the_fit_needs_it(db):
    put_patch(db, _grid(10.0))
    required = osi.InputRequest(outer_shell='required')
    with pytest.raises(osi.ObjectStoreInputError, match=f'no surface is tagged {osi.OUTER_SHELL_TAG}'):
        osi.read_snapshot(db, required)
    assert osi.read_snapshot(db, request()).outer_shell is None
    first = put_patch(db, _grid(30.0), tags=[osi.OUTER_SHELL_TAG])
    # A fit that does not read the shell ignores it.
    assert osi.read_snapshot(db, osi.InputRequest(outer_shell='off')).outer_shell is None
    second = put_patch(db, _grid(40.0), tags=[osi.OUTER_SHELL_TAG])
    with pytest.raises(osi.ObjectStoreInputError, match='2 surfaces are tagged') as refused:
        osi.read_snapshot(db, request())
    assert first in str(refused.value) and second in str(refused.value)
    # The configuration decides the mode.
    from config import Config
    assert osi.request_from_config(Config().as_dict()).outer_shell == 'required'
    assert osi.request_from_config(Config({
        'loss_weight_shell_outer': 0, 'dense_spacing_mode': 'grad_mag'}).as_dict()
    ).outer_shell == 'optional'
    assert osi.request_from_config(Config({
        'input_use_outer_shell': False, 'loss_weight_shell_outer': 0,
        'dense_spacing_mode': 'grad_mag'}).as_dict()).outer_shell == 'off'
