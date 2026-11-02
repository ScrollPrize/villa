"""Committing a published preview generation to the object store.

A generation is written here as lasagna_publish leaves it on disk (a schema-3
manifest, the flattened tifxyz and the float32 winding map), registered as a
preview artifact, and committed through the service against a real store.
The session is a stand-in reporting real object-store inputs.
"""
import dataclasses
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('vc_object_store')
import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402
from vc_object_store.surfaces import read_surface  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import spiral_service  # noqa: E402
from lasagna_publish import flatten_profile  # noqa: E402
from object_store_fixture import VOLUME  # noqa: E402
from service_http import ApiError  # noqa: E402
from service_project import UMBILICUS  # noqa: E402
from test_object_store_startup import fit_config, store_path  # noqa: E402,F401

SESSION = 'spiral-1-0123456789'
MODEL_STATE = 'a' * 64
BASE_SHAPE = [64, 64, 64]
STEP = flatten_profile()['output_step_vx']


def _grid(rows=5, columns=7):
    """A flattened grid with a NaN vertex and a -1 vertex, both invalid."""
    r, c = np.mgrid[:rows, :columns].astype(np.float32)
    xyz = np.stack([10.0 + c * 2.5, 20.0 + r * 1.5, 30.0 + r + c], axis=-1)
    xyz[0, 0] = np.nan
    xyz[1, 2] = -1.0
    windings = np.where(c < 3, 4.0, 5.0).astype(np.float32)
    windings[0, 0] = windings[1, 2] = -1.0
    return xyz, windings


def publish(root, *, input_digest, input_position=None, xyz=None, windings=None,
            surface_meta=None, run_diff=False, **manifest):
    """Write one published generation as lasagna_publish does; returns its root."""
    root = Path(root)
    surface_id = 'preview-lasagna.tifxyz'
    surface = root / surface_id
    surface.mkdir(parents=True)
    if xyz is None:
        xyz, windings = _grid()
    for axis, name in enumerate(('x.tif', 'y.tif', 'z.tif')):
        Image.fromarray(np.ascontiguousarray(xyz[..., axis]), mode='F').save(surface / name)
    meta = {'scale': [1.0 / STEP, 1.0 / STEP], 'export_coordinate_scale': 1.0,
            'base_shape_zyx': BASE_SHAPE, 'winding_id_map': 'winding-ids.tif'}
    meta.update(surface_meta or {})
    meta = {key: value for key, value in meta.items() if value is not None}
    (surface / 'meta.json').write_text(json.dumps(meta))
    Image.fromarray(windings, mode='F').save(root / 'winding-ids.tif')
    bounds = [{'winding': 4, 'columns': [0, 2]}, {'winding': 5, 'columns': [3, 6]}]
    document = {
        'schema_version': 3, 'surface_id': surface_id,
        'surface_path': str(surface), 'output_step_vx': STEP,
        'flatten_profile': flatten_profile(), 'grid_shape': list(xyz.shape[:2]),
        'winding_ids': [4, 5], 'winding_id_map': 'winding-ids.tif',
        'winding_id_dtype': 'float32_integer', 'winding_bounds': bounds,
        'loss_maps': [], 'source_fit_iteration': 120,
        'model_state_sha256': MODEL_STATE, 'input_digest': input_digest,
        'input_position': input_position, 'base_shape_zyx': BASE_SHAPE}
    if run_diff:
        Image.fromarray(np.zeros((*xyz.shape[:2], 4), np.uint8), mode='RGBA').save(
            root / 'run-diff.png')
        document['run_diff'] = {'path': 'run-diff.png', 'changed_source_pixels': 3,
                                'supported_pixels': 3}
    document.update(manifest)
    document = {key: value for key, value in document.items() if value is not None}
    (root / 'manifest.json').write_text(json.dumps(document))
    return root


def _fit(binding, tmp_path):
    """A service bound to ``binding``, with a stand-in session whose inputs
    were read (linked and pinned) from its store."""
    config = fit_config(binding.path, object_store_author='svc')
    inputs = osi.load_fit_inputs(config, log=lambda line: None)
    assert inputs.manifest['position'] is not None
    output = tmp_path / 'output'
    output.mkdir()
    manifest_path = output / 'object_store_inputs.json'
    manifest_path.write_text(json.dumps({'digest': inputs.digest, 'manifest': inputs.manifest}))
    status = {'state': 'Idle', 'applied_config': config, 'input_digest': inputs.digest,
              'input_manifest_path': str(manifest_path), 'checkpoint_state': None}
    state = spiral_service.ServiceState(
        bindings=spiral_service.bind_service_inputs(output, tmp_path / 'cache'),
        object_store=binding)
    state.session = SimpleNamespace(status=lambda: dict(status), close=lambda: None)
    state.session_id = SESSION
    state.session_request = {'project': {
        'name': 's1', 'volume': VOLUME, 'base_shape_zyx': BASE_SHAPE,
        'umbilicus': {'uuid': UMBILICUS['uuid'], 'revision': UMBILICUS['revision'],
                      'points': 2}}}
    serial = iter(range(1, 1000))

    def generation(**kwargs):
        number = next(serial)
        kwargs.setdefault('input_digest', inputs.digest)
        kwargs.setdefault('input_position', inputs.manifest['position'])
        root = publish(output / '.spiral-published' / SESSION / f'generation-{number}', **kwargs)
        return state.artifacts.register_directory(
            'spiral-preview', SESSION, number, root, 'manifest.json',
            delete_root_on_prune=True)['id']
    return SimpleNamespace(state=state, store=binding.path, config=config, inputs=inputs,
                           status=status, generation=generation, output=output)


@pytest.fixture()
def fit(store_path, tmp_path):
    """A service sharing the store in place."""
    fit = _fit(spiral_service.bind_object_store(store_path, 'svc'), tmp_path)
    yield fit
    fit.state.close()


@pytest.fixture()
def hub(store_path, tmp_path):
    """The startup fixture's store as a hub, and where the service's
    replica of it goes."""
    vos.hub_init(store_path, 'ingest')
    return SimpleNamespace(hub=store_path,
                           replica=str(tmp_path / 'replicas' / 'service.vcobj'))


@pytest.fixture()
def hosted(hub, tmp_path):
    """A service bound to its own replica of the hub."""
    fit = _fit(spiral_service.bind_object_store(hub.replica, 'svc', hub.hub), tmp_path)
    fit.hub = hub.hub
    yield fit
    fit.state.close()


def settle(state, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        commit = state.status()['commit']
        if commit is not None and commit['state'] != 'running':
            return commit
        time.sleep(0.01)
    raise AssertionError('the commit did not finish')


def commit(fit, artifact):
    fit.state.commit({'artifact': artifact})
    return settle(fit.state)


def _committed(path):
    with ObjectStore.open(path, read_only=True) as db:
        return db.objects_by_tag(osi.SPIRAL_FIT_TAG)


def _channel(path, revision, name):
    with ObjectStore.open(path, read_only=True) as db:
        descriptor = db.geometry_revision(revision)
        array = next(a for a in descriptor['arrays'] if a['name'] == name)
        data = db.read_geometry_array(revision, name)
    return np.frombuffer(data, dtype=np.dtype(array['dtype'])).reshape(array['shape'])


def test_a_commit_is_the_published_generation_with_its_provenance(fit):
    assert fit.state.binding_advertisement()['capabilities']['commits'] is True
    result = commit(fit, fit.generation())
    assert result['state'] == 'done', result['error']
    assert result['reused'] is False and result['pushed'] is None
    uuid = result['surface_uuid']

    xyz, windings = _grid()
    valid = np.isfinite(xyz).all(axis=-1) & ~np.all(xyz == -1.0, axis=-1)
    surface = read_surface(fit.store, uuid, volume=VOLUME)
    assert surface.geometry_revision == result['geometry_revision']
    np.testing.assert_array_equal(surface.valid, valid)
    np.testing.assert_array_equal(surface.points[valid], xyz[valid])
    assert surface.scale == (1.0 / STEP, 1.0 / STEP)
    assert surface.channels == {'winding_id'}
    winding = _channel(fit.store, surface.geometry_revision, 'winding_id')
    assert winding.dtype == np.dtype('<i4')
    np.testing.assert_array_equal(winding, windings.astype(np.int32))

    record = surface.record
    meta = record['meta']
    assert record['created_by'] == 'svc'
    assert record['tags'] == [osi.SPIRAL_FIT_TAG]
    assert record['name'] == 's1-spiral-1-0123456789-g1'
    for key in ('winding_bounds', 'component_winding_ids', 'flatten_profile',
                'model_state_sha256', 'source_fit_iteration', 'input_digest',
                'input_position', 'spiral_api_version'):
        assert key in meta, key
    assert meta['component_winding_ids'] == [4, 5]
    assert meta['flatten_profile'] == flatten_profile()
    assert meta['model_state_sha256'] == MODEL_STATE
    assert meta['input_digest'] == fit.inputs.digest
    assert meta['input_position'] == fit.inputs.manifest['position']
    assert meta['umbilicus'] == {'uuid': UMBILICUS['uuid'], 'revision': UMBILICUS['revision']}
    assert meta['coordinates']['volume_uuid'] == VOLUME
    provenance = meta['provenance']
    assert provenance['tool'] == 'spiral-fitting'
    # The anchors, never the whole selection.
    assert provenance['sources'] == [VOLUME]
    # The configuration and the recorded position reproduce the selection.
    config = provenance['parameters']['config']
    position = osi.InputPosition.from_provenance(meta['input_position'])
    with ObjectStore.open(fit.store, read_only=True) as db:
        reproduced = osi.resolve_selection(db, dataclasses.replace(
            osi.request_from_config(config), position=position))
        current = osi.read_snapshot(db, osi.request_from_config(fit.config))
    assert reproduced == {kind: [obj.uuid for obj in fit.inputs.snapshot.objects_of(kind)]
                          for kind in osi.KINDS}
    # The commit is not an input: the session's selection and digest are
    # unchanged, and naming it is refused.
    assert uuid not in [obj.uuid for obj in current.objects_of('surface')]
    assert current.digest() == fit.inputs.snapshot.digest()
    with ObjectStore.open(fit.store, read_only=True) as db:
        with pytest.raises(osi.ObjectStoreInputError, match='committed Spiral fit'):
            request = osi.request_from_config(fit.config)
            osi.read_snapshot(db, dataclasses.replace(request, selectors={
                **request.selectors, 'surface': osi.Selector(uuids=(uuid,))}))


def test_the_checkpoint_of_the_same_model_state_is_recorded(fit):
    fit.status['checkpoint_state'] = {'path': '/out/checkpoints/a.ckpt',
                                      'model_state_sha256': MODEL_STATE}
    uuid = commit(fit, fit.generation())['surface_uuid']
    fit.status['checkpoint_state'] = {'path': '/out/checkpoints/b.ckpt',
                                      'model_state_sha256': 'b' * 64}
    other = commit(fit, fit.generation(model_state_sha256='c' * 64))['surface_uuid']
    with ObjectStore.open(fit.store, read_only=True) as db:
        parameters = {u: db.object(u)['meta']['provenance']['parameters'] for u in (uuid, other)}
    assert parameters[uuid]['checkpoint'] == '/out/checkpoints/a.ckpt'
    assert 'checkpoint' not in parameters[other]


def test_committing_the_same_fit_again_reuses_its_surface(fit):
    first = commit(fit, fit.generation())
    # Another generation of the same model state, inputs and flatten
    # profile is the same surface.
    again = commit(fit, fit.generation())
    assert again['state'] == 'done'
    assert again['reused'] is True
    assert again['surface_uuid'] == first['surface_uuid']
    assert again['geometry_revision'] == first['geometry_revision']
    # A different model state is a different surface.
    other = commit(fit, fit.generation(model_state_sha256='b' * 64))
    assert other['reused'] is False
    assert len(_committed(fit.store)) == 2


def _held_reads(monkeypatch):
    """Make the commit worker wait, once it holds the generation, until
    released. Returns (reading, release) events."""
    reading, release = threading.Event(), threading.Event()
    real = spiral_service.read_published_generation

    def held(root, entry_point):
        reading.set()
        assert release.wait(10)
        return real(root, entry_point)
    monkeypatch.setattr(spiral_service, 'read_published_generation', held)
    return reading, release


def test_concurrent_commits_write_one_surface(fit, monkeypatch):
    first, second = fit.generation(), fit.generation()
    reading, release = _held_reads(monkeypatch)
    fit.state.commit({'artifact': first})
    assert reading.wait(10)
    # One commit runs at a time; a second request is refused meanwhile...
    with pytest.raises(ApiError) as refused:
        fit.state.commit({'artifact': second})
    assert refused.value.status == 409
    release.set()
    assert settle(fit.state)['reused'] is False
    # ...and its retry finds the first one's surface.
    assert commit(fit, second)['reused'] is True
    assert len(_committed(fit.store)) == 1


@pytest.mark.parametrize('surface_meta, message', [
    ({'export_coordinate_scale': 0.5}, 'export_coordinate_scale 0.5'),
    ({'export_coordinate_scale': None}, 'export_coordinate_scale None'),
])
def test_a_surface_outside_the_fit_frame_is_refused(fit, surface_meta, message):
    result = commit(fit, fit.generation(surface_meta=surface_meta))
    assert result['state'] == 'failed'
    assert message in result['error']
    assert _committed(fit.store) == []


def test_a_shape_other_than_the_fit_volumes_is_refused(fit):
    fit.state.session_request['project']['base_shape_zyx'] = [64, 64, 65]
    result = commit(fit, fit.generation())
    assert result['state'] == 'failed'
    assert "is not the fit volume's shape" in result['error']
    assert _committed(fit.store) == []


def test_generations_a_commit_cannot_vouch_for_are_refused(fit):
    for kwargs, message in (
            ({'flatten_profile': None}, 'publish a new preview'),
            ({'input_digest': 'f' * 64}, 'since replaced')):
        result = commit(fit, fit.generation(**kwargs))
        assert result['state'] == 'failed'
        assert message in result['error']
    assert _committed(fit.store) == []


def test_a_pruned_generation_is_gone(fit):
    artifact = fit.generation()
    fit.state.artifacts.prune('spiral-preview', SESSION, 0)
    with pytest.raises(ApiError) as refused:
        fit.state.commit({'artifact': artifact})
    assert refused.value.status == 410
    assert fit.state.status()['commit'] is None
    assert _committed(fit.store) == []


def test_a_generation_pruned_during_the_commit_is_read_first(fit, monkeypatch):
    artifact = fit.generation()
    reading, release = _held_reads(monkeypatch)
    fit.state.commit({'artifact': artifact})
    assert reading.wait(10)
    root = fit.output / '.spiral-published' / SESSION / 'generation-1'
    fit.state.artifacts.prune('spiral-preview', SESSION, 0)
    assert root.is_dir()
    release.set()
    assert settle(fit.state)['state'] == 'done'
    # The last reader gone, the pruned root goes with it.
    assert not root.exists()


def test_a_commit_holds_no_overlays(fit):
    artifact = fit.generation(run_diff=True)
    # A diagnostics wave for the same generation, never read by a commit.
    diagnostics = fit.output / 'diagnostics-1'
    diagnostics.mkdir()
    (diagnostics / 'manifest.json').write_text(json.dumps({'loss_maps': [
        {'name': 'patch_radius', 'path': 'patch_radius.png', 'mean': 1.0}]}))
    fit.state.artifacts.register_directory(
        'spiral-preview-diagnostics', SESSION, 1, diagnostics, 'manifest.json')
    uuid = commit(fit, artifact)['surface_uuid']
    surface = read_surface(fit.store, uuid, volume=VOLUME)
    assert surface.channels == {'winding_id'}
    meta = surface.record['meta']
    assert not {'run_diff', 'loss_maps', 'loss_map_statistics'} & set(meta)


def test_vertices_the_store_cannot_hold_are_committed_invalid(fit):
    xyz, windings = _grid()
    xyz[2, 3, 2] = 0.0  # valid when flattened; the store needs z > 0
    uuid = commit(fit, fit.generation(xyz=xyz, windings=windings))['surface_uuid']
    surface = read_surface(fit.store, uuid, volume=VOLUME)
    assert not surface.valid[2, 3]
    assert int(np.count_nonzero(surface.valid)) == xyz.shape[0] * xyz.shape[1] - 3
    winding = _channel(fit.store, surface.geometry_revision, 'winding_id')
    # The winding map is the published one, whose bounds still hold.
    np.testing.assert_array_equal(winding, windings.astype(np.int32))
    assert any('1 vertices of the preview are invalid in the store' in event['text']
               for event in fit.state.events.read_after(0)['events'])


def test_only_a_preview_of_this_session_is_committed(fit):
    checkpoint = fit.output / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'manifest.json').write_text('{}')
    other = fit.state.artifacts.register_directory(
        'spiral-checkpoint', SESSION, 1, checkpoint, 'manifest.json')['id']
    with pytest.raises(ApiError, match='not a preview of this session'):
        fit.state.commit({'artifact': other})
    with pytest.raises(ApiError, match='artifact must name'):
        fit.state.commit({})
    assert fit.state.status()['commit'] is None


# ---------------------------------------------------------------------------
# A hub replica: commits are pushed with a plain push, or stay pending
# ---------------------------------------------------------------------------

def _names(path):
    with ObjectStore.open(path, read_only=True) as db:
        return {record['name']: record['uuid'] for record in db.objects_by_kind('surface')}


def _on_hub(hub):
    """UUIDs of the committed fits on ``hub``, read through a fresh clone."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        clone = str(Path(directory) / 'reader.vcobj')
        vos.hub_clone(hub, clone, 'reader')
        return sorted(record['uuid'] for record in _committed(clone))


def _hub_edges(hub, kind):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        clone = str(Path(directory) / 'reader.vcobj')
        vos.hub_clone(hub, clone, 'reader')
        with ObjectStore.open(clone, read_only=True) as db:
            return db.edges_by_kind(kind)


def _pending(path):
    with ObjectStore.open(path, read_only=True) as db:
        return db.hub_status()['pending']


class _HubAway:
    """The hub unreachable while inside: its file moved aside."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.rename(self.path.with_suffix('.away'))

    def __exit__(self, *exc):
        self.path.with_suffix('.away').rename(self.path)


def test_a_commit_reaches_the_hub(hosted):
    assert hosted.state.binding_advertisement()['capabilities']['commits'] is True
    result = commit(hosted, hosted.generation())
    assert result['state'] == 'done', result['error']
    assert result['pushed'] is True and result['push_error'] is None
    assert hosted.state.status()['pending_commits'] == []
    assert _on_hub(hosted.hub) == [result['surface_uuid']]


def test_unpublished_links_are_reverted_before_the_push(hosted):
    names = _names(hosted.store)
    # Links a refresh left behind when its revert failed.
    with ObjectStore.open(hosted.store, author='svc') as db:
        db.add_edge({'kind': 'related', 'a': names['near'], 'b': names['far'],
                     'a_anchor': {'whole_geometry': True}, 'b_anchor': {'whole_geometry': True}})
    result = commit(hosted, hosted.generation())
    assert result['pushed'] is True
    assert _hub_edges(hosted.hub, 'related') == []
    assert _on_hub(hosted.hub) == [result['surface_uuid']]
    assert _pending(hosted.store) == []


def test_a_revert_that_fails_writes_nothing(hosted, monkeypatch):
    def refuse(self):
        raise RuntimeError('the hub is locked')
    monkeypatch.setattr(ObjectStore, 'hub_revert', refuse)
    result = commit(hosted, hosted.generation())
    assert result['state'] == 'failed'
    assert 'the hub is locked' in result['error']
    assert _committed(hosted.store) == []


def test_a_commit_the_hub_did_not_receive_stays_pending(hosted):
    with _HubAway(hosted.hub):
        result = commit(hosted, hosted.generation())
        assert result['state'] == 'done'
        assert result['pushed'] is False and result['push_error']
        uuid = result['surface_uuid']
        assert hosted.state.status()['pending_commits'] == [uuid]
    assert _on_hub(hosted.hub) == []
    # Once the hub is back, the next input preparation publishes it with its
    # links.
    with ObjectStore.open(hosted.store, author='svc') as store:
        osi.prepare_position(store, hosted.config, osi.request_from_config(hosted.config),
                             log=lambda line: None)
    assert _pending(hosted.store) == []
    assert _on_hub(hosted.hub) == [uuid]


def test_repeating_a_commit_retries_its_push(hosted):
    artifact = hosted.generation()
    with _HubAway(hosted.hub):
        first = commit(hosted, artifact)
    assert first['pushed'] is False
    again = commit(hosted, artifact)
    assert again['reused'] is True and again['pushed'] is True
    assert again['surface_uuid'] == first['surface_uuid']
    assert hosted.state.status()['pending_commits'] == []
    assert _on_hub(hosted.hub) == [first['surface_uuid']]


def test_startup_pushes_pending_commits_and_refuses_anything_else(hosted, hub):
    with _HubAway(hosted.hub):
        uuid = commit(hosted, hosted.generation())['surface_uuid']
    hosted.state.close()
    # A restarted service starts, and publishes the commit.
    restarted = spiral_service.bind_object_store(hub.replica, 'svc', hub.hub)
    assert restarted.hub == hosted.state.object_store.hub
    assert _pending(hub.replica) == []
    assert _on_hub(hub.hub) == [uuid]
    # A surface the service did not commit is still refused.
    from vc_object_store.surfaces import create_surface
    xyz, _ = _grid()
    create_surface(hub.replica, np.nan_to_num(xyz, nan=-1.0), name='stray', scale=0.05,
                   volume=VOLUME, tool='other', author='svc')
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='unpublished edits'):
        spiral_service.bind_object_store(hub.replica, 'svc', hub.hub)


def test_a_commit_waits_for_a_link_refresh(hosted):
    # A refresh holds the store-write lock while it links and pushes.
    with hosted.state._store_write_lock:
        hosted.state.commit({'artifact': hosted.generation()})
        time.sleep(0.2)
        commit_state = hosted.state.status()['commit']
        assert commit_state['state'] == 'running'
        assert commit_state['stage_name'] == 'Waiting for the object store'
        assert _committed(hosted.store) == []
    assert settle(hosted.state)['pushed'] is True
