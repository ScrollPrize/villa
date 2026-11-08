#!/usr/bin/env python3
"""Exercise a VC3D project store: transfer and immediate edits on a disposable database.

Requires the vc_object_store Python extension in PYTHONPATH. The input database
must contain at least one fiber; neither it nor the user's settings are modified.
Without --database, create a current-format fixture and check lazy span details.
The database is the project VC3D opens: it references one volume and holds a
surface bound to it and an Open Data catalog surface, so the bridge's surface
workflow (list, activate, rename, tags, review, editing a catalog surface in
place) runs against a real store. Two VC3D clients then share a copy of it
through vc_object_server (--object-server, default: beside --vc3d).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid as uuidlib

import numpy as np

from bridge_client import BridgeClient, BridgeError
from smoke_offscreen import create_test_volume
from vc3d_process import VC3DProcess
from vc_object_store import ObjectStore


def add_surface(store, name, volume, invalid_rows=0):
    """A 3x3 surface inside the 16^3 smoke volume, bound to `volume`, followed by
    `invalid_rows` rows of invalid points; returns its UUID."""
    rows, cols = np.mgrid[:3 + invalid_rows, :3]
    points = np.stack([cols * 2.0 + 4, rows * 2.0 + 4, np.full(rows.shape, 8.0)], axis=-1).astype('<f4')
    points[3:] = -1
    uuid = store.upsert_object(dict(kind='surface', name=name,
                                    meta=dict(coordinates=dict(volume_uuid=volume, scale_factor=1))))
    store.put_geometry(uuid, dict(kind='surface', attributes=dict(scale=[1, 1]), arrays=[
        dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())]))
    return uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vc3d', required=True)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--object-server', type=Path)
    args = parser.parse_args()
    server_binary = args.object_server or Path(args.vc3d).with_name('vc_object_server')
    with tempfile.TemporaryDirectory(prefix='vc3d-store-bridge-') as temporary:
        root = Path(temporary)
        database = root / 'smoke.vcobj'
        if args.database:
            with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(database) as target:
                    source.backup(target)
        else:
            fixture = root / 'fiber.json'
            fixture.write_text(json.dumps(dict(type='vc3d_fiber', version=4,
                uuid='12345678-1234-1234-1234-123456789abc', name='span details',
                control_points=[dict(position=[x, 1, 1]) for x in (1, 3, 5)],
                line_points=[[x, 1, 1] for x in range(1, 6)])))
            with ObjectStore.create(str(database), author='bridge-fixture') as store:
                assert not store.ingest_geometry_source('fiber', fixture)['problems']
        with ObjectStore.open(str(database), read_only=True) as store:
            count = len(store.objects_by_kind('fiber'))
        assert count > 0, 'The smoke fixture must contain fibers'
        # The database is the project: it references the volume, relative to itself.
        create_test_volume(root / 'volumes' / 'vol1', 'vol1')
        with ObjectStore.open(str(database), author='bridge-fixture') as store:
            store.add_project_resource('volume', 'volumes')
            user_surface = add_surface(store, 'smoke surface', 'vol1', invalid_rows=2)
            # A surface of another scan, registered to vol1 by a shift.
            registered_surface = add_surface(store, 'registered surface', 'scanb')
            store.register_volume_transform('scanb', 'vol1', [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 0]], 'smoke registration')
            # An Open Data catalog surface of the sample volume "catvol" (vol1 shows it at L0), as fetched.
            catalog_surface = add_surface(store, 'catalog surface', 'catvol')
            store.set_meta(catalog_surface, 'catalog', dict(sample='smp', segment='seg', url='https://example.invalid/seg'))
            store.set_meta(catalog_surface, 'catalog_geometry_revision', store.current_geometry(catalog_surface)['uuid'])
            store.set_meta(catalog_surface, 'catalog_write', 'fetched')
            store.register_volume_transform('catvol', 'vol1', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], 'catalog level 0')
        # A second store of the project, routed for no kind: imports go there explicitly.
        side = root / 'side.vcobj'
        ObjectStore.create(str(side), author='bridge-fixture').close()
        config = root / 'config'
        config.mkdir()
        (config / 'VC3D.ini').write_text(
            '[viewer]\nusername=bridge-smoke\nremote_cache_dir=' + str(root / 'remote-cache') +
            '\n[project]\nshow_open_data_catalog_on_startup=false\n')
        tag = 'bridge_smoke_' + str(os.getpid())

        def launch(index, store=database, endpoint=None):
            project = root / f'hosted-{index}.vcproj' if endpoint else store.with_suffix('.vcproj')
            entry = dict(id='main', location=endpoint, hosted=True) if endpoint else \
                dict(id='main', location=str(store), read_only=False)
            if not project.exists():
                project.write_text(json.dumps(dict(
                    format='vcproj', version=1, name='Store smoke',
                    stores=[entry] +
                           ([dict(id='side', location=str(side), read_only=False)] if store == database and not endpoint else []),
                    selections={kind: 'main' for kind in ('surface', 'fiber', 'point_collection', 'umbilicus', 'volume_transform')},
                    volumes=[dict(location=str(root / 'volumes'), tags=[])],
                    normal_grids=[], lasagna_datasets=[], view={})))
            process = VC3DProcess(args.vc3d,
                ['--agent-bridge', '--agent-bridge-name', f'vc3d-store-{os.getpid()}-{index}',
                 '--project', str(project)],
                env_overrides={'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM', 'offscreen'),
                               'XDG_CONFIG_HOME': str(config), 'VC3D_CONFIG_DIR': str(config)})
            client = None
            try:
                client = BridgeClient(process.wait_for_handshake(120))
                deadline = time.monotonic() + 120
                while True:
                    try:
                        result, _ = client.call('fiber.list', timeout=120)
                        if len(result['fibers']) == count:
                            return process, client, result['fibers']
                    except BridgeError as error:
                        if error.code != -32000:
                            raise
                    if time.monotonic() > deadline:
                        raise RuntimeError('Object-store catalog did not attach')
                    time.sleep(.1)
            except BaseException:
                if client:
                    client.close()
                process.terminate()
                print('\n'.join(process.tail_log(100)))
                raise

        process, client, fibers = launch(1)
        try:
            if not args.database:
                assert len(fibers[0]['spans']) == 2, 'Fresh fiber.list must include span details before any edit'
            exported = root / 'export.json'
            client.call('fiber.export', {'path': str(exported)})
            with ObjectStore.create(str(root / 'roundtrip.vcobj'), author='bridge-roundtrip') as store:
                report = store.ingest_geometry_source('fiber', exported)
                assert not report['problems'] and report['ingested'] == count, report
            fiber_id = fibers[0]['fiberId']
            client.call('fiber.set_tag', {'fiberId': fiber_id, 'tag': tag, 'enabled': True})
            with ObjectStore.open(str(database), read_only=True) as store:
                edited = [fiber for fiber in store.objects_by_kind('fiber') if tag in fiber['tags']]
                assert len(edited) == 1, 'Tag must persist before Save'
                uuid = edited[0]['uuid']
            with ObjectStore.open(str(database), author='external-smoke') as store:
                store.set_meta(uuid, 'smoke_external', True)
            time.sleep(.5)
            client.call('fiber.set_tag', {'fiberId': fiber_id, 'tag': tag + '_external', 'enabled': True})
            with ObjectStore.open(str(database), read_only=True) as store:
                edited = [fiber for fiber in store.objects_by_kind('fiber') if fiber['uuid'] == uuid]
                assert tag + '_external' in edited[0]['tags'], 'Edit after external commit must persist'
            client.call('fiber.save', {}, timeout=120)
            print(f'Attached {count} fibers; immediate edit, external journal refresh and Save barrier passed', flush=True)
            render(client, root, user_surface, registered_surface)
            surfaces(client, database, user_surface, catalog_surface)
            legacy = objects(client, root, database, side, user_surface, uuid)
            routed(client, root, legacy)
        finally:
            client.close()
            process.terminate()
        process, client, fibers = launch(2)
        try:
            assert len([fiber for fiber in fibers if tag in fiber['tags']]) == 1
            assert len([fiber for fiber in fibers if tag + '_external' in fiber['tags']]) == 1
            print('New VC3D process reopened the database and retained both edits', flush=True)
            # Imported objects keep their identities across a reopen.
            _, status = run_job(client, 'objects.export', {'uuid': legacy, 'path': str(root / 'reopened-legacy'),
                                                           'store': 'side'})
            assert status['state'] == 'succeeded', status
            assert json.loads((root / 'reopened-legacy' / 'meta.json').read_text())['uuid'] == legacy
            print('Imported objects export by the same UUID after a reopen', flush=True)
        finally:
            client.close()
            process.terminate()
        hosted(root, database, launch, tag, user_surface, server_binary)


def listed_surfaces(client, wanted):
    """The surface listing once it holds `wanted` (it is read in the background)."""
    deadline = time.monotonic() + 60
    while True:
        try:
            result, _ = client.call('segments.list', timeout=120)
            rows = {row['id']: row for row in result['segments']}
            if wanted <= rows.keys():
                return rows
        except BridgeError as error:
            if error.code != -32000:
                raise
        if time.monotonic() > deadline:
            raise RuntimeError(f'surfaces {wanted} were not listed')
        time.sleep(.1)


def expect_error(client, method, params, code):
    try:
        client.call(method, params, timeout=120)
    except BridgeError as error:
        assert error.code == code, (method, error)
        return error
    raise AssertionError(f'{method} must fail with {code}')


def surfaces(client, database, user_surface, catalog_surface):
    """Surfaces of the project are edited through the bridge, Open Data catalog
    surfaces among them, in place."""
    rows = listed_surfaces(client, {user_surface, catalog_surface})
    assert not rows[user_surface]['catalog'] and rows[user_surface]['bound'], rows[user_surface]
    assert rows[catalog_surface]['catalog'] and rows[catalog_surface]['bound'], rows[catalog_surface]

    activated, _ = client.call('segments.activate', {'segmentId': user_surface}, timeout=120)
    assert activated['segment']['id'] == user_surface and activated['segment']['active'], activated
    client.call('tags.set', {'segmentId': user_surface, 'tag': 'approved', 'enabled': True}, timeout=120)
    client.call('segments.rename', {'segmentId': user_surface, 'newName': 'renamed surface'}, timeout=120)
    with ObjectStore.open(str(database), read_only=True) as store:
        record = store.object(user_surface)
        assert 'approved' in record['tags'] and record['name'] == 'renamed surface', record
    deadline = time.monotonic() + 30
    while True:  # the review reads the listing, refreshed by the commit events
        review, _ = client.call('segments.review', {'filter': {'approved': True}}, timeout=120)
        if [row['id'] for row in review['segments']] == [user_surface]:
            break
        assert time.monotonic() < deadline, review
        time.sleep(.1)
    assert review['segments'][0]['reviewState'] == 'approved', review
    with ObjectStore.open(str(database), read_only=True) as store:
        before = store.current_geometry(user_surface)
    client.call('segment.crop_bounds', {'segmentId': user_surface}, timeout=120)
    with ObjectStore.open(str(database), read_only=True) as store:
        after = store.current_geometry(user_surface)
    assert after['parent_uuid'] == before['uuid'], (before, after)
    assert [array['shape'] for array in after['arrays'] if array['name'] == 'points'] == [[3, 3, 3]], after

    # A catalog surface is the project's like any other: tagged, renamed and edited in place.
    client.call('segments.activate', {'segmentId': catalog_surface}, timeout=120)
    enabled, _ = client.call('segmentation.enable_editing', {'enabled': True}, timeout=120)
    assert enabled['enabled'], enabled
    client.call('segmentation.enable_editing', {'enabled': False}, timeout=120)
    client.call('tags.set', {'segmentId': catalog_surface, 'tag': 'approved', 'enabled': True}, timeout=120)
    client.call('segments.rename', {'segmentId': catalog_surface, 'newName': 'renamed catalog surface'}, timeout=120)
    with ObjectStore.open(str(database), read_only=True) as store:
        record = store.object(catalog_surface)
        assert record['meta']['coordinates']['volume_uuid'] == 'catvol', record['meta']
        assert 'approved' in record['tags'] and record['name'] == 'renamed catalog surface', record
    rows = listed_surfaces(client, {catalog_surface})
    assert rows[catalog_surface]['catalog'], rows[catalog_surface]
    print('Surfaces: list, activate, tags, rename, review and crop edit the project; a catalog surface '
          'is edited in place', flush=True)


def run_job(client, method, params, timeout=120):
    """Start a job and wait for its terminal job.status record."""
    started, _ = client.call(method, params, timeout=120)
    deadline = time.monotonic() + timeout
    while True:
        status, _ = client.call('job.status', {'jobId': started['jobId']}, timeout=30)
        if status['state'] in ('succeeded', 'failed'):
            return started, status
        assert time.monotonic() < deadline, status
        time.sleep(.1)


def objects(client, root, database, side, surface, fiber):
    """Objects travel between native files and the project's stores: each kind
    exports and re-imports into a store routed for nothing with its UUID and
    frame; UUID-less sources import by namespace, stably; a binding scales.
    Returns the UUID of an imported legacy surface."""
    # Created and edited objects report their store identities next to their runtime ids.
    added, _ = client.call('points.add_collection', {'name': 'exported collection'})
    assert added['uuid'] and added['store'] == 'main', added
    committed, _ = client.call('points.commit', {'collection': 'exported collection',
                                                 'points': [{'x': 5, 'y': 5, 'z': 8}, {'x': 6, 'y': 5, 'z': 8}]})
    assert committed['uuid'] == added['uuid'], committed
    listed, _ = client.call('points.list', {'collection': 'exported collection'})
    assert listed['collections'][0]['uuid'] == added['uuid'] and listed['collections'][0]['store'] == 'main'
    listed, _ = client.call('fiber.list')
    [row] = [row for row in listed['fibers'] if row['uuid'] == fiber]
    assert row['store'] == 'main', row
    rows = listed_surfaces(client, {surface})
    assert rows[surface]['store'] == 'main' and rows[surface]['writable'], rows[surface]

    exports = root / 'exports'
    exports.mkdir()
    exported = {'surface': (surface, exports / 'surface'), 'fiber': (fiber, exports / 'fiber.json'),
                'point_collection': (added['uuid'], exports / 'collection.json')}
    for kind, (uuid, path) in exported.items():
        started, status = run_job(client, 'objects.export', {'uuid': uuid, 'path': str(path)})
        assert started['objectKind'] == kind and started['store'] == 'main', started
        assert status['state'] == 'succeeded', status
        with ObjectStore.open(str(database), read_only=True) as store:
            version = status['result']['version']
            assert version['object'] == uuid and version['version'] == store.current_version(uuid), status
            assert status['result']['geometryRevision'] == store.current_geometry(uuid)['uuid'], status
    meta = json.loads((exports / 'surface' / 'meta.json').read_text())
    assert meta['uuid'] == surface and meta['coordinates'] == dict(volume_uuid='vol1', scale_factor=1), meta
    collection = json.loads((exports / 'collection.json').read_text())
    [entry] = collection['collections'].values()
    assert entry['uuid'] == added['uuid'] and len(entry['points']) == 2, collection

    # Into a store routed for no kind, without binding or namespace: identities and frames are kept.
    for kind, (uuid, path) in exported.items():
        _, status = run_job(client, 'objects.import', {'kind': kind, 'path': str(path), 'store': 'side', 'tags': []})
        assert status['state'] == 'succeeded', status
        assert status['result']['uuids'] == [uuid] and status['result']['store'] == 'side', status
    with ObjectStore.open(str(database), read_only=True) as main, ObjectStore.open(str(side), read_only=True) as copy:
        for kind, (uuid, _) in exported.items():
            assert copy.object(uuid)['meta']['coordinates'] == main.object(uuid)['meta']['coordinates'], kind

    # The same UUID in two attached stores: the export names the candidates until one is chosen.
    error = expect_error(client, 'objects.export', {'uuid': surface, 'path': str(exports / 'either')}, -32602)
    assert sorted(error.data['candidates']) == ['main', 'side'], error.data
    _, status = run_job(client, 'objects.export', {'uuid': surface, 'path': str(exports / 'side'), 'store': 'side'})
    assert status['state'] == 'succeeded' and status['result']['store'] == 'side', status
    expect_error(client, 'objects.export', {'uuid': str(uuidlib.uuid4()), 'path': str(exports / 'none')}, -32007)
    expect_error(client, 'objects.export', {'uuid': fiber, 'path': str(exports / 'f.json'), 'validOnly': True}, -32602)
    expect_error(client, 'objects.export', {'uuid': fiber, 'path': 'relative.json'}, -32602)
    expect_error(client, 'objects.import', {'kind': 'surface', 'path': str(exports / 'surface'), 'store': 'nowhere'}, -32007)
    expect_error(client, 'objects.import', {'kind': 'surface', 'path': str(exports / 'surface'),
                                            'uuidNamespace': 'not-a-uuid'}, -32602)

    # VC3D-era surfaces name themselves, and declare no frame.
    legacy = root / 'legacy' / 'paths'
    for name in ('auto_grown_a', 'auto_grown_b'):
        shutil.copytree(exports / 'surface', legacy / name)
        meta = json.loads((legacy / name / 'meta.json').read_text())
        meta['uuid'] = name
        meta.pop('coordinates')
        (legacy / name / 'meta.json').write_text(json.dumps(meta))
    _, status = run_job(client, 'objects.import', {'kind': 'surface', 'path': str(legacy), 'store': 'side'})
    report = status['result']
    assert status['state'] == 'succeeded' and report['uuids'] == [] and len(report['problems']) == 2, status
    namespace = str(uuidlib.uuid4())
    derived = {name: str(uuidlib.uuid5(uuidlib.UUID(namespace), 'patch:' + name)) for name in ('auto_grown_a', 'auto_grown_b')}
    _, status = run_job(client, 'objects.import', {'kind': 'surface', 'path': str(legacy / 'auto_grown_a'), 'store': 'side',
                                                   'uuidNamespace': namespace, 'bind': {'scale': 2}})
    assert status['result']['uuids'] == [derived['auto_grown_a']], status
    _, status = run_job(client, 'objects.import', {'kind': 'surface', 'path': str(legacy), 'store': 'side',
                                                   'uuidNamespace': namespace})
    report = status['result']
    assert report['uuids'] == [derived['auto_grown_b']] and report['existing'] == [derived['auto_grown_a']], status
    _, status = run_job(client, 'objects.import', {'kind': 'surface', 'path': str(legacy), 'store': 'side',
                                                   'uuidNamespace': namespace})
    assert status['state'] == 'succeeded' and status['result']['ingested'] == 0, status
    assert sorted(status['result']['existing']) == sorted(derived.values()), status
    with ObjectStore.open(str(side), read_only=True) as store:
        bound = store.object(derived['auto_grown_a'])
        assert bound['meta']['coordinates'] == dict(volume_uuid='vol1', scale_factor=2), bound['meta']
        assert 'auto_grown_a' in bound['tags'], bound['tags']
        unbound = store.object(derived['auto_grown_b'])['meta']['coordinates']
        assert unbound['volume_uuid'] is None and unbound['scale_factor'] is None, unbound
        assert 'paths' in store.object(derived['auto_grown_b'])['tags']
    _, status = run_job(client, 'objects.export', {'uuid': derived['auto_grown_b'], 'path': str(exports / 'unbound')})
    meta = json.loads((exports / 'unbound' / 'meta.json').read_text())
    assert meta['coordinates'] == dict(volume_uuid=None, scale_factor=None), meta
    # An explicit binding that conflicts with the file's own frame imports nothing.
    conflict = root / 'conflict'
    shutil.copytree(exports / 'surface', conflict)
    meta = json.loads((conflict / 'meta.json').read_text())
    meta['uuid'] = str(uuidlib.uuid4())
    (conflict / 'meta.json').write_text(json.dumps(meta))
    _, status = run_job(client, 'objects.import', {'kind': 'surface', 'path': str(conflict), 'store': 'side',
                                                   'bind': {'scale': 3}})
    report = status['result']
    assert status['state'] == 'succeeded' and report['ingested'] == 0 and not report['existing'], status
    assert 'coordinate conflict' in ' '.join(report['problems']), status

    # A fiber without a UUID needs a namespace through the bridge.
    document = json.loads((exports / 'fiber.json').read_text())
    document.pop('uuid')
    (root / 'legacy' / 'fiber.json').write_text(json.dumps(document))
    _, status = run_job(client, 'objects.import', {'kind': 'fiber', 'path': str(root / 'legacy' / 'fiber.json'),
                                                   'store': 'side'})
    assert status['result']['uuids'] == [] and 'UUID namespace' in ' '.join(status['result']['problems']), status

    # Cancelling stops between objects with a partial report, or arrives after the last commit.
    many = root / 'many'
    for index in range(40):
        shutil.copytree(legacy / 'auto_grown_a', many / f'copy_{index:02}')
    started, _ = client.call('objects.import', {'kind': 'surface', 'path': str(many), 'store': 'side',
                                                'uuidNamespace': str(uuidlib.uuid4())}, timeout=120)
    try:
        client.call('job.cancel', {'jobId': started['jobId']}, timeout=30)
    except BridgeError as error:
        assert error.code == -32007, error  # it already finished
    deadline = time.monotonic() + 120
    while (status := client.call('job.status', {'jobId': started['jobId']}, timeout=30)[0])['state'] == 'running':
        assert time.monotonic() < deadline, status
        time.sleep(.1)
    report = status['result']
    if status['state'] == 'failed':
        assert report['cancelled'] and report['ingested'] < 40 and len(report['uuids']) == report['ingested'], status
    else:
        assert not report['cancelled'] and report['ingested'] == 40, status
    print(f"Objects: export/re-import of each kind, namespace-stable legacy import, binding and "
          f"cancellation ({status['state']} with {report['ingested']} of 40)", flush=True)
    return derived['auto_grown_a']


def routed(client, root, surface):
    """Objects imported into another store show once that store is routed for their kind."""
    assert surface not in listed_surfaces(client, set())
    result, _ = client.call('stores.route', {'kind': 'surface', 'storeId': 'side'}, timeout=120)
    assert result['routes']['surface'] == 'side', result
    rows = listed_surfaces(client, {surface})
    assert rows[surface]['store'] == 'side' and rows[surface]['bound'], rows[surface]
    activated, _ = client.call('segments.activate', {'segmentId': surface}, timeout=120)
    assert activated['segment']['store'] == 'side', activated
    shot, _ = client.call('screenshot.capture', {'filePath': str(root / 'imported.png')}, timeout=120)
    assert (root / 'imported.png').stat().st_size > 0, shot
    client.call('stores.route', {'kind': 'surface', 'storeId': 'main'}, timeout=120)
    print('Routing the import store lists and activates the imported surface', flush=True)


def render(client, root, *surface_ids):
    """Surfaces render in the open volume's voxels: one bound to it, and one of
    another scan through the store's registration."""
    for surface in surface_ids:
        out = root / ('render-' + surface)
        started, _ = client.call('render.tifxyz', {'segmentId': surface, 'outputFormat': 'tif_stack',
                                                   'outputDir': str(out), 'numSlices': 1}, timeout=120)
        deadline = time.monotonic() + 120
        while True:
            status, _ = client.call('job.status', {'jobId': started['jobId']}, timeout=30)
            if status['state'] in ('succeeded', 'failed'):
                break
            assert time.monotonic() < deadline, status
            time.sleep(.2)
        assert status['state'] == 'succeeded', status
        assert list((out / 'layers').glob('*.tif')), list(out.rglob('*'))
    print('Render: a bound surface and a registered one render through the store', flush=True)


def hosted(root, database, launch, tag, surface, server_binary):
    """Two VC3D clients of one vc_object_server exchange fiber and surface edits."""
    if not server_binary.exists():
        print(f'Hosted: skipped, no vc_object_server at {server_binary}', flush=True)
        return
    shared = root / 'shared.vcobj'
    with sqlite3.connect(database) as source, sqlite3.connect(shared) as target:
        source.backup(target)
    server = subprocess.Popen([str(server_binary), str(shared), '--port', '0'], stderr=subprocess.PIPE, text=True)
    line = server.stderr.readline()
    port = re.search(r':(\d+)\s*$', line)
    if not port:
        server.terminate()
        raise RuntimeError(f'vc_object_server did not start: {line}')
    endpoint = f'http://127.0.0.1:{port.group(1)}'

    def wait_for_tag(client, wanted):
        """Fiber IDs are per process: find the fiber by tag in this one."""
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result, _ = client.call('fiber.list', timeout=120)
            for fiber in result['fibers']:
                if wanted in fiber['tags']:
                    return fiber['fiberId']
            time.sleep(.5)
        raise RuntimeError(f'{wanted} did not arrive through the hosted store')

    first = first_client = second = second_client = None
    try:
        first, first_client, fibers = launch(3, endpoint=endpoint)
        second, second_client, _ = launch(4, endpoint=endpoint)
        first_id = fibers[0]['fiberId']
        first_client.call('fiber.set_tag', {'fiberId': first_id, 'tag': tag + '_first', 'enabled': True})
        second_id = wait_for_tag(second_client, tag + '_first')
        second_client.call('fiber.set_tag', {'fiberId': second_id, 'tag': tag + '_second', 'enabled': True})
        wait_for_tag(first_client, tag + '_second')
        # Surfaces travel the same way: a tag set in one client reaches the other.
        listed_surfaces(first_client, {surface})
        first_client.call('tags.set', {'segmentId': surface, 'tag': 'reviewed', 'enabled': True}, timeout=120)
        deadline = time.monotonic() + 90
        while 'reviewed' not in listed_surfaces(second_client, {surface})[surface]['tags']:
            assert time.monotonic() < deadline, 'surface tag did not arrive through the hosted store'
            time.sleep(.5)
        print('Two clients of one hosted store exchanged fiber and surface edits', flush=True)
    finally:
        for client, process in ((first_client, first), (second_client, second)):
            if client:
                client.close()
            if process:
                process.terminate()
        server.terminate()
        server.wait(timeout=30)


if __name__ == '__main__':
    main()
