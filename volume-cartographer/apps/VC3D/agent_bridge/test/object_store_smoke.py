#!/usr/bin/env python3
"""Exercise a VC3D project store: transfer and immediate edits on a disposable database.

Requires the vc_object_store Python extension in PYTHONPATH. The input database
must contain at least one fiber; neither it nor the user's settings are modified.
Without --database, create a current-format fixture and check lazy span details.
The database is the project VC3D opens: it references one volume and holds a
surface bound to it and an Open Data catalog surface, so the bridge's surface
workflow (list, activate, rename, tags, review, editing a catalog surface in
place) runs against a real store. Hosted replicas of it open as projects too.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

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
    args = parser.parse_args()
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
        config = root / 'config'
        config.mkdir()
        (config / 'VC3D.ini').write_text(
            '[viewer]\nusername=bridge-smoke\nremote_cache_dir=' + str(root / 'remote-cache') +
            '\n[project]\nshow_open_data_catalog_on_startup=false\n')
        tag = 'bridge_smoke_' + str(os.getpid())

        def launch(index, store=database):
            process = VC3DProcess(args.vc3d,
                ['--agent-bridge', '--agent-bridge-name', f'vc3d-store-{os.getpid()}-{index}',
                 '--project', str(store)],
                env_overrides={'QT_QPA_PLATFORM': 'offscreen',
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
        finally:
            client.close()
            process.terminate()
        process, client, fibers = launch(2)
        try:
            assert len([fiber for fiber in fibers if tag in fiber['tags']]) == 1
            assert len([fiber for fiber in fibers if tag + '_external' in fiber['tags']]) == 1
            print('New VC3D process reopened the database and retained both edits', flush=True)
        finally:
            client.close()
            process.terminate()
        hosted(root, database, launch, tag, user_surface)


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


def hosted(root, database, launch, tag, surface):
    """Two VC3D replicas of one hub exchange edits; a lease held elsewhere blocks editing."""
    import vc_object_store as vos
    hub = root / 'share' / 'hub.vcobj'
    hub.parent.mkdir()
    with sqlite3.connect(database) as source, sqlite3.connect(hub) as target:
        source.backup(target)
    vos.hub_init(str(hub), 'curator')
    for name in ('first', 'second', 'script'):
        vos.hub_clone(str(hub), str(root / f'{name}.vcobj'), name)

    def wait_for_tag(client, wanted):
        """Fiber IDs are per process: find the fiber by tag in this one."""
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result, _ = client.call('fiber.list', timeout=120)
            for fiber in result['fibers']:
                if wanted in fiber['tags']:
                    return fiber['fiberId']
            time.sleep(.5)
        raise RuntimeError(f'{wanted} did not arrive through the hub')

    first, first_client, fibers = launch(3, root / 'first.vcobj')
    second = second_client = None
    try:
        second, second_client, _ = launch(4, root / 'second.vcobj')
        first_id = fibers[0]['fiberId']
        first_client.call('fiber.set_tag', {'fiberId': first_id, 'tag': tag + '_first', 'enabled': True})
        second_id = wait_for_tag(second_client, tag + '_first')
        second_client.call('fiber.set_tag', {'fiberId': second_id, 'tag': tag + '_second', 'enabled': True})
        wait_for_tag(first_client, tag + '_second')
        with ObjectStore.open(str(root / 'script.vcobj'), author='script') as script:
            script.hub_sync()
            [uuid] = [fiber['uuid'] for fiber in script.objects_by_kind('fiber') if tag + '_second' in fiber['tags']]
            deadline = time.monotonic() + 60
            while True:  # the second replica releases its lease after pushing
                try:
                    script.acquire_leases([uuid])
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
                    script.hub_sync()
            try:
                first_client.call('fiber.set_tag', {'fiberId': first_id, 'tag': tag + '_blocked', 'enabled': True})
                raise AssertionError('Editing a fiber leased by another replica must fail')
            except BridgeError as error:
                assert 'leased to script' in str(error), error
        # Surfaces travel the same way: a tag set in one replica reaches the other.
        listed_surfaces(first_client, {surface})
        first_client.call('tags.set', {'segmentId': surface, 'tag': 'reviewed', 'enabled': True}, timeout=120)
        deadline = time.monotonic() + 90
        while 'reviewed' not in listed_surfaces(second_client, {surface})[surface]['tags']:
            assert time.monotonic() < deadline, 'surface tag did not arrive through the hub'
            time.sleep(.5)
        print('Hosted replicas exchanged fiber and surface edits through the hub; a held lease blocked editing',
              flush=True)
    finally:
        for client, process in ((first_client, first), (second_client, second)):
            if client:
                client.close()
            if process:
                process.terminate()


if __name__ == '__main__':
    main()
