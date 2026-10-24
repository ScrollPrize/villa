#!/usr/bin/env python3
"""Exercise VC3D attachment, transfer and immediate edits on a disposable database.

Requires the vc_object_store Python extension in PYTHONPATH. The input database
must contain at least one fiber; neither it nor the user's settings are modified.
Without --database, create a current-format fixture and check lazy span details.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from bridge_client import BridgeClient, BridgeError
from vc3d_process import VC3DProcess
from vc_object_store import ObjectStore


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
        config = root / 'config'
        config.mkdir()
        (config / 'VC3D.ini').write_text(
            '[viewer]\nremote_cache_dir=' + str(root / 'remote-cache') +
            '\n[project]\nshow_open_data_catalog_on_startup=false\n')
        tag = 'bridge_smoke_' + str(os.getpid())

        def launch(index, store=database):
            process = VC3DProcess(args.vc3d,
                ['--agent-bridge', '--agent-bridge-name', f'vc3d-store-{os.getpid()}-{index}',
                 '--object-store', str(store)],
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
        hosted(root, database, launch, tag)


def hosted(root, database, launch, tag):
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
        print('Hosted replicas exchanged edits through the hub; a held lease blocked editing', flush=True)
    finally:
        for client, process in ((first_client, first), (second_client, second)):
            if client:
                client.close()
            if process:
                process.terminate()


if __name__ == '__main__':
    main()
