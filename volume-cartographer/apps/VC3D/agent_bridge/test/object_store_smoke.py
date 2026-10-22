#!/usr/bin/env python3
"""Exercise real VC3D attachment and immediate edits on a disposable database copy.

Requires the vc_object_store Python extension in PYTHONPATH. The input database
must contain at least one fiber; neither it nor the user's settings are modified.
"""
import argparse
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
    parser.add_argument('--database', required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='vc3d-store-bridge-') as temporary:
        root = Path(temporary)
        database = root / 'smoke.vcobj'
        with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(database) as target:
                source.backup(target)
        with ObjectStore.open(str(database), read_only=True) as store:
            count = len(store.objects_by_kind('fiber'))
        assert count > 0, 'The smoke fixture must contain fibers'
        config = root / 'config'
        config.mkdir()
        (config / 'VC3D.ini').write_text(
            '[viewer]\nremote_cache_dir=' + str(root / 'remote-cache') +
            '\n[project]\nshow_open_data_catalog_on_startup=false\n')
        tag = 'bridge_smoke_' + str(os.getpid())

        def launch(index):
            process = VC3DProcess(args.vc3d,
                ['--agent-bridge', '--agent-bridge-name', f'vc3d-store-{os.getpid()}-{index}',
                 '--object-store', str(database)],
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


if __name__ == '__main__':
    main()
