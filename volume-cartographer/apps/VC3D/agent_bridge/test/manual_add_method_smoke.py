#!/usr/bin/env python3
"""Check that finishing Manual Add preserves the growth method on a real surface.

Uses a disposable project/store and a 4x4 crop of --surface (a real tifxyz mesh).
Requires numpy, tifffile and vc_object_store, as well as the built VC3D binary.
Neither input data nor the user's settings are modified.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import tifffile
from vc_object_store import ObjectStore

from bridge_client import BridgeClient
from smoke_offscreen import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vc3d', required=True)
    parser.add_argument('--platform', default='offscreen', help='Qt platform (xcb for a running X display)')
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--volume', required=True, help='Matching CT zarr path or URL')
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--follow-sheet', action='store_true', help='Also check editing locks during a real RK4 walk')
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    points = np.stack([tifffile.imread(args.surface / f'{axis}.tif') for axis in 'xyz'], axis=-1)
    valid = np.isfinite(points).all(-1) & (points >= 0).all(-1)
    candidates = np.argwhere(valid)
    for row, col in candidates[len(candidates) // 2:]:
        if valid[row:row + 4, col:col + 4].shape == (4, 4) and valid[row:row + 4, col:col + 4].all():
            points = np.ascontiguousarray(points[row:row + 4, col:col + 4], dtype='<f4')
            break
    else:
        raise RuntimeError('No valid 4x4 crop found in the input surface')
    with tempfile.TemporaryDirectory(prefix='vc3d-manual-method-') as temporary:
        root = Path(temporary)
        database = root / 'test.vcobj'
        with ObjectStore.create(str(database), author='manual-add-test'):
            pass
        project = root / 'test.vcproj'
        project.write_text(json.dumps(dict(
            format='vcproj', version=1, name='Manual Add method regression',
            stores=[dict(id='main', location=str(database), read_only=False)],
            selections={kind: 'main' for kind in ('surface', 'fiber', 'point_collection', 'umbilicus', 'volume_transform')},
            volumes=[dict(location=args.volume, tags=[])], normal_grids=[], lasagna_datasets=[], view={})))
        config = root / 'config'
        config.mkdir()
        settings = config / 'VC3D.ini'
        settings.write_text('[viewer]\nusername=manual-add-test\nremote_cache_dir=' + str(root / 'cache') +
                            '\n[project]\nshow_open_data_catalog_on_startup=false\n' +
                            '[segmentation_edit]\ngrowth_method=0\nmanual_add_follow_sheet=' + str(args.follow_sheet).lower() +
                            '\nmanual_add_normal_step=1\nmanual_add_walk_mode=1\nmanual_add_integrator=2\n')
        process = launch(args.vc3d, f'manual-method-{os.getpid()}-{time.time_ns()}', str(project), str(config), args.platform)
        client = None
        try:
            client = BridgeClient(process.wait_for_handshake(120))
            print('Bridge connected; waiting for CT volume', flush=True)
            volumes, _ = client.call('volume.list', timeout=15)
            assert volumes['volumeIds'], volumes
            volume_id = volumes['volumeIds'][0]
            client.call('volume.select', {'volumeId': volume_id}, timeout=120)
            state, _ = client.call('state.get', timeout=120)
            assert state.get('volume'), state
            print('Volume loaded; attaching real surface crop', flush=True)
            with ObjectStore.open(str(database), author='manual-add-test') as store:
                surface = store.upsert_object(dict(kind='surface', name='Real surface crop',
                    meta=dict(coordinates=dict(volume_uuid=state['volume']['id'], scale_factor=1))))
                store.put_geometry(surface, dict(kind='surface', attributes=dict(scale=[1, 1]), arrays=[
                    dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())]))
            # Reopen to load the newly inserted real surface into the listing.
            client.call('volume.open', {'path': str(project), 'volumeId': volume_id}, timeout=120)
            client.call('segments.activate', {'segmentId': surface}, timeout=120)
            client.call('viewer.center_on_point', {'point': dict(zip('xyz', points.mean(axis=(0,1)).tolist()))})
            client.call('segmentation.enable_editing', {'enabled': True}, timeout=120)
            client.call('segmentation.manual_add.set_interpolation', {'mode': 'thin_plate_spline'})

            def assert_method(label):
                state, _ = client.call('state.get', timeout=120)
                assert not state['manualAddMode'], state
                method = state['segmentationGrowthMethod']
                client.call('screenshot.capture', {'filePath': str((args.artifacts / f'{label}.png').resolve())}, timeout=120)
                assert method == 'Manual Add', f'{label}: expected Manual Add, got {method}'
                print(f'PASS {label}: session ended; Manual Add remains selected', flush=True)

            for apply in (True, False):
                print('Testing ' + ('apply' if apply else 'cancel'), flush=True)
                extension, _ = client.call('segmentation.manual_add.extend',
                    {'side': 'right', 'cells': 8 if args.follow_sheet else 1}, timeout=120)
                if args.follow_sheet:
                    assert extension['followingSheet'], extension
                    result, _ = client.call('segmentation.manual_add.finish', {'apply': True})
                    assert not result['applied'], result
                    result, _ = client.call('segmentation.manual_add.finish', {'apply': False})
                    assert not result['applied'], result
                    result, _ = client.call('segmentation.enable_editing', {'enabled': False})
                    assert result['enabled'], result
                    state, _ = client.call('state.get')
                    assert state['manualAddFollowingSheet'] and state['manualAddMode'], state
                    client.call('screenshot.capture', {'filePath': str((args.artifacts /
                        ('busy-apply.png' if apply else 'busy-cancel.png')).resolve())}, timeout=120)
                    deadline = time.monotonic() + 180
                    while state['manualAddFollowingSheet']:
                        assert time.monotonic() < deadline, 'Sheet walk did not finish'
                        time.sleep(.1)
                        state, _ = client.call('state.get')
                    assert state['manualAddMode'], state
                    print('PASS busy: apply/cancel/editing changes blocked until completion', flush=True)
                client.call('segmentation.manual_add.finish', {'apply': apply}, timeout=120)
                client.call('segmentation.save', timeout=120)
                deadline = time.monotonic() + 120
                while True:
                    state, _ = client.call('state.get')
                    save = state['autosave']
                    if not (save['pending'] or save['saveInProgress'] or save['dirtyAfterSave']):
                        break
                    assert time.monotonic() < deadline, f'Autosave did not finish: {save}'
                    time.sleep(.1)
                assert_method('apply' if apply else 'cancel')
            client.call('segmentation.manual_add.begin', timeout=120)
            client.call('segmentation.enable_editing', {'enabled': False}, timeout=120)
            assert_method('disable_editing')
        finally:
            if client:
                client.close()
            process.terminate()
            (args.artifacts / 'settings.ini').write_text(settings.read_text())
            (args.artifacts / 'vc3d.log').write_text('\n'.join(process.tail_log(8000)))


if __name__ == '__main__':
    main()
