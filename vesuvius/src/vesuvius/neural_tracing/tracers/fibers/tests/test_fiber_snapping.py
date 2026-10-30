"""Presence snapping of fiber labels (data/fiber_snapping.py, scripts/snap_fibers.py), of fiber stores and VC3D JSONs."""
import json

import numpy as np
import pytest
import zarr
from vc_reader.reader import open_zarr

from vesuvius.neural_tracing.tracers.fibers.data.fiber_snapping import SnapConfig, resample, snap_polylines


def tube_volume(path, centre, shape=(48, 48, 96), sigma=.8):
    """uint8 presence: a Gaussian tube (sigma voxels) around ``centre`` (xyz samples), written as a local Zarr v2
    array at ``path`` and opened with vc_reader."""
    from scipy.spatial import cKDTree
    z, y, x = np.indices(shape)
    distance = cKDTree(resample(centre, .1)).query(np.c_[x.ravel(), y.ravel(), z.ravel()])[0].reshape(shape)
    array = zarr.create_array(str(path), shape=shape, chunks=(16, 16, 32), dtype='u1', fill_value=0, zarr_format=2,
                              compressors=None)
    array[:] = (255*np.exp(-.5*(distance/sigma)**2)).astype(np.uint8)
    return open_zarr(path)


def wavy(offset=(0., 0., 0.)):
    x = np.arange(4., 92., 1.)
    return np.c_[x, 24+3*np.sin(x/9.), 24+2*np.cos(x/13.)]+np.asarray(offset)


def test_snapping_moves_an_offset_label_onto_the_presence_centre(tmp_path):
    centre = wavy()
    presence = tube_volume(tmp_path/'presence', centre)
    label = wavy((0., 1.4, -.9))  # off the fiber by ~1.7 voxels
    snapped = snap_polylines([label], presence, SnapConfig(tile=32, threads=2), device='cpu')[0]
    from scipy.spatial import cKDTree
    error = cKDTree(resample(centre, .1)).query(snapped[8:-8])[0]
    before = cKDTree(resample(centre, .1)).query(label[8:-8])[0]
    assert before.mean() > 1.5 and error.mean() < .25 and error.max() < .6
    # A label already on the fiber stays there.
    kept = snap_polylines([centre], presence, SnapConfig(tile=32, threads=2), device='cpu')[0]
    assert cKDTree(resample(centre, .1)).query(kept[8:-8])[0].max() < .25


A = '00000000-0000-4000-8000-00000000000a'
B = '00000000-0000-4000-8000-00000000000b'


def test_store_snapping_keeps_uuids_names_families_and_meta_and_reads_back(tmp_path):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import FiberStore, read_polylines, write_fibers
    from vesuvius.neural_tracing.tracers.fibers.data import fiber_snapping as snap_fibers
    scale = 8.
    centre = wavy()
    write_fibers(tmp_path/'in.vcobj', [(dict(id=B, name='b', family='V', meta={'origin': 'test'}), centre*scale),
                                       (dict(id=A, name='a', family='H', meta={'origin': 'test'}),
                                        wavy((0., 1.4, -.9))*scale)], volume='1')
    presence = tube_volume(tmp_path/'presence', centre)
    result = snap_fibers.snap_store(tmp_path/'in.vcobj', tmp_path/'out.vcobj', presence, scale, 0.,
                                    SnapConfig(tile=32, threads=2), 'cpu', 1., 0, dict(method='test'))
    assert result['fibers'] == 2
    rows = read_polylines(tmp_path/'out.vcobj')
    assert [(r['id'], r['name'], r['family']) for r, _ in rows] == [(A, 'a', 'H'), (B, 'b', 'V')]
    assert rows[0][0]['meta']['origin'] == 'test'
    assert rows[0][0]['meta']['snapping'] == dict(method='test', snapped_from=str((tmp_path/'in.vcobj').resolve()))
    fibers = FiberStore(tmp_path/'out.vcobj', '1', scale, split='all')
    assert len(fibers) == 2 and fibers.catalog[0]['id'] == A
    from scipy.spatial import cKDTree
    assert cKDTree(resample(centre, .1)).query(fibers[0].points[10:-10])[0].mean() < .3
    with pytest.raises(FileExistsError):
        snap_fibers.snap_store(tmp_path/'in.vcobj', tmp_path/'out.vcobj', presence, scale, 0., SnapConfig(tile=32),
                               'cpu', 1., 0, {})


def test_json_snapping_rewrites_the_line_and_keeps_controls_on_it(tmp_path):
    from vesuvius.neural_tracing.tracers.fibers.data import fiber_snapping as snap_fibers
    scale = 8.
    centre = wavy()
    line = wavy((0., 1.4, -.9))*scale
    document = dict(type='vc3d_fiber', version=3, line_points=line.tolist(),
                    control_points=[dict(position=line[i].tolist()) for i in (0, 40, len(line)-1)])
    (tmp_path/'in').mkdir()
    (tmp_path/'in'/'f.json').write_text(json.dumps(document))
    presence = tube_volume(tmp_path/'presence', centre)
    snap_fibers.snap_json([tmp_path/'in'/'f.json'], tmp_path/'out', presence, scale, 0., SnapConfig(tile=32, threads=2),
                          'cpu', 1., dict(method='test'))
    out = json.loads((tmp_path/'out'/'f.json').read_text())
    new = np.asarray(out['line_points'])
    assert all(np.min(np.linalg.norm(new-np.asarray(c['position']), axis=1)) < 1e-9 for c in out['control_points'])
    from scipy.spatial import cKDTree
    assert cKDTree(resample(centre, .1)).query(new[10:-10]/scale)[0].mean() < .3
    assert json.loads((tmp_path/'out'/'snapping.json').read_text())['files'] == ['f.json']
