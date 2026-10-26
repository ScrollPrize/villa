"""Surfaces read from and created in a VC object store (vesuvius.tifxyz.store) and the displacement-copy store
mode built on them."""
import numpy as np
import pytest
from vc_object_store import ObjectStore

from vesuvius.neural_tracing.inference import infer_rowcol_triplet_wraps as copy
from vesuvius.tifxyz.store import create_surface, read_surface


def _store_with_surface(path, *, scale_factor=1, valid=None):
    rows, cols = np.mgrid[:6, :8]
    points = np.stack([cols * 2.0 + 10, rows * 2.0 + 20, np.full(rows.shape, 30.0)], axis=-1).astype('<f4')
    points[0, 0] = -1
    arrays = [dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())]
    if valid is not None:
        arrays.append(dict(name='valid', dtype='|u1', shape=list(valid.shape), data=valid.astype('u1').tobytes()))
    with ObjectStore.create(str(path), author='test') as db:
        uid = db.upsert_object(dict(kind='surface', name='source',
                                    meta=dict(coordinates=dict(volume_uuid='vol', scale_factor=scale_factor))))
        db.put_geometry(uid, dict(kind='surface', attributes=dict(scale=[0.5, 0.5]), arrays=arrays))
        revision = db.current_geometry(uid)['uuid']
    return uid, revision, points


def test_read_surface_applies_validity_and_volume_scale(tmp_path):
    valid = np.full((6, 8), 255, 'u1')
    valid[2, 3] = 0
    uid, _, points = _store_with_surface(tmp_path / 'objects.vcobj', scale_factor=2, valid=valid)
    surface = read_surface(tmp_path / 'objects.vcobj', uid, volume='vol')
    assert surface.uuid == uid
    assert surface.get_scale_tuple() == (1.0, 1.0)  # stored scale 0.5 per stored unit = 1 per voxel
    assert not surface._mask[0, 0] and not surface._mask[2, 3] and surface._mask[1, 1]
    assert surface._x[2, 3] == -1 and surface._x[0, 0] == -1
    assert surface._x[1, 1] == pytest.approx(points[1, 1, 0] / 2)
    with pytest.raises(ValueError, match='does not map'):
        read_surface(tmp_path / 'objects.vcobj', uid, volume='other')


def test_read_surface_uses_the_store_missing_vertex_rule(tmp_path):
    """Any -1 or non-finite coordinate is a missing vertex; z = 0 is a slice, not a missing vertex."""
    rows, cols = np.mgrid[:2, :3]
    points = np.stack([cols + 10.0, rows + 20.0, np.full(rows.shape, 30.0)], axis=-1).astype('<f4')
    points[0, 1] = (11, -1, 30)
    points[0, 2] = (12, 20, np.nan)
    points[1, 0] = (10, 21, 0)
    with ObjectStore.create(str(tmp_path / 'objects.vcobj'), author='test') as db:
        uid = db.upsert_object(dict(kind='surface', meta=dict(coordinates=dict(volume_uuid='vol', scale_factor=1))))
        db.put_geometry(uid, dict(kind='surface', attributes=dict(scale=[1, 1]), arrays=[
            dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())]))
    surface = read_surface(tmp_path / 'objects.vcobj', uid, volume='vol')
    assert surface._mask.tolist() == [[True, False, False], [True, True, True]]


def test_create_surface_records_binding_provenance_and_missing_vertices(tmp_path):
    source, _, _ = _store_with_surface(tmp_path / 'objects.vcobj')
    zyxs = np.full((3, 4, 3), 5.0, np.float32)
    zyxs[1, 1] = -1
    zyxs[2, 2, 0] = np.nan
    uid = create_surface(tmp_path / 'objects.vcobj', zyxs, name='copy', scale=(0.25, 0.25), volume='vol',
                         tool='test-tool', parameters={'k': 1}, sources=[source], meta={'area_vx2': 4})
    with ObjectStore.open(str(tmp_path / 'objects.vcobj'), read_only=True) as db:
        record = db.object(uid)
        assert record['name'] == 'copy'
        assert record['meta']['coordinates'] == {'volume_uuid': 'vol', 'scale_factor': 1}
        assert record['meta']['provenance'] == {'tool': 'test-tool', 'parameters': {'k': 1}, 'sources': [source]}
        assert record['meta']['area_vx2'] == 4
    surface = read_surface(tmp_path / 'objects.vcobj', uid, volume='vol')
    assert surface.get_scale_tuple() == (0.25, 0.25)
    assert not surface._mask[1, 1] and not surface._mask[2, 2] and surface._mask[0, 0]
    assert surface._z[0, 0] == 5.0  # zyx in, xyz stored


def test_copy_store_mode_reads_the_pinned_revision_and_creates_sourced_outputs(tmp_path):
    store = tmp_path / 'objects.vcobj'
    source, revision, _ = _store_with_surface(store)
    with ObjectStore.open(str(store), author='test') as db:  # a later revision must not be read
        edited = np.zeros((6, 8, 3), '<f4')
        db.put_geometry(source, dict(kind='surface', attributes=dict(scale=[0.5, 0.5]),
                                     arrays=[dict(name='points', dtype='<f4', shape=[6, 8, 3], data=edited.tobytes())]),
                        expected_revision=revision)
    surfaces = copy._StoredSurfaces(str(store), 'vol', source, revision)
    _, grid, valid = surfaces.load(source, 1)
    assert grid[1, 1, 2] == pytest.approx(12.0)  # zyx of the pinned revision (x = 1*2 + 10)
    assert not valid[0, 0]
    assert surfaces.name(source) == 'source'
    created = surfaces.save(grid, valid, 'source_front', step_size=2, voxel_size_um=8.0, source='ckpt',
                            metadata={'surface_role': 'front'}, apply_mesh_cleanup=False)
    with ObjectStore.open(str(store), read_only=True) as db:
        meta = db.object(created)['meta']
        assert meta['provenance']['sources'] == [source]
        assert meta['surface_role'] == 'front' and meta['area_vx2'] > 0


def test_copy_cli_takes_exactly_one_input_mode():
    base = ['--volume-path', 'v.zarr', '--checkpoint-path', 'c.pt']
    args = copy.parse_args(base + ['--store', 's.vcobj', '--surface', 'u', '--volume-uuid', 'vol'])
    assert (args.store, args.surface_uuid, args.volume_uuid) == ('s.vcobj', 'u', 'vol')
    for bad in (base, base + ['--tifxyz-path', 't', '--store', 's'], base + ['--store', 's'],
                base + ['--tifxyz-path', 't', '--surface', 'u']):
        with pytest.raises(SystemExit):
            copy.parse_args(bad)
    argv = copy._copy_args_to_argv({'store': 's.vcobj', 'surface_uuid': 'u', 'volume_uuid': 'vol',
                                    'geometry_revision': 'r'})
    assert argv == ['--store', 's.vcobj', '--surface', 'u', '--geometry-revision', 'r', '--volume-uuid', 'vol']
