"""Runtime patches materialized from an object store."""
import numpy as np
import pytest

pytest.importorskip('vc_object_store')

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    frame, put_patch, write_reference_store)


pytestmark = pytest.mark.usefixtures('cpu')


def materialize(db, **request):
    snapshot = osi.read_snapshot(db, osi.InputRequest(**request))
    config = reference.fit_config()
    return osi.materialize(db, snapshot, z_begin=config['z_begin'],
                           z_end=config['z_end'],
                           z_margin=config['patch_loss_z_margin'],
                           erode_cells_default=config['patch_erode_patches'])


def adapter_context(db, tmp_path):
    """A FitContext loaded from the store instead of files."""
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)
    context = FitContext(
        FitConfig(reference.fit_config()), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), object_store_inputs=materialize(db))
    context.load_host_inputs()
    return context


def test_identical_patches_reproduce_the_reference(db, tmp_path):
    uuids = write_reference_store(db)
    context = adapter_context(db, tmp_path)
    names = {uuid: name for name, uuid in uuids.items()}
    summary = reference.constraint_summary(context, names)
    assert summary == reference.load_reference()['constraints']


def _grid(z0=4.0, size=8, step=4.0):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([z0 + rows * step, np.full(rows.shape, 100.0),
                     40.0 + cols * step], axis=-1)


@pytest.mark.parametrize('mask', [False, True])
def test_raw_surface_validity_uses_shared_reader(db, mask):
    from types import SimpleNamespace
    points = np.full((2, 3, 3), 10., dtype='<f4')
    points[0, 0] = [-1, -1, -1]
    points[0, 1] = [10, -1, 10]
    points[0, 2] = [10, 10, np.nan]
    points[1, 0] = [10, 10, 0]
    arrays = [dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())]
    if mask:
        valid = np.ones((2, 3), dtype='u1')
        valid[1, 1] = 0
        arrays.append(dict(name='valid', dtype='|u1', shape=[2, 3], data=valid.tobytes()))
    uid = db.upsert_object(dict(kind='surface', meta={'coordinates': frame(2)}))
    revision = db.put_geometry(uid, dict(kind='surface', attributes=dict(scale=[1, 1]), arrays=arrays))
    selected = SimpleNamespace(uuid=uid, geometry_revision=revision,
                               frame=osi.ObjectFrame('vol', 2, .5))
    actual, invalid, scale = osi._surface_zyxs(db, selected, 'surface')
    with db.geometry_snapshot() as snapshot:
        _, expected_valid = snapshot.surface(uid, revision).read_arrays()
    np.testing.assert_array_equal(invalid, ~expected_valid)
    np.testing.assert_array_equal(actual[invalid], -1)
    np.testing.assert_allclose(actual[~invalid], 5)
    assert scale == [2, 2]


@pytest.mark.parametrize('scale', [0.5, 1, 4])
def test_patch_frames_invalid_vertices_and_erosion(db, scale):
    import torch
    from spiral_helpers import prepare_patch_for_fit
    from tifxyz import Patch
    grid = _grid()
    grid[3, 3] = -1
    a = put_patch(db, grid, (0.25, 0.25), coordinates=frame(scale), name='a')
    b = put_patch(db, _grid(), (0.25, 0.25), coordinates=frame(1), name='b',
                  meta={'source': {'spiral_patch_erode_cells': 0}})
    inputs = materialize(db)
    patch = inputs.patches[a]
    # The file loader's erosion of the same grid, with the hole kept at -1.
    expected = Patch(torch.from_numpy(grid.astype(np.float32)),
                     torch.tensor([0.25, 0.25]), None, None)
    assert prepare_patch_for_fit(expected, 0, 64, 1) is None
    valid = patch.valid_vertex_mask.numpy()
    np.testing.assert_array_equal(valid, expected.valid_vertex_mask.numpy())
    assert not valid[3, 3] and (patch.zyxs.numpy()[3, 3] == -1).all()
    np.testing.assert_allclose(patch.zyxs.numpy(), expected.zyxs.numpy(), atol=1e-4)
    np.testing.assert_allclose(patch.scale.numpy(), [0.25, 0.25])
    # Its own erosion request keeps the other patch whole.
    assert inputs.patches[b].valid_vertex_mask.all()


def test_surface_validity_channel_marks_holes(db):
    # Ingested mask.tif: a masked vertex keeps its coordinates but is a hole.
    valid = np.ones(_grid().shape[:2], dtype=bool)
    valid[2, 5] = False
    uid = put_patch(db, _grid(), (0.25, 0.25), valid=valid,
                    meta={'source': {'spiral_patch_erode_cells': 0}})
    patch = materialize(db).patches[uid]
    np.testing.assert_array_equal(patch.valid_vertex_mask.numpy(), valid)
    assert (patch.zyxs.numpy()[2, 5] == -1).all()


def test_patches_outside_the_z_window_are_dropped(db):
    put_patch(db, _grid(z0=200.0))
    inputs = materialize(db)
    assert inputs.patches == {}
    assert inputs.counts['patches outside z ROI'] == 1


def test_report_materializes_the_inputs(db, capsys):
    put_patch(db, _grid())
    assert osi.main([db.path, '--z-begin', '0', '--z-end', '64']) == 0
    assert 'patches 1' in capsys.readouterr().out
