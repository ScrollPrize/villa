"""Runtime patches and point constraints materialized from an object store."""
import math

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    frame, pcl_patch_edge, put_patch, put_pcl, write_reference_store)


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / 'fit.vcobj'), author='test',
                            hostname='host') as store:
        yield store


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def materialize(db, fiber_min_point_spacing=0.0, **request):
    snapshot = osi.read_snapshot(db, osi.InputRequest(**request))
    config = reference.fit_config()
    return osi.materialize(db, snapshot, z_begin=config['z_begin'],
                           z_end=config['z_end'],
                           erode_cells_default=config['patch_erode_patches'],
                           fiber_min_point_spacing=fiber_min_point_spacing)


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


def _without_fibers(constraints):
    """The patch and regular-PCL constraints of a summary."""
    def regular(entries):
        return [entry for entry in entries if entry[0] != 'fibers']
    return {
        'patches': constraints['patches'],
        'cross_patch': [pcl for pcl in constraints['cross_patch']
                        if pcl['group'] != 'fibers'],
        'strips': [strip for strip in constraints['strips']
                   if strip['group'] != 'fibers'],
        'cross_patch_strata': regular(constraints['cross_patch_strata']),
    }


def test_identical_attachments_reproduce_the_reference_patches_and_pcls(db, tmp_path):
    expected = reference.load_reference()
    uuids = write_reference_store(db, expected, fibers=False)
    context = adapter_context(db, tmp_path)
    names = {uuid: name for name, uuid in uuids.items()}
    summary = reference.constraint_summary(context, names)
    assert _without_fibers(summary) == _without_fibers(expected['constraints'])
    point = context.cross_patch_pcls[0]['points'][2]
    assert point['on_patch']['edge_uuid'] in {
        rel.uuid for rel in context.object_store_inputs.snapshot.relationships}


def _grid(z0=4.0, size=8, step=4.0):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([z0 + rows * step, np.full(rows.shape, 100.0),
                     40.0 + cols * step], axis=-1)


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


def test_patches_outside_the_z_window_are_dropped(db):
    put_patch(db, _grid(z0=200.0))
    inputs = materialize(db)
    assert inputs.patches == {}
    assert inputs.counts['patches outside z ROI'] == 1


def _point(point_id, wind=None):
    return [10.0 + point_id, 100.0, 40.0 + 4 * point_id], wind


def test_points_keep_stable_ids_and_aligned_annotations(db):
    patch = put_patch(db, _grid())
    # Stored out of order, with noncontiguous IDs.
    pcl = put_pcl(db, {9: _point(9, 2.0), 3: _point(3, 1.0), 5: _point(5, 1.5)})
    pcl_patch_edge(db, pcl, 5, patch, (1.5, 2.25))
    inputs = materialize(db)
    [collection] = inputs.point_collections.values()
    assert list(collection['points']) == [3, 5, 9]
    assert [p['winding_annotation'] for p in collection['points'].values()] == [1.0, 1.5, 2.0]
    assert collection['points'][5]['on_patch']['ij'] == [1.5, 2.25]
    assert collection['points'][5]['on_patch']['id'] == patch
    np.testing.assert_allclose(collection['points'][9]['zyx'], _point(9)[0])
    assert collection['sampling_group'] == osi.RELATIVE_PCL_GROUP
    assert inputs.counts['pcl attachments'] == 1


def test_absolute_role_comes_from_metadata_not_names(db):
    patch = put_patch(db, _grid())
    absolute = put_pcl(db, {0: _point(0, 3.0)}, absolute=True, name='same_windings')
    relative = put_pcl(db, {0: _point(0, 3.0)}, name='abs_winding')
    unannotated = put_pcl(db, {0: _point(0)})
    for uid in (absolute, relative, unannotated):
        pcl_patch_edge(db, uid, 0, patch, (1.0, 1.0))
    inputs = materialize(db)
    by_uuid = {pcl['metadata']['object_uuid']: pcl
               for pcl in inputs.point_collections.values()}
    assert by_uuid[absolute]['metadata']['winding_is_absolute']
    assert by_uuid[absolute]['sampling_group'] == osi.ABSOLUTE_PCL_GROUP
    assert not by_uuid[relative]['metadata']['winding_is_absolute']
    assert by_uuid[unannotated]['sampling_group'] == osi.SAME_WINDING_PCL_GROUP


def test_malformed_roles_fail(db):
    put_pcl(db, {0: _point(0, 3.0)}, source={'winding_is_absolute': 'yes'})
    with pytest.raises(osi.ObjectStoreInputError, match='non-boolean'):
        materialize(db)


def test_detected_role_must_match_the_metadata(db):
    patch = put_patch(db, _grid())
    pcl = put_pcl(db, {0: _point(0, 3.0)}, absolute=True)
    edge = pcl_patch_edge(db, pcl, 0, patch, (1.0, 1.0))
    record = db.edge(edge)
    record['meta']['point_collection_patch']['winding_is_absolute'] = False
    with db.write() as tx:
        tx.update_edge(record)
    with pytest.raises(osi.ObjectStoreInputError, match='detected as relative'):
        materialize(db)


def test_absolute_points_need_explicit_positive_windings(db, tmp_path):
    from pcl_views import regular_pcl_views
    patch = put_patch(db, _grid())
    pcl = put_pcl(db, {0: _point(0), 1: _point(1, 4.0)}, absolute=True)
    pcl_patch_edge(db, pcl, 0, patch, (1.0, 1.0))
    inputs = materialize(db)
    # A missing annotation is never treated as winding zero.
    assert math.isnan(inputs.point_collections[0]['points'][0]['winding_annotation'])
    with pytest.raises(ValueError, match='without a winding annotation'):
        regular_pcl_views(inputs.point_collections, inputs.patches, z_begin=0,
                          z_end=64, z_margin=0, min_point_spacing=0)


def test_unmaterializable_attachments_are_reported(db):
    patch = put_patch(db, _grid())
    pcl = put_pcl(db, {0: _point(0, 1.0), 1: _point(1, 1.0)})
    pcl_patch_edge(db, pcl, 0, patch, (1.0, 1.0))
    pcl_patch_edge(db, pcl, 0, put_patch(db, _grid(z0=8.0)), (1.0, 1.0))
    with pytest.raises(osi.ObjectStoreInputError) as error:
        materialize(db)
    assert 'attaches to patches' in str(error.value)


def test_attachments_to_unfitted_patches_are_counted(db):
    pcl = put_pcl(db, {0: _point(0, 1.0)})
    pcl_patch_edge(db, pcl, 0, put_patch(db, _grid(z0=200.0)), (1.0, 1.0))
    inputs = materialize(db)
    assert 'on_patch' not in inputs.point_collections[0]['points'][0]
    assert inputs.counts['pcl attachments to unfitted patches'] == 1


def test_report_lists_materialized_constraints(db, capsys):
    patch = put_patch(db, _grid())
    pcl = put_pcl(db, {0: _point(0, 1.0), 1: _point(1, 2.0)})
    pcl_patch_edge(db, pcl, 0, patch, (1.0, 1.0))
    assert osi.main([db.path, '--z-begin', '0', '--z-end', '64']) == 0
    report = capsys.readouterr().out
    assert 'constraints: ' in report and 'pcl attachments 1' in report
