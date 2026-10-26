"""Fibers, fiber-patch contacts and branches materialized from an object store."""
import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    branch, fiber_patch_edge, put_fiber, put_patch, write_reference_store)
from object_store_reference import spiral_zyx  # noqa: E402
from test_object_store_materialize import adapter_context, materialize  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / 'fit.vcobj'), author='test',
                            hostname='host') as store:
        yield store


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def test_identical_attachments_reproduce_the_reference(db, tmp_path):
    expected = reference.load_reference()
    uuids = write_reference_store(db, expected)
    context = adapter_context(db, tmp_path)
    names = {uuid: name for name, uuid in uuids.items()}
    reference.assert_matches_reference(reference.reference_summary(context, names))
    [link] = context.resolved_links
    assert link['edge_uuid'] == context.object_store_inputs.fiber_links[0]['edge_uuid']


def _grid(z0=4.0, size=12, step=4.0, y=100.0, x0=40.0):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([z0 + rows * step, np.full(rows.shape, y),
                     x0 + cols * step], axis=-1)


def _dense_line(n=60, y=104.0):
    """A fiber along x, one voxel apart, beside the patch plane y=100."""
    return [[30.0, y, 40.0 + i] for i in range(n)]


def test_contacts_and_branch_endpoints_survive_trimming_and_decimation(db):
    patch = put_patch(db, _grid(), (0.25, 0.25))
    other_patch = put_patch(db, _grid(y=120.0), (0.25, 0.25))
    line = _dense_line()
    # Controls span line points 5..50 with reordered, noncontiguous IDs.
    fiber = put_fiber(db, line, [5, 23, 50], [70, 11, 42])
    partner = put_fiber(db, [[30.0, 140.0, 40.0 + i] for i in range(20)],
                        [0, 19], [3, 9])
    fiber_patch_edge(db, fiber, 17, patch, (6.5, 4.25))
    fiber_patch_edge(db, fiber, 17, other_patch, (6.5, 4.25))
    fiber_patch_edge(db, fiber, 3, patch, (6.5, 1.75))
    edge = branch(db, fiber, 11, partner, 9)
    inputs = materialize(db, fiber_min_point_spacing=10.0)
    catalog = inputs.fibers[fiber]
    keys = list(catalog['points'])
    # Raw line indices survive: trimmed to the control span (widened to the
    # contact at 3), decimated to ~10 voxels, contacts and branch kept.
    assert keys[0] == 3 and keys[-1] <= 50
    assert {3, 17, 23} <= set(keys)
    assert all(b - a <= 10 for a, b in zip(keys, keys[1:]))
    assert inputs.counts['fiber spans widened to keep contacts'] == 1
    attachments = catalog['points'][17]['attachments']
    assert sorted(a['id'] for a in attachments) == sorted([patch, other_patch])
    assert all(a['ij'] == [6.5, 4.25] for a in attachments)
    [link] = inputs.fiber_links
    a_point, b_point = ((link['a_point'], link['b_point'])
                        if link['a_coll'] == catalog['id'] else
                        (link['b_point'], link['a_point']))
    # Control ID 11 is control position 1, line point 23; ID 9 is line 19.
    assert (a_point, b_point) == (23, 19)
    assert link['edge_uuid'] == edge and link['pending']


def test_missing_anchored_control_ids_fail(db):
    a = put_fiber(db, _dense_line(10), [0, 9], [1, 2])
    b = put_fiber(db, _dense_line(10, y=130.0), [0, 9], [3, 4])
    branch(db, a, 2, b, 3)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    [rel] = snapshot.relationships_of('fiber_branch')
    rel.a_anchor = {'points': [{'array': 'control_point_ids', 'index': [99]}]}
    with pytest.raises(osi.ObjectStoreInputError, match='absent from selected geometry'):
        osi.materialize_fibers(db, snapshot, {}, min_point_spacing=0.0)


def test_adjacent_branches_never_link(db):
    a = put_fiber(db, _dense_line(10), [0, 9], [1, 2])
    b = put_fiber(db, _dense_line(10, y=130.0), [0, 9], [3, 4])
    branch(db, a, 2, b, 3, adjacent=True)
    inputs = materialize(db)
    assert inputs.fiber_links == []
    assert inputs.snapshot.exclusions['adjacent fiber branch'] == 1


def test_attachments_to_several_patches_share_one_physical_node(db, tmp_path):
    a = put_patch(db, _grid(), (0.25, 0.25))
    b = put_patch(db, _grid(x0=60.0), (0.25, 0.25))
    fiber = put_fiber(db, _dense_line(40), [0, 39], [1, 2])
    fiber_patch_edge(db, fiber, 22, a, (6.5, 5.5))
    fiber_patch_edge(db, fiber, 22, b, (6.5, 1.5))
    context = adapter_context(db, tmp_path)
    [pcl] = [p for p in context.cross_patch_pcls
             if p['metadata'].get('logical_input_kind') == 'fiber']
    views = [entry for entries in pcl['points_by_patch'].values() for entry in entries]
    assert sorted(view['on_patch']['id'] for view in views) == sorted([a, b])
    assert views[0]['physical'] is views[1]['physical'] is pcl['points'][22]
    # The strip and the chain each visit the point once.
    assert len(pcl['chain'].iter_chain()) == len(pcl['points'])
    assert len(context.unattached_pcl_strips) == 1


def _seam_patch(winding, theta_begin, theta_end):
    zs = np.arange(4.0, 61.0, 4.0)
    radius = reference.DR * (winding + (theta_begin + theta_end) / (4 * np.pi))
    thetas = np.arange(theta_begin, theta_end + 1e-9, 4.0 / radius)
    return np.asarray([[spiral_zyx(t, winding, z) for t in thetas] for z in zs])


def test_fiber_across_the_seam_links_consecutive_windings(db, tmp_path):
    before = put_patch(db, _seam_patch(8, 5.6, 6.2), (0.25, 0.25), name='before')
    after = put_patch(db, _seam_patch(9, 0.1, 0.7), (0.25, 0.25), name='after')
    # One sheet, between the patches' radii, crossing theta = 0: winding 8.5
    # before the seam continues as winding 9.5 after it. It dips onto each
    # patch once.
    samples = [(t, 8.5) for t in np.arange(5.75, 6.2832, 0.05)] + \
              [(t, 9.5) for t in np.arange(0.02, 0.55, 0.05)]
    dips = {4: 8, len(samples) - 3: 9}
    line = [spiral_zyx(t, dips.get(i, w), 30.0) for i, (t, w) in enumerate(samples)]
    fiber = put_fiber(db, line, [0, len(line) - 1], [1, 2])
    for index, patch, winding, theta0 in (
            (4, before, 8, 5.6), (len(samples) - 3, after, 9, 0.1)):
        theta = samples[index][0]
        radius = reference.DR * (winding + (theta0 + theta0 + 0.6) / (4 * np.pi))
        fiber_patch_edge(db, fiber, index, patch,
                         (6.5, (theta - theta0) * radius / 4.0))
    context = adapter_context(db, tmp_path)
    losses = reference.loss_summary(context)
    # Without the seam correction the pair would be a winding (12 voxels) off.
    assert max(losses['rel_winding']) < 0.5
    assert len(context.cross_patch_pcls) == 1


@pytest.mark.parametrize('manual,certainty,expected', [
    (True, 0.1, 'V'), (False, 0.9, 'V'), (False, 0.1, 'H'),
])
def test_hv_metadata_uses_pinned_typed_fields(db, tmp_path, manual, certainty, expected):
    import json
    from spiral_helpers import classify_fiber_hv

    source_hv = {'automatic_certainty': certainty, 'automatic_score': 0.7,
                 'manual_tag': 'H', 'automatic_tag': 'H'}
    uid = put_fiber(db, _dense_line(10), [0, 9],
                    meta={'source': {'hv_classification': source_hv}},
                    fiber={'family': 'V', 'manual_family': manual})
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    revision = snapshot.objects[uid].object_revision

    # Change the moving head after capture; materialization must use the pin.
    record = db.object(uid)
    record['fiber']['family'] = 'H'
    db.upsert_object(record)
    catalog, _, _ = osi.materialize_fibers(db, snapshot, {}, min_point_spacing=0.0)
    hv = catalog[uid]['metadata']['hv_classification']
    assert hv['automatic_certainty'] == certainty
    assert hv['automatic_score'] == 0.7
    assert hv['manual_tag'] == ('V' if manual else '')
    assert hv['automatic_tag'] == ('H' if manual else 'V')
    assert classify_fiber_hv(hv, _dense_line(10), min_z_fraction=0.8,
                             min_auto_certainty=0.5) == expected

    from pcl_views import fiber_strip
    strip = fiber_strip(0, catalog[uid], set(), z_begin=0, z_end=100,
                        z_margin=0, min_point_spacing=0,
                        vertical_min_z_fraction=0.8,
                        vertical_min_auto_certainty=0.5,
                        vertical_radial_offset=3.0)
    assert strip['hv_tag'] == expected
    np.testing.assert_array_equal(
        strip['radial_offsets'], np.full(10, 3.0 if expected == 'V' else 0.0))

    output = tmp_path / 'fiber.json'
    db.export_geometry(uid, output, object_revision_uuid=revision)
    assert json.loads(output.read_text())['hv_classification'] == hv
    fresh, _, _ = osi.materialize_fibers(
        db, osi.read_snapshot(db, osi.InputRequest()), {}, min_point_spacing=0.0)
    assert fresh[uid]['metadata']['hv_classification'] != hv
