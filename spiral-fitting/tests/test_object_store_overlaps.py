"""Patch-overlap constraints materialized from an object store."""
import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import VOLUME, frame, put_patch  # noqa: E402
from object_store_reference import spiral_zyx  # noqa: E402
from test_object_store_materialize import adapter_context, materialize  # noqa: E402

LOW = '00000000-0000-4000-8000-000000000001'
HIGH = 'ffffffff-0000-4000-8000-000000000001'


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / 'fit.vcobj'), author='test',
                            hostname='host') as store:
        yield store


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def _plane(dx=0.0, size=24, step=2.0):
    """A sheet in the plane y=100: row r at z 4+2r, column c at x 40+2c+dx."""
    rows, cols = np.mgrid[:size, :size]
    return np.stack([4.0 + rows * step, np.full(rows.shape, 100.0),
                     40.0 + cols * step + dx], axis=-1)


def _put(db, grid, scale=1, uuid=None, **record):
    if uuid is not None:
        record['uuid'] = uuid
    return put_patch(db, grid, (0.5, 0.5), coordinates=frame(scale), **record)


def _detect(db, expected=1):
    report = db.link_patch_overlaps(volume_uuid=VOLUME, min_inliers=16)
    assert report['edges_created'] == expected and not report['failed'], report
    return db.edges_by_kind('same-surface')


def _at(patch, ij):
    """Fit-frame ZYX of a patch's grid at fractional (row, col) ``ij``."""
    zyxs = patch.zyxs.numpy().astype(np.float64)
    row, col = ij
    r, c = min(int(row), zyxs.shape[0] - 2), min(int(col), zyxs.shape[1] - 2)
    fr, fc = row - r, col - c
    return ((1 - fr) * (1 - fc) * zyxs[r, c] + (1 - fr) * fc * zyxs[r, c + 1]
            + fr * (1 - fc) * zyxs[r + 1, c] + fr * fc * zyxs[r + 1, c + 1])


def _pairs(inputs, edge_uuid=None):
    return [pcl for pcl in inputs.patch_overlaps.values()
            if edge_uuid is None or pcl['metadata']['edge_uuid'] == edge_uuid]


def assert_pairs_lie_on_their_patches(inputs):
    """Each pair's points sit at their own patch grid positions, a fraction
    of a voxel apart; a reversed endpoint order puts them ~10 voxels off."""
    assert inputs.patch_overlaps
    for pcl in inputs.patch_overlaps.values():
        a, b = pcl['points'][0], pcl['points'][1]
        assert a['on_patch']['id'] != b['on_patch']['id']
        for point in (a, b):
            patch = inputs.patches[point['on_patch']['id']]
            np.testing.assert_allclose(
                point['zyx'], _at(patch, point['on_patch']['ij']), atol=1e-3)
            assert point['p'] == point['zyx'][::-1].tolist()
            assert point['winding_annotation'] == 0.0
            assert osi.on_retained_quad(patch, point['on_patch']['ij'])
        np.testing.assert_allclose(a['zyx'], b['zyx'], atol=0.5)


@pytest.mark.parametrize('first,second', [(LOW, HIGH), (HIGH, LOW)])
def test_pairs_follow_the_stored_endpoint_order(db, first, second):
    _put(db, _plane(), uuid=first)
    _put(db, _plane(10.0), uuid=second)
    [edge] = _detect(db)
    assert edge['a'] == first
    inputs = materialize(db)
    snapshot_edge = inputs.snapshot.relationships_of('patch_overlap')[0]
    # The snapshot orders same-kind endpoints by UUID; pairs keep stored order.
    assert (snapshot_edge.a, snapshot_edge.b) == (LOW, HIGH)
    pairs = _pairs(inputs)
    assert len(pairs) > 1
    assert all(pcl['points'][0]['on_patch']['id'] == first for pcl in pairs)
    assert_pairs_lie_on_their_patches(inputs)
    shift = {first: 0.0, second: 10.0}
    for pcl in pairs:
        a, b = pcl['points'][0]['on_patch'], pcl['points'][1]['on_patch']
        assert b['ij'][1] == pytest.approx(
            a['ij'][1] + (shift[first] - shift[second]) / 2, abs=0.25)


@pytest.mark.parametrize('scales', [(2, 0.5), (0.25, 4)])
def test_non_unit_scales_land_in_fit_voxels(db, scales):
    a = _put(db, _plane(), scale=scales[0])
    b = _put(db, _plane(10.0), scale=scales[1])
    _detect(db)
    inputs = materialize(db)
    assert {inputs.snapshot.objects[a].frame.scale_factor,
            inputs.snapshot.objects[b].frame.scale_factor} == set(scales)
    assert_pairs_lie_on_their_patches(inputs)
    zs = [pcl['points'][0]['zyx'][0] for pcl in inputs.patch_overlaps.values()]
    assert 4.0 <= min(zs) and max(zs) <= 50.0


def test_pairs_off_sampled_quads_are_dropped(db):
    a = _put(db, _plane())
    _put(db, _plane(10.0))
    [edge] = _detect(db)
    store_pairs = db.overlap_correspondences(
        [edge['uuid']], VOLUME, 8.0, 16)[0]['pairs']
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    # The losses sample only quads below z 24, so the pairs above it go.
    inputs = osi.materialize(db, snapshot, z_begin=0, z_end=24, z_margin=0,
                             erode_cells_default=1)
    kept = _pairs(inputs)
    assert all(pcl['points'][0]['zyx'][0] < 26 for pcl in kept)
    lost = inputs.counts['overlap pairs off sampled quads']
    assert lost > 0 and len(kept) + lost == len(store_pairs)
    assert_pairs_lie_on_their_patches(inputs)
    assert inputs.dropped_attachments == []
    assert all(pcl['sampling_member_weight'] == pytest.approx(1 / len(kept))
               and pcl['sampling_unit'] == edge['uuid'] for pcl in kept)
    assert inputs.counts['patch overlaps'] == 1
    assert inputs.counts['overlap pairs'] == len(kept)
    assert a in inputs.patches


def test_an_edge_with_no_sampled_pair_is_reported(db):
    _put(db, _plane())
    # Overlapping only in its first ten columns, which its erosion removes.
    _put(db, _plane(28.0, size=40), meta={'source': {'spiral_patch_erode_cells': 12}})
    [edge] = _detect(db)
    inputs = materialize(db)
    assert inputs.patch_overlaps == {}
    assert inputs.dropped_attachments == [(edge['uuid'], 'no pair on sampled quads')]
    assert inputs.counts['patch overlaps off sampled quads'] == 1
    warning = [line for line in inputs.report() if line.startswith('WARNING')]
    assert len(warning) == 1 and edge['uuid'] in warning[0]


def test_edges_to_unfitted_patches_are_counted(db):
    _put(db, _plane())
    _put(db, _plane(10.0))
    _detect(db)
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    collections, counts = osi.materialize_patch_overlaps(
        db, snapshot, {}, pair_spacing=8.0, max_pairs=16)
    assert collections == {}
    assert counts['patch overlaps with unfitted patches'] == 1


def test_materialization_errors_and_stale_geometry_are_reported(db):
    a = _put(db, _plane())
    b = _put(db, _plane(10.0))
    [edge] = _detect(db)
    whole = {'whole_geometry': True}
    c = _put(db, _plane(200.0))
    plain = db.add_edge({'kind': 'same-surface', 'a': a, 'b': c,
                         'a_anchor': whole, 'b_anchor': whole})
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    assert len(snapshot.relationships_of('patch_overlap')) == 2
    with pytest.raises(osi.ObjectStoreInputError) as error:
        osi.materialize(db, snapshot, z_begin=0, z_end=64, z_margin=0,
                        erode_cells_default=1)
    assert error.value.problems == [f'patch overlap {plain}: edge has no overlap pose']
    # Pairs computed from geometry the snapshot did not freeze are refused.
    snapshot = osi.read_snapshot(db, osi.InputRequest())
    snapshot.relationships = [rel for rel in snapshot.relationships
                              if rel.uuid == edge['uuid']]
    patches, _ = osi.materialize_patches(
        db, snapshot, z_begin=0, z_end=64, z_margin=0, erode_cells_default=1)
    snapshot.objects[b].geometry_revision = '00000000-0000-4000-8000-0000000000aa'
    with pytest.raises(osi.ObjectStoreInputError, match="other than the snapshot's"):
        osi.materialize_patch_overlaps(db, snapshot, patches, pair_spacing=8.0,
                                       max_pairs=16)


def test_edges_weigh_the_same_whatever_their_pair_count(db, tmp_path):
    _put(db, _plane())
    _put(db, _plane(10.0))
    # A second, smaller overlap elsewhere.
    _put(db, _plane(size=12) + [0.0, 40.0, 0.0])
    _put(db, _plane(6.0, size=12) + [0.0, 40.0, 0.0])
    edges = _detect(db, expected=2)
    context = adapter_context(db, tmp_path)
    pairs = [pcl for pcl in context.cross_patch_pcls
             if pcl['metadata'].get('input_role') == 'patch_overlap']
    counts = {edge['uuid']: len([p for p in pairs if p['metadata']['edge_uuid'] == edge['uuid']])
              for edge in edges}
    assert len(set(counts.values())) == 2 and min(counts.values()) >= 1
    # Every draw of a batch, not just the first, picks the two edges equally,
    # stratified or from the combined pool.
    import losses
    for stratified in (True, False):
        cfg = dict(context.config, pcl_stratified_pcl_sampling=stratified)
        np.random.seed(0)
        drawn = {edge: 0 for edge in counts}
        for _ in range(2000):
            for index in losses._choose_pcl_indices(
                    context.pcl_sampling_strata['cross_patch'], 2, cfg):
                drawn[context.cross_patch_pcls[index]['metadata']['edge_uuid']] += 1
        assert min(drawn.values()) / max(drawn.values()) > 0.97, drawn
    # Each pair is its own two-node chain on two patches.
    for pcl in pairs:
        assert len(pcl['chain'].iter_chain()) == 2
        assert len(pcl['points_by_patch']) == 2


def _seam_sheet(winding, theta_begin, theta_end):
    zs = np.arange(4.0, 61.0, 4.0)
    radius = reference.DR * (winding + 0.5)
    thetas = np.arange(theta_begin, theta_end + 1e-9, 4.0 / radius)
    return np.asarray([[spiral_zyx(t, winding, z) for t in thetas] for z in zs])


def test_an_overlap_across_the_seam_is_satisfied_by_the_same_sheet(db, tmp_path):
    # One sheet seen by two patches: winding 8 up to theta 2pi + 0.35 is
    # winding 9 from theta 0, so their overlap straddles theta = 0.
    before = put_patch(db, _seam_sheet(8, 5.2, 2 * np.pi + 0.35), (0.25, 0.25))
    after = put_patch(db, _seam_sheet(9, -0.35, 1.0), (0.25, 0.25))
    _detect(db)
    context = adapter_context(db, tmp_path)
    pairs = [pcl for pcl in context.cross_patch_pcls
             if pcl['metadata'].get('input_role') == 'patch_overlap']
    assert pairs
    # Some pairs lie on either side of the seam.
    thetas = [np.arctan2(p['points'][0]['zyx'][1] - reference.CENTRE_YX[0],
                         p['points'][0]['zyx'][2] - reference.CENTRE_YX[1])
              for p in pairs]
    assert min(thetas) < 0 < max(thetas)
    assert {before, after} == {pid for p in pairs for pid in p['points_by_patch']}
    losses = reference.loss_summary(context)
    # Without the seam corrections the patches would be a winding (12 voxels) apart.
    assert max(losses['rel_winding']) < 0.5
    assert max(losses['rel_winding']) >= 0.0


def test_sampling_units_draw_units_then_members():
    """The reviewer's case: edges of 1 and 16 pairs, two draws per step."""
    import losses
    cfg = {'pcl_sampling_weights': None, 'pcl_stratified_pcl_sampling': True}
    groups = [osi.PATCH_OVERLAP_GROUP] * 17
    weights = [1.0] + [1 / 16] * 16
    units = ['small'] + ['large'] * 16
    strata = losses.build_pcl_sampling_strata(groups, cfg, weights, units)
    assert strata['effective_size'] == 2
    np.random.seed(1)
    drawn = {'small': 0, 'large': 0}
    for _ in range(5000):
        for index in losses._choose_pcl_indices(strata, 2, cfg):
            drawn[units[index]] += 1
    assert drawn == {'small': 5000, 'large': 5000}
    # Pools without multi-member units draw exactly as before.
    plain = losses.build_pcl_sampling_strata(groups, cfg, weights,
                                             [None] * 17)
    assert plain['member_units'] == [None] and plain['all_units'] is None
    np.random.seed(2)
    expected = np.random.choice(plain['strata'][0], 3, replace=False,
                                p=plain['member_probs'][0])
    np.random.seed(2)
    np.testing.assert_array_equal(losses._choose_pcl_indices(plain, 3, cfg), expected)
