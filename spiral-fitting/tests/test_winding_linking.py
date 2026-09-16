"""Analysis linking uses real geometry through both projection backends."""
import copy
import json

import numpy as np
import pytest
import torch

import point_collection as pc
from fit_spiral import inward_winding_direction
from tifxyz import Patch
from winding_linking import (
    AnalysisLinkingConfig, AnalysisPointLinker, model_inward_direction, sampled_fiber_view,
)
from test_point_collection import _collection, _grid_patch, _z_plane, _attached


@pytest.fixture(params=['fallback', 'native'])
def backend(request, monkeypatch):
    if request.param == 'fallback':
        monkeypatch.setattr(pc, '_load_surface_index_backend', lambda: None)
    elif pc._load_surface_index_backend() is None:
        pytest.skip('native surface index is unavailable')
    return request.param


def fiber(cid, points, tag='V'):
    return _collection(cid, points, metadata={
        'logical_input_kind': 'fiber', 'hv_classification': {'manual_tag': tag}})


def policy(**kwargs):
    return AnalysisLinkingConfig.from_json(kwargs).resolve_distance(2.5)


def inward_z(points):
    return np.tile([1., 0., 0.], (len(points), 1))


def test_default_config_inherits_distance_and_keeps_side_filters_off():
    cfg = policy()
    assert cfg.distance == 2.5
    assert cfg.window_points == 5 and cfg.min_points == 1
    assert cfg.vertical_max_behind_fraction is None
    assert cfg.horizontal_max_front_fraction is None
    assert cfg.fiber_stride == cfg.patch_stride == 1


@pytest.mark.parametrize('settings', [
    None, [], {'unknown': 2}, {'distance': True}, {'distance': None},
    {'distance': 0}, {'distance': float('inf')}, {'side_margin_voxels': -1},
    {'vertical_max_behind_fraction': 1.1}, {'horizontal_max_front_fraction': float('nan')},
    {'window_points': 4}, {'window_points': 3.0}, {'min_points': 6},
    {'fiber_stride': True}, {'patch_stride': 0}, {'selection': 'area'},
    {'fiber_stride': 2, 'fiber-stride': 3},
])
def test_bad_config(settings):
    with pytest.raises(ValueError):
        AnalysisLinkingConfig.from_json(settings)


def test_distance_is_fractional_and_inclusive_for_annotations(backend):
    patch = _z_plane(50, 0, 0)
    pcl = _collection(1, [(51.75, 5, 5), (51.8, 5, 5)])
    linker = AnalysisPointLinker(policy(distance=1.75, min_points=5), inward_z)
    linker({'p': patch}, {1: pcl}, {})
    assert _attached(pcl) == ['p', None]
    assert linker.report['counts']['annotations']['attached_points'] == 1


def test_strict_support_does_not_relax_for_short_fibers(backend):
    pcl = fiber(1, [(50, 5, x) for x in (5, 10, 15)])
    linker = AnalysisPointLinker(policy(min_points=4), inward_z)
    linker({'p': _z_plane(50, 0, 0)}, {1: pcl}, {1: pcl})
    assert _attached(pcl) == [None] * 3
    assert linker.report['counts']['fibers']['support_rejected_candidates'] == 3


@pytest.mark.parametrize('absolute', [False, True])
def test_absolute_anchors_ignore_checkpoint_annotation_support(backend, absolute):
    pcl = _collection(1, [(50, 5, 5), (60, 5, 5), (70, 5, 5)],
                      metadata={'winding_is_absolute': absolute})
    linker = AnalysisPointLinker(
        policy(distance=1.75, min_points=3), inward_z,
        annotation_options=pc.PatchLinkOptions(window_points=5, window_min_points=3))
    linker({'p': _z_plane(50, 0, 0)}, {1: pcl}, {})
    assert _attached(pcl) == (['p', None, None] if absolute else [None] * 3)
    counts = linker.report['counts']['annotations']
    assert counts['attached_points'] == int(absolute)
    assert counts['support_rejected_candidates'] == int(not absolute)


def test_absolute_anchors_ignore_side_rules_in_shared_linker(backend):
    pcl = _collection(1, [(50, 5, 5), (60, 5, 5)],
                      metadata={'winding_is_absolute': True})

    def unexpected_direction(points):
        pytest.fail('absolute anchors must not evaluate fiber side rules')

    pc.link_points_to_patches(
        {'p': _z_plane(49, 0, 0)}, {1: pcl}, tolerance=1.75,
        surface_index_tolerance=1.75,
        options=pc.PatchLinkOptions(
            window_points=5, window_min_points=3, strict_min_points=True,
            rank_by_support=True, side_rules={1: pc.SIDE_FRONT},
            max_wrong_side_fractions={1: 0.0}, inward_direction=unexpected_direction))
    assert _attached(pcl) == ['p', None]


def test_support_beats_closest_point_and_largest_patch(backend):
    broad = _z_plane(50.8, -20, -20)
    small = _grid_patch(2, 2, lambda i, j: (50.1, -2 + 4*i, -2 + 4*j), scale=.01)
    pcl = fiber(1, [(50, 0, x) for x in (-10, 0, 10)])
    linker = AnalysisPointLinker(policy(window_points=3, distance=1.), inward_z)
    linker({'broad': broad, 'small': small}, {1: pcl}, {1: pcl})
    assert _attached(pcl) == ['broad'] * 3


@pytest.mark.parametrize('selection', ['nearest', 'largest_area'])
def test_alternative_selection(backend, selection):
    broad = _z_plane(50.8, -20, -20)
    small = _grid_patch(2, 2, lambda i, j: (50.1, -2 + 4*i, -2 + 4*j), scale=.01)
    pcl = fiber(1, [(50, 0, x) for x in (-10, 0, 10)])
    AnalysisPointLinker(policy(window_points=3, distance=1., selection=selection), inward_z)(
        {'broad': broad, 'small': small}, {1: pcl}, {1: pcl})
    assert _attached(pcl) == ['broad', 'small', 'broad']


def test_rms_ranks_equal_support_before_center_distance():
    # A nearly touches the center but diverges from its neighbors; B fits all.
    pcl = fiber(1, [(50, 0, i) for i in range(3)])
    items = list(pcl['points'].items())
    pc._link_from_hits(
        pcl, 1, items, pc._point_zyxs(items),
        np.array([0, 1, 2, 0, 1, 2]), np.array([0, 0, 0, 1, 1, 1]),
        np.array([1.5, .1, 1.5, .5, .5, .5]), np.zeros((6, 2)),
        distance_scale=1, surface_ids=['a', 'b'], patch_for_surface=lambda _: None,
        patch_areas=np.array([100, 1]),
        options=pc.PatchLinkOptions(window_points=3, strict_min_points=True, rank_by_support=True))
    assert _attached(pcl) == ['b'] * 3


def test_ties_use_patch_id_independent_of_input_order(backend):
    patch = _z_plane(50, 0, 0)
    for names in [('z', 'a'), ('a', 'z')]:
        pcl = fiber(1, [(50, 5, 5)])
        AnalysisPointLinker(policy(), inward_z)(dict.fromkeys(names, patch), {1: pcl}, {1: pcl})
        assert _attached(pcl) == ['a']


@pytest.mark.parametrize(('tag', 'key', 'sign'), [
    ('V', 'vertical_max_behind_fraction', 1),
    ('H', 'horizontal_max_front_fraction', -1),
])
@pytest.mark.parametrize(('threshold', 'expected'), [(.2, 'p'), (.19, None)])
def test_wrong_side_fraction_counts_window_and_accepts_equality(backend, tag, key, sign, threshold, expected):
    zs = [50 - sign, 50 - sign, 50 + sign, 50 - sign, 50 - sign]
    pcl = fiber(1, [(z, 5, 5*i) for i, z in enumerate(zs)], tag)
    config = policy(**{key: threshold}, side_margin_voxels=0, min_points=3)
    linker = AnalysisPointLinker(config, inward_z)
    linker({'p': _z_plane(50, 0, 0)}, {1: pcl}, {1: pcl})
    # A wrong-side center is permitted when its window's fraction passes.
    assert _attached(pcl)[2] == expected


def test_side_margin_and_independent_orientation_switches(backend):
    v = fiber(1, [(50.5, 5, 5)], 'V')
    h = fiber(2, [(49, 5, 5)], 'H')
    linker = AnalysisPointLinker(policy(vertical_max_behind_fraction=0), inward_z)
    linker({'p': _z_plane(50, 0, 0)}, {1: v, 2: h}, {1: v, 2: h})
    assert _attached(v) == ['p']  # exactly at allowed signed margin
    assert _attached(h) == ['p']  # horizontal rule is disabled


def test_undefined_directions_do_not_support_filtered_candidates(backend):
    pcl = fiber(1, [(50, 5, x) for x in (5, 10, 15)])
    linker = AnalysisPointLinker(policy(vertical_max_behind_fraction=1), lambda p: np.zeros_like(p))
    linker({'p': _z_plane(50, 0, 0)}, {1: pcl}, {1: pcl})
    assert _attached(pcl) == [None] * 3
    assert linker.report['counts']['fibers']['undefined_side_hits'] == 3


def test_exact_winding_gradient_pulls_back_through_scan_transform():
    matrix = torch.tensor([[1., 0., 0.], [.3, 1.5, .2], [.1, -.4, .8]])
    points = torch.tensor([[20., 10., 30.], [30., 20., 40.]])
    with torch.enable_grad():
        differentiable = points.clone().requires_grad_()
        image = differentiable @ matrix.T
        winding = torch.linalg.vector_norm(image[:, 1:], dim=-1) / 8 - torch.atan2(image[:, 1], image[:, 2]) / (2*torch.pi)
        grad = torch.autograd.grad(winding.sum(), differentiable)[0]
    expected = -torch.nn.functional.normalize(grad, dim=-1)
    with torch.no_grad():
        actual = inward_winding_direction(lambda p: p @ matrix.T, points.numpy(), dr_per_winding=8.)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6)


def test_winding_direction_is_continuous_at_seam_and_defined_inside_winding_zero():
    points = np.array([[5, 1e-5, 10], [5, -1e-5, 10], [5, .1, 0], [5, 0, 0]])
    directions = inward_winding_direction(lambda p: p, points, dr_per_winding=8.)
    np.testing.assert_allclose(directions[0], directions[1], atol=3e-6)
    assert np.linalg.norm(directions[2]) == pytest.approx(1.)
    assert np.linalg.norm(directions[3]) == 0
    assert abs(directions[0, 1]) > .1  # includes the angular derivative


def test_out_of_domain_points_never_reach_model():
    def transform(points):
        assert torch.isfinite(points).all()
        return points
    inward = model_inward_direction(transform, 8., 'cpu', 0, 100)
    points = np.array([[-5, 2, 2], [50, 2, 2], [101, 2, 2], [np.nan, 2, 2]])
    actual = inward(points)
    assert np.linalg.norm(actual[1]) == pytest.approx(1.)
    assert not actual[[0, 2, 3]].any()


def test_stride_keeps_ids_chain_endpoints_and_branch_targets():
    pcl = fiber(1, [(50, 5, i) for i in range(10)])
    pcl['control_line_indices'] = np.arange(10)
    for i, point in pcl['points'].items():
        point['orig_index'] = i
    pcl['branches'] = [{'local_index': 2}]
    pcl['chain'] = object()
    view = sampled_fiber_view(pcl, 4, forced_controls={7})
    assert list(view['points']) == [0, 2, 4, 7, 8, 9]
    assert len(pcl['points']) == 10 and view['chain'] is pcl['chain']
    assert all(point is pcl['points'][i] for i, point in view['points'].items())


def test_stride_only_attaches_sampled_points_and_preserves_one_sided_branch(backend):
    a = fiber(1, [(50, 5, i) for i in range(10)])
    b = fiber(2, [(50, 6, i) for i in range(10)])
    for pcl, name in [(a, 'a.json'), (b, 'b.json')]:
        pcl.update(file_basename=name, control_line_indices=np.arange(10))
        for i, point in pcl['points'].items():
            point['orig_index'] = i
    a['branches'] = [dict(local_index=2, branch_file='b.json', branch_index=7)]
    inputs = {1: a, 2: b}
    linker = AnalysisPointLinker(policy(fiber_stride=4), inward_z)
    linker({'p': _z_plane(50, 0, 0)}, inputs, inputs)
    assert [i for i, p in a['points'].items() if 'on_patch' in p] == [0, 2, 4, 8, 9]
    assert [i for i, p in b['points'].items() if 'on_patch' in p] == [0, 4, 7, 8, 9]
    assert linker.report['counts']['fibers']['source_points'] == 20
    assert linker.report['counts']['fibers']['sampled_points'] == 10


def test_patch_stride_maps_nonmultiple_boundary_to_original_grid(backend):
    patch = _grid_patch(6, 8, lambda i, j: (50, 10+i, 20+j))
    original = patch.zyxs.clone()
    pcl = _collection(1, [(50.25, 14.5, 26.5), (50, 15, 27)])
    AnalysisPointLinker(policy(patch_stride=3, distance=.5), inward_z)({'p': patch}, {1: pcl}, {})
    assert _attached(pcl) == ['p', 'p']
    np.testing.assert_allclose(pcl['points'][0]['on_patch']['ij'], [4.5, 6.5], atol=1e-5)
    np.testing.assert_allclose(pcl['points'][1]['on_patch']['ij'], [5, 7], atol=1e-5)
    assert pcl['points'][0]['on_patch']['distance'] == pytest.approx(.25)
    torch.testing.assert_close(patch.zyxs, original)


def test_coarse_patch_does_not_bridge_holes(backend):
    grid = _grid_patch(7, 7, lambda i, j: (50, i, j)).zyxs.clone()
    grid[1, 1] = -1
    patch = Patch(grid, torch.ones(2), None, None)
    pcl = _collection(1, [(50, 1.5, 1.5), (50, 4.5, 4.5)])
    AnalysisPointLinker(policy(patch_stride=3, distance=.1), inward_z)({'p': patch}, {1: pcl}, {})
    assert _attached(pcl) == [None, 'p']


def test_removed_coarse_pair_never_links_unrelated_patch(backend):
    grid = _grid_patch(4, 4, lambda i, j: (50, i, j)).zyxs.clone()
    grid[1, 1] = -1
    hole = Patch(grid, torch.ones(2), None, None)
    other = _grid_patch(4, 4, lambda i, j: (50, i, j))
    pcl = _collection(1, [(50, 2.5, 2.5)], name='between_patches__a__b')
    AnalysisPointLinker(policy(patch_stride=3), inward_z)(
        {'a': hole, 'b': hole, 'other': other}, {1: pcl}, {})
    assert _attached(pcl) == [None]


def test_native_and_fallback_match_curved_coarse_surface(monkeypatch):
    native = pc._load_surface_index_backend()
    if native is None:
        pytest.skip('native surface index is unavailable')
    patch = _grid_patch(6, 8, lambda i, j: (50+.02*i*j, 10+i, 20+j))
    pcl = fiber(1, [(50.6, 12.5, x) for x in (22.5, 23.5, 24.5)])
    results = []
    for module in (None, native):
        monkeypatch.setattr(pc, '_load_surface_index_backend', lambda: module)
        current = copy.deepcopy(pcl)
        AnalysisPointLinker(policy(patch_stride=3, vertical_max_behind_fraction=.4), inward_z)(
            {'p': patch}, {1: current}, {1: current})
        results.append([p.get('on_patch') for p in current['points'].values()])
    for a, b in zip(*results):
        assert a is not None and b is not None
        assert a['id'] == b['id']
        np.testing.assert_allclose(a['ij'], b['ij'], atol=2e-5)
        assert a['distance'] == pytest.approx(b['distance'], abs=2e-5)


def test_loader_links_raw_fibers_before_deriving_components(tmp_path, monkeypatch, backend):
    from config import Config
    from find_inconsistent_windings import build_fit_inputs
    from fit_spiral import FitContext

    points = [[4*x, 20, 200] for x in range(10)]
    (tmp_path / 'fiber.json').write_text(json.dumps({
        'control_points': [points[0], points[-1]], 'line_points': points}))
    cfg = Config(dict(z_begin=0, z_end=100, pcl_fiber_min_point_spacing=40)).as_dict()
    fit_cfg, scroll, paths, _, _ = build_fit_inputs(
        dict(cfg=cfg, z_begin=0, z_end=100), 'unused', [], 0, 100,
        'unused-umbilicus.json', fibers_path=str(tmp_path))
    ctx = FitContext(fit_cfg, scroll=scroll, paths=paths)
    monkeypatch.setattr(ctx, 'umbilicus_z_to_yx', lambda: lambda z: np.zeros((len(z), 2)))
    monkeypatch.setattr(ctx, '_load_patches_from_dir', lambda *a: {'p': _z_plane(50, 0, 0)})
    linker = AnalysisPointLinker(policy(fiber_stride=4, min_points=3), inward_z)
    derive = ctx._derive_point_inputs

    class Derived(Exception):
        pass

    def stop_after_derivation(*args, **kwargs):
        derive(*args, **kwargs)
        raise Derived

    monkeypatch.setattr(ctx, '_derive_point_inputs', stop_after_derivation)
    with pytest.raises(Derived):
        ctx.load_host_inputs(point_linker=linker, fiber_min_point_spacing=0.)
    assert len(ctx.fiber_catalog['fiber']['points']) == 10
    assert linker.report['counts']['fibers']['sampled_points'] == 4
    assert linker.report['counts']['fibers']['attached_points'] == 4
    assert len(ctx.cross_patch_pcls) == 1
    assert set(ctx.cross_patch_pcls[0]['points_by_patch']) == {'p'}
    assert len(ctx.cross_patch_pcls[0]['points_by_patch']['p']) == 4
    assert ctx.config['pcl_fiber_min_point_spacing'] == 40


def test_side_gate_uses_triangle_projection_not_bilinear_surface(backend):
    patch = _grid_patch(2, 2, lambda i, j: (50 + 4*i*j, 10+10*i, 20+10*j))
    # This point lies exactly on the upper triangle, but the bilinear surface
    # is in front of it. A zero-slack horizontal rule must still attach it.
    pcl = fiber(1, [(52, 17.5, 27.5)], 'H')
    AnalysisPointLinker(policy(distance=.1, horizontal_max_front_fraction=0,
                               side_margin_voxels=0), inward_z)({'p': patch}, {1: pcl}, {1: pcl})
    assert _attached(pcl) == ['p']


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_writes_report_with_linking_before_host_derivation(tmp_path, monkeypatch, backend, enabled):
    from click.testing import CliRunner
    import checkpoint_io
    import find_inconsistent_windings as analysis
    from config import Config
    from fit_spiral import FitContext

    class CpuTorch:
        def __getattr__(self, name):
            return (lambda *a, **kw: torch.device('cpu')) if name == 'device' else getattr(torch, name)

    monkeypatch.setattr(analysis, 'torch', CpuTorch())
    checkpoint = tmp_path / 'model.ckpt'
    checkpoint.touch()
    umbilicus = tmp_path / 'umbilicus.json'
    umbilicus.write_text(json.dumps({'control_points': [
        {'z': 0, 'y': 0, 'x': 0}, {'z': 100, 'y': 0, 'x': 0}]}))
    annotations = tmp_path / 'abs_winding.json'
    annotations.write_text(json.dumps({
        'vc_pointcollections_json_version': '1', 'collections': {
            '0': {'name': 'anchor', 'metadata': {'winding_is_absolute': True},
                  'points': {'0': {'p': [5, 5, 50], 'wind_a': 3}}}}}))
    fibers_dir = tmp_path / 'fibers'
    fibers_dir.mkdir()
    points = [[4*x, 20, 200] for x in range(10)]
    (fibers_dir / 'fiber.json').write_text(json.dumps({
        'control_points': [points[0], points[-1]], 'line_points': points,
        'hv_classification': {'manual_tag': 'V'}}))
    monkeypatch.setattr(checkpoint_io, 'load_checkpoint_cpu', lambda _: dict(
        cfg=Config().as_dict(), z_begin=0, z_end=100))
    events = []

    def transform(*args, **kwargs):
        events.append('transform')
        return lambda p: p, torch.tensor(8.)

    def load_patches(*args):
        events.append('patches')
        return {'p': _z_plane(50, 0, 0)}

    monkeypatch.setattr(analysis, 'build_transform', transform)
    monkeypatch.setattr(FitContext, '_load_patches_from_dir', load_patches)
    output = tmp_path / 'report.json'
    config = dict(checkpoint=str(checkpoint), patches_dir=str(tmp_path),
                  umbilicus=str(umbilicus), patch_id='p', pcl=[str(annotations)],
                  fibers=str(fibers_dir), z_range='0,100', output=str(output),
                  max_hops=0, detect_loops=False, min_fix=False)
    if enabled:
        config['linking'] = dict(distance=.75, min_points=3, fiber_stride=2,
                                 patch_stride=3, vertical_max_behind_fraction=0)
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(config))
    # Avoid a global no-grad change leaking from the command into other tests.
    with torch.enable_grad():
        result = CliRunner().invoke(analysis.main, ['--config', str(params)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['num_absolute_pcls'] == 1
    assert report['num_direct_votes'] == 1
    if enabled:
        assert events == ['transform', 'patches']
        assert report['linking']['settings']['distance'] == .75
        assert report['linking']['counts']['fibers']['sampled_points'] == 6
        assert report['linking']['counts']['fibers']['attached_points'] == 6
    else:
        assert events == ['patches', 'transform']
        assert 'linking' not in report
