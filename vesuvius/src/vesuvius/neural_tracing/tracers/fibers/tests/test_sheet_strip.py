"""Sheet-strip follower: the tube channel starts without influence, strip targets of analytic surfaces, tube
rendering, and gradients reach the strip head."""
import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import (
    SheetStripConfig, SheetStripFollower, band_layout, render_tube, strip_grid)
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.targets import strip_targets
from vesuvius.neural_tracing.tracers.models.crop_transformer import RegressionConfig, RegressionFollower
from vesuvius.neural_tracing.tracers.models.weights import add_input_channels, load_matching_weights
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec

SMALL = dict(fine=CropSpec(depth=32, width=32, behind=16, spacing=.5), hidden=32, layers=2, heads=2, ffn=64,
             n_future=6, gate_plane=4, n_history=8, max_recovery_distance=3., cnn_channels=(8, 16), cnn_blocks=(1, 1), cnn_strides=(2, 2),
             cnn_stem=False, cell_tokens='polar', patch_size=8, patch_points=2, polar_tube_radius=3.,
             polar_ring_widths=(3., 4.), polar_ring_growth=1.5, polar_pool_spacing=4., multiscale_fine_width=16,
             multiscale_coarse_channels=(4, 8), recurrent_refinement_steps=1, scoring_pass=True)
STRIP = dict(model_type='sheet_strip', strip_half_width=6., strip_v_spacing=2.)


def inputs(cfg, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = cfg.fine
    x = dict(fine=torch.randn(b, 1, c.depth, c.width, c.width, generator=g),
             path_geometry=torch.zeros(b, 4, 8), path_geometry_valid=torch.zeros(b, 4, dtype=torch.bool))
    hist = torch.randn(b, cfg.n_history, 3, generator=g)
    hist[..., 2] = -torch.arange(cfg.n_history, dtype=torch.float32)*.5
    return x, hist, torch.ones(b, cfg.n_history)


def test_tube_channel_starts_without_influence():
    torch.manual_seed(0)
    base = RegressionFollower(RegressionConfig(**SMALL)).eval()
    strip = SheetStripFollower(SheetStripConfig(**SMALL, **STRIP)).eval()
    report = load_matching_weights(strip, add_input_channels(strip, base.state_dict()))
    assert not report['mismatched'] and not report['unexpected']
    assert all(name.startswith('strip_') for name in report['fresh'])
    x, hist, hmask = inputs(base.cfg)
    threshold = torch.tensor(2.)
    with torch.no_grad():
        expected = base.training_forward(x, hist, hmask, threshold)
        tube = dict(x, fine=torch.cat((x['fine'], torch.rand_like(x['fine'])), 1))
        output = strip.training_forward(tube, hist, hmask, threshold)
    for key in ('refinement_points', 'refinement_hazard_logits'):
        torch.testing.assert_close(output[key], expected[key], rtol=0, atol=1e-5)
    assert output['strip_heights'].shape == (2, 2, *(len(g) for g in strip_grid(strip.cfg)))


def test_path_keeps_its_lateral_limit_in_a_wider_crop():
    wide = dict(SMALL, fine=CropSpec(depth=32, width=48, behind=16, spacing=.5), multiscale_fine_width=16)
    cfg = SheetStripConfig(**wide, model_type='sheet_strip', path_lateral_limit=6.75, strip_half_width=10.)
    assert cfg.lateral_limit == 6.75 and cfg.input_channels == 2


def surface_window(height, extent=40., cell=2.):
    """Crop-local node grid (h, w, 3) of the surface u = height(v, f) over [-extent, extent]^2."""
    grid = np.arange(-extent, extent+1e-9, cell)
    v, f = np.meshgrid(grid, grid, indexing='ij')
    return np.stack((height(v, f), v, f), -1), np.ones(v.shape, bool)


def test_strip_targets_of_a_tilted_plane():
    f, v = np.linspace(-10, 10, 6), np.linspace(-8, 8, 5)
    plane = lambda vv, ff: .3*vv-.2*ff+1.
    heights, mask = strip_targets([surface_window(plane)], f, v)
    assert mask.all()
    np.testing.assert_allclose(heights, plane(*np.meshgrid(v, f)), atol=1e-5)
    # A second patch agreeing within the tolerance is averaged; one a sheet away masks every point.
    shifted = lambda d: surface_window(lambda vv, ff: plane(vv, ff)+d)
    heights, mask = strip_targets([surface_window(plane), shifted(1.)], f, v, fold_tolerance=2.)
    assert mask.all() and np.allclose(heights, plane(*np.meshgrid(v, f))+.5, atol=1e-5)
    _, mask = strip_targets([surface_window(plane), shifted(6.)], f, v, fold_tolerance=2.)
    assert not mask.any()


def test_strip_targets_mask_steep_holes_and_range():
    f, v = np.linspace(-10, 10, 6), np.linspace(-8, 8, 5)
    steep = lambda vv, ff: 5.*vv  # 79 degrees from the (v, f) plane
    assert not strip_targets([surface_window(steep)], f, v, max_slope_deg=70.)[1].any()
    local, valid = surface_window(lambda vv, ff: 0.*vv)
    valid[:20] = False  # the patch ends at v = 0 (axis 0 is v): no triangle covers v < 0
    _, mask = strip_targets([(local, valid)], f, v)
    assert not mask[:, v < 0].any() and mask[:, v > 0].all()
    _, mask = strip_targets([surface_window(lambda vv, ff: 0.*vv+9.)], f, v, max_height=8.)
    assert not mask.any()


def test_render_tube_peaks_on_the_curve():
    crop = CropSpec(depth=32, width=24, behind=16, spacing=.5)
    c = np.arange(-8., 8., .25)
    points = torch.tensor(np.stack((np.full_like(c, 1.), np.full_like(c, -2.), c), -1), dtype=torch.float32)[None]
    tube = render_tube(points, torch.ones(1, len(c), dtype=torch.bool), crop, sigma=1.)[0, 0]
    d, h, w = 16, int((-2./.5)+11.5), int(1./.5+11.5)  # sample of (a, b, c) = (1, -2, 0): w, h at .5 offsets
    profile = tube[d, h, :]
    assert profile.argmax() in (w, w+1)
    assert abs(float(profile.max())-np.exp(-.5/2)) < 1e-4  # half a sample off in h and w: squared distance .5
    assert float(tube[d, h, 0]) == 0. and abs(float(tube[0, h, w])-np.exp(-.5/2)) < 1e-4  # the whole crop depth


def test_strip_head_receives_gradients():
    cfg = SheetStripConfig(**SMALL, **STRIP)
    model = SheetStripFollower(cfg)
    x, hist, hmask = inputs(cfg)
    x['fine'] = torch.cat((x['fine'], torch.rand_like(x['fine'])), 1)
    heights = model.training_forward(x, hist, hmask, torch.tensor(2.))['strip_heights']
    heights.square().mean().backward()
    for name in ('strip_query.0.weight', 'strip_height.weight', 'strip_refinement.0.weight', 'cnn.stages.0.blocks.0.conv1.conv.weight'):
        assert model.get_parameter(name).grad is not None, name
    assert model.get_parameter('cnn.stages.0.blocks.0.conv1.conv.weight').shape[1] == 2


def test_stitch_joins_decisions_with_opposite_in_sheet_axes():
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import stitch
    # The sheet z = .2 x + .1 y around a fiber along y; heights u in frames whose normal is z and whose in-sheet axis
    # v is +-x. Decisions every 2 units, f every 1; the second's v (and u) axes are flipped, its columns reversed.
    f, v = np.arange(-4., 4.5, 1.), np.linspace(-3, 3, 5)
    arcs = np.array([10., 12.])
    surface = lambda x, y: .2*x+.1*y
    frames, heights = [], []
    for k, (t, sign) in enumerate(zip(arcs, (1., -1.))):
        frames.append(np.array([[0., sign, 0.], [0., 0., 1.], [sign, 0., 0.]]))  # columns u, v, f: z, +-x, y
        x, y = np.meshgrid(v*sign, t+f, indexing='xy')
        heights.append(surface(x, y)*sign)
    positions = np.stack([np.array([0., t, 0.]) for t in arcs])
    flip = np.array([False, True])
    oriented = np.stack([h[:, ::-1]*-1 if fl else h for h, fl in zip(heights, flip)])
    frames = np.stack(frames)
    frames[1, :, :2] *= -1
    rows, weight, spread = stitch(oriented, frames, positions, arcs, f, v, blend=3.)
    keep = weight > 0
    assert keep.sum() == (4+2+1)*len(v)  # rows within the blend of either head (|f| < 3)
    np.testing.assert_allclose(rows[keep][:, 2], surface(rows[keep][:, 0], rows[keep][:, 1]), atol=1e-9)
    np.testing.assert_allclose(rows[keep][:, 0].reshape(-1, len(v)), np.broadcast_to(v, (keep.sum()//len(v), len(v))))
    assert spread[keep].max() < 1e-6


def test_strip_profiles_read_the_crop_along_u():
    cfg = SheetStripConfig(**SMALL, **STRIP, strip_profile_reach=1.)
    model = SheetStripFollower(cfg).eval()
    c = cfg.fine
    d, h, w = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in (c.depth, c.width, c.width)), indexing='ij')
    image = torch.stack((w, h+100*d), 0)[None]  # channel 0: the u sample index; channel 1: v and f
    points = torch.tensor([[[[1., -2., 3.]]]])  # (B, T, n, 3) crop-local (u, v, f)
    out = model.strip_profiles(dict(image=image), points).reshape(len(model.strip_profile_offsets), 2)
    u = (1.+model.strip_profile_offsets)/c.spacing+(c.width-1)/2
    assert torch.allclose(out[:, 0], u)
    assert torch.allclose(out[:, 1], torch.full_like(u, -2./c.spacing+(c.width-1)/2+100*(3./c.spacing+c.behind)))


def test_strip_options_start_without_influence():
    torch.manual_seed(0)
    base = SheetStripFollower(SheetStripConfig(**SMALL, **STRIP)).eval()
    x, hist, hmask = inputs(base.cfg)
    x['fine'] = torch.cat((x['fine'], torch.rand_like(x['fine'])), 1)
    with torch.no_grad():
        reference = base.forward_strip(x, hist, hmask)
    for option in (dict(strip_profile_reach=1.), dict(strip_full_fine=True)):
        model = SheetStripFollower(SheetStripConfig(**SMALL, **STRIP, **option)).eval()
        report = load_matching_weights(model, base.state_dict())
        assert all(name.startswith('strip_profile') for name in report['fresh']) and not report['mismatched']
        with torch.no_grad():
            out = model.forward_strip(x, hist, hmask)
        if 'strip_full_fine' in option:  # the fine features change outside the central box only
            assert out.shape == reference.shape and torch.isfinite(out).all()
        else:
            assert torch.allclose(out, reference)


def test_band_tokens_tile_the_sheet_band():
    cfg = SheetStripConfig(**SMALL, **STRIP, strip_full_fine=True, strip_band_half_height=4., strip_band_cell=4)
    points, centers = band_layout(cfg)
    c = cfg.fine
    slabs, rows, columns = c.depth//cfg.patch_size, 2, c.width//4
    assert points.shape == (slabs*rows*columns, cfg.patch_points**3, 3) and centers.shape == (len(points), 3)
    assert torch.allclose(points.mean(1), centers, atol=1e-5)
    assert (points[..., 0].abs() <= 4*c.spacing).all() and (points[..., 1].abs() <= (c.width-1)/2*c.spacing).all()
    model = SheetStripFollower(cfg)
    x, hist, hmask = inputs(cfg)
    x['fine'] = torch.cat((x['fine'], torch.rand_like(x['fine'])), 1)
    plain = SheetStripFollower(SheetStripConfig(**SMALL, **STRIP, strip_full_fine=True))
    assert model.context(x, hist, hmask)['padding'].shape[1] == plain.context(x, hist, hmask)['padding'].shape[1]+len(points)
    model.strip(model.context(x, hist, hmask)).sum().backward()
    assert model.strip_band.weight.grad.abs().sum() > 0


def test_crop_orientation_against_the_scroll():
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import crop_orientation
    umbilicus = np.array([[10., 20., 0.], [10., 20., 100.]])
    # A crop 5 voxels in +x of the axis; u along +x (outward), v along z, f along -y.
    frame = np.array([[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]])
    out = crop_orientation(np.array([[15., 20., 50.]])*.5, frame, umbilicus, .5)
    assert np.allclose(out, [[1., 0., 0., 1., 0.]])
    cfg = SheetStripConfig(**SMALL, **STRIP, strip_orientation=True)
    model = SheetStripFollower(cfg)
    x, hist, hmask = inputs(cfg)
    x['fine'] = torch.cat((x['fine'], torch.rand_like(x['fine'])), 1)
    x['orientation'] = torch.from_numpy(np.repeat(out, 2, 0))
    model.strip(model.context(x, hist, hmask)).sum().backward()
    assert model.strip_orient[0].weight.grad is not None


def test_uncertainty_probe_leaves_the_heights_alone():
    torch.manual_seed(0)
    cfg = SheetStripConfig(**SMALL, **STRIP, strip_uncertainty=True)
    model = SheetStripFollower(cfg)
    x, hist, hmask = inputs(cfg)
    x['fine'] = torch.cat((x['fine'], torch.rand_like(x['fine'])), 1)
    out = model.training_forward(x, hist, hmask, torch.tensor(2.))
    assert out['strip_log_scale'].shape == out['strip_heights'].shape[:1]+out['strip_heights'].shape[2:]
    out['strip_log_scale'].sum().backward()
    assert model.strip_log_scale[0].weight.grad.abs().sum() > 0
    assert model.strip_height.weight.grad is None and model.strip_query[0].weight.grad is None


def test_stitch_values_average_decisions_per_node():
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import stitch_values
    f = np.arange(-8., 8.)
    arcs = np.array([20., 24., 28.])
    values = np.broadcast_to(np.array([1., 2., 3.])[:, None, None], (3, len(f), 2)).copy()
    out = stitch_values(values, arcs, f, blend=4.)
    rows = np.flatnonzero(np.isfinite(out[:, 0]))
    assert np.allclose(out[rows[0], :], 1.) and np.allclose(out[rows[-1], :], 3.)
    assert ((out[rows] >= 1.) & (out[rows] <= 3.)).all()
    # Between two heads the nearer decision weighs more.
    assert out[int(np.rint(arcs[1]-arcs[0]-f[0]-1)), 0] < 2.


def test_remote_seed_heading_is_skipped():
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import skip_remote_seed_heading
    near = dict(seed_heading_family='H', seed_pos=np.zeros(3), pos=np.array([0., 0., 10.]))
    far = dict(seed_heading_family='V', seed_pos=np.zeros(3), pos=np.array([0., 0., 100.]))
    skip_remote_seed_heading(near, 60.)
    skip_remote_seed_heading(far, 60.)
    assert near['seed_heading_family'] == 'H' and 'fiber_family' not in near
    assert 'seed_heading_family' not in far and far['fiber_family'] == 'V'


def test_seed_heading_is_reused_per_seed(monkeypatch):
    """reuse_seed_heading leaves an item as data/targets.resolve_trace_seed would, computing each seed's heading once."""
    from vesuvius.neural_tracing.tracers.fibers.data import targets
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer
    calls = []

    def heading(vol, pos, family, direction):
        calls.append(family)
        return np.array([0., 0., 1.]) if np.dot([0., 0., 1.], direction) >= 0 else np.array([0., 0., -1.])

    monkeypatch.setattr(infer, 'oriented_seed_heading', heading)
    monkeypatch.setattr(targets, 'oriented_seed_heading', heading)
    item = lambda: dict(seed_heading_family='H', seed_pos=np.zeros(3), seed_tangent=np.array([.1, 0., .9]))
    expected = targets.resolve_trace_seed(item(), None)
    cache, items = {}, [item() for _ in range(3)]
    for it in items:
        infer.reuse_seed_heading(it, None, cache)
    assert calls == ['H', 'H']  # once for resolve_trace_seed, once for the three reused items
    for it in items:
        assert it.keys() == expected.keys() and it['fiber_family'] == 'H'
        assert np.array_equal(it['seed_tangent'], expected['seed_tangent'])
        assert targets.resolve_trace_seed(it, None) is it and len(calls) == 2  # nothing left to resolve
    pending = dict(item(), _pending_seed_heading=('fiber', 0., False, None))
    infer.reuse_seed_heading(pending, None, cache)
    assert pending['seed_heading_family'] == 'H'  # a seed decision's crop heading stays with resolve_trace_seed


def test_display_grid_puts_higher_z_on_top():
    """An H strip runs along the columns and a V strip along the rows, row 0 the highest z."""
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all import display_grid
    R, C = 7, 3
    f, v = np.meshgrid(np.arange(R, dtype=float), np.arange(C, dtype=float), indexing='ij')
    keep = np.ones((R, C), bool)
    keep[0, 0] = False
    # H fiber along x, its strip's v axis rising in z.
    h = np.stack([f, np.zeros_like(f), v], -1)
    rows, kept, expected, layout = display_grid(h, keep, f*10+v, 'H')
    assert rows.shape == (C, R, 3) and layout == dict(fiber_along='columns', fiber_reversed=False, across_reversed=True)
    assert (np.diff(rows[..., 2], axis=0) < 0).all() and (np.diff(rows[..., 0], axis=1) > 0).all()
    assert not kept[-1, 0] and kept.sum() == keep.sum() and expected[-1, 1] == 10.
    # V fiber rising in z along arclength: reversed so its top end is row 0.
    vert = np.stack([v, np.zeros_like(f), f], -1)
    rows, kept, expected, layout = display_grid(vert, keep, None, 'V')
    assert rows.shape == (R, C, 3) and expected is None
    assert layout == dict(fiber_along='rows', fiber_reversed=True, across_reversed=False)
    assert (np.diff(rows[..., 2], axis=0) < 0).all() and not kept[-1, 0]
    # Already highest at the start: unchanged.
    rows, kept, _, layout = display_grid(vert[::-1], keep[::-1], None, 'V')
    assert not layout['fiber_reversed'] and np.array_equal(rows, vert[::-1])


def test_trim_ends_drops_rows_near_and_beyond_the_fiber_ends():
    """Rows (arclength -6 + 4 r) within 10 of either end of a 40-long fiber, or beyond it, are dropped; masked nodes
    stay masked."""
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all import trim_ends
    keep = np.ones((14, 3), bool)
    keep[5, 1] = False
    trimmed = trim_ends(keep, -6., 4., 40., 10.)
    arc = -6.+4.*np.arange(14)
    assert np.array_equal(trimmed.any(1), (arc >= 10) & (arc <= 30))
    assert not trimmed[5, 1] and trimmed[5, 0]
    assert np.array_equal(trim_ends(keep, -6., 4., 40., 0.).any(1), (arc >= 0) & (arc <= 40))


def test_transfer_to_single_lattice_keeps_what_it_can():
    """transfer.py: the widened fine CNN keeps the source's features (context block off at the start), a point
    reader gives the source's output when the dropped coarse inputs sit at their mean, and the result runs."""
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.transfer import POINT_READERS, map_columns, transfer
    torch.manual_seed(0)
    two_level = SheetStripConfig(**SMALL, **STRIP, strip_full_fine=True)
    source = SheetStripFollower(two_level).eval()
    cfg = SheetStripConfig(**{**two_level.to_dict(), 'multiscale_fine_width': None, 'strip_full_fine': False,
                              'cnn_channels': (8, 24), 'cnn_context_channels': 16})
    fine, coarse = two_level.cnn_channels[-1], two_level.multiscale_coarse_channels[-1]
    means = {name: torch.randn(getattr(source, name)[0].in_features) for name in POINT_READERS}
    state, new = transfer(source, cfg, means)
    target = SheetStripFollower(cfg).eval()
    target.load_state_dict(state)
    assert new and all(n.startswith('cnn.context_') for n in new)
    image = torch.randn(2, cfg.input_channels, cfg.fine.depth, cfg.fine.width, cfg.fine.width)
    with torch.no_grad():
        assert torch.allclose(target.cnn(image)[:, :fine], source.cnn(image), atol=1e-5)
        # strip_query: one point (fine, coarse, support) and two anchor inputs; coarse at its mean.
        old = source.strip_query[0]
        x = torch.randn(5, old.in_features)
        x[:, fine:fine+coarse] = means['strip_query'][fine:fine+coarse]
        y = torch.cat((x[:, :fine], torch.randn(5, cfg.cnn_channels[-1]-fine), x[:, fine+coarse:]), 1)
        w, b = map_columns((old.weight, old.bias), (target.strip_query[0].weight, target.strip_query[0].bias), 1,
                           fine, coarse, True, means['strip_query'])
        assert torch.allclose(y@w.T+b, old(x), atol=1e-5)
        x_in, hist, hmask = inputs(cfg)
        x_in['fine'] = torch.cat((x_in['fine'], torch.zeros_like(x_in['fine'])), 1)
        heights = target.forward_strip(x_in, hist, hmask)
    assert torch.isfinite(heights).all() and not hasattr(target, 'coarse_cnn')
