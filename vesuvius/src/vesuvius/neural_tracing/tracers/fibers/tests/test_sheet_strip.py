"""Sheet-strip follower: the tube channel starts without influence, strip targets of analytic surfaces, tube
rendering, and gradients reach the strip head."""
import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import (
    SheetStripConfig, SheetStripFollower, render_tube, strip_grid)
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
