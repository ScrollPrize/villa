"""Crop-heading model: in-crop target, inputs from the follower's code, predictor contract, trainer config."""
import json

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.data import HeadingSampling, build_states, cone, heading_state
from vesuvius.neural_tracing.fiber_follow.heading_model.model import (
    HeadingConfig, HeadingNet, HeadingPredictor, main_crop_forward, path_features, prior_frames)
from vesuvius.neural_tracing.fiber_follow.heading_model.targets import in_crop_heading, lateral_extent
from vesuvius.neural_tracing.fiber_follow.heading_model.train import learning_rate, read_config
from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, frame_from_heading, normalize, tangent_at



def volume(root):
    """Tiny CT-only volume (80^3 native CT, trace grid at half resolution) with per-crop z-score records."""
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_EPSILON, ZSCORE_METHOD, volume_key
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
    path = root/'ct/0'
    path.mkdir(parents=True)
    (path/'.zarray').write_text(json.dumps(dict(shape=[80, 80, 80], chunks=[80, 80, 80], dtype='|u1', fill_value=0,
                                                order='C', filters=None, compressor=None, zarr_format=2)))
    z, y, x = np.indices((80, 80, 80))
    (path/'0.0.0').write_bytes(((x+2*y+z) % 256).astype(np.uint8).tobytes())
    spec = FiberVolumeSpec(str(root/'fields'), ct_zarr=str(root/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct',
                           load_presence=False)
    spec.ct_normalization = dict(method=ZSCORE_METHOD, volume=volume_key(spec), epsilon=ZSCORE_EPSILON)
    return FiberVolume(spec, cache_bytes=1 << 20)


def line_fiber(length=200.):
    p = np.c_[np.arange(length+1.), np.zeros(int(length)+1), np.zeros(int(length)+1)]
    return TracedFiber('line', p, arclength(p), 'H')


def test_target_follows_a_straight_fiber_and_beats_tangent_and_chord_on_a_bend():
    straight = np.c_[np.arange(1., 36.), np.zeros(35), np.zeros(35)]
    h = in_crop_heading(straight[None], normalize(np.array([[1., .3, .1]])))[0]
    assert np.degrees(np.arccos(abs(h[0]))) < .5
    u = np.arange(1., 36.)
    arc = np.c_[u, .02*u**2, np.zeros_like(u)]  # bends away by ~25 voxels over the crop span
    h = in_crop_heading(arc[None], normalize(arc[-1:]))[0]
    tangent, chord = np.array([1., 0, 0]), normalize(arc[-1])
    assert lateral_extent(arc, h) < min(lateral_extent(arc, tangent), lateral_extent(arc, chord))


def test_untrained_network_returns_its_prior_and_forward_span_matches_follower_crop():
    cfg = HeadingConfig()
    assert cfg.forward == main_crop_forward() == pytest.approx(35.5)
    out = HeadingNet(cfg)(torch.randn(3, 1, *[cfg.patch.depth, cfg.patch.width, cfg.patch.width]), torch.randn(3, 4*cfg.window))
    torch.testing.assert_close(out, torch.tensor([[0., 0., 1.]]*3))


def test_path_features_mask_missing_history_and_use_the_given_frame():
    frame = frame_from_heading(np.array([1., 0, 0]))
    seed = path_features(np.zeros((1, 3)), np.zeros(3), frame, 8).reshape(8, 4)
    assert not seed.any()
    path = np.c_[np.arange(5.), np.zeros(5), np.zeros(5)]  # 4 voxels behind the head
    features = path_features(path, path[-1], frame, 8).reshape(8, 4)
    np.testing.assert_array_equal(features[:, 3], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(features[:4, 2], -np.arange(1., 5.), atol=1e-6)  # behind the head along -z


def test_predictor_returns_untrained_prior_signed_and_roll_independent(tmp_path):
    vol = volume(tmp_path)
    predictor = HeadingPredictor(HeadingNet(HeadingConfig()))
    priors = [normalize(np.array([.3, .4, .87])), normalize(np.array([-1., .2, .1]))]
    heads = predictor.predict(vol, [np.full(3, 20.), np.full(3, 20.)], priors, [np.zeros((1, 3))+20., np.zeros((1, 3))+20.])
    for h, p in zip(heads, priors):
        np.testing.assert_allclose(h, p, atol=1e-6)
    rolled = prior_frames(priors, np.random.default_rng(0))
    for frame, p in zip(rolled, priors):
        np.testing.assert_allclose(frame[:, 2], p, atol=1e-12)


def test_states_come_from_make_sample_with_seed_priors_not_given_the_answer():
    f, cfg, rng = line_fiber(), HeadingConfig(), np.random.default_rng(3)
    sampling = HeadingSampling(seed_probability=1.)
    states = [heading_state(f, rng, cfg, sampling, sampling.follower_sample_config()) for _ in range(20)]
    for state in states:
        assert state['history'] == 0 and len(state['path']) == 1
        np.testing.assert_array_equal(state['pos'], state['path'][-1])
        assert np.degrees(np.arccos(np.clip(state['prior'] @ state['init'], -1, 1))) <= 75+1e-6
    assert np.mean([state['prior'] @ state['init'] < .9999 for state in states]) > .8  # seed priors are tilted
    sampling = HeadingSampling(seed_probability=0., long_cone_probability=0.)
    long = [s for s in (heading_state(f, rng, cfg, sampling, sampling.follower_sample_config()) for _ in range(60))
            if s['history'] >= 12]
    assert long
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import linear12_heading
    for state in long:
        np.testing.assert_allclose(state['prior'], linear12_heading(state['path'], 0), atol=1e-12)


def test_built_states_express_targets_in_their_patch_frames(tmp_path):
    vol = volume(tmp_path)
    p = np.c_[np.arange(0., 41.), np.full(41, 20.), np.full(41, 20.)]
    f = TracedFiber('ct', p, arclength(p), 'H')
    cfg = HeadingConfig(patch=HeadingConfig().patch.__class__(depth=8, width=8, behind=2, spacing=1.), forward=6.)
    states, patch, path, local = build_states([f], np.ones(1), vol, 4, np.random.default_rng(0), cfg,
                                              HeadingSampling(), roll_rng=np.random.default_rng(1))
    assert patch.shape == (4, 1, 8, 8, 8) and path.shape == (4, 4*cfg.window)
    for state, target in zip(states, local.numpy()):
        np.testing.assert_allclose(state['frame'] @ target, state['target'], atol=1e-5)
        assert abs(state['target'] @ tangent_at(p, f.s, 10.)) > .9  # along the fiber, in either traversal direction


def test_config_resolves_relative_paths_and_schedule_warms_then_decays(tmp_path):
    path = tmp_path/'heading.json'
    path.write_text(json.dumps(dict(name='h', dataset_config='data.json', ct_normalization='norm.json', out_root='out')))
    config = read_config(path, dict(steps=100, warmup=10, lr=1.))
    assert config['dataset_config'] == str(tmp_path/'data.json') and config['run_dir'] == str(tmp_path/'out'/'h')
    rates = [learning_rate(step, config) for step in range(100)]
    assert rates[0] == pytest.approx(.1) and max(rates) == pytest.approx(1.) and rates[-1] < .01


def test_cone_tilts_within_cap():
    rng = np.random.default_rng(0)
    d = np.array([0., 0., 1.])
    tilts = [np.degrees(np.arccos(np.clip(cone(d, rng, 25., 75.) @ d, -1, 1))) for _ in range(500)]
    assert max(tilts) <= 75+1e-6 and 10 < np.median(tilts) < 25


def test_heading_batches_flow_through_the_follower_loader_pipeline(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.data import Source, mixed_heading_states
    vol = volume(tmp_path)
    p = np.c_[np.arange(0., 41.), np.full(41, 20.), np.full(41, 20.)]
    f = TracedFiber('ct', p, arclength(p), 'H')
    cfg = HeadingConfig(patch=HeadingConfig().patch.__class__(depth=8, width=8, behind=2, spacing=1.), forward=6.)
    source = Source('fixture', 'paris4', 1., vol.spec, [f], [f])
    mixed, per_source = mixed_heading_states([source, source], cfg, HeadingSampling(), batch=4)
    batch = next(iter(mixed))
    assert batch['patch'].shape == (4, 1, 8, 8, 8) and batch['target'].shape == (4, 3)
    assert batch['dataset_id'].shape == (4,) and per_source[0].remote_prefetch is None
    torch.testing.assert_close(batch['target'].norm(dim=-1), torch.ones(4))


def test_downsampled_patch_lands_where_the_follower_ct_would(tmp_path, monkeypatch):
    """Level 1 holds 2x block means; on a linear field (trilinear-exact) its raw patch must equal the level-0 patch."""
    from vesuvius.neural_tracing.fiber_follow.data import crop_sampling
    monkeypatch.setattr(crop_sampling, 'normalize_ct', lambda image, record: None)  # z-score would hide offsets/scale
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import model_inputs, patch_volume_spec
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_EPSILON, ZSCORE_METHOD, volume_key
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
    z, y, x = np.indices((42, 42, 42))
    level0 = 2*(x+y+z)
    level1 = level0.reshape(21, 2, 21, 2, 21, 2).mean((1, 3, 5))
    for level, data in (('0', level0), ('1', level1)):
        path = tmp_path/'ct'/level
        path.mkdir(parents=True)
        (path/'.zarray').write_text(json.dumps(dict(shape=list(data.shape), chunks=list(data.shape), dtype='|u1', fill_value=0,
                                                    order='C', filters=None, compressor=None, zarr_format=2)))
        (path/'0.0.0').write_bytes(data.astype(np.uint8).tobytes())
    spec = FiberVolumeSpec(str(tmp_path/'fields'), ct_zarr=str(tmp_path/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct',
                           load_presence=False)
    spec.ct_normalization = dict(method=ZSCORE_METHOD, volume=volume_key(spec), epsilon=ZSCORE_EPSILON)
    fine, coarse = FiberVolume(spec), FiberVolume(patch_volume_spec(spec, 1))
    assert coarse.input_scale == 1 and coarse.spec.ct_level == 1 and coarse.spec.ct_normalization['volume'].endswith('::1')
    patch = HeadingConfig().patch.__class__(depth=6, width=6, behind=2, spacing=1.)
    cfg0, cfg1 = HeadingConfig(patch=patch), HeadingConfig(patch=patch, ct_downsample_levels=1)
    rng = np.random.default_rng(0)
    positions = [10.5+rng.uniform(-1, 1, 3) for _ in range(4)]
    frames = prior_frames([normalize(rng.normal(size=3)) for _ in range(4)], rng)
    paths = [p[None] for p in positions]
    expected, _ = model_inputs(fine, cfg0, positions, frames, paths)
    actual, _ = model_inputs(coarse, cfg1, positions, frames, paths)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    unshifted, _ = model_inputs(coarse, cfg0, positions, frames, paths)  # ignoring the half-voxel offset is detectable
    assert (unshifted-expected).abs().max() > 1e-2


def test_encoder_cells_are_mirror_image_octants():
    """With mirror-symmetric kernels, the opposite final cell's sensitivity is the exact mirror image (v1's
    3-wide stride-2 kernels put the low cell on the patch edge, so this failed)."""
    torch.manual_seed(0)
    net = HeadingNet(HeadingConfig())
    with torch.no_grad():
        for layer in net.features:
            if isinstance(layer, torch.nn.Conv3d):
                layer.weight.copy_(layer.weight.abs()+layer.weight.abs().flip(2, 3, 4))
    x = torch.ones(1, 1, 32, 32, 32, requires_grad=True)
    cells = net.features[:-2](x)
    assert cells.shape[2:] == (2, 2, 2)
    low, = torch.autograd.grad(cells[0, :, 0, 0, 0].sum(), x, retain_graph=True)
    high, = torch.autograd.grad(cells[0, :, 1, 1, 1].sum(), x)
    torch.testing.assert_close(high, low.flip(2, 3, 4))


def test_v1_checkpoint_converts_exactly_and_is_not_loaded_directly(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import load_heading_model
    from vesuvius.neural_tracing.fiber_follow.scripts.convert_heading_v1 import convert, v1_forward
    torch.manual_seed(1)
    cfg = HeadingConfig()
    state = {k: torch.randn(v.shape[:2]+(3, 3, 3))*.2 if v.ndim == 5 else torch.randn(v.shape)*.2
             for k, v in HeadingNet(cfg).state_dict().items()}
    config = dict(cfg.to_dict(), patch=dict(cfg.to_dict()['patch'], gate_direction=False, history_render='points',
                                            history_sigma=1.))  # as v1 runs saved it
    torch.save(dict(architecture='crop_heading_ct_path_v1', config=config, state=state, step=7, optimizer={}),
               tmp_path/'v1.pt')
    with pytest.raises(ValueError, match='convert_heading_v1'):
        load_heading_model(tmp_path/'v1.pt')
    converted = convert(tmp_path/'v1.pt', tmp_path/'v2.pt')
    assert converted['step'] == 7 and 'optimizer' not in converted
    model, _ = load_heading_model(tmp_path/'v2.pt')
    patch, path = torch.randn(5, 1, 32, 32, 32), torch.randn(5, 4*cfg.window)
    with torch.no_grad():
        expected = v1_forward(state, patch, path)
        torch.testing.assert_close(model(patch, path), expected)
    assert (expected-torch.tensor([0., 0., 1.])).abs().max() > 1e-2  # a real, non-prior output was compared
    with pytest.raises(FileExistsError):
        convert(tmp_path/'v1.pt', tmp_path/'v2.pt')


def test_constant_schedule_holds_lr_and_unknown_schedules_are_rejected(tmp_path):
    config = dict(lr=1.5e-4, warmup=0, steps=20000, lr_schedule='constant')
    assert {learning_rate(s, config) for s in (0, 7000, 19999)} == {1.5e-4}
    assert learning_rate(19999, dict(config, lr_schedule='cosine')) < 1e-8
    path = tmp_path/'c.json'
    path.write_text(json.dumps(dict(dataset_config='d.json', lr_schedule='linear')))
    with pytest.raises(ValueError, match='lr_schedule'):
        read_config(path)
