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
from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, frame_from_heading, normalize, tangent_at



def volume(root):
    """Tiny CT-only volume (80^3 native CT, trace grid at half resolution) with per-crop z-score records."""
    from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import ZSCORE_EPSILON, ZSCORE_METHOD, volume_key
    from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec
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
    from vesuvius.neural_tracing.fiber_follow.shared.heading import linear12_heading
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
