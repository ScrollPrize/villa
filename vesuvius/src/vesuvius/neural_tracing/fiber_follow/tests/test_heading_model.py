"""Frame (heading/normal) model: in-crop heading target, normal supervision, and its trainer."""
import json

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingConfig, HeadingNet, load_heading_model
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import normal_loss, normal_target
from vesuvius.neural_tracing.fiber_follow.heading_model.targets import in_crop_heading, lateral_extent
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, normalize


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


def test_targets_axes_reliability_and_sign_invariant_gradients():
    z, y, x = np.indices((17, 18, 18))
    axis = np.array([.8, -.4, .2]); axis /= np.linalg.norm(axis)
    field = 100+20*np.sin((axis[0]*x+axis[1]*y+axis[2]*z)*.5)
    n, w = normal_target(field, [8, 8.5, 8.5], 2.5)
    assert abs(n @ axis) > .995 and w > .9
    n0, w0 = normal_target(np.ones_like(field), [8, 8.5, 8.5], 2.5)
    assert w0 == 0 and not n0.any()
    pred = torch.tensor([[1., 0, 0], [0., 1, 0]], requires_grad=True)
    target = torch.tensor([[.8, .6, 0], [0., 0, 1]])
    weight = torch.tensor([.7, 0.])
    loss = normal_loss(pred, target, weight)
    torch.testing.assert_close(loss, normal_loss(pred, -target, weight))
    loss.backward()
    assert pred.grad[0].abs().sum() > 0 and pred.grad[1].abs().sum() == 0
    empty = normal_loss(pred, target, torch.zeros(2))
    assert empty.item() == 0
    empty.backward()


def test_joint_loader_training_validation_checkpoint_and_resume(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.data import (
        HeadingSampling, Source, mixed_heading_states, build_states, fiber_weights)
    from vesuvius.neural_tracing.fiber_follow.heading_model.train import DEFAULTS, train
    from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY
    vol = volume(tmp_path)
    cfg, sampling = HeadingConfig(width=2, predict_normals=True), HeadingSampling()
    fiber = line_fiber()
    source = Source('fixture', 'paris4', 1., vol.spec, [fiber], [fiber])
    mixed, _ = mixed_heading_states([source], cfg, sampling, batch=2)
    loader = torch.utils.data.DataLoader(mixed, batch_size=None, num_workers=0)
    batch = next(iter(loader))
    assert batch['normal_target'].shape == (2, 3) and torch.isfinite(batch['patch']).all()
    held_out = {'fixture': build_states([fiber], fiber_weights([fiber], cfg.forward+8), vol, 4,
                                       np.random.default_rng(42), cfg, sampling)}
    run = tmp_path/'run'; run.mkdir()
    model = HeadingNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = dict(DEFAULTS, steps=1, batch=2, log_every=1, val_every=1, ckpt_every=1)
    provenance = dict(normal_target_policy=NORMAL_TARGET_POLICY, normal_loss_weight=.25)
    train(config, run, model, opt, 0, float('inf'), loader, held_out, provenance, 'cpu')
    loaded, ckpt = load_heading_model(run/'last.pt')
    assert ckpt['step'] == 1 and (run/'best.pt').exists()
    reports = [json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
    assert 'normal_loss' in reports[0]
    assert all('normal' in b for b in reports[1]['reports']['fixture'].values())
    opt = torch.optim.AdamW(loaded.parameters())
    opt.load_state_dict(ckpt['optimizer'])
    train(dict(config, steps=2), run, loaded, opt, 1, ckpt['best'], loader, held_out, provenance, 'cpu',
          best_normal=ckpt['best_normal'])
    assert load_heading_model(run/'last.pt')[1]['step'] == 2
