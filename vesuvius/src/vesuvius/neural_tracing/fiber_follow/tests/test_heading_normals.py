"""Shared CT sampling, unsigned normal supervision, and checkpoint compatibility."""
import json

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.model import (
    HeadingConfig, HeadingNet, model_inputs, prior_frames, save_heading_model, load_heading_model)
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import normal_loss, normal_target, training_crop
from vesuvius.neural_tracing.fiber_follow.tests.test_heading_model import volume, line_fiber


def test_shared_sample_preserves_input_and_reads_once(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.data import crop_sampling
    vol = volume(tmp_path)
    cfg = HeadingConfig(predict_normals=True)
    rng = np.random.default_rng(42)
    positions = [rng.uniform(15, 25, 3) for _ in range(4)]
    frames = prior_frames([v/np.linalg.norm(v) for v in rng.normal(size=(4, 3))], rng)
    paths = [p[None] for p in positions]
    expected, path = model_inputs(vol, cfg, positions, frames, paths)
    calls = []
    original = crop_sampling.read_tight_blocks
    def read(items, vol, crop, pool, **kw):
        calls.append(crop)
        return original(items, vol, crop, pool, **kw)
    monkeypatch.setattr(crop_sampling, 'read_tight_blocks', read)
    actual, actual_path, normals, weights = model_inputs(vol, cfg, positions, frames, paths, normal_targets=True)
    assert len(calls) == 1
    assert (calls[0].depth, calls[0].width, calls[0].behind) == (44, 34, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_path, path, rtol=0, atol=0)
    assert normals.shape == (4, 3) and weights.shape == (4,)


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


def test_conversion_preserves_trained_heading_and_exposes_normals(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.scripts.convert_heading_normals import convert
    torch.manual_seed(1)
    old = HeadingNet(HeadingConfig(width=2))
    torch.nn.init.normal_(old.head[-1].weight, std=.1)
    save_heading_model(tmp_path/'old.pt', old, step=123, optimizer={})
    converted = convert(tmp_path/'old.pt', tmp_path/'new.pt')
    loaded, checkpoint = load_heading_model(tmp_path/'new.pt')
    assert 'optimizer' not in checkpoint and 'step' not in checkpoint
    assert checkpoint['initialized_from']['step'] == 123
    patch, path = torch.randn(3, 1, 32, 32, 32), torch.randn(3, 128)
    torch.testing.assert_close(old(patch, path), converted(patch, path), rtol=0, atol=0)
    out = loaded.forward_outputs(patch, path)
    torch.testing.assert_close(out['normal'].norm(dim=-1), torch.ones(3))
    loss = normal_loss(out['normal'], torch.tensor([[1., 0, 0]]*3), torch.ones(3))
    loss.backward()
    assert loaded.normal_head[-1].weight.grad.abs().sum() > 0
    assert loaded.features[0].weight.grad.abs().sum() > 0
    with pytest.raises(FileExistsError):
        convert(tmp_path/'old.pt', tmp_path/'new.pt')


def test_normal_inference_rotates_outputs_without_generating_labels(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
    from vesuvius.neural_tracing.fiber_follow.heading_model import normals
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not generate normal labels')
    monkeypatch.setattr(normals, 'sample_training_crops', forbidden)
    vol = volume(tmp_path)
    model = HeadingNet(HeadingConfig(width=2, predict_normals=True))
    with torch.no_grad():
        model.normal_head[-1].weight.zero_()
        model.normal_head[-1].bias.copy_(torch.tensor([1., 0, 0]))
    predictor = HeadingPredictor(model)
    priors = [np.array([1., 0, 0]), np.array([0., 1, 0])]
    positions = [np.full(3, 20.)]*2
    headings, axes = predictor.predict_with_normals(vol, positions, priors, [p[None] for p in positions])
    for h, n, p, f in zip(headings, axes, priors, prior_frames(priors)):
        np.testing.assert_allclose(h, p, atol=1e-6)
        np.testing.assert_allclose(n, f[:, 0], atol=1e-6)


def test_joint_loader_training_validation_checkpoint_and_resume(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.data import (
        HeadingSampling, Source, mixed_heading_states, build_states, fiber_weights, HeadingBatchBuilder)
    from vesuvius.neural_tracing.fiber_follow.heading_model.train import DEFAULTS, train
    from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY
    from vesuvius.neural_tracing.fiber_follow.data.data import tight_block
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
    state = held_out['fixture'][0][0]
    bounds = list(HeadingBatchBuilder(cfg).prefetch_bounds(state, vol))
    expected = tight_block(state['pos'], state['frame'], training_crop(cfg, vol)[0], vol.input_scale)
    assert len(bounds) == 1
    for a, b in zip(bounds[0], expected):
        np.testing.assert_array_equal(a, b)
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


def test_default_target_matches_the_selected_sweep_setting():
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor
    from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY
    assert NORMAL_TARGET_POLICY['context_radius_ct'] == 38.
    raw = np.random.default_rng(3).uniform(0, 255, (33, 34, 34))
    center = [16, 16.5, 16.5]
    normal, weight = normal_target(raw, center, 2.5)
    values, vectors = np.linalg.eigh(ct_structure_tensor(raw, center, sample_spacing=2.5,
                                                        derivative_sigma=2., integration_sigma=8.))
    assert abs(normal @ vectors[:, -1]) > 1-1e-6
    assert weight == pytest.approx((values[-1]-values[-2])/values[-1])


def test_resume_upgrades_only_known_policy_and_preserves_current_metric(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY_1_4, NORMAL_TARGET_POLICY
    from vesuvius.neural_tracing.fiber_follow.heading_model.train import normal_resume_state, archive_normal_best
    checkpoint = dict(step=8000, normal_target_policy=NORMAL_TARGET_POLICY_1_4,
                      normal_loss_weight=.25, best_normal=.01)
    score, change = normal_resume_state(checkpoint, .25)
    assert score == float('inf') and change['step'] == 8000 and change['new'] == NORMAL_TARGET_POLICY
    current = dict(checkpoint, normal_target_policy=NORMAL_TARGET_POLICY)
    assert normal_resume_state(current, .25) == (.01, None)
    with pytest.raises(ValueError, match='loss weight'):
        normal_resume_state(checkpoint, .5)
    with pytest.raises(ValueError, match='Unsupported normal target policy'):
        normal_resume_state(dict(checkpoint, normal_target_policy=None), .25)
    torch.save(checkpoint, tmp_path/'best_normal.pt')
    archive = archive_normal_best(tmp_path, change)
    assert archive == 'best_normal_sigma1_4_before_step_008000.pt'
    assert not (tmp_path/'best_normal.pt').exists()
    assert torch.load(tmp_path/archive, weights_only=False) == checkpoint
    assert archive_normal_best(tmp_path, change) is None
