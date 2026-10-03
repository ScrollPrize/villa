"""Family conditioning, fiber correction, frame geometry and resumable conversion."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.frames import (
    family_ids, smoothed_tangent, fiber_corrected_normal, orthonormal_frame, roll_supervision, frame_angles)
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingConfig, HeadingNet, save_heading_model, load_heading_model
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import normal_target_policy
from vesuvius.neural_tracing.fiber_follow.tests.test_heading_model import volume, line_fiber


def test_tangent_and_projection_geometry():
    f = line_fiber()
    for at in (0., 1., 50., 199., 200.):
        np.testing.assert_allclose(smoothed_tangent(f.points, f.s, at), [1., 0, 0], atol=1e-12)
    n, w = fiber_corrected_normal(np.array([.6, .8, 0]), np.array([1., 0, 0]), .5)
    np.testing.assert_allclose(n, [0., 1, 0]); assert w == pytest.approx(.32)
    n2, w2 = fiber_corrected_normal(-np.array([.6, .8, 0]), -np.array([1., 0, 0]), .5)
    np.testing.assert_allclose(n2, -n); assert w2 == w
    n, w = fiber_corrected_normal(np.array([1., 0, 0]), np.array([1., 0, 0]), 1.)
    assert w == 0 and not n.any()


def test_frames_are_right_handed_and_continuous_with_degenerate_inputs():
    torch.manual_seed(42)
    h, n = torch.randn(16, 3), torch.randn(16, 3)
    h[0] = 0; n[1] = h[1]; n[2] = 0
    frame = orthonormal_frame(h, n, torch.zeros_like(h))
    torch.testing.assert_close(frame.transpose(-1, -2) @ frame, torch.eye(3).expand(16, 3, 3), atol=2e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(frame), torch.ones(16), atol=2e-6, rtol=0)
    continuity = orthonormal_frame(h, -n, frame[..., 0])
    torch.testing.assert_close(continuity, frame, atol=2e-6, rtol=0)
    torch.testing.assert_close(frame_angles(frame, frame*frame.new_tensor([-1., -1., 1.])),
                               torch.zeros(16), atol=.06, rtol=0)
    n = torch.randn(16, 3, requires_grad=True)
    heading = torch.nn.functional.normalize(h+torch.tensor([0., 0., 1.]), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    loss, _, _ = roll_supervision(n, target, heading, torch.ones(16))
    loss.backward(); assert torch.isfinite(n.grad).all() and n.grad.abs().sum() > 0
    flipped, _, _ = roll_supervision(n, -target, heading, torch.ones(16))
    torch.testing.assert_close(loss, flipped)
    empty, _, _ = roll_supervision(n, target, heading, torch.zeros(16))
    assert empty.item() == 0


def test_conversion_preserves_predictions_optimizer_and_learns_family(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.scripts.convert_heading_frame import convert
    torch.manual_seed(4)
    cfg = HeadingConfig(width=2, predict_normals=True)
    old = HeadingNet(cfg)
    opt = torch.optim.AdamW(old.parameters(), lr=1e-4)
    patch, path = torch.randn(4, 1, 32, 32, 32), torch.randn(4, 128)
    out = old.forward_outputs(patch, path)
    (out['normal'][:, 0].sum()+out['heading'][:, 0].sum()).backward(); opt.step()
    save_heading_model(tmp_path/'source.pt', old, step=20000, optimizer=opt.state_dict(),
                       normal_target_policy=normal_target_policy(cfg), normal_loss_weight=.25)
    model, extra = convert(tmp_path/'source.pt', tmp_path/'frame.pt')
    loaded, ckpt = load_heading_model(tmp_path/'frame.pt')
    assert ckpt['step'] == 20000 and ckpt['normal_target_policy']['tangent_half_span_trace'] == 6
    for key, state in opt.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, ckpt['optimizer']['state'][key][name], rtol=0, atol=0)
    for f in (0, 1):
        for key in ('heading', 'normal'):
            torch.testing.assert_close(old.forward_outputs(patch, path)[key],
                loaded.forward_outputs(patch, path, torch.full((4,), f))[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='family IDs'):
        loaded(patch, path)
    with pytest.raises(ValueError):
        family_ids(['unknown'])
    opt2 = torch.optim.AdamW(loaded.parameters()); opt2.load_state_dict(ckpt['optimizer'])
    outputs = loaded.forward_outputs(patch, path, family_ids(['H', 'V', 'H', 'V']))
    outputs['normal'][:, 0].sum().backward(); opt2.step()
    assert loaded.family_embedding.weight.abs().sum() > 0
    assert not torch.equal(loaded.family_embedding.weight[0], loaded.family_embedding.weight[1])
    with pytest.raises(FileExistsError):
        convert(tmp_path/'source.pt', tmp_path/'frame.pt')


def test_family_batches_correct_targets_and_train_frames(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.heading_model.data import (
        Source, HeadingSampling, mixed_heading_states, build_states, fiber_weights)
    from vesuvius.neural_tracing.fiber_follow.heading_model.train import DEFAULTS, train
    vol = volume(tmp_path)
    cfg = HeadingConfig(width=2, predict_normals=True, predict_frames=True)
    fibers = [line_fiber(), replace(line_fiber(), name='vertical', tag='V')]
    sampling = HeadingSampling()
    source = Source('fixture', 'paris4', 1., vol.spec, fibers, fibers)
    mixed, _ = mixed_heading_states([source], cfg, sampling, batch=4)
    loader = torch.utils.data.DataLoader(mixed, batch_size=None, num_workers=0)
    batch = next(iter(loader)); assert batch['family'].shape == (4,)
    assert batch['family'].dtype == torch.long
    held = build_states(fibers, fiber_weights(fibers, cfg.forward+8), vol, 8, np.random.default_rng(44), cfg, sampling)
    for state in held[0]:
        if state['normal_weight']:
            assert abs(state['normal_target'] @ (state['fiber_tangent'] @ state['frame'])) < 1e-6
    assert {s['family'] for s in held[0]} == {0, 1}
    model = HeadingNet(cfg); opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    run = tmp_path/'run'; run.mkdir()
    config = dict(DEFAULTS, steps=1, batch=4, log_every=1, val_every=1, ckpt_every=1)
    provenance = dict(normal_target_policy=normal_target_policy(cfg), normal_loss_weight=.25, roll_loss_weight=.25)
    train(config, run, model, opt, 0, float('inf'), loader, {'fixture': held}, provenance, 'cpu')
    ckpt = load_heading_model(run/'last.pt')[1]
    assert ckpt['step'] == 1
    rows = [json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()]
    assert 'roll_loss' in rows[0]
    assert all('frame' in b for b in rows[1]['reports']['fixture'].values())
    # Continue the saved optimizer/model and verify the new branch participates after resume.
    resumed, ckpt = load_heading_model(run/'last.pt')
    opt2 = torch.optim.AdamW(resumed.parameters()); opt2.load_state_dict(ckpt['optimizer'])
    train(dict(config, steps=2), run, resumed, opt2, ckpt['step'], ckpt['best'], loader, {'fixture': held}, provenance,
          'cpu', best_normal=ckpt['best_normal'], best_frame=ckpt['best_frame'])
    assert load_heading_model(run/'last.pt')[1]['step'] == 2


def test_predict_frames_uses_family_and_preserves_previous_roll_sign(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
    from vesuvius.neural_tracing.fiber_follow.heading_model import normals
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not compute tensor/tangent labels')
    monkeypatch.setattr(normals, 'sample_training_crops', forbidden)
    vol = volume(tmp_path)
    model = HeadingNet(HeadingConfig(width=2, predict_normals=True, predict_frames=True))
    with torch.no_grad():
        model.normal_head[-1].weight.zero_(); model.normal_head[-1].bias.copy_(torch.tensor([1., 0, 0]))
    predictor = HeadingPredictor(model)
    pos = [np.full(3, 20.)]*2
    priors = [np.array([1., 0, 0]), np.array([0., 1, 0])]
    frames = predictor.predict_frames(vol, pos, priors, [p[None] for p in pos], ['H', 'V'])
    previous = [f*np.array([-1., -1., 1.]) for f in frames]
    continuous = predictor.predict_frames(vol, pos, priors, [p[None] for p in pos], ['H', 'V'], previous=previous)
    np.testing.assert_allclose(continuous, previous, atol=1e-6)
    for f, h in zip(frames, priors):
        np.testing.assert_allclose(f[:, 2], h, atol=1e-6)
        np.testing.assert_allclose(f.T @ f, np.eye(3), atol=1e-6)
        assert np.linalg.det(f) == pytest.approx(1.)
