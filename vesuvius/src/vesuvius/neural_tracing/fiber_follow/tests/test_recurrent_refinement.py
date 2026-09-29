"""Shared-decoder refinement, differentiable caches and checkpoint migration."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from test_trajectory_memory import cfg, memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, PathDecoderLayer
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    checkpoint_config, optimizer_update, build_parser, compile_training_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.upgrade_refinement import upgrade_checkpoint, options_argv
from vesuvius.neural_tracing.fiber_follow.shared.runloop import resume_training, training_rng_state


def test_cached_decoder_matches_recomputed_forward_and_gradients():
    torch.manual_seed(41)
    layer = PathDecoderLayer(16, 2, 32, dropout=0., batch_first=True, norm_first=True)
    other = copy.deepcopy(layer)
    memory = torch.randn(2, 23, 16, requires_grad=True)
    copied = memory.detach().clone().requires_grad_()
    x = torch.randn(2, 4, 16)
    padding = torch.zeros(2, 23, dtype=torch.bool)
    padding[0, 17:] = True
    kv = other.project_memory(copied)
    a = layer(x, memory, memory_key_padding_mask=padding)
    a = layer(a, memory, memory_key_padding_mask=padding)
    b = other.forward_cached(x, kv, padding)
    b = other.forward_cached(b, kv, padding)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(memory.grad, copied.grad, atol=1e-7, rtol=1e-4)
    for p, q in zip(layer.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_refinement_starts_neutral_encodes_writes_and_projects_once(monkeypatch):
    torch.manual_seed(42)
    old = build_model(cfg()).eval()
    model = build_model(replace(old.cfg, recurrent_refinement_steps=2)).eval()
    model.load_state_dict(old.state_dict(), strict=False)
    b = memory_batch(model.cfg)
    calls = dict(encoder=0, writer=0, projection=0, decoder=0)
    for obj, method, key in ((model.encoder, 'forward', 'encoder'),
                             (model.recurrent_memory, 'observe_features', 'writer'),
                             (model.decoder.layers[0], 'project_memory', 'projection'),
                             (model.decoder.layers[0], 'forward_cached', 'decoder')):
        original = getattr(obj, method)
        def count(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(obj, method, count)
    expected = old(b['x'], b['hist'], b['hmask'])
    out = model(b['x'], b['hist'], b['hmask'])
    assert calls == dict(encoder=1, writer=1, projection=1, decoder=3)
    torch.testing.assert_close(out['points'], expected['points'], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(out['points'], out['initial_points'], atol=1e-6, rtol=1e-5)
    assert out['refinement_points'].shape[1] == 3
    for key in state_from(model, out):
        torch.testing.assert_close(out['memory_'+key], expected['memory_'+key], rtol=0, atol=0)


def test_refinement_geometry_reaches_proposal_stage_fusion_and_old_observations():
    torch.manual_seed(43)
    model = build_model(cfg(recurrent_refinement_steps=1))
    torch.nn.init.normal_(model.refinement_delta.weight, std=.03)
    earlier = memory_batch(model.cfg, step=0)
    b = memory_batch(model.cfg, step=1)
    earlier['x']['fine'].requires_grad_()
    old = model(earlier['x'], earlier['hist'], earlier['hmask'])
    out = model(b['x'], b['hist'], b['hmask'], memory=state_from(model, old))
    gradient, = torch.autograd.grad(out['points'].square().mean(), out['initial_points'], retain_graph=True)
    assert gradient.abs().sum() > 0
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert earlier['x']['fine'].grad.abs().sum() > 0
    for name, p in model.named_parameters():
        if name.startswith(('refinement_', 'coordinates.')):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bounds_and_masked_geometry():
    c = cfg(recurrent_refinement_steps=2, max_recovery_distance=2.)
    model = build_model(c)
    with torch.no_grad():
        model.refinement_delta.bias.fill_(100.)
    b = memory_batch(c)
    b['offtrack'][0] = 1
    b['dense_mask'][1].zero_()
    out = model(b['x'], b['hist'], b['hmask'])
    curves = out['refinement_points']
    assert curves[..., :2].diff(dim=1).norm(dim=-1).max() <= c.recurrent_refinement_limit+1e-5
    assert curves[:, :, 0].norm(dim=-1).max() <= c.max_recovery_distance+1e-5
    assert curves[..., :2].abs().max() <= c.lateral_limit
    torch.testing.assert_close(curves[..., 2], model.planes.expand_as(curves[..., 2]))
    loss = loss_terms(out, b, c)['geometry_per_state']
    assert loss.eq(0).all()
    loss.sum().backward()
    assert model.refinement_delta.weight.grad.eq(0).all()


def test_upgrade_preserves_model_ema_optimizer_rng_and_next_update():
    torch.manual_seed(44)
    model = build_model(cfg())
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer_update(model, ema, opt, [training_chunk(model.cfg)], 17, .001, device='cpu', compute_metrics=False)
    ck = dict(model_cfg=model.cfg.to_dict(), architecture=model.architecture,
              model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
              rng=training_rng_state(), step=17, samples_seen=123, replay_seen=11)
    upgraded = upgrade_checkpoint(ck)
    new = build_model(checkpoint_config(upgraded))
    new_ema = copy.deepcopy(new)
    new_opt = torch.optim.AdamW(new.parameters(), lr=.001)
    assert resume_training(upgraded, new, new_ema, new_opt) == (17, 11)
    assert upgraded['samples_seen'] == 123
    torch.testing.assert_close(torch.get_rng_state(), ck['rng']['torch'], rtol=0, atol=0)
    for before, after in ((model, new), (ema, new_ema)):
        for name, value in before.state_dict().items():
            torch.testing.assert_close(value, after.state_dict()[name], rtol=0, atol=0)
    old_params = dict(model.named_parameters())
    for name, p in new.named_parameters():
        if name.startswith('refinement_'):
            assert not new_opt.state.get(p)
        else:
            for key, value in opt.state[old_params[name]].items():
                torch.testing.assert_close(new_opt.state[p][key], value, rtol=0, atol=0)
    metrics = optimizer_update(new, new_ema, new_opt, [training_chunk(new.cfg)], 18, .001,
                               device='cpu', compute_metrics=False)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert new.refinement_delta.weight.abs().sum() > 0
    options = vars(build_parser().parse_args(['--name', 'test', '--fiber-zarrs', '/a', '--fibers', '/b',
        '--ct', '/c', '--manifest', '/d', '--memory-version', '4', '--no-correction',
        '--recurrent-refinement-steps', '1']))
    assert json.dumps(vars(build_parser().parse_args(options_argv(options))), sort_keys=True) == json.dumps(options, sort_keys=True)


def test_refinement_fullgraph_capture():
    model = build_model(cfg(recurrent_refinement_steps=1))
    b = memory_batch(model.cfg)
    wrapped = torch.compile(model, backend='eager', fullgraph=True)
    out = wrapped(b['x'], b['hist'], b['hmask'])
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert torch.isfinite(model.refinement_delta.weight.grad).all()


def test_run_migration_uses_runtime_options_and_preserves_fixture_replay(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.regression.upgrade_refinement import main
    model = build_model(cfg())
    source = tmp_path/'old'
    source.mkdir()
    path = source/'ckpt_000017.pt'
    options = vars(build_parser().parse_args(['--name', 'old', '--fiber-zarrs', '/a', '--fibers', '/b',
        '--ct', '/c', '--manifest', '/d', '--out-root', str(tmp_path), '--batch', '8',
        '--microbatch', '4', '--memory-version', '4', '--no-correction', '--resume', str(path)]))
    options.pop('recurrent_refinement_steps')
    options.pop('recurrent_refinement_limit')
    options = json.loads(json.dumps(options))
    fixture = source/'monitor_recovery.npz'
    fixture.write_bytes(b'unchanged monitor fixture')
    ck = dict(model_cfg=model.cfg.to_dict(), architecture=model.architecture,
              model=model.state_dict(), ema=model.state_dict(), step=17,
              optimizer=torch.optim.AdamW(model.parameters()).state_dict(), rng=training_rng_state(),
              monitor_recovery_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
              training_options=options, feature_sampling_revision=3)
    torch.save(ck, path)
    # Deliberately stale launch config: the runtime event is authoritative.
    (source/'config.json').write_text(json.dumps(dict(options, batch=16)))
    (source/'log.jsonl').write_text(json.dumps(dict(event='resume_configuration', training_options=options))+'\n')
    replay = tmp_path/'replay'
    replay.mkdir()
    (source/'dagger').mkdir()
    (source/'dagger'/'replay.json').write_text(json.dumps([str(replay)]))
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(runtime_options=options, model_cfg=ck['model_cfg'], cli_event_differences={})))
    monkeypatch.setattr('sys.argv', ['upgrade_refinement', '--checkpoint', str(path),
        '--runtime-audit', str(audit), '--name', 'new'])
    main()
    destination = tmp_path/'new'
    configuration = json.loads((destination/'config.json').read_text())
    assert configuration['batch'] == 8 and configuration['recurrent_refinement_steps'] == 1
    assert (destination/'monitor_recovery.npz').read_bytes() == fixture.read_bytes()
    assert json.loads((destination/'dagger'/'replay.json').read_text()) == [str(replay)]
    upgraded = torch.load(destination/'last.pt', weights_only=False)
    assert upgraded['step'] == 17 and upgraded['feature_sampling_revision'] == 3
    argv = json.loads((destination/'resume_argv.json').read_text())
    assert vars(build_parser().parse_args(argv)) == upgraded['training_options']
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compiled_refinement_bf16_two_decision_gradients():
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    torch.manual_seed(45)
    c = cfg(recurrent_refinement_steps=1)
    eager = build_model(c).cuda()
    torch.nn.init.normal_(eager.refinement_delta.weight, std=.03)
    compiled = copy.deepcopy(eager)
    chunk = [move_batch(b, 'cuda') for b in training_chunk(c)['feature_sequence']]
    losses = []
    for model in (eager, compile_training_model(compiled)):
        state = None
        loss = 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for b in chunk:
                candidates = b['hist'].new_zeros(2, 2, c.n_future, 3)
                candidates[..., 2] = eager.planes
                candidates[:, 1, :, 0] = 2.
                b['candidate_mask'] = torch.ones_like(candidates[..., 0], dtype=torch.bool)
                b['candidate_labels'] = torch.zeros_like(candidates[..., 0])
                b['candidate_labels'][:, 0] = 1.
                out = model(b['x'], b['hist'], b['hmask'], memory=state,
                            queries=b['identity_points'], candidates=candidates)
                state = state_from(eager, out)
                terms = loss_terms(out, b, c)
                loss = (loss+terms['geometry_per_state'].mean()+terms['confidence_per_state'].mean()
                        +out['candidate_confidence_logits'].square().mean())
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(*losses, atol=.005, rtol=.02)
    for prefix in ('encoder.', 'decoder.', 'refinement_', 'recurrent_memory.'):
        grads = [torch.cat([p.grad.flatten() for name, p in model.named_parameters()
                           if name.startswith(prefix) and p.grad is not None]) for model in (eager, compiled)]
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.nn.functional.cosine_similarity(*grads, dim=0).item() > .99
        assert (grads[1].norm()/grads[0].norm()).item() == pytest.approx(1., rel=.05)
