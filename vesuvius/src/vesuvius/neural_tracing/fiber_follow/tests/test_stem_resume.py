"""Pretrained initialization, fresh history/optimizer, and resumable continuation."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from slab_fixtures import cfg, slab_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, STEM_ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.stem_resume import expand_stem_checkpoint, fork_run
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    build_parser, checkpoint_config, initialize_training_optimizer, prepare_training, training_prediction,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state, lr_at


def source_checkpoint(tmp_path):
    model = build_model(cfg(encoder='patch4', token_only=True, recurrent_refinement_steps=2))
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for p in model.parameters():
        p.grad = torch.full_like(p, .001)
    optimizer.step()
    options = vars(build_parser().parse_args(['--name', 'source', '--fiber-zarrs', 'z', '--fibers', 'f',
        '--ct', 'c', '--manifest', 'm', '--out-root', str(tmp_path), '--encoder', 'patch4', '--token-only',
        '--batch', '16', '--microbatch', '16', '--remote-prefetch-connections', '48']))
    # Exercise checkpoints written before the stem options existed.
    options.pop('stem_channels'); options.pop('stem_blocks')
    config = model.cfg.to_dict()
    config.pop('stem_channels'); config.pop('stem_blocks')
    return dict(architecture=model.architecture, model_cfg=config, model=model.state_dict(), ema=ema.state_dict(),
        optimizer=optimizer.state_dict(), rng=training_rng_state(), step=100000, samples_seen=1600000,
        lr_restart_step=0, training_options=options)


def test_migration_transfers_weights_resets_history_and_restarts_training(tmp_path):
    torch.manual_seed(42)
    source = source_checkpoint(tmp_path)
    rng = torch.get_rng_state().clone()
    grown = expand_stem_checkpoint(source)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(grown['rng']['torch'], source['rng']['torch'])
    assert source['optimizer']['state'] and not grown['optimizer']['state']
    assert 'stem_channels' not in source['model_cfg']
    assert grown['architecture'] == STEM_ARCHITECTURE
    old = build_model(checkpoint_config(source)).eval()
    new = build_model(checkpoint_config(grown)).eval()
    batch = slab_batch(old.cfg)
    candidates = torch.zeros(2, 2, old.cfg.n_future, 3)
    candidates[..., 2] = old.planes
    for section in ('model', 'ema'):
        for name, value in source[section].items():
            if not name.startswith('history_encoder.'):
                torch.testing.assert_close(grown[section][name], value, rtol=0, atol=0)
        new.load_state_dict(grown[section])
        # Isolate the silent new image branch; history is deliberately reset.
        baseline = dict(source[section])
        baseline.update({k: v for k, v in grown[section].items() if k.startswith('history_encoder.')})
        old.load_state_dict(baseline)
        with torch.no_grad():
            a = old(batch['x'], batch['hist'], batch['hmask'], candidates, confidence_threshold=1.)
            b = new(batch['x'], batch['hist'], batch['hmask'], candidates, confidence_threshold=1.)
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
    key = 'history_encoder.convolution.0.weight'
    assert not torch.equal(grown['model'][key], source['model'][key])
    for name in grown['model']:
        if name.startswith('history_encoder.'):
            torch.testing.assert_close(grown['model'][name], grown['ema'][name], rtol=0, atol=0)
    args = SimpleNamespace(lr=1e-4, reset_optimizer=False)
    opt, done, origin = initialize_training_optimizer(new, copy.deepcopy(new), args, grown)
    assert (done, origin) == (100000, 100000)
    assert not opt.state and opt.param_groups[0]['weight_decay'] == 1e-4
    settings = grown['training_options']
    assert settings['steps'] == 200000 and settings['warmup'] == 5000
    assert settings['lr'] == opt.param_groups[0]['lr'] == 1e-4
    assert settings['fresh_fraction'] == .9 and settings['bank_following_probability'] == 0
    assert settings['decision_fraction'] == .3 and not settings['reset_optimizer']
    assert lr_at(1, settings['lr'], settings['warmup'], settings['steps']-origin) == pytest.approx(2e-8)
    assert lr_at(5000, settings['lr'], settings['warmup'], settings['steps']-origin) == pytest.approx(1e-4, rel=.01)
    # A later ordinary resume retains the new optimizer state and LR position.
    for p in new.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    grown.update(model=new.state_dict(), optimizer=opt.state_dict(), step=100001)
    again, done, origin = initialize_training_optimizer(new, copy.deepcopy(new), args, grown)
    assert (done, origin) == (100001, 100000)
    assert all(state['step'] == 1 for state in again.state.values())


@pytest.mark.parametrize('activation_checkpointing', [False, True])
def test_stem_learns_through_compiled_geometry_and_candidate_scoring(activation_checkpointing):
    torch.manual_seed(21)
    model = build_model(cfg(encoder='patch4', token_only=True, stem_channels=32,
        stem_blocks=2, recurrent_refinement_steps=1, activation_checkpointing=activation_checkpointing))
    prepare_training(model, backend='eager')
    batch = slab_batch(model.cfg)
    candidates = torch.zeros(2, 1, model.cfg.n_future, 3)
    candidates[..., 2] = model.planes
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = training_prediction(model, batch['x'], batch['hist'], batch['hmask'],
                                  candidates, confidence_threshold=1.)
        terms = loss_terms({k: v for k, v in out.items() if not k.startswith('candidate_')}, batch, model.cfg)
        loss = (terms['geometry_per_state']+.5*terms['confidence_per_state']).mean()
        loss = loss+out['candidate_hazard_logits'].square().mean()
        loss.backward()
        projection = model.encoder.stem.projection.weight.grad
        assert torch.isfinite(projection).all() and projection.abs().sum() > 0
        if step:
            for name, p in model.encoder.stem.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                if p.ndim == 5:
                    assert p.grad.abs().sum() > 0, name
        opt.step()


@pytest.mark.parametrize('shape', [(24, 20, 20), (28, 24, 24)])
def test_stem_grid_matches_existing_patch_tokens(shape):
    config = cfg(encoder='patch4', token_only=True, stem_channels=32)
    config = replace(config, fine=replace(config.fine, depth=shape[0], width=shape[1]))
    model = build_model(config)
    stem = model.encoder.stem
    image = torch.randn(1, config.input_channels, *shape)
    actual = stem(image)
    assert actual.shape == model.encoder.patch_projection(image).shape == (1, config.hidden, *config.token_shape)
    assert torch.count_nonzero(actual) == 0
    original = build_model(replace(config, stem_channels=0))
    torch.testing.assert_close(model.encoder.token_xyz, original.encoder.token_xyz, rtol=0, atol=0)


def test_stem_uses_shared_basicblock_d_at_full_half_and_quarter_resolution():
    from vesuvius.models.build.resblocks import BasicBlockD
    config = cfg(encoder='patch4', token_only=True, stem_channels=32, stem_blocks=2)
    stem = build_model(config).encoder.stem
    blocks = [m for m in stem.modules() if isinstance(m, BasicBlockD)]
    assert len(blocks) == 5
    assert [b.output_channels for b in blocks] == [32, 32, 32, 64, 64]
    assert [tuple(b.stride) for b in blocks] == [(1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)]
    image = torch.randn(1, config.input_channels, 24, 20, 20)
    with torch.no_grad():
        full = stem.input(image)
        half = stem.blocks[0](full)
        quarter = stem.blocks[1](half)
    assert full.shape == (1, 32, 24, 20, 20)
    assert half.shape == (1, 32, 12, 10, 10)
    assert quarter.shape == (1, 64, 6, 5, 5)
    assert all(isinstance(b.conv1.norm, torch.nn.InstanceNorm3d) and
               isinstance(b.nonlin2, torch.nn.ReLU) for b in blocks)
    assert isinstance(blocks[1].skip[0], torch.nn.AvgPool3d)
    assert isinstance(blocks[3].skip[0], torch.nn.AvgPool3d)


def test_fork_preserves_fixture_and_source_and_serializes_exact_continuation(tmp_path):
    source = source_checkpoint(tmp_path)
    directory = tmp_path/'source'; directory.mkdir()
    fixture = b'fixed monitor fixture'
    (directory/'monitor_recovery.npz').write_bytes(fixture)
    source['monitor_recovery_sha256'] = hashlib.sha256(fixture).hexdigest()
    path = directory/'ckpt_100000.pt'; torch.save(source, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = fork_run(path, 'stem')
    dest = tmp_path/'stem'
    saved = torch.load(dest/'last.pt', weights_only=False)
    assert saved['step'] == 100000 and saved['training_options']['steps'] == 200000
    assert saved['lr_restart_step'] == 100000 and not saved['optimizer']['state']
    assert (dest/'monitor_recovery.npz').read_bytes() == fixture
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert json.loads((dest/'migration.json').read_text()) == info
    options = vars(build_parser().parse_args(info['command'][4:]))
    assert json.dumps(options, sort_keys=True) == json.dumps(saved['training_options'], sort_keys=True)
    assert (options['batch'], options['microbatch'], options['remote_prefetch_connections']) == (16, 16, 48)
    with pytest.raises(FileExistsError):
        fork_run(path, 'stem')


def test_original_checkpoint_crop_is_rounded_and_old_history_is_discarded(tmp_path):
    source = source_checkpoint(tmp_path)
    source['architecture'] = 'axial_patch4_overlap_tokens_fiber_slabs_v11'
    source['model_cfg']['fine']['width'] = 17
    source['crop'] = copy.deepcopy(source['model_cfg']['fine'])
    source['sample_cfg'] = dict(crop=copy.deepcopy(source['crop']))
    for section in ('model', 'ema'):
        source[section] = {k: v for k, v in source[section].items() if not k.startswith('history_encoder.')}
        source[section]['history_encoder.old_weight'] = torch.full((3,), float('nan'))
    grown = expand_stem_checkpoint(source)
    assert grown['model_cfg']['fine']['width'] == 20
    assert grown['crop'] == grown['sample_cfg']['crop'] == grown['model_cfg']['fine']
    model = build_model(checkpoint_config(grown))
    for section in ('model', 'ema'):
        model.load_state_dict(grown[section], strict=True)
        assert all(torch.isfinite(p).all() for p in model.history_encoder.parameters())


@pytest.mark.parametrize('kwargs', [dict(channels=0), dict(blocks=0), dict(additional_steps=0),
    dict(lr=float('nan')), dict(warmup=100001), dict(fresh_fraction=1.1), dict(bank_following_probability=.8)])
def test_invalid_migrations_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        expand_stem_checkpoint(source_checkpoint(tmp_path), **kwargs)


def test_reject_repeated_expansion_and_non_token_models(tmp_path):
    source = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='without a residual stem'):
        expand_stem_checkpoint(expand_stem_checkpoint(source))
    with pytest.raises(ValueError, match='token-only'):
        cfg(encoder='patch4', stem_channels=32)


def test_fork_updates_fixture_crop_without_changing_frozen_states(tmp_path):
    import numpy as np
    source = source_checkpoint(tmp_path)
    source['architecture'] = 'axial_patch4_overlap_tokens_fiber_slabs_v11'
    source['model_cfg']['fine']['width'] = 17
    directory = tmp_path/'source'; directory.mkdir()
    fixture = directory/'monitor_recovery.npz'
    metadata = dict(provenance=dict(sample_cfg=dict(crop=source['model_cfg']['fine'])))
    geometry = np.arange(24, dtype=np.float32).reshape(8, 3)
    np.savez(fixture, __metadata__=json.dumps(metadata), pos=geometry)
    source['monitor_recovery_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    path = directory/'ckpt_100000.pt'; torch.save(source, path)
    info = fork_run(path, 'stem')
    dest = tmp_path/'stem'
    saved = torch.load(dest/'last.pt', weights_only=False)
    digest = hashlib.sha256((dest/'monitor_recovery.npz').read_bytes()).hexdigest()
    assert digest == saved['monitor_recovery_sha256'] == info['monitor_recovery_sha256']
    assert digest != source['monitor_recovery_sha256']
    with np.load(dest/'monitor_recovery.npz') as archive:
        np.testing.assert_array_equal(archive['pos'], geometry)
        actual = json.loads(str(archive['__metadata__']))
    assert actual['provenance']['sample_cfg']['crop'] == saved['model_cfg']['fine']
    assert not (dest/'monitor_recovery_mmap_v6').exists()
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == source['monitor_recovery_sha256']
