import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slab_fixtures import cfg
from test_history_slabs import observation
from test_memory_improvements import path_batch
from vesuvius.neural_tracing.fiber_follow.regression.geometry_resume import expand_geometry_checkpoint
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.path_geometry import (
    COUNT, OFFSETS, path_geometry_inputs, path_geometry_samples)
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    build_parser, checkpoint_config, initialize_training_optimizer, optimizer_update, prepare_training)
from vesuvius.neural_tracing.fiber_follow.regression.datasets import read_dataset_config
from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state


def geometry_batch(c, count=2, length=100.):
    batch = path_batch(c, count)
    path = np.c_[np.zeros(int(length)+1), np.zeros(int(length)+1), np.arange(int(length)+1.)]
    batch['x'].update(path_geometry_inputs([observation(path) for _ in range(count)]))
    return batch


def test_samples_follow_observed_path_without_crop_mask():
    path = np.c_[np.r_[np.zeros(60), np.linspace(0, 6, 41)], np.zeros(101), np.arange(101.)]
    features, valid = path_geometry_samples(observation(path))
    offsets = np.asarray(OFFSETS)
    assert features.shape == (COUNT, 8) and valid.shape == (COUNT,)
    np.testing.assert_array_equal(valid[:-1], offsets <= 100)
    assert valid[-1] and features[-1, 7] == 1 and features[:-1, 7].max() == 0
    # Positions are relative to the head in its frame; samples far outside a crop stay valid.
    far = int(np.flatnonzero(offsets == 96)[0])
    # The head is 6 voxels lateral of the path's straight start: that offset must survive.
    assert valid[far] and features[far, 2] < -90 and abs(features[far, 0]+6) < 1e-6
    np.testing.assert_allclose(features[-1, :3], path[0]-path[-1], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(features[valid, 3:6], axis=-1), 1, atol=1e-6)
    assert features[~valid].max() == 0 == features[~valid].min()


def test_seed_only_state_has_no_geometry():
    _, valid = path_geometry_samples(observation(np.zeros((1, 3))))
    assert not valid.any()


def test_pairs_with_different_older_paths_differ_only_beyond_shared_tail():
    shared = np.c_[np.zeros(13), np.zeros(13), 88+np.arange(13.)]
    a = np.concatenate((np.c_[np.full(88, 4.), np.zeros(88), np.arange(88.)], shared))
    b = np.concatenate((np.c_[np.full(88, -4.), np.zeros(88), np.arange(88.)], shared))
    fa, va = path_geometry_samples(observation(a))
    fb, vb = path_geometry_samples(observation(b))
    near, older = np.asarray(OFFSETS) <= 8, np.asarray(OFFSETS) >= 24
    np.testing.assert_allclose(fa[:-1][near], fb[:-1][near], atol=1e-6)
    assert np.abs(fa[:-1][older & va[:-1], 0]-fb[:-1][older & vb[:-1], 0]).min() > 7


def test_masked_geometry_reproduces_path_token_model_exactly():
    base, extended = cfg(history_path_tokens=True), cfg(history_path_tokens=True, path_geometry_tokens=True)
    assert extended.architecture.endswith('_v17')
    torch.manual_seed(0)
    old = build_model(base).eval()
    new = build_model(extended).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(name.startswith('path_geometry.') for name in missing)
    batch = geometry_batch(extended)
    hidden = dict(batch['x'], path_geometry_valid=torch.zeros_like(batch['x']['path_geometry_valid']))
    with torch.no_grad():
        a = old(batch['x'], batch['hist'], batch['hmask'])
        b = new(hidden, batch['hist'], batch['hmask'])
    for key in ('points', 'confidence'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=1e-6)


def test_geometry_tokens_start_at_zero_and_receive_gradients():
    c = cfg(history_path_tokens=True, path_geometry_tokens=True)
    model = build_model(c)
    batch = geometry_batch(c)
    tokens = model.path_geometry(batch['x']['path_geometry'], batch['x']['path_geometry_valid'])
    assert tokens.eq(0).all()
    out = model(batch['x'], batch['hist'], batch['hmask'])
    (out['points'].square().sum()+out['confidence'].sum()).backward()
    assert model.path_geometry.embed[-1].weight.grad.abs().sum() > 0


def test_compiled_update_trains_geometry_tokens():
    c = cfg(history_path_tokens=True, path_geometry_tokens=True)
    model = prepare_training(build_model(c), 2, backend='eager')
    batch = geometry_batch(c)
    ema = copy.deepcopy(build_model(c))
    before = model.path_geometry.embed[-1].weight.detach().clone()
    metrics = optimizer_update(model, ema, torch.optim.AdamW(model.parameters()), [batch], 1, .0001,
                               device='cpu', compute_metrics=False)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before, model.path_geometry.embed[-1].weight)


def test_cli_flag_is_optional_and_inferred_by_default():
    assert build_parser().parse_args(['--name', 'run', '--path-geometry-tokens']).path_geometry_tokens is True
    assert build_parser().parse_args(['--name', 'run']).path_geometry_tokens is None


def dataset_config(tmp_path, name, weight):
    document = dict(version=1, cache_dir='cache', sources=[dict(
        name='paris4', kind='paris4', weight=weight, fibers='fibers', fiber_zarrs='zarrs', ct='ct.zarr',
        manifest='seeds.json', negative_bank='bank', val_z=[0., 1.],
        validation=dict(strategy='fiber_hash', fraction=.1, seed=7))])
    path = tmp_path/name
    path.write_text(json.dumps(document))
    return path


def source_checkpoint(tmp_path):
    model = build_model(cfg(history_path_tokens=True, recurrent_refinement_steps=1))
    opt = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=1e-4)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    original = dataset_config(tmp_path, 'original.json', .1)
    args = vars(build_parser().parse_args(['--name', 'source', '--fiber-zarrs', 'z', '--fibers', 'f', '--ct', 'c',
        '--manifest', 'm', '--out-root', str(tmp_path), '--lr', '.0001', '--recurrent-refinement-steps', '1',
        '--history-path-tokens', '--dataset-config', str(original)]))
    document, digest = read_dataset_config(original)
    return dict(architecture=model.architecture, model_cfg=model.cfg.to_dict(), model=model.state_dict(),
                ema=copy.deepcopy(model).state_dict(), optimizer=opt.state_dict(), rng=training_rng_state(),
                step=22000, samples_seen=264000, lr_restart_step=0, training_options=args,
                dataset_config=document, dataset_config_sha256=digest)


def test_fork_preserves_weights_resets_optimizer_and_restarts_schedule(tmp_path):
    original = source_checkpoint(tmp_path)
    rng = torch.get_rng_state().clone()
    changed = expand_geometry_checkpoint(original, dataset_config=dataset_config(tmp_path, 'paris50.json', .5),
                                         rest_grad_clip=60.)
    assert torch.equal(rng, torch.get_rng_state())
    assert changed['architecture'].endswith('_v17') and changed['lr_restart_step'] == changed['step'] == 22000
    assert changed['dataset_config']['sources'][0]['weight'] == .5
    assert changed['dataset_config_sha256'] != original['dataset_config_sha256']
    new = build_model(checkpoint_config(changed))
    for section in ('model', 'ema'):
        for name, value in original[section].items():
            torch.testing.assert_close(changed[section][name], value, rtol=0, atol=0)
        new.load_state_dict(changed[section], strict=True)
        assert changed[section]['path_geometry.embed.2.weight'].eq(0).all()
    assert changed['optimizer']['state'] == {}
    options = SimpleNamespace(lr=.0001, reset_optimizer=False)
    opt, done, origin = initialize_training_optimizer(new, copy.deepcopy(new), options, changed)
    assert (done, origin) == (22000, 22000) and not opt.state
    differences = {k for k in original['training_options'] if original['training_options'][k] != changed['training_options'][k]}
    assert differences == {'path_geometry_tokens', 'rest_grad_clip', 'dataset_config'}


def test_fork_rejects_dataset_changes_beyond_weights(tmp_path):
    original = source_checkpoint(tmp_path)
    other = dataset_config(tmp_path, 'other.json', .5)
    document = json.loads(other.read_text()); document['sources'][0]['ct'] = 'different.zarr'
    other.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='Only dataset sampling weights'):
        expand_geometry_checkpoint(original, dataset_config=other, rest_grad_clip=60.)
