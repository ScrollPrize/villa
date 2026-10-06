"""One-off conversion of checkpoints from before the model cleanup (scripts/convert_checkpoint.py)."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from model_fixtures import config, dataset_document, flow_config
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.data.observations import IdentitySampling
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.models.sequence import SequenceConfig
from vesuvius.neural_tracing.fiber_follow.scripts import convert_checkpoint as convert
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint

DEAD = dict(encoder_ffn=256, decoder_ffn=2048, decoder_layers=6, scorer_layers=4, stem_channels=32, stem_blocks=2,
            stem='residual', memory='none', identity_dim=0, identity_temperature=.1, identity_objective='infonce',
            identity_map=False, identity_feedback=False, path_planes='future', tube_head=False)
SHARES = dict(fresh=.30, live=.45, dagger_pre_excursion=.06, dagger_recoverable=.08, dagger_terminal=.08,
              dagger_premature_stop=.03, dagger_ordinary=0., synthetic_terminal=0., synthetic_identity=0.)
LEGACY = dict(regression='unified', flow='unified_flow', sequence='sequence')


def small(kind):
    if kind == 'regression':
        return config(recurrent_refinement_steps=2)
    if kind == 'flow':
        return flow_config()
    return SequenceConfig(fine=CropSpec(depth=32, width=16, behind=8, spacing=.5), n_future=8, gate_plane=4, hidden=32,
                          layers=2, heads=2, ffn=64, cnn_channels=(4, 8, 8, 16), cnn_blocks=1, n_history=32)


def legacy_checkpoint(kind, shares=SHARES, **dead):
    """A checkpoint as the code before the cleanup wrote it, holding a current small model's tensors."""
    torch.manual_seed(0)
    cfg = small(kind)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    model_cfg = dict(cfg.to_dict(), **dict(DEAD, **dead), model_type=LEGACY[kind])
    if kind != 'regression':
        model_cfg['query_scale'] = None
    sample = dict(asdict(SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future)), crop_targets=False)
    options = dict(model=LEGACY[kind], name='legacy', out_root='/runs', init_weights=None, resume=None, steps=500,
                   batch=4, grad_steps=2, rest_grad_clip=60., history_grad_clip=5., memory='slabs', identity_weight=.1,
                   task_share=['fresh=.30'], n_commit=4, gate='full', workers=3, device='cpu', threads=2,
                   dataset_config='/configs/old.json', fibers='/data/fibers')
    return dict(model_type=LEGACY[kind], data_policy='controlled_spans_v3', model=model.state_dict(),
                ema=model.state_dict(), model_cfg=model_cfg, optimizer=opt.state_dict(), rng={}, step=7,
                lr_restart_step=0, samples_seen=56, sample_cfg=sample, sample_contract=sample,
                history_sampling_revision='live_observed_slabs_learned_roll_v4', seed_manifest_sha256='x',
                negative_bank_provenance={}, bank_role_provenance={}, monitor_recovery_sha256='y',
                dataset_config=dataset_document(), dataset_config_sha256='z',
                task_budget=dict(shares=dict(shares), terminal_fallback_cap=.5, replay_max_age=12000, replay_event_cap=64),
                identity_sampling=dict(asdict(IdentitySampling()), identity_switch_tail=(48., 128.),
                                       memory_augmentation='shared'),
                training_options=options, vol_spec={}, crop=asdict(cfg.fine), n_history=cfg.n_history, n_commit=4)


@pytest.mark.parametrize('kind', ['regression', 'flow', 'sequence'])
def test_legacy_checkpoints_convert_to_the_current_model_and_a_resumable_run(kind):
    ck = legacy_checkpoint(kind, shares=dict(SHARES, fresh=.25, live=.45, synthetic_terminal=.05))
    out, run, notes = convert.convert(ck)
    assert out['model_type'] == kind == run['model']['type']
    assert not set(convert.DROPPED_KEYS) & set(out)
    assert not set(DEAD) & set(out['model_cfg']) and 'query_scale' not in out['model_cfg'] or kind == 'regression'
    assert 'crop_targets' not in out['sample_cfg'] and 'synthetic_identity' not in out['task_budget']['shares']
    assert not {'identity_switch_tail', 'memory_augmentation'} & set(out['identity_sampling'])
    for key in ('model', 'ema', 'optimizer', 'step', 'samples_seen', 'rng'):
        assert out[key] is ck[key]
    convert.verify(out)
    # The run configuration resolves to the checkpoint's model and its training settings.
    # Fitted flow scales stay with the checkpoint; a new run from the configuration fits its own.
    converted = config_from_checkpoint(out)
    assert run_config.model_config(run) == (replace(converted, flow_sigma=()) if kind == 'flow' else converted)
    assert run == run_config.resolve(json.loads(json.dumps(run)))
    t = run['training']
    assert (run['name'], run['out_root'], t['batch'], t['grad_steps'], t['grad_clip'], t['steps']) == ('legacy', '/runs', 4, 2, 60., 500)
    assert (run['runtime']['workers'], run['runtime']['device'], t['n_commit']) == (3, 'cpu', 4)
    assert out['training_options']['grad_clip'] == 60. and out['training_options']['model'] == kind
    assert 'model fields dropped' in notes[0]
    if kind == 'sequence':
        assert (t['task_shares']['fresh'], t['task_shares']['live'], t['task_shares']['synthetic_terminal']) == (pytest.approx(.75), 0., 0.)
        assert any('moved to fresh' in note for note in notes)
    else:
        assert t['task_shares'] == {k: v for k, v in dict(SHARES, fresh=.25, live=.45, synthetic_terminal=.05).items()
                                    if k != 'synthetic_identity'}


@pytest.mark.parametrize('change', [dict(memory='slabs'), dict(memory='decisions'), dict(identity_dim=8),
                                    dict(identity_objective='verify'), dict(path_planes='crop'), dict(tube_head=True)])
def test_models_with_removed_components_are_refused(change):
    with pytest.raises(ValueError, match='no counterpart'):
        convert.convert(legacy_checkpoint('regression', **change))


def test_removed_tasks_targets_and_types_are_refused():
    with pytest.raises(ValueError, match='synthetic_identity'):
        convert.convert(legacy_checkpoint('regression', shares=dict(SHARES, fresh=.2, synthetic_identity=.1)))
    ck = legacy_checkpoint('regression')
    ck['sample_cfg'] = dict(ck['sample_cfg'], crop_targets=True)
    with pytest.raises(ValueError, match='Whole-crop'):
        convert.convert(ck)
    for kind in ('coordinate_regression', 'flow_matching', 'regression'):
        with pytest.raises(ValueError, match='legacy'):
            convert.convert(dict(legacy_checkpoint('regression'), model_type=kind))


def test_main_writes_a_loadable_checkpoint_and_never_overwrites(tmp_path, capsys):
    source, destination, run = tmp_path/'ckpt_000007.pt', tmp_path/'ckpt_000007.converted.pt', tmp_path/'run.converted.json'
    torch.save(legacy_checkpoint('flow'), source)
    convert.main([str(source), str(destination), '--run-config', str(run)])
    assert 'Resume: python train/train.py' in capsys.readouterr().out
    ck = read_checkpoint(destination, 'cpu')
    assert ck['model_type'] == 'flow' and ck['run_config'] == run_config.load(run)
    model = build_model(config_from_checkpoint(ck))
    model.load_state_dict(ck['ema'])
    before = source.read_bytes()
    for target, config_path in ((destination, tmp_path/'other.json'), (tmp_path/'other.pt', run)):
        with pytest.raises(FileExistsError):
            convert.main([str(source), str(target), '--run-config', str(config_path)])
    assert source.read_bytes() == before and not (tmp_path/'other.pt').exists()
