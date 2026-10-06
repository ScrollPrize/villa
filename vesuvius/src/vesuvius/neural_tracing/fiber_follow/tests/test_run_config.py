"""The run configuration (train/run_config.py): per-model defaults, strict keys, paths and the trainer's settings."""
import json
import re
from pathlib import Path

import pytest

from model_fixtures import dataset_document, run_document
from vesuvius.neural_tracing.fiber_follow.data.data import TASKS
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train import run_config

ROOT = Path(__file__).resolve().parents[1]


def resolved(model='regression', **options):
    return run_config.resolve(run_document(model, **options))


def test_regression_defaults_are_its_checkpoint_settings():
    run = resolved()
    cfg, t = run_config.model_config(run), run['training']
    assert cfg.model_type == 'regression' and cfg.fine == CropSpec(depth=144, width=104, behind=72, spacing=.5)
    assert (cfg.n_future, cfg.gate_plane, cfg.recurrent_refinement_steps, cfg.query_scale) == (35, 16, 3, None)
    assert (cfg.hidden, cfg.layers, cfg.heads, cfg.ffn, cfg.cnn_channels, cfg.cnn_blocks) == (256, 12, 8, 2048, (32, 64, 128), (1, 2, 2))
    assert (t['steps'], t['batch'], t['grad_steps'], t['warmup'], t['grad_clip']) == (30000, 16, 2, 2000, 60.)
    assert (t['n_commit'], t['gate'], t['refinement_loss'], t['terminal_fallback_cap']) == (8, 'full', 'all', 0.)
    assert (t['dagger_every'], t['dagger_fibers'], t['afv_dagger_fibers'], t['dagger_batch'], t['dagger_forward_chunk'],
            t['dagger_trace_len']) == (1000, 128, 384, 24, 8, 6000.)
    assert t['task_shares'] == dict(fresh=.30, live=.45, dagger_pre_excursion=.06, dagger_recoverable=.08,
                                    dagger_terminal=.08, dagger_premature_stop=.03, dagger_ordinary=0., synthetic_terminal=0.)
    assert (t['lr'], t['ema_decay'], t['tolerance'], t['trace_confidence'], t['afv_length_power']) == (1e-4, .999, 2., .5, 3.)
    assert t['live_continuation_steps'] == [32, 256] and t['synthetic_tail'] == [4., 16.]


def test_flow_defaults_are_its_checkpoint_settings():
    run = resolved('flow')
    cfg, t = run_config.model_config(run), run['training']
    assert cfg.model_type == 'flow' and cfg.fine == CropSpec(depth=144, width=104, behind=72, spacing=.5)
    assert (cfg.n_future, cfg.gate_plane, cfg.flow_steps, cfg.flow_draws, cfg.flow_samples) == (16, None, 4, 64, 4)
    assert (cfg.flow_time_conditioning, cfg.flow_sigma_floor, cfg.flow_unknown_planes, cfg.flow_loss, cfg.flow_huber_c) == (
        'adaln_zero', 3., 'own_path', 'pseudo_huber', 1.)
    assert (cfg.flow_selection, cfg.flow_zero_start, cfg.flow_sample_threshold, cfg.flow_geometry_weight) == ('retry', True, 0., 0.)
    assert cfg.flow_sigma == ()  # fitted at the start of training
    assert (t['steps'], t['batch'], t['grad_steps'], t['warmup'], t['grad_clip']) == (40000, 48, 1, 1000, 10.)
    assert (t['n_commit'], t['gate'], t['refinement_loss'], t['dagger_every'], t['terminal_fallback_cap']) == (16, 'prefix', 'policy', 500, .5)
    assert t['task_shares']['fresh'] == .27 and t['task_shares']['synthetic_terminal'] == .03
    assert t['synthetic_tail'] == [4., 128.] and t['flow_calibration_states'] == 2048


def test_sequence_defaults_are_its_checkpoint_settings_without_live_or_synthetic_tasks():
    run = resolved('sequence')
    cfg, t = run_config.model_config(run), run['training']
    assert cfg.model_type == 'sequence' and cfg.fine == CropSpec(depth=80, width=64, behind=16, spacing=.5)
    assert (cfg.n_future, cfg.gate_plane, cfg.cnn_channels, cfg.cnn_blocks, cfg.history_limit) == (31, 16, (16, 32, 64, 128), 2, 512)
    assert (cfg.hidden, cfg.layers, cfg.heads, cfg.ffn) == (256, 12, 8, 2048)
    assert (t['batch'], t['grad_steps'], t['n_commit'], t['episode_steps'], t['episode_supervised'], t['episode_commit']) == (20, 1, 12, 64, 16, 12)
    assert (t['dagger_every'], t['dagger_fibers'], t['afv_dagger_fibers'], t['dagger_trace_len'], t['batch_diag_every']) == (0, 64, None, 768., 0)
    assert t['task_shares']['fresh'] == pytest.approx(.70)
    assert t['task_shares']['live'] == t['task_shares']['synthetic_terminal'] == 0.
    assert t['live_continuation_steps'] == [12, 32]
    with pytest.raises(ValueError, match='live and synthetic'):
        resolved('sequence', training=dict(task_shares=dict(fresh=.75, live=.25)))


@pytest.mark.parametrize('section', ['top-level', 'model', 'training', 'runtime', 'task share'])
def test_unknown_keys_are_errors(section):
    document = run_document()
    if section == 'top-level':
        document['optimizer'] = {}
    elif section == 'task share':
        document['training'] = dict(task_shares=dict(fresh=1., synthetic_identity=0.))
    else:
        document[section] = dict(document.get(section, {}), not_a_setting=1)
    with pytest.raises(ValueError, match=f'Unknown {section}'):
        run_config.resolve(document)


def test_a_name_dataset_and_known_model_type_are_required():
    for change in (dict(name=''), dict(model=dict(type='unified'))):
        with pytest.raises(ValueError):
            run_config.resolve(dict(run_document(), **change))
    document = run_document()
    del document['dataset']
    with pytest.raises(ValueError, match='dataset'):
        run_config.resolve(document)


def test_task_shares_replace_the_default_and_absent_tasks_get_zero():
    shares = resolved(training=dict(task_shares=dict(fresh=.6, dagger_terminal=.4)))['training']['task_shares']
    assert shares == {task: dict(fresh=.6, dagger_terminal=.4).get(task, 0.) for task in TASKS}


def test_relative_paths_resolve_against_the_configuration_folder(tmp_path):
    folder = tmp_path/'configs'
    folder.mkdir()
    document = dict(run_document(), dataset=dataset_document('data'), out_root='runs', init_weights='../w/ckpt.pt',
                    training=dict(onpolicy=['replay/a.npz']))
    document['model']['frame_checkpoint'] = 'frames/ckpt.pt'
    (folder/'run.json').write_text(json.dumps(document))
    run = run_config.load(folder/'run.json')
    assert run['out_root'] == str(folder/'runs') and run['init_weights'] == str(tmp_path/'w'/'ckpt.pt')
    assert run['model']['frame_checkpoint'] == str(folder/'frames'/'ckpt.pt')
    assert run['training']['onpolicy'] == [str(folder/'replay'/'a.npz')]
    primary = run['dataset']['sources'][0]
    assert primary['fibers'] == str(folder/'data'/'fibers') and run['dataset']['cache_dir'] == str(folder/'data'/'cache')
    assert run_config.resolve(dict(run_document(), dataset=dataset_document('s3://bucket')))['dataset']['cache_dir'] == 's3://bucket/cache'


@pytest.mark.parametrize('model', ['regression', 'flow', 'sequence'])
def test_a_resolved_configuration_loads_back_unchanged(tmp_path, model):
    run = resolved(model, training=dict(batch=3), runtime=dict(workers=2))
    (tmp_path/'run.json').write_text(json.dumps(run, indent=2))
    assert run_config.load(tmp_path/'run.json') == run
    assert run_config.resolve(json.loads(json.dumps(run)), '/elsewhere') == run


def test_the_namespace_holds_every_setting_the_trainer_reads():
    read = set(re.findall(r'\bargs\.([a-z_]+)', (ROOT/'train'/'train.py').read_text()))
    read |= {'near_negative_bank', 'continuation_bank'}  # read through getattr(args, role)
    args = run_config.namespace(resolved(), resume='/runs/test/last.pt')
    assert read <= set(vars(args)), read-set(vars(args))
    assert (args.model, args.resume, args.name, args.fibers, args.val_z) == ('regression', '/runs/test/last.pt', 'test', '/data/fibers', [45000., 48500.])
    assert args.negative_bank == '/data/bank' and args.near_negative_bank is None


def test_example_run_configurations_resolve():
    examples = sorted((ROOT/'configs'/'runs').glob('*.json'))
    for path in examples:
        run = run_config.load(path)
        run_config.model_config(run)
        assert run['name'] and run['model']['type'] in ('regression', 'flow', 'sequence')
