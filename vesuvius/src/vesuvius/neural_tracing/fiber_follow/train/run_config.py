"""The run configuration: one JSON file holding the dataset, model, training and runtime settings of a training run.

    {
      "name": "my_run",                  # run folder under out_root
      "out_root": "output",              # optional; default fiber_follow/output
      "init_weights": null,              # optional checkpoint whose model/EMA tensors start the run (name+shape match)
      "init_exclude": [],                # tensor-name prefixes kept at initialization
      "reset_optimizer": false,          # with --resume: fresh AdamW and LR schedule from the resumed step
      "dataset": {...},                  # the dataset configuration (data/datasets.py: version, cache_dir, sources)
      "model": {"type": "regression"},   # 'regression', 'flow' or 'sequence', plus any model field
      "training": {...},                 # optimizer, schedule, losses, sampling, task shares, DAgger, diagnostics
      "runtime": {...}                   # device, workers, threads, CT prefetch: settings of the machine
    }

Every omitted field takes its model type's default (DEFAULTS below and the model configuration classes): these
are the settings of the runs each model type was established with. Unknown keys are errors. Relative paths resolve
against the configuration file's folder. ``resolve`` returns the complete configuration, which the trainer saves as
``run.json`` in the run folder and in every checkpoint (``run_config``); it loads again unchanged.
"""
import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.models.model import MODEL_TYPES, RETIRED_FIELDS, config_class
from vesuvius.neural_tracing.fiber_follow.shared.retired import ANY, retire as retire_settings

OUT_ROOT = Path(__file__).resolve().parents[1]/'output'
TOP_LEVEL = dict(name=None, out_root=None, init_weights=None, init_exclude=[], reset_optimizer=False)
SECTIONS = ('dataset', 'model', 'training', 'runtime')

# Training settings shared by every model type.
TRAINING = dict(
    steps=30000, batch=16, grad_steps=1, lr=1e-4, warmup=2000, ema_decay=.999, grad_clip=60., seed=0,
    confidence_weight=.5, tolerance=2., refinement_loss='policy', refinement_weight=1.,
    flow_calibration_states=2048,
    # Operating policy: live chains, DAgger collection, monitor rollouts and proposal selection.
    n_commit=8, trace_confidence=.5, gate='full',
    # Sequence episodes (model 'sequence').
    episode_steps=64, episode_supervised=16, episode_commit=None,
    # Task budget (data/data.py TASKS); a given dict replaces the whole default, absent tasks get 0.
    task_shares=dict(fresh=.30, live=.45, dagger_pre_excursion=.06, dagger_recoverable=.08, dagger_terminal=.08,
                     dagger_premature_stop=.03, dagger_ordinary=0., synthetic_terminal=0.),
    terminal_fallback_cap=.5, replay_max_age=3000, replay_event_cap=64,
    # Simulated traces and labels.
    startup_shares=[.15, .17, .17, .51], excursion_probability=.2, excursion_amplitude=[3., 6.],
    excursion_rise=[16., 128.], synthetic_tail=[4., 16.], live_continuation_steps=[32, 256],
    seed_offset=[0., 0.], seed_offset_ramp=16., live_seed_start=.5,
    # Neighbour banks, augmentation and AFV sampling.
    bank_coverage_probability=.2, bank_hard_fraction=.5, bank_switch_tolerance=.75, bank_own_tolerance=1.5,
    blur_probability=.25, blur_sigma=[.5, 1.25], lateral_fraction=.1, afv_length_power=3.,
    # On-policy collection (DAgger) and replay.
    dagger_every=1000, dagger_fibers=128, afv_dagger_fibers=384, dagger_batch=24, dagger_forward_chunk=0,
    dagger_trace_len=6000., dagger_before=48., dagger_after=64., dagger_stride=16., replay_keep=4,
    # Logging, checkpoints and diagnostics.
    log_every=50, ckpt_every=1000, diag_every=5000, batch_diag_every=1000, diag_max_len=400., recovery_every=1000,
    recovery_seeds=8, recovery_length=32.)

# Per model type: the settings of mixed_ct_afv_unified_v1 (regression), unified_flow_v6_adalnzero (flow) and
# sequence_v1 (sequence) where they differ from TRAINING. The sequence run's live and synthetic shares (which sequence
# training does not support) are folded into fresh.
DEFAULTS = dict(
    regression=dict(batch=16, grad_steps=2, refinement_loss='all', terminal_fallback_cap=0., dagger_forward_chunk=8),
    flow=dict(steps=40000, batch=48, warmup=1000, grad_clip=10., n_commit=16, gate='prefix', dagger_every=500,
              task_shares=dict(fresh=.27, live=.45, dagger_pre_excursion=.06, dagger_recoverable=.08,
                               dagger_terminal=.08, dagger_premature_stop=.03, dagger_ordinary=0.,
                               synthetic_terminal=.03),
              synthetic_tail=[4., 128.]),
    sequence=dict(batch=20, n_commit=12, episode_commit=12, dagger_every=0, dagger_fibers=64, afv_dagger_fibers=None,
                  dagger_batch=8, dagger_trace_len=768., batch_diag_every=0, live_continuation_steps=[12, 32],
                  task_shares=dict(fresh=.70, live=0., dagger_pre_excursion=.08, dagger_recoverable=.06,
                                   dagger_terminal=.08, dagger_premature_stop=.03, dagger_ordinary=.05,
                                   synthetic_terminal=0.)))

RUNTIME = dict(device='cuda', workers=10, worker_cache_gb=.5, threads=4, dagger_threads=4,
               remote_prefetch_connections=48, remote_prefetch_queue_size=512, remote_prefetch_lookahead=16,
               remote_prefetch_timeout=120.)

# Model fields set by the trainer, not by a configuration.
DERIVED_MODEL_FIELDS = ('model_type', 'frame_checkpoint_sha256')

# Removed settings, which run files written before their removal still hold: the listed value (ANY: every value) had
# the effect of the current code, so the setting is dropped on load; any other value is refused.
RETIRED = dict(model=RETIRED_FIELDS,
               training=dict(negative_bank_refresh_seconds=ANY, negative_bank_cache_mb=ANY, retry_threshold=None,
                             lr_step_offset=0, onpolicy=[], long_diag_every=0, long_diag_max_len=ANY),
               runtime=dict(dagger_device=None))


def model_fields(model_type):
    return {f.name for f in fields(config_class(model_type))}-set(DERIVED_MODEL_FIELDS)


def training_defaults(model_type):
    return json.loads(json.dumps(dict(TRAINING, **DEFAULTS[model_type])))


def model_defaults(model_type):
    """The model section's defaults: the model configuration class's fields, plus the default frame checkpoint."""
    from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import DEFAULT_FRAME_CHECKPOINT
    values = {}
    for f in fields(config_class(model_type)):
        if f.name not in DERIVED_MODEL_FIELDS:
            values[f.name] = f.default_factory() if callable(f.default_factory) else f.default
    values['fine'] = asdict(values['fine'])
    values['frame_checkpoint'] = values.get('frame_checkpoint') or str(DEFAULT_FRAME_CHECKPOINT)
    return json.loads(json.dumps(values))


def retire(section, values):
    return retire_settings(values, RETIRED[section], section)


def unknown(section, given, allowed):
    extra = sorted(set(given)-set(allowed))
    if extra:
        raise ValueError(f'Unknown {section} settings: {", ".join(extra)}')


def resolve(document, base='.'):
    """The complete run configuration: ``document`` with every default filled in and paths made absolute."""
    from vesuvius.neural_tracing.fiber_follow.data.datasets import parse_dataset_config
    from vesuvius.neural_tracing.fiber_follow.data.data import TASKS
    base = Path(base).resolve()
    document = json.loads(json.dumps(document))
    unknown('top-level', document, (*TOP_LEVEL, *SECTIONS))
    if not document.get('name'):
        raise ValueError('The run configuration needs a name')
    if 'dataset' not in document:
        raise ValueError('The run configuration needs a dataset section')
    path = lambda value: value if value is None or '://' in str(value) else str((base/value).resolve())
    run = {key: document.get(key, json.loads(json.dumps(default))) for key, default in TOP_LEVEL.items()}
    run['out_root'] = path(run['out_root']) if run['out_root'] else str(OUT_ROOT)
    run['init_weights'] = path(run['init_weights'])
    run['dataset'] = parse_dataset_config(document['dataset'], base)[0]

    model = retire('model', dict(document.get('model') or {}))
    model_type = model.pop('type', 'regression')
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unknown model type {model_type!r} (supported: {", ".join(MODEL_TYPES)})')
    unknown('model', model, model_fields(model_type))
    values = model_defaults(model_type)
    values.update(model)
    values['frame_checkpoint'] = path(values['frame_checkpoint'])
    run['model'] = dict(type=model_type, **values)
    model_config(run)  # validate the model section now

    training = retire('training', dict(document.get('training') or {}))
    defaults = training_defaults(model_type)
    unknown('training', training, defaults)
    if 'task_shares' in training:
        unknown('task share', training['task_shares'], TASKS)
        training['task_shares'] = {task: float(training['task_shares'].get(task, 0.)) for task in TASKS}
    else:
        defaults['task_shares'] = {task: float(defaults['task_shares'].get(task, 0.)) for task in TASKS}
    defaults.update(training)
    run['training'] = defaults

    runtime = retire('runtime', dict(document.get('runtime') or {}))
    unknown('runtime', runtime, RUNTIME)
    run['runtime'] = dict(RUNTIME, **runtime)
    validate(run)
    return run


def load(path):
    path = Path(path).resolve()
    return resolve(json.loads(path.read_text()), path.parent)


def model_config(run):
    """The model configuration of a resolved run (``frame_checkpoint_sha256`` is bound by the trainer)."""
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
    values = {k: v for k, v in run['model'].items() if k != 'type'}
    values['fine'] = CropSpec(**values['fine'])
    for key in ('cnn_channels', 'cnn_blocks', 'flow_sigma'):
        if isinstance(values.get(key), list):
            values[key] = tuple(tuple(v) if isinstance(v, list) else v for v in values[key])
    return config_class(run['model']['type'])(**values)


def validate(run):
    from vesuvius.neural_tracing.fiber_follow.data.data import TASKS, TaskBudget
    t, r = run['training'], run['runtime']
    TaskBudget(tuple(t['task_shares'][name] for name in TASKS), terminal_fallback_cap=t['terminal_fallback_cap'],
               replay_max_age=t['replay_max_age'], replay_event_cap=t['replay_event_cap'])
    if t['refinement_loss'] not in ('policy', 'all'):
        raise ValueError("refinement_loss is 'policy' or 'all'")
    if not math.isfinite(t['refinement_weight']) or t['refinement_weight'] < 0:
        raise ValueError('refinement_weight must be finite and nonnegative')
    if not math.isfinite(t['grad_clip']) or t['grad_clip'] < 0:
        raise ValueError('grad_clip must be finite and nonnegative (0 disables clipping)')
    if (r['remote_prefetch_connections'] < 0 or r['remote_prefetch_queue_size'] < 1 or r['remote_prefetch_lookahead'] < 0
            or not math.isfinite(r['remote_prefetch_timeout']) or r['remote_prefetch_timeout'] <= 0):
        raise ValueError('Prefetch connections and lookahead must be nonnegative; queue size and timeout must be positive')
    if min(t['steps'], t['batch'], t['grad_steps'], t['log_every'], t['ckpt_every'], r['threads'], r['dagger_threads'],
           t['flow_calibration_states'], t['replay_keep'], t['dagger_fibers'], t['dagger_batch'], t['recovery_seeds']) < 1:
        raise ValueError('Positive counts required, including batch and grad steps')
    if t['afv_dagger_fibers'] is not None and t['afv_dagger_fibers'] < 1:
        raise ValueError('AFV collection fiber count must be positive')
    if min(r['workers'], t['warmup'], t['diag_every'], t['batch_diag_every'], t['dagger_every'],
           t['recovery_every'], t['confidence_weight']) < 0:
        raise ValueError('Invalid training settings')
    if not 0 <= t['ema_decay'] < 1 or min(t['lr'], t['tolerance'], r['worker_cache_gb'], t['dagger_after'],
                                          t['diag_max_len'], t['dagger_trace_len'],
                                          t['recovery_length']) <= 0:
        raise ValueError('Invalid loss, learning rate, cache, or rollout settings')
    if not math.isfinite(t['afv_length_power']) or t['afv_length_power'] < 0:
        raise ValueError('AFV length power must be finite and nonnegative')
    if not 1 <= t['live_continuation_steps'][0] <= t['live_continuation_steps'][1]:
        raise ValueError('Live continuation requires 1 <= minimum steps <= maximum steps')
    if run['model']['type'] == 'sequence' and any(t['task_shares'].get(name) for name in ('live', 'synthetic_terminal')):
        # Episodes are simulated traces ('fresh') or windows of collected traces ('dagger_*'), nothing else.
        raise ValueError('Sequence training uses fresh and DAgger episodes; set the live and synthetic task shares to 0')


def namespace(run, resume=None):
    """The flat settings the trainer reads (``args``), from a resolved run configuration."""
    from vesuvius.neural_tracing.fiber_follow.data.datasets import PRIMARY_OPTIONS
    primary = next(s for s in run['dataset']['sources'] if s['kind'] == 'paris4')
    values = dict({k: run[k] for k in TOP_LEVEL}, **run['training'], **run['runtime'],
                  **{key: primary.get(key) for key in PRIMARY_OPTIONS},
                  model=run['model']['type'], frame_checkpoint=run['model']['frame_checkpoint'], resume=resume)
    values['init_exclude'] = list(values['init_exclude'])
    return argparse.Namespace(**values)
