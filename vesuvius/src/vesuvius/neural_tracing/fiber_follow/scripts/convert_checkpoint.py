"""One-off conversion of checkpoints written before the model cleanup to the current format.

    python scripts/convert_checkpoint.py SOURCE.pt DESTINATION.pt --run-config RUN.json

Model types 'unified' -> 'regression', 'unified_flow' -> 'flow', 'sequence' -> 'sequence'. The model, EMA, optimizer,
RNG and step are kept; the model configuration keeps the fields the current model reads (removed memory, identity,
whole-crop and tube fields must hold their inert values, or the conversion refuses). The run configuration that
resumes the run is rebuilt from the checkpoint's dataset configuration, model configuration and training options,
stored in the checkpoint (``run_config``) and written to RUN.json:

    python train/train.py --config RUN.json --resume DESTINATION.pt

Paths stay those of the machine that wrote the checkpoint. The source is never modified and an existing destination
is never overwritten. The converted model, EMA and optimizer state are loaded into the current model before saving.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

LEGACY_TYPES = dict(unified='regression', unified_flow='flow', sequence='sequence')
# Removed model fields and the values under which they had no effect.
INERT_MODEL_FIELDS = dict(memory='none', identity_dim=0, identity_objective='infonce', identity_map=False,
                          identity_feedback=False, path_planes='future', tube_head=False)
DROPPED_KEYS = ('data_policy', 'history_sampling_revision', 'sample_contract', 'negative_bank_provenance',
                'bank_role_provenance', 'monitor_recovery_sha256', 'seed_manifest_sha256', 'dataset_config_sha256')
RENAMED_OPTIONS = dict(rest_grad_clip='grad_clip')
SEQUENCE_UNSUPPORTED_TASKS = ('live', 'synthetic_terminal')


def convert_model_config(model_type, recorded):
    from vesuvius.neural_tracing.fiber_follow.models.model import RETIRED_FIELDS, config_class
    for key, inert in {**INERT_MODEL_FIELDS, **RETIRED_FIELDS}.items():
        if recorded.get(key, inert) != inert:
            raise ValueError(f'Model field {key}={recorded[key]!r} has no counterpart in the current model')
    kept = {f.name for f in fields(config_class(model_type))}
    values = {k: v for k, v in recorded.items() if k in kept}
    values['model_type'] = model_type
    return config_class(model_type)(**values), sorted(set(recorded)-kept)


def task_shares(budget, model_type, notes):
    shares = dict(budget['shares'])
    if shares.pop('synthetic_identity', 0.):
        raise ValueError('The synthetic_identity task no longer exists')
    if model_type == 'sequence':
        moved = {task: shares[task] for task in SEQUENCE_UNSUPPORTED_TASKS if shares.get(task)}
        if moved:
            shares['fresh'] += sum(moved.values())
            for task in moved:
                shares[task] = 0.
            notes.append(f'sequence training supports no {"/".join(moved)} tasks: their shares '
                         f'({", ".join(f"{k} {v:g}" for k, v in moved.items())}) moved to fresh ({shares["fresh"]:g})')
    return shares


def run_document(ck, cfg, notes):
    """The run configuration (unresolved) reproducing a legacy checkpoint's run."""
    from vesuvius.neural_tracing.fiber_follow.train import run_config
    options = dict(ck['training_options'])
    for old, new in RENAMED_OPTIONS.items():
        if old in options:
            options[new] = options.pop(old)
    model = {k: v for k, v in json.loads(json.dumps(cfg.to_dict())).items()
             if k not in run_config.DERIVED_MODEL_FIELDS and k != 'flow_sigma'}
    # Retired settings pass through, so resolving refuses one that held an unsupported value.
    training = {k: options[k] for k in (*run_config.training_defaults(cfg.model_type), *run_config.RETIRED['training'])
                if k in options}
    training['task_shares'] = task_shares(ck['task_budget'], cfg.model_type, notes)
    training['terminal_fallback_cap'] = ck['task_budget']['terminal_fallback_cap']
    training['replay_max_age'] = ck['task_budget']['replay_max_age']
    training['replay_event_cap'] = ck['task_budget']['replay_event_cap']
    runtime = {k: options[k] for k in (*run_config.RUNTIME, *run_config.RETIRED['runtime']) if k in options}
    return dict(name=options['name'], out_root=options['out_root'], init_weights=options.get('init_weights'),
                init_exclude=[], reset_optimizer=False, dataset=ck['dataset_config'],
                model=dict(type=cfg.model_type, **model), training=training, runtime=runtime)


def convert(ck):
    """(converted checkpoint, resolved run configuration, notes)."""
    from vesuvius.neural_tracing.fiber_follow.train import run_config
    if ck.get('model_type') not in LEGACY_TYPES:
        raise ValueError(f'Not a legacy checkpoint of a kept model type: {ck.get("model_type")!r}')
    if ck.get('dataset_config') is None:
        raise ValueError('The checkpoint records no dataset configuration')
    notes = []
    cfg, dropped = convert_model_config(LEGACY_TYPES[ck['model_type']], ck['model_cfg'])
    if dropped:
        notes.append(f'model fields dropped: {", ".join(dropped)}')
    run = run_config.resolve(run_document(ck, cfg, notes), '/')
    out = {k: v for k, v in ck.items() if k not in DROPPED_KEYS}
    out.update(model_type=cfg.model_type, model_cfg=cfg.to_dict(), run_config=run,
               training_options=vars(run_config.namespace(run)))
    sample = dict(ck['sample_cfg'])
    if sample.pop('crop_targets', False):
        raise ValueError('Whole-crop targets have no counterpart in the current model')
    out['sample_cfg'] = sample
    out['task_budget'] = dict(ck['task_budget'], shares={k: v for k, v in ck['task_budget']['shares'].items()
                                                          if k != 'synthetic_identity'})
    if 'identity_sampling' in ck:
        out['identity_sampling'] = {k: v for k, v in ck['identity_sampling'].items()
                                    if k not in ('identity_switch_tail', 'memory_augmentation')}
    return out, run, notes


def verify(out):
    """Load the converted model, EMA and optimizer state into the current model."""
    from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
    model = build_model(config_from_checkpoint(out))
    model.load_state_dict(out['model'])
    ema = build_model(config_from_checkpoint(out))
    ema.load_state_dict(out['ema'])
    if 'optimizer' in out:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        opt.load_state_dict(out['optimizer'])
        shapes = [tuple(p.shape) for p in model.parameters()]
        moments = [tuple(opt.state[p]['exp_avg'].shape) for p in model.parameters() if p in opt.state]
        if moments and moments != shapes[:len(moments)]:
            raise ValueError('Optimizer moments do not match the model parameters')
    return sum(p.numel() for p in model.parameters())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    ap.add_argument('--run-config', type=Path, required=True, help='Where to write the run configuration')
    args = ap.parse_args(argv)
    for path in (args.destination, args.run_config):
        if path.exists():
            raise FileExistsError(f'{path} exists; choose another destination')
    ck = torch.load(args.source, map_location='cpu', weights_only=False)
    out, run, notes = convert(ck)
    parameters = verify(out)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix('.partial.pt')
    torch.save(out, temporary)
    temporary.replace(args.destination)
    args.run_config.write_text(json.dumps(run, indent=2)+'\n')
    print(f'{args.source}: {ck["model_type"]} -> {out["model_type"]} at step {out.get("step")}, '
          f'{parameters:,} parameters; model, EMA and optimizer load into the current model')
    for note in notes:
        print(f'  {note}')
    print(f'Wrote {args.destination} and {args.run_config}')
    print(f'Resume: python train/train.py --config {args.run_config} --resume {args.destination}')


if __name__ == '__main__':
    main()
