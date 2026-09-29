"""Upgrade a resumable v4 checkpoint into a separate recurrent-refinement run.

Uses checkpoint model settings and verified runtime training options, never a
launcher recipe. Preserves optimizer moments by parameter name, EMA, RNG, step,
monitor fixture and published replay. Does not stop or launch any process.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import shutil

import torch

from .model import build_model
from .train import build_parser, checkpoint_config


def upgrade_checkpoint(ck, steps=1, limit=1.):
    cfg = checkpoint_config(ck)
    if cfg.memory_version != 4 or cfg.recurrent_refinement_steps:
        raise ValueError('Upgrade requires a single-pass feature-memory v4 checkpoint')
    if steps < 1 or not all(k in ck for k in ('optimizer', 'rng', 'step', 'ema')):
        raise ValueError('Positive refinement steps and a resumable training checkpoint are required')
    cfg = replace(cfg, recurrent_refinement_steps=steps, recurrent_refinement_limit=limit)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        old = build_model(checkpoint_config(ck))
        model = build_model(cfg)
    old.load_state_dict(ck['model'], strict=True)
    fresh = model.state_dict()
    new_names = set(fresh)-set(ck['model'])
    if not new_names or any(not k.startswith('refinement_') for k in new_names):
        raise ValueError('Unexpected refinement parameter migration')
    upgraded = dict(ck, model_cfg=cfg.to_dict())
    for key in ('model', 'ema'):
        state = {**{k: fresh[k].clone() for k in new_names}, **ck[key]}
        model.load_state_dict(state, strict=True)
        upgraded[key] = state
    # The trainer uses one AdamW group. Parameter registration order can change
    # when modules are added; map saved moment IDs through the old model names.
    optimizer = copy.deepcopy(ck['optimizer'])
    if len(optimizer['param_groups']) != 1:
        raise ValueError('Expected the trainer single AdamW parameter group')
    group = optimizer['param_groups'][0]
    old_names = [name for name, _ in old.named_parameters()]
    if len(group['params']) != len(old_names):
        raise ValueError('Optimizer parameter count differs from the source model')
    by_name = dict(zip(old_names, group['params'], strict=True))
    new_order = [name for name, _ in model.named_parameters()]
    optimizer['state'] = {i: optimizer['state'][by_name[name]] for i, name in enumerate(new_order)
                          if name in by_name and by_name[name] in optimizer['state']}
    group['params'] = list(range(len(new_order)))
    upgraded['optimizer'] = optimizer
    upgraded['refinement_upgrade'] = dict(source_step=ck['step'], new_parameters=sorted(new_names),
        recurrent_refinement_steps=steps, recurrent_refinement_limit=limit,
        optimizer='existing moments retained by parameter name; new parameters start without moments')
    return upgraded


def options_argv(options):
    argv = []
    for action in build_parser()._actions:
        if action.dest == 'help' or action.dest not in options:
            continue
        value = options[action.dest]
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(action.option_strings[0 if value else 1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
        else:
            argv.append(action.option_strings[0])
            argv.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime-audit', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--limit', type=float, default=1.)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('Run name must be a single directory name')
    source = args.checkpoint.resolve()
    audit = json.loads(args.runtime_audit.read_text())
    if audit['cli_event_differences']:
        raise ValueError('Live process arguments did not match the recorded resume configuration')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    if checkpoint_config(ck) != checkpoint_config(dict(ck, model_cfg=audit['model_cfg'])):
        raise ValueError('Checkpoint model configuration differs from verified live configuration')
    # Require the most recent logged runtime configuration still to match the audit.
    events = [json.loads(line) for line in (source.parent/'log.jsonl').read_text().splitlines()]
    runtime = [e['training_options'] for e in events if e.get('event') == 'resume_configuration'][-1]
    if runtime != audit['runtime_options']:
        raise ValueError('Runtime configuration changed since the process audit')
    upgraded = upgrade_checkpoint(ck, args.steps, args.limit)
    options = dict(runtime)
    destination = Path(options['out_root']).resolve()/args.name
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = destination/f'ckpt_{ck["step"]:06d}.pt'
    options.update(name=args.name, resume=str(checkpoint), init_tracer=None,
                   recurrent_refinement_steps=args.steps, recurrent_refinement_limit=args.limit)
    upgraded['training_options'] = options
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    upgraded['refinement_upgrade'].update(source_checkpoint=str(source), source_sha256=source_hash,
        runtime_audit=audit, option_changes={k: [runtime.get(k), v] for k, v in options.items()
                                          if runtime.get(k) != v})
    config = json.loads((source.parent/'config.json').read_text())
    config.update(options)
    for key in ('architecture', 'model_cfg', 'sample_cfg', 'vol_spec', 'data_policy',
                'identity_sampling', 'negative_bank_provenance', 'bank_role_provenance',
                'seed_manifest_sha256', 'fiber_manifest', 'monitor_recovery_sha256',
                'feature_sampling_revision', 'refinement_upgrade'):
        if key in upgraded:
            config[key] = upgraded[key]
    config['parameter_count'] = sum(p.numel() for p in build_model(checkpoint_config(upgraded)).parameters())
    fixture = source.parent/'monitor_recovery.npz'
    if options['recovery_every']:
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != ck['monitor_recovery_sha256']:
            raise ValueError('Original monitor recovery fixture hash differs from checkpoint')
    replay = source.parent/'dagger'/'replay.json'
    replay_paths = json.loads(replay.read_text()) if replay.exists() else options['onpolicy']
    if any(not Path(p).exists() for p in replay_paths):
        raise FileNotFoundError('Published replay bank is missing')
    argv = options_argv(options)
    parsed = vars(build_parser().parse_args(argv))
    if json.dumps(parsed, sort_keys=True) != json.dumps(options, sort_keys=True):
        raise ValueError('Generated trainer arguments do not reproduce the effective configuration')
    destination.mkdir()
    torch.save(upgraded, checkpoint)
    shutil.copy2(checkpoint, destination/'last.pt')
    if options['recovery_every']:
        shutil.copy2(fixture, destination/fixture.name)
    (destination/'dagger').mkdir()
    (destination/'dagger'/'replay.json').write_text(json.dumps(replay_paths, indent=2))
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    (destination/'upgrade.json').write_text(json.dumps(upgraded['refinement_upgrade'], indent=2))
    (destination/'resume_argv.json').write_text(json.dumps(argv, indent=2))
    command = ['bash', 'scripts/launch_regression.sh', args.name, *argv[2:]]
    (destination/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        + 'export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1\n'
        + 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1\n'
        + 'export TORCHINDUCTOR_COMPILE_THREADS=4\n'
        + 'cd '+shlex.quote(str(Path(__file__).resolve().parents[1]))+'\n'
        + shlex.join(command)+'\n')
    print(json.dumps(dict(destination=str(destination), checkpoint=str(checkpoint), step=ck['step'],
                         changes=upgraded['refinement_upgrade']['option_changes']), indent=2))


if __name__ == '__main__':
    main()
