"""Fork a resumable checkpoint with additional learned refinement stages."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from .model import build_model
from .train import build_parser, checkpoint_config, options_argv


STAGE = 'refinement_stage.weight'


def expand_refinement_checkpoint(checkpoint, steps):
    """Keep existing weights/moments; copy the last stage into added stage rows.

    New rows get zero Adam moments, retaining the tensor's existing Adam step.
    Only the standard single-group optimizer layout is supported.
    """
    cfg = checkpoint_config(checkpoint)
    old = cfg.recurrent_refinement_steps
    if type(steps) is not int or not 1 <= old < steps:
        raise ValueError('Require an existing refinement stage and a larger integer stage count')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable training checkpoint: missing {key}')
    if checkpoint['training_options']['recurrent_refinement_steps'] != old:
        raise ValueError('Checkpoint model and training refinement settings disagree')
    # Building validation models must not consume the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        model = build_model(cfg)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.load_state_dict(checkpoint['ema'], strict=True)
        names = [name for name, _ in model.named_parameters()]
        groups = checkpoint['optimizer']['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(names):
            raise ValueError('Expected the standard complete single-group AdamW layout')
        stage_id = groups[0]['params'][names.index(STAGE)]
        migrated = copy.deepcopy(checkpoint)
        for section in ('model', 'ema'):
            value = checkpoint[section][STAGE]
            if tuple(value.shape) != (old, cfg.hidden):
                raise ValueError('Unexpected stage embedding shape')
            migrated[section][STAGE] = torch.cat((value, value[-1:].expand(steps-old, -1)), 0).clone()
        for key, value in migrated['optimizer']['state'].get(stage_id, {}).items():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if tuple(value.shape) != (old, cfg.hidden):
                    raise ValueError(f'Unexpected stage optimizer shape: {key}')
                migrated['optimizer']['state'][stage_id][key] = torch.cat(
                    (value, value.new_zeros(steps-old, cfg.hidden)), 0)
            elif key != 'step':
                raise ValueError(f'Unsupported stage optimizer field: {key}')
        migrated['model_cfg']['recurrent_refinement_steps'] = steps
        migrated['training_options']['recurrent_refinement_steps'] = steps
        expanded = build_model(checkpoint_config(migrated))
        expanded.load_state_dict(migrated['model'], strict=True)
        expanded.load_state_dict(migrated['ema'], strict=True)
        optimizer = torch.optim.AdamW(expanded.parameters())
        optimizer.load_state_dict(migrated['optimizer'])
        for param, state in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and state[key].shape != param.shape:
                    raise ValueError(f'Optimizer tensor shape mismatch: {key}')
    return migrated


def fork_run(source, name, steps):
    """Prepare an independent run directory using only checkpoint settings.

    The frozen monitor fixture is copied byte-for-byte. Published source replay
    caches are referenced read-only; subsequent replay publication is independent.
    This prepares artifacts and a command, and does not stop or launch processes.
    """
    source = Path(source).resolve()
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a single directory name')
    original = torch.load(source, map_location='cpu', weights_only=False)
    migrated = expand_refinement_checkpoint(original, steps)
    options = migrated['training_options']
    if options['reset_optimizer']:
        raise ValueError('Refusing a continuation that would reset optimizer state')
    destination = Path(options['out_root']).resolve()/name
    if destination.exists():
        raise FileExistsError(destination)
    fixture = source.parent/'monitor_recovery.npz'
    if options['recovery_every'] and hashlib.sha256(fixture.read_bytes()).hexdigest() != original['monitor_recovery_sha256']:
        raise ValueError('Source monitor fixture does not match checkpoint')
    replay_file = source.parent/'dagger/replay.json'
    replay = json.loads(replay_file.read_text()) if replay_file.exists() else options['onpolicy']
    if any(not Path(p).exists() for p in replay):
        raise ValueError('A source replay cache is missing')
    target = destination/f"ckpt_{original['step']:06d}.pt"
    options.update(name=name, out_root=str(destination.parent), resume=str(target))
    argv = options_argv(options)
    parsed = vars(build_parser().parse_args(argv))
    if json.dumps(parsed, sort_keys=True) != json.dumps(options, sort_keys=True):
        raise ValueError('Checkpoint options do not round-trip through the current trainer CLI')
    changes = {k:dict(before=original['training_options'][k], after=v)
               for k,v in options.items() if v != original['training_options'][k]}
    if changes.keys() - {'name', 'out_root', 'resume', 'recurrent_refinement_steps'}:
        raise ValueError('Unexpected training setting change')
    info = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_step=original['step'], source_refinements=original['model_cfg']['recurrent_refinement_steps'],
                refinement_steps=steps, initialization='copy last learned stage; zero new Adam moment rows; retain Adam step',
                changes=changes, replay_paths=replay,
                command=[sys.executable, '-u', '-m', 'vesuvius.neural_tracing.fiber_follow.regression.train', *argv])
    migrated['refinement_migration'] = info
    config = dict(options)
    config.update({k:v for k,v in migrated.items() if k not in
                   ('model','ema','optimizer','rng','training_options','step','samples_seen','lr_restart_step')})
    with torch.random.fork_rng(devices=[]):
        config['parameter_count'] = sum(p.numel() for p in build_model(checkpoint_config(migrated)).parameters())
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=destination.parent))
    try:
        torch.save(migrated, staging/target.name)
        shutil.copyfile(staging/target.name, staging/'last.pt')
        if options['recovery_every']:
            shutil.copyfile(fixture, staging/'monitor_recovery.npz')
        (staging/'dagger').mkdir()
        (staging/'dagger/replay.json').write_text(json.dumps(replay, indent=2)+'\n')
        (staging/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        (staging/'migration.json').write_text(json.dumps(info, indent=2)+'\n')
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--refinement-steps', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, args.refinement_steps), indent=2))


if __name__ == '__main__':
    main()
