"""Publish an independent resumable run after a validated checkpoint migration."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from .model import build_model
from .train import build_parser, checkpoint_config, options_argv


def fork_checkpoint(source, name, transform, *, allowed_changes, migration_key, describe, transform_fixture=None):
    """Prepare an independent run directory using only checkpoint settings.

    The frozen monitor fixture is copied, optionally adapting its metadata. Published source replay
    caches are referenced read-only; subsequent replay publication is independent.
    This prepares artifacts and a command, and does not stop or launch processes.
    """
    source = Path(source).resolve()
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a single directory name')
    original = torch.load(source, map_location='cpu', weights_only=False)
    migrated = transform(original)
    options = migrated['training_options']
    parser = build_parser()
    missing_defaults = {'remote_prefetch_connections','remote_prefetch_queue_size','remote_prefetch_timeout',
                         'remote_prefetch_lookahead','stem_channels','stem_blocks'}-options.keys()
    for key in sorted(missing_defaults):
        options.setdefault(key,parser.get_default(key))
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
    changes = {k:dict(before=original['training_options'].get(k), after=v)
               for k,v in options.items() if v != original['training_options'].get(k)}
    if changes.keys() - {'name', 'out_root', 'resume'} - allowed_changes - missing_defaults:
        raise ValueError('Unexpected training setting change')
    info = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_step=original['step'], **describe(original, migrated),
                changes=changes, replay_paths=replay,
                command=[sys.executable, '-u', '-m', 'vesuvius.neural_tracing.fiber_follow.regression.train', *argv])
    migrated[migration_key] = info
    config = dict(options)
    config.update({k:v for k,v in migrated.items() if k not in
                   ('model','ema','optimizer','rng','training_options','step','samples_seen','lr_restart_step')})
    with torch.random.fork_rng(devices=[]):
        config['parameter_count'] = sum(p.numel() for p in build_model(checkpoint_config(migrated)).parameters())
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=destination.parent))
    try:
        if options['recovery_every']:
            shutil.copyfile(fixture, staging/'monitor_recovery.npz')
            if transform_fixture is not None:
                transform_fixture(staging/'monitor_recovery.npz', original, migrated)
                digest = hashlib.sha256((staging/'monitor_recovery.npz').read_bytes()).hexdigest()
                migrated['monitor_recovery_sha256'] = config['monitor_recovery_sha256'] = digest
                info['monitor_recovery_sha256'] = digest
        torch.save(migrated, staging/target.name)
        shutil.copyfile(staging/target.name, staging/'last.pt')
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
