"""Fork a checkpoint into a new run trained on tracing-consistent samples, with a fresh optimizer.

Preserves every model/EMA tensor, RNG and update count. AdamW starts without moments and the
LR schedule restarts its warmup at the fork step, decaying to the new ``--steps``. Retired light-GT
perturbation options are dropped and the trace-start shares are recorded explicitly. Prepares the
run directory and its launch command only; nothing is started.
"""
import argparse
import copy
import json
from pathlib import Path
import shlex

import torch

from .checkpoint_migration import fork_checkpoint
from .model import build_model
from .train import checkpoint_config

RETIRED_OPTIONS = ('gt_perturb_max_offset', 'gt_perturb_max_angle_deg')


def trace_sampling_checkpoint(checkpoint, *, steps, warmup, no_history_prob, short_history_prob):
    if int(steps) <= int(checkpoint['step']):
        raise ValueError('The fork must train beyond its source step')
    if int(warmup) < 0 or not all(0 <= p <= 1 for p in (no_history_prob, short_history_prob)):
        raise ValueError('Warmup must be nonnegative and trace-start shares in [0, 1]')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable checkpoint: missing {key}')
    migrated = copy.copy(checkpoint)
    options = migrated['training_options'] = copy.deepcopy(checkpoint['training_options'])
    with torch.random.fork_rng(devices=[]):
        model = build_model(checkpoint_config(checkpoint))
        model.load_state_dict(checkpoint['model'], strict=True)
        # The trainer's AdamW layout with no moments; resume loads it unchanged.
        migrated['optimizer'] = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=1e-4).state_dict()
    migrated['lr_restart_step'] = int(checkpoint['step'])
    for key in RETIRED_OPTIONS:
        options.pop(key, None)
    options.update(steps=int(steps), warmup=int(warmup), no_history_prob=float(no_history_prob),
                   short_history_prob=float(short_history_prob))
    return migrated


def fork_run(source, name, *, steps, warmup, no_history_prob, short_history_prob):
    settings = dict(steps=steps, warmup=warmup, no_history_prob=no_history_prob, short_history_prob=short_history_prob)
    info = fork_checkpoint(source, name, lambda ck: trace_sampling_checkpoint(ck, **settings),
        allowed_changes=set(settings), migration_key='trace_sampling_migration',
        describe=lambda original, migrated: dict(
            initialization='preserve all model/EMA tensors, RNG and update count; fresh AdamW (no moments); '
                           f"LR warmup restarts at step {original['step']}",
            sampling='simulated GT traces (annotated seed, OU tracing error, tracer heading/history/CT seed heading), '
                     'tracer-consistent synthetic switches and decision pairs, roll flip+jitter, evaluation departure rule',
            retired_options={k: original['training_options'].get(k) for k in RETIRED_OPTIONS},
            lr_restart_step=migrated['lr_restart_step']))
    command = info['command']
    run = Path(command[command.index('--out-root')+1])/command[command.index('--name')+1]
    (run/'launch_command.sh').write_text(' '.join(shlex.quote(a) for a in command)+'\n')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, required=True, help='Final update count of the forked run')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--no-history-prob', type=float, default=.05)
    parser.add_argument('--short-history-prob', type=float, default=.1)
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, steps=args.steps, warmup=args.warmup,
                              no_history_prob=args.no_history_prob, short_history_prob=args.short_history_prob),
                     indent=2))


if __name__ == '__main__':
    main()
