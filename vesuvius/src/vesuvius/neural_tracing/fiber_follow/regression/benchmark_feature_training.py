"""Bounded real-data training fork with steady wall-clock throughput and VRAM.

Preserves the source run and learning-rate schedule. Copies its resumable
checkpoint into a new directory and stops after --updates without changing the
trainer's total step count. No evaluation/collector boundary may be crossed.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import train
from .train import options_argv
from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state


class BenchmarkComplete(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--updates', type=int, default=80)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args()
    if args.out.exists() or not 0 <= args.warmup < args.updates or args.workers < 0:
        ap.error('Need a fresh directory, updates > warmup, and nonnegative workers')
    source = args.checkpoint.resolve()
    ck = torch.load(source, map_location='cpu', weights_only=False)
    options = vars(train.build_parser().parse_args(options_argv(ck['training_options'])))
    if ck['step']+args.updates >= options['steps']:
        ap.error('Source training schedule must extend beyond the requested benchmark')
    for name in ('ckpt_every', 'diag_every', 'long_diag_every', 'dagger_every', 'recovery_every'):
        interval = options[name]
        if interval and ck['step']//interval != (ck['step']+args.updates)//interval:
            ap.error(f'Bounded benchmark must not cross {name} boundary')
    destination = args.out.resolve()
    cp = destination/source.name
    options.update(name=destination.name, out_root=str(destination.parent), resume=str(cp),
                   workers=args.workers, log_every=10)
    options.update({key: ck['model_cfg'][key] for key in train.FEATURE_OPTIONS})
    destination.mkdir(parents=True)
    staged = dict(ck, training_options=options)
    torch.save(staged, cp)
    config = json.loads((source.parent/'config.json').read_text())
    config.update(options, model_cfg=ck['model_cfg'])
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    if options['recovery_every']:
        shutil.copy2(source.parent/'monitor_recovery.npz', destination/'monitor_recovery.npz')
    (destination/'dagger').mkdir()
    shutil.copy2(source.parent/'dagger/replay.json', destination/'dagger/replay.json')
    original = train.optimizer_update
    rows = []
    observed = 0
    last_return = None
    test_started = time.perf_counter()

    def measured(model, ema, opt, batches, step, lr, **kwargs):
        nonlocal observed, last_return
        torch.cuda.synchronize()
        now = time.perf_counter()
        wait = None if last_return is None else now-last_return
        torch.cuda.reset_peak_memory_stats()
        from torch._dynamo.utils import counters
        graphs_before = counters['stats']['unique_graphs']
        metrics = original(model, ema, opt, batches, step, lr, **kwargs)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        elapsed = finished-now
        observed += metrics['observed_states']
        row = dict(step=step, optimizer_ms=elapsed*1000, between_updates_seconds=wait,
                   compiled_graphs=counters['stats']['unique_graphs']-graphs_before,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   **{k: v for k, v in metrics.items() if k in ('loss', 'grad_norm', 'observed_states', 'endpoint_states', 'supervised_states', 'history_encoder_crops', 'memory_replay_observations') or k.startswith('replay_')})
        rows.append(row)
        with (destination/'timings.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print('MEASURED '+json.dumps(row), flush=True)
        last_return = finished
        if len(rows) == args.updates:
            saved = dict(staged, model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                         rng=training_rng_state(), step=step, samples_seen=ck['samples_seen']+observed)
            torch.save(saved, destination/f'ckpt_{step:06d}.pt')
            raise BenchmarkComplete()
        return metrics

    train.optimizer_update = measured
    try:
        train.main(options_argv(options))
    except BenchmarkComplete:
        pass
    finally:
        train.optimizer_update = original
    summary, measured_rows = steady_summary(rows, args.warmup)
    values = [r['optimizer_ms'] for r in measured_rows]
    result = dict(source_checkpoint=str(source), hardware=torch.cuda.get_device_name(),
        config=ck['model_cfg'], options=options, updates=len(rows), warmup=args.warmup,
        elapsed_seconds=time.perf_counter()-test_started, observed_crops=observed,
        measured_updates=len(measured_rows), excluded_compilation_updates=sum(bool(r['compiled_graphs']) for r in rows[args.warmup:]),
        **summary,
        mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in measured_rows),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in measured_rows),
        supervised_states=sum(r['supervised_states'] for r in measured_rows),
        memory_replay_observations=sum(r['memory_replay_observations'] for r in measured_rows),
        history_encoder_crops=sum(r['history_encoder_crops'] for r in measured_rows),
        samples=rows)
    (destination/'benchmark.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'options', 'config')}), flush=True)


def steady_summary(rows, warmup):
    """Use the contiguous suffix after warmup and the final compilation event."""
    start = max(1, warmup)
    for i, row in enumerate(rows):
        if row['compiled_graphs']:
            start = max(start, i+1)
    measured = rows[start:]
    if not measured:
        raise RuntimeError('No contiguous compilation-free measured updates; increase --updates')
    optimizer = sum(r['optimizer_ms'] for r in measured)/1000
    between = sum(r['between_updates_seconds'] for r in measured)
    wall = optimizer+between
    durations = [r['optimizer_ms']+1000*r['between_updates_seconds'] for r in measured]
    result = dict(first_measured_step=measured[0]['step'], last_measured_step=measured[-1]['step'],
                  steady_wall_seconds=wall, steady_optimizer_seconds=optimizer,
                  steady_between_updates_seconds=between,
                  wall_mean_ms=float(np.mean(durations)), wall_p50_ms=float(np.median(durations)),
                  wall_p95_ms=float(np.percentile(durations, 95)))
    for label, key in (('observations', 'observed_states'), ('endpoints', 'endpoint_states'),
                       ('supervised_decisions', 'supervised_states')):
        count = sum(r[key] for r in measured)
        result[label+'_per_second'] = count/wall
        result[label+'_per_optimizer_second'] = count/optimizer
    return result, measured


if __name__ == '__main__':
    main()
