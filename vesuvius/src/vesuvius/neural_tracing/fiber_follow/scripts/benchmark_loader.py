"""CPU-only real-data loader benchmark from a saved regression config.

Run with --workers 0 --profile PATH to locate worker hotspots. --hash-batches
records exact tensor digests (excluding timing fields) outside timed regions.
No model, checkpoint, replay index, or source data is modified.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import (
    FollowDataset, SampleConfig, ZBand, load_fibers, split_fibers, OnPolicyStates,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.training_options import normalize_batch_options
from vesuvius.neural_tracing.fiber_follow.shared.runloop import raise_open_file_limit
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank


def tensor_hashes(batch, prefix=''):
    result = {}
    for key, value in batch.items():
        name = prefix+key
        if isinstance(value, dict):
            result.update(tensor_hashes(value, name+'.'))
        elif isinstance(value, torch.Tensor) and not key.endswith('_seconds'):
            result[name] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batches', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--profile', type=Path)
    ap.add_argument('--hash-batches', action='store_true')
    args = ap.parse_args()
    if args.batches < 1 or min(args.warmup, args.workers) < 0 or (args.profile and args.workers):
        ap.error('Positive batches, nonnegative warmup/workers; profiling requires workers=0')
    raise_open_file_limit()
    torch.set_num_threads(1)
    c = normalize_batch_options(json.loads(args.config.read_text()))
    cfg = DirectConfig(**dict(c['model_cfg'], fine=CropSpec(**c['model_cfg']['fine'])))
    sample = SampleConfig(**dict(c['sample_cfg'], crop=cfg.fine))
    spec = FiberVolumeSpec(**c['vol_spec'])
    band = ZBand(*(v/spec.grid_scale for v in c['val_z']))
    fibers, _ = split_fibers(load_fibers(c['fibers'], grid_scale=spec.grid_scale), band)
    banks = {role: NeighborBank(c[role], fibers, band, grid_scale=spec.grid_scale,
                refresh_seconds=c['negative_bank_refresh_seconds'],
                cache_bytes=int(c['negative_bank_cache_mb']*(1 << 20)))
             for role in ('negative_bank', 'near_negative_bank', 'following_bank', 'continuation_bank')
             if c.get(role)}
    builder = IdentityObservationBuilder(cfg, fibers, IdentitySampling(**c['identity_sampling']),
                                         augment=True, **banks)
    dataset = FollowDataset(fibers, spec, sample, band, chunk=c['batch'], seed=args.seed,
        cache_bytes=int(c['worker_cache_gb']*(1 << 30)), batch_builder=builder,
        onpolicy=[OnPolicyStates.load(p) for p in c['onpolicy']], fresh_fraction=c['fresh_fraction'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    iterator = iter(loader)
    for _ in range(args.warmup):
        next(iterator)
    profiler = cProfile.Profile() if args.profile else None
    rows = []
    for i in range(args.batches):
        if profiler:
            profiler.enable()
        started = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter()-started
        if profiler:
            profiler.disable()
        row = dict(batch=i, seconds=elapsed)
        if args.hash_batches:
            row['hashes'] = tensor_hashes(batch)
        rows.append(row)
        print(f'{i+1}/{args.batches}: {elapsed:.3f}s', flush=True)
    if profiler:
        profiler.dump_stats(str(args.profile))
    values = [r['seconds'] for r in rows]
    report = dict(config=str(args.config.resolve()), workers=args.workers, seed=args.seed,
        warmup=args.warmup, batches=args.batches, batch=c['batch'],
        torch_version=torch.__version__, numpy_version=np.__version__,
        mean_seconds=float(np.mean(values)), p50_seconds=float(np.median(values)),
        p95_seconds=float(np.percentile(values, 95)),
        samples_per_second=c['batch']*args.batches/sum(values), rows=rows)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
