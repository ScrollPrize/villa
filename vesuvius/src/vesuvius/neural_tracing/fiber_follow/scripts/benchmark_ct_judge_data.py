"""Profile real CT-judge batch preparation without loading models or changing a run.

Use --workers 0 for a CPU profile; --workers 4 measures the production loader.
Reports tensor hashes so runs with the same worker count can check exact inputs.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import (
    FollowDataset, OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.direct.data import ObservationBuilder
from vesuvius.neural_tracing.fiber_follow.direct.judge_options import configs
from vesuvius.neural_tracing.fiber_follow.direct.judge_supervision import JointObservationBuilder
from vesuvius.neural_tracing.fiber_follow.direct.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.direct.train import raise_open_file_limit
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def tensor_digest(value, digest):
    if isinstance(value, dict):
        for key, item in value.items():
            digest.update(key.encode())
            tensor_digest(item, digest)
    elif isinstance(value, list):
        for item in value:
            tensor_digest(item, digest)
    else:
        digest.update(value.numpy().tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='JSON report; profile saved alongside it')
    ap.add_argument('--count', type=int, default=32, help='Number of microbatches')
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()
    if args.count < 1 or args.workers < 0:
        ap.error('Count must be positive and workers nonnegative')
    c = json.loads(args.config.read_text())
    if not c['judge']:
        ap.error('Config must enable the CT judge')
    raise_open_file_limit()
    torch.set_num_threads(1)
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    cfg = DirectConfig(**c['model_cfg'])
    spec = FiberVolumeSpec(**c['vol_spec'])
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                          future_step=cfg.future_step, recent_history_points=cfg.n_history,
                          no_history_prob=c['no_history_prob'], short_history_prob=c['short_history_prob'])
    band = ZBand(*(v/spec.grid_scale for v in c['val_z']))
    print('Loading training annotations and replay banks', flush=True)
    fibers, _ = split_fibers(load_fibers(c['fibers'], grid_scale=spec.grid_scale), band)
    slices, _, _ = configs(argparse.Namespace(**c), spec)
    builder = JointObservationBuilder(ObservationBuilder(cfg), slices, band, c['judge_synthetic_fraction'], fibers)
    dataset = FollowDataset(fibers, spec, sample, band, chunk=c['microbatch'], seed=c['seed'],
        cache_bytes=int(c['worker_cache_gb']*(1 << 30)),
        fixed=[OnPolicyStates.load(c['fixed_bank'])] if c['fixed_bank'] else [],
        onpolicy=[OnPolicyStates.load(p) for p in c.get('onpolicy', [])],
        batch_builder=builder, additional_crops=(cfg.coarse,))
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers,
        **dict(prefetch_factor=2, persistent_workers=True) if args.workers else {})
    start = time.perf_counter()
    iterator = iter(loader)
    startup_seconds = time.perf_counter()-start
    profile = cProfile.Profile()
    rows = []
    for i in range(args.count):
        start = time.perf_counter()
        profile.enable()
        batch = next(iterator)
        profile.disable()
        elapsed = time.perf_counter()-start
        digest = hashlib.sha256()
        tensor_digest(batch, digest)
        row = dict(microbatch=i, seconds=elapsed, sha256=digest.hexdigest(),
                   judge_lengths=[int(s['images'].shape[1]) for s in batch['judge']])
        rows.append(row)
        print(json.dumps(row), flush=True)
    values = np.array([r['seconds'] for r in rows])
    per_update = c['batch']//c['microbatch']
    updates = [float(values[i:i+per_update].sum()) for i in range(0, len(values)-per_update+1, per_update)]
    report = dict(config=str(args.config), workers=args.workers, rows=rows,
                  loader_startup_seconds=startup_seconds, update_data_seconds=updates,
                  mean=float(values.mean()), p50=float(np.median(values)), p95=float(np.quantile(values, .95)),
                  min=float(values.min()), max=float(values.max()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    profile.dump_stats(str(args.out)+'.prof')
    pstats.Stats(profile).sort_stats('cumtime').print_stats(25)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
