"""Bounded full-size synthetic observation-stream benchmark, with fresh weights.

Measures complete fixed streams, including all image observations, sparse task
losses, chronological writer replay, selected historical encoder gradients,
AdamW, clipping and EMA. Excludes volume I/O and does not measure tracing quality.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .feature_sequences import decision_plan
from .model import DirectConfig, build_model
from .train import conv_memory_format, optimizer_update, prepare_training


def synthetic_streams(cfg, length, streams, seed):
    """Fixed images and causal poses, with four candidates at each endpoint."""
    rng = torch.Generator().manual_seed(seed)
    plans = [decision_plan(length, cfg, np.random.default_rng(seed+j)) for j in range(streams)]
    rows = []
    for t in range(length):
        hist = torch.zeros(streams, cfg.n_history, 3)
        hist[..., 2] = -torch.arange(1, cfg.n_history+1)
        curves = torch.zeros(streams, 4, cfg.n_future, 3)
        curves[..., 2] = torch.arange(1, cfg.n_future+1)*cfg.future_step
        curves[..., 0] = torch.arange(4)[None, :, None]*.5
        mask = torch.full((streams, 4, cfg.n_future), t == length-1)
        pose = torch.zeros(streams, 3)
        pose[:, 2] = t*cfg.memory_stride
        seed_pos = torch.zeros(streams, 1, 3)
        seed_pos[..., 2] = -t*cfg.memory_stride
        x = dict(fine=torch.rand(streams, cfg.input_channels, cfg.fine.depth, cfg.fine.width,
                                 cfg.fine.width, generator=rng),
                 seed=seed_pos, seed_mask=torch.ones(streams, 1),
                 seed_tangent=torch.tensor([0., 0., 1.]).expand(streams, -1).clone(),
                 seed_age=torch.full((streams,), float(t*cfg.memory_stride)),
                 query_frame=torch.eye(3).expand(streams, -1, -1).clone(), query_position=pose,
                 feature_seed_here=torch.full((streams,), t == 0))
        q = 4*(cfg.n_future-1)+1
        rows.append(dict(x=x, hist=hist, hmask=(torch.arange(cfg.n_history)[None] < t*cfg.memory_stride).expand(streams, -1).float(),
            dense_ab=torch.zeros(streams, q, 2), dense_mask=torch.ones(streams, q),
            offtrack=torch.zeros(streams), endpoint_known=torch.zeros(streams), end_local=torch.zeros(streams, 3),
            source=torch.zeros(streams),
            foreign=torch.zeros(streams, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8),
            candidate_points=curves, candidate_mask=mask, candidate_labels=torch.ones_like(mask).float(),
            stream_id=torch.arange(streams), stream_reset=torch.full((streams,), t == 0),
            stream_end=torch.full((streams,), t == length-1), stream_index=torch.full((streams,), t),
            decision_mask=torch.tensor([plan[0][t] > 0 for plan in plans]),
            loss_weight=torch.tensor([plan[0][t] for plan in plans], dtype=torch.float32),
            encoder_indices=torch.tensor(np.stack([plan[1][t] for plan in plans])),
            retain_until=torch.tensor([plan[2][t] for plan in plans])))
    return [dict(feature_sequence=rows[t:t+cfg.feature_sequence_length])
            for t in range(0, length, cfg.feature_sequence_length)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--save-fixture', type=Path)
    ap.add_argument('--length', type=int, default=16)
    ap.add_argument('--streams', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seed', type=int, default=194)
    ap.add_argument('--history-decisions', type=int, default=2)
    args = ap.parse_args()
    if min(args.length, args.streams, args.warmup, args.repeats) < 1 or args.history_decisions < 0:
        ap.error('Positive dimensions/repetitions and nonnegative history decisions required')
    if args.out.exists():
        ap.error('Use a fresh result path')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cfg = DirectConfig(direction_inputs=True, feature_history_decisions=args.history_decisions)
    chunks = synthetic_streams(cfg, args.length, args.streams, args.seed)
    if args.save_fixture:
        if args.save_fixture.exists():
            ap.error('Use a fresh fixture path')
        args.save_fixture.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=cfg.to_dict(), chunks=chunks, seed=args.seed), args.save_fixture)
    model = build_model(cfg).to('cuda', memory_format=conv_memory_format('cuda'))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    prepare_training(model, args.streams)
    from torch._dynamo.utils import counters
    samples = []
    for i in range(args.warmup+args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = counters['stats']['unique_graphs']
        started = time.perf_counter()
        metrics = optimizer_update(model, ema, opt, chunks, i+1, .0003, device='cuda',
                                   n_commit=cfg.n_future, compute_metrics=False)
        torch.cuda.synchronize()
        row = dict(iteration=i+1, ms=1000*(time.perf_counter()-started),
            graphs=counters['stats']['unique_graphs']-before,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            **{k: metrics[k] for k in ('loss', 'grad_norm', 'observed_states', 'supervised_states',
                                      'endpoint_states', 'history_encoder_crops', 'memory_replay_observations')})
        samples.append(row)
        print(json.dumps(row), flush=True)
    steady = [r for r in samples[args.warmup:] if not r['graphs']]
    if not steady:
        raise RuntimeError('No compilation-free measurements; increase warmup/repetitions')
    times = [r['ms'] for r in steady]
    seconds = sum(times)/1000
    result = dict(config=cfg.to_dict(), hardware=torch.cuda.get_device_name(), torch=torch.__version__,
        seed=args.seed, input='Fixed synthetic full-size images; complete causal streams; candidates at endpoints',
        precision='Existing BF16 autocast / FP32 memory and survival policy',
        compiled=True, warmup=args.warmup, measured=len(steady),
        mean_ms=float(np.mean(times)), p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
        min_ms=min(times), max_ms=max(times),
        observations_per_second=sum(r['observed_states'] for r in steady)/seconds,
        decisions_per_second=sum(r['supervised_states'] for r in steady)/seconds,
        endpoints_per_second=sum(r['endpoint_states'] for r in steady)/seconds,
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in steady), samples=samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples', 'config')}, indent=2))


if __name__ == '__main__':
    main()
