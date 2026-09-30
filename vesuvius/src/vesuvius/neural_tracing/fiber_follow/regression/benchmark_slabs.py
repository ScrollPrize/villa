"""Independent-decision training benchmark, excluding volume I/O and tracing quality."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .history_slabs import selected_arcs
from .model import DirectConfig, build_model
from .train import conv_memory_format, optimizer_update, prepare_training


def synthetic_decisions(cfg, count=6, seed=194):
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for j in range(count):
        age = (j+1)*184./count
        arcs = selected_arcs(age)
        valid = torch.arange(8)[None] < len(arcs)
        hist = torch.zeros(1, cfg.n_history, 3)
        hist[..., 2] = -torch.arange(1, cfg.n_history+1)
        pose = torch.zeros(1, 8, 14)
        pose[0, :len(arcs), 2] = torch.tensor(arcs-age)/128.
        pose[0, :len(arcs), 3:12] = torch.eye(3).flatten()
        pose[0, :len(arcs), 12] = torch.tensor(np.log1p(age-arcs)/8.)
        pose[0, 0, 13] = 1.
        ages = torch.zeros(1, 8)
        ages[0, :len(arcs)] = torch.tensor(age-arcs)
        x = dict(fine=torch.rand(1, cfg.input_channels, cfg.fine.depth, cfg.fine.width, cfg.fine.width, generator=rng),
                 seed=torch.zeros(1, 1, 3), seed_mask=torch.zeros(1, 1),
                 seed_tangent=torch.zeros(1, 3), seed_age=torch.zeros(1),
                 history_slabs=torch.rand(1, 8, 2, 8, 65, 65, generator=rng),
                 history_valid=valid, history_pose=pose, history_ages=ages,
                 history_overlap=torch.zeros(1, 8), history_load_seconds=torch.zeros(1))
        q = 4*(cfg.n_future-1)+1
        batch = dict(x=x, hist=hist, hmask=(torch.arange(cfg.n_history)[None] < age).float(),
            dense_ab=torch.zeros(1, q, 2), dense_mask=torch.ones(1, q),
            offtrack=torch.zeros(1), endpoint_known=torch.zeros(1), end_local=torch.zeros(1, 3), source=torch.zeros(1),
            foreign=torch.zeros(1, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8))
        # Same supplied-candidate count as the old six-decision / two-endpoint benchmark.
        if j >= count-2:
            curves = torch.zeros(1, 4, cfg.n_future, 3)
            curves[..., 2] = torch.arange(1, cfg.n_future+1)*cfg.future_step
            curves[..., 0] = torch.arange(4)[None, :, None]*.5
            batch.update(candidate_points=curves, candidate_mask=torch.ones(1, 4, cfg.n_future),
                         candidate_labels=torch.ones(1, 4, cfg.n_future))
        rows.append(batch)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--decisions', type=int, default=6)
    ap.add_argument('--seed', type=int, default=194)
    ap.add_argument('--encoder', choices=('conv', 'patch4'), default='patch4')
    ap.add_argument('--token-only', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.out.exists() or min(args.decisions, args.repeats, args.warmup) < 1:
        ap.error('Need a fresh output path and positive dimensions')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cfg = DirectConfig(direction_inputs=True, encoder=args.encoder, token_only=args.token_only)
    batches = synthetic_decisions(cfg, args.decisions, args.seed)
    model = build_model(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    prepare_training(model, 1)
    from torch._dynamo.utils import counters
    rows = []
    cuda = torch.device(args.device).type == 'cuda'
    for i in range(args.warmup+args.repeats):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        before = counters['stats']['unique_graphs']
        started = time.perf_counter()
        metrics = optimizer_update(model, ema, opt, batches, i+1, .0003, device=args.device,
                                   n_commit=cfg.n_future, compute_metrics=False)
        if cuda:
            torch.cuda.synchronize()
        row = dict(iteration=i+1, ms=1000*(time.perf_counter()-started), graphs=counters['stats']['unique_graphs']-before,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None,
                   **{k: metrics[k] for k in ('loss', 'grad_norm', 'history_grad_norm', 'history_valid_slabs_mean',
                                              'history_encode_seconds', 'supervised_states')})
        rows.append(row)
        print(json.dumps(row), flush=True)
    measured = [r for r in rows[args.warmup:] if not r['graphs']]
    if not measured:
        raise RuntimeError('No compilation-free measurements')
    times = [r['ms'] for r in measured]
    result = dict(config=cfg.to_dict(), hardware=torch.cuda.get_device_name() if cuda else 'CPU',
        torch=torch.__version__, seed=args.seed, precision='BF16 autocast / FP32 survival' if cuda else 'FP32',
        parameters=sum(p.numel() for p in model.parameters()),
        history_parameters=sum(p.numel() for p in model.history_encoder.parameters()),
        input='Synthetic full-size current crops and live CT/path slabs; no volume I/O',
        warmup=args.warmup, measured=len(measured), mean_ms=float(np.mean(times)),
        p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
        decisions_per_second=1000*sum(r['supervised_states'] for r in measured)/sum(times),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in measured) if cuda else None, samples=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
