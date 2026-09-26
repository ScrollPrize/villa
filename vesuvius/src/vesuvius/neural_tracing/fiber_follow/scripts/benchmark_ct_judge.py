"""Compute-only training benchmark of the CT judge plan, not a production judge.

Example (from fiber_follow):
  PYTHONPATH=../../.. python scripts/benchmark_ct_judge.py --out /tmp/ct-judge-bench.json

Uses repeated preview CT for full-size judge tensors and synthetic follower
inputs/targets. Measures no CT I/O, augmentation, collection, or model accuracy.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.direct.model import DirectConfig, DirectFollower
from vesuvius.neural_tracing.fiber_follow.direct.supervision import loss_terms


class JudgePrototype(nn.Module):
    def __init__(self, positions=33, view_batch=33, encoder_variant='baseline', center_sampler='grid'):
        super().__init__()
        self.positions, self.view_batch = positions, view_batch
        if encoder_variant not in ('baseline', 'feature_pool2', 'dual_scale'):
            raise ValueError('Unknown encoder variant')
        self.encoder_variant = encoder_variant
        if center_sampler not in ('grid', 'index'):
            raise ValueError('Unknown center sampler')
        self.center_sampler = center_sampler

        def stage(a, b, stride):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(math.gcd(8, b), b), nn.SiLU(),
                nn.Conv2d(b, b, 3, padding=1, bias=False),
                nn.GroupNorm(math.gcd(8, b), b), nn.SiLU())

        self.local = stage(3, 16, 1)
        self.down = nn.Sequential(stage(16, 32, 2), stage(32, 64, 2))
        self.full_projection = nn.Linear(64, 128)
        self.center_projection = nn.Linear(16, 128)
        self.metadata = nn.Linear(20, 128)
        self.query = nn.Linear(20, 128)
        layer = nn.TransformerDecoderLayer(128, 4, 256, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(128))
        self.head = nn.Linear(128, 1)
        offsets = torch.arange(-2, 3).float() * 8 / 128
        yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('center_grid', torch.stack((xx, yy), -1)[None])

    def forward(self, images, metadata, query_metadata):
        batch = images.shape[0]
        flat = images.reshape(-1, 3, 257, 257)
        tokens = []
        for chunk in flat.split(self.view_batch):
            chunk = chunk.contiguous(memory_format=torch.channels_last)
            if self.encoder_variant == 'dual_scale':
                # Keep native center pixels; low-pass and decimate broad context.
                # Odd 3x3 pooling preserves the image-center lattice (129x129).
                local = self.local(chunk[:, :, 96:161, 96:161].contiguous(memory_format=torch.channels_last))
                context = self.local(F.avg_pool2d(chunk, 3, 2, 1, count_include_pad=False))
                grid = self.center_grid * 4  # 65-pixel crop, same physical offsets
            else:
                local = self.local(chunk)
                context = (F.avg_pool2d(local, 3, 2, 1, count_include_pad=False)
                           if self.encoder_variant == 'feature_pool2' else local)
                grid = self.center_grid
            full = F.adaptive_avg_pool2d(self.down(context), (8, 8)).flatten(2).transpose(1, 2)
            if self.center_sampler == 'index':
                # All fixed center-grid positions lie exactly on feature pixels.
                # Select first; avoid converting the entire native map to FP32.
                mid = local.shape[-1] // 2
                center = local[:, :, mid-16:mid+17:8, mid-16:mid+17:8].float()
            else:
                center = F.grid_sample(local.float(), grid.expand(len(chunk), -1, -1, -1), align_corners=True)
            center = center.flatten(2).transpose(1, 2)
            tokens.append(torch.cat((self.full_projection(full), self.center_projection(center)), 1))
        memory = torch.cat(tokens).reshape(batch, self.positions * 3 * 89, 128)
        memory = memory + self.metadata(metadata)
        return self.head(self.decoder(self.query(query_metadata), memory)).squeeze(-1).float()


def follower_batch(cfg, count, device):
    x = {name: torch.rand(count, 2, crop.depth, crop.width, crop.width, device=device)
         for name, crop in [('fine', cfg.fine), ('coarse', cfg.coarse)]}
    hist = torch.zeros(count, cfg.n_history, 3, device=device)
    hist[..., 2] = -torch.arange(1, cfg.n_history + 1, device=device)
    return dict(x=x, hist=hist, hmask=torch.ones(count, cfg.n_history, device=device),
                dense_ab=torch.zeros(count, 61, 2, device=device),
                dense_mask=torch.ones(count, 61, device=device),
                offtrack=torch.zeros(count, device=device),
                endpoint_known=torch.zeros(count, device=device),
                end_local=torch.zeros(count, 3, device=device))


def judge_batch(path, count, positions, device):
    with np.load(path) as source:
        indices = np.arange(positions) % len(source['ct'])
        ct = source['ct'][indices]
        marker = np.broadcast_to(source['marker'], ct.shape)
        support = source['support'][indices].astype(np.float32)
        images = np.stack((ct, marker, support), axis=2)
    images = torch.from_numpy(images).to(device)[None].repeat(count, 1, 1, 1, 1, 1)
    # Placeholder metadata has the planned dimensions, but is not real path supervision.
    metadata = torch.randn(count, positions * 3 * 89, 20, device=device)
    queries = torch.randn(count, positions, 20, device=device)
    labels = (torch.arange(positions, device=device) < positions // 2).float()[None].expand(count, -1)
    return images, metadata, queries, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--encoder-variant', choices=['baseline', 'feature_pool2', 'dual_scale'], default='baseline')
    ap.add_argument('--center-sampler', choices=['grid', 'index'], default='grid')
    ap.add_argument('--positions', type=int, default=33)
    ap.add_argument('--view-batch', type=int, default=33)
    ap.add_argument('--microbatch', type=int, default=2)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--updates', type=int, default=10)
    ap.add_argument('--cases', nargs='+', choices=['follower', 'judge', 'combined'],
                    default=['follower', 'judge', 'combined'])
    ap.add_argument('--slices', type=Path, default=Path(__file__).resolve().parents[1] /
                    'direct/ct_slice_judge_examples/slices.npz')
    args = ap.parse_args()
    if args.batch % args.microbatch or (args.batch + args.batch // 4) % args.microbatch:
        raise ValueError('Microbatch must divide follower and combined judge batch counts')
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA; CPU timing is not a GPU substitute')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = 'cuda'
    cfg = DirectConfig()
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), precision='BF16 autocast; FP32 weights',
                  compiled=args.compile, encoder_variant=args.encoder_variant, center_sampler=args.center_sampler,
                  positions=args.positions, views_per_position=3,
                  view_batch=args.view_batch, microbatch=args.microbatch, batch=args.batch,
                  warmup=args.warmup, updates=args.updates, seed=0, cpu_threads=4,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  slices=str(args.slices),
                  scope='GPU-resident compute; synthetic follower; repeated preview CT; no I/O',
                  cases={})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        print(f'Starting {case}, compiled={args.compile}', flush=True)
        models, forwards, opts, emas = {}, {}, {}, {}
        if case in ('follower', 'combined'):
            models['follower'] = DirectFollower(cfg).to(device, memory_format=torch.channels_last_3d)
        if case in ('judge', 'combined'):
            models['judge'] = JudgePrototype(args.positions, args.view_batch, args.encoder_variant,
                                            args.center_sampler).to(device, memory_format=torch.channels_last)
        for name, model in models.items():
            forwards[name] = torch.compile(model) if args.compile else model
            opts[name] = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            emas[name] = copy.deepcopy(model).requires_grad_(False)
        fb = follower_batch(cfg, args.microbatch, device) if 'follower' in models else None
        jb = judge_batch(args.slices, args.microbatch, args.positions, device) if 'judge' in models else None
        judge_count = args.batch + args.batch // 4 if case == 'combined' else args.batch

        def update():
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            if fb is not None:
                for _ in range(args.batch // args.microbatch):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = forwards['follower'](fb['x'], fb['hist'], fb['hmask'])
                        terms = loss_terms(out, fb, cfg, n_commit=8)
                        loss = (terms['geometry_per_state'] + .5 * terms['confidence_per_state']).sum() / args.batch
                    loss.backward()
            if jb is not None:
                images, metadata, queries, labels = jb
                for _ in range(judge_count // args.microbatch):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = forwards['judge'](images, metadata, queries)
                        loss = .5 * F.binary_cross_entropy_with_logits(logits, labels) * args.microbatch / judge_count
                    loss.backward()
            for name, model in models.items():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opts[name].step()
                with torch.no_grad():
                    for ep, p in zip(emas[name].parameters(), model.parameters()):
                        ep.lerp_(p, .001)

        warmup, elapsed = [], []
        for i in range(args.warmup + args.updates):
            if i == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            update()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            (warmup if i < args.warmup else elapsed).append(seconds)
            if i < args.warmup:
                print(f'  warmup {i + 1}: {seconds:.3f}s', flush=True)
        stats = dict(mean_ms=float(np.mean(elapsed) * 1000),
                     p50_ms=float(np.median(elapsed) * 1000),
                     p95_ms=float(np.percentile(elapsed, 95) * 1000),
                     peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                     follower_states=args.batch if fb is not None else 0,
                     judge_states=judge_count if jb is not None else 0,
                     warmup_seconds=warmup, update_seconds=elapsed)
        result['cases'][case] = stats
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(case=case, **stats)), flush=True)
        del update, model, forwards, models, opts, emas, fb, jb
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
