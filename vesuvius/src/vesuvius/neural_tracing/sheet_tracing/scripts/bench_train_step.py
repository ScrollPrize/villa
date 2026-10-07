"""Time training iterations of a config on fixed batches: host-to-device copy (``train.device_batches``) plus
``train.optimize_step`` (rollout, backward, clip, optimizer, EMA), without the data loader's variance.

Draws ``--batches`` training batches once (loader workers, like training), pins them, then runs ``--warmup`` +
``--iterations`` iterations cycling through them and prints seconds per iteration (mean, p50, p95, min, max) and
the loss of the first iteration (same weights, batch and generator seed every run, before any update: compare it
across code changes for numerical equality). ``--profile`` adds torch.profiler tables (top ops by CUDA time and by
host time) over three iterations. Run it with nothing else on the GPU; host load from other processes shows up in
the timings, since the step is close to launch-bound.

  python scripts/bench_train_step.py --config remote_l0_joint.json --init output/l0_joint/last.pt
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch

from vesuvius.neural_tracing.sheet_tracing import config as C
from vesuvius.neural_tracing.sheet_tracing.data.episodes import EpisodeSampler, EpisodeStream, collate, episode_pad
from vesuvius.neural_tracing.sheet_tracing.data.patches import build_index
from vesuvius.neural_tracing.sheet_tracing.model import build_model, compile_model, init_model_state
from vesuvius.neural_tracing.sheet_tracing.rollout import Tracer
from vesuvius.neural_tracing.sheet_tracing.train import EMA, _pinned, device_batches, optimize_step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.FIELD=VALUE')
    ap.add_argument('--init', help='model weights (EMA if saved)')
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--iterations', type=int, default=10)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--stream', action='store_true', help='draw every batch from the loader, as training does '
                    '(end-to-end throughput, with the loader and pinning times), instead of cycling fixed batches')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    cfg = C.load(args.config, args.set)
    tc, d = cfg.train, cfg.data
    device = torch.device(args.device)
    torch.manual_seed(tc.seed)
    if tc.compile_geometry:
        from vesuvius.neural_tracing.sheet_tracing.geometry import use_compiled_projection
        use_compiled_projection()

    sampler = EpisodeSampler(d, build_index(d.patches_dir, C.OUTPUT/'cache'), 'train', d.steps, d.roi,
                             episode_pad(cfg.model, tc))
    loader = torch.utils.data.DataLoader(EpisodeStream(sampler, tc.seed), batch_size=tc.batch,
                                         num_workers=tc.workers, collate_fn=collate,
                                         prefetch_factor=tc.prefetch if tc.workers else None)
    timings = {}
    if args.stream:
        stream = device_batches(loader, device, timings=timings, pad_blocks=not d.unpadded_blocks)
    else:
        t0 = time.time()
        batches = [b for b, _ in zip(loader, range(args.batches))]
        del loader
        batches = [_pinned(b) for b in batches]
        print(f'{len(batches)} batches of {tc.batch} in {time.time()-t0:.0f} s; block GB '
              f'{[round(b["block"]["data"].numel()/1e9, 2) for b in batches]}', flush=True)
        stream = device_batches(itertools.cycle(batches), device, pad_blocks=not d.unpadded_blocks)

    model = build_model(cfg).to(device)
    if args.init:
        saved = torch.load(args.init, map_location='cpu', weights_only=False)
        init_model_state(model, saved.get('ema') or saved['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    for group in opt.param_groups:
        group['lr'] = tc.lr*.1
    ema = EMA(model, tc.ema) if tc.ema else None
    if tc.compile_model:
        for net in (model, ema.model) if ema else (model,):
            compile_model(net)
    tracer = Tracer(model, cfg, device)
    generator = torch.Generator(device=device)

    def iteration(i):
        generator.manual_seed(tc.seed+i)
        b = next(stream)
        loss, *_ = optimize_step(tracer, opt, ema, b, cfg, sampler.spec, generator)
        return loss

    times, first = [], None
    for i in range(args.warmup+args.iterations):
        if i == args.warmup:
            timings.clear()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = iteration(i)
        torch.cuda.synchronize(device)
        first = float(loss) if first is None else first
        if i >= args.warmup:
            times.append(time.perf_counter()-start)
    t = np.array(times)
    print(f's/it mean {t.mean():.3f} p50 {np.median(t):.3f} p95 {np.percentile(t, 95):.3f} min {t.min():.3f} '
          f'max {t.max():.3f} over {len(t)} iterations; loss of the first iteration {first!r}; '
          f'peak memory {torch.cuda.max_memory_allocated(device)/2**30:.1f} GiB', flush=True)
    if timings:
        print('per measured iteration: ' + ', '.join(f'{k} {v/len(t):.3f} s' for k, v in timings.items())
              + ' (background thread)', flush=True)
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in itertools.islice(itertools.count(), 3):
                iteration(i)
            torch.cuda.synchronize(device)
        for key in ('cuda_time_total', 'self_cpu_time_total'):
            print(prof.key_averages().table(sort_by=key, row_limit=30, max_name_column_width=60))


if __name__ == '__main__':
    main()
