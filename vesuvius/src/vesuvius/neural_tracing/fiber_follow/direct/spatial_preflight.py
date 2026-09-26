"""Bounded real-data geometry, prompt fitting, and full-input resource checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..data import load_fibers, split_fibers, ZBand, SampleConfig
from ..volume import FiberVolume, FiberVolumeSpec
from .contacts import ContactIndex, SpatialObservationBuilder
from .spatial_model import SpatialConfig, SpatialFollower
from .spatial_supervision import spatial_loss_terms, teaching_candidates, plane_mask
from .train import move_batch, conv_memory_format


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', default='output/direct_ct_judge_run2/config.json')
    ap.add_argument('--out', default='output/direct_spatial_preflight')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fit-steps', type=int, default=100)
    ap.add_argument('--microbatch', type=int, default=8, help='Additional full-input training memory/throughput check')
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.baseline).read_text())
    cfg = SpatialConfig()
    sample = SampleConfig(crop=cfg.fine, n_future=cfg.n_future, n_history=cfg.n_history,
                          no_history_prob=.15, short_history_prob=.4, unique_crossings=True)
    spec = FiberVolumeSpec(base['fiber_zarrs'], ct_zarr=base['ct'], ct_level=0, ct_grid_scale=4.,
                           inputs='ct', load_presence=False)
    band = ZBand(*(v/spec.grid_scale for v in base['val_z']))
    fibers, _ = split_fibers(load_fibers(base['fibers']), band)
    contacts = ContactIndex(fibers, band, out/'contacts.json')
    builder = SpatialObservationBuilder(cfg, sample, fibers, band, contacts, contact_fraction=1.)
    summary = contacts.summary(cfg.n_future)
    print('Contact index', json.dumps(summary), flush=True)
    all_items, fitted_pairs = [], {}
    for _ in range(128):
        items = builder.contact_batch(rng, 2)
        all_items.extend(items)
        if len(items) == 2:
            masks = [i['plane_mask'].astype(bool) & (np.abs(i['plane_ab']).max(-1) < cfg.lateral_limit-1) for i in items]
            geometry_ok = all(np.max(np.linalg.norm(np.diff(i['plane_ab'], axis=0), axis=-1)) < cfg.lateral_step-0.3 for i in items)
            separation = np.mean(np.linalg.norm(items[0]['plane_ab']-items[1]['plane_ab'], axis=-1))
            direction = bool(items[0]['contact_reverse'])
            if direction not in fitted_pairs and all(m.all() for m in masks) and geometry_ok and separation > 1.5:
                fitted_pairs[direction] = items
    if len(fitted_pairs) != 2:
        raise RuntimeError('No fully supported prompt pairs in both directions for fitting')
    fitted = fitted_pairs[False]+fitted_pairs[True]
    if not all_items:
        raise RuntimeError('Contact sampler produced no usable states')
    known = np.stack([i['plane_mask'] for i in all_items]).astype(bool)
    supported = np.stack([np.abs(i['plane_ab']).max(-1) <= cfg.lateral_limit for i in all_items])
    summary.update(sampled_states=len(all_items), unique_plane_fraction=float(known.mean()),
                   complete_unique_fraction=float(known.all(-1).mean()),
                   complete_in_crop_fraction=float((known & supported).all(-1).mean()),
                   lateral_excursion_p95=float(np.quantile(np.concatenate([np.abs(i['plane_ab']).max(-1)[i['plane_mask'] > 0] for i in all_items]), .95)))
    summary['contact_sampling_counts'] = dict(builder.sampling_counts)
    vol = FiberVolume(spec, cache_bytes=1 << 30)
    began = time.perf_counter()
    cpu = builder(fitted, vol)
    read_seconds = time.perf_counter()-began
    assert vol.presence is None and builder._coarse.presence is None
    assert torch.equal(cpu['x']['fine'][0], cpu['x']['fine'][1])
    assert torch.equal(cpu['x']['coarse'][0], cpu['x']['coarse'][1])
    data = move_batch(cpu, args.device)
    model = SpatialFollower(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    is_cuda = torch.device(args.device).type == 'cuda'
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    losses, times, errors = [], [], []
    model.train()
    for step in range(args.fit_steps):
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=is_cuda):
            prediction = model(data['x'], data['hist'], data['hmask'], teaching_candidates(data, cfg))
            terms = spatial_loss_terms(prediction, data, cfg, n_commit=4, compute_metrics=step == args.fit_steps-1)
            loss = terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite preflight loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter()-began)
        losses.append(loss.item())
        peak = prediction['heatmap_logits'].flatten(2).argmax(-1)
        error = (model.lateral_grid[peak]-data['plane_ab']).norm(dim=-1)
        errors.append(error.mean().item())
        if step % 10 == 0 or step == args.fit_steps-1:
            print(f'Fit {step+1}/{args.fit_steps}: loss={losses[-1]:.4f}, peak_error={errors[-1]:.3f}, seconds={times[-1]:.3f}', flush=True)
    train_peak = torch.cuda.max_memory_allocated() if is_cuda else None
    if not losses[-1] < losses[0] or errors[-1] > 1.5:
        raise RuntimeError(f'Prompt pair did not fit: initial/final loss {losses[0]:.3f}/{losses[-1]:.3f}, error {errors[-1]:.3f}')
    model.eval()
    timings = []
    with torch.no_grad():
        for _ in range(6):
            began = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=is_cuda):
                prediction = model(data['x'], data['hist'], data['hmask'])
            if is_cuda:
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-began)
        metrics = spatial_loss_terms(prediction, data, cfg)['spatial_metrics']
        labels = model.lateral_grid[prediction['heatmap_logits'].flatten(2).argmax(-1)]
        own = (labels-data['plane_ab']).norm(dim=-1).mean(-1)
        swapped = (labels-data['plane_ab'][torch.tensor([1, 0, 3, 2], device=labels.device)]).norm(dim=-1).mean(-1)
        if not (own < swapped).all():
            raise RuntimeError('Fixed-crop prompt swap failed to distinguish the two fibers')
    # Benchmark the intended training microbatch, including real CT reads,
    # search, teacher candidates, backward, and the optimizer. Model weights
    # from this bounded check are never used to initialize production training.
    if args.microbatch < 2 or args.microbatch % 2:
        raise ValueError('Preflight microbatch must be an even number >= 2')
    full_items = builder.contact_batch(rng, args.microbatch)
    if len(full_items) != args.microbatch:
        raise RuntimeError('Could not fill the full-input contact microbatch')
    began = time.perf_counter()
    full_cpu = builder(full_items, vol)
    full_read_seconds = time.perf_counter()-began
    full = move_batch(full_cpu, args.device)
    del prediction, terms, loss
    optimizer.zero_grad(set_to_none=True)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    full_times = []
    model.train()
    for _ in range(6):
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=is_cuda):
            prediction = model(full['x'], full['hist'], full['hmask'], teaching_candidates(full, cfg))
            terms = spatial_loss_terms(prediction, full, cfg, compute_metrics=False)
            loss = terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize()
        full_times.append(time.perf_counter()-began)
    full_peak = torch.cuda.max_memory_allocated() if is_cuda else None
    report = dict(status='passed', config=cfg.to_dict(), sample=asdict(sample), volume=spec.to_dict(),
                  contact_audit=summary, fit_steps=args.fit_steps, fit_initial_loss=losses[0], fit_final_loss=losses[-1],
                  initial_peak_error=errors[0], final_peak_error=errors[-1], prompt_own_error=own.tolist(),
                  prompt_swapped_error=swapped.tolist(), fit_metrics={k: v.item() for k, v in metrics.items()},
                  microbatch=len(fitted), precision='bf16 autocast' if is_cuda else 'float32',
                  peak_training_allocated_bytes=train_peak, crop_read_seconds=read_seconds,
                  training_update_mean_seconds=float(np.mean(times[5:])),
                  training_samples_per_second=len(fitted)/float(np.mean(times[5:])),
                  inference_batch_mean_seconds=float(np.mean(timings[1:])),
                  inference_batch_p95_seconds=float(np.quantile(timings[1:], .95)),
                  full_microbatch=args.microbatch, full_crop_read_seconds=full_read_seconds,
                  full_peak_training_allocated_bytes=full_peak,
                  full_training_update_mean_seconds=float(np.mean(full_times[1:])),
                  full_training_samples_per_second=args.microbatch/float(np.mean(full_times[1:])))
    (out/'report.json').write_text(json.dumps(report, indent=2))
    torch.save(dict(config=cfg.to_dict(), state=model.state_dict()), out/'fit_check.pt')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
