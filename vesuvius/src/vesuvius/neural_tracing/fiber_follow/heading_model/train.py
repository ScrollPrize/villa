"""Train the crop-heading model on the follower's Paris 4 + AFV sources (see configs/heading_model.json).

Batches are generated online in loader workers from the follower's own sampling code (heading_model.data). The
run directory holds config.json, ct_normalization.json, log.jsonl, last.pt (resumable), ckpt_STEP.pt and best.pt
(lowest held-out p90 off-axis distance of the fiber over the follower crop's forward extent).

Usage (from fiber_follow/, or with -m from anywhere):
  python heading_model/train.py                      # configs/heading_model.json
  python heading_model/train.py --config configs/heading_model.json --name heading_model_v2
  python -m vesuvius.neural_tracing.fiber_follow.heading_model.train --resume
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from vesuvius.neural_tracing.fiber_follow.heading_model.data import (
    load_sources, mixed_heading_states, sampling_from_dict, sampling_to_dict, validation_states)
from vesuvius.neural_tracing.fiber_follow.heading_model.evaluate import (
    alignment_report, format_report, model_headings, summary_metric)
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingConfig, HeadingNet, save_heading_model
from vesuvius.neural_tracing.fiber_follow.data.remote_prefetch import attach_remote_prefetch
from vesuvius.neural_tracing.fiber_follow.train.runloop import raise_open_file_limit

DEFAULTS = dict(name='heading_model', steps=30000, batch=128, workers=8, lr=3e-3, weight_decay=1e-4, warmup=500,
                log_every=100, val_every=2000, ckpt_every=2000, val_states_per_source=1500, seed=0,
                worker_cache_gb=.5, device='cuda', model={}, sampling={}, ct_normalization=None, out_root=None,
                # The follower trainer's remote CT prefetch (separate async process; workers read the cache only).
                remote_prefetch_connections=48, remote_prefetch_queue_size=512, remote_prefetch_lookahead=16,
                remote_prefetch_timeout=120.)


PACKAGE = Path(__file__).resolve().parents[1]  # fiber_follow/
DEFAULT_CONFIG = PACKAGE/'configs'/'heading_model.json'


def config_path(path):
    """A config path given relative to the working directory, fiber_follow/, vesuvius/src/ or configs/."""
    path = Path(path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd()/path, PACKAGE/path, PACKAGE.parents[2]/path,
                                                    PACKAGE/'configs'/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Heading-model config not found: {path} (tried {", ".join(map(str, candidates))})')


def read_config(path, overrides=None):
    """Config JSON with defaults; relative paths inside it resolve against the config file's directory."""
    path = config_path(path)
    config = dict(DEFAULTS, **json.loads(path.read_text()))
    config.update(overrides or {})
    for key in ('dataset_config', 'ct_normalization', 'out_root'):
        if config.get(key) and '://' not in str(config[key]):
            config[key] = str((path.parent/config[key]).resolve())
    if not config.get('dataset_config'):
        raise ValueError('The heading-model config needs a dataset_config')
    out_root = config['out_root'] or str(Path(__file__).resolve().parents[1]/'output')
    config['run_dir'] = str(Path(out_root)/config['name'])
    return config


def learning_rate(step, config):
    """Linear warmup, then cosine decay to zero at the final step."""
    if step < config['warmup']:
        return config['lr']*(step+1)/config['warmup']
    progress = (step-config['warmup'])/max(1, config['steps']-config['warmup'])
    return config['lr']*.5*(1+math.cos(math.pi*min(1., progress)))


def validation_stats(reports, metric, best):
    """Compact held-out numbers for the progress bar: selection metric, and model angle at seeds / long paths."""
    def mean(bin_name):
        values = [r[bin_name]['angle_model'][0] for r in reports.values() if bin_name in r]
        return float(np.mean(values)) if values else float('nan')
    return dict(off90=metric, best=best, seed=mean('seed'), trace=mean('32+'))


def progress_text(train, val):
    text = (f"loss {train['loss']:.4f} | err {train['angle_deg']:.1f}\N{DEGREE SIGN} | lr {train['lr']:.1e} | "
            f"{train['samples_per_second']:.0f} samples/s, data wait {train['data_fraction']:.0%}") if train else ''
    if val:
        text += (f" || val off-axis p90 {val['off90']:.2f} (best {val['best']:.2f}) vox | median err seed "
                 f"{val['seed']:.1f}\N{DEGREE SIGN}, 32+ {val['trace']:.1f}\N{DEGREE SIGN}")
    return text


def validate(model, held_out, device):
    model.eval()
    reports = {name: alignment_report(states, model_headings(model, patch, path, [s['frame'] for s in states], device),
                                      model.cfg.forward)
               for name, (states, patch, path, _) in held_out.items()}
    model.train()
    return reports, summary_metric(reports)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT_CONFIG),
                    help='config JSON (default: configs/heading_model.json); relative to cwd, fiber_follow/ or configs/')
    ap.add_argument('--name')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--device')
    ap.add_argument('--resume', action='store_true', help='continue last.pt in the run directory')
    args = ap.parse_args(argv)
    # CT chunks are memory-mapped files; raise the descriptor limit before volumes or workers open them.
    raise_open_file_limit()
    overrides = {k: v for k, v in dict(name=args.name, steps=args.steps, workers=args.workers, device=args.device).items()
                 if v is not None}
    config = read_config(args.config, overrides)
    run = Path(config['run_dir'])
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError(f'{run} exists; use --resume or another --name')
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    cfg, sampling = HeadingConfig(**config['model']), sampling_from_dict(config['sampling'])
    print('Loading sources', flush=True)
    _, digest, sources, normalization = load_sources(config['dataset_config'], run,
                                                     ct_normalization=config['ct_normalization'],
                                                     ct_downsample_levels=cfg.ct_downsample_levels)
    for s in sources:
        print(f'  {s.name} ({s.kind}): {len(s.train)} training / {len(s.validation)} held-out fibers, weight {s.weight:g}', flush=True)
    print(f"Building {config['val_states_per_source']} held-out states per source", flush=True)
    with ThreadPoolExecutor(16) as pool:
        held_out = validation_states(sources, cfg, sampling, config['val_states_per_source'], config['seed'], pool)
    device = config['device']
    model = HeadingNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    step, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        if HeadingConfig(**checkpoint['config']).to_dict() != cfg.to_dict():
            raise ValueError(f'{run}/last.pt was trained with a different model config (inputs); use another --name')
        model.load_state_dict(checkpoint['state'])
        opt.load_state_dict(checkpoint['optimizer'])
        step, best = int(checkpoint['step']), float(checkpoint.get('best', best))
    provenance = dict(sampling=sampling_to_dict(sampling), dataset_config=config['dataset_config'],
                      dataset_config_sha256=digest, ct_normalization=normalization,
                      sources=[dict(name=s.name, kind=s.kind, weight=s.weight, train=len(s.train),
                                    validation=len(s.validation)) for s in sources])
    (run/'config.json').write_text(json.dumps(dict(config, model_cfg=cfg.to_dict(), **provenance), indent=2)+'\n')
    print(f'Model: {sum(p.numel() for p in model.parameters())} parameters; training from step {step}', flush=True)
    workers = config['workers']
    dataset, per_source = mixed_heading_states(sources, cfg, sampling, config['batch'], seed=config['seed']+step,
                                               cache_bytes=int(config['worker_cache_gb']*(1 << 30)))
    prefetcher, remote = attach_remote_prefetch(per_source, config['remote_prefetch_connections'],
                                                config['remote_prefetch_queue_size'], config['remote_prefetch_timeout'],
                                                config['remote_prefetch_lookahead'], workers)
    if prefetcher is not None:
        print(f"Remote CT prefetch for {len(remote)} sources: {config['remote_prefetch_connections']} connections, "
              f"{config['remote_prefetch_lookahead']} planned batches per source/worker", flush=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
                                         pin_memory=device.startswith('cuda'), persistent_workers=workers > 0,
                                         prefetch_factor=4 if workers else None)
    try:
        train(config, run, model, opt, step, best, loader, held_out, provenance, device)
    finally:
        if prefetcher is not None:
            prefetcher.close()


def train(config, run, model, opt, step, best, loader, held_out, provenance, device):
    batches = iter(loader)
    interval = dict(loss=0., angle=0., n=0, data=0., started=time.monotonic())
    train_stats = val_stats = None
    reports = {}
    bar = tqdm(total=config['steps'], initial=step, desc=config['name'], unit='step', dynamic_ncols=True, smoothing=.05)
    with open(run/'log.jsonl', 'a') as log:
        while step < config['steps']:
            began = time.monotonic()
            batch = next(batches)
            interval['data'] += time.monotonic()-began
            for group in opt.param_groups:
                group['lr'] = learning_rate(step, config)
            pred = model(batch['patch'].to(device, non_blocking=True), batch['path'].to(device, non_blocking=True))
            cosine = (pred*batch['target'].to(device, non_blocking=True)).sum(-1)
            loss = (1-cosine).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            bar.update(1)
            interval['loss'] += loss.item()
            interval['angle'] += float(torch.rad2deg(torch.arccos(cosine.detach().clamp(-1, 1))).mean())
            interval['n'] += 1
            if step % config['log_every'] == 0:
                elapsed = time.monotonic()-interval['started']
                row = dict(step=step, lr=opt.param_groups[0]['lr'], loss=interval['loss']/interval['n'],
                           angle_deg=interval['angle']/interval['n'],
                           samples_per_second=interval['n']*config['batch']/elapsed, data_fraction=interval['data']/elapsed)
                log.write(json.dumps(row)+'\n'); log.flush()
                train_stats = row
                bar.set_postfix_str(progress_text(train_stats, val_stats))
                interval = dict(loss=0., angle=0., n=0, data=0., started=time.monotonic())
            if step % config['val_every'] == 0 or step == config['steps']:
                reports, metric = validate(model, held_out, device)
                log.write(json.dumps(dict(step=step, split='validation', metric=metric, reports=reports))+'\n'); log.flush()
                if metric < best:
                    best = metric
                    save_heading_model(run/'best.pt', model, step=step, metric=metric, **provenance)
                val_stats = validation_stats(reports, metric, best)
                bar.set_postfix_str(progress_text(train_stats, val_stats))
            if step % config['ckpt_every'] == 0 or step == config['steps']:
                save_heading_model(run/f'ckpt_{step:06d}.pt', model, step=step, **provenance)
                save_heading_model(run/'last.pt', model, step=step, best=best, optimizer=opt.state_dict(), **provenance)
    bar.close()
    for name, report in reports.items():
        print(format_report(f'{name} (held-out) at step {step}', report), flush=True)
    print(f'Done: {run}/best.pt (held-out crop off-axis p90 {best:.3f} voxels); per-validation details in {run}/log.jsonl',
          flush=True)


if __name__ == '__main__':
    main()
