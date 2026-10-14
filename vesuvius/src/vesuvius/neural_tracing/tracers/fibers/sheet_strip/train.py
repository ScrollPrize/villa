"""Fine-tune a fiber follower checkpoint into the sheet-strip follower (sheet_strip/model.py).

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.train --config configs/sheet_strip/paris4_slim50k.json
    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.train --config RUN/run.json --resume RUN/last.pt

The model starts from the checkpoint's EMA weights: its crop CNNs gain the tube input channel with zero weights and the
strip modules start fresh (``new_lr_scale`` times the learning rate). Every update trains

  * the strip: smooth-L1 of the strip heights against the patch height field, over supervised grid points only, per
    decision averaged over its points (first pass full weight, refinement passes ``strip_refinement_weight`` x their
    mean);
  * the path, on decisions shown only their observed trace: the fiber follower's objective (train/supervision.py,
    every refinement pass), ``path_weight`` x (geometry + ``confidence_weight`` x survival).

Run folder ``out_root/name``: ``run.json`` (resolved configuration), ``log.jsonl`` and ``train.log``, ``last.pt``,
``best.pt`` (lowest validation strip error), ``ckpt_STEP.pt``, ``curves.png`` and ``figures/`` (validation decisions
every ``figure_every`` updates: CT slices with the patch sheet, the predicted sheet and the tube).
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.fibers.data.observations import IdentitySampling
from vesuvius.neural_tracing.tracers.fibers.data.samples import SampleConfig
from vesuvius.neural_tracing.tracers.fibers.data.tasks import TASKS, TaskBudget
from vesuvius.neural_tracing.tracers.fibers.frames.learned import frame_predictor, prediction_window
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import RunFibers, StripBuilder, StripDataset, StripIndex
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import (
    SheetStripConfig, SheetStripFollower, extend_strip_stages, render_tube)
from vesuvius.neural_tracing.tracers.fibers.train.supervision import loss_terms
from vesuvius.neural_tracing.tracers.models.checkpoint import read_checkpoint
from vesuvius.neural_tracing.tracers.models.model import config_from_checkpoint
from vesuvius.neural_tracing.tracers.models.weights import add_input_channels, conv_memory_format, load_matching_weights
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.io import atomic_torch_save
from vesuvius.neural_tracing.tracers.shared.system import raise_open_file_limit
from vesuvius.neural_tracing.tracers.shared.tensors import move_batch
from vesuvius.neural_tracing.tracers.shared.training import (
    RunLog, clip_training_gradients, lr_at, prepare_run_dir, restore_training_rng, training_rng_state, update_ema)

REFINE_ALL = 2.  # retry threshold above any confidence: every path refinement pass runs


@dataclass
class DataOptions:
    index: str = ''  # sheet_strip/build_index.py output
    afv: str = ''  # the AFV the index was built from
    full_tube_share: float = .5  # decisions shown the whole (matched) fiber, not just the trace up to the head
    max_patches: int = 4
    length_power: float = 1.  # fibers drawn in proportion to matched length**power
    fold_tolerance: float = 2.  # crop units: covering triangles/patches disagreeing more than this mask the point
    max_slope_deg: float = 70.
    span_margin: float = 8.  # patch voxels of fiber shown beyond its matched spans (full tube)
    umbilicus: str = ''  # umbilicus.json (patch voxels): the batches carry the crops' orientation to the scroll


@dataclass
class TrainOptions:
    steps: int = 40000
    batch: int = 24
    lr: float = 1e-4
    new_lr_scale: float = 3.  # learning-rate multiplier of the modules the checkpoint lacks
    warmup: int = 500
    weight_decay: float = 1e-4
    ema_decay: float = .999
    grad_clip: float = 60.
    seed: int = 0
    strip_weight: float = 1.
    strip_beta: float = .5  # smooth-L1 transition, crop units (one 9.6 um voxel)
    strip_refinement_weight: float = 1.
    path_weight: float = 1.
    confidence_weight: float = .5
    tolerance: float = 2.
    n_commit: int = 8
    confidence: float = .7
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 8
    figure_every: int = 500
    figure_rows: int = 6
    ckpt_every: int = 2000


@dataclass
class RuntimeOptions:
    device: str = 'cuda'
    workers: int = 12
    val_workers: int = 4
    threads: int = 4
    worker_cache_gb: float = .5


@dataclass
class RunConfig:
    name: str = ''
    out_root: str = 'output'
    init_checkpoint: str = ''
    model: dict = field(default_factory=dict)  # SheetStripConfig fields over the checkpoint's configuration
    data: DataOptions = field(default_factory=DataOptions)
    training: TrainOptions = field(default_factory=TrainOptions)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        document = json.loads(path.read_text())
        sections = dict(data=DataOptions, training=TrainOptions, runtime=RuntimeOptions)
        unknown = set(document)-{f.name for f in fields(cls)}
        for name, section in sections.items():
            known = {f.name for f in fields(section)}
            unknown |= {f'{name}.{k}' for k in document.get(name, {}) if k not in known}
        if unknown:
            raise ValueError(f'Unknown run configuration keys: {", ".join(sorted(unknown))}')
        resolve = lambda value: str((path.parent/value).resolve()) if value and not Path(value).is_absolute() else value
        run = cls(**{k: v for k, v in document.items() if k not in sections},
                  **{k: s(**document.get(k, {})) for k, s in sections.items()})
        run.out_root, run.init_checkpoint = resolve(run.out_root), resolve(run.init_checkpoint)
        run.data.index, run.data.afv, run.data.umbilicus = (resolve(run.data.index), resolve(run.data.afv),
                                                            resolve(run.data.umbilicus))
        return run

    def to_dict(self):
        return asdict(self)


def load_init_checkpoint(path):
    """A fiber follower checkpoint, or a sheet-strip run's checkpoint to continue from."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck if ck.get('model_type') == 'sheet_strip' else read_checkpoint(path, 'cpu')


def strip_model_config(checkpoint, overrides):
    """The checkpoint's model configuration as a SheetStripConfig with ``overrides``; the path keeps the checkpoint's
    lateral bound unless overridden."""
    if checkpoint.get('model_type') == 'sheet_strip':
        return SheetStripConfig(**{**checkpoint['model_cfg'], **overrides})
    base = config_from_checkpoint(checkpoint)
    values = dict(base.to_dict(), model_type='sheet_strip', path_lateral_limit=base.lateral_limit)
    return SheetStripConfig(**{**values, **overrides})


def initialize(cfg, checkpoint, device):
    """Model and EMA from the checkpoint's EMA weights (tube input channels zero, strip modules fresh). From a
    sheet-strip checkpoint, added refinement stages start as copies of its last one and other strip modules whose
    shape changed start fresh."""
    model = SheetStripFollower(cfg).to(device, memory_format=conv_memory_format(device))
    strip = checkpoint.get('model_type') == 'sheet_strip'
    state = checkpoint.get('ema', checkpoint['model'])
    report = load_matching_weights(model, extend_strip_stages(state, cfg) if strip else add_input_channels(model, state))
    if strip:
        report['mismatched'] = [n for n in report['mismatched'] if not n.startswith('strip_')]
    if report['mismatched'] or report['unexpected']:
        raise ValueError(f'Checkpoint weights do not fit: mismatched {report["mismatched"]}, unused {report["unexpected"]}')
    if any(not name.startswith('strip_') for name in report['fresh']):
        raise ValueError(f'Weights missing from the checkpoint: {[n for n in report["fresh"] if not n.startswith("strip_")]}')
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    return model, ema, report


def load_strip_model(path, device='cuda'):
    """The EMA model, its configuration and the checkpoint of a sheet-strip run."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = SheetStripConfig(**ck['model_cfg'])
    model = SheetStripFollower(cfg).to(device, memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    return model.eval(), cfg, ck


def with_tube(model, batch):
    """The batch's model input with the rendered tube channel appended to the CT crop."""
    cfg = model.cfg
    x = dict(batch['x'])
    if 'crop_orientation' in batch:
        x['orientation'] = batch['crop_orientation']
    if cfg.tube_channel:
        tube = render_tube(batch['tube_points'], batch['tube_valid'], cfg.fine, cfg.tube_sigma)
        x['fine'] = torch.cat((x['fine'], tube.to(x['fine'].dtype)), 1).contiguous(
            memory_format=conv_memory_format(x['fine'].device))
    return x


def strip_errors(heights, batch, beta):
    """Per decision: smooth-L1 of every pass (B, A) averaged over its supervised points, and the supervised count."""
    mask = batch['strip_mask'][:, None].float()
    error = F.smooth_l1_loss(heights, batch['strip_target'][:, None].expand_as(heights), beta=beta, reduction='none')
    count = mask[:, 0].sum((1, 2))
    return (error*mask).sum((2, 3))/count.clamp_min(1)[:, None], count


def forward_losses(model, batch, run):
    """Loss, the update's summed metrics and the model output for one batch on the device."""
    t = run.training
    x = with_tube(model, batch)
    threshold = batch['hist'].new_full((), REFINE_ALL)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=batch['hist'].is_cuda):
        output = model.training_forward(x, batch['hist'], batch['hmask'], threshold)
    output = {k: v.float() if torch.is_floating_point(v) else v for k, v in output.items()}
    selected = model.select_prediction(output, t.confidence, t.n_commit)
    terms = loss_terms(selected, batch, model.cfg, t.tolerance, n_commit=t.n_commit, refinement_loss='all',
                       policy_threshold=t.confidence)
    rows = len(batch['hist'])
    path_rows = (~batch['tube_full']).float()
    geometry = (terms['geometry_per_state']*path_rows).sum()/rows
    confidence = (terms['confidence_per_state']*path_rows).sum()/rows
    passes, count = strip_errors(output['strip_heights'], batch, t.strip_beta)
    strip = passes[:, 0]+(t.strip_refinement_weight*passes[:, 1:].mean(1) if passes.shape[1] > 1 else 0.)
    strip = strip.sum()/rows
    loss = t.path_weight*(geometry+t.confidence_weight*confidence)+t.strip_weight*strip
    metrics = dict(loss=loss.detach(), strip=strip.detach(), geometry=geometry.detach(), confidence=confidence.detach())
    if 'strip_log_scale' in output:
        # Laplace likelihood of the last pass's (fixed) error under the predicted scale, per supervised point.
        s = output['strip_log_scale'].clamp(-6., 6.)
        error = (output['strip_heights'][:, -1].detach()-batch['strip_target']).abs()
        mask = batch['strip_mask'].float()
        uncertainty = ((error*torch.exp(-s)+s)*mask).sum()/mask.sum().clamp_min(1)
        loss = loss+uncertainty
        metrics['uncertainty'] = uncertainty.detach()
    metrics.update(strip_metrics(output['strip_heights'][:, -1].detach(), batch, model.cfg))
    metrics.update(path_error_sum=terms['error_sum'].detach(), path_points=terms['geometry_count'].detach().float())
    return loss, metrics, output


def strip_metrics(heights, batch, cfg, trace_per_voxel=.5):
    """Sums over supervised strip points, in voxels (``trace_per_voxel`` crop units each): absolute error of the
    prediction and of the frame model's plane (u = 0), points within 1 and 2 voxels; per tube mode."""
    mask = batch['strip_mask']
    error = (heights-batch['strip_target']).abs()/trace_per_voxel
    plane = batch['strip_target'].abs()/trace_per_voxel
    out = {}
    for name, rows in (('', torch.ones_like(batch['tube_full'])), ('full_', batch['tube_full']),
                       ('observed_', ~batch['tube_full'])):
        m = mask & rows[:, None, None]
        out.update({f'{name}strip_points': m.sum().float(), f'{name}strip_abs_sum': error[m].sum(),
                    f'{name}plane_abs_sum': plane[m].sum(), f'{name}within1_count': (error[m] <= 1).sum().float(),
                    f'{name}within2_count': (error[m] <= 2).sum().float()})
    return out


def summarize(sums):
    """Means of summed metrics (strip errors per supervised point, in voxels)."""
    out = {}
    if sums.get('path_points'):
        out['path_error'] = sums['path_error_sum']/sums['path_points']  # crop units, every row's selected path
    for prefix in ('', 'full_', 'observed_'):
        n = sums.get(f'{prefix}strip_points', 0.)
        if n:
            out[f'{prefix}strip_mae_vox'] = sums[f'{prefix}strip_abs_sum']/n
            out[f'{prefix}plane_mae_vox'] = sums[f'{prefix}plane_abs_sum']/n
            out[f'{prefix}within1'] = sums[f'{prefix}within1_count']/n
            out[f'{prefix}within2'] = sums[f'{prefix}within2_count']/n
    return out


def build_data(run, cfg, spec, checkpoint, split, *, augment, seed, workers, full_tube_share=None,
               builder_type=StripBuilder):
    """The decision loader of one index split (batches of ``training.batch`` decisions, built by ``builder_type``)."""
    d = run.data
    sample_values = dict(checkpoint['sample_cfg'], crop=cfg.fine, prediction_window=prediction_window(cfg))
    sample = SampleConfig(**sample_values)
    sampling_values = dict(checkpoint['identity_sampling'])
    for key in ('blur_sigma', 'synthetic_tail', 'synthetic_prefix'):
        sampling_values[key] = tuple(sampling_values[key])
    sampling = IdentitySampling(**sampling_values)
    index = StripIndex(d.index, split)
    fibers = RunFibers(d.afv, index, spec.grid_scale, d.length_power)
    trace_per_patch = spec.ct_grid_scale/spec.grid_scale
    builder = builder_type(cfg, fibers, sampling, augment=augment, trace_per_patch=trace_per_patch,
                           full_tube_share=d.full_tube_share if full_tube_share is None else full_tube_share,
                           max_patches=d.max_patches, fold_tolerance=d.fold_tolerance,
                           max_slope_deg=d.max_slope_deg, span_margin=d.span_margin, umbilicus=d.umbilicus or None)
    budget = TaskBudget(tuple(1. if name == 'fresh' else 0. for name in TASKS))
    dataset = StripDataset(fibers, spec, sample, chunk=run.training.batch, seed=seed, batch_builder=builder,
                           budget=budget, cache_bytes=int(run.runtime.worker_cache_gb*(1 << 30)),
                           trace_per_patch=trace_per_patch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    return loader, index


def save(path, model, ema, opt, step, run, spec, checkpoint, best=None):
    atomic_torch_save(dict(model_type='sheet_strip', model_cfg=model.cfg.to_dict(), model=model.state_dict(),
                           ema=ema.state_dict(), optimizer=opt.state_dict(), step=step, best=best,
                           run_config=run.to_dict(), vol_spec=spec.to_dict(), rng=training_rng_state(),
                           init_checkpoint=run.init_checkpoint, sample_cfg=checkpoint['sample_cfg'],
                           identity_sampling=checkpoint['identity_sampling']), path)


@torch.no_grad()
def validate(ema, batches, run, device):
    sums = {}
    for cpu in batches:
        batch = move_batch(cpu, device)
        _, metrics, _ = forward_losses(ema, batch, run)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.)+float(value)
    out = {k: sums[k]/len(batches) for k in ('loss', 'strip', 'geometry', 'confidence', 'uncertainty') if k in sums}
    out.update(summarize(sums))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--resume')
    cli = ap.parse_args(argv)
    run = RunConfig.load(cli.config)
    t, rt = run.training, run.runtime
    raise_open_file_limit()
    torch.set_num_threads(rt.threads)
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    device = rt.device
    checkpoint = load_init_checkpoint(run.init_checkpoint)
    cfg = strip_model_config(checkpoint, run.model)
    if cfg.strip_orientation and not run.data.umbilicus:
        raise ValueError('strip_orientation needs data.umbilicus')
    frame_predictor(cfg)  # the frozen frame model loads (and its digest is checked) before workers start
    spec = CTVolumeSpec.from_dict(checkpoint['vol_spec'])
    out = prepare_run_dir(run.out_root, run.name, resume=cli.resume is not None)
    (out/'figures').mkdir(exist_ok=True)
    (out/'trainer.pid').write_text(f'{os.getpid()}\n')
    (out/'run.json').write_text(json.dumps(run.to_dict(), indent=1)+'\n')
    (out/'config.json').write_text(json.dumps(cfg.to_dict(), indent=1, default=str)+'\n')
    model, ema, report = initialize(cfg, checkpoint, device)
    fresh = set(report['fresh'])
    params = dict(model.named_parameters())
    groups = [dict(params=[p for n, p in params.items() if n not in fresh], lr_scale=1.),
              dict(params=[p for n, p in params.items() if n in fresh], lr_scale=t.new_lr_scale)]
    opt = torch.optim.AdamW(groups, lr=t.lr, weight_decay=t.weight_decay)
    step, best = 0, None
    if cli.resume:
        state = torch.load(cli.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        restore_training_rng(state['rng'])
        step, best = int(state['step']), state.get('best')
    log = RunLog(out/'log.jsonl', formatter=format_line)
    print(f'[{run.name}] {sum(p.numel() for p in model.parameters()):,} parameters; fresh: '
          f'{sorted({n.split(".")[0] for n in fresh})}; crop {cfg.fine}; strip {len(model.strip_f)} x {len(model.strip_v)} '
          f'heights, {len(model.strip_anchor)} strip tokens', flush=True)
    print(f'[{run.name}] preparing {t.val_batches} validation batches', flush=True)
    val_loader, val_index = build_data(run, cfg, spec, checkpoint, 'validation', augment=False, seed=12345,
                                       workers=rt.val_workers)
    val_iter = iter(val_loader)
    val = [next(val_iter) for _ in range(t.val_batches)]
    del val_iter, val_loader
    loader, index = build_data(run, cfg, spec, checkpoint, 'train', augment=True, seed=t.seed+step, workers=rt.workers)
    print(f'[{run.name}] {len(index)} training fibers ({index.unit_length.sum()/1e6:.1f}M voxels matched), '
          f'{len(val_index)} validation fibers; training from update {step+1} to {t.steps}', flush=True)
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.figures import plot_curves, plot_decisions
    sums, started, waited = {}, time.time(), 0.
    batches = iter(loader)
    while step < t.steps:
        tick = time.time()
        cpu = next(batches)
        waited += time.time()-tick
        step += 1
        lr = lr_at(step, t.lr, t.warmup, t.steps)
        for group in opt.param_groups:
            group['lr'] = lr*group['lr_scale']
        model.train()
        batch = move_batch(cpu, device)
        loss, metrics, _ = forward_losses(model, batch, run)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at update {step}')
        clip = clip_training_gradients(model, t.grad_clip)
        opt.step()
        update_ema(ema, model, step, t.ema_decay, ramp=False)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.)+value.detach().double()
        sums['grad_norm'] = sums.get('grad_norm', 0.)+clip['grad_norm']
        sums['updates'] = sums.get('updates', 0)+1
        if step % t.log_every == 0:
            n = sums.pop('updates')
            values = {k: float(v) for k, v in sums.items()}
            record = dict(step=step, lr=lr, seconds=time.time()-started, data_wait=waited/n,
                          **{k: values[k]/n for k in ('loss', 'strip', 'geometry', 'confidence', 'uncertainty', 'grad_norm')
                             if k in values},
                          **summarize(values))
            log.record(dict(kind='train', **record))
            sums, started, waited = {}, time.time(), 0.
        if step % t.figure_every == 0 or step == 1:
            plot_decisions(ema, val[0], run, out/'figures'/f'val_{step:06d}.png', step, t.figure_rows)
            plot_decisions(model, cpu, run, out/'figures'/f'train_{step:06d}.png', step, t.figure_rows)
        if step % t.val_every == 0:
            result = validate(ema, val, run, device)
            improved = best is None or result['strip_mae_vox'] < best
            best = result['strip_mae_vox'] if improved else best
            log.record(dict(kind='val', step=step, best=best, **result))
            plot_curves(out/'log.jsonl', out/'curves.png')
            save(out/'last.pt', model, ema, opt, step, run, spec, checkpoint, best)
            if improved:
                save(out/'best.pt', model, ema, opt, step, run, spec, checkpoint, best)
        if step % t.ckpt_every == 0:
            save(out/f'ckpt_{step:06d}.pt', model, ema, opt, step, run, spec, checkpoint, best)
    save(out/'last.pt', model, ema, opt, step, run, spec, checkpoint, best)
    print(f'[{run.name}] finished at update {step}', flush=True)


def format_line(values):
    keys = ('loss', 'strip', 'geometry', 'confidence', 'uncertainty', 'strip_mae_vox', 'plane_mae_vox', 'full_strip_mae_vox',
            'observed_strip_mae_vox', 'within2', 'path_error', 'grad_norm', 'data_wait', 'lr')
    shown = ' '.join(f'{k} {values[k]:.4g}' for k in keys if values.get(k) is not None)
    return f'{values["kind"]:5s} {values["step"]:6d} {shown}'


if __name__ == '__main__':
    main()
