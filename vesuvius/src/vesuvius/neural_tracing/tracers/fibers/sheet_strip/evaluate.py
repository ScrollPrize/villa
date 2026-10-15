"""Evaluate sheet-strip checkpoints on one fixed, cached set of validation decisions.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.evaluate build --checkpoint RUN/best.pt \\
        --out EVAL [--batches 250] [--seed 777]
    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.evaluate run --eval EVAL --checkpoint CK [--name NAME]

``build`` draws decisions from the validation split of the checkpoint run's index exactly as the trainer validates
(no augmentation) and caches them (EVAL/batches/, CT crops as float16), each with the tube curves of both modes
(observed trace and full fiber), the strip targets and the height field of every patch on its own
(``patch_heights``, NaN where a patch does not supervise a point), so the label noise between patches can be measured.
EVAL/meta.npz collects the targets and per-decision metadata. The destination must be new or empty.

``run`` evaluates a checkpoint (``--override``: under a changed model configuration, e.g. more strip refinement
passes) on every cached decision in both tube modes: EVAL/results/NAME.npz holds the strip
heights of every pass (N, passes, Nf, Nv) per mode (and the predicted log error scale, ``strip_uncertainty``) and, for the observed mode, the path error per decision; the
printed summary is in voxels (strip errors over supervised points).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import StripBuilder, crop_orientation, load_umbilicus
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig, SheetStripFollower, extend_strip_stages
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.targets import strip_targets
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import (
    REFINE_ALL, RunConfig, build_data, load_strip_model, with_tube)
from vesuvius.neural_tracing.tracers.fibers.train.supervision import loss_terms
from vesuvius.neural_tracing.tracers.models.weights import conv_memory_format, load_matching_weights
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.tensors import move_batch

MODES = ('observed', 'full')
META_KEYS = ('strip_target', 'strip_mask', 'strip_patches', 'tube_full', 'patch_heights', 'fiber_id', 'fiber_t',
             'fiber_reverse', 'fiber_length', 'crop_pos', 'crop_frame')


class EvalBuilder(StripBuilder):
    """StripBuilder plus the tube curves of both modes, every patch's own height field and the fiber reference."""

    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        n, nf, nv = len(items), len(self.f), len(self.v)
        height = (self.cfg.fine.width-1)*self.cfg.fine.spacing/2
        for mode in MODES:
            points = np.zeros((n, self.tube_points, 3), np.float32)
            valid = np.zeros((n, self.tube_points), bool)
            for row, item in enumerate(items):
                drawn, item['tube_full'] = item['tube_full'], mode == 'full'
                curve = self.tube_curve(item)
                item['tube_full'] = drawn
                points[row, :len(curve)], valid[row, :len(curve)] = curve, True
            batch[f'tube_points_{mode}'], batch[f'tube_valid_{mode}'] = torch.from_numpy(points), torch.from_numpy(valid)
        per_patch = np.full((n, self.max_patches, nf, nv), np.nan, np.float32)
        for row, item in enumerate(items):
            for k, window in enumerate(self.windows(item)[0]):
                heights, mask = strip_targets([window], self.f, self.v, fold_tolerance=self.fold_tolerance,
                                              max_slope_deg=self.max_slope_deg, max_height=height)
                per_patch[row, k][mask] = heights[mask]
        units = [item['fiber_ref'][0] for item in items]
        batch.update(patch_heights=torch.from_numpy(per_patch),
                     fiber_id=torch.tensor([int(self.index.units[u]) for u in units]),
                     fiber_t=torch.tensor([float(item['fiber_ref'][1]) for item in items]),
                     fiber_reverse=torch.tensor([bool(item['fiber_ref'][2]) for item in items]),
                     fiber_length=torch.tensor([float(self.fibers[u].length) for u in units]))
        return batch


def build(args):
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f'Evaluation cache destination must be new or empty: {out}')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    run = RunConfig(**{k: v for k, v in ck['run_config'].items() if k not in ('data', 'training', 'runtime')},
                    **{k: type(getattr(RunConfig(), k))(**ck['run_config'][k]) for k in ('data', 'training', 'runtime')})
    cfg = SheetStripConfig(**ck['model_cfg'])
    spec = CTVolumeSpec.from_dict(ck['vol_spec'])
    (out/'batches').mkdir(parents=True, exist_ok=True)
    loader, index = build_data(run, cfg, spec, ck, 'validation', augment=False, seed=args.seed, workers=args.workers,
                               builder_type=EvalBuilder)
    (out/'eval.json').write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()), seed=args.seed,
                                                 batches=args.batches, batch=run.training.batch,
                                                 fibers=len(index)), indent=1)+'\n')
    started = time.time()
    meta = {k: [] for k in META_KEYS}
    for b, batch in zip(range(args.batches), loader):
        batch['x']['fine'] = batch['x']['fine'].half()
        torch.save(batch, out/'batches'/f'{b:05d}.pt')
        for k in META_KEYS:
            meta[k].append(batch[k].numpy())
        if b % 10 == 0:
            print(f'batch {b}/{args.batches} {time.time()-started:.0f}s', flush=True)
    np.savez(out/'meta.npz', **{k: np.concatenate(v) for k, v in meta.items()})
    print(f'{args.batches} batches in {time.time()-started:.0f}s -> {out}', flush=True)


def override_model(ck, overrides, device):
    """The checkpoint's EMA model under configuration ``overrides`` (KEY=JSON); added strip refinement stages start as
    copies of the last trained one."""
    values = {k: json.loads(v) for k, v in (o.split('=', 1) for o in overrides)}
    cfg = SheetStripConfig(**{**ck['model_cfg'], **values})
    model = SheetStripFollower(cfg).to(device, memory_format=conv_memory_format(device))
    report = load_matching_weights(model, extend_strip_stages(ck['ema'], cfg))
    if report['fresh'] or report['mismatched']:
        raise ValueError(f'Overridden model does not fit the checkpoint: {report}')
    return model.eval(), cfg


@torch.no_grad()
def evaluate(args):
    root = Path(args.eval)
    model, cfg, ck = load_strip_model(args.checkpoint, args.device)
    if args.override:
        model, cfg = override_model(ck, args.override, args.device)
    run = RunConfig()
    run.training.__dict__.update(ck['run_config']['training'])
    t = run.training
    name = args.name or f'{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}'
    files = sorted((root/'batches').glob('*.pt'))
    umbilicus = ck['run_config']['data'].get('umbilicus')
    umbilicus = load_umbilicus(umbilicus) if cfg.strip_orientation else None
    spec = CTVolumeSpec.from_dict(ck['vol_spec'])
    heights, scales = {m: [] for m in MODES}, {}
    path_error, path_count = [], []
    for path in files:
        batch = move_batch(torch.load(path, map_location='cpu', weights_only=False), args.device)
        batch['x']['fine'] = batch['x']['fine'].float()
        if umbilicus is not None and 'crop_orientation' not in batch:
            batch['crop_orientation'] = torch.from_numpy(crop_orientation(
                batch['crop_pos'].cpu().numpy(), batch['crop_frame'].cpu().numpy(), umbilicus,
                spec.ct_grid_scale/spec.grid_scale)).to(args.device)
        for mode in MODES:
            batch['tube_points'], batch['tube_valid'] = batch[f'tube_points_{mode}'], batch[f'tube_valid_{mode}']
            x = with_tube(model, batch)
            threshold = batch['hist'].new_full((), REFINE_ALL)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=batch['hist'].is_cuda):
                output = model.training_forward(x, batch['hist'], batch['hmask'], threshold)
            output = {k: v.float() if torch.is_floating_point(v) else v for k, v in output.items()}
            heights[mode].append(output['strip_heights'].cpu().numpy())
            if 'strip_log_scale' in output:
                scales.setdefault(mode, []).append(output['strip_log_scale'].cpu().numpy())
            if mode == 'observed':
                selected = model.select_prediction(output, t.confidence, t.n_commit)
                terms = loss_terms(selected, batch, cfg, t.tolerance, n_commit=t.n_commit, refinement_loss='all',
                                   policy_threshold=t.confidence)
                path_error.append(terms['geometry_per_state'].cpu().numpy())
    (root/'results').mkdir(exist_ok=True)
    result = {f'heights_{m}': np.concatenate(v) for m, v in heights.items()}
    result['path_geometry_observed'] = np.concatenate(path_error)
    result.update({f'log_scale_{m}': np.concatenate(v) for m, v in scales.items()})
    np.savez(root/'results'/f'{name}.npz', checkpoint=str(Path(args.checkpoint).resolve()), step=ck.get('step', -1),
             **result)
    meta = np.load(root/'meta.npz')
    target, mask = meta['strip_target'], meta['strip_mask']
    summary = dict(name=name, step=ck.get('step'), decisions=len(target), plane_mae_vox=float(np.abs(target[mask]).mean()*2))
    for mode in MODES:
        error = np.abs(result[f'heights_{mode}'][:, -1]-target)[mask]*2  # trace -> voxels
        summary[f'{mode}_mae_vox'] = float(error.mean())
        summary[f'{mode}_within2'] = float((error <= 2).mean())
    (root/'results'/f'{name}.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--checkpoint', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--batches', type=int, default=250)
    b.add_argument('--seed', type=int, default=777)
    b.add_argument('--workers', type=int, default=12)
    r = sub.add_parser('run')
    r.add_argument('--eval', required=True)
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--name')
    r.add_argument('--device', default='cuda')
    r.add_argument('--override', nargs='*', default=[], help='model configuration KEY=JSON, e.g. strip_refinement_steps=2')
    args = ap.parse_args(argv)
    build(args) if args.command == 'build' else evaluate(args)


if __name__ == '__main__':
    main()
