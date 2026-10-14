"""Predict the sheet an AFV fiber lies on along the fiber and write it as a tifxyz strip.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer --checkpoint RUN/best.pt --fiber 469373 \\
        --out OUT [--afv FIBERS.afv] [--index INDEX_DIR] [--arc START END] [--step 4] [--blend 16]

Decisions are made every ``--step`` trace voxels along the fiber, each built as in training (data/samples.
decision_on_path; frame model crop, no augmentation): the observed path is the fiber up to the head, the crop axis
follows the fiber ahead, and the tube shows the whole fiber through the crop. Each decision predicts the strip's
height field u(f, v) on the model's grid (sheet_strip/model.strip_grid); its points ``pos + frame @ (u, v, f)``
become rows of the strip at fiber arclength ``t + f``. Overlapping decisions are averaged per (row, column) with
weight cos^2(pi f / 2 blend) (|f| < ``--blend``), so each row mostly follows the decisions centred near it. The
in-sheet axis v of a decision is turned to agree with the previous decision's (the frame model's roll sign is
arbitrary), so columns keep one side of the fiber.

Output (OUT): ``strip/`` (the tifxyz, patch voxels: the CT level of the patches; rows every f spacing, columns every
v spacing), ``strip.npz`` (per decision: arclength, position, frame, heights; per node: the spread of the decisions'
points) and ``summary.json``. When the fiber is in a sheet-strip index (``--index``, default: the checkpoint run's),
the patch sheet it lies on is scored and stitched the same way: ``patch_sheet/`` (the supervised nodes only) and
per-decision errors. ``render.png``: CT sampled on the predicted strip, every voxel, in bands along the fiber, each
with a bar of the error against the patch sheet where there is one (``--render-only`` redraws it).
Models reading the crop orientation (``strip_orientation``) need the scroll's umbilicus (``--umbilicus``, default: the
checkpoint run's); models with ``strip_uncertainty`` also save their expected error per decision and grid point
(``expected_error_vox`` in strip.npz, its mean over the stitched rows in summary.json).
Arclengths and ``--step``/``--blend`` are trace voxels; ``--arc`` patch voxels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.data.observations import (
    IdentityObservationBuilder, IdentitySampling, image_crop)
from vesuvius.neural_tracing.tracers.fibers.data.samples import STARTUP_CATEGORIES, SampleConfig, decision_on_path
from vesuvius.neural_tracing.tracers.fibers.frames.ct import oriented_seed_heading
from vesuvius.neural_tracing.tracers.fibers.frames.learned import prediction_window
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import (
    RunFibers, StripBuilder, StripIndex, crop_orientation, load_umbilicus, polyline_points)
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import strip_grid
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.targets import strip_targets
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import load_strip_model, with_tube
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at
from vesuvius.neural_tracing.tracers.shared.tensors import move_batch
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface

MIN_HEAD_ARC = 12.  # trace voxels: a shorter observed path would take its heading from CT (a seed), not the fiber


def decision_item(fiber, t, sample_cfg, n_history, window):
    """The decision at arclength ``t`` of a fiber traced exactly up to its head, its crop axis from the fiber ahead."""
    p, s = fiber.points, fiber.s
    count = int(min(t//sample_cfg.history_step, n_history))
    arcs = t-np.arange(count, -1, -1)*sample_cfg.history_step
    item = decision_on_path(fiber, t, False, sample_cfg, interp_at(p, s, arcs), arcs,
                            STARTUP_CATEGORIES.index('established'), 0., {})
    if window:
        item['prev_prediction'] = interp_at(p, s, np.minimum(t+np.arange(1., window+1e-9), s[-1]))
    item['fiber_ref'] = (0, float(t), False)
    return item


def skip_remote_seed_heading(item, reach):
    """Skip the CT seed heading (data/targets.resolve_trace_seed: a structure tensor of the CT at the seed) of a
    decision whose seed lies farther than ``reach`` from its head: outside the crop it is never read (the seed
    reference and its tangent are masked, observations.reference_layout), so the model's inputs do not change. The
    family the heading would set is kept."""
    if item.get('seed_heading_family') is None or item.get('_pending_seed_heading') is not None:
        return
    if np.linalg.norm(np.asarray(item['seed_pos'], np.float64)-np.asarray(item['pos'], np.float64)) > reach:
        item['fiber_family'] = item.pop('seed_heading_family')



def reuse_seed_heading(item, vol, cache):
    """Resolve a decision's CT seed heading as data/targets.resolve_trace_seed does, computing it once per seed:
    every decision of a fiber has the same seed, family and seed tangent, so ``cache`` (one per fiber) holds it.
    Decisions starting at the seed (a pending crop heading) are left to resolve_trace_seed."""
    if item.get('seed_heading_family') is None or item.get('_pending_seed_heading') is not None:
        return
    family = item.pop('seed_heading_family')
    item.pop('_pending_seed_heading', None)
    seed, tangent = (np.asarray(item[k], np.float64) for k in ('seed_pos', 'seed_tangent'))
    key = (family, seed.tobytes(), tangent.tobytes())
    if key not in cache:
        cache[key] = oriented_seed_heading(vol, item['seed_pos'], family, item['seed_tangent'])
    item['fiber_family'] = family
    item['seed_tangent'] = cache[key].copy()

class StripPredictor:
    """A sheet-strip checkpoint and what its decisions need (frame model and crop builder, CT volume, umbilicus),
    loaded once; ``predict`` runs one fiber's decisions."""

    def __init__(self, checkpoint, device='cuda', umbilicus=None, batch=16, crop_threads=1):
        self.model, self.cfg, self.ck = load_strip_model(checkpoint, device)
        self.device, self.batch = device, batch
        self.data = self.ck['run_config']['data']
        self.spec = CTVolumeSpec.from_dict(self.ck['vol_spec'])
        self.trace_per_patch = self.spec.ct_grid_scale/self.spec.grid_scale
        self.f, self.v = (a.numpy().astype(np.float64) for a in strip_grid(self.cfg))
        self.sample_cfg = SampleConfig(**dict(self.ck['sample_cfg'], crop=self.cfg.fine,
                                              prediction_window=prediction_window(self.cfg)))
        self.builder = IdentityObservationBuilder(self.cfg, None, IdentitySampling(), augment=False)
        if crop_threads > 1:
            # A batch's crops are read and sampled on a thread pool (shared/crop_sampling/crops.scalar_crops: the same
            # values; the sampler releases the GIL).
            pool = ThreadPoolExecutor(crop_threads)
            self.builder.crop_images = lambda items, vol, crop, _=None: image_crop(items, vol, crop, pool)
        self.umbilicus = None
        if self.cfg.strip_orientation:
            path = umbilicus or self.data.get('umbilicus')
            if not path:
                raise ValueError('This model reads the crop orientation: an umbilicus is needed')
            self.umbilicus = load_umbilicus(path)
        self.vol = CTVolume(self.spec)
        self.rng = np.random.default_rng(0)
        c = self.cfg.fine  # the crop's farthest point from its head, plus a sample
        self.crop_reach = math.sqrt((max(c.behind, c.depth-c.behind)*c.spacing)**2+2*((c.width-1)/2*c.spacing)**2)+c.spacing

    def predict(self, fiber, arcs, on_items=None, log=None):
        """Strip heights (K, Nf, Nv), log error scales (K, Nf, Nv; None without ``strip_uncertainty``), frames (K, 3, 3)
        and positions (K, 3) of the decisions at fiber arclengths ``arcs`` (trace voxels). ``on_items`` sees each
        batch's decision items."""
        cfg, crop = self.cfg, self.cfg.fine
        heights, scales, frames, positions = [], [], [], []
        seed_headings = {}
        for start in range(0, len(arcs), self.batch):
            items = [self.builder.prepare(decision_item(fiber, float(t), self.sample_cfg, cfg.n_history,
                                                        self.sample_cfg.prediction_window), fiber, self.rng)
                     for t in arcs[start:start+self.batch]]
            for item in items:
                skip_remote_seed_heading(item, self.crop_reach)
                reuse_seed_heading(item, self.vol, seed_headings)
            batch = self.builder(items, self.vol)  # orients each crop with the frame model, then crops the CT
            tube = np.zeros((len(items), 2048, 3), np.float32)
            valid = np.zeros((len(items), 2048), bool)
            for row, item in enumerate(items):
                curve = polyline_points(fiber.points, np.asarray(item['pos']), np.asarray(item['frame']), crop,
                                        crop.spacing/2, 3*cfg.tube_sigma*crop.spacing, len(tube[row]))
                tube[row, :len(curve)], valid[row, :len(curve)] = curve, True
            if on_items is not None:
                on_items(items)
            inputs = dict(x=batch['x'], hist=batch['hist'], hmask=batch['hmask'], tube_points=torch.from_numpy(tube),
                          tube_valid=torch.from_numpy(valid))
            if self.umbilicus is not None:
                inputs['crop_orientation'] = torch.from_numpy(crop_orientation(
                    np.stack([i['pos'] for i in items]), np.stack([i['frame'] for i in items]), self.umbilicus,
                    self.trace_per_patch))
            device_batch = move_batch(inputs, self.device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                ctx = self.model.context(with_tube(self.model, device_batch), device_batch['hist'], device_batch['hmask'])
                out, scale = self.model.strip_outputs(ctx)
            heights.append(out[:, -1].float().cpu().numpy())
            if scale is not None:
                scales.append(scale.float().cpu().numpy())
            frames.extend(np.asarray(item['frame'], np.float64) for item in items)
            positions.extend(np.asarray(item['pos'], np.float64) for item in items)
            if log is not None:
                log(min(start+self.batch, len(arcs)))
        return (np.concatenate(heights).astype(np.float64), np.concatenate(scales) if scales else None,
                np.stack(frames), np.stack(positions))


def orient_decisions(heights, frames):
    """Keep one side of the fiber in every column: each decision's v axis turned to agree with the previous one's.
    Returns the oriented heights, frames and which decisions flipped (K,)."""
    flipped = np.zeros(len(frames), bool)
    for k in range(1, len(frames)):
        flipped[k] = flipped[k-1] ^ (float(frames[k, :, 1] @ frames[k-1, :, 1]) < 0)
    oriented = np.where(flipped[:, None, None], heights[:, :, ::-1], heights)
    oriented_frames = frames.copy()
    oriented_frames[flipped, :, 1] *= -1  # the v column reversed is v -> -v: the same points
    oriented_frames[flipped, :, 0] *= -1  # and u -> -u keeps the frame right-handed; heights follow
    oriented[flipped] *= -1
    return oriented, oriented_frames, flipped


def stitch_values(values, arcs, f, blend, weights=None):
    """Per strip node (R, Nv): the stitch weights' average of a per-point value (K, Nf, Nv) of the decisions (as
    ``stitch``, which averages their points), NaN where no decision weighs in."""
    df = f[1]-f[0]
    rows = np.rint((arcs[:, None]-arcs[0]+f[None]-f[0])/df).astype(np.int64)
    window = np.where(np.abs(f) < blend, np.cos(np.pi*f/(2*blend))**2, 0.)
    w = window[None, :, None]*(np.ones_like(values) if weights is None else weights)
    shape = (int(rows.max())+1, values.shape[2])
    total, first = np.zeros(shape), np.zeros(shape)
    flat_rows = np.broadcast_to(rows[..., None], values.shape).ravel()
    flat_cols = np.broadcast_to(np.arange(values.shape[2]), values.shape).ravel()
    np.add.at(total, (flat_rows, flat_cols), w.ravel())
    np.add.at(first, (flat_rows, flat_cols), (w*values).ravel())
    return np.where(total > 0, first/np.maximum(total, 1e-12), np.nan)


class Scorer:
    """Patch-sheet targets of decisions on a fiber of a sheet-strip index (StripBuilder.windows, strip_targets)."""

    def __init__(self, index_dir, afv, catalog_index, cfg, spec, data):
        self.unit = None
        for split in ('validation', 'train', 'unused'):
            index = StripIndex(index_dir, split)
            hit = np.flatnonzero(index.units == catalog_index)
            if len(hit):
                self.unit, self.split = int(hit[0]), split
                break
        if self.unit is None:
            return
        trace_per_patch = spec.ct_grid_scale/spec.grid_scale
        self.builder = StripBuilder(cfg, RunFibers(afv, index, spec.grid_scale), IdentitySampling(), augment=False,
                                    trace_per_patch=trace_per_patch, max_patches=data.get('max_patches', 4))
        self.options = dict(fold_tolerance=data.get('fold_tolerance', 2.), max_slope_deg=data.get('max_slope_deg', 70.),
                            max_height=(cfg.fine.width-1)*cfg.fine.spacing/2)

    def __call__(self, item):
        windows, _ = self.builder.windows(dict(item, fiber_ref=(self.unit, *item['fiber_ref'][1:])))
        return strip_targets(windows, self.builder.f, self.builder.v, **self.options)


def stitch(heights, frames, positions, arcs, f, v, blend, weights=None):
    """Average every decision's strip points (world, trace voxels) into rows at arclength ``arcs[0]+f[0]+r*df``.
    ``heights`` (K, Nf, Nv) in each decision's frame; ``weights`` (K, Nf, Nv) masks points (None: all). Returns the
    rows (R, Nv, 3), the summed weight (R, Nv) and the weighted RMS distance of the decisions' points from the mean."""
    df = f[1]-f[0]
    rows = np.rint((arcs[:, None]-arcs[0]+f[None]-f[0])/df).astype(np.int64)  # (K, Nf)
    window = np.where(np.abs(f) < blend, np.cos(np.pi*f/(2*blend))**2, 0.)
    w = window[None, :, None]*(np.ones_like(heights) if weights is None else weights)
    local = np.stack(np.broadcast_arrays(heights, v[None, None], f[None, :, None]), -1)  # (K, Nf, Nv, 3): u, v, f
    world = positions[:, None, None]+np.einsum('kfvj,kij->kfvi', local, frames)
    shape = (int(rows.max())+1, len(v))
    total, first, second = np.zeros(shape), np.zeros((*shape, 3)), np.zeros(shape)
    flat_rows = np.broadcast_to(rows[..., None], heights.shape).ravel()
    flat_cols = np.broadcast_to(np.arange(len(v)), heights.shape).ravel()
    np.add.at(total, (flat_rows, flat_cols), w.ravel())
    np.add.at(first, (flat_rows, flat_cols), w.reshape(-1, 1)*world.reshape(-1, 3))
    np.add.at(second, (flat_rows, flat_cols), w.ravel()*(world**2).sum(-1).ravel())
    mean = first/np.maximum(total, 1e-12)[..., None]
    spread = np.sqrt(np.maximum(second/np.maximum(total, 1e-12)-(mean**2).sum(-1), 0.))
    return mean, total, spread


def render_strip(vol, rows, keep, row_step, column_step):
    """CT (nearest) on a strip of patch-voxel rows (R, C, 3), its grid bilinearly refined to one voxel; NaN off it."""
    from scipy.ndimage import map_coordinates
    r = np.arange(0, len(rows)-1+1e-9, 1./row_step)
    c = np.arange(0, rows.shape[1]-1+1e-9, 1./column_step)
    grid = np.meshgrid(r, c, indexing='ij')
    points = np.stack([map_coordinates(rows[..., i], grid, order=1) for i in range(3)], -1)
    inside = map_coordinates(keep.astype(np.float32), grid, order=1) > .999
    image = np.full(points.shape[:2], np.nan, np.float32)
    zyx = points[inside][:, ::-1]*(vol.spec.ct_grid_scale/vol.spec.grid_scale)*vol.input_scale  # -> trace -> CT
    image[inside] = vol.ct.sample_nearest(zyx)
    return image


def save_render(path, image, origin, title, error=None, length=1200):
    """The rendered strip (L, C), one voxel per row from arclength ``origin``, as equal bands of at most ``length``
    voxels; under each band its error against the patch sheet per row (``error`` (L,), NaN: no patch there)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pieces = max(1, math.ceil(len(image)/length))
    length = math.ceil(len(image)/pieces)
    band = 18.*image.shape[1]/length  # inches: about one voxel per voxel along and across over an 18-inch axis
    ratios = [band, .22] if error is not None else [band]
    height = pieces*(sum(ratios)+.6)+1.6
    fig, axes = plt.subplots(pieces*len(ratios), 1, figsize=(20, height), squeeze=False,
                             gridspec_kw=dict(height_ratios=ratios*pieces, hspace=.3/(sum(ratios)/len(ratios)),
                                              top=1-.7/height, bottom=.9/height))
    lo, hi = np.nanpercentile(image, (1, 99))
    errors = plt.get_cmap('RdYlGn_r').copy()
    errors.set_bad(alpha=0.)  # no patch: no bar
    width = image.shape[1]
    for piece in range(pieces):
        part = slice(piece*length, (piece+1)*length)
        a = origin+piece*length
        b = a+len(image[part])
        ax = axes[piece*len(ratios), 0]
        ax.imshow(image[part].T, cmap='gray', origin='lower', aspect='auto', vmin=lo, vmax=hi,
                  extent=(a, b, -width/2, width/2))
        ax.set_xlim(a, a+length)
        ax.set_ylabel('across (vox)', fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=error is None)
        if error is not None:
            ax = axes[piece*len(ratios)+1, 0]
            shown = ax.imshow(error[part][None], cmap=errors, vmin=0, vmax=5, aspect='auto', extent=(a, b, 0, 1))
            ax.set_xlim(a, a+length)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            for side in ('top', 'left', 'right'):
                ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel('arclength along the fiber (vox)', fontsize=9)
    if error is not None:
        bar = fig.colorbar(shown, ax=axes[:, 0].tolist(), orientation='horizontal', fraction=.25/fig.get_figheight(),
                           pad=.6/fig.get_figheight(), aspect=80)
        bar.set_label('bar under each band, where verified patches cover the strip: |error| against them (vox)', fontsize=9)
    fig.suptitle(title, fontsize=11, y=1-.1/height)
    fig.savefig(path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def render_saved(out, vol):
    """``render.png`` of an inference output folder: CT on its strip and the decisions' errors along it."""
    import tifffile
    out = Path(out)
    summary = json.loads((out/'summary.json').read_text())
    saved = np.load(out/'strip.npz')
    rows = np.stack([tifffile.imread(out/'strip'/f'{axis}.tif') for axis in 'xyz'], -1).astype(np.float64)
    keep = np.all(rows != -1, -1)
    image = render_strip(vol, rows, keep, summary['row_step_voxels'], summary['column_step_voxels'])
    filled = np.flatnonzero(np.isfinite(image).any(1))
    image = image[filled[0]:filled[-1]+1]
    origin = summary['first_row_arc_patch_voxels']+filled[0]
    error, title = None, f'fiber {summary["fiber"]}: CT on the predicted strip ({Path(summary["checkpoint"]).name}, update {summary["step"]})'
    if 'decision_error_vox' in saved:
        arcs = saved['arcs']/float(saved['trace_per_patch'])  # decision heads, patch voxels
        along = origin+np.arange(len(image))
        nearest = np.clip(np.searchsorted(arcs, along), 1, len(arcs)-1)
        nearest = np.where(np.abs(arcs[nearest-1]-along) <= np.abs(arcs[nearest]-along), nearest-1, nearest)
        spacing = float(np.median(np.diff(arcs))) if len(arcs) > 1 else 1.
        error = np.where(np.abs(arcs[nearest]-along) <= spacing, saved['decision_error_vox'][nearest], np.nan)
        score = summary['patch_score']
        covered = float(np.isfinite(saved['decision_error_vox']).mean())
        title += (f'\nwhere verified patches cover it ({covered:.0%} of the fiber): mean error {score["strip_mae_vox"]:.2f} '
                  f'vox (frame plane {score["plane_mae_vox"]:.2f}), {score["within2"]:.0%} within 2 vox')
    save_render(out/'render.png', image, origin, title, error)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fiber', type=int, required=True, help='AFV fiber id')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--afv', help="default: the checkpoint run's")
    ap.add_argument('--index', help="sheet-strip index for scoring (default: the checkpoint run's; 'none': skip)")
    ap.add_argument('--arc', type=float, nargs=2, help='fiber arclength range, patch voxels (default: all of it)')
    ap.add_argument('--step', type=float, default=4., help='trace voxels between decisions (a multiple of the f spacing)')
    ap.add_argument('--blend', type=float, default=16., help='trace voxels: decisions weigh their rows within this of the head')
    ap.add_argument('--umbilicus', help="umbilicus.json for models reading the crop orientation (default: the checkpoint run's)")
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--voxel-size-um', type=float, default=None, help='patch voxel size, for area_cm2 in meta.json')
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--render-only', action='store_true', help='redraw render.png of an existing --out, no inference')
    args = ap.parse_args(argv)
    started = time.time()
    if args.render_only:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        render_saved(args.out, CTVolume(CTVolumeSpec.from_dict(ck['vol_spec'])))
        return
    try:
        predictor = StripPredictor(args.checkpoint, args.device, args.umbilicus, args.batch)
    except ValueError as error:
        raise SystemExit(f'{error} (--umbilicus)')
    cfg, ck, data, spec = predictor.cfg, predictor.ck, predictor.data, predictor.spec
    trace_per_patch, f, v = predictor.trace_per_patch, predictor.f, predictor.v
    afv_path = args.afv or data['afv']
    afv = AFVFibers(afv_path, spec.grid_scale, split='all')
    if args.fiber not in afv.id_to_index:
        raise SystemExit(f'Fiber {args.fiber} is not in {afv_path}')
    catalog_index = afv.id_to_index[args.fiber]
    fiber = afv[catalog_index]
    df = f[1]-f[0]
    if abs(args.step/df-round(args.step/df)) > 1e-6 or args.step <= 0:
        raise SystemExit(f'--step must be a positive multiple of the strip row spacing {df:g}')
    lo, hi = (0., fiber.length) if args.arc is None else (args.arc[0]*trace_per_patch, args.arc[1]*trace_per_patch)
    lo, hi = max(lo, MIN_HEAD_ARC), min(hi, fiber.length)
    if hi <= lo:
        raise SystemExit('The arclength range holds no decision')
    arcs = np.arange(lo, hi+1e-9, args.step)
    index_dir = None if args.index == 'none' else (args.index or data.get('index'))
    scorer = Scorer(index_dir, afv_path, catalog_index, cfg, spec, data) if index_dir else None
    scorer = scorer if scorer is not None and scorer.unit is not None else None
    vol = predictor.vol
    targets, masks = [], []

    def score(items):
        for item in items:
            target, mask = scorer(item)
            targets.append(target)
            masks.append(mask)

    heights, scales, frames, positions = predictor.predict(
        fiber, arcs, on_items=score if scorer is not None else None,
        log=lambda done: print(f'[infer] {done}/{len(arcs)} decisions, {time.time()-started:.0f} s', flush=True))
    oriented, oriented_frames, flipped = orient_decisions(heights, frames)
    rows, weight, spread = stitch(oriented, oriented_frames, positions, arcs, f, v, args.blend)
    keep = weight > 0
    row_step, column_step = df/trace_per_patch, (v[1]-v[0])/trace_per_patch
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck.get('step'), afv=str(afv_path),
                      fiber=args.fiber, arc_trace=[float(arcs[0]), float(arcs[-1])], decision_step=args.step,
                      blend=args.blend, units='patch voxels', trace_per_patch=trace_per_patch,
                      first_row_arc_patch_voxels=float((arcs[0]+f[0])/trace_per_patch))
    write_surface(args.out/'strip', rows/trace_per_patch, keep, row_step, column_step, provenance, overwrite=True,
                  voxel_size_um=args.voxel_size_um)
    summary = dict(provenance, decisions=len(arcs), rows=int(rows.shape[0]), columns=int(rows.shape[1]),
                   row_step_voxels=row_step, column_step_voxels=column_step, v_flips=int(flipped.sum()),
                   spread_vox=dict(median=float(np.median(spread[keep])/trace_per_patch),
                                   p95=float(np.percentile(spread[keep], 95)/trace_per_patch)))
    saved = dict(arcs=arcs, positions=positions, frames=frames, heights=heights, flipped=flipped, f=f, v=v,
                 spread=spread/trace_per_patch, trace_per_patch=trace_per_patch)
    if scales is not None:
        # The model's expected |error| per decision and grid point (voxels; strip_uncertainty), and per decision over
        # the rows stitching weighs (|f| < blend).
        expected = np.exp(scales)/trace_per_patch
        saved['expected_error_vox'] = expected
        central = np.abs(f) < args.blend
        per_decision = expected[:, central].mean((1, 2))
        summary['expected_error_vox'] = dict(mean=float(per_decision.mean()), p95=float(np.percentile(per_decision, 95)))
    errors, gt = None, None
    if scorer is not None:
        targets, masks = np.stack(targets).astype(np.float64), np.stack(masks)
        count = masks.sum((1, 2))
        has = count > 0
        error = np.where(masks, np.abs(heights-targets), 0.).sum((1, 2))/np.maximum(count, 1)/trace_per_patch
        plane = np.where(masks, np.abs(targets), 0.).sum((1, 2))/np.maximum(count, 1)/trace_per_patch
        errors = dict(strip=np.where(has, error, np.nan), plane=np.where(has, plane, np.nan))
        diff = np.abs(heights-targets)[masks]/trace_per_patch
        summary['patch_score'] = dict(split=scorer.split, scored_decisions=int(has.sum()),
                                      supervised_points=int(masks.sum()), strip_mae_vox=float(diff.mean()),
                                      plane_mae_vox=float((np.abs(targets)[masks]/trace_per_patch).mean()),
                                      within1=float((diff <= 1).mean()), within2=float((diff <= 2).mean()),
                                      p95_vox=float(np.percentile(diff, 95)))
        oriented_targets = np.where(flipped[:, None, None], -targets[:, :, ::-1], targets)
        oriented_masks = np.where(flipped[:, None, None], masks[:, :, ::-1], masks)
        gt, gt_weight, _ = stitch(oriented_targets, oriented_frames, positions, arcs, f, v, args.blend,
                                  oriented_masks.astype(np.float64))
        gt_keep = gt_weight > 1e-6
        write_surface(args.out/'patch_sheet', gt/trace_per_patch, gt_keep, row_step, column_step,
                      dict(provenance, source='patch sheet targets'), overwrite=True, voxel_size_um=args.voxel_size_um)
        both = keep & gt_keep
        summary['patch_score']['stitched_node_distance_vox'] = dict(
            median=float(np.median(np.linalg.norm(rows-gt, axis=-1)[both])/trace_per_patch),
            p95=float(np.percentile(np.linalg.norm(rows-gt, axis=-1)[both], 95)/trace_per_patch))
        saved.update(targets=targets, masks=masks, decision_error_vox=errors['strip'], decision_plane_vox=errors['plane'])
    np.savez_compressed(args.out/'strip.npz', **saved)
    summary['seconds'] = time.time()-started
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    if not args.no_render:
        render_saved(args.out, vol)


if __name__ == '__main__':
    main()
