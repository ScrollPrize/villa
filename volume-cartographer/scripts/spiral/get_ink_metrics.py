#!/usr/bin/env python3
"""Run the Dataset001_Ink nnU-Net ensemble over the ink strips produced by render_ink.py
and report how much inked surface area the ensemble finds and how well that ink is
organised into the layout expected of a written scroll (text lines and columns).

render_ink.py writes one 8-bit grayscale strip jpg per winding-range chunk into an `ink/`
folder (e.g. `w010-027.jpg`). This script feeds those strips through the trained 2d nnU-Net
ensemble (3 folds, Dataset001_Ink), whose job is to segment inked (foreground) pixels.
More foreground => more legible ink, so the total foreground area is a scalar quality metric
for a render_ink output: higher is better. Ink that additionally forms regularly spaced
text lines and clean, correctly sized columns is more likely to be real writing than
scattered false positives, so two layout scores complement the area.

For each strip it:
  1. converts the strip to the single-channel PNG nnU-Net expects (`<stem>_0000.png`),
  2. predicts each fold's softmax probabilities (the folds are run as separate processes,
     spread across the available GPUs so they run in parallel),
  3. averages the folds' probabilities (the standard nnU-Net ensemble) and thresholds them
     into a binary ink mask,
  4. scores the layout of the ensemble ink probabilities: line-ness (text-line pitch close
     to the expected ~80-120 px) and column-ness (columns close to the expected ~850 px
     width, separated by ink-free gaps),
  5. saves the mask (png) and a red ink-on-strip overlay (jpg, detected column edges in
     cyan),
  6. counts foreground pixels (the surface area of predicted ink).

Finally it aggregates the per-strip areas and layout scores, writes
metrics.json / metrics.csv (the json additionally carries the full per-strip column
detection detail: run/gap positions and scores plus a downsampled density profile),
and renders everything into a single self-contained report.html next to them
(see make_ink_report.py, which can also rebuild the report on its own).

Usage:
    get_ink_metrics.py /path/to/<run>/meshes/<...>/ink
    get_ink_metrics.py /path/to/ink --output /somewhere/ink_metric --gpus 0,1,2

The nnU-Net model is pulled from HuggingFace (DEFAULT_MODEL below, cached locally by
huggingface_hub on first use); pass --model to use another HF repo id or a local
nnU-Net model folder instead.
"""

import os
import sys
import csv
import glob
import json
import math
import time
import shutil
import argparse
import subprocess
import multiprocessing

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from make_ink_report import build_report

# ---------------------------------------------------------------------------
# The model: a HuggingFace repo id (downloaded to the HF cache on first use) or a
# local nnU-Net model folder. Override with --model.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = 'scrollprize/ink-coverage-32um'
DEFAULT_CHECKPOINT = 'checkpoint_final.pth'

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def set_nnunet_env():
    """nnU-Net insists on these three env vars at import time even though predicting
    from an explicit model folder never reads them; point them at a scratch location.
    setdefault so an explicit outer environment still wins."""
    base = os.path.join(os.path.expanduser('~'), '.cache', 'ink-coverage-nnunet')
    for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        os.environ.setdefault(key, os.path.join(base, key))


def resolve_model(model):
    """Return a local nnU-Net model folder for `model`: either an existing local
    directory, or a HuggingFace repo id (e.g. 'scrollprize/ink-coverage-32um') whose
    snapshot is downloaded to the HF cache on first use."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=model)
    # The model folder (plans.json + dataset.json + fold_*/) is normally the repo
    # root, but tolerate it being nested one or more levels down.
    for root, _dirs, files in sorted(os.walk(local)):
        if 'plans.json' in files and 'dataset.json' in files:
            return root
    raise SystemExit(f'no nnU-Net model (plans.json + dataset.json) found in {model} ({local})')


# ===========================================================================
# Worker mode: predict one fold on one (already-masked) GPU. Invoked as a
# subprocess by the orchestrator, one per fold, each pinned to a GPU via
# CUDA_VISIBLE_DEVICES so the folds run in parallel.
# ===========================================================================
def run_worker(argv):
    ap = argparse.ArgumentParser(prog='get_ink_metrics __worker__')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True, help='folder of <stem>_0000.png')
    ap.add_argument('--output', required=True, help='per-fold output folder')
    ap.add_argument('--step', type=float, default=0.5, help='tile_step_size')
    ap.add_argument('--no-tta', action='store_true', help='disable mirroring TTA (faster)')
    ap.add_argument('--procs', type=int, default=8)
    args = ap.parse_args(argv)

    # nnU-Net was written assuming the fork start method; Python 3.14 defaults to
    # forkserver, which re-imports the main module in every pool worker (slow, and it
    # trips over non-file mains). Force fork to restore the behaviour nnU-Net expects.
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    set_nnunet_env()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=args.step,
        use_gaussian=True,
        use_mirroring=not args.no_tta,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),  # CUDA_VISIBLE_DEVICES already narrowed to one GPU
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model, use_folds=(args.fold,), checkpoint_name=args.checkpoint,
    )
    predictor.predict_from_files(
        args.input, args.output,
        save_probabilities=True,   # we ensemble the folds' softmax ourselves
        overwrite=True,
        num_processes_preprocessing=args.procs,
        num_processes_segmentation_export=args.procs,
    )
    print(f'[fold {args.fold}] done -> {args.output}', flush=True)


# ===========================================================================
# Layout ("grid-ness") metrics. Ancient writing has a grid: text lines at a
# regular pitch inside columns of a roughly fixed width, separated by clean
# vertical gaps (intercolumnia). Both scores work on 1D projections of the
# ensemble ink-probability map and use the expected dimensions as a prior:
# structure at the wrong scale (e.g. beautifully periodic 100 px "columns")
# scores near zero.
# ===========================================================================
COL_WIDTH_PX = 850.0         # expected column width in strip pixels
COL_WIDTH_TOL = 0.15         # fractional width variation that still scores 1.0
LINE_PITCH_PX = (80.0, 120.0)  # expected text-line pitch band in strip pixels
LINE_WIN_PX = 512            # x-window for line detection (~half a column)
LINE_STEP_PX = 256
LINE_MIN_WINDOW_INK = 0.01   # mean ink probability for a window to count
LINE_MIN_GAPS = 10           # below this many observed gaps, scale the score down
PROFILE_DS = 4               # downsampling of the density profile kept in metrics.json


def band_score(value, lo, hi, log_sigma=0.22):
    """1.0 for value inside [lo, hi], gaussian falloff outside, measured in log
    space so 2x too large is punished like 2x too small. With the default sigma
    a value 25% past a band edge scores ~0.5 and 2x past it ~0."""
    if value <= 0 or lo <= 0:
        return 0.0
    d = max(0.0, math.log(lo / value), math.log(value / hi))
    return math.exp(-0.5 * (d / log_sigma) ** 2)


def score_columns(prob, width_px=COL_WIDTH_PX, tol=COL_WIDTH_TOL):
    """Column-ness of one strip from the ensemble ink-probability map.

    Pool the probabilities vertically into an ink-density profile along x: text
    columns show up as wide high-density runs separated by low-density gaps.
    The score is (mean prior conformity of detected column widths) x (how
    ink-free the gaps are), so it punishes columns at the wrong scale, merged
    columns (dirty gaps fuse runs into one over-wide run), and ink smeared into
    the intercolumnia.

    Returns (metrics dict, detail dict). The detail carries the detected runs
    (with per-column width scores), the gaps between them (with per-gap
    cleanliness), the detection threshold, and a downsampled density profile --
    enough to redraw the detection later without the probability maps."""
    h, w = prob.shape
    metrics = {'col_score': 0.0, 'col_width_conformity': 0.0,
               'col_gap_contrast': 0.0, 'col_count': 0, 'col_median_width_px': 0}
    profile = prob.mean(axis=0)
    detail = {'threshold': 0.0, 'runs': [], 'gaps': [],
              'profile_ds': [round(float(v), 4) for v in profile[::PROFILE_DS]],
              'profile_ds_factor': PROFILE_DS}
    smooth = gaussian_filter1d(profile, sigma=max(3.0, width_px / 16.0))
    lo_v, hi_v = np.percentile(smooth, [5, 95])
    if hi_v < 0.02:  # essentially no ink anywhere
        return metrics, detail

    thr = float(lo_v + 0.3 * (hi_v - lo_v))
    detail['threshold'] = thr
    above = smooth >= thr
    steps = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts, ends = np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)
    runs = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= 25]
    if not runs:
        return metrics, detail

    col_px = np.zeros(w, dtype=bool)
    for s, e in runs:
        col_px[s:e] = True
    col_density = float(profile[col_px].mean())
    gap_density = float(profile[~col_px].mean()) if (~col_px).any() else col_density
    contrast = max(0.0, 1.0 - gap_density / col_density) if col_density > 0 else 0.0

    # Runs touching the strip edge are (potentially) truncated columns; they
    # count for gap cleanness but their width says nothing about the prior.
    for s, e in runs:
        interior = s > 0 and e < w
        detail['runs'].append({
            'start': s, 'end': e, 'width': e - s, 'interior': interior,
            'width_score': band_score(e - s, width_px * (1 - tol), width_px * (1 + tol))
                           if interior else None,
        })
    bounds = [0] + [x for s, e in runs for x in (s, e)] + [w]
    for i in range(len(runs) + 1):
        gs, ge = bounds[2 * i], bounds[2 * i + 1]
        if ge - gs < 5:
            continue
        gd = float(profile[gs:ge].mean())
        detail['gaps'].append({
            'start': int(gs), 'end': int(ge),
            'cleanliness': max(0.0, 1.0 - gd / col_density) if col_density > 0 else 0.0,
        })

    scores = [r['width_score'] for r in detail['runs'] if r['interior']]
    conformity = float(np.mean(scores)) if scores else 0.0
    widths = [r['width'] for r in detail['runs'] if r['interior']]
    metrics.update({
        'col_score': conformity * contrast,
        'col_width_conformity': conformity,
        'col_gap_contrast': contrast,
        'col_count': len(runs),
        'col_median_width_px': int(np.median(widths)) if widths else 0,
    })
    return metrics, detail


def score_lines(prob, pitch=LINE_PITCH_PX, win=LINE_WIN_PX, step=LINE_STEP_PX):
    """Line-ness of one strip from the ensemble ink-probability map.

    Sweep windows along x (narrow enough to sit inside a single column); in
    each window with enough ink, pool horizontally into a vertical profile,
    detrend it, and find text-line peaks. The score is the mean prior
    conformity of the gaps between consecutive lines: 1.0 when every observed
    pitch is inside the expected band, ~0 for random spacings or structure at
    the wrong scale. Windows without ink contribute nothing (missing coverage
    is already measured by the area metric)."""
    h, w = prob.shape
    lo_p, hi_p = pitch
    gaps = []
    for x0 in range(0, max(1, w - win + 1), step):
        chunk = prob[:, x0:x0 + win]
        if chunk.mean() < LINE_MIN_WINDOW_INK:
            continue
        profile = chunk.mean(axis=1)
        detrended = profile - gaussian_filter1d(profile, sigma=2.0 * hi_p)
        detrended = gaussian_filter1d(detrended, sigma=lo_p / 8.0)
        # Real text lines modulate the profile by roughly its own mean (ink is
        # concentrated in the lines); gating the peak prominence on that mean
        # keeps the ripple of pooled diffuse noise from posing as lines.
        prominence = max(0.25 * float(detrended.std()), 0.2 * float(profile.mean()))
        if prominence <= 0:
            continue
        peaks, _ = find_peaks(detrended, distance=max(1, int(0.3 * lo_p)),
                              prominence=prominence)
        gaps.extend(np.diff(peaks).tolist())

    metrics = {'line_score': 0.0, 'line_median_pitch_px': 0, 'line_gap_count': len(gaps)}
    if gaps:
        evidence = min(1.0, len(gaps) / LINE_MIN_GAPS)
        metrics['line_score'] = evidence * float(np.mean(
            [band_score(g, lo_p, hi_p) for g in gaps]))
        metrics['line_median_pitch_px'] = int(np.median(gaps))
    return metrics


# ===========================================================================
# Orchestrator helpers
# ===========================================================================
def list_strips(ink_dir):
    """Every image file directly inside ink_dir, sorted by name."""
    out = []
    for name in sorted(os.listdir(ink_dir)):
        p = os.path.join(ink_dir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            out.append(p)
    return out


def detect_folds(model_dir):
    folds = []
    for name in sorted(os.listdir(model_dir)):
        if name.startswith('fold_') and os.path.isdir(os.path.join(model_dir, name)):
            suffix = name[len('fold_'):]
            if suffix.isdigit():
                folds.append(int(suffix))
    return sorted(folds)


def detect_gpus():
    """GPU ids to spread folds across. Honour CUDA_VISIBLE_DEVICES if set, else ask torch."""
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env is not None and env.strip() != '':
        return [tok.strip() for tok in env.split(',') if tok.strip() != '']
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] if n > 0 else ['0']


def load_fold_prob(npz_path):
    """Load a fold's saved softmax probabilities as (num_classes, H, W) float32."""
    with np.load(npz_path) as d:
        p = d['probabilities'].astype(np.float32)
    if p.ndim == 4:      # (C, Z, H, W) for the 2d config -> drop the singleton z
        p = p[:, 0]
    return p


def save_predictions(out_dir, stem, gray, mask, col_runs=()):
    """Write the binary mask (png) and a red-ink overlay (jpg) for one strip,
    with the detected column edges drawn in cyan on the overlay."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, f'{stem}_mask.png'))

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    color = np.array([255.0, 40.0, 40.0])
    alpha = 0.45
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color
    cyan = np.array([40.0, 255.0, 255.0])
    for s, e in col_runs:
        for x in (s, e - 1):
            x0, x1 = max(0, x - 1), min(rgb.shape[1], x + 2)
            rgb[:, x0:x1] = 0.4 * rgb[:, x0:x1] + 0.6 * cyan
    Image.fromarray(rgb.astype(np.uint8)).save(
        os.path.join(out_dir, f'{stem}_overlay.jpg'), quality=95)


def run_main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ink_dir', help="folder of ink strip images (render_ink's ink/ output)")
    ap.add_argument('--output', default=None,
                    help='output folder (default: <ink_dir>/../ink_metric)')
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='HuggingFace repo id or local nnU-Net model folder')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                    help='checkpoint_final.pth (default) or checkpoint_best.pth')
    ap.add_argument('--folds', default=None,
                    help='comma list of folds to ensemble (default: all present)')
    ap.add_argument('--gpus', default=None,
                    help='comma list of GPU ids to spread folds across (default: auto)')
    ap.add_argument('--fg-threshold', type=float, default=0.5,
                    help='ensemble foreground-probability threshold for the mask '
                         '(0.5 == argmax; lower = more permissive)')
    ap.add_argument('--step', type=float, default=0.5, help='nnU-Net tile_step_size')
    ap.add_argument('--col-width-px', type=float, default=COL_WIDTH_PX,
                    help='expected text column width in strip pixels (layout prior)')
    ap.add_argument('--col-width-tol', type=float, default=COL_WIDTH_TOL,
                    help='fractional column-width variation that still scores 1.0')
    ap.add_argument('--line-pitch-px', default=f'{LINE_PITCH_PX[0]:g},{LINE_PITCH_PX[1]:g}',
                    help='expected text-line pitch band in strip pixels, as "min,max"')
    ap.add_argument('--no-tta', action='store_true',
                    help='disable mirroring test-time augmentation (~faster, slightly worse)')
    ap.add_argument('--procs', type=int, default=8,
                    help='CPU worker processes per fold for pre/post-processing')
    ap.add_argument('--pixel-size-um', type=float, default=None,
                    help='if given, also report physical ink area in cm^2 using this '
                         'strip pixel size (micrometres per pixel edge)')
    ap.add_argument('--keep-work', action='store_true',
                    help='keep the intermediate per-fold prediction folders')
    args = ap.parse_args(argv)

    ink_dir = os.path.abspath(args.ink_dir)
    if not os.path.isdir(ink_dir):
        ap.error(f'ink_dir not found: {ink_dir}')
    strips = list_strips(ink_dir)
    if not strips:
        ap.error(f'no image strips found in {ink_dir}')

    out_dir = os.path.abspath(args.output) if args.output else \
        os.path.join(os.path.dirname(ink_dir), 'ink_metric')
    pred_dir = os.path.join(out_dir, 'predictions')
    work_dir = os.path.join(out_dir, '_work')
    in_dir = os.path.join(work_dir, 'input')
    for d in (pred_dir, in_dir):
        os.makedirs(d, exist_ok=True)

    model_dir = resolve_model(args.model)
    folds = [int(x) for x in args.folds.split(',')] if args.folds else detect_folds(model_dir)
    if not folds:
        ap.error(f'no folds found in {model_dir}')
    gpus = [g.strip() for g in args.gpus.split(',')] if args.gpus else detect_gpus()

    print(f'ink_dir : {ink_dir}')
    print(f'strips  : {len(strips)}')
    print(f'model   : {args.model}' + (f' ({model_dir})' if model_dir != args.model else ''))
    print(f'folds   : {folds}   gpus: {gpus}   checkpoint: {args.checkpoint}')
    print(f'output  : {out_dir}')

    # 1. Stage strips as nnU-Net inputs: single-channel PNG named <stem>_0000.png.
    #    Remember each strip's original grayscale for the overlays and its size.
    stems = []          # nnU-Net case id per strip
    gray_by_stem = {}   # stem -> original grayscale array
    src_by_stem = {}    # stem -> original strip path
    for p in strips:
        stem = os.path.splitext(os.path.basename(p))[0]
        gray = np.asarray(Image.open(p).convert('L'))
        Image.fromarray(gray).save(os.path.join(in_dir, f'{stem}_0000.png'))
        stems.append(stem)
        gray_by_stem[stem] = gray
        src_by_stem[stem] = p

    # 2. Predict every fold in parallel: one subprocess per fold, each pinned to a GPU.
    t0 = time.time()
    procs = []
    fold_out = {}
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    for i, fold in enumerate(folds):
        gpu = gpus[i % len(gpus)]
        fold_out[fold] = os.path.join(work_dir, f'pred_fold_{fold}')
        os.makedirs(fold_out[fold], exist_ok=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        set_nnunet_env()
        for k in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
            env[k] = os.environ[k]
        cmd = [sys.executable, os.path.abspath(__file__), '__worker__',
               '--fold', str(fold), '--model', model_dir, '--checkpoint', args.checkpoint,
               '--input', in_dir, '--output', fold_out[fold], '--step', str(args.step),
               '--procs', str(args.procs)]
        if args.no_tta:
            cmd.append('--no-tta')
        logf = open(os.path.join(log_dir, f'fold_{fold}.log'), 'w')
        print(f'  launching fold {fold} on GPU {gpu} -> {os.path.basename(fold_out[fold])}')
        procs.append((fold, gpu, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))

    failed = []
    for fold, gpu, proc, logf in procs:
        rc = proc.wait()
        logf.close()
        status = 'ok' if rc == 0 else f'FAILED (rc={rc})'
        print(f'  fold {fold} (GPU {gpu}) {status}')
        if rc != 0:
            failed.append(fold)
    if failed:
        for fold in failed:
            print(f'\n----- fold {fold} log tail -----')
            with open(os.path.join(log_dir, f'fold_{fold}.log')) as f:
                print(''.join(f.readlines()[-40:]))
        raise SystemExit(f'prediction failed for folds {failed}; see logs in {log_dir}')
    print(f'prediction done in {time.time() - t0:.1f}s')

    # 3. Ensemble the folds' softmax probabilities and threshold into ink masks;
    #    4. score line-ness/column-ness of the probabilities; 5. save;
    #    6. count foreground pixels (the ink surface area).
    px_area_cm2 = None
    if args.pixel_size_um:
        px_area_cm2 = (args.pixel_size_um * 1e-4) ** 2  # (um -> cm)^2 per pixel
    line_band = tuple(float(x) for x in args.line_pitch_px.split(','))
    if len(line_band) != 2 or line_band[0] >= line_band[1]:
        ap.error(f'--line-pitch-px must be "min,max", got {args.line_pitch_px!r}')

    rows = []
    total_fg = 0
    total_px = 0
    for stem in stems:
        probs = [load_fold_prob(os.path.join(fold_out[f], f'{stem}.npz')) for f in folds]
        avg = np.mean(probs, axis=0)          # (num_classes, H, W)
        fg_prob = avg[1]                       # class 1 == ink
        mask = fg_prob >= args.fg_threshold
        col_metrics, col_detail = score_columns(fg_prob, args.col_width_px, args.col_width_tol)
        line_metrics = score_lines(fg_prob, line_band)
        gray = gray_by_stem[stem]
        # Guard against any resampling size drift between prob map and source strip.
        if mask.shape != gray.shape:
            gray = np.asarray(Image.fromarray(gray).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST))
        save_predictions(pred_dir, stem, gray, mask,
                         [(r['start'], r['end']) for r in col_detail['runs']])

        fg = int(mask.sum())
        n = int(mask.size)
        total_fg += fg
        total_px += n
        row = {
            'strip': stem,
            'source': src_by_stem[stem],
            'width': int(mask.shape[1]),
            'height': int(mask.shape[0]),
            'total_pixels': n,
            'fg_pixels': fg,
            'fg_fraction': (fg / n) if n else 0.0,
        }
        if px_area_cm2 is not None:
            row['fg_area_cm2'] = fg * px_area_cm2
        row.update(line_metrics)
        row.update(col_metrics)
        # Full per-run/per-gap detection detail: goes to metrics.json, but stays
        # out of the flat metrics.csv.
        row['columns'] = col_detail
        rows.append(row)
        print(f'  {stem:14s} {mask.shape[1]:5d}x{mask.shape[0]:<5d} '
              f'ink={fg:>10,d}px  ({row["fg_fraction"]*100:5.2f}%)  '
              f'lines={row["line_score"]:.2f}@{row["line_median_pitch_px"]}px  '
              f'cols={row["col_score"]:.2f} '
              f'({row["col_count"]} x ~{row["col_median_width_px"]}px)')

    # 7. Aggregate + persist. Layout scores aggregate as strip-area-weighted
    #    means, so a big structureless strip drags the overall score down.
    def area_weighted(key):
        return sum(r[key] * r['total_pixels'] for r in rows) / total_px if total_px else 0.0

    summary = {
        'ink_dir': ink_dir,
        'model': args.model,
        'checkpoint': args.checkpoint,
        'folds': folds,
        'fg_threshold': args.fg_threshold,
        'col_width_px': args.col_width_px,
        'col_width_tol': args.col_width_tol,
        'line_pitch_px': list(line_band),
        'num_strips': len(rows),
        'total_fg_pixels': total_fg,
        'total_pixels': total_px,
        'overall_fg_fraction': (total_fg / total_px) if total_px else 0.0,
        'overall_line_score': area_weighted('line_score'),
        'overall_column_score': area_weighted('col_score'),
    }
    if px_area_cm2 is not None:
        summary['total_fg_area_cm2'] = total_fg * px_area_cm2
        summary['pixel_size_um'] = args.pixel_size_um

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'summary': summary, 'strips': rows}, f, indent=2)
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'columns'],
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    report_path = build_report(out_dir)

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print('\n' + '=' * 64)
    print('INK METRICS')
    print('=' * 64)
    print(f'strips scored          : {len(rows)}')
    print(f'TOTAL ink foreground   : {total_fg:,} px   (metric; higher = more ink)')
    print(f'total strip area       : {total_px:,} px')
    print(f'overall ink fraction   : {summary["overall_fg_fraction"]*100:.3f} %')
    print(f'overall line-ness      : {summary["overall_line_score"]:.3f}   '
          f'(1 = all line pitches in {line_band[0]:g}-{line_band[1]:g} px band)')
    print(f'overall column-ness    : {summary["overall_column_score"]:.3f}   '
          f'(1 = clean columns of ~{args.col_width_px:g} px +/- {args.col_width_tol*100:g}%)')
    if px_area_cm2 is not None:
        print(f'TOTAL ink area         : {summary["total_fg_area_cm2"]:.3f} cm^2')
    print(f'predictions            : {pred_dir}')
    print(f'metrics                : {os.path.join(out_dir, "metrics.json")}')
    print(f'report                 : {report_path}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '__worker__':
        run_worker(sys.argv[2:])
    else:
        run_main(sys.argv[1:])


if __name__ == '__main__':
    main()
