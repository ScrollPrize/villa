#!/usr/bin/env python3
"""Run the Dataset001_Ink nnU-Net ensemble over the ink strips produced by render_ink.py
and report how much inked surface area the segmentation ensemble finds.

render_ink.py writes one 8-bit grayscale strip jpg per winding-range chunk into an `ink/`
folder (e.g. `w010-027.jpg`). This script feeds those strips through the trained 2d nnU-Net
ensemble (3 folds, Dataset001_Ink), whose job is to segment inked (foreground) pixels.
More foreground => more legible ink, so the total foreground area is a scalar quality metric
for a render_ink output: higher is better.

For each strip it:
  1. converts the strip to the single-channel PNG nnU-Net expects (`<stem>_0000.png`),
  2. predicts each fold's softmax probabilities (the folds are run as separate processes,
     spread across the available GPUs so they run in parallel),
  3. averages the folds' probabilities (the standard nnU-Net ensemble) and thresholds them
     into a binary ink mask,
  4. saves the mask (png) and a red ink-on-strip overlay (jpg),
  5. counts foreground pixels (the surface area of predicted ink).

Finally it aggregates the per-strip areas into a single total (the metric) and writes
metrics.json / metrics.csv.

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
import time
import shutil
import argparse
import subprocess
import multiprocessing

import numpy as np
from PIL import Image

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


def save_predictions(out_dir, stem, gray, mask):
    """Write the binary mask (png) and a red-ink overlay (jpg) for one strip."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, f'{stem}_mask.png'))

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    color = np.array([255.0, 40.0, 40.0])
    alpha = 0.45
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color
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

    # 3. Ensemble the folds' softmax probabilities and threshold into ink masks; 4. save;
    #    5. count foreground pixels (the ink surface area).
    px_area_cm2 = None
    if args.pixel_size_um:
        px_area_cm2 = (args.pixel_size_um * 1e-4) ** 2  # (um -> cm)^2 per pixel

    rows = []
    total_fg = 0
    total_px = 0
    for stem in stems:
        probs = [load_fold_prob(os.path.join(fold_out[f], f'{stem}.npz')) for f in folds]
        avg = np.mean(probs, axis=0)          # (num_classes, H, W)
        fg_prob = avg[1]                       # class 1 == ink
        mask = fg_prob >= args.fg_threshold
        gray = gray_by_stem[stem]
        # Guard against any resampling size drift between prob map and source strip.
        if mask.shape != gray.shape:
            gray = np.asarray(Image.fromarray(gray).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST))
        save_predictions(pred_dir, stem, gray, mask)

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
        rows.append(row)
        print(f'  {stem:14s} {mask.shape[1]:5d}x{mask.shape[0]:<5d} '
              f'ink={fg:>10,d}px  ({row["fg_fraction"]*100:5.2f}%)')

    # 6. Aggregate + persist.
    summary = {
        'ink_dir': ink_dir,
        'model': args.model,
        'checkpoint': args.checkpoint,
        'folds': folds,
        'fg_threshold': args.fg_threshold,
        'num_strips': len(rows),
        'total_fg_pixels': total_fg,
        'total_pixels': total_px,
        'overall_fg_fraction': (total_fg / total_px) if total_px else 0.0,
    }
    if px_area_cm2 is not None:
        summary['total_fg_area_cm2'] = total_fg * px_area_cm2
        summary['pixel_size_um'] = args.pixel_size_um

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'summary': summary, 'strips': rows}, f, indent=2)
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print('\n' + '=' * 64)
    print('INK SURFACE-AREA METRIC')
    print('=' * 64)
    print(f'strips scored          : {len(rows)}')
    print(f'TOTAL ink foreground   : {total_fg:,} px   (metric; higher = more ink)')
    print(f'total strip area       : {total_px:,} px')
    print(f'overall ink fraction   : {summary["overall_fg_fraction"]*100:.3f} %')
    if px_area_cm2 is not None:
        print(f'TOTAL ink area         : {summary["total_fg_area_cm2"]:.3f} cm^2')
    print(f'predictions            : {pred_dir}')
    print(f'metrics                : {os.path.join(out_dir, "metrics.json")}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '__worker__':
        run_worker(sys.argv[2:])
    else:
        run_main(sys.argv[1:])


if __name__ == '__main__':
    main()
