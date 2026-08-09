"""Phantom-vs-real correlation study: does the synthetic benchmark's score
predict real-scroll performance? (The evidence bar named when the phantom
harness PR was closed.)

Runs a grid of reference-fitter method variants over phantom datasets and
real classically-traced datasets, with an IDENTICAL sector holdout carved
from both by vertex angle, then reports Spearman rank correlation between
each variant's mean phantom truth-MAE and its mean real held-out-arc MAE.

Usage (real datasets come from trace_real_windings.py, e.g. three slabs):

  for Z in 883 1405 1927; do
      python trace_real_windings.py --out work/ds_z$Z --z0 $Z \
          --slab-cache work/slab_z$Z.npy
  done
  python phantom_real_correlation.py --work work \
      --real-dataset work/ds_z883 --real-dataset work/ds_z1405 \
      --real-dataset work/ds_z1927

Resumable: completed (variant, case) pairs in <work>/results.jsonl are
skipped, so interrupted runs continue and real-fit rows can be reused when
only the phantom side changes (e.g. adding --phantom-slips).

Results on record (12 reference-fitter variants; real reference = classical
traces from trace_real_windings.py; Spearman rho of phantom truth-MAE vs real
held-out-arc MAE):

  in-family, clean labels                              rho=0.31 (p=0.32)
  + out-of-family slips (--phantom-slips 3)            rho=0.40 (p=0.20)
  + deformation calibrated to the real fit             rho=0.72 (p=0.008)
    + contradictory labels (--phantom-index-noise 0.2)

Each critique the PR review raised, when implemented, moved the benchmark
toward reality. The dominant fix was deformation MAGNITUDE: the phantom's
warps had been far milder than a real scroll's, so undertrained configs
passed on phantoms while failing on real data; matching the phantom's
flow/gap/linear std to the real fitted checkpoint (in winding units) fixed it.

Metric-choice transparency (not cherry-picking): the x-axis above is phantom
MAE-vs-exact-truth -- the benchmark's actual headline output, the thing real
data cannot provide. The methodological TWIN of the real metric (phantom
held-out-arc MAE, same sector mask as the real side) correlates even better
at round 3 -- rho=0.78 vs the 0.72 reported for truth -- so the reported
number is the CONSERVATIVE of the two, not the flattering one.

COLD CROSS-SCROLL (calibration measured on PHerc0172, frozen, then tested on
PHerc0332 slabs 700/1050/1400 -- a scroll the calibration never saw):
rho=0.80 (p=0.002), i.e. transfer without recalibration. The two real scrolls
rank the 12 methods at rho=0.95 with each other, so method quality is largely
scroll-independent -- the precondition that makes the benchmark viable.

Reproduce cross-scroll (frozen calibration): seed a fresh work dir's
results.jsonl with the PHANTOM rows from the calibration run (so phantoms are
reused, not recalibrated), then:

  python phantom_real_correlation.py --work <new> \
      --real-dataset <PHerc0332 ds700> --real-dataset ... \
      --phantom-slips 3 --phantom-index-noise 0.2 \
      --phantom-flow-std 0.0137 --phantom-gap-log-std 0.107 \
      --phantom-linear-std 0.0046

Honest scope: n=12 variants, one fitter FAMILY (this reference fitter, not
production fit_spiral), two scrolls, a classical-trace real reference with a
~0.6-winding floor. The benchmark reliably separates good configs from
catastrophic ones; it does not yet finely rank within the good cluster.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import click
import numpy as np
from PIL import Image
import torch

from checkpoint_io import load_checkpoint_cpu
from sample_spiral import get_theta_and_radii
from synth_phantom import build_model_from_phantom_checkpoint

SPIRAL = os.path.dirname(os.path.abspath(__file__))
SECTORS = [(40., 80.), (200., 240.)]
PHANTOM_SEEDS = [101, 202]

VARIANTS = {
    'undertrained_150': ['--steps', '150'],
    'undertrained_400': ['--steps', '400'],
    'noreg_1200': ['--steps', '1200'],
    'mildreg': ['--steps', '1200', '--reg-gap', '0.01', '--reg-flow', '0.1'],
    'canonical': ['--steps', '1200', '--reg-gap', '0.03', '--reg-flow', '1.0',
                  '--reg-flow-smooth', '100'],
    'oversmooth': ['--steps', '1200', '--reg-gap', '0.03', '--reg-flow', '1.0',
                   '--reg-flow-smooth', '1000'],
    'strongflow_10': ['--steps', '1200', '--reg-gap', '0.03', '--reg-flow', '10.'],
    'rigid_flow100': ['--steps', '1200', '--reg-flow', '100.'],
    'lowlr': ['--steps', '1200', '--lr', '3e-5'],
    'bighuber': ['--steps', '1200', '--huber-delta', '1.0', '--reg-gap', '0.03',
                 '--reg-flow', '1.0', '--reg-flow-smooth', '100'],
    'long_2400': ['--steps', '2400', '--reg-gap', '0.03', '--reg-flow', '1.0',
                  '--reg-flow-smooth', '100'],
    'frozen_gaps': ['--steps', '1200', '--reg-gap', '1.0', '--reg-flow', '1.0',
                    '--reg-flow-smooth', '100'],
}


def run(cmd):
    p = subprocess.run([sys.executable, *cmd], cwd=SPIRAL,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[0]} failed: {p.stderr[-500:]}')


def umbilicus_fn(dataset_dir):
    pts = json.load(open(os.path.join(dataset_dir, 'umbilicus.json')))['control_points']
    pts = sorted(pts, key=lambda p: p['z'])
    zs = np.array([p['z'] for p in pts])
    yx = np.array([[p['y'], p['x']] for p in pts])
    return lambda z: np.stack([np.interp(z, zs, yx[:, 0]), np.interp(z, zs, yx[:, 1])], -1)


def mask_sectors(src, dst, heldout_npz):
    """Copy dataset, invalidate vertices whose angle about the per-z
    umbilicus falls in SECTORS; save withheld (zyx, k) for evaluation.
    Format-agnostic: works on phantom exports and real traced datasets."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    ufn = umbilicus_fn(dst)
    held_zyx, held_k = [], []
    pdirs = os.path.join(dst, 'verified_patches')
    for entry in sorted(os.listdir(pdirs)):
        pdir = os.path.join(pdirs, entry)
        coords = {c: np.array(Image.open(f'{pdir}/{c}.tif')) for c in 'zyx'}
        ann = np.array(Image.open(f'{pdir}/winding.tif'))
        valid = (coords['z'] != -1) | (coords['y'] != -1) | (coords['x'] != -1)
        umb = ufn(coords['z'])
        theta = np.degrees(np.arctan2(coords['y'] - umb[..., 0],
                                      coords['x'] - umb[..., 1])) % 360.
        withheld = np.zeros_like(valid)
        for lo, hi in SECTORS:
            withheld |= (theta >= lo) & (theta < hi)
        sel = valid & withheld
        held_zyx.append(np.stack([coords[c][sel] for c in 'zyx'], -1))
        held_k.append(ann[sel])
        for c in 'zyx':
            arr = coords[c].copy()
            arr[sel] = -1.
            Image.fromarray(arr.astype(np.float32)).save(f'{pdir}/{c}.tif')
    np.savez(heldout_npz, zyx=np.concatenate(held_zyx).astype(np.float32),
             k=np.concatenate(held_k).astype(np.float32))


@torch.no_grad()
def heldout_mae(ckpt_path, npz_path):
    ck = load_checkpoint_cpu(ckpt_path)
    model = build_model_from_phantom_checkpoint(ck, torch.device('cpu'))
    gauge = float(ck['reference_fit']['gauge'])
    dr = model.get_dr_per_winding()
    h = np.load(npz_path)
    spiral = model.get_slice_to_spiral_transform()(torch.from_numpy(h['zyx']))
    _, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    resid = (shifted / dr).numpy() - gauge - h['k']
    resid = resid - np.median(resid)
    return float(np.abs(resid).mean()), float((np.abs(resid) > 0.5).mean())


@click.command()
@click.option('--work', required=True, type=click.Path(file_okay=False))
@click.option('--real-dataset', 'real_datasets', multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help='trace_real_windings output dirs (repeatable).')
@click.option('--phantom-slips', default=0, type=int,
              help='Out-of-family slip dislocations per phantom (see '
                   'synth_phantom --num-slips).')
@click.option('--phantom-coverage', default=0.45, type=float)
@click.option('--phantom-noise', default=1.0, type=float)
@click.option('--phantom-index-noise', default=0., type=float,
              help='Per-patch +-1 winding mis-index rate (self-INconsistent '
                   'constraints; PHerc0172 classical traces measured ~0.24).')
@click.option('--phantom-flow-std', default=None, type=float,
              help='Override synth_phantom --flow-std (deformation calibration).')
@click.option('--phantom-gap-log-std', default=None, type=float)
@click.option('--phantom-linear-std', default=None, type=float)
def main(work, real_datasets, phantom_slips, phantom_coverage, phantom_noise,
         phantom_index_noise, phantom_flow_std, phantom_gap_log_std,
         phantom_linear_std):
    os.makedirs(work, exist_ok=True)
    results_path = os.path.join(work, 'results.jsonl')

    cases = []
    for seed in PHANTOM_SEEDS:
        ph = os.path.join(work, f'phantom_{seed}')
        ds = os.path.join(work, f'phds_{seed}')
        train = os.path.join(work, f'phtrain_{seed}')
        npz = os.path.join(work, f'phheld_{seed}.npz')
        if not os.path.exists(os.path.join(ph, 'phantom_checkpoint.pt')):
            extra = []
            for flag, value in (('--flow-std', phantom_flow_std),
                                ('--gap-log-std', phantom_gap_log_std),
                                ('--linear-std', phantom_linear_std)):
                if value is not None:
                    extra += [flag, str(value)]
            run(['synth_phantom.py', '--out', ph, '--seed', str(seed),
                 '--z-size', '32', '--yx-size', '640', '--dr-per-winding', '12',
                 '--num-tears', '0', '--num-slips', str(phantom_slips), *extra])
            click.echo(f'phantom {seed} generated (slips={phantom_slips})')
        if not os.path.exists(npz):
            run(['export_phantom_dataset.py', '--phantom', ph, '--out', ds,
                 '--step', '2', '--coverage', str(phantom_coverage),
                 '--position-noise', str(phantom_noise),
                 '--index-noise-rate', str(phantom_index_noise),
                 '--seed', str(seed)])
            mask_sectors(ds, train, npz)
            click.echo(f'phantom {seed} dataset + sector holdout built')
        cases.append(('phantom', f'ph{seed}', train, npz, ph))
    for src in real_datasets:
        name = os.path.basename(os.path.normpath(src))
        train = os.path.join(work, f'realtrain_{name}')
        npz = os.path.join(work, f'realheld_{name}.npz')
        if not os.path.exists(npz):
            mask_sectors(src, train, npz)
            click.echo(f'real {name} sector holdout built')
        cases.append(('real', name, train, npz, None))

    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            r = json.loads(line)
            done.add((r['variant'], r['case']))

    for vname, vargs in VARIANTS.items():
        for kind, cname, train, npz, ph in cases:
            if (vname, cname) in done:
                continue
            t0 = time.time()
            ckpt = os.path.join(work, 'tmp_fit.pt')
            try:
                run(['fit_phantom_reference.py', '--dataset', train,
                     '--out', ckpt, '--seed', '0', *vargs])
                row = {'variant': vname, 'case': cname, 'kind': kind, 'status': 'ok'}
                row['heldout_mae'], row['heldout_switch'] = heldout_mae(ckpt, npz)
                if kind == 'phantom':
                    out_json = os.path.join(work, 'tmp_score.json')
                    run(['winding_error.py', '--phantom', ph,
                         '--candidate-checkpoint', ckpt, '--num-points', '50000',
                         '--output', out_json])
                    s = json.load(open(out_json))
                    row['truth_mae'] = s['overall']['mae']
                    row['truth_switch'] = s['overall']['switch_rate']
            except Exception as e:
                row = {'variant': vname, 'case': cname, 'kind': kind,
                       'status': 'failed', 'error': str(e)[-300:]}
            row['seconds'] = round(time.time() - t0, 1)
            with open(results_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            click.echo(f"{vname}/{cname}: {row.get('heldout_mae', 'FAIL')} "
                       f"({row['seconds']}s)")

    # Aggregate + verdict.
    rows = [json.loads(l) for l in open(results_path)]
    ok = [r for r in rows if r['status'] == 'ok']
    variants = sorted({r['variant'] for r in ok})
    x = np.array([np.mean([r['truth_mae'] for r in ok
                           if r['variant'] == v and r['kind'] == 'phantom'])
                  for v in variants])
    y = np.array([np.mean([r['heldout_mae'] for r in ok
                           if r['variant'] == v and r['kind'] == 'real'])
                  for v in variants])
    from scipy import stats
    rho = stats.spearmanr(x, y)
    click.echo(f'Spearman(phantom truth MAE, real heldout MAE): '
               f'rho={rho.statistic:.3f} p={rho.pvalue:.4f} over {len(variants)} variants')


if __name__ == '__main__':
    main()
