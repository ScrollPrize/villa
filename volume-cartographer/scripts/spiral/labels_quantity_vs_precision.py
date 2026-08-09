"""USING the harness to answer an open-problems question: for a fixed spiral
fit, does annotation QUANTITY (arc coverage) or annotation PRECISION (label
position noise) matter more, and where is the cliff?

The open-problems writeup floats that "a smaller set of precise labels in hard
regions may be more useful than a larger set of approximate labels in easy
regions." This measures it directly, against exact truth.

Realistic calibrated phantoms (deformation matched to a real fit + out-of-
family slips), N seeds, grid of coverage x position-noise; each cell exported,
fitted with a fixed config, scored against exact truth (winding MAE). Writes
results.jsonl (resumable) and a heatmap PNG.

Result on record (2 seeds, reference fitter, calibrated phantom; avg winding
MAE vs truth, lower better):

           n=0    n=0.5  n=1    n=2    n=4      (label noise, voxels; dr=12)
  cov0.15  0.43   0.46   0.52   0.58   0.68
  cov0.90  0.15   0.16   0.21   0.27   0.35

Coverage 0.15->0.90 improves MAE 2.34x; noise 0->4vox worsens it 2.04x. Model-
free crossover: abundant+very-noisy (cov0.90, noise 4vox = 1/3 winding, MAE
0.35) BEATS sparse+perfect (cov0.15, noise 0, MAE 0.43). For this fitter,
coverage has more leverage than precision -- the opposite of the doc's hunch --
and below ~0.30 coverage no label precision rescues the fit.

Scope: reference fitter, NOT production fit_spiral (whose dense-surface and
between-sheet losses may weight precision differently); a hypothesis about
production labeling strategy, not a ruling. Not a formal iso-effort study, but
the crossover needs no cost model -- the noisier set also has more labels AND
wins outright.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import click
import numpy as np

SPIRAL = os.path.dirname(os.path.abspath(__file__))
CANON = ['--steps', '1200', '--reg-gap', '0.03', '--reg-flow', '1.0',
         '--reg-flow-smooth', '100']
# Deformation calibrated to the real PHerc0172 fit (see phantom_real_correlation).
CALIB = ['--flow-std', '0.0137', '--gap-log-std', '0.107', '--linear-std', '0.0046',
         '--num-slips', '3']
COVERAGE = [0.15, 0.30, 0.45, 0.60, 0.90]
NOISE = [0.0, 0.5, 1.0, 2.0, 4.0]


def run(cmd):
    p = subprocess.run([sys.executable, *cmd], cwd=SPIRAL, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[0]}: {p.stderr[-400:]}')


@click.command()
@click.option('--work', required=True, type=click.Path(file_okay=False))
@click.option('--seeds', default='101,202', help='Comma-separated phantom seeds.')
def main(work, seeds):
    os.makedirs(work, exist_ok=True)
    results = os.path.join(work, 'results.jsonl')
    seed_list = [int(s) for s in seeds.split(',')]

    done = set()
    if os.path.exists(results):
        for line in open(results):
            r = json.loads(line)
            done.add((r['seed'], r['coverage'], r['noise']))

    for seed in seed_list:
        ph = os.path.join(work, f'phantom_{seed}')
        if not os.path.exists(os.path.join(ph, 'phantom_checkpoint.pt')):
            run(['synth_phantom.py', '--out', ph, '--seed', str(seed),
                 '--z-size', '32', '--yx-size', '640', '--dr-per-winding', '12',
                 '--num-tears', '0', *CALIB])
            click.echo(f'phantom {seed} built')
        for cov in COVERAGE:
            for noise in NOISE:
                if (seed, cov, noise) in done:
                    continue
                t0 = time.time()
                ds = os.path.join(work, f'ds_{seed}_{cov}_{noise}')
                ckpt = os.path.join(work, 'tmp.pt')
                score = os.path.join(work, 'tmp.json')
                try:
                    run(['export_phantom_dataset.py', '--phantom', ph, '--out', ds,
                         '--step', '2', '--coverage', str(cov),
                         '--position-noise', str(noise), '--seed', str(seed)])
                    run(['fit_phantom_reference.py', '--dataset', ds, '--out', ckpt,
                         '--seed', '0', *CANON])
                    run(['winding_error.py', '--phantom', ph,
                         '--candidate-checkpoint', ckpt, '--num-points', '50000',
                         '--output', score])
                    s = json.load(open(score))
                    row = {'seed': seed, 'coverage': cov, 'noise': noise,
                           'status': 'ok', 'truth_mae': s['overall']['mae'],
                           'switch': s['overall']['switch_rate']}
                except Exception as e:
                    row = {'seed': seed, 'coverage': cov, 'noise': noise,
                           'status': 'failed', 'error': str(e)[-200:]}
                row['seconds'] = round(time.time() - t0, 1)
                shutil.rmtree(ds, ignore_errors=True)
                with open(results, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                click.echo(f"seed{seed} cov{cov} noise{noise}: "
                           f"{row.get('truth_mae', 'FAIL')} ({row['seconds']}s)")

    rows = [json.loads(l) for l in open(results) if json.loads(l)['status'] == 'ok']

    def cell(c, n):
        v = [r['truth_mae'] for r in rows
             if abs(r['coverage'] - c) < 1e-6 and abs(r['noise'] - n) < 1e-6]
        return float(np.mean(v)) if v else np.nan

    M = np.array([[cell(c, n) for n in NOISE] for c in COVERAGE])
    click.echo('\navg truth MAE, rows=coverage, cols=noise:')
    click.echo('        ' + ''.join(f'n={n:<5}' for n in NOISE))
    for i, c in enumerate(COVERAGE):
        click.echo(f'cov{c:<5}' + ' '.join(f'{M[i, j]:.3f}' for j in range(len(NOISE))))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(M, cmap='viridis_r', aspect='auto')
        ax.set_xticks(range(len(NOISE))); ax.set_xticklabels([str(n) for n in NOISE])
        ax.set_yticks(range(len(COVERAGE))); ax.set_yticklabels([str(c) for c in COVERAGE])
        ax.set_xlabel('label position noise (voxels)')
        ax.set_ylabel('annotation arc coverage')
        ax.set_title('fit error vs label quantity x precision\n(truth MAE, reference fitter; lower=better)')
        for i in range(len(COVERAGE)):
            for j in range(len(NOISE)):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='w')
        plt.colorbar(im, label='winding MAE vs exact truth')
        plt.tight_layout()
        out_png = os.path.join(work, 'labels_heatmap.png')
        plt.savefig(out_png, dpi=115)
        click.echo(f'wrote {out_png}')
    except ImportError:
        click.echo('(matplotlib unavailable; skipped heatmap)')


if __name__ == '__main__':
    main()
