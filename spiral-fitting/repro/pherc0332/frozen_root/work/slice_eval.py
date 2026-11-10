"""Score spiral fits against the dense slice line annotations extracted from the old hand-traced VC segments
(/media/djosey/nvme4/fibers/PHerc0332_vcslices.volpkg.json: 51 lines, each the trace of ONE sheet across many windings
in one slice of the 7.91 um scan, mapped to the 2.399 um frame).

Each line lies on a single sheet, so in a correct fit its continuous winding coordinate
    W = (r_spiral - theta_unwrapped / 2pi * dr) / dr
(spiral space, theta unwrapped along the line) is CONSTANT along the line. The residual W - median(W) per line
measures the fit directly, without any model-vs-prediction convention: |res| < 0.25 = on the right sheet and
well inside the gap tolerance, |res| >= 0.5 = the fit assigns the point to a different winding than the rest of
the line (a sheet jump, a doubled or missing winding, or a core collapse).

    python slice_eval.py RUN_DIR [RUN_DIR ...] [--step 8] [--lines DIR] [--tsv OUT]
one row per run: lines used, points, pooled fraction |res| < 0.25 / < 0.5, median and p90 |res| (windings),
jumps (|dW| > 0.5 between consecutive points) per 1000 points, median per-line fraction < 0.25.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get('SPIRAL_ROOT', '/home/djosey/Documents/villa/spiral-fitting'))
import flatten_spiral_checkpoint as fsc, sample_spiral
from checkpoint_io import load_checkpoint_cpu

D = '/media/djosey/nvme4/spiral_0332'
LINES = '/media/djosey/nvme4/fibers/PHerc0332_vcslices.volpkg.json'


def load_lines(d, step, max_dr_per_rev=60.0, min_revs=1.0, verbose=True):
    """(seq, zyx L2) per line. Lines whose umbilicus radius changes by more than max_dr_per_rev voxels per
    revolution (a sheet moves ~20 vx/rev; a degenerate vcano row sweeps hundreds) or that span less than min_revs
    revolutions are excluded, independently of any fit."""
    u = json.load(open(f'{D}/dataset/umbilicus.json'))['control_points']
    uz = np.array([p['z'] for p in u]); o = np.argsort(uz); ux = np.array([p['x'] for p in u])[o]; uy = np.array([p['y'] for p in u])[o]; uz = uz[o]
    out = []; dropped = []
    for f in sorted(glob.glob(f'{d}/*.json')):
        j = json.load(open(f))
        L = np.array([p['position'] if isinstance(p, dict) else p for p in j['line_points']], float)[::step] / 4.0  # L0 xyz -> L2
        seq = os.path.basename(f).rsplit('_', 1)[-1][:-5]
        dx = L[:, 0] - np.interp(L[:, 2], uz, ux); dy = L[:, 1] - np.interp(L[:, 2], uz, uy)
        theta = np.unwrap(np.arctan2(dy, dx)); r = np.hypot(dx, dy); revs = (theta.max() - theta.min()) / (2 * np.pi)
        slope = abs(np.polyfit(theta / (2 * np.pi), r, 1)[0]) if revs > 0.1 else np.inf
        if revs < min_revs or slope > max_dr_per_rev:
            dropped.append(f'{seq} (revs {revs:.1f}, {slope:.0f} vx/rev)'); continue
        out.append((seq, np.ascontiguousarray(L[:, ::-1])))  # (seq, zyx)
    if verbose and dropped:
        print(f'excluded {len(dropped)} degenerate lines: ' + ', '.join(dropped))
    return out


def score(run, lines, min_points=200, per_line_out=None):
    ck = load_checkpoint_cpu(f'{run}/checkpoint_fitted.ckpt'); cfg = fsc._checkpoint_config(ck)
    z0, z1 = ck['z_begin'], ck['z_end']
    m = fsc._build_model(ck, cfg, fsc.Path(f'{D}/dataset/umbilicus.json'), torch.device('cpu'))
    t = m.get_slice_to_spiral_transform(); dr = float(m.get_dr_per_winding().detach())
    res_all = []; jumps = 0; npts = 0; per_line = []; off_model = 0
    for seq, zyx in lines:
        s = (zyx[:, 0] >= z0 + 2) & (zyx[:, 0] < z1 - 2)
        P = zyx[s]
        if len(P) < min_points:
            continue
        with torch.inference_mode():
            sp = torch.cat([t(torch.tensor(P[i:i + 20000], dtype=torch.float32)) for i in range(0, len(P), 20000)])
        theta, radius = sample_spiral.get_theta(sp[:, 1:])[0].numpy(), torch.linalg.norm(sample_spiral.get_theta(sp[:, 1:])[1], dim=-1).numpy()
        W = (radius - np.unwrap(theta) / (2 * np.pi) * dr) / dr
        res = W - np.median(W)
        # points the model maps more than 3 windings away from the rest of the line are outside the modelled
        # sheet stack (beyond the outer shell the flow is unconstrained); counted, not scored
        off = np.abs(res) > 3
        off_model += int(off.sum()); res = res[~off]; W = W[~off]
        res_all.append(res); npts += len(res); jumps += int(np.sum(np.abs(np.diff(W)) > 0.5))
        per_line.append(float(np.mean(np.abs(res) < 0.25)))
        if per_line_out is not None:
            per_line_out.append(dict(seq=seq, z=float(P[:, 0].mean()), points=len(P), f025=per_line[-1],
                                     f05=float(np.mean(np.abs(res) < 0.5)), sd=float(np.std(res)),
                                     jumps=int(np.sum(np.abs(np.diff(W)) > 0.5)), Wmed=float(np.median(W))))
    if not res_all:
        return None
    r = np.abs(np.concatenate(res_all))
    return dict(lines=len(per_line), points=npts, f025=float(np.mean(r < 0.25)), f05=float(np.mean(r < 0.5)),
                med=float(np.median(r)), p90=float(np.percentile(r, 90)), jumps_per_1k=1000.0 * jumps / npts,
                line_med_f025=float(np.median(per_line)), z=(z0, z1), off_model=off_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--step', type=int, default=8); ap.add_argument('--lines', default=LINES)
    ap.add_argument('--tsv'); ap.add_argument('--per-line', help='TSV with one row per (run, line)')
    a = ap.parse_args()
    lines = load_lines(a.lines, a.step)
    rows = []; per_line = []
    print(f'{"run":36s} {"z":>10s} {"lines":>5s} {"pts":>7s} {"<0.25":>6s} {"<0.5":>6s} {"med":>6s} {"p90":>6s} {"jmp/1k":>7s} {"line<.25":>8s} {"off":>4s}')
    for run in a.runs:
        tag = os.path.basename(run.rstrip('/')).split('_', 4)[-1]
        pl = []
        s = score(run, lines, per_line_out=pl)
        per_line += [dict(run=tag, **p) for p in pl]
        if s is None:
            print(f'{tag:36s} no line inside the z range'); continue
        print(f'{tag:36s} {s["z"][0]:>4d}-{s["z"][1]:<5d} {s["lines"]:>5d} {s["points"]:>7d} {s["f025"]:>6.1%} {s["f05"]:>6.1%} {s["med"]:>6.2f} {s["p90"]:>6.2f} {s["jumps_per_1k"]:>7.1f} {s["line_med_f025"]:>8.1%} {s["off_model"]:>4d}', flush=True)
        rows.append((tag, s))
    if a.tsv:
        with open(a.tsv, 'w') as f:
            f.write('run\tz0\tz1\tlines\tpoints\tfrac_lt_0.25\tfrac_lt_0.5\tmed_abs\tp90_abs\tjumps_per_1k\tline_median_frac_lt_0.25\n')
            for tag, s in rows:
                f.write(f'{tag}\t{s["z"][0]}\t{s["z"][1]}\t{s["lines"]}\t{s["points"]}\t{s["f025"]:.4f}\t{s["f05"]:.4f}\t{s["med"]:.4f}\t{s["p90"]:.4f}\t{s["jumps_per_1k"]:.2f}\t{s["line_med_f025"]:.4f}\n')
    if a.per_line and per_line:
        keys = list(per_line[0])
        with open(a.per_line, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for p in per_line:
                f.write('\t'.join(f'{p[k]:.4f}' if isinstance(p[k], float) else str(p[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
