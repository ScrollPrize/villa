"""Frozen common-point slice-line score (astra review 2026-10-26): every run is scored on the SAME fixed point set, so
runs with different z ranges (window vs full scroll) or that push points off the model are comparable.

    python slice_eval_frozen.py RUN_DIR [RUN_DIR ...] [--z 2902 5098] [--step 8] [--tsv OUT] [--per-line OUT]

Point set = the slice-line points (slice_eval.load_lines, --step) with L2 z in [z0, z1). For each run every point of the
set is scored: points outside the run's modelled z range or mapped > 3 windings from the line's median (off model)
are FAILURES (counted in the denominators of frac<0.25 / frac<0.5, excluded from med/p90). Jumps (|dW| > 0.5) are
counted only between consecutive scored points, never across a gap. Pooled fractions weight points; line_mean_* weight
every line equally. Default --z = the lines' full extent, which with full-scroll runs equals slice_eval's set.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get('WORK_ROOT', '/media/djosey/nvme4/spiral_0332/work')); sys.path.insert(0, os.environ.get('SPIRAL_ROOT', '/home/djosey/Documents/villa/spiral-fitting'))
import slice_eval as S
import flatten_spiral_checkpoint as fsc, sample_spiral
from checkpoint_io import load_checkpoint_cpu


def score_frozen(run, lines, z0f, z1f, per_line_out=None):
    ck = load_checkpoint_cpu(f'{run}/checkpoint_fitted.ckpt'); cfg = fsc._checkpoint_config(ck)
    z0, z1 = ck['z_begin'], ck['z_end']
    m = fsc._build_model(ck, cfg, fsc.Path(f'{S.D}/dataset/umbilicus.json'), torch.device('cpu'))
    t = m.get_slice_to_spiral_transform(); dr = float(m.get_dr_per_winding().detach())
    n_total = n_scored = n_outside = n_off = n025 = n05 = jumps = 0; res_scored = []; pl = []
    for seq, zyx in lines:
        sel = (zyx[:, 0] >= z0f) & (zyx[:, 0] < z1f)
        P = zyx[sel]
        if len(P) == 0:
            continue
        n_total += len(P)
        inside = (P[:, 0] >= z0 + 2) & (P[:, 0] < z1 - 2)
        ok = np.zeros(len(P), bool); res = np.full(len(P), np.inf); W = np.full(len(P), np.nan)
        if inside.sum() >= 2:
            Q = P[inside]
            with torch.inference_mode():
                sp = torch.cat([t(torch.tensor(Q[i:i + 20000], dtype=torch.float32)) for i in range(0, len(Q), 20000)])
            th = sample_spiral.get_theta(sp[:, 1:])
            theta, radius = th[0].numpy(), torch.linalg.norm(th[1], dim=-1).numpy()
            Wq = (radius - np.unwrap(theta) / (2 * np.pi) * dr) / dr
            rq = Wq - np.median(Wq); off = np.abs(rq) > 3
            idx = np.flatnonzero(inside)
            res[idx[~off]] = rq[~off]; W[idx[~off]] = Wq[~off]; ok[idx[~off]] = True
            n_off += int(off.sum())
        n_outside += int((~inside).sum())
        a = np.abs(res); f025 = float(np.mean(a < 0.25)); f05 = float(np.mean(a < 0.5))
        n025 += int((a < 0.25).sum()); n05 += int((a < 0.5).sum()); n_scored += int(ok.sum()); res_scored.append(a[ok])
        both = ok[1:] & ok[:-1]; j = int(np.sum(np.abs(np.diff(W))[both] > 0.5)); jumps += j
        pl.append(dict(seq=seq, z=float(P[:, 0].mean()), points=len(P), scored=int(ok.sum()), f025=f025, f05=f05, jumps=j,
                       Wmed=float(np.nanmedian(W)) if ok.any() else float('nan')))
    if n_total == 0:
        return None
    r = np.concatenate(res_scored) if res_scored else np.array([np.inf])
    if per_line_out is not None:
        per_line_out.extend(pl)
    return dict(lines=len(pl), points=n_total, scored=n_scored, outside=n_outside, off_model=n_off,
                f025=n025 / n_total, f05=n05 / n_total, med=float(np.median(r)), p90=float(np.percentile(r, 90)),
                jumps_per_1k=1000.0 * jumps / max(n_scored, 1), line_mean_f025=float(np.mean([p['f025'] for p in pl])),
                line_mean_f05=float(np.mean([p['f05'] for p in pl])), z=(z0, z1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('runs', nargs='+'); ap.add_argument('--step', type=int, default=8); ap.add_argument('--lines', default=S.LINES)
    ap.add_argument('--z', type=float, nargs=2, default=(-1e9, 1e9), help='frozen point set: L2 z range [z0, z1)')
    ap.add_argument('--tsv'); ap.add_argument('--per-line')
    a = ap.parse_args()
    lines = S.load_lines(a.lines, a.step)
    rows = []; per_line = []
    print(f'frozen set z [{a.z[0]:.0f}, {a.z[1]:.0f})')
    print(f'{"run":36s} {"model z":>10s} {"lines":>5s} {"pts":>7s} {"out":>5s} {"off":>4s} {"<0.25":>6s} {"<0.5":>6s} {"med":>6s} {"p90":>6s} {"jmp/1k":>7s} {"line<.25":>8s} {"line<.5":>7s}')
    for run in a.runs:
        tag = os.path.basename(run.rstrip('/')).split('_', 4)[-1]
        pl = []; s = score_frozen(run, lines, a.z[0], a.z[1], per_line_out=pl)
        if s is None:
            print(f'{tag:36s} no points'); continue
        per_line += [dict(run=tag, **p) for p in pl]; rows.append((tag, s))
        print(f'{tag:36s} {s["z"][0]:>4d}-{s["z"][1]:<5d} {s["lines"]:>5d} {s["points"]:>7d} {s["outside"]:>5d} {s["off_model"]:>4d} {s["f025"]:>6.1%} {s["f05"]:>6.1%} {s["med"]:>6.2f} {s["p90"]:>6.2f} {s["jumps_per_1k"]:>7.1f} {s["line_mean_f025"]:>8.1%} {s["line_mean_f05"]:>7.1%}')
    if a.tsv:
        with open(a.tsv, 'w') as f:
            f.write('run\tz0\tz1\tlines\tpoints\tscored\toutside\toff_model\tfrac_lt_0.25\tfrac_lt_0.5\tmed_abs\tp90_abs\tjumps_per_1k\tline_mean_frac_lt_0.25\tline_mean_frac_lt_0.5\n')
            for tag, s in rows:
                f.write(f'{tag}\t{a.z[0]:.0f}\t{a.z[1]:.0f}\t{s["lines"]}\t{s["points"]}\t{s["scored"]}\t{s["outside"]}\t{s["off_model"]}\t{s["f025"]:.4f}\t{s["f05"]:.4f}\t{s["med"]:.4f}\t{s["p90"]:.4f}\t{s["jumps_per_1k"]:.2f}\t{s["line_mean_f025"]:.4f}\t{s["line_mean_f05"]:.4f}\n')
    if a.per_line and per_line:
        keys = list(per_line[0])
        with open(a.per_line, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for p in per_line:
                f.write('\t'.join(f'{p[k]:.4f}' if isinstance(p[k], float) else str(p[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
