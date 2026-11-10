"""Stage-1 scoring of one run into a full-precision RUN_DIR/scores.json (PLAN-S1-final sections 2, 3, 4, 6).
No value is rounded; every statistic is stored beside the per-unit integer counts it is computed from, so
s1_decide.py can recompute it exactly.

    CUDA_VISIBLE_DEVICES= python s1/s1_score.py RUN_DIR [--out RUN_DIR/scores.json] [--step 8]
        [--train-records F] [--held-records F] [--eval-set F] [--no-anchors]

slice     slice_eval_frozen.score_frozen on the 51 slice lines (slice_eval.load_lines, --step) for the full extent
          ("all") and the frozen windows z [1900,2900), [2902,5098), [3350,4300). Per line: points (in window),
          inside (inside the model z range, z_begin+2 <= z < z_end-2), outside, off_model (inside but > 3 windings from
          the line median), scored = inside - off_model, n025 / n05 (scored points with |res| < 0.25 / < 0.5), jumps.
            f025      = sum n025 / sum scored over lines with inside >= min_points (200). This is slice_eval.py's SLICE
                        f025: off_model and outside points are NOT in the denominator, short lines are dropped.
            f025_cens = sum n025 / sum points over all lines = score_frozen's f025: off_model AND outside-z points
                        are failures in the denominator, no line is dropped. For runs sharing the z ROI the outside
                        count is run-invariant, so the gap f025 -> f025_cens measures off-model censoring.
            f05 / f05_cens likewise; jumps_per_1k = 1000 * sum jumps / sum scored over the f025 line set (score_frozen
            jumps: only between consecutive scored points).
held_patches  per patch of the frozen eval set: satisfied_area from RUN/satisfied_fitted.json (0 if absent) and the
          frozen area; fraction = sum satisfied / sum frozen.
anchors_train / anchors_held  frozen records {source_fiber, src_index, p (L2 xyz), wind_a, phi_ref, w_cont}, section-2
          rule at the run's model: phi, q = shifted_radius / dr from sample_spiral.get_theta_and_radii (float32, as in
          the fitter), t = round(w_cont - phi/2pi), rho = w_cont - phi/2pi - t (float64 from the float32 phi);
          accepted iff |rho| <= 0.4; off iff non-finite, not accepted, or |q - t| >= 0.5. Records with p z outside
          the model's [z_begin, z_end) are not evaluated (counted per fiber as outside_z), as in holdout_score.py.
          Per fiber: evaluated, off, rejected (finite, |rho| > 0.4), nonfinite, outside_z. Statistics: equal-fiber
          mean off-rate, pooled off-rate, correct_fraction = 1 - pooled off-rate, and the legacy median-centred
          agreement mean(|dev - median(dev)| < 0.5), dev = q - w_cont (finite points).
Anchor sections are null when their record file is absent (or with --no-anchors).
"""
import argparse, functools, hashlib, json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
torch.set_num_threads(min(8, torch.get_num_threads()))
WORK = os.environ.get('WORK_ROOT', '/media/djosey/nvme4/spiral_0332/work')
SPIRAL = os.environ.get('SPIRAL_ROOT', '/home/djosey/Documents/villa/spiral-fitting')
sys.path.insert(0, WORK); sys.path.insert(0, SPIRAL)
import slice_eval as S, slice_eval_frozen as SF
import flatten_spiral_checkpoint as fsc, sample_spiral

S1 = f'{WORK}/s1'
WINDOWS = {'all': (-1e9, 1e9), 'z1900_2900': (1900, 2900), 'z2902_5098': (2902, 5098), 'z3350_4300': (3350, 4300)}
MIN_POINTS = 200          # slice_eval.score's default line filter
RHO_MAX = 0.4
TWO_PI = 2 * np.pi


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def exact_count(frac, n):
    """Integer k from frac = mean of n booleans (= k/n correctly rounded), exact for any n < 2**50."""
    k = round(frac * n)
    assert abs(frac * n - k) < 1e-6, (frac, n)
    return int(k)


def slice_window(run, lines, ck, z0f, z1f):
    pl = []; s = SF.score_frozen(run, lines, z0f, z1f, per_line_out=pl)
    if s is None:
        return None
    zb, ze = ck['z_begin'], ck['z_end']
    rows = []; it = iter(pl)
    for seq, zyx in lines:
        P = zyx[(zyx[:, 0] >= z0f) & (zyx[:, 0] < z1f)]
        if len(P) == 0:
            continue
        p = next(it); assert p['seq'] == seq and p['points'] == len(P)
        inside = int(((P[:, 0] >= zb + 2) & (P[:, 0] < ze - 2)).sum())
        if inside < 2:  # score_frozen evaluates nothing on such a line
            inside_eval = 0
        else:
            inside_eval = inside
        rows.append(dict(seq=seq, points=len(P), inside=inside, outside=len(P) - inside, scored=p['scored'],
                         off_model=inside_eval - p['scored'], n025=exact_count(p['f025'], len(P)),
                         n05=exact_count(p['f05'], len(P)), jumps=p['jumps']))
    out = dict(z=[z0f, z1f], per_line=rows, **slice_stats(rows))
    # consistency with score_frozen's own pooled values
    assert out['points'] == s['points'] and out['off_model'] == s['off_model'] and out['scored'] == s['scored']
    assert out['f025_cens'] == s['f025'] and out['f05_cens'] == s['f05'], (out['f025_cens'], s['f025'])
    return out


def slice_stats(rows, min_points=MIN_POINTS):
    """Pooled statistics from per-line counts (also used by s1_decide.py)."""
    sel = [r for r in rows if r['inside'] >= min_points]
    sc = sum(r['scored'] for r in sel); pts = sum(r['points'] for r in rows)
    return dict(lines=len(rows), lines_f025=len(sel), points=pts, scored=sum(r['scored'] for r in rows),
                outside=sum(r['outside'] for r in rows), off_model=sum(r['off_model'] for r in rows),
                f025=sum(r['n025'] for r in sel) / sc if sc else None, f05=sum(r['n05'] for r in sel) / sc if sc else None,
                f025_cens=sum(r['n025'] for r in rows) / pts if pts else None,
                f05_cens=sum(r['n05'] for r in rows) / pts if pts else None,
                jumps_per_1k=1000.0 * sum(r['jumps'] for r in sel) / sc if sc else None)


def anchor_rule(phi, q, w_cont):
    """Section-2 rule. phi, q: float32 arrays from the transform; w_cont: float64. Returns (t, rho, accepted, off,
    nonfinite, rejected). t/rho are computed in float64 from the float32 angle (the value the fitter sees)."""
    phi = np.asarray(phi, np.float32).astype(np.float64); q = np.asarray(q, np.float32).astype(np.float64)
    w = np.asarray(w_cont, np.float64)
    nonfinite = ~(np.isfinite(phi) & np.isfinite(q) & np.isfinite(w))
    with np.errstate(invalid='ignore'):
        x = w - phi / TWO_PI; t = np.round(x); rho = x - t
        accepted = ~nonfinite & (np.abs(rho) <= RHO_MAX)
        off = ~accepted | (np.abs(q - t) >= 0.5)
    rejected = ~nonfinite & ~accepted
    return t, rho, accepted, off, nonfinite, rejected


def score_anchors(recs, model, ck):
    t_fn, dr = model
    z = np.array([r['p'][2] for r in recs]); inz = (z >= ck['z_begin']) & (z < ck['z_end'])
    P = torch.tensor(np.array([r['p'][::-1] for r in recs], np.float32)[inz])
    with torch.inference_mode():
        sp = torch.cat([t_fn(P[i:i + 20000]) for i in range(0, len(P), 20000)]) if len(P) else torch.zeros((0, 3))
        phi, _, shifted = sample_spiral.get_theta_and_radii(sp[:, 1:], dr)
        q = (shifted / dr).numpy(); phi = phi.numpy()
    w = np.array([r['w_cont'] for r in recs], np.float64)[inz]
    _, _, acc, off, nonf, rej = anchor_rule(phi, q, w)
    names = np.array([r['source_fiber'] for r in recs]); ev_names = names[inz]
    fibers = {}
    for f in sorted(set(names)):
        m = ev_names == f
        fibers[f] = dict(records=int((names == f).sum()), outside_z=int((names == f).sum() - m.sum()), evaluated=int(m.sum()),
                         off=int(off[m].sum()), rejected=int(rej[m].sum()), nonfinite=int(nonf[m].sum()))
    out = dict(fibers=fibers, **anchor_stats(fibers))
    fin = np.isfinite(q) & np.isfinite(w)
    dev = q.astype(np.float64)[fin] - w[fin]
    out['legacy_median_centred'] = float(np.mean(np.abs(dev - np.median(dev)) < 0.5)) if len(dev) else None
    return out


def anchor_stats(fibers):
    ev = [f for f in fibers.values() if f['evaluated'] > 0]
    n = sum(f['evaluated'] for f in ev); o = sum(f['off'] for f in ev)
    return dict(n_fibers=len(ev), evaluated=n, off=o, rejected=sum(f['rejected'] for f in ev),
                nonfinite=sum(f['nonfinite'] for f in ev), outside_z=sum(f['outside_z'] for f in fibers.values()),
                equal_fiber_off_rate=float(np.mean([f['off'] / f['evaluated'] for f in ev])) if ev else None,
                pooled_off_rate=o / n if n else None, correct_fraction=1 - o / n if n else None)


def held_patches(run, eval_set):
    frozen = json.load(open(eval_set))['patches']
    sp = f'{run}/satisfied_fitted.json'
    sat = {p['id']: p['satisfied_area'] for p in json.load(open(sp))['patches']} if os.path.exists(sp) else {}
    pp = {k: dict(satisfied_area=sat.get(k, 0.0), frozen_area=a, present=k in sat) for k, a in frozen.items()}
    return dict(eval_set=eval_set, eval_set_sha256=sha256(eval_set), satisfied_file_present=os.path.exists(sp),
                patches=pp, missing=sum(not p['present'] for p in pp.values()),
                fraction=sum(p['satisfied_area'] for p in pp.values()) / sum(p['frozen_area'] for p in pp.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run'); ap.add_argument('--out'); ap.add_argument('--step', type=int, default=8)
    ap.add_argument('--lines', default=S.LINES)
    ap.add_argument('--train-records', default=f'{S1}/anchors_train_records.json')
    ap.add_argument('--held-records', default=f'{S1}/anchors_held_records.json')
    ap.add_argument('--eval-set', default=f'{S1}/holdout_eval_set.json'); ap.add_argument('--no-anchors', action='store_true')
    a = ap.parse_args()
    run = a.run.rstrip('/'); out_path = a.out or f'{run}/scores.json'
    ckp = f'{run}/checkpoint_fitted.ckpt'
    ck = S.load_checkpoint_cpu(ckp)
    cached = functools.lru_cache(maxsize=1)(lambda p: ck)  # score_frozen reloads the checkpoint per window
    SF.load_checkpoint_cpu = lambda p: cached(str(p))
    man = f'{run}/manifest.json'
    res = dict(run=os.path.abspath(run), checkpoint_sha256=sha256(ckp), model_state_sha256=ck.get('model_state_sha256'),
               completed_steps=ck.get('completed_iterations'), z_begin=ck['z_begin'], z_end=ck['z_end'],
               manifest_hash=sha256(man) if os.path.exists(man) else None)
    lines = S.load_lines(a.lines, a.step)
    res['slice'] = dict(lines_dir=a.lines, step=a.step, min_points=MIN_POINTS,
                        windows={k: slice_window(run, lines, ck, *z) for k, z in WINDOWS.items()})
    res['held_patches'] = held_patches(run, a.eval_set)
    model = None
    for key, path in (('anchors_train', a.train_records), ('anchors_held', a.held_records)):
        if a.no_anchors or not os.path.exists(path):
            res[key] = None; print(f'{key}: skipped ({"--no-anchors" if a.no_anchors else path + " absent"})'); continue
        if model is None:
            m = fsc._build_model(ck, fsc._checkpoint_config(ck), fsc.Path(f'{S.D}/dataset/umbilicus.json'), torch.device('cpu'))
            model = (m.get_slice_to_spiral_transform(), m.get_dr_per_winding().detach())
        res[key] = dict(records_file=path, records_sha256=sha256(path), **score_anchors(json.load(open(path)), model, ck))
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1, allow_nan=False)
    for k, w in res['slice']['windows'].items():
        if w:
            print(f'SLICE {k:11s} f025 {w["f025"]!r} f025_cens {w["f025_cens"]!r} f05 {w["f05"]!r} '
                  f'jumps/1k {w["jumps_per_1k"]!r} off_model {w["off_model"]} outside {w["outside"]} lines {w["lines_f025"]}/{w["lines"]}')
    print(f'HELD_PATCHES {res["held_patches"]["fraction"]!r} ({len(res["held_patches"]["patches"])} patches, {res["held_patches"]["missing"]} missing)')
    for key in ('anchors_train', 'anchors_held'):
        x = res[key]
        if x:
            print(f'{key.upper()} equal_fiber_off {x["equal_fiber_off_rate"]!r} pooled_off {x["pooled_off_rate"]!r} '
                  f'legacy_median_centred {x["legacy_median_centred"]!r} n={x["evaluated"]} fibers={x["n_fibers"]} '
                  f'rejected={x["rejected"]} nonfinite={x["nonfinite"]} outside_z={x["outside_z"]}')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
