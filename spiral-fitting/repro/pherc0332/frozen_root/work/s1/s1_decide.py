"""Stage-1 decision (PLAN-S1-final section 6) from full-precision scores.json files written by s1_score.py.
Every statistic is recomputed EXACTLY (fractions.Fraction) from the per-unit counts; stored floats are never used.

    python s1/s1_decide.py --r0 R0/scores.json --r1 R1/scores.json [--r1b R1b/scores.json]
        [--r0s2 ... --r1s2 ... --r1bs2 ...] [--json OUT]

Statistics (all-51 slice set; anchors on the frozen record populations):
  f025, f025_cens   s1_score.slice_stats definitions (f025 = slice_eval SLICE f025, cens = off-model/outside as failures)
  held_correct      held anchors, pooled fraction accepted-and-|q-t|<0.5 (= 1 - pooled off-rate)
  held_patches      sum satisfied / sum frozen area
  held_off, train_off   equal-fiber mean off-rate of held / training fibers
Guards vs reference (all required): f025 and f025_cens >= ref - 0.2 pt; held_correct drop <= 0.01; held_patches
drop <= 0.005. Qualifying (at least one): (i) f025 >= +0.5 pt, (ii) held_off <= -3 pt, (iii) train_off <= -5 pt.
Boundaries are inclusive (+0.5 qualifies, -0.2 passes). Attribution: R1 vs R0, R1b vs R1 (reported only).
Adoption: an arm is adopted only if, against R0, it passes every guard and meets a qualifying improvement on seed 1
AND on seed 2 with the SAME statistic (missing seed-2 -> "pending confirmation", not adopted). If both R1 and R1b
are adoptable, R1b is the finalist only if R1b vs R1 also passes the guards and meets a qualifying improvement on
the same seeds; otherwise R1 (the simpler arm). Supporting only: paired bootstrap (2000 resamples, rng seed 0, 80%
interval) of each statistic's difference over its units (slice lines / fibers).
"""
import argparse, json, sys
from fractions import Fraction as F
import numpy as np

PT = F(1, 100)
GUARDS = (('f025', -F(2, 10) * PT), ('f025_cens', -F(2, 10) * PT), ('held_correct', -F(1, 100)), ('held_patches', -F(5, 1000)))
QUALIFY = (('i', 'f025', F(5, 10) * PT, +1), ('ii', 'held_off', 3 * PT, -1), ('iii', 'train_off', 5 * PT, -1))
MIN_POINTS = 200


def _lines(sc):
    return sc['slice']['windows']['all']['per_line'], sc['slice'].get('min_points', MIN_POINTS)


def _fibers(sc, key):
    a = sc.get(key)
    return None if a is None else {k: v for k, v in a['fibers'].items() if v['evaluated'] > 0}


def stats(sc):
    """Exact statistics of one scores.json (None where the section is absent)."""
    rows, mp = _lines(sc); sel = [r for r in rows if r['inside'] >= mp]
    out = dict(f025=F(sum(r['n025'] for r in sel), sum(r['scored'] for r in sel)),
               f025_cens=F(sum(r['n025'] for r in rows), sum(r['points'] for r in rows)))
    hp = sc['held_patches']['patches'].values()
    out['held_patches'] = sum(F(p['satisfied_area']) for p in hp) / sum(F(p['frozen_area']) for p in hp)
    for key, name in (('anchors_held', 'held_off'), ('anchors_train', 'train_off')):
        fb = _fibers(sc, key)
        out[name] = None if not fb else sum(F(f['off'], f['evaluated']) for f in fb.values()) / len(fb)
        if key == 'anchors_held':
            out['held_correct'] = None if not fb else 1 - F(sum(f['off'] for f in fb.values()), sum(f['evaluated'] for f in fb.values()))
    return out


def check_same_population(a, b):
    ra, rb = _lines(a)[0], _lines(b)[0]
    if [(r['seq'], r['points']) for r in ra] != [(r['seq'], r['points']) for r in rb]:
        raise SystemExit('slice line populations differ')
    if set(a['held_patches']['patches']) != set(b['held_patches']['patches']) or any(
            F(a['held_patches']['patches'][k]['frozen_area']) != F(b['held_patches']['patches'][k]['frozen_area']) for k in a['held_patches']['patches']):
        raise SystemExit('held patch populations differ')
    for key in ('anchors_held', 'anchors_train'):
        fa, fb = _fibers(a, key), _fibers(b, key)
        if fa is not None and fb is not None and {k: v['evaluated'] for k, v in fa.items()} != {k: v['evaluated'] for k, v in fb.items()}:
            raise SystemExit(f'{key} populations differ')


def compare(cand, ref):
    """Guards and qualifying improvements of cand vs ref. Returns dict with diffs, guard results, fired set."""
    check_same_population(cand, ref)
    sc, sr = stats(cand), stats(ref)
    diff = {k: (None if sc[k] is None or sr[k] is None else sc[k] - sr[k]) for k in sc}
    guards = {k: (None if diff[k] is None else diff[k] >= thr) for k, thr in GUARDS}
    fired = {q for q, k, thr, sign in QUALIFY if diff[k] is not None and sign * diff[k] >= thr}
    return dict(diff=diff, guards=guards, guards_ok=all(g is True for g in guards.values()), qualifying=fired,
                cand=sc, ref=sr)


def bootstrap(cand, ref, stat, n=2000, seed=0, level=0.8):
    """Paired bootstrap of the difference cand - ref of `stat` over its units (supporting only)."""
    rng = np.random.default_rng(seed)
    if stat in ('f025', 'f025_cens'):
        ra, rb = _lines(cand)[0], _lines(ref)[0]; mp = _lines(ref)[1]
        if stat == 'f025':
            keep = [i for i, (x, y) in enumerate(zip(ra, rb)) if x['inside'] >= mp and y['inside'] >= mp]
            num_a = np.array([ra[i]['n025'] for i in keep], float); den_a = np.array([ra[i]['scored'] for i in keep], float)
            num_b = np.array([rb[i]['n025'] for i in keep], float); den_b = np.array([rb[i]['scored'] for i in keep], float)
        else:
            num_a = np.array([r['n025'] for r in ra], float); den_a = np.array([r['points'] for r in ra], float)
            num_b = np.array([r['n025'] for r in rb], float); den_b = np.array([r['points'] for r in rb], float)
        est = lambda i: num_a[i].sum() / den_a[i].sum() - num_b[i].sum() / den_b[i].sum()
        m = len(num_a)
    else:
        key = {'held_off': 'anchors_held', 'train_off': 'anchors_train'}[stat]
        fa, fb = _fibers(cand, key), _fibers(ref, key)
        if not fa or not fb:
            return None
        names = sorted(fa)
        d = np.array([fa[k]['off'] / fa[k]['evaluated'] - fb[k]['off'] / fb[k]['evaluated'] for k in names])
        est = lambda i: d[i].mean(); m = len(d)
    vals = np.array([est(rng.integers(0, m, m)) for _ in range(n)])
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return dict(units=m, point=float(est(np.arange(m))), lo=float(lo), hi=float(hi))


def fmt(x, pct=True):
    return 'n/a' if x is None else (f'{float(x) * 100:+.6f} pt' if pct else f'{float(x):.6f}')


def report(title, c):
    print(f'== {title}')
    for k in ('f025', 'f025_cens', 'held_correct', 'held_patches', 'held_off', 'train_off'):
        print(f'   {k:13s} cand {fmt(c["cand"][k], False)}  ref {fmt(c["ref"][k], False)}  diff {fmt(c["diff"][k])}'
              + (f'  guard {"PASS" if c["guards"][k] else ("n/a" if c["guards"][k] is None else "FAIL")}' if k in c['guards'] else ''))
    print(f'   guards {"PASS" if c["guards_ok"] else "FAIL"}; qualifying {sorted(c["qualifying"]) or "none"}')


def decide(r0, r1, r1b=None, r0s2=None, r1s2=None, r1bs2=None, verbose=True, boot=True):
    """Returns {'adopted': arm or None, 'arms': {...}, 'attribution': {...}}."""
    arms = {'R1': (r1, r1s2), 'R1b': (r1b, r1bs2)}
    res = dict(arms={}, attribution={})
    for name, a, b in (('R1 vs R0', r1, r0), ('R1b vs R1', r1b, r1)):
        if a is not None and b is not None:
            res['attribution'][name] = c = compare(a, b)
            if verbose:
                report(f'attribution {name} (seed 1, not an adoption test)', c)
    for name, (s1, s2) in arms.items():
        if s1 is None:
            continue
        c1 = compare(s1, r0); c2 = compare(s2, r0s2) if (s2 is not None and r0s2 is not None) else None
        ok1 = c1['guards_ok'] and bool(c1['qualifying'])
        common = (c1['qualifying'] & c2['qualifying']) if c2 is not None and c2['guards_ok'] else set()
        status = ('rejected (seed 1)' if not ok1 else 'pending seed-2 confirmation' if c2 is None
                  else 'adoptable' if common else 'rejected (seed 2)')
        res['arms'][name] = dict(seed1=c1, seed2=c2, common=common, status=status)
        if verbose:
            report(f'{name} vs R0 seed 1', c1)
            if c2 is not None:
                report(f'{name} vs R0 seed 2', c2)
            print(f'   -> {name}: {status}' + (f' (statistic {sorted(common)} on both seeds)' if common else ''))
            if boot:
                for q, k, thr, sign in QUALIFY:
                    b = bootstrap(s1, r0, k)
                    if b:
                        print(f'   bootstrap {name}-R0 {k} (supporting): diff {b["point"]*100:+.4f} pt, 80% [{b["lo"]*100:+.4f}, {b["hi"]*100:+.4f}] over {b["units"]} units')
    ok = [n for n, v in res['arms'].items() if v['status'] == 'adoptable']
    adopted = ok[0] if len(ok) == 1 else None
    if len(ok) == 2:
        # finalist: R1b only if it beats R1 on its own, guards + qualifying with a common statistic on both seeds
        cb1 = res['attribution'].get('R1b vs R1')
        cb2 = compare(r1bs2, r1s2) if (r1bs2 is not None and r1s2 is not None) else None
        better = (cb1 is not None and cb2 is not None and cb1['guards_ok'] and cb2['guards_ok']
                  and bool(cb1['qualifying'] & cb2['qualifying']))
        adopted = 'R1b' if better else 'R1'
    res['adopted'] = adopted
    if verbose:
        print(f'VERDICT: {"adopt " + adopted if adopted else "adopt nothing (keep R0)"}'
              + ('' if adopted else '; ' + ', '.join(f'{n}: {v["status"]}' for n, v in res['arms'].items())))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for k in ('r0', 'r1', 'r1b', 'r0s2', 'r1s2', 'r1bs2'):
        ap.add_argument(f'--{k}', required=k in ('r0', 'r1'))
    ap.add_argument('--json'); ap.add_argument('--no-bootstrap', action='store_true')
    a = ap.parse_args()
    L = lambda p: json.load(open(p)) if p else None
    res = decide(L(a.r0), L(a.r1), L(a.r1b), L(a.r0s2), L(a.r1s2), L(a.r1bs2), boot=not a.no_bootstrap)
    if a.json:
        conv = lambda o: str(o) if isinstance(o, F) else sorted(o) if isinstance(o, set) else o
        json.dump(res, open(a.json, 'w'), indent=1, default=conv)


if __name__ == '__main__':
    main()
