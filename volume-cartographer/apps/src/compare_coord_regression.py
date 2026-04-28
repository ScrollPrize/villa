#!/usr/bin/env python3
"""Compare two vc_coord_regression outputs with epsilon tolerance.

Each non-comment line is "TAG id v1 v2 v3 ...", all floats. Lines must
match 1:1 by (TAG, id). Any numeric field whose absolute diff exceeds
--abs-tol OR relative diff exceeds --rel-tol is flagged.

Exit 0 on pass, 1 on any mismatch.
"""
import argparse, sys

def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tag, idx = parts[0], parts[1]
            vals = [float(v) for v in parts[2:]]
            rows[(tag, idx)] = vals
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('current')
    p.add_argument('--abs-tol', type=float, default=1e-4)
    p.add_argument('--rel-tol', type=float, default=1e-5)
    p.add_argument('--max-show', type=int, default=10)
    args = p.parse_args()

    base = load(args.baseline)
    curr = load(args.current)

    missing = set(base) - set(curr)
    extra = set(curr) - set(base)
    if missing or extra:
        print(f"FAIL: key sets differ. missing: {len(missing)}, extra: {len(extra)}")
        return 1

    mismatches = []
    for key in base:
        b, c = base[key], curr[key]
        if len(b) != len(c):
            mismatches.append((key, 'len', b, c))
            continue
        for i, (bi, ci) in enumerate(zip(b, c)):
            adiff = abs(bi - ci)
            rdiff = adiff / max(abs(bi), 1e-20)
            if adiff > args.abs_tol and rdiff > args.rel_tol:
                mismatches.append((key, i, bi, ci, adiff, rdiff))

    if not mismatches:
        print(f"PASS: {len(base)} cases match (abs_tol={args.abs_tol}, rel_tol={args.rel_tol})")
        return 0

    print(f"FAIL: {len(mismatches)} mismatches (abs_tol={args.abs_tol}, rel_tol={args.rel_tol})")
    for m in mismatches[:args.max_show]:
        if len(m) == 4:
            (tag, idx), what, b, c = m
            print(f"  {tag} {idx}: {what} differ")
        else:
            (tag, idx), field, b, c, ad, rd = m
            print(f"  {tag} {idx}[{field}]: base={b:.9f} curr={c:.9f} abs={ad:.2e} rel={rd:.2e}")
    if len(mismatches) > args.max_show:
        print(f"  ... and {len(mismatches) - args.max_show} more")
    return 1

if __name__ == '__main__':
    sys.exit(main())
