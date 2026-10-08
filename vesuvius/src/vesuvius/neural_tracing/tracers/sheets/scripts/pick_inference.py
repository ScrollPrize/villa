"""Pick an inference configuration (front points x confidence threshold) from trace-benchmark reports of one model on
the same seeds: a table of coverage, precision, real switches and lost traces per configuration, ranked by
``covered - PENALTY * off-sheet kept points`` (scored points; a kept point on the wrong sheet costs PENALTY right
ones). Wider fronts cover more surface per trace, so the count, not the coverage fraction, is what is ranked.

  python scripts/pick_inference.py --model NAME output/trace_bench/sweep_* [--json best.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PENALTY = 10.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', type=Path)
    ap.add_argument('--model', required=True)
    ap.add_argument('--penalty', type=float, default=PENALTY)
    ap.add_argument('--json', type=Path, help='write the chosen configuration here')
    args = ap.parse_args(argv)
    rows = []
    for folder in args.folders:
        report = json.loads((folder/'report.json').read_text())
        if args.model not in report['models']:
            continue
        s = report['models'][args.model]['summary']
        points = int(round(report['front']/60.))+1
        threshold = float(re.search(r'_t([\d.]+)$', folder.name).group(1))
        utility = s['points_covered']-args.penalty*s['off_kept_points']
        rows.append(dict(folder=str(folder), points=points, threshold=threshold, seeds=len(report['seeds']),
                         coverage=s['coverage_mean'], precision=s['precision_mean'], real_switches=s['traces_real_switch'],
                         lost=s['traces_lost'], covered=s['points_covered'], off_kept=s['off_kept_points'],
                         utility=utility))
    rows.sort(key=lambda r: -r['utility'])
    print(f"{'points':>6} {'thr':>5} {'coverage':>8} {'precision':>9} {'switches':>8} {'lost':>5} {'covered':>8} "
          f"{'off kept':>8} {'utility':>9}")
    for r in rows:
        print(f"{r['points']:6d} {r['threshold']:5.2f} {r['coverage']:8.3f} {r['precision']:9.3f} {r['real_switches']:8d} "
              f"{r['lost']:5d} {r['covered']:8d} {r['off_kept']:8d} {r['utility']:9.0f}")
    if args.json and rows:
        args.json.write_text(json.dumps(dict(best=rows[0], penalty=args.penalty, table=rows), indent=1))


if __name__ == '__main__':
    main()
