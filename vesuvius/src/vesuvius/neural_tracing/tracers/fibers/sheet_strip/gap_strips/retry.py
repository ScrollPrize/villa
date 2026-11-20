"""Retry fibers for chains infer_all could not seed at their first point.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.retry CHAINS.vcobj RETRY.vcobj ID [ID ...]

The strip of a chain (mirror.py) starts on the mirrored extension, which can lie outside the scanned material; the
tracer then fails or skips that fiber. The retry fiber is the chain without its leading extension, so the strip starts
on the fiber's own first point (that end then gets the tracer's normal end trim); the trailing extension is kept.
UUIDs are unchanged and RETRY.vcobj.joined.json holds the shifted point indices, so split.py works on the retry run
as on the first one.
"""
import argparse
import json

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume, read_polylines, write_fibers


def without_leading_extension(entry, points):
    """A chain's joined.json entry and points with the leading mirrored extension cut off."""
    entry = dict(entry)
    ext = list(entry['ext'])
    cut = ext[0][1]+1 if ext and ext[0][0] == 0 else 0
    entry['pieces'] = [dict(s, start=s['start']-cut, end=s['end']-cut) for s in entry['pieces']]
    entry['ext'] = [[lo-cut, hi-cut] for lo, hi in ext if lo >= cut]
    entry['points'] = len(points)-cut
    entry['leading_extension_removed'] = cut
    return entry, points[cut:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chains')
    ap.add_argument('out')
    ap.add_argument('ids', nargs='+')
    a = ap.parse_args(argv)
    chain_log = {e['id']: e for e in json.load(open(a.chains+'.joined.json'))}
    rows, log = [], []
    for row, points in read_polylines(a.chains, a.ids):
        entry, points = without_leading_extension(chain_log[row['id']], points)
        meta = {k: v for k, v in row['meta'].items() if k != 'coordinates'}
        rows.append((dict(id=row['id'], name=row['name'], family=row['family'], meta=meta), points))
        log.append(entry)
    print(write_fibers(a.out, rows, volume=fiber_volume(a.chains)) | dict(ids=len(rows)))
    json.dump(log, open(a.out+'.joined.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
