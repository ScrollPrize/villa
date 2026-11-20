"""Retry fibers for chains infer_all could not start.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.retry CHAINS.vcobj RETRY.vcobj ID [ID ...]

The strip of a chain (mirror.py) starts on the mirrored extension, which can lie outside the scanned material; the
tracer then fails or skips that fiber. The retry fiber is the chain without its leading extension, so the strip starts
on the fiber's own first point (that end then gets the tracer's normal end trim); the trailing extension is kept.
UUIDs and names are unchanged and ``meta.gap_strips`` holds the shifted point indices (and
``leading_extension_removed``, the number of points cut), so split.py works on the retry run as on the first one.
"""
import argparse

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.pieces import write_derived
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import META_KEY, read_derived
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume


def without_leading_extension(derived, points):
    """A chain's ``meta.gap_strips`` and points with the leading mirrored extension cut off."""
    derived = dict(derived)
    ext = list(derived['ext'])
    cut = ext[0][1]+1 if ext and ext[0][0] == 0 else 0
    derived['pieces'] = [dict(s, start=s['start']-cut, end=s['end']-cut) for s in derived['pieces']]
    derived['ext'] = [[lo-cut, hi-cut] for lo, hi in ext if lo >= cut]
    derived['leading_extension_removed'] = cut+derived.get('leading_extension_removed', 0)
    return derived, points[cut:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chains')
    ap.add_argument('out')
    ap.add_argument('ids', nargs='+')
    a = ap.parse_args(argv)
    try:
        chains = read_derived(a.chains, 'chain', a.ids)
    except ValueError as error:
        raise SystemExit(str(error))
    rows = []
    for row, points in chains.values():
        derived, points = without_leading_extension(row['gap_strips'], points)
        rows.append((dict(id=row['id'], name=row['name'], family=row['family'], meta={META_KEY: derived}), points))
    write_derived(a.out, rows, fiber_volume(a.chains), 'retry chains')


if __name__ == '__main__':
    main()
