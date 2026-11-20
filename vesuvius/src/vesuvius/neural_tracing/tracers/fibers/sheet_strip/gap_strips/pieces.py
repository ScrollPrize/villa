"""A fiber store of the valid pieces of a store's fibers, one fiber per piece.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.pieces SOURCE.vcobj PIECES.vcobj \\
        [--only NAMES.txt] [--spacing 2]

Each fiber is cut into its valid pieces (spans.py): no piece bridges a gap/damaged span and none reaches past the
outermost control points. A piece is the line points between its first and last control point, densified to
--spacing volume voxels, written as the fiber '<name>#piece<k>' with a UUID derived from the source fiber's;
``meta.gap_strips`` records the source fiber, the piece and its control-index run. PIECES.vcobj is bound to the
source's volume, so ``infer_all --fibers PIECES.vcobj`` traces it. PIECES.vcobj.pieces.json lists the pieces.
--only: a text file of fiber UUIDs or names, one per line (default: every fiber).
"""
import argparse
import json

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import (
    fiber_row, pieces_of, read_annotated, read_only_list)
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers


def piece_rows(fibers, spacing):
    """(write_fibers rows with points, log entries) of every valid piece of ``fibers``."""
    rows, log = [], []
    for fiber in fibers:
        for piece in pieces_of(fiber, spacing):
            rows.append((fiber_row(piece, fiber['reference'], piece=piece['piece'], controls=piece['controls'],
                                   pieces=piece['pieces']), piece['points']))
            log.append({k: piece[k] for k in ('id', 'name', 'source', 'piece', 'controls', 'pieces')}
                       | dict(points=len(piece['points'])))
    return rows, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('out')
    ap.add_argument('--only', help='text file of fiber UUIDs or names, one per line')
    ap.add_argument('--spacing', type=float, default=2.0, help='volume voxels between the written line points')
    args = ap.parse_args(argv)
    volume, fibers = read_annotated(args.source, read_only_list(args.only))
    rows, log = piece_rows(fibers, args.spacing)
    print(write_fibers(args.out, rows, volume=volume) | dict(ids=len(rows)))
    json.dump(log, open(str(args.out)+'.pieces.json', 'w'), indent=1)
    print(f'{len(fibers)} fibers -> {len(rows)} pieces')


if __name__ == '__main__':
    main()
