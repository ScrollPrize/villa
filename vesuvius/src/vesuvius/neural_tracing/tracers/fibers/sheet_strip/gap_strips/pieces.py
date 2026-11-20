"""A fiber store of the valid pieces of a store's fibers, one fiber per piece.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.pieces SOURCE.vcobj PIECES.vcobj \\
        [--only NAMES.txt] [--spacing 2]

Each fiber is cut into its valid pieces (spans.py): no piece bridges a gap/damaged span and none reaches past the
outermost control points. A piece is the line points from its first to its last control point, densified to
--spacing volume voxels, written as the fiber '<name>#piece<k>' with a UUID derived from the source fiber's;
its ``meta.gap_strips`` records the source fiber and version, the piece number and its control-index run.
PIECES.vcobj is bound to the source's volume, so ``infer_all --fibers PIECES.vcobj`` traces it.
--only: a text file of fiber UUIDs or names, one per line (default: every fiber).
"""
import argparse

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import (
    fiber_row, pieces_of, read_annotated, read_only_list)
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers


def piece_row(piece, reference):
    """The write_fibers row and points of a piece (spans.pieces_of)."""
    return (fiber_row(piece, 'piece', reference, piece=piece['piece'], controls=piece['controls'],
                      pieces=piece['pieces']), piece['points'])


def write_derived(path, rows, volume, what):
    """Write a derived store; an empty one is refused."""
    if not rows:
        raise SystemExit(f'No {what} to write to {path}: the selection has no fiber with a valid piece')
    result = write_fibers(path, rows, volume=volume)
    print(f"{path}: {result['fibers']} {what}, {result['points']} points")


def add_source_arguments(ap):
    ap.add_argument('--only', help='text file of fiber UUIDs or names, one per line')
    ap.add_argument('--spacing', type=float, default=2.0, help='volume voxels between the written line points')


def read_source(args):
    try:
        return read_annotated(args.source, read_only_list(args.only))
    except ValueError as error:
        raise SystemExit(str(error))


def pieces_or_exit(fiber, spacing):
    try:
        return pieces_of(fiber, spacing)
    except ValueError as error:
        raise SystemExit(str(error))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('out')
    add_source_arguments(ap)
    args = ap.parse_args(argv)
    volume, fibers = read_source(args)
    rows = [piece_row(piece, fiber['reference']) for fiber in fibers for piece in pieces_or_exit(fiber, args.spacing)]
    write_derived(args.out, rows, volume, 'pieces')


if __name__ == '__main__':
    main()
