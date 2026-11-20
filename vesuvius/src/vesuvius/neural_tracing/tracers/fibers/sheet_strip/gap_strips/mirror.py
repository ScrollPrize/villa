"""Fiber stores of the valid pieces of a store's fibers and of their joined, mirror-extended chains.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.mirror SOURCE.vcobj PIECES.vcobj \\
        CHAINS.vcobj [--only NAMES.txt] [--join 75] [--ext 80] [--level-scale 4] [--spacing 2]

infer_all does not keep strip nodes within its --trim (75 patch voxels) of either end of a fiber, so a fiber cut into
short pieces at its gaps loses both ends of every piece. Here the pieces (as pieces.py makes them) of one fiber are
JOINED into a chain when the straight step from one piece's last point to the next piece's first point is at most
--join patch voxels; the excluded line points between them stay cut out. Each chain is then EXTENDED at both ends by
--ext patch voxels of arclength by point reflection through the end point (p' = 2 p_end - p: the line keeps its
direction, its bend is flipped); a chain shorter than --ext is reflected again about the new end until --ext is
reached. The tracer's end trim then removes about the mirrored part, and split.py drops every remaining mirrored or
joint column and cuts the chain strips back into one strip per piece.

  PIECES.vcobj  one fiber per piece, '<name>#piece<k>' (what the split strips refer to)
  CHAINS.vcobj  one fiber per chain with its extensions, '<name>#chain<c>' (what infer_all traces)

A chain's ``meta.gap_strips`` records, besides its source: ``pieces`` (per piece its UUID, its version in
PIECES.vcobj (``reference``), number and inclusive [start, end] point indices in the chain), ``ext`` (the inclusive point index ranges of the extensions) and
``level_scale``. --join and --ext are in patch voxels; stores hold volume voxels, --level-scale volume voxels per
patch voxel, which must be the scale infer_all traces at (its --ct-grid-scale).
"""
import argparse

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.pieces import (
    add_source_arguments, piece_row, pieces_or_exit, read_source, write_derived)
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import derived_id, fiber_row
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_catalog


def arclen(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def extension(points, length):
    """Points beyond points[-1] (moving away from the line) by repeated point reflection, ``length`` of arclength
    long (the units of ``points``): the last point is cut to that length exactly."""
    if not np.isfinite(length) or length < 0:
        raise ValueError('The extension length must be finite and not negative')
    out, S, have = [], points, 0.0
    if len(points) < 2 or not arclen(points) > 0 or length == 0:
        return np.zeros((0, 3))
    while have < length:
        R = 2*S[-1]-S[::-1][1:]  # S reflected through its last point, ordered outward
        prev = S[-1]
        for q in R:
            step = float(np.linalg.norm(q-prev))
            if have+step >= length:
                if step > 0:
                    out.append(prev+(q-prev)*((length-have)/step))
                have = length
                break
            have += step
            prev = q
            out.append(q)
        S = np.concatenate([S[-1:], R])
    return np.array(out)


def chains_of(pieces, join):
    """Consecutive pieces (spans.pieces_of) of one fiber grouped into chains: a piece joins the chain before it when
    the step between them is at most ``join``."""
    chains = [[pieces[0]]]
    for piece in pieces[1:]:
        if float(np.linalg.norm(piece['points'][0]-chains[-1][-1]['points'][-1])) <= join:
            chains[-1].append(piece)
        else:
            chains.append([piece])
    return chains


def extended_chain(chain, ext):
    """A chain's points with both mirrored extensions, its pieces' point spans in them, the extension index ranges
    and the chain's own arclength."""
    core = np.concatenate([piece['points'] for piece in chain])
    pre, post = extension(core[::-1], ext)[::-1], extension(core, ext)
    points = np.concatenate([pre, core, post])
    off, spans = len(pre), []
    for piece in chain:
        spans.append(dict(id=piece['id'], piece=piece['piece'], start=off, end=off+len(piece['points'])-1))
        off += len(piece['points'])
    ranges = [r for r in ([0, len(pre)-1], [off, len(points)-1]) if r[1] >= r[0]]
    return points, spans, ranges, arclen(core)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('pieces_out')
    ap.add_argument('chains_out')
    add_source_arguments(ap)
    ap.add_argument('--join', type=float, default=75.0, help='patch voxels: join pieces no farther apart than this')
    ap.add_argument('--ext', type=float, default=80.0, help='patch voxels of mirrored line added at each chain end')
    ap.add_argument('--level-scale', type=float, default=4.0, help='volume voxels per patch voxel')
    a = ap.parse_args(argv)
    scale = a.level_scale
    if not (np.isfinite(scale) and scale > 0 and a.join >= 0 and a.ext >= 0):
        raise SystemExit('--level-scale must be positive; --join and --ext must not be negative')
    volume, fibers = read_source(a)
    piece_rows, chain_entries, joins = [], [], 0
    for fiber in fibers:
        pieces = pieces_or_exit(fiber, a.spacing)
        if not pieces:
            continue
        piece_rows += [piece_row(piece, fiber['reference']) for piece in pieces]
        for c, chain in enumerate(chains_of(pieces, a.join*scale)):
            points, spans, ranges, _ = extended_chain(chain, a.ext*scale)
            entry = dict(id=derived_id(fiber['id'], f'chain{c}'), name=f"{fiber['name']}#chain{c}",
                         family=fiber['family'], source=fiber['id'])
            chain_entries.append((entry, fiber['reference'], dict(chain=c, pieces=spans, ext=ranges, join=a.join,
                                                                  ext_length=a.ext, level_scale=scale), points))
            joins += len(chain)-1
    write_derived(a.pieces_out, piece_rows, volume, 'pieces')
    # A chain names the piece versions just written: split.py cuts only with those.
    written = {row['id']: row['reference'] for row in fiber_catalog(a.pieces_out)}
    chain_rows = []
    for entry, reference, fields, points in chain_entries:
        fields['pieces'] = [dict(s, reference=written[s['id']]) for s in fields['pieces']]
        chain_rows.append((fiber_row(entry, 'chain', reference, **fields), points))
    write_derived(a.chains_out, chain_rows, volume, 'chains')
    print(f'{len(fibers)} fibers -> {len(piece_rows)} pieces, {len(chain_rows)} chains ({joins} joins)')


if __name__ == '__main__':
    main()
