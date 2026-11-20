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

  PIECES.vcobj             one fiber per piece, '<name>#piece<k>' (what the split strips refer to)
  CHAINS.vcobj             one fiber per chain with its extensions, '<name>#chain<c>' (what infer_all traces)
  CHAINS.vcobj.joined.json per chain: its pieces' [start, end] point indices and the extension ranges ('ext')

--join and --ext are in patch voxels; stores hold volume voxels (--level-scale volume voxels per patch voxel).
"""
import argparse
import json

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.pieces import piece_rows
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import (
    derived_id, fiber_row, pieces_of, read_annotated, read_only_list)
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers


def arclen(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def extension(points, length):
    """Points beyond points[-1] (moving away from the line), by repeated point reflection, cut at ``length`` of
    arclength (the units of ``points``)."""
    out, S, have = [], points, 0.0
    if len(points) < 2:
        return np.zeros((0, 3))
    while have < length:
        R = 2*S[-1]-S[::-1][1:]  # S reflected through its last point, ordered outward
        prev = S[-1]
        for q in R:
            have += float(np.linalg.norm(q-prev))
            prev = q
            out.append(q)
            if have >= length:
                break
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
    ap.add_argument('--only', help='text file of fiber UUIDs or names, one per line')
    ap.add_argument('--join', type=float, default=75.0, help='patch voxels: join pieces no farther apart than this')
    ap.add_argument('--ext', type=float, default=80.0, help='patch voxels of mirrored line added at each chain end')
    ap.add_argument('--level-scale', type=float, default=4.0, help='volume voxels per patch voxel')
    ap.add_argument('--spacing', type=float, default=2.0, help='volume voxels between the written line points')
    a = ap.parse_args(argv)
    scale = a.level_scale
    volume, fibers = read_annotated(a.source, read_only_list(a.only))
    chain_rows, chain_log = [], []
    stats = dict(joins=0, chains=0, shorter_than_ext=0)
    for fiber in fibers:
        pieces = pieces_of(fiber, a.spacing)
        if not pieces:
            continue
        for c, chain in enumerate(chains_of(pieces, a.join*scale)):
            points, spans, ranges, length = extended_chain(chain, a.ext*scale)
            entry = dict(id=derived_id(fiber['id'], f'chain{c}'), name=f"{fiber['name']}#chain{c}",
                         family=fiber['family'], source=fiber['id'])
            chain_rows.append((fiber_row(entry, fiber['reference'], chain=c, pieces=[s['id'] for s in spans],
                                         join=a.join, ext=a.ext, level_scale=scale), points))
            chain_log.append(dict(id=entry['id'], name=entry['name'], source=fiber['id'], chain=c, points=len(points),
                                  length=round(length/scale, 1), pieces=spans, ext=ranges))
            stats['joins'] += len(chain)-1
            stats['chains'] += 1
            stats['shorter_than_ext'] += length < a.ext*scale
    rows, log = piece_rows(fibers, a.spacing)
    print(write_fibers(a.pieces_out, rows, volume=volume) | dict(ids=len(rows)))
    json.dump(log, open(str(a.pieces_out)+'.pieces.json', 'w'), indent=1)
    print(write_fibers(a.chains_out, chain_rows, volume=volume) | dict(ids=len(chain_rows)))
    json.dump(chain_log, open(str(a.chains_out)+'.joined.json', 'w'), indent=1)
    print(f'{len(fibers)} fibers -> {len(rows)} pieces, {len(chain_rows)} chains; {stats}')


if __name__ == '__main__':
    main()
