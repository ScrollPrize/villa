"""Stylised visualisation of a find_inconsistent_windings.py result.

find_inconsistent_windings traces a high-level graph -- quadmesh *patches* are
nodes, point-collections (pcls) are edges where their attached points link two
patches -- outward from a seed patch, accumulating relative winding numbers and
voting (via absolute-winding anchors) on the seed patch's absolute winding. This
tool draws that graph.

The recorded subgraph is the union of every voter's path: a tree rooted at the
seed patch, whose edges are the relative-winding pcls actually traversed and
whose terminal patches carry the absolute anchors that voted.

Layout (the stylised picture the result implies):

  * x-axis = absolute model winding number. A patch's *entry* point sits at
    `seed_model_winding - acc`, where acc = winding(seed) - winding(entry) is
    rebuilt by walking the path hops (acc -= intra_patch_strip_delta + edge_delta
    per hop, exactly as the BFS accumulates it). Integer x are the theta=0 seams.
  * y = one row per patch. Rows are ordered by BFS depth (the seed on top); the
    patches at a given depth each take their own slot, stacked just below one
    another, so a depth occupies several slightly-offset rows and no two patches
    ever share a row. Depth bands are shaded alternately and depth numbers run
    down the right edge.
  * Each patch is a short horizontal line spanning the winding extent of its
    known points (entry, the departure point of each outgoing edge, any anchors),
    with a crimson tick at each integer winding (theta=0 seam) genuinely crossed
    *between* those known points. The crossings come from the integer seam-count
    deltas recorded on the within-patch strips (intra_patch_strip_delta /
    anchor_to_entry_delta), so a patch with a single known winding-point shows
    none. (A short min-width line is still drawn for such a patch so it stays
    visible, but it carries no tick.) Its full patch id is listed in a left-hand
    column, aligned to its row.
  * Each relative-pcl edge runs from the parent's departure point (entry winding +
    intra_patch_strip_delta) to the child's entry point; its horizontal run equals
    the edge's winding delta (e minus d), so it stays near-vertical in x.
  * Each absolute anchor is a star on its patch at `entry - anchor_to_entry_delta`,
    coloured by the seed winding its annotation implies, so disagreeing votes show
    up as different colours scattered across the tree.

Two outputs are written from the same geometry: a static PNG (matplotlib) and an
interactive HTML5 page. In the HTML the main plot region scrolls while the patch
names (left), absolute-winding axis (bottom), title (top) and BFS-depth scale
(right) stay pinned. The plot is an SVG whose patches, edges and anchors are each
their own element carrying data-* attributes, so mouse events on individual lines
can be wired up later (a hover tooltip + click handler are already attached as a
starting point).

Run, e.g.:

    python plot_winding_graph.py --votes winding_votes.json --output winding_graph.png
"""

import os
import json
import math
import html as html_lib
from collections import defaultdict

import click
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


def load_graph(votes):
    """Rebuild the patch tree from a find_inconsistent_windings votes dict.

    Returns (nodes, edges, anchors, meta) where:
      nodes:   patch_id -> {'depth', 'acc'} (acc = winding(seed) - winding(entry))
      edges:   child_patch_id -> hop record (the parent edge; child is `to_patch`)
      anchors: list of voter dicts (each carries abs_patch_id / annotation / vote)
      meta:    handy scalars pulled off the top level
    """
    seed = votes['patch_id']
    all_voters = [v for vs in votes['votes'].values() for v in vs]

    nodes = {seed: {'depth': 0, 'acc': 0.0}}
    edges = {}
    for v in all_voters:
        acc = 0.0
        for hop_i, h in enumerate(v['path']):
            acc = acc - h['intra_patch_strip_delta'] - h['edge_winding_delta']
            child = h['to_patch']
            # BFS reaches each patch once, so this is idempotent across voters.
            nodes.setdefault(child, {'depth': hop_i + 1, 'acc': acc})
            edges.setdefault(child, h)

    meta = {
        'seed': seed,
        'seed_model_winding': votes['model_raw_winding_at_seed'],
        'dr_per_winding': votes['dr_per_winding'],
        'num_patches_reached': votes['num_patches_reached'],
        'num_direct_votes': votes['num_direct_votes'],
        'num_long_range_votes': votes['num_long_range_votes'],
        'votes_agree': votes['votes_agree'],
        'max_hops': votes['max_hops'],
    }
    return nodes, edges, all_voters, meta


def assign_slots(depth_to_nodes, node_extent, slot_height, depth_gap):
    """Give every patch its own row: one patch per slot, depth-major.

    Walks depths top-down; within a depth the patches are sorted by winding and
    each dropped into the next slot below (so a depth occupies several adjacent,
    slightly-offset rows and no two patches share a row). A `depth_gap` of extra
    space separates one depth's block of rows from the next.

    Returns (node_y, depth_band) where node_y maps patch_id -> y, and depth_band
    maps depth -> (y_top, y_bottom) of its block (for shading / depth ticks)."""
    node_y = {}
    depth_band = {}
    cur = 0.0  # grows downward; negated into y so the seed (depth 0) sits at y=0
    for depth in sorted(depth_to_nodes):
        ids = sorted(depth_to_nodes[depth], key=lambda p: node_extent[p][0])
        top = cur
        for pid in ids:
            node_y[pid] = -cur
            cur += slot_height
        depth_band[depth] = (-top + slot_height / 2, -(cur - slot_height) - slot_height / 2)
        cur += depth_gap
    return node_y, depth_band


def integer_ticks(lo, hi):
    """Integer values strictly inside (lo, hi) -- the theta=0 crossings."""
    return [n for n in range(int(np.ceil(lo)), int(np.floor(hi)) + 1) if lo < n < hi]


def _star_points(cx, cy, r):
    """Points string for a 5-point SVG star polygon centred at (cx, cy)."""
    ri = r * 0.45
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else ri
        pts.append(f'{cx + rr * math.cos(ang):.2f},{cy + rr * math.sin(ang):.2f}')
    return ' '.join(pts)


# CSS / JS kept as plain strings (lots of braces / template literals) so they
# don't fight with the f-strings that assemble the dynamic markup.
_HTML_CSS = r"""
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
body { display: flex; flex-direction: column; background: #fff; color: #222; }
.header { flex: none; padding: 8px 12px; border-bottom: 1px solid #ddd; }
.header .title { font-size: 13px; font-weight: 600; white-space: pre-line; line-height: 1.35; }
.legend { margin-top: 6px; font-size: 12px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
.legend .item { display: inline-flex; align-items: center; gap: 5px; }
.layout { flex: 1 1 auto; min-height: 0; display: grid; }
.leftcol, .rightcol, .bottomaxis { overflow: hidden; background: #fff; }
.plotwrap { overflow: auto; }
.leftcol { border-right: 1px solid #ddd; }
.rightcol { border-left: 1px solid #ddd; }
.bottomaxis { border-top: 1px solid #ddd; }
.corner { background: #fafafa; display: flex; align-items: center; justify-content: center;
          font-size: 11px; color: #666; padding: 2px 6px; text-align: center; }
svg.plot { display: block; background: #fff; }
svg.inner { display: block; will-change: transform; }
.band { fill: #f2f2f2; }
.seam { stroke: #ededed; stroke-width: 1; }
.edgeline { stroke: #999; stroke-width: 1.1; fill: none; }
.depdot { fill: #888; }
.hitdot { fill: transparent; }
.patchline { stroke: #000; stroke-width: 1.8; }
.cap { stroke: #000; stroke-width: 1.8; }
.patch.seed .patchline, .patch.seed .cap { stroke-width: 2.8; }
.seamtick { stroke: crimson; stroke-width: 1.4; }
.star { stroke: none; }
.votelabel { font-size: 10px; font-weight: 700; }
.hit { stroke: transparent; stroke-width: 12; fill: none; }
.lbl { font-size: 11px; fill: #555; dominant-baseline: middle; }
.lbl.seed { fill: #111; font-weight: 700; }
.dlbl { font-size: 11px; fill: #444; dominant-baseline: middle; }
.albl { font-size: 11px; fill: #333; }
.atick { stroke: #333; stroke-width: 1; }
.band, .seam, .votelabel, .lbl, .dlbl, .albl, .atick { pointer-events: none; }
.patch, .edge, .anchor, .dot { cursor: pointer; }
.patch.hl .patchline, .patch.hl .cap { stroke: #ff8c00; }
.edge.hl .edgeline { stroke: #ff8c00; stroke-width: 2.4; }
.anchor.hl .star { stroke: #ff8c00; stroke-width: 2; }
.dot.hl .depdot { fill: #ff8c00; }
.tooltip { position: fixed; pointer-events: none; display: none; z-index: 10;
           background: rgba(20,20,20,.92); color: #fff; font-size: 11px;
           padding: 5px 7px; border-radius: 4px; max-width: 420px; white-space: pre-line; }
"""

_HTML_JS = r"""
const wrap = document.querySelector('.plotwrap');
const li = document.querySelector('.leftcol .inner');
const ri = document.querySelector('.rightcol .inner');
const bi = document.querySelector('.bottomaxis .inner');
function sync() {
  const tx = -wrap.scrollLeft, ty = -wrap.scrollTop;
  li.style.transform = 'translateY(' + ty + 'px)';
  ri.style.transform = 'translateY(' + ty + 'px)';
  bi.style.transform = 'translateX(' + tx + 'px)';
}
wrap.addEventListener('scroll', sync);
sync();

const tip = document.querySelector('.tooltip');
function xyz(d) {
  if (d.x === undefined) return '\nxyz (not recorded in this votes file)';
  return '\nxyz ' + Math.round(+d.x) + ', ' + Math.round(+d.y) + ', ' + Math.round(+d.z);
}
function info(el) {
  const d = el.dataset;
  if (d.kind === 'patch') {
    const lo = +d.windingLo, hi = +d.windingHi;
    const w = (lo === hi) ? ('winding ' + lo)
                          : ('windings ' + lo + '–' + hi + '  (crosses theta=0)');
    return 'patch ' + d.patch + '\ndepth ' + d.depth + '   ' + w;
  }
  if (d.kind === 'edge')
    return 'edge ' + d.from + '\n  → ' + d.to + '\nΔwinding ' + d.edgeDelta + '   strip ' + d.stripDelta;
  if (d.kind === 'anchor')
    return 'abs anchor — pcl ' + d.pcl + ' "' + (d.pclName || '') + '"  point ' + d.point
         + '\nannotation ' + d.annotation + ' → seed ' + d.vote + xyz(d) + '\non patch ' + d.patch;
  if (d.kind === 'dot')
    return 'rel-pcl point — pcl ' + d.pcl + ' "' + (d.pclName || '') + '"  point ' + d.point
         + '\non patch ' + d.patch + xyz(d);
  return '';
}
// Per-element mouse handling. The data-* attributes are the hook for richer
// behaviour later; for now we highlight + show a tooltip and log clicks.
document.querySelectorAll('.patch, .edge, .anchor, .dot').forEach(el => {
  el.addEventListener('mouseenter', () => {
    el.classList.add('hl');
    tip.textContent = info(el);
    tip.style.display = 'block';
  });
  el.addEventListener('mousemove', e => {
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top = (e.clientY + 12) + 'px';
  });
  el.addEventListener('mouseleave', () => {
    el.classList.remove('hl');
    tip.style.display = 'none';
  });
  el.addEventListener('click', () => { console.log('clicked', el.dataset.kind, el.dataset); });
});
"""


def write_html(geom, output_html):
    """Emit the interactive HTML5 page (frozen panes + per-element SVG)."""
    nodes = geom['nodes']
    edges = geom['edges']
    voters = geom['voters']
    seed = geom['seed']
    w0 = geom['w0']
    ex = geom['entry_x']
    departure_x = geom['departure_x']
    anchors_by_patch = geom['anchors_by_patch']
    node_extent = geom['node_extent']
    node_raw_extent = geom['node_raw_extent']
    node_y = geom['node_y']
    depth_band = geom['depth_band']
    vote_color = geom['vote_color']
    vote_windings = geom['vote_windings']
    vote_counts = geom['vote_counts']
    node_id = geom['node_id']
    title = geom['title']
    x_lo, x_hi, y_min = geom['x_lo'], geom['x_hi'], geom['y_min']
    slot_height, depth_gap = geom['slot_height'], geom['depth_gap']
    annotate_votes, label_mode = geom['annotate_votes'], geom['label_mode']
    sx = geom['px_per_winding']
    row_px = geom['row_px']
    LW = geom['label_width']
    RW, BH = 92, 46

    # --- data -> pixel transform (SVG y grows downward; matplotlib y up). ---
    Xmin, Xmax = x_lo - 1, x_hi + 1
    Ymax, Ymin = slot_height, y_min - slot_height
    sy = row_px / slot_height

    def X(x):
        return (x - Xmin) * sx

    def Y(y):
        return (Ymax - y) * sy

    PW = (Xmax - Xmin) * sx
    PH = (Ymax - Ymin) * sy
    cap = 0.18 * row_px

    esc = html_lib.escape

    def a(s):
        return esc(str(s), quote=True)

    def xyz_attrs(zyx):
        """data-x/y/z (volume xyz) from a downsampled [z, y, x] triple, if present."""
        if not zyx or len(zyx) != 3:
            return ''
        z, y, x = zyx
        return f' data-x="{x:.2f}" data-y="{y:.2f}" data-z="{z:.2f}"'

    def odd_bands(width):
        """Shaded rects for odd depths, spanning `width` px -- reused per panel."""
        out = []
        for d, (yt, yb) in depth_band.items():
            if d % 2 == 1:
                ry = Y(yt + depth_gap / 2)
                rh = (yt - yb + depth_gap) * sy
                out.append(f'<rect class="band" x="0" y="{ry:.1f}" width="{width:.0f}" height="{rh:.1f}"/>')
        return out

    seam_lo, seam_hi = int(np.floor(x_lo)), int(np.ceil(x_hi))

    # ---------------------------------------------------------------- plot svg
    P = [f'<svg class="plot" width="{PW:.0f}" height="{PH:.0f}" '
         f'viewBox="0 0 {PW:.0f} {PH:.0f}" xmlns="http://www.w3.org/2000/svg">']
    P += odd_bands(PW)
    for n in range(seam_lo, seam_hi + 1):
        xx = X(n)
        P.append(f'<line class="seam" x1="{xx:.1f}" y1="0" x2="{xx:.1f}" y2="{PH:.1f}"/>')

    # edges: parent departure point -> child entry point.
    for child, h in edges.items():
        parent = h['from_patch']
        x1, y1 = X(departure_x[child]), Y(node_y[parent])
        x2, y2 = X(ex[child]), Y(node_y[child])
        P.append(f'<g class="edge" data-kind="edge" data-from="{a(parent)}" data-to="{a(child)}" '
                 f'data-edge-delta="{h["edge_winding_delta"]:g}" data-strip-delta="{h["intra_patch_strip_delta"]:g}">')
        P.append(f'<line class="hit" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        P.append(f'<line class="edgeline" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        P.append('</g>')

    # patches: horizontal line + end caps + crimson seam ticks.
    for pid in nodes:
        lo, hi = node_extent[pid]
        y = node_y[pid]
        xl, xr, yc = X(lo), X(hi), Y(y)
        is_seed = pid == seed
        # winding span of the known points (integers, post-rounding). A patch whose
        # span covers more than one winding crosses theta=0, so a single winding
        # number is meaningless -- the hover reports the range instead.
        rlo, rhi = node_raw_extent[pid]
        wlo, whi = int(round(rlo)), int(round(rhi))
        P.append(f'<g class="patch{" seed" if is_seed else ""}" data-kind="patch" data-patch="{a(pid)}" '
                 f'data-depth="{nodes[pid]["depth"]}" data-winding-lo="{wlo}" data-winding-hi="{whi}">')
        P.append(f'<line class="hit" x1="{xl:.1f}" y1="{yc:.1f}" x2="{xr:.1f}" y2="{yc:.1f}"/>')
        P.append(f'<line class="patchline" x1="{xl:.1f}" y1="{yc:.1f}" x2="{xr:.1f}" y2="{yc:.1f}"/>')
        for xe in (xl, xr):
            P.append(f'<line class="cap" x1="{xe:.1f}" y1="{yc - cap:.1f}" x2="{xe:.1f}" y2="{yc + cap:.1f}"/>')
        for n in integer_ticks(rlo, rhi):
            xx = X(n)
            P.append(f'<line class="seamtick" x1="{xx:.1f}" y1="{yc - cap * 1.3:.1f}" '
                     f'x2="{xx:.1f}" y2="{yc + cap * 1.3:.1f}"/>')
        P.append('</g>')

    # departure dots: where each relative-pcl edge leaves its parent patch. Drawn
    # after the patches so they sit on top and own the hover at that point; each
    # carries the source rel-pcl / point id and the point's volume xyz.
    for child, h in edges.items():
        parent = h['from_patch']
        cx, cy = X(departure_x[child]), Y(node_y[parent])
        P.append(f'<g class="dot" data-kind="dot" data-patch="{a(parent)}" data-to="{a(child)}" '
                 f'data-pcl="{a(h["rel_pcl_id"])}" data-pcl-name="{a(h.get("rel_pcl_name") or "")}" '
                 f'data-point="{a(h["from_point_id"])}"{xyz_attrs(h.get("from_zyx_downsampled"))}>')
        P.append(f'<circle class="hitdot" cx="{cx:.1f}" cy="{cy:.1f}" r="6"/>')
        P.append(f'<circle class="depdot" cx="{cx:.1f}" cy="{cy:.1f}" r="2.4"/>')
        P.append('</g>')

    # absolute anchors: stars coloured by the seed winding they imply.
    for pid, anchs in anchors_by_patch.items():
        yc = Y(node_y[pid])
        for ax_x, v in anchs:
            cx = X(ax_x)
            col = vote_color[v['expected_seed_winding_rounded']]
            P.append(f'<g class="anchor" data-kind="anchor" data-patch="{a(pid)}" '
                     f'data-pcl="{a(v["abs_pcl_id"])}" data-pcl-name="{a(v.get("abs_pcl_name") or "")}" '
                     f'data-point="{a(v["abs_point_id"])}" '
                     f'data-annotation="{v["abs_winding_annotation"]:g}" '
                     f'data-vote="{v["expected_seed_winding_rounded"]}"'
                     f'{xyz_attrs(v.get("abs_zyx_downsampled"))}>')
            P.append(f'<polygon class="star" points="{_star_points(cx, yc, 8)}" fill="{col}"/>')
            if annotate_votes:
                P.append(f'<text class="votelabel" x="{cx:.1f}" y="{yc - 11:.1f}" fill="{col}" '
                         f'text-anchor="middle">{v["abs_winding_annotation"]:g}&#8594;'
                         f'{v["expected_seed_winding_rounded"]}</text>')
            P.append('</g>')

    P.append('</svg>')

    # --------------------------------------------------------------- left col
    Lp = [f'<svg class="inner" width="{LW}" height="{PH:.0f}" xmlns="http://www.w3.org/2000/svg">']
    Lp += odd_bands(LW)
    if label_mode != 'none':
        for pid in nodes:
            txt = str(node_id[pid]) if label_mode == 'id' else pid
            cls = 'lbl seed' if pid == seed else 'lbl'
            Lp.append(f'<text class="{cls}" x="{LW - 8}" y="{Y(node_y[pid]):.1f}" '
                      f'text-anchor="end" data-patch="{a(pid)}">{esc(txt)}</text>')
    Lp.append('</svg>')

    # --------------------------------------------------------------- right col
    Rp = [f'<svg class="inner" width="{RW}" height="{PH:.0f}" xmlns="http://www.w3.org/2000/svg">']
    Rp += odd_bands(RW)
    for d, (yt, yb) in depth_band.items():
        mid = 0.5 * (yt + yb)
        Rp.append(f'<text class="dlbl" x="{RW / 2:.0f}" y="{Y(mid):.1f}" text-anchor="middle">{d}</text>')
    Rp.append('</svg>')

    # --------------------------------------------------------------- bottom axis
    Bp = [f'<svg class="inner" width="{PW:.0f}" height="{BH}" xmlns="http://www.w3.org/2000/svg">']
    for n in range(seam_lo, seam_hi + 1):
        xx = X(n)
        Bp.append(f'<line class="atick" x1="{xx:.1f}" y1="0" x2="{xx:.1f}" y2="7"/>')
        Bp.append(f'<text class="albl" x="{xx:.1f}" y="21" text-anchor="middle">{n}</text>')
    Bp.append('</svg>')

    # --------------------------------------------------------------- legend
    leg = [
        '<span class="item"><svg width="22" height="12"><line x1="1" y1="6" x2="21" y2="6" '
        'stroke="#000" stroke-width="2"/></svg>patch</span>',
        '<span class="item"><svg width="14" height="14"><line x1="7" y1="1" x2="7" y2="13" '
        'stroke="crimson" stroke-width="1.6"/></svg>theta=0 seam crossing</span>',
        '<span class="item"><svg width="22" height="12"><line x1="1" y1="6" x2="21" y2="6" '
        'stroke="#999" stroke-width="1.4"/></svg>relative-pcl edge</span>',
    ]
    for w in vote_windings:
        col = vote_color[w]
        leg.append(f'<span class="item"><svg width="16" height="16"><polygon '
                   f'points="{_star_points(8, 8, 7)}" fill="{col}"/></svg>seed = {w} '
                   f'({vote_counts[w]}&times; anchors)</span>')

    grid_style = (f'grid-template-columns:{LW}px 1fr {RW}px;'
                  f'grid-template-rows:1fr {BH}px;')
    html_doc = (
        '<!DOCTYPE html>\n'
        f'<html lang="en"><head><meta charset="utf-8">'
        f'<title>winding graph {esc(seed)}</title>\n'
        f'<style>{_HTML_CSS}</style></head>\n<body>\n'
        f'<div class="header"><div class="title">{esc(title)}</div>'
        f'<div class="legend">{"".join(leg)}</div></div>\n'
        f'<div class="layout" style="{grid_style}">\n'
        f'  <div class="leftcol">{"".join(Lp)}</div>\n'
        f'  <div class="plotwrap">{"".join(P)}</div>\n'
        f'  <div class="rightcol">{"".join(Rp)}</div>\n'
        f'  <div class="corner corner-bl">absolute winding &rarr;</div>\n'
        f'  <div class="bottomaxis">{"".join(Bp)}</div>\n'
        f'  <div class="corner corner-br">BFS depth</div>\n'
        f'</div>\n'
        f'<div class="tooltip"></div>\n'
        f'<script>{_HTML_JS}</script>\n'
        f'</body></html>\n'
    )
    with open(output_html, 'w') as f:
        f.write(html_doc)


@click.command()
@click.option('--votes', 'votes_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='winding_votes json written by find_inconsistent_windings.py.')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output image path (default: winding_graph_<patch>.png next to the votes json).')
@click.option('--seg-min-width', default=0.18, type=float,
              help='How far (in windings) each end of a patch line pokes out past its extreme known '
                   'points, so the ends clear the abs-winding seams (and a single-point patch is '
                   'still visible).')
@click.option('--slot-height', default=1.0, type=float,
              help='Vertical spacing between adjacent patch rows (each patch gets its own row).')
@click.option('--depth-gap', default=0.8, type=float,
              help='Extra vertical space inserted between one BFS depth''s block of rows and the next.')
@click.option('--annotate-votes/--no-annotate-votes', default=True,
              help='Label each absolute anchor with "annotation->vote" (the abs winding it carries '
                   'and the seed winding it then implies).')
@click.option('--label', 'label_mode', type=click.Choice(['patch', 'id', 'none']), default='patch',
              help='Left-hand column content per row: the patch id (folder name), a compact integer '
                   'id (id->patch mapping printed to stdout), or nothing.')
@click.option('--label-fontsize', default=6.0, type=float, help='Font size of the left-column labels.')
@click.option('--dpi', default=150, type=int, help='Output resolution.')
@click.option('--html/--no-html', 'want_html', default=True,
              help='Also write an interactive HTML5 page next to the PNG.')
@click.option('--html-output', default=None, type=click.Path(dir_okay=False),
              help='HTML output path (default: same base as the PNG with a .html extension).')
@click.option('--html-px-per-winding', default=44.0, type=float,
              help='Horizontal scale of the HTML plot, in pixels per winding.')
@click.option('--html-row-px', default=26.0, type=float,
              help='Vertical scale of the HTML plot, in pixels per patch row (slot-height units).')
@click.option('--html-label-width', default=260, type=int,
              help='Width (px) of the pinned left-hand patch-name column in the HTML.')
def main(votes_path, output, seg_min_width, slot_height, depth_gap, annotate_votes, label_mode,
         label_fontsize, dpi, want_html, html_output, html_px_per_winding, html_row_px, html_label_width):
    with open(votes_path) as f:
        votes = json.load(f)
    nodes, edges, voters, meta = load_graph(votes)

    seed = meta['seed']
    w0 = meta['seed_model_winding']

    # --- per-patch x coordinates: entry, the departure of each outgoing edge,
    #     and any anchors sitting on the patch (all as absolute model winding). ---
    def entry_x(pid):
        return w0 - nodes[pid]['acc']

    departure_x = {}      # child_patch -> x of the departure point on its parent
    point_xs = defaultdict(list)
    for pid in nodes:
        point_xs[pid].append(entry_x(pid))
    for child, h in edges.items():
        parent = h['from_patch']
        dx = entry_x(parent) + h['intra_patch_strip_delta']
        departure_x[child] = dx
        point_xs[parent].append(dx)
    anchors_by_patch = defaultdict(list)
    for v in voters:
        ax = entry_x(v['abs_patch_id']) - v['anchor_to_entry_delta']
        anchors_by_patch[v['abs_patch_id']].append((ax, v))
        point_xs[v['abs_patch_id']].append(ax)

    # patch segment extent [lo, hi]. node_raw_extent keeps the *unpadded* span of
    # the known points: theta=0 ticks are drawn over that, since the gaps between
    # known points are integer seam-crossing counts (intra_patch_strip_delta /
    # anchor_to_entry_delta), so integers strictly inside the raw span are exactly
    # the genuine crossings. node_extent pads both ends by seg_min_width so the line
    # always pokes out a little past its extreme known points (the abs-winding
    # seams) -- this is what gives a single-known-point patch a visible length, and
    # for a multi-winding patch it keeps the ends clear of the seam lines. Ticks use
    # the raw span, so the padding never fakes a crossing.
    node_extent = {}
    node_raw_extent = {}
    for pid in nodes:
        lo, hi = min(point_xs[pid]), max(point_xs[pid])
        node_raw_extent[pid] = (lo, hi)
        node_extent[pid] = (lo - seg_min_width, hi + seg_min_width)

    # --- y layout: one patch per row, depth-major (same-depth patches stacked) ---
    depth_to_nodes = defaultdict(list)
    for pid, nd in nodes.items():
        depth_to_nodes[nd['depth']].append(pid)
    node_y, depth_band = assign_slots(depth_to_nodes, node_extent, slot_height, depth_gap)

    # row order top-to-bottom (== the left-column label order); stable integer id.
    ordered = sorted(nodes, key=lambda p: -node_y[p])
    node_id = {pid: i for i, pid in enumerate(ordered)}

    # --- vote colour map: one colour per distinct rounded predicted seed winding.
    #     A hand-picked palette of deep, saturated colours -- all clearly legible
    #     (as both star and text) on a white background, no pale yellows/greens. ---
    DEEP_COLORS = [
        '#1f4e9b',  # blue
        '#1a7a3c',  # green
        '#7d2b8b',  # purple
        '#b5651d',  # ochre
        '#0d6b6b',  # teal
        '#8c2d5f',  # wine
        '#3b3b3b',  # near-black
        '#4b3fa3',  # indigo
    ]
    vote_windings = sorted({v['expected_seed_winding_rounded'] for v in voters})
    vote_color = {w: DEEP_COLORS[i % len(DEEP_COLORS)] for i, w in enumerate(vote_windings)}
    vote_counts = defaultdict(int)
    for v in voters:
        vote_counts[v['expected_seed_winding_rounded']] += 1

    # ------------------------------------------------------------------ figure
    x_lo = min(e[0] for e in node_extent.values())
    x_hi = max(e[1] for e in node_extent.values())
    y_min = min(node_y.values())
    y_span = -y_min + slot_height
    fig_w = max(16.0, (x_hi - x_lo) * 0.46)
    fig_h = max(8.0, (y_span + 2) * 0.17)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # alternate depth bands shaded so the per-depth blocks of rows read as groups.
    for d, (y_top, y_bot) in depth_band.items():
        if d % 2 == 1:
            ax.axhspan(y_bot - depth_gap / 2, y_top + depth_gap / 2,
                       color='0.95', zorder=-2)

    # theta=0 seam loci: faint vertical lines at every integer model winding.
    for n in range(int(np.floor(x_lo)), int(np.ceil(x_hi)) + 1):
        ax.axvline(n, color='0.90', lw=0.8, zorder=0)

    # edges: from the parent's departure point to the child's entry point.
    for child, h in edges.items():
        parent = h['from_patch']
        ax.plot([departure_x[child], entry_x(child)], [node_y[parent], node_y[child]],
                color='0.6', lw=0.9, alpha=0.8, zorder=1, solid_capstyle='round')
        # a small dot at the departure point on the parent (where the pcl leaves).
        ax.plot([departure_x[child]], [node_y[parent]], marker='o', ms=2.0, color='0.5', zorder=2)

    # patches: short horizontal lines with end-caps and theta=0 ticks.
    cap = 0.18 * slot_height
    for pid in nodes:
        lo, hi = node_extent[pid]
        y = node_y[pid]
        is_seed = pid == seed
        ax.plot([lo, hi], [y, y], color='black', lw=2.2 if is_seed else 1.6,
                solid_capstyle='butt', zorder=3)
        for xe in (lo, hi):
            ax.plot([xe, xe], [y - cap, y + cap], color='black',
                    lw=2.2 if is_seed else 1.6, zorder=3)
        # theta=0 discontinuities genuinely crossed by this patch -- ticked over
        # the RAW (unpadded) known-point span so the count/position reflect the
        # recorded seam crossings (a single-known-point patch shows none).
        rlo, rhi = node_raw_extent[pid]
        for n in integer_ticks(rlo, rhi):
            ax.plot([n, n], [y - cap * 1.3, y + cap * 1.3], color='crimson', lw=1.4, zorder=4)

    # absolute anchors: stars coloured by the seed winding their annotation implies.
    for pid, anchs in anchors_by_patch.items():
        for ax_x, v in anchs:
            y = node_y[pid]
            col = vote_color[v['expected_seed_winding_rounded']]
            ax.plot([ax_x], [y], marker='*', ms=11, color=col,
                    markeredgecolor='none', zorder=6)
            if annotate_votes:
                ax.annotate(f"{v['abs_winding_annotation']:g}→{v['expected_seed_winding_rounded']}",
                            (ax_x, y), textcoords='offset points', xytext=(0, 8),
                            ha='center', va='bottom', fontsize=6.5, color=col, zorder=7,
                            fontweight='bold')

    # left-hand column: one label per row, right-aligned just outside the y-axis.
    if label_mode != 'none':
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for pid in nodes:
            txt = str(node_id[pid]) if label_mode == 'id' else pid
            ax.text(-0.008, node_y[pid], txt, transform=trans, ha='right', va='center',
                    fontsize=label_fontsize, color='0.10' if pid == seed else '0.35',
                    fontweight='bold' if pid == seed else 'normal', zorder=5)

    ax.set_xlabel('absolute model winding number  (integer = theta=0 seam)')
    # BFS-depth scale on the right edge (the left edge is the patch-id column).
    depths_sorted = sorted(depth_band)
    ax.set_yticks([0.5 * (depth_band[d][0] + depth_band[d][1]) for d in depths_sorted])
    ax.set_yticklabels([str(d) for d in depths_sorted])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('BFS depth (hops from seed)')
    ax.set_xlim(x_lo - 1, x_hi + 1)
    ax.set_ylim(y_min - slot_height, slot_height)

    agree = 'AGREE' if meta['votes_agree'] else 'DISAGREE'
    title = (f"winding graph for seed {seed}\n"
             f"{len(nodes)} patches shown ({meta['num_patches_reached']} reached); "
             f"{meta['num_direct_votes']} direct + {meta['num_long_range_votes']} long-range votes; "
             f"votes {agree} ({', '.join(str(w) for w in vote_windings)}); "
             f"dr/winding={meta['dr_per_winding']:.3f}")
    ax.set_title(title, fontsize=10)

    # legend.
    handles = [Line2D([0], [0], color='black', lw=1.6, label='patch'),
               Line2D([0], [0], color='crimson', linestyle='none', marker='|', markersize=10,
                      markeredgewidth=1.4, label='theta=0 seam crossing'),
               Line2D([0], [0], color='0.6', lw=0.9, label='relative-pcl edge')]
    handles.append(Line2D([], [], linestyle='none', label='abs winding annotation implying seed winding ='))
    for w in vote_windings:
        handles.append(Line2D([0], [0], marker='*', color=vote_color[w], lw=0,
                              markeredgecolor='none', markersize=11,
                              label=f'     = {w}  ({vote_counts[w]}× anchors)'))
    # lower-left tends to be empty (the tree drifts down-right as winding grows).
    leg = ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.95)
    for txt in leg.get_texts():
        if txt.get_text().startswith('abs winding annotation'):
            txt.set_fontweight('bold')

    fig.tight_layout()
    if output is None:
        base = os.path.splitext(os.path.basename(votes_path))[0]
        output = os.path.join(os.path.dirname(os.path.abspath(votes_path)), f'{base}_graph.png')
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    print(f'wrote {output}  ({len(nodes)} patches, {len(edges)} edges, {len(voters)} votes)')

    if want_html:
        if html_output is None:
            html_output = os.path.splitext(os.path.abspath(output))[0] + '.html'
        geom = {
            'nodes': nodes, 'edges': edges, 'voters': voters, 'seed': seed, 'w0': w0,
            'entry_x': {pid: entry_x(pid) for pid in nodes},
            'departure_x': departure_x, 'anchors_by_patch': anchors_by_patch,
            'node_extent': node_extent, 'node_raw_extent': node_raw_extent,
            'node_y': node_y, 'depth_band': depth_band,
            'vote_color': vote_color, 'vote_windings': vote_windings, 'vote_counts': vote_counts,
            'node_id': node_id, 'title': title,
            'x_lo': x_lo, 'x_hi': x_hi, 'y_min': y_min,
            'slot_height': slot_height, 'depth_gap': depth_gap,
            'annotate_votes': annotate_votes, 'label_mode': label_mode,
            'px_per_winding': html_px_per_winding, 'row_px': html_row_px, 'label_width': html_label_width,
        }
        write_html(geom, html_output)
        print(f'wrote {html_output}')

    if label_mode == 'id':
        print('\npatch id legend (id: depth  winding  patch):')
        for pid in ordered:
            mark = ' [seed]' if pid == seed else (' [anchor]' if pid in anchors_by_patch else '')
            print(f'  {node_id[pid]:3d}: d{nodes[pid]["depth"]:<2d} '
                  f'w={entry_x(pid):6.2f}  {pid}{mark}')


if __name__ == '__main__':
    main()
