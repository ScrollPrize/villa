"""Stylised visualisation of a find_inconsistent_windings.py result.

find_inconsistent_windings traces a high-level graph -- quadmesh *patches* are
nodes, point-collections (pcls) are edges where their attached points link two
patches -- outward from a seed patch, accumulating relative winding numbers and
voting (via absolute-winding anchors) on the seed patch's absolute winding. This
tool draws that graph.

The drawn subgraph is the union of every voter's path: a tree rooted at the seed
patch, whose edges are the relative-winding pcls actually traversed and whose
terminal patches carry the absolute anchors that voted.

find_inconsistent_windings also records "loops": non-tree relative edges whose
winding holonomy is nonzero (the cycle they close does not return to the same
winding -- an inconsistency in the relative annotations themselves). Rather than
overlay these on the (already busy) main graph, we render them in a separate
output, one loop at a time: each inconsistent loop is drawn on its own as a
vertical stack of the patches around its cycle, with the relative-pcl tree edges
as downward arrows between consecutive patches and one "closing" arrow bowing from
the bottom patch back up to the top (the non-tree edge whose holonomy makes the
loop inconsistent). Each pcl edge is labelled with its winding delta; each patch is
labelled (in crimson, to the right of its line) with the within-patch winding delta
between its two cycle points, when the strip across it crosses theta=0. The cycle
reconstruction (patches + pcls in order, with the per-patch strip deltas) is
precomputed by find_inconsistent_windings and stored under "loop_cycles" in the
votes json; this per-loop view is always produced alongside the main graph.

Layout (the stylised picture the result implies):

  * x-axis = absolute model winding number. A patch's *entry* point has winding
    `seed_model_winding - acc`, where acc = winding(seed) - winding(entry) is
    rebuilt by walking the path hops (acc -= intra_patch_strip_delta + edge_delta
    per hop, exactly as the BFS accumulates it). Every recorded winding delta is
    an integer seam count, so a point's winding number is the integer w0 - acc.
    Integer x are the theta=0 seams (the transitions between windings); a winding
    is the *range* between two consecutive seams. We therefore draw each point
    half a winding in from its seam -- at its band centre, x = winding + 0.5 --
    and label each winding number there too. Points (entries, departures,
    anchors) thus sit *inside* their winding, never on a boundary; the seams
    carry only tick marks.
  * y = one row per patch. Rows are ordered by BFS depth (the seed on top); the
    patches at a given depth each take their own slot, stacked just below one
    another, so a depth occupies several slightly-offset rows and no two patches
    ever share a row. Depth bands are shaded alternately and depth numbers run
    down the right edge.
  * Each patch is a short horizontal line spanning its known points (entry, the
    departure point of each outgoing edge, any anchors), each placed at its band
    centre. The line pokes seg_min_width past the extreme stars on each side, so it
    extends a little beyond its end points while staying inside the bounding
    windings (with the default it reaches the 1/4 and 3/4 marks of its first and
    last winding box); a single-winding patch becomes a short stub centred in its
    one band. A crimson tick marks each integer winding (theta=0 seam) genuinely
    crossed *between* those known points; the crossings come from the integer
    seam-count deltas recorded on the within-patch strips (intra_patch_strip_delta
    / anchor_to_entry_delta), so a single-winding patch shows none. Its full patch
    id is listed in a left-hand column, aligned to its row.
  * Each relative-pcl edge runs from the parent's departure point (entry winding +
    intra_patch_strip_delta) to the child's entry point; its horizontal run equals
    the edge's winding delta (e minus d), so it stays near-vertical in x.
  * Each absolute anchor is a star on its patch at `entry - anchor_to_entry_delta`,
    coloured by the seed winding its annotation implies, so disagreeing votes show
    up as different colours scattered across the tree.

Outputs: a static PNG of the main graph, a static PNG of the per-loop view, and --
unless --no-html -- a single interactive HTML5 page holding both as tabs. In the
"Winding graph" tab the main plot region scrolls while the patch names (left),
absolute-winding axis (bottom), title (top) and BFS-depth scale (right) stay
pinned; the "Loop cycles" tab shows each inconsistent loop as its own patch/pcl
stack. Both are SVG whose elements carry data-* attributes with hover tooltips, and
they live in isolated <iframe> panes inside the page so their styles never collide.

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
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import blended_transform_factory


def load_graph(votes):
    """Rebuild the patch tree (the anchor subtree) from a find_inconsistent_windings
    votes dict.

    Returns (nodes, edges, anchors, meta) where:
      nodes:   patch_id -> {'depth', 'acc'} (acc = winding(seed) - winding(entry))
      edges:   child_patch_id -> hop record (the parent edge; child is `to_patch`)
      anchors: list of voter dicts (each carries abs_patch_id / annotation / vote)
      meta:    handy scalars pulled off the top level

    The graph is the union of every voter's path: a tree rooted at the seed whose
    leaves carry the absolute anchors that voted. (The detected loops are not drawn
    on this graph -- they get their own per-loop view; see plot_loop_cycles.)
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
.loopline { stroke: #c026d3; stroke-width: 1.3; fill: none; opacity: .7; stroke-dasharray: 5 3; }
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
.patch, .edge, .anchor, .dot, .loop { cursor: pointer; }
.patch.hl .patchline, .patch.hl .cap { stroke: #ff8c00; }
.edge.hl .edgeline { stroke: #ff8c00; stroke-width: 2.4; }
.loop.hl .loopline { stroke: #ff8c00; stroke-width: 2.6; opacity: 1; }
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
    return 'edge — rel pcl ' + d.pcl + ' "' + (d.pclName || '') + '"'
         + '\npoint ' + d.fromPoint + ' on ' + d.from
         + '\n  → point ' + d.toPoint + ' on ' + d.to
         + '\nΔwinding ' + d.edgeDelta + '   strip ' + d.stripDelta;
  if (d.kind === 'loop')
    return 'inconsistent loop — rel pcl ' + d.pcl + ' "' + (d.pclName || '') + '"'
         + '\npoint ' + d.fromPoint + ' on ' + d.from
         + '\n  ↔ point ' + d.toPoint + ' on ' + d.to
         + '\nwinding holonomy ' + d.holo + '   (edge Δwinding ' + d.edgeDelta + ')';
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
document.querySelectorAll('.patch, .edge, .anchor, .dot, .loop').forEach(el => {
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


def build_main_html(geom):
    """Build the interactive main-graph HTML document (frozen panes + per-element
    SVG) and return it as a string -- embedded as a tab in the combined page."""
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
        """data-x/y/z (full-res volume xyz) from a raw, un-downsampled [z, y, x] triple, if present."""
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
                 f'data-pcl="{a(h["rel_pcl_id"])}" data-pcl-name="{a(h.get("rel_pcl_name") or "")}" '
                 f'data-from-point="{a(h["from_point_id"])}" data-to-point="{a(h["to_point_id"])}" '
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
        # winding span of the known points. Points sit at band centres (winding +
        # 0.5), so the winding number a centre belongs to is floor(x). A patch whose
        # span covers more than one winding crosses theta=0, so a single winding
        # number is meaningless -- the hover reports the range instead.
        rlo, rhi = node_raw_extent[pid]
        wlo, whi = int(math.floor(rlo)), int(math.floor(rhi))
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
                 f'data-point="{a(h["from_point_id"])}"{xyz_attrs(h.get("from_zyx_raw"))}>')
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
                     f'{xyz_attrs(v.get("abs_zyx_raw"))}>')
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
    # Tick marks sit on the integer seams (theta=0, the transitions between
    # windings)...
    for n in range(seam_lo, seam_hi + 1):
        xx = X(n)
        Bp.append(f'<line class="atick" x1="{xx:.1f}" y1="0" x2="{xx:.1f}" y2="7"/>')
    # ...but the winding number labels the *range* between two seams, so it is
    # centred in the gap (at n + 0.5), not on the seam at the winding's start.
    for n in range(seam_lo, seam_hi):
        xx = X(n + 0.5)
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
    return html_doc


def _short_patch(pid, n=16):
    """Patch id trimmed to fit a loop-stack cell (keep the tail, where ids differ)."""
    s = str(pid)
    return s if len(s) <= n else '…' + s[-(n - 1):]


def _select_loop_cycles(loop_cycles, max_cycles):
    """Sort the inconsistent loops by |holonomy| (largest first) and optionally cap
    the count. Returns (cycles, total, truncated) -- total is the full count before
    any cap, for the caption / a truncation note."""
    cycles = sorted(loop_cycles, key=lambda c: -abs(c.get('loop_winding_delta', 0)))
    total = len(cycles)
    truncated = bool(max_cycles and total > max_cycles)
    if truncated:
        cycles = cycles[:max_cycles]
    return cycles, total, truncated


def plot_loop_cycles(cycles, total, output, ncols=6, dpi=150):
    """Render each inconsistent closed loop on its own, as a vertical patch stack
    (matplotlib PNG). `cycles` are the (already selected) loops to draw; `total` is
    the full count before any cap, used only for the caption.

    Each loop is a vertical stack of its patches (the departure patch at the top,
    the arrival patch at the bottom), the relative-pcl tree edges as downward arrows
    between consecutive patches (labelled with the pcl id and its Δwinding), and one
    magenta 'closing' arrow bowing -- to the right, clear of the patch names -- from
    the bottom patch back up to the top (the non-tree edge whose nonzero holonomy
    makes the loop fail to close; labelled with its pcl id and Δwinding too)."""
    n = len(cycles)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    max_patches = max(len(c['patches']) for c in cycles)

    LOOP_COLOR = '#c026d3'
    cell_w, cell_h = 2.8, 0.34 * max_patches + 1.1
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * cell_w, nrows * cell_h))

    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.axis('off')
        if k >= n:
            continue
        c = cycles[k]
        patches, steps = c['patches'], c['steps']
        m = len(patches)
        pdeltas = c.get('patch_strip_deltas') or [None] * m
        ys = [-i for i in range(m)]

        # patches: a dot per row, id labelled to the left; the within-patch winding
        # delta (when the strip across the patch crosses theta=0) in crimson to the right.
        for i, pid in enumerate(patches):
            ax.plot([0], [ys[i]], marker='o', ms=4, color='black', zorder=3)
            ax.text(-0.16, ys[i], _short_patch(pid), fontsize=6, va='center', ha='right',
                    zorder=4, fontweight='bold' if i in (0, m - 1) else 'normal')
            if pdeltas[i]:
                ax.text(0.12, ys[i], f"Δ{pdeltas[i]:+d}", fontsize=5, color='crimson',
                        va='center', ha='left', zorder=4, fontweight='bold')

        # tree-edge pcls: downward arrows, label centred on the arrow (white bbox).
        for i, st in enumerate(steps):
            ax.annotate('', xy=(0, ys[i + 1]), xytext=(0, ys[i]),
                        arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.0,
                                        shrinkA=5, shrinkB=5), zorder=2)
            ax.text(0.0, 0.5 * (ys[i] + ys[i + 1]),
                    f"pcl {st['rel_pcl_id']}  Δ{st['edge_winding_delta']:+d}",
                    fontsize=5, color='0.3', va='center', ha='center', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.85))

        # closing edge: one arrow bowing to the RIGHT (clear of the patch names on the
        # left) from the bottom patch back up to the top. arc3 'rad' is a fraction of
        # the chord, so scale it by the stack height to keep the bow a roughly constant
        # width regardless of how tall the loop is.
        cs = c['closing_step']
        rad = 0.7 / max(1, m - 1)
        ax.add_patch(FancyArrowPatch(
            (0, ys[-1]), (0, ys[0]), connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=12, color=LOOP_COLOR, lw=1.3,
            shrinkA=6, shrinkB=6, zorder=2))
        ax.text(0.9, 0.5 * (ys[0] + ys[-1]),
                f"pcl {cs['rel_pcl_id']}  Δ{cs['edge_winding_delta']:+d}",
                fontsize=5.5, color=LOOP_COLOR, va='center', ha='left',
                fontweight='bold', zorder=4)

        ax.set_title(f"holonomy {c['loop_winding_delta']:+d}", fontsize=7, color=LOOP_COLOR)
        ax.set_xlim(-1.5, 1.9)
        ax.set_ylim(ys[-1] - 0.6, 0.6)

    cap_note = '' if n == total else f' (showing {n} with the largest |holonomy|)'
    fig.suptitle(f'{total} inconsistent closed loop(s){cap_note}  --  patches stacked top-to-bottom, '
                 f'relative-pcl edges as arrows (with Δwinding), magenta = closing edge (holonomy ≠ 0); '
                 f'crimson Δ = within-patch winding delta (theta=0 crossing)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {output}  ({n} loop diagram(s) of {total})')


# CSS / JS for the per-loop HTML view (kept as plain strings; same pattern as the
# main-graph page above).
_LOOPS_CSS = r"""
* { box-sizing: border-box; }
html, body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #fff; color: #222; }
.header { padding: 8px 12px; border-bottom: 1px solid #ddd; }
.header .title { font-size: 13px; font-weight: 600; }
.legend { margin-top: 6px; font-size: 12px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
.legend .item { display: inline-flex; align-items: center; gap: 5px; }
.grid { display: grid; gap: 10px; padding: 12px; }
.cell { border: 1px solid #eee; border-radius: 6px; padding: 4px 6px 8px; }
.cell .cap { font-size: 12px; font-weight: 700; color: #c026d3; text-align: center; margin-bottom: 2px; }
svg.loop { display: block; margin: 0 auto; }
.plbl { font-size: 11px; fill: #333; dominant-baseline: middle; }
.plbl.end { font-weight: 700; fill: #111; }
.elbl { font-size: 9.5px; fill: #444; dominant-baseline: middle; }
.clbl { font-size: 10px; fill: #c026d3; font-weight: 700; dominant-baseline: middle; }
.pslbl { font-size: 9px; fill: crimson; font-weight: 700; dominant-baseline: middle; pointer-events: none; }
.patchv { stroke: #000; stroke-width: 1.8; }
.patchv.end { stroke-width: 2.6; }
.tedge { stroke: #999; stroke-width: 1; fill: none; }
.closing { stroke: #c026d3; stroke-width: 1.6; fill: none; }
.pdot { fill: #000; }
.phit { fill: transparent; }
.ledge, .lclose, .lpatch, .ldot { cursor: pointer; }
.ledge:hover .tedge { stroke: #ff8c00; stroke-width: 2; }
.lclose:hover .closing { stroke: #ff8c00; stroke-width: 2.6; }
.lpatch:hover .patchv { stroke: #ff8c00; }
.lpatch:hover .plbl { fill: #ff8c00; }
.ldot:hover .pdot { fill: #ff8c00; }
.tooltip { position: fixed; pointer-events: none; display: none; z-index: 10;
           background: rgba(20,20,20,.92); color: #fff; font-size: 11px;
           padding: 5px 7px; border-radius: 4px; max-width: 420px; white-space: pre-line; }
"""

_LOOPS_JS = r"""
const tip = document.querySelector('.tooltip');
function xyz(px, py, pz) {
  if (px === undefined) return '  (xyz not recorded)';
  return '  xyz ' + Math.round(+px) + ', ' + Math.round(+py) + ', ' + Math.round(+pz);
}
function info(el) {
  const d = el.dataset;
  if (d.kind === 'patch') return 'patch ' + d.patch
        + (d.stripDelta !== undefined ? '\nΔwinding across patch ' + d.stripDelta : '');
  if (d.kind === 'dot') return 'rel-pcl point — pcl ' + d.pcl + ' "' + (d.pclName || '') + '"  point ' + d.point
        + '\non patch ' + d.patch + xyz(d.x, d.y, d.z);
  if (d.kind === 'edge') return 'rel-pcl edge ' + d.pcl + ' "' + (d.pclName || '') + '"'
        + '\npoint ' + d.fromPoint + ' on ' + d.from + xyz(d.fromX, d.fromY, d.fromZ)
        + '\n  →  point ' + d.toPoint + ' on ' + d.to + xyz(d.toX, d.toY, d.toZ)
        + '\nΔwinding ' + d.delta;
  if (d.kind === 'closing') return 'closing edge — rel-pcl ' + d.pcl + ' "' + (d.pclName || '') + '"'
        + '\npoint ' + d.fromPoint + ' on ' + d.from + xyz(d.fromX, d.fromY, d.fromZ)
        + '\n  →  point ' + d.toPoint + ' on ' + d.to + xyz(d.toX, d.toY, d.toZ)
        + '\nΔwinding ' + d.delta + '\nwinding holonomy ' + d.holo + '  (loop does not close)';
  return '';
}
document.querySelectorAll('.lpatch, .ledge, .lclose, .ldot').forEach(el => {
  el.addEventListener('mouseenter', () => { tip.textContent = info(el); tip.style.display = 'block'; });
  el.addEventListener('mousemove', e => { tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def build_loops_html(cycles, total, ncols=6):
    """Build the per-loop view as an HTML document string -- embedded as a tab in
    the combined page. One small inline SVG per loop in a grid: the patches stacked
    vertically, each drawn as a short bold black vertical line with a black dot at
    either end for the two pcl-points attaching there (the upper from the edge above,
    the lower from the edge below; each hoverable for its volume xyz). Each relative-
    pcl tree edge is two fine gray fragments running between adjacent patches' dots,
    with its pcl id + Δwinding in the gap between them; one magenta closing arrow bows
    to the right from the bottom patch back up to the top. When the strip across a patch
    between its two cycle points crosses theta=0, its winding delta is shown in crimson
    just right of the patch line. Each patch / edge / point / closing edge is its own
    <g> carrying data-* attributes, with a hover tooltip."""
    esc = html_lib.escape

    def a(s):
        return esc(str(s), quote=True)

    def xyz_attrs(zyx, prefix=''):
        """data-[prefix]x/y/z (full-res volume xyz) from a raw, un-downsampled [z, y, x]
        triple, if present. prefix='from-'/'to-' tags the two ends of a pcl edge."""
        if not zyx or len(zyx) != 3:
            return ''
        z, y, x = zyx
        return f' data-{prefix}x="{x:.2f}" data-{prefix}y="{y:.2f}" data-{prefix}z="{z:.2f}"'

    ROW, TOP, BOT, CX, BOW = 34, 16, 16, 150, 58
    # patch half-line height; pcl-point dot radius; half-gap left for the name.
    PHH, PR, NAMEGAP = 5.0, 2.6, 7
    # the closing caption sits just right of the bow's apex (CX + 3/4·BOW), close to
    # the line rather than out past its full width.
    LBLX, RLBL = CX - 14, CX + BOW - 8

    cells = []
    for c in cycles:
        patches, steps, cs = c['patches'], c['steps'], c['closing_step']
        m = len(patches)
        # within-patch winding delta between each patch's two cycle points (older votes
        # files predate this; fall back to no labels).
        pdeltas = c.get('patch_strip_deltas') or [None] * m
        ys = [TOP + i * ROW for i in range(m)]
        H = TOP + (m - 1) * ROW + BOT
        W = RLBL + 96
        midc = 0.5 * (ys[0] + ys[-1])
        ctxt = f"pcl {cs['rel_pcl_id']}  Δ{cs['edge_winding_delta']:+d}"

        # the pcl-point attaching at each patch's top end (edge from above) and bottom
        # end (edge below). Interior patches take the adjacent tree steps; the two loop
        # endpoints take the closing edge for their outer end. Each is
        # (pcl_id, pcl_name, point_id, patch_id, zyx) -> a hoverable black dot.
        top_pt, bot_pt = [None] * m, [None] * m
        for j, st in enumerate(steps):
            bot_pt[j] = (st['rel_pcl_id'], st.get('rel_pcl_name'), st['from_point_id'],
                         patches[j], st.get('from_zyx_raw'))
            top_pt[j + 1] = (st['rel_pcl_id'], st.get('rel_pcl_name'), st['to_point_id'],
                             patches[j + 1], st.get('to_zyx_raw'))
        bot_pt[m - 1] = (cs['rel_pcl_id'], cs.get('rel_pcl_name'), cs['from_point_id'],
                         patches[m - 1], cs.get('from_zyx_raw'))
        top_pt[0] = (cs['rel_pcl_id'], cs.get('rel_pcl_name'), cs['to_point_id'],
                     patches[0], cs.get('to_zyx_raw'))

        S = [f'<svg class="loop" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'xmlns="http://www.w3.org/2000/svg">']
        # tree-edge pcls: two fine gray fragments (above and below the centred name)
        # running between the bottom dot of the upper patch and the top dot of the
        # lower one, so the name sits in a clear gap rather than over the line.
        for i, st in enumerate(steps):
            ya, yb = ys[i] + PHH, ys[i + 1] - PHH   # the adjacent patches' end-dots
            mid = 0.5 * (ys[i] + ys[i + 1])
            txt = f"pcl {st['rel_pcl_id']}  Δ{st['edge_winding_delta']:+d}"
            S.append(f'<g class="ledge" data-kind="edge" data-pcl="{a(st["rel_pcl_id"])}" '
                     f'data-pcl-name="{a(st.get("rel_pcl_name") or "")}" '
                     f'data-from="{a(st["from_patch"])}" data-to="{a(st["to_patch"])}" '
                     f'data-from-point="{a(st["from_point_id"])}" data-to-point="{a(st["to_point_id"])}" '
                     f'data-delta="{st["edge_winding_delta"]:+d}"'
                     f'{xyz_attrs(st.get("from_zyx_raw"), "from-")}'
                     f'{xyz_attrs(st.get("to_zyx_raw"), "to-")}>')
            S.append(f'<line class="tedge" x1="{CX}" y1="{ya:.1f}" x2="{CX}" y2="{mid - NAMEGAP:.1f}"/>')
            S.append(f'<line class="tedge" x1="{CX}" y1="{mid + NAMEGAP:.1f}" x2="{CX}" y2="{yb:.1f}"/>')
            S.append(f'<text class="elbl" x="{CX}" y="{mid:.1f}" text-anchor="middle">{esc(txt)}</text>')
            S.append('</g>')

        # closing edge: bow to the right, from the bottom patch's bottom dot up into
        # the top patch's top dot.
        yb0, yt0 = ys[-1] + PHH, ys[0] - PHH
        S.append(f'<g class="lclose" data-kind="closing" data-pcl="{a(cs["rel_pcl_id"])}" '
                 f'data-pcl-name="{a(cs.get("rel_pcl_name") or "")}" '
                 f'data-from="{a(cs["from_patch"])}" data-to="{a(cs["to_patch"])}" '
                 f'data-from-point="{a(cs["from_point_id"])}" data-to-point="{a(cs["to_point_id"])}" '
                 f'data-delta="{cs["edge_winding_delta"]:+d}" data-holo="{c["loop_winding_delta"]:+d}"'
                 f'{xyz_attrs(cs.get("from_zyx_raw"), "from-")}'
                 f'{xyz_attrs(cs.get("to_zyx_raw"), "to-")}>')
        S.append(f'<path class="closing" d="M {CX},{yb0:.1f} C {CX + BOW},{yb0:.1f} '
                 f'{CX + BOW},{yt0:.1f} {CX},{yt0:.1f}" marker-end="url(#ahm)"/>')
        S.append(f'<text class="clbl" x="{RLBL}" y="{midc:.1f}">{esc(ctxt)}</text>')
        S.append('</g>')

        # patches: a short bold black vertical line, a black pcl-point dot at each end
        # (top = edge from above, bottom = edge below; each hoverable for its xyz), and
        # the id label to the left (departure/arrival ends bold).
        for i, pid in enumerate(patches):
            is_end = i in (0, m - 1)
            pcls = 'patchv end' if is_end else 'patchv'
            lcls = 'plbl end' if is_end else 'plbl'
            pd = pdeltas[i]
            pd_attr = f' data-strip-delta="{pd:+d}"' if pd is not None else ''
            S.append(f'<g class="lpatch" data-kind="patch" data-patch="{a(pid)}"{pd_attr}>')
            S.append(f'<line class="{pcls}" x1="{CX}" y1="{ys[i] - PHH:.1f}" '
                     f'x2="{CX}" y2="{ys[i] + PHH:.1f}"/>')
            S.append(f'<text class="{lcls}" x="{LBLX}" y="{ys[i]}" text-anchor="end">{esc(_short_patch(pid))}</text>')
            # crimson within-patch Δwinding, just right of the line (only when nonzero:
            # the strip across the patch crosses theta=0).
            if pd:
                S.append(f'<text class="pslbl" x="{CX + PHH + 3}" y="{ys[i]}" '
                         f'text-anchor="start">&#916;{pd:+d}</text>')
            S.append('</g>')
            for yy, pt in ((ys[i] - PHH, top_pt[i]), (ys[i] + PHH, bot_pt[i])):
                if pt is None:
                    continue
                pcl_id, pcl_name, pt_id, patch_id, zyx = pt
                S.append(f'<g class="ldot" data-kind="dot" data-pcl="{a(pcl_id)}" '
                         f'data-pcl-name="{a(pcl_name or "")}" '
                         f'data-point="{a(pt_id)}" data-patch="{a(patch_id)}"{xyz_attrs(zyx)}>')
                S.append(f'<circle class="phit" cx="{CX}" cy="{yy:.1f}" r="4"/>')
                S.append(f'<circle class="pdot" cx="{CX}" cy="{yy:.1f}" r="{PR}"/>')
                S.append('</g>')
        S.append('</svg>')
        cells.append(f'<div class="cell"><div class="cap">holonomy {c["loop_winding_delta"]:+d}</div>'
                     f'{"".join(S)}</div>')

    note = '' if len(cycles) == total else f' (showing {len(cycles)} with the largest |holonomy|)'
    title = (f'{total} inconsistent closed loop(s){note} — each: patches (bold black lines) stacked '
             f'top-to-bottom, relative-pcl edges as fine gray links (with Δwinding), '
             f'magenta = closing edge (holonomy ≠ 0)')
    legend = (
        '<span class="item"><svg width="20" height="14">'
        '<line x1="10" y1="2" x2="10" y2="12" stroke="#000" stroke-width="2.2"/>'
        '<circle cx="10" cy="2" r="2" fill="#000"/><circle cx="10" cy="12" r="2" fill="#000"/>'
        '</svg>patch (with its two pcl-points)</span>'
        '<span class="item"><svg width="24" height="12"><line x1="4" y1="6" x2="20" y2="6" '
        'stroke="#999" stroke-width="1"/></svg>relative-pcl tree edge</span>'
        '<span class="item"><svg width="24" height="12"><line x1="1" y1="6" x2="17" y2="6" '
        'stroke="#c026d3" stroke-width="1.8" marker-end="url(#ahm)"/></svg>closing edge (holonomy ≠ 0)</span>'
        '<span class="item"><svg width="22" height="14"><line x1="6" y1="2" x2="6" y2="12" '
        'stroke="#000" stroke-width="2.2"/><text x="10" y="11" font-size="11" fill="crimson" '
        'font-weight="700">&#916;</text></svg>within-patch &#916;winding (theta=0 crossing)</span>'
    )
    # Arrowhead markers, defined once and referenced from every loop svg.
    defs = ('<svg width="0" height="0" style="position:absolute"><defs>'
            '<marker id="ahg" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto" '
            'markerUnits="userSpaceOnUse"><path d="M0,0 L6,3 L0,6 z" fill="#777"/></marker>'
            '<marker id="ahm" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto" '
            'markerUnits="userSpaceOnUse"><path d="M0,0 L6,3 L0,6 z" fill="#c026d3"/></marker>'
            '</defs></svg>')
    grid_style = f'grid-template-columns: repeat({max(1, ncols)}, max-content); justify-content: start;'
    html_doc = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<title>loop cycles</title>\n'
        f'<style>{_LOOPS_CSS}</style></head>\n<body>\n{defs}\n'
        f'<div class="header"><div class="title">{esc(title)}</div>'
        f'<div class="legend">{legend}</div></div>\n'
        f'<div class="grid" style="{grid_style}">{"".join(cells)}</div>\n'
        f'<div class="tooltip"></div>\n'
        f'<script>{_LOOPS_JS}</script>\n'
        '</body></html>\n'
    )
    return html_doc


def write_combined_html(main_doc, loops_doc, output_html, seed, loops_total):
    """Write one HTML file holding both views as tabs. Each view keeps its own
    full-page layout and scripts, isolated inside an <iframe srcdoc=...> so their
    CSS/JS never collide; the shell only ever shows one iframe at a time. When
    loops_doc is None (no inconsistent loops) the page has just the winding-graph
    tab."""
    esc = html_lib.escape
    tabs = [('graph', 'Winding graph', main_doc)]
    if loops_doc is not None:
        tabs.append(('loops', f'Loop cycles ({loops_total})', loops_doc))
    btns, panes = [], []
    for i, (key, label, doc) in enumerate(tabs):
        active = ' active' if i == 0 else ''
        btns.append(f'<button class="tabbtn{active}" data-target="pane-{key}">{esc(label)}</button>')
        # The whole sub-document is HTML-escaped into the srcdoc attribute; the
        # browser un-escapes it and parses it as that iframe's isolated document.
        panes.append(f'<iframe id="pane-{key}" class="pane{active}" '
                     f'srcdoc="{esc(doc, quote=True)}"></iframe>')
    css = (
        '* { box-sizing: border-box; }\n'
        'html, body { margin: 0; height: 100vh; }\n'
        'body { display: flex; flex-direction: column; font-family: -apple-system, '
        '"Segoe UI", Roboto, Helvetica, Arial, sans-serif; }\n'
        '.tabs { flex: none; display: flex; gap: 4px; align-items: flex-end; '
        'background: #f3f3f3; border-bottom: 1px solid #ccc; padding: 5px 8px 0; }\n'
        '.tabbtn { padding: 6px 14px; border: 1px solid #ccc; border-bottom: none; '
        'background: #e6e6e6; color: #333; cursor: pointer; font-size: 13px; '
        'border-radius: 6px 6px 0 0; }\n'
        '.tabbtn.active { background: #fff; font-weight: 600; color: #111; margin-bottom: -1px; }\n'
        '.panes { flex: 1 1 auto; min-height: 0; position: relative; }\n'
        '.pane { position: absolute; inset: 0; width: 100%; height: 100%; border: none; display: none; }\n'
        '.pane.active { display: block; }\n'
    )
    js = (
        "const btns = document.querySelectorAll('.tabbtn');\n"
        "const panes = document.querySelectorAll('.pane');\n"
        "btns.forEach(b => b.addEventListener('click', () => {\n"
        "  btns.forEach(x => x.classList.remove('active'));\n"
        "  panes.forEach(x => x.classList.remove('active'));\n"
        "  b.classList.add('active');\n"
        "  document.getElementById(b.dataset.target).classList.add('active');\n"
        "}));\n"
    )
    html_doc = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<title>winding graph + loops {esc(seed)}</title>\n'
        f'<style>{css}</style></head>\n<body>\n'
        f'<div class="tabs">{"".join(btns)}</div>\n'
        f'<div class="panes">{"".join(panes)}</div>\n'
        f'<script>{js}</script>\n'
        '</body></html>\n'
    )
    with open(output_html, 'w') as f:
        f.write(html_doc)


@click.command()
@click.option('--votes', 'votes_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='winding_votes json written by find_inconsistent_windings.py.')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output image path (default: winding_graph_<patch>.png next to the votes json).')
@click.option('--seg-min-width', default=0.25, type=float,
              help='How far (in windings, must be < 0.5) each end of a patch line pokes past its '
                   'extreme known point. The default 0.25 runs the ends to the 1/4 and 3/4 marks of '
                   'the first / last winding box; it also gives a single-winding patch a visible stub.')
@click.option('--slot-height', default=1.0, type=float,
              help='Vertical spacing between adjacent patch rows (each patch gets its own row).')
@click.option('--depth-gap', default=0.8, type=float,
              help='Extra vertical space inserted between one BFS depth''s block of rows and the next.')
@click.option('--annotate-votes/--no-annotate-votes', default=True,
              help='Label each absolute anchor with "annotation->vote" (the abs winding it carries '
                   'and the seed winding it then implies).')
@click.option('--loops-output', default=None, type=click.Path(dir_okay=False),
              help='Output path for the per-loop view PNG (default: <votes>_loops.png next to the '
                   'votes json). The interactive version is a tab in the combined HTML page (see '
                   '--html-output). Each inconsistent loop is drawn on its own as a patch/pcl stack.')
@click.option('--loops-ncols', default=6, type=int,
              help='Number of loop diagrams per row in the per-loop view.')
@click.option('--max-loop-cycles', default=48, type=int,
              help='Cap on how many inconsistent loops to draw in the per-loop view (those with the '
                   'largest |holonomy| first); 0 = all. The full set is always in the votes json.')
@click.option('--label', 'label_mode', type=click.Choice(['patch', 'id', 'none']), default='patch',
              help='Left-hand column content per row: the patch id (folder name), a compact integer '
                   'id (id->patch mapping printed to stdout), or nothing.')
@click.option('--label-fontsize', default=6.0, type=float, help='Font size of the left-column labels.')
@click.option('--dpi', default=150, type=int, help='Output resolution.')
@click.option('--html/--no-html', 'want_html', default=True,
              help='Also write the combined interactive HTML5 page (winding-graph + loop-cycles tabs).')
@click.option('--html-output', default=None, type=click.Path(dir_okay=False),
              help='Combined HTML output path -- one file with a winding-graph tab and a loop-cycles '
                   'tab (default: same base as the main PNG with a .html extension).')
@click.option('--html-px-per-winding', default=44.0, type=float,
              help='Horizontal scale of the HTML plot, in pixels per winding.')
@click.option('--html-row-px', default=26.0, type=float,
              help='Vertical scale of the HTML plot, in pixels per patch row (slot-height units).')
@click.option('--html-label-width', default=260, type=int,
              help='Width (px) of the pinned left-hand patch-name column in the HTML.')
def main(votes_path, output, seg_min_width, slot_height, depth_gap, annotate_votes, loops_output,
         loops_ncols, max_loop_cycles, label_mode, label_fontsize, dpi, want_html, html_output,
         html_px_per_winding, html_row_px, html_label_width):
    with open(votes_path) as f:
        votes = json.load(f)
    nodes, edges, voters, meta = load_graph(votes)
    loop_cycles = votes.get('loop_cycles', [])

    seed = meta['seed']
    w0 = meta['seed_model_winding']

    # --- per-patch x coordinates: entry, the departure of each outgoing edge,
    #     and any anchors sitting on the patch. The point's winding number is the
    #     integer w0 - acc; we plot it at that winding's band CENTRE (winding +
    #     0.5) so it sits inside the winding range rather than on the seam. Every
    #     departure / anchor / edge offset is an integer winding delta, so they
    #     inherit the +0.5 and land on band centres too. ---
    def entry_x(pid):
        return w0 - nodes[pid]['acc'] + 0.5

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

    # patch segment extent [lo, hi]. node_raw_extent keeps the band-centre span of
    # the known points (the extreme stars): theta=0 ticks are drawn over that, since
    # the gaps between known points are integer seam-crossing counts
    # (intra_patch_strip_delta / anchor_to_entry_delta), so integers strictly inside
    # the band-centre span are exactly the genuine crossings. node_extent then pokes
    # each end seg_min_width past the extreme star, so the line runs a little beyond
    # its end points (and a single-winding patch, whose raw span is zero, becomes a
    # visible stub centred in its band). seg_min_width < 0.5 keeps both ends inside
    # the bounding windings -- with the default 0.25 the ends land at the 1/4 and 3/4
    # marks of the first / last winding box. Ticks use the raw span, so the poke
    # never fakes a crossing.
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

    ax.set_xlabel('absolute model winding number  (number sits in its winding; ticks = theta=0 seams)')
    # x ticks: the integer seams (theta=0, transitions between windings) carry the
    # tick marks, but each winding number labels the *range* between two seams, so
    # it is centred in the gap (at n + 0.5) rather than on the seam at its start.
    seam_lo, seam_hi = int(np.floor(x_lo)), int(np.ceil(x_hi))
    gaps = list(range(seam_lo, seam_hi))
    ax.set_xticks([n + 0.5 for n in gaps])
    ax.set_xticklabels([str(n) for n in gaps])
    ax.set_xticks(list(range(seam_lo, seam_hi + 1)), minor=True)
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='x', which='minor', length=5)
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

    # --- per-loop view PNG: each inconsistent closed loop drawn on its own (the
    #     interactive version is folded into the combined HTML below, as a tab) ---
    loops_selected = None
    if loop_cycles:
        cycles, total, truncated = _select_loop_cycles(loop_cycles, max_loop_cycles)
        if truncated:
            print(f'per-loop view: drawing the {max_loop_cycles} of {total} inconsistent loops '
                  f'with the largest |holonomy|; all {total} are in the votes json')
        if loops_output is None:
            base = os.path.splitext(os.path.basename(votes_path))[0]
            loops_output = os.path.join(os.path.dirname(os.path.abspath(votes_path)),
                                        f'{base}_loops.png')
        plot_loop_cycles(cycles, total, loops_output, ncols=loops_ncols, dpi=dpi)
        loops_selected = (cycles, total)
    else:
        print('no "loop_cycles" in this votes file; the per-loop view will be absent '
              '(rerun find_inconsistent_windings.py if it predates this field)')

    # --- interactive HTML: one file with a winding-graph tab and a loop-cycles tab ---
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
        main_doc = build_main_html(geom)
        loops_doc = (build_loops_html(loops_selected[0], loops_selected[1], ncols=loops_ncols)
                     if loops_selected else None)
        loops_total = loops_selected[1] if loops_selected else 0
        write_combined_html(main_doc, loops_doc, html_output, seed, loops_total)
        print(f'wrote {html_output}'
              + ('  (winding-graph + loop-cycles tabs)' if loops_doc else '  (winding-graph only)'))

    if label_mode == 'id':
        print('\npatch id legend (id: depth  winding  patch):')
        for pid in ordered:
            mark = ' [seed]' if pid == seed else (' [anchor]' if pid in anchors_by_patch else '')
            print(f'  {node_id[pid]:3d}: d{nodes[pid]["depth"]:<2d} '
                  f'w={entry_x(pid) - 0.5:6.2f}  {pid}{mark}')


if __name__ == '__main__':
    main()
