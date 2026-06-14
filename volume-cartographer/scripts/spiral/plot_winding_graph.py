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

Run, e.g.:

    python plot_winding_graph.py --votes winding_votes.json --output winding_graph.png
"""

import os
import json
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


@click.command()
@click.option('--votes', 'votes_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='winding_votes json written by find_inconsistent_windings.py.')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output image path (default: winding_graph_<patch>.png next to the votes json).')
@click.option('--seg-min-width', default=0.18, type=float,
              help='Minimum half-width (in windings) of a patch line, so a patch with a single known '
                   'point is still visible.')
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
def main(votes_path, output, seg_min_width, slot_height, depth_gap, annotate_votes, label_mode, label_fontsize, dpi):
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
    # the genuine crossings. node_extent pads to a minimum width purely so a patch
    # with a single known point still draws as a visible (tick-free) line; ticking
    # the padded span would fake a crossing whenever frac(seed winding) lands
    # within seg_min_width of an integer.
    node_extent = {}
    node_raw_extent = {}
    for pid in nodes:
        lo, hi = min(point_xs[pid]), max(point_xs[pid])
        node_raw_extent[pid] = (lo, hi)
        if hi - lo < 2 * seg_min_width:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - seg_min_width, mid + seg_min_width
        node_extent[pid] = (lo, hi)

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

    # seed: just the word next to its (thicker) patch line -- no marker.
    ax.annotate('seed', (w0, 0), textcoords='offset points', xytext=(10, 0),
                ha='left', va='center', fontsize=9, fontweight='bold')

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
    vote_counts = defaultdict(int)
    for v in voters:
        vote_counts[v['expected_seed_winding_rounded']] += 1
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

    if label_mode == 'id':
        print('\npatch id legend (id: depth  winding  patch):')
        for pid in ordered:
            mark = ' [seed]' if pid == seed else (' [anchor]' if pid in anchors_by_patch else '')
            print(f'  {node_id[pid]:3d}: d{nodes[pid]["depth"]:<2d} '
                  f'w={entry_x(pid):6.2f}  {pid}{mark}')


if __name__ == '__main__':
    main()
