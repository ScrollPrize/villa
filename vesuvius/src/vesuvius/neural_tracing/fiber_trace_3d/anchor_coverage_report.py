"""Analysis only: pool native anchor spacing measurements, never extract anchors."""
import math
from collections import defaultdict
import numpy as np

EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, math.inf]


def distribution(values):
    values = np.asarray(values, dtype=float)
    counts = np.histogram(values, bins=EDGES)[0].tolist()
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                p95=float(np.quantile(values, .95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None,
                histogram_edges=EDGES[:-1] + [None], histogram_counts=counts)


def summarize(cases):
    interior, endpoints, empty_runs = [], [], []
    angles = []
    missing_axes = 0
    directional_empty = {10: 0, 20: 0, 30: 0}
    length = no_anchor_length = empty_bin_length = 0.
    empty_bins = bins = accepted = zero = one = fallback = tests = 0
    for c in cases:
        cell = c['cell_side_base']
        interior.extend(v / cell for v in c['interior_gaps_base'])
        endpoints.extend(c[k] / cell for k in ('leading_gap_base', 'trailing_gap_base')
                         if c[k] is not None)
        empty_runs.extend(c['empty_arc_bin_runs'])
        scale = c['base_voxel_size_um'] / 1000
        length += c['reference_length_base'] * scale
        no_anchor_length += c['no_anchor_length_base'] * scale
        empty_bin_length += c['empty_arc_bin_length_base'] * scale
        empty_bins += c['empty_arc_bins']
        bins += c['arc_bins']
        accepted += len(c['accepted_anchors'])
        for anchor in c['accepted_anchors']:
            if anchor.get('direction_error_degrees') is None:
                missing_axes += 1
            else:
                angles.append(anchor['direction_error_degrees'])
        for threshold in directional_empty:
            occupied = {min(c['arc_bins']-1, int(a['reference_arc_base']/cell))
                        for a in c['accepted_anchors']
                        if a.get('direction_error_degrees') is not None and a['direction_error_degrees'] <= threshold}
            directional_empty[threshold] += c['arc_bins'] - len(occupied)
        zero += len(c['unique_arcs_base']) == 0
        one += len(c['unique_arcs_base']) == 1
        fallback += c['invalid_normal_tests']
        tests += c['ellipse_tests']
    return dict(pieces=len(cases), length_mm=length, accepted_anchors=accepted,
                no_anchor_pieces=zero, one_position_pieces=one,
                no_anchor_length_mm=no_anchor_length,
                interior_gap_cells=distribution(interior), endpoint_gap_cells=distribution(endpoints),
                empty_bin_runs_cells=distribution(empty_runs), arc_bins=bins, empty_arc_bins=empty_bins,
                empty_arc_bins_percent=100*empty_bins/bins if bins else None,
                empty_arc_bin_length_percent=100*empty_bin_length/length if length else None,
                invalid_normal_tests=fallback, ellipse_tests=tests,
                direction_aware_empty_arc_bins_percent={str(k): 100*v/bins if bins else None
                                                        for k, v in directional_empty.items()},
                direction=dict(count=len(angles), missing=missing_axes,
                    median=float(np.median(angles)) if angles else None,
                    p95=float(np.quantile(angles,.95)) if angles else None,
                    within_degrees_percent={str(d): 100*sum(a<=d for a in angles)/len(angles)
                                            if angles else None for d in (10,20,30)}))


def coverage_report(cases):
    groups = defaultdict(list)
    for c in cases:
        # Collection name PHerc1229 is historical; embedded scan identity is1299.
        scroll = 'PHercParis4' if c['source_id'].startswith('PHercParis4') else c['source_id']
        if scroll == 'PHerc1229':
            scroll = 'PHerc1299'
        groups[scroll].append(c)
    return dict(protocol='anchor-nearest-projection-flat-caps-v1',
                cell_sides_base=sorted({c['cell_side_base'] for c in cases}),
                overall=summarize(cases), scrolls={k: summarize(v) for k, v in groups.items()})


def markdown_report(report):
    rows = ["# Anchor placement coverage", "",
            "Canonical anchors, nearest GT projection, configured GT acceptance ellipse and flat endpoint caps.",
            "Gaps are GT arclength / anchor-cell side; not literal crossed grid cells.",
            "Interior and endpoint gaps are separate; empty pieces remain in occupancy statistics.", "",
            f"Cell sides: {report['cell_sides_base']} base voxels. Runtime: {report['wall_seconds']:.2f}s.", "",
            "| Scroll | Pieces | Accepted anchors | Empty pieces | Median gap (cells) | P95 | Max | Empty arc bins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(v):
        return "—" if v is None else f"{v:.3f}"
    for name, s in [("All", report["overall"]), *report["scrolls"].items()]:
        g = s["interior_gap_cells"]
        rows.append(f"| {name} | {s['pieces']} | {s['accepted_anchors']} | {s['no_anchor_pieces']} | "
                    f"{number(g['median'])} | {number(g['p95'])} | {number(g['maximum'])} | "
                    f"{s['empty_arc_bins_percent']:.2f}% |")
    rows += ["", "## Pooled interior-gap histogram", "",
             "| Gap (cell-side units) | Count | Percent |", "|---|---:|---:|"]
    h = report["overall"]["interior_gap_cells"]
    for lo, hi, count in zip(EDGES, EDGES[1:], h["histogram_counts"]):
        rows.append(f"| [{lo:g}, {'∞' if math.isinf(hi) else format(hi, 'g')}) | {count} | "
                    f"{100*count/max(1,h['count']):.3f}% |")
    rows += ["", "All quantiles pool individual gaps. Arc-bin occupancy is phase-dependent; final partial bins count.",
             "This targeted failure set is not an unbiased sample of the full scrolls.",
             "See metrics.json for endpoint-gap and empty-bin-run distributions, and piece_ranking.json",
             "for worst-gap-first inspection links. Native anchors.json files contain positions and original GT arcs.", ""]
    return "\n".join(rows)
