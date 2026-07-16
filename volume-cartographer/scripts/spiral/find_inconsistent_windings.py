"""Standalone debug tool: derive the expected spiral-space winding number of a
seed point from the winding annotations of the point-collections (pcls) that
reach its patch -- both directly-attached absolute-winding pcls and, by a
backwards BFS over patches, absolute-winding pcls on *other* patches that are
linked to the seed patch through a chain of relative-winding pcls.

It reuses fit_spiral's patch/point-collection loading phase and its spiral
transform, but does no fitting/training.

Two kinds of winding information feed the votes:

  * **Absolute-winding pcls** (metadata.winding_is_absolute): each attached point
    carries an *absolute* spiral winding number. A point on a patch is a winding
    "anchor" on that patch.
  * **Relative-winding pcls** (everything else): the difference of two of the
    pcl's points' wind_a annotations is the *relative* winding number between
    those two points (regardless of which patches they sit on). Treated as graph
    edges, they let an absolute anchor on one patch propagate to another.

The propagation, starting from the seed patch S (seed = a valid vertex near the
centre of S's grid):

  0. Direct votes: every absolute anchor attached to S votes for the seed's
     winding, propagating its absolute number from the anchor to the seed along
     a within-patch strip by counting discrete theta=0 branch crossings.
  1. Long-range votes: BFS the patch graph backwards from S. Relative pcls
     supply the edges -- for each relative pcl we walk its attached points in
     annotation order and connect each *consecutive* cross-patch pair (so the
     edge set stays a chain; more distant patches are reached transitively).
     Crossing an edge from patch P (departure point d) to patch R (arrival point
     e) costs the within-P strip delta from the entry point to d, plus the edge's
     wind_a difference (e minus d), adjusted when the two adjacent PCL points
     straddle theta=0. At every patch reached, its absolute anchors vote,
     propagated back to the seed through the accumulated chain.

Concretely, with strip_delta(P, a, b) := winding(b) - winding(a) measured by the
integer theta=0 branch transport of a within-P ij strip, each BFS state is
(patch P, entry point q on P, acc) with acc := winding(seed) - winding(q):

  * a vote from anchor a (absolute winding w) on P is
        winding(seed) = acc + w + strip_delta(P, a, q);
  * crossing a relative edge (P:d -> R:e, edge_delta = wind_a(e) - wind_a(d),
    with the adjacent-PCL theta=0 seam adjustment applied) sets
        acc' = acc - strip_delta(P, q, d) - edge_delta,  entry q' = e.

BFS marks each patch visited the first (fewest-hops) time it is reached, so it
explores a spanning tree of the reachable patches; --max-hops bounds the edge
depth (0 = direct anchors only, the original behaviour).

Relative edges that are *not* used as tree edges (they reconnect two patches both
already reached) are not discarded: each such edge closes a cycle, and we measure
its winding holonomy -- the difference between the winding it implies for the
arrival point and the winding the BFS tree already assigns there. A nonzero
holonomy means the relative-winding annotations do not close around that loop,
i.e. a winding inconsistency in the graph itself (independent of, and detectable
without, the absolute anchors). These are reported under "loops" in the output
(disable with --no-detect-loops). For each inconsistent loop we additionally
reconstruct the full cycle it closes -- the BFS-tree path between its two patches
plus the closing edge -- and record it under "loop_cycles" as an ordered patch/pcl
stack, which plot_winding_graph renders one loop at a time. Each cycle additionally
carries, per patch, the within-patch winding delta between its two cycle points (the
strip across a patch may itself cross theta=0), measured the same way as the tree
edges' intra-patch deltas.

Each within-patch winding delta is measured along a strip that follows a
fringe-avoiding shortest path through the patch's *valid quads only* (a weighted
Dijkstra on the quad grid, exactly like connect_overlapping_patches.py but within
one patch), then transforms that path to spiral space and theta=0-unwraps its
shifted radii. Sampling only valid quads keeps theta continuous so seam-crossing
counts reflect real branch transport (and whole windings on multi-winding patches)
instead of being thrown off by the wild theta swings of invalid regions. The
endpoints (annotated point / seed) are spliced onto the path's ends so the delta
is between the true points. The residual unwrapped shifted-radius gap is retained
as a diagnostic, but propagation uses the integer seam-crossing delta.

The transform's flow field is shaped by the checkpoint's own z-range; the
--z-range arg only filters patches/pcls (and must lie within the checkpoint's
range, like fit_spiral's resume path). Both are expressed in full-resolution
scan z, matching fit_spiral's z_begin/z_end constants.

Run from the spiral/ directory, e.g.

    python find_inconsistent_windings.py \
        --checkpoint out/.../checkpoint_fitted.ckpt \
        --patches-dir /path/to/dataset/patches \
        --patch-id <patch-folder-name> \
        --pcl /path/to/abs_windings.json \
        --pcl /path/to/rel_windings.json \
        --z-range 10000,13000
"""

import os
import sys
import json
from collections import deque

import click
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra, connected_components
from scipy.optimize import milp, LinearConstraint, Bounds

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import fit_spiral as fs
from sample_spiral import (
    get_theta_and_radii,
    get_theta_crossing_step_adjustments,
    unwrap_shifted_radii,
)
# Reuse connect_overlapping_patches' valid-quad graph + path reconstruction so a
# within-patch strip follows a fringe-avoiding path through valid quads only
# (never slicing through invalid regions, where theta would jump and the unwrap
# would miscount whole windings).
import connect_overlapping_patches as cop


def resolve_umbilicus_path(patches_dir, explicit_path=None):
    """Resolve the umbilicus used to rebuild a checkpoint transform.

    Checkpoints do not carry the source umbilicus samples, and fit_spiral's module
    default may point at the machine where the checkpoint was trained. Prefer an
    explicit path, then the dataset root inferred from --patches-dir, then the
    fit_spiral default if it exists.
    """
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))

    patches_dir = os.path.abspath(os.path.expanduser(patches_dir))
    inferred = os.path.join(os.path.dirname(patches_dir), 'umbilicus.json')
    if os.path.exists(inferred):
        return inferred

    default_dataset_path = getattr(fs, 'dataset_path', None)
    if default_dataset_path:
        default_path = os.path.join(os.path.expanduser(default_dataset_path), 'umbilicus.json')
        if os.path.exists(default_path):
            return os.path.abspath(default_path)

    raise SystemExit(
        f'could not find umbilicus.json next to --patches-dir ({inferred}); '
        'pass --umbilicus /path/to/umbilicus.json'
    )


def _to_py(x):
    """Convert numpy / torch scalars+arrays to plain python for json."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def install_globals(checkpoint, patches_dir, pcl_paths, filter_z_begin, filter_z_end, umbilicus_path):
    """Point fit_spiral's module globals at the checkpoint's cfg and the
    user-supplied patches/pcls/filter-z-range so its loaders behave identically.
    Returns the checkpoint's model z-range, which shapes the transform's flow
    field independently of the filtering z-range."""
    cfg = dict(fs.default_config)
    cfg.update(checkpoint['cfg'])
    fs.cfg = cfg
    fs.verified_patches_path = patches_dir
    # Attachment is over the verified patch set only, so skip the (slow, unrelated)
    # unverified patches; they don't change the cross-patch / attached pcl set.
    fs.unverified_patches_path = None
    fs.fibers_path = None
    fs.pcl_json_paths = list(pcl_paths)
    fs.z_begin = filter_z_begin
    fs.z_end = filter_z_end
    fs.umbilicus_z_to_yx = (
        lambda path=umbilicus_path: fs.json_umbilicus_z_to_yx(path, coordinate_scale=1.0)
    )
    # We don't compute shell losses; stop prepare_patches from loading the shell.
    fs.shell_losses_enabled = lambda: False
    return int(checkpoint['z_begin']), int(checkpoint['z_end'])


def build_transform(checkpoint, model_z_begin, model_z_end):
    """Reconstruct the slice->spiral transform and dr_per_winding from the
    checkpoint, using fit_spiral's own model class. The flow field is shaped by
    the checkpoint's (model) z-range, not the filtering z-range."""
    device = torch.device('cuda')
    cfg = fs.cfg

    all_zs = np.arange(model_z_begin, model_z_end)
    umbilicus_fn = fs.umbilicus_z_to_yx()
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], umbilicus_fn(all_zs)], axis=-1).astype(np.float32)
    ).to(device)

    r = cfg['flow_bounds_radius']
    flow_min_corner = torch.tensor([model_z_begin - cfg['flow_bounds_z_margin'], -r, -r], dtype=torch.int64, device=device)
    flow_max_corner = torch.tensor([model_z_end + cfg['flow_bounds_z_margin'], r, r], dtype=torch.int64, device=device)

    model = fs.SpiralAndTransform(
        flow_integration_steps=cfg['num_flow_integration_steps'],
        flow_integration_solver=cfg['flow_integration_solver'],
        umbilicus_zyx=umbilicus_zyx,
        flow_min_corner_zyx=flow_min_corner,
        flow_max_corner_zyx=flow_max_corner,
        config=cfg,
        spiral_outward_sense=fs.spiral_outward_sense,
    )
    model.to(device)
    model.load_state_dict(checkpoint['spiral_and_transform'])
    model.eval()
    # The high-res flow field is stored "pre-scale": at forward time the hr params
    # are multiplied by flow_scales[1], which training ramps and which is NOT part
    # of the state_dict. A fully-fitted checkpoint ends the ramp at its 'final'
    # value, so pin the scale to that to reproduce the saved transform.
    model.flow_field.flow_scales[1] = cfg['flow_field_high_res_lr_scale_final']

    transform = model.get_slice_to_spiral_transform()
    dr_per_winding = model.get_dr_per_winding()
    return transform, dr_per_winding


def pick_centre_seed(patch):
    """Pick a valid grid point near the centre of the patch's vertex grid. We pick
    the valid quad whose min-corner is closest to the grid centre, so the seed is a
    real vertex that lies on a valid quad (usable by ij_to_zyx / strip sampling)."""
    valid_quad = patch.valid_quad_mask.cpu().numpy()
    qi, qj = np.nonzero(valid_quad)
    if len(qi) == 0:
        raise SystemExit('patch has no valid quads')
    H, W = patch.zyxs.shape[:2]
    centre = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    d2 = (qi - centre[0]) ** 2 + (qj - centre[1]) ** 2
    k = int(np.argmin(d2))
    return np.array([float(qi[k]), float(qj[k])], dtype=np.float32)


def winding_at_points(transform, dr, zyx):
    """Per-point shifted-radius / dr (the model's raw winding number) for an
    (N, 3) scroll-space tensor; no unwrap (each point treated independently)."""
    spiral = transform(zyx)
    _, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    return shifted / dr


def _nearest_valid_quad_node(graph, ij):
    """Local node id of the valid quad containing `ij`, else the nearest valid quad."""
    h, w = graph.valid_quad.shape
    qi = int(np.clip(np.floor(ij[0]), 0, h - 1))
    qj = int(np.clip(np.floor(ij[1]), 0, w - 1))
    if graph.valid_quad[qi, qj]:
        return int(graph.ids[qi, qj])
    d2 = (graph.node_qi - qi) ** 2 + (graph.node_qj - qj) ** 2
    return int(np.argmin(d2))


def _valid_quad_path_centres(graph, start_node, end_node):
    """Quad-centre (i, j) list of the fringe-avoiding shortest path between two
    valid quads, or None if they are not connected through valid quads. The
    per-start Dijkstra predecessor array is memoised on the graph, so expanding a
    patch (many edges share one entry quad) costs a single Dijkstra."""
    if start_node == end_node:
        return [(graph.node_qi[start_node] + 0.5, graph.node_qj[start_node] + 0.5)]
    pred = graph._dij_cache.get(start_node)
    if pred is None:
        _dist, pred = csgraph_dijkstra(graph.csr, directed=True, indices=start_node,
                                       return_predecessors=True)
        graph._dij_cache[start_node] = pred
    path = cop._reconstruct(pred, start_node, end_node)
    if path is None:
        return None
    return [(graph.node_qi[k] + 0.5, graph.node_qj[k] + 0.5) for k in path]


def _polyline_ijs(points, step_size):
    """Densely sample a polyline through `points` (each (i, j)) at <= step_size
    spacing, dropping the duplicated junction vertex between consecutive segments."""
    segs = []
    for a, b in zip(points, points[1:]):
        seg_len = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg_len / step_size))) + 1
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        seg = a[None] * (1.0 - t) + b[None] * t
        if segs:
            seg = seg[1:]
        segs.append(seg)
    return np.concatenate(segs, axis=0).astype(np.float32)


def strip_winding_delta(transform, dr, graph, from_ij, to_ij, step_size):
    """Winding-number delta winding(to) - winding(from) between two ij points on
    one patch, measured as the integer theta=0 branch transport along a
    fringe-avoiding path through valid quads only.

    Routes from `from_ij` to `to_ij` via a weighted Dijkstra on the patch's
    valid-quad grid (connect_overlapping_patches' medial-axis weighting, so the
    path hugs the middle of valid strips), splices the true endpoints onto the
    path's ends, samples the polyline at <= `step_size` (grid/vertex units),
    transforms to spiral space, counts theta=0 seam crossings and returns the
    discrete winding delta `to` minus `from`, with diagnostics. Returns None if
    the two points are not connected through valid quads, or fewer than 2 samples
    survive.

    Confining the strip to valid quads keeps theta continuous, so the unwrap
    (sample_spiral.unwrap_shifted_radii) stitches only genuine theta=0 seam
    crossings -- a straight line through invalid regions would swing theta wildly
    and make the unwrap add or drop whole windings.
    """
    device = dr.device
    from_ij = np.asarray(from_ij, dtype=np.float32)
    to_ij = np.asarray(to_ij, dtype=np.float32)

    centres = _valid_quad_path_centres(
        graph,
        _nearest_valid_quad_node(graph, from_ij),
        _nearest_valid_quad_node(graph, to_ij),
    )
    if centres is None:
        return None

    # from_ij .. valid-quad path .. to_ij (true endpoints spliced on).
    polyline = [from_ij] + [np.array(c, dtype=np.float32) for c in centres] + [to_ij]
    ijs = torch.from_numpy(_polyline_ijs(polyline, step_size)).to(device)

    zyx_all, valid = graph.patch.ij_to_zyx(ijs)
    num_invalid = int((~valid).sum().item())
    zyx = zyx_all[valid]
    if zyx.shape[0] < 2:
        return None

    spiral = transform(zyx)
    theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    step_adjustment_windings = get_theta_crossing_step_adjustments(theta, dr) / dr.detach()
    cumulative_adjustment_windings = step_adjustment_windings.sum()
    delta_windings = int(round(float((-cumulative_adjustment_windings).item())))

    shifted_uw, _ = unwrap_shifted_radii(theta[None], shifted[None], dr)
    shifted_uw = shifted_uw[0]
    residual_windings = float(((shifted_uw[-1] - shifted_uw[0]) / dr).item())
    return {
        'delta_windings': delta_windings,
        'residual_unwrapped_delta_windings': residual_windings,
        'unwrap_adjustment_windings': float(cumulative_adjustment_windings.item()),
        'from_raw_winding': float((shifted[0] / dr).item()),
        'to_raw_winding': float((shifted[-1] / dr).item()),
        'strip_num_path_quads': len(centres),
        'strip_num_points': int(zyx.shape[0]),
        'strip_num_invalid_dropped': num_invalid,
    }


def classify_pcl(pcl):
    """Classify a pcl by how it carries winding information:

      * 'absolute' -- metadata.winding_is_absolute; each point is an absolute winding
        anchor.
      * 'relative' -- not absolute, but the pcl carried wind_a annotations in the
        source json (deliberate same-wrap relative-winding annotations, increasing).
      * 'neither'  -- not absolute and no wind_a in the source (fibers and
        new_same_wind pcls): they assert all their points sit on the *same* winding,
        so they contribute delta-0 edges but no winding assignment.

    The 'relative' vs 'neither' split relies on `has_winding_annotations`, stamped by
    normalise_pcl_winding_annotations before it 0-fills the all-unannotated case (post
    load both kinds have finite annotations, so finiteness can't tell them apart)."""
    if pcl.get('metadata', {}).get('winding_is_absolute', False):
        return 'absolute'
    if pcl.get('has_winding_annotations', False):
        return 'relative'
    return 'neither'


def build_abs_anchors_by_patch(cross_patch_pcls, patches):
    """patch_id -> list of absolute-winding anchors attached to that patch.

    Each anchor is a dict carrying the source absolute-winding pcl, the point and
    its absolute winding annotation, and the point's on-patch ij. Anchors with a
    non-finite annotation are skipped (warned about elsewhere)."""
    anchors = {}
    for pid, pcl in cross_patch_pcls.items():
        if not pcl.get('metadata', {}).get('winding_is_absolute', False):
            continue
        for patch_id, points in pcl.get('points_by_patch', {}).items():
            if patch_id not in patches:
                continue
            for p in points:
                w = float(p['winding_annotation'])
                if not np.isfinite(w):
                    continue
                anchors.setdefault(patch_id, []).append({
                    'pcl_id': int(pid),
                    'pcl_name': pcl.get('name'),
                    'source_file': pcl.get('source_file'),
                    'point_id': int(p['id']),
                    # Absolute-winding annotations are integer winding numbers.
                    'winding': int(round(w)),
                    'ij': np.array(p['on_patch']['ij'], dtype=np.float32),
                    'distance': float(p['on_patch'].get('distance', float('nan'))),
                    # Source [z, y, x]; point['p'] is [x, y, z] from the pcl json.
                    'zyx': np.asarray(p['p'], dtype=np.float32)[::-1],
                })
    return anchors


def pcl_edge_unwrap_adjustment_windings(transform, dr, pa, pb):
    """Unwrap the adjacent PCL segment pa -> pb and return only its theta=0
    cumulative branch-cut adjustment in winding units."""
    device = dr.device
    zyx = torch.as_tensor(
        np.stack([pa['zyx'], pb['zyx']], axis=0).astype(np.float32),
        device=device,
    )
    with torch.no_grad():
        spiral = transform(zyx)
        theta, _, _ = get_theta_and_radii(spiral[..., 1:], dr)
        zero_shifted = torch.zeros_like(theta)
        _, adjustments = unwrap_shifted_radii(theta[None], zero_shifted[None], dr)
        adjustments = adjustments[0]
    return int(round(float((adjustments[-1] / dr).item())))


def build_rel_adjacency(cross_patch_pcls, patches, transform, dr):
    """patch_id -> list of relative-winding edges leaving that patch.

    For each relative-winding pcl we order its attached points by annotation
    (int-json-key) order and connect each *consecutive* pair that lands on two
    different (loaded) patches. Each such pair yields a pair of directed edges,
    one per direction. An edge records the departure point (`from_ij`) on this
    patch, the arrival point (`to_ij`) on the neighbour, and the edge's winding
    delta = wind_a(arrival) - wind_a(departure), corrected for any theta=0 seam
    crossing between the adjacent PCL points. Only consecutive pairs are used, so
    the edge set stays a chain through the pcl's points; more distant patches are
    reached transitively by BFS."""
    adjacency = {}
    for pid, pcl in cross_patch_pcls.items():
        kind = classify_pcl(pcl)
        if kind == 'absolute':
            continue
        # Attached points (on a loaded patch, finite annotation) in annotation order.
        seq = []
        for _, p in sorted(pcl['points'].items(), key=lambda kv: int(kv[0])):
            on_patch = p.get('on_patch')
            if on_patch is None or on_patch['id'] not in patches:
                continue
            if not np.isfinite(float(p['winding_annotation'])):
                continue
            seq.append(p)
        for pa, pb in zip(seq, seq[1:]):
            ida, idb = pa['on_patch']['id'], pb['on_patch']['id']
            if ida == idb:
                continue
            wa, wb = float(pa['winding_annotation']), float(pb['winding_annotation'])
            # Relative-winding annotations are integers; the edge delta is the integer
            # number of windings between the two attached points, expressed in the
            # local theta branch frame. The unwrap helper returns c, the cumulative
            # shifted-radius adjustment; the diagnostic's integer branch transport
            # convention is -c (same sign as strip_winding_delta).
            raw_dwind = int(round(wb - wa))
            unwrap_adjustment = pcl_edge_unwrap_adjustment_windings(transform, dr, pa, pb)
            branch_delta = -unwrap_adjustment
            dwind = raw_dwind + branch_delta
            ija = np.array(pa['on_patch']['ij'], dtype=np.float32)
            ijb = np.array(pb['on_patch']['ij'], dtype=np.float32)
            # Source [z, y, x] of each attached point, so the departure/arrival
            # dots can be located in the volume later.
            za = np.asarray(pa['p'], dtype=np.float32)[::-1]
            zb = np.asarray(pb['p'], dtype=np.float32)[::-1]
            common = {'pcl_id': int(pid), 'pcl_name': pcl.get('name'), 'source_file': pcl.get('source_file'),
                      'kind': kind}
            adjacency.setdefault(ida, []).append({
                **common, 'neighbor': idb, 'from_ij': ija, 'to_ij': ijb,
                'from_point_id': int(pa['id']), 'to_point_id': int(pb['id']),
                'from_zyx': za, 'to_zyx': zb,
                'winding_delta': dwind,
                'raw_winding_delta': raw_dwind,
                'pcl_unwrap_adjustment': unwrap_adjustment,
                'pcl_branch_delta': branch_delta,
            })
            adjacency.setdefault(idb, []).append({
                **common, 'neighbor': ida, 'from_ij': ijb, 'to_ij': ija,
                'from_point_id': int(pb['id']), 'to_point_id': int(pa['id']),
                'from_zyx': zb, 'to_zyx': za,
                'winding_delta': -dwind,
                'raw_winding_delta': -raw_dwind,
                'pcl_unwrap_adjustment': -unwrap_adjustment,
                'pcl_branch_delta': -branch_delta,
            })
    return adjacency


def build_loop_cycles(loops, tree_edge_by_child, strip_delta_fn):
    """For each inconsistent loop (a non-tree relative edge whose winding holonomy
    is nonzero), reconstruct the full fundamental cycle it closes: the BFS-tree
    path between the edge's two patches, plus the closing edge itself.

    Returns a list of cycle dicts. Each lists `patches` as an ordered stack -- the
    departure patch P at the top, walking up the tree to the lowest common ancestor
    and back down to the arrival patch R at the bottom -- with one `steps` entry per
    consecutive pair (the relative-pcl tree edge between them, with its winding
    delta oriented in walk order) and a single `closing_step` for the non-tree edge
    that loops R back up to P. It also carries `patch_strip_deltas`, one per patch
    (aligned with `patches`): the within-patch winding delta between that patch's two
    cycle points -- where the edge above attaches (top) and where the edge below
    attaches (bottom) -- measured downward via `strip_delta_fn(patch, top_ij,
    bot_ij)` (winding(bottom) - winding(top); None if unmeasurable). The strip
    across a patch may itself cross theta=0, exactly like the tree edges'
    intra_patch_strip_delta. plot_winding_graph renders each cycle as a vertical
    patch/pcl stack; precomputing it here keeps the tree-walking reconstruction next
    to the BFS that produced the tree."""
    def path_to_root(p):
        """Patch ids from p up to the seed (root) via tree-edge parent pointers."""
        path = [p]
        while True:
            hop = tree_edge_by_child.get(path[-1])
            if hop is None:  # reached the seed (no tree edge into it)
                break
            path.append(hop['from_patch'])
        return path

    def tree_step(a, b):
        """Step record for walking the tree edge between adjacent patches a -> b
        (a, b are parent/child in some order; orient the pcl delta along a -> b)."""
        hop = tree_edge_by_child.get(b)
        if hop is not None and hop['from_patch'] == a:
            return {'from_patch': a, 'to_patch': b,
                    'rel_pcl_id': hop['rel_pcl_id'], 'rel_pcl_name': hop['rel_pcl_name'],
                    'from_point_id': hop['from_point_id'], 'to_point_id': hop['to_point_id'],
                    'from_ij': hop['from_ij'], 'to_ij': hop['to_ij'],
                    'from_zyx_raw': hop['from_zyx_raw'],
                    'to_zyx_raw': hop['to_zyx_raw'],
                    'edge_winding_delta': hop['edge_winding_delta'],
                    'raw_edge_winding_delta': hop['raw_edge_winding_delta'],
                    'pcl_unwrap_adjustment': hop['pcl_unwrap_adjustment'],
                    'pcl_branch_delta': hop['pcl_branch_delta'],
                    'kind': 'tree'}
        hop = tree_edge_by_child.get(a)
        assert hop is not None and hop['from_patch'] == b, (a, b)
        # Stored as b -> a; we walk a -> b, so swap the points (and their coords) and
        # negate the delta.
        return {'from_patch': a, 'to_patch': b,
                'rel_pcl_id': hop['rel_pcl_id'], 'rel_pcl_name': hop['rel_pcl_name'],
                'from_point_id': hop['to_point_id'], 'to_point_id': hop['from_point_id'],
                'from_ij': hop['to_ij'], 'to_ij': hop['from_ij'],
                'from_zyx_raw': hop['to_zyx_raw'],
                'to_zyx_raw': hop['from_zyx_raw'],
                'edge_winding_delta': -hop['edge_winding_delta'],
                'raw_edge_winding_delta': -hop['raw_edge_winding_delta'],
                'pcl_unwrap_adjustment': -hop['pcl_unwrap_adjustment'],
                'pcl_branch_delta': -hop['pcl_branch_delta'],
                'kind': 'tree'}

    cycles = []
    for L in loops:
        if not L.get('is_inconsistent'):
            continue
        P, R = L['from_patch'], L['to_patch']
        pp, pr = path_to_root(P), path_to_root(R)
        pr_index = {pid: i for i, pid in enumerate(pr)}
        lca_i = next(i for i, pid in enumerate(pp) if pid in pr_index)
        lca = pp[lca_i]
        # P .. LCA (up P's branch) then LCA-exclusive .. R (down R's branch, reversed).
        patches = pp[:lca_i + 1] + pr[:pr_index[lca]][::-1]
        steps = [tree_step(a, b) for a, b in zip(patches, patches[1:])]
        # the non-tree edge, drawn as one arrow from R (bottom) back up to P (top).
        closing_step = {
            'from_patch': R, 'to_patch': P,
            'rel_pcl_id': L['rel_pcl_id'], 'rel_pcl_name': L['rel_pcl_name'],
            'from_point_id': L['to_point_id'], 'to_point_id': L['from_point_id'],
            'from_ij': L['to_ij'], 'to_ij': L['from_ij'],
            'from_zyx_raw': L['to_zyx_raw'],
            'to_zyx_raw': L['from_zyx_raw'],
            'edge_winding_delta': -L['edge_winding_delta'],
            'raw_edge_winding_delta': -L['raw_edge_winding_delta'],
            'pcl_unwrap_adjustment': -L['pcl_unwrap_adjustment'],
            'pcl_branch_delta': -L['pcl_branch_delta'],
            'kind': 'closing',
        }
        # Per patch: the within-patch winding delta between its two cycle points -- the
        # pcl-point where the edge above attaches (top) and where the edge below
        # attaches (bottom). The loop endpoints take the closing edge for their outer
        # end (the same top/bottom assignment plot_winding_graph draws). Measured
        # downward (winding(bottom) - winding(top)) as integer theta=0 seam transport;
        # None where the strip can't be measured.
        m = len(patches)
        patch_strip_deltas = []
        for i in range(m):
            top_ij = closing_step['to_ij'] if i == 0 else steps[i - 1]['to_ij']
            bot_ij = closing_step['from_ij'] if i == m - 1 else steps[i]['from_ij']
            patch_strip_deltas.append(strip_delta_fn(patches[i], top_ij, bot_ij))
        cycles.append({
            'closing_rel_pcl_id': L['rel_pcl_id'],
            'closing_rel_pcl_name': L['rel_pcl_name'],
            'loop_winding_delta': L['loop_winding_delta'],
            'loop_winding_residual': L['loop_winding_residual'],
            'from_patch': P, 'to_patch': R, 'lca': lca,
            'patches': patches,
            'steps': steps,
            'patch_strip_deltas': patch_strip_deltas,
            'closing_step': closing_step,
        })
    return cycles


def solve_min_edge_fix(reached, rel_adjacency, strip_delta, allowed_edge_keys=None, time_limit=120.0):
    """Find the *fewest* relative edges whose winding-delta must change so the whole
    relative-edge graph closes (every loop has zero holonomy).

    Model each reached patch p as an integer node potential u_p = winding number at
    p's BFS entry point q_p (gauge-fixed u_seed = 0). A relative edge from departure
    point d on P to arrival point a on R with annotation delta `winding_delta`
    requires, for a consistent graph,

        u_R - u_P == D_e,   D_e = winding_delta + strip(P, q_P -> d) - strip(R, q_R -> a)

    where strip(.) is the integer theta=0 within-patch transport (same machinery as
    the BFS's intra_patch_strip_delta). We minimise the number of violated edges with
    one binary z_e per edge and a big-M residual bound (HiGHS via scipy.optimize.milp):

        minimise  sum_e z_e
        s.t.      -M z_e <= (u_R - u_P) - D_e <= M z_e,   u integer, z in {0, 1}.

    `strip_delta(patch_id, from_ij, to_ij)` must return the integer within-patch delta
    winding(to) - winding(from) (or None if unmeasurable). Returns a result dict; an
    edge with z_e == 1 is one to change, with its suggested corrected delta.

    `allowed_edge_keys`, when given, restricts the model to that set of edge keys
    (`(pcl_id, frozenset((from_point_id, to_point_id)))`) -- pass the edges lying on the
    inconsistent fundamental cycles so the solve only touches edges that can actually
    close a broken loop. Edges on only-consistent cycles cannot help and only add
    gauge freedom (a hub patch's potential can then float to its box bound, producing
    huge spurious residuals), so excluding them keeps the suggestions sane and tight.
    """
    # --- collect unique undirected rel edges among reached patches ---
    strip_memo = {}

    def sd(pid, a, b):
        key = (pid, tuple(np.round(np.asarray(a, dtype=float), 3)),
               tuple(np.round(np.asarray(b, dtype=float), 3)))
        if key not in strip_memo:
            strip_memo[key] = strip_delta(pid, a, b)
        return strip_memo[key]

    seen = set()
    edges = []
    num_unmeasurable = 0
    for P, edge_list in rel_adjacency.items():
        if P not in reached:
            continue
        for e in edge_list:
            R = e['neighbor']
            if R not in reached:
                continue
            key = (e['pcl_id'], frozenset((e['from_point_id'], e['to_point_id'])))
            if allowed_edge_keys is not None and key not in allowed_edge_keys:
                continue
            if key in seen:
                continue
            seen.add(key)  # also blocks the reverse-direction copy from R's list
            sP = sd(P, reached[P]['entry_ij'], e['from_ij'])
            sR = sd(R, reached[R]['entry_ij'], e['to_ij'])
            if sP is None or sR is None:
                num_unmeasurable += 1
                continue
            edges.append({
                'P': P, 'R': R,
                'D': int(e['winding_delta'] + sP - sR),
                'pcl_id': e['pcl_id'], 'pcl_name': e['pcl_name'], 'source_file': e['source_file'],
                'kind': e.get('kind'),
                'from_point_id': e['from_point_id'], 'to_point_id': e['to_point_id'],
                'from_zyx': e['from_zyx'], 'to_zyx': e['to_zyx'],
                'winding_delta': int(e['winding_delta']),
            })

    if not edges:
        return {
            'num_edges_considered': 0, 'num_edges_changed': 0, 'num_edges_unmeasurable': num_unmeasurable,
            'solver_status': 'no edges', 'objective': 0.0, 'edges': [],
        }

    # --- variable layout: [u_p for nodes] + [z_e for edges] ---
    nodes = sorted({e['P'] for e in edges} | {e['R'] for e in edges})
    node_idx = {p: i for i, p in enumerate(nodes)}
    n, m = len(nodes), len(edges)

    # BFS gives a feasible labelling u0_p = -acc_p (acc = winding(seed) - winding(entry)),
    # so the box only needs to span the BFS winding range plus a small correction margin.
    u0 = np.array([-reached[p]['acc'] for p in nodes], dtype=float)
    big = float(np.max(np.abs(u0 - u0.mean())) + m + 10)  # generous, but contains u0
    M = 2.0 * big + max(abs(e['D']) for e in edges) + 1.0

    # constraints: for edge i (u_R - u_P) - D_i <=  M z_i   and  -(...) <= -... + M z_i
    rows, cols, vals, ub = [], [], [], []
    for i, e in enumerate(edges):
        ip, ir, iz = node_idx[e['P']], node_idx[e['R']], n + i
        # row 2i:   u_R - u_P - M z_i <= D_i
        rows += [2 * i, 2 * i, 2 * i]; cols += [ir, ip, iz]; vals += [1.0, -1.0, -M]; ub.append(float(e['D']))
        # row 2i+1: -u_R + u_P - M z_i <= -D_i
        rows += [2 * i + 1, 2 * i + 1, 2 * i + 1]; cols += [ir, ip, iz]; vals += [-1.0, 1.0, -M]; ub.append(float(-e['D']))
    A = coo_matrix((vals, (rows, cols)), shape=(2 * m, n + m)).tocsr()
    constraints = LinearConstraint(A, -np.inf, np.array(ub))

    lb = np.concatenate([u0 - big, np.zeros(m)])
    hi = np.concatenate([u0 + big, np.ones(m)])
    if 'seed' not in node_idx:  # gauge-fix one node to break the per-component constant
        seed_node = nodes[int(np.argmin(np.abs(u0)))]
    else:
        seed_node = 'seed'
    si = node_idx[seed_node]
    lb[si] = hi[si] = u0[si]

    c = np.concatenate([np.zeros(n), np.ones(m)])
    res = milp(c, constraints=constraints, integrality=np.ones(n + m),
               bounds=Bounds(lb, hi), options={'time_limit': time_limit})

    out_edges = []
    if res.x is not None:
        u = np.round(res.x[:n]).astype(int)
        for i, e in enumerate(edges):
            residual = int(u[node_idx[e['R']]] - u[node_idx[e['P']]] - e['D'])
            if residual == 0:
                continue
            out = {
                'rel_pcl_id': e['pcl_id'], 'rel_pcl_name': e['pcl_name'], 'rel_source_file': e['source_file'],
                'kind': e['kind'],
                'from_patch': e['P'], 'to_patch': e['R'],
                'from_point_id': e['from_point_id'], 'to_point_id': e['to_point_id'],
                'from_zyx_raw': _to_py(e['from_zyx']), 'to_zyx_raw': _to_py(e['to_zyx']),
                'residual': residual,
            }
            # A 'neither' pcl (fiber / new_same_wind) carries no editable winding number -- it
            # only asserts its points share a winding. So the fix is not a delta edit but a
            # mis-attached point: delete one of the two points, or re-attach it to the right
            # winding's patch. (Deleting an intermediate point is not identical to applying
            # `residual` to one edge -- it makes the point's neighbours consecutive -- so this
            # flags the likely culprit, it is not a mechanical delta change.)
            if e['kind'] == 'neither':
                out['action'] = 'detach_or_reattach'
            else:
                out['action'] = 'change_winding_delta'
                out['current_winding_delta'] = e['winding_delta']
                out['suggested_winding_delta'] = e['winding_delta'] + residual
            out_edges.append(out)
    return {
        'num_edges_considered': m,
        'num_edges_changed': len(out_edges),
        'num_edges_unmeasurable': num_unmeasurable,
        'solver_status': res.message,
        'objective': float(res.fun) if res.fun is not None else None,
        'edges': out_edges,
    }


def find_disconnected_subgraphs(rel_adjacency, patches, seed_patch_id, top_patches):
    """Connected components of the relative-edge patch graph that do NOT contain the
    seed patch -- the patch clusters the seed-rooted BFS never reaches.

    The BFS in main() only ever explores the seed's connected component (the "main
    tree"); patches wired together by relative/neither pcls but not linked to the
    seed are otherwise invisible. This finds those other components and ranks them by
    patch count so the biggest disconnected clusters (and good seed candidates within
    them) can be surfaced for separate analysis.

    Nodes are the patches carrying at least one cross-patch relative/neither edge
    (`rel_adjacency.keys()`); isolated patches (no such edge) form no graph and are
    excluded. Edges are the (undirected) cross-patch pcl links. Component membership
    is computed with scipy's connected_components on a symmetric adjacency.

    Returns a dict with the total component count, the seed component's size, and a
    `components` list (excluding the seed's component) sorted by patch count desc
    (tie-break total area desc), each carrying a 1-based `rank`, `num_patches`,
    `num_edges` (unique undirected cross-patch edges with both ends in the component),
    `total_area`, the full `patch_ids`, and `largest_patches` -- the top `top_patches`
    members by `patch.area`."""
    nodes = sorted(rel_adjacency.keys())
    if not nodes:
        return {'num_components_total': 0, 'seed_component_num_patches': 0,
                'num_disconnected_components': 0, 'components': []}
    node_idx = {pid: i for i, pid in enumerate(nodes)}
    n = len(nodes)

    rows, cols = [], []
    for P, edge_list in rel_adjacency.items():
        for e in edge_list:
            R = e['neighbor']
            if R not in node_idx:
                continue
            rows.append(node_idx[P]); cols.append(node_idx[R])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)

    # The seed's component (the "main tree") is excluded. If the seed carries no
    # relative edge it is not a node here, so every component is disconnected from it.
    seed_label = labels[node_idx[seed_patch_id]] if seed_patch_id in node_idx else None
    seed_component_num_patches = (
        int(np.sum(labels == seed_label)) if seed_label is not None else 0
    )

    members_by_label = {}
    for pid in nodes:
        members_by_label.setdefault(labels[node_idx[pid]], []).append(pid)

    def patch_area(pid):
        return float(patches[pid].area)

    components = []
    for label, members in members_by_label.items():
        if label == seed_label:
            continue
        member_set = set(members)
        edge_keys = set()
        for P in members:
            for e in rel_adjacency.get(P, []):
                if e['neighbor'] in member_set:
                    edge_keys.add((e['pcl_id'], frozenset((e['from_point_id'], e['to_point_id']))))
        members_sorted = sorted(members, key=patch_area, reverse=True)
        components.append({
            'num_patches': len(members),
            'num_edges': len(edge_keys),
            'total_area': float(sum(patch_area(pid) for pid in members)),
            'patch_ids': members_sorted,
            'largest_patches': [
                {'patch_id': pid, 'area': patch_area(pid),
                 'grid': list(patches[pid].zyxs.shape[:2])}
                for pid in members_sorted[:top_patches]
            ],
        })

    components.sort(key=lambda c: (c['num_patches'], c['total_area']), reverse=True)
    for rank, c in enumerate(components, start=1):
        c['rank'] = rank
    return {
        'num_components_total': int(n_comp),
        'seed_component_num_patches': seed_component_num_patches,
        'num_disconnected_components': len(components),
        'components': components,
    }


def parse_z_range(z_range):
    lo, hi = (int(v) for v in z_range.split(','))
    if lo >= hi:
        raise click.BadParameter('--z-range must be "zlo,zhi" with zlo < zhi (full-resolution scan z)')
    return lo, hi


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to a checkpoint_*.ckpt saved by fit_spiral.')
@click.option('--patches-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory of patch tifxyz folders (the full set, for attachment).')
@click.option('--umbilicus', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Path to umbilicus.json. Defaults to umbilicus.json next to --patches-dir.')
@click.option('--patch-id', required=True, help='Folder name (within --patches-dir) of the seed patch to debug.')
@click.option('--pcl', 'pcl_paths', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False),
              help='Point-collection json file(s). Repeat --pcl for several. Both absolute- and '
                   'relative-winding pcls are used (absolute supply votes, relative supply the '
                   'long-range graph edges).')
@click.option('--z-range', required=True,
              help='Patch/pcl filtering z-range as "zlo,zhi". Must lie within the checkpoint z-range.')
@click.option('--step-size', default=1.0, type=float,
              help='Sampling spacing (in tifxyz grid cells) along the valid-quad path within each '
                   'patch (spacing between samples is <= step-size).')
@click.option('--medial-weight', default=4.0, type=float,
              help='How strongly within-patch strips are pulled toward the perpendicular middle of '
                   'valid regions (0 = shortest valid path; larger keeps the path away from '
                   'valid-region boundaries, where theta is noisy). Same meaning as in '
                   'connect_overlapping_patches.py.')
@click.option('--max-hops', default=None, type=int,
              help='Maximum number of relative-pcl edges to traverse from the seed patch (BFS depth). '
                   '0 = direct absolute anchors on the seed patch only (original behaviour); '
                   'omit for unlimited (BFS to every reachable patch).')
@click.option('--detect-loops/--no-detect-loops', default=True,
              help='Check every non-tree relative edge (one that reconnects two already-reached '
                   'patches) for winding holonomy: compare the winding it implies for the arrival '
                   'point against the winding the BFS tree already assigns there. A nonzero '
                   'difference is a loop that does not close -- a winding inconsistency in the '
                   'relative-edge graph itself. Recorded under "loops" in the output.')
@click.option('--min-fix/--no-min-fix', default=True,
              help='After detecting loop inconsistencies, solve a MILP (scipy/HiGHS) for the FEWEST '
                   'relative-edge winding-delta annotations that must change so the whole relative-edge '
                   'graph closes (every loop has zero holonomy). Report-only; recorded under '
                   '"min_edge_fix" in the output. Requires --detect-loops.')
@click.option('--min-fix-time-limit', default=120.0, type=float,
              help='Wall-clock time limit (seconds) for the --min-fix MILP solve. A non-optimal stop '
                   'is reported via min_edge_fix.solver_status.')
@click.option('--max-subgraphs', default=20, type=int,
              help='How many disconnected relative-edge subgraphs (connected components of the '
                   'patch graph that do NOT contain the seed patch) to report, largest by patch '
                   'count first. Reported component-count totals stay accurate regardless.')
@click.option('--top-patches-per-subgraph', default=10, type=int,
              help='Within each reported disconnected subgraph, how many of its largest patches '
                   '(by patch area) to list.')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output json path (default: winding_votes_<patch>.json next to the checkpoint).')
@click.option('--plot/--no-plot', default=False,
              help='After writing the json, also render the full winding graph with plot_winding_graph '
                   '(every reached patch positioned at its inferred winding, all connections, plus a '
                   'max-winding-reached readout). Writes the same PNG(s) + combined HTML the standalone '
                   'plot_winding_graph.py produces, next to the json.')
@click.option('--plot-output', default=None, type=click.Path(dir_okay=False),
              help='Output image path for --plot (default: winding_votes_<patch>_graph.png next to '
                   'the json). Ignored without --plot.')
def main(checkpoint, patches_dir, umbilicus, patch_id, pcl_paths, z_range, step_size, medial_weight, max_hops,
         detect_loops, min_fix, min_fix_time_limit, max_subgraphs, top_patches_per_subgraph, output,
         plot, plot_output):
    torch.set_grad_enabled(False)
    device = torch.device('cuda')

    if step_size <= 0:
        raise click.BadParameter('--step-size must be > 0 (patch grid/vertex units)')
    if medial_weight < 0:
        raise click.BadParameter('--medial-weight must be >= 0')
    if max_hops is not None and max_hops < 0:
        raise click.BadParameter('--max-hops must be >= 0 (or omitted for unlimited)')

    z_lo_raw, z_hi_raw = parse_z_range(z_range)
    filter_z_begin, filter_z_end = z_lo_raw, z_hi_raw

    umbilicus_path = resolve_umbilicus_path(patches_dir, umbilicus)
    print(f'using umbilicus {umbilicus_path}')

    print(f'loading checkpoint {checkpoint}')
    ckpt = torch.load(checkpoint, map_location='cpu')
    model_z_begin, model_z_end = install_globals(
        ckpt, patches_dir, pcl_paths, filter_z_begin, filter_z_end, umbilicus_path
    )
    print(f'checkpoint (model) z-range: [{model_z_begin}, {model_z_end}); '
          f'filtering z-range: [{filter_z_begin}, {filter_z_end})')
    if not (filter_z_begin >= model_z_begin and filter_z_end <= model_z_end):
        raise SystemExit(
            f'filtering z-range [{filter_z_begin}, {filter_z_end}) extends beyond the checkpoint '
            f'z-range [{model_z_begin}, {model_z_end}); the transform is undefined outside its domain.'
        )

    print('loading + z-filtering patches and point-collections')
    patches, _unverified, _shell, cross_patch_pcls, _unattached = fs.main(
        load_only_patches_and_point_collections=True
    )
    if patch_id not in patches:
        raise SystemExit(
            f'patch id {patch_id!r} not among {len(patches)} loaded patches (after z-filtering); '
            'check --patch-id / --z-range'
        )
    patch = patches[patch_id]

    print('rebuilding spiral transform from checkpoint')
    transform, dr = build_transform(ckpt, model_z_begin, model_z_end)
    dr_value = float(dr.item())
    print(f'dr_per_winding = {dr_value:.4f}')

    # --- seed: a valid grid vertex near the centre of the patch ---
    seed_ij = pick_centre_seed(patch)
    seed_zyx_ds, seed_valid = patch.ij_to_zyx(torch.from_numpy(seed_ij).to(device))
    assert bool(seed_valid.item())
    seed_zyx_ds = seed_zyx_ds.cpu().numpy()
    seed_model_winding_raw = float(
        winding_at_points(transform, dr, torch.from_numpy(seed_zyx_ds).to(device)[None])[0].item()
    )
    # Every other winding quantity here is integer (relative-pcl deltas, theta=0
    # strip adjustments, absolute anchors). Round the model's continuous seed winding
    # to match, so all reported windings -- and every patch's entry position -- are
    # clean integers.
    seed_model_winding = int(round(seed_model_winding_raw))
    print(f'patch {patch_id}: grid {tuple(patch.zyxs.shape[:2])}; '
          f'seed at grid-centre vertex ij={seed_ij.tolist()}; '
          f'model raw winding = {seed_model_winding_raw:.3f} (rounded to {seed_model_winding})')

    # --- build the absolute anchors and the relative-pcl patch graph ---
    abs_anchors_by_patch = build_abs_anchors_by_patch(cross_patch_pcls, patches)
    rel_adjacency = build_rel_adjacency(cross_patch_pcls, patches, transform, dr)
    # Per-pcl classification (absolute / relative / neither) so it is clear what each
    # pcl is doing: absolute pcls anchor windings, relative pcls supply long-range edges
    # with real winding deltas, and 'neither' pcls (fibers / new_same_wind) supply
    # same-winding (delta-0) edges only.
    pcl_summary = []
    for pid, pcl in sorted(cross_patch_pcls.items()):
        num_attached = sum(1 for p in pcl['points'].values() if 'on_patch' in p)
        pcl_summary.append({
            'id': int(pid),
            'name': pcl.get('name'),
            'kind': classify_pcl(pcl),
            'winding_is_absolute': bool(pcl.get('metadata', {}).get('winding_is_absolute', False)),
            'has_winding_annotations': bool(pcl.get('has_winding_annotations', False)),
            'num_points': len(pcl['points']),
            'num_attached_patches': len(pcl.get('points_by_patch', {})),
        })
    num_abs_pcls = sum(1 for s in pcl_summary if s['kind'] == 'absolute')
    num_rel_pcls = sum(1 for s in pcl_summary if s['kind'] == 'relative')
    num_neither_pcls = sum(1 for s in pcl_summary if s['kind'] == 'neither')
    num_rel_edges = sum(len(v) for v in rel_adjacency.values()) // 2
    print(f'{num_abs_pcls} absolute, {num_rel_pcls} relative, {num_neither_pcls} neither '
          f'(fiber/same-wind) pcl(s); absolute anchor {len(abs_anchors_by_patch)} patch(es); '
          f'relative+neither -> {num_rel_edges} cross-patch edge(s) over {len(rel_adjacency)} patch(es)')

    # Lazily-built, memoised valid-quad graph per patch (built once the first time
    # a strip is needed on that patch; carries a per-start Dijkstra cache).
    graph_cache = {}

    def graph_for(pid):
        g = graph_cache.get(pid)
        if g is None:
            g = cop.build_patch_graph(patches[pid], medial_weight)
            g._dij_cache = {}
            graph_cache[pid] = g
        return g

    # --- backwards BFS over patches from the seed patch ---
    # State: (patch P, entry ij q on P, acc) with acc = winding(seed) - winding(q).
    # At each patch its absolute anchors vote; each relative edge expands to an
    # unvisited neighbour. See module docstring for the propagation algebra.
    votes = {}
    voter_details = []
    # reached[P] = {'acc', 'entry_ij', 'hops'} for the first (tree) time P is reached;
    # acc = winding(seed) - winding(entry). Retaining the entry point lets a later
    # non-tree edge close the loop and measure its winding holonomy.
    reached = {patch_id: {'acc': 0, 'entry_ij': seed_ij, 'hops': 0}}
    tree_edges = set()   # undirected keys of edges used to grow the BFS tree
    tree_edge_by_child = {}  # child patch -> tree-edge hop record that first reached it (full BFS tree)
    loops = []           # non-tree closures: each carries the cycle's winding holonomy
    loops_seen = set()   # undirected keys already recorded as a loop (skip the reverse dir)
    num_loops_unmeasurable = 0
    # acc = winding(seed) - winding(entry); integer throughout (sum of integer
    # strip + edge deltas), so the predicted seed windings come out as integers.
    queue = deque([{'patch': patch_id, 'entry_ij': seed_ij, 'acc': 0, 'hops': 0, 'path': []}])

    def record_vote(expected, anchor, hops, acc, anchor_strip, path):
        expected_int = int(np.round(expected))
        voter = {
            'expected_seed_winding': expected,
            'expected_seed_winding_rounded': expected_int,
            'hops': hops,
            'abs_pcl_id': anchor['pcl_id'],
            'abs_pcl_name': anchor['pcl_name'],
            'abs_source_file': anchor['source_file'],
            'abs_point_id': anchor['point_id'],
            'abs_winding_annotation': anchor['winding'],
            'abs_patch_id': path[-1]['to_patch'] if path else patch_id,
            'abs_ij': anchor['ij'].tolist(),
            'abs_distance_to_patch': anchor['distance'],
            'abs_zyx_raw': _to_py(anchor['zyx']),
            # winding(seed) = acc_seed_minus_entry + abs_winding + anchor_to_entry_delta
            'acc_seed_minus_entry': acc,
            'anchor_to_entry_delta': anchor_strip['delta_windings'],
            'anchor_to_entry_residual_unwrapped_delta': anchor_strip['residual_unwrapped_delta_windings'],
            'anchor_to_entry_unwrap_adjustment': anchor_strip['unwrap_adjustment_windings'],
            'anchor_strip_num_path_quads': anchor_strip['strip_num_path_quads'],
            'anchor_strip_num_points': anchor_strip['strip_num_points'],
            'anchor_strip_num_invalid_dropped': anchor_strip['strip_num_invalid_dropped'],
            'path': path,
        }
        voter_details.append(voter)
        votes.setdefault(str(expected_int), []).append(voter)
        chain = '' if not path else ' via ' + ' -> '.join(
            [path[0]['from_patch']] + [h['to_patch'] for h in path])
        print(f'  [{hops} hop(s)] abs pcl {anchor["pcl_id"]} ({anchor["pcl_name"]!r}) '
              f'point {anchor["point_id"]} on patch {voter["abs_patch_id"]}: '
              f'annotation={anchor["winding"]:g} -> expected seed winding={expected:.3f} '
              f'(round {expected_int}){chain}')

    while queue:
        state = queue.popleft()
        P, q, acc, hops, path = state['patch'], state['entry_ij'], state['acc'], state['hops'], state['path']
        graph_P = graph_for(P)

        # (a) votes from absolute anchors on this patch: propagate from anchor to
        #     the entry point (strip_delta(P, anchor, q) = winding(q) - winding(anchor)).
        for anchor in abs_anchors_by_patch.get(P, []):
            anchor_strip = strip_winding_delta(transform, dr, graph_P, anchor['ij'], q, step_size)
            if anchor_strip is None:
                print(f'  [{hops} hop(s)] abs pcl {anchor["pcl_id"]} ({anchor["pcl_name"]!r}) '
                      f'point {anchor["point_id"]} on patch {P}: anchor->entry strip had <2 valid '
                      f'points; skipped')
                continue
            expected = acc + anchor['winding'] + anchor_strip['delta_windings']
            record_vote(expected, anchor, hops, acc, anchor_strip, path)

        # (b) relative edges: grow the BFS tree into unreached neighbours, and check
        #     edges that reconnect an already-reached patch for loop holonomy.
        for edge in rel_adjacency.get(P, []):
            R = edge['neighbor']
            edge_key = (edge['pcl_id'], frozenset((edge['from_point_id'], edge['to_point_id'])))

            if R in reached:
                # Non-tree edge -> it closes a cycle. Express winding(seed) - winding(e)
                # at the edge's arrival point e two ways and difference them:
                #   * via this edge from P: acc - strip(P, q->d) - edge_delta
                #   * via R's stored tree path: acc_R - strip(R, entry_R->e)
                # The gap is the cycle's winding holonomy (0 == the loop closes).
                if (not detect_loops) or edge_key in tree_edges or edge_key in loops_seen:
                    continue
                strip = strip_winding_delta(transform, dr, graph_P, q, edge['from_ij'], step_size)
                rstate = reached[R]
                rstrip = strip_winding_delta(transform, dr, graph_for(R), rstate['entry_ij'],
                                             edge['to_ij'], step_size)
                if strip is None or rstrip is None:
                    num_loops_unmeasurable += 1
                    continue  # can't measure one side; the reverse direction may still succeed
                acc_arrival_via_edge = acc - strip['delta_windings'] - edge['winding_delta']
                acc_arrival_via_tree = rstate['acc'] - rstrip['delta_windings']
                holonomy = acc_arrival_via_edge - acc_arrival_via_tree
                holonomy_int = int(round(holonomy))
                loops_seen.add(edge_key)
                loops.append({
                    'rel_pcl_id': edge['pcl_id'],
                    'rel_pcl_name': edge['pcl_name'],
                    'rel_source_file': edge['source_file'],
                    'from_patch': P,
                    'to_patch': R,
                    'from_ij': edge['from_ij'].tolist(),
                    'to_ij': edge['to_ij'].tolist(),
                    'from_point_id': edge['from_point_id'],
                    'to_point_id': edge['to_point_id'],
                    'from_zyx_raw': _to_py(edge['from_zyx']),
                    'to_zyx_raw': _to_py(edge['to_zyx']),
                    'edge_winding_delta': edge['winding_delta'],
                    'raw_edge_winding_delta': edge['raw_winding_delta'],
                    'pcl_unwrap_adjustment': edge['pcl_unwrap_adjustment'],
                    'pcl_branch_delta': edge['pcl_branch_delta'],
                    'from_depth': hops,
                    'to_depth': rstate['hops'],
                    # winding(seed) - winding(entry) of each end, so the plot can place the
                    # loop's departure (from_patch) and arrival (to_patch) ends in x.
                    'from_acc': acc,
                    'to_acc': rstate['acc'],
                    'intra_patch_strip_delta_from': strip['delta_windings'],
                    'intra_patch_strip_delta_to': rstrip['delta_windings'],
                    'loop_winding_delta': holonomy_int,
                    'loop_winding_residual': float(holonomy),
                    'is_inconsistent': holonomy_int != 0,
                })
                if holonomy_int != 0:
                    print(f'  LOOP inconsistency: rel pcl {edge["pcl_id"]} ({edge["pcl_name"]!r}) '
                          f'reconnects {P} (d{hops}) <-> {R} (d{rstate["hops"]}): winding holonomy '
                          f'{holonomy_int:+d} (residual {holonomy:+.3f})')
                continue

            # Unreached neighbour: grow the tree into it (bounded by --max-hops).
            if max_hops is not None and hops >= max_hops:
                continue
            # Propagate within P from the entry point to the edge's departure point.
            strip = strip_winding_delta(transform, dr, graph_P, q, edge['from_ij'], step_size)
            if strip is None:
                continue  # can't carry the winding across this edge; another path may reach R
            acc2 = acc - strip['delta_windings'] - edge['winding_delta']
            tree_edges.add(edge_key)
            reached[R] = {'acc': acc2, 'entry_ij': edge['to_ij'], 'hops': hops + 1}
            hop_record = {
                'rel_pcl_id': edge['pcl_id'],
                'rel_pcl_name': edge['pcl_name'],
                'rel_source_file': edge['source_file'],
                'from_patch': P,
                'to_patch': R,
                'from_ij': edge['from_ij'].tolist(),
                'to_ij': edge['to_ij'].tolist(),
                'from_point_id': edge['from_point_id'],
                'to_point_id': edge['to_point_id'],
                'from_zyx_raw': _to_py(edge['from_zyx']),
                'to_zyx_raw': _to_py(edge['to_zyx']),
                'edge_winding_delta': edge['winding_delta'],
                'raw_edge_winding_delta': edge['raw_winding_delta'],
                'pcl_unwrap_adjustment': edge['pcl_unwrap_adjustment'],
                'pcl_branch_delta': edge['pcl_branch_delta'],
                'intra_patch_strip_delta': strip['delta_windings'],
                'intra_patch_residual_unwrapped_delta': strip['residual_unwrapped_delta_windings'],
                'intra_patch_unwrap_adjustment': strip['unwrap_adjustment_windings'],
                'intra_patch_strip_num_path_quads': strip['strip_num_path_quads'],
                'intra_patch_strip_num_points': strip['strip_num_points'],
                'intra_patch_strip_num_invalid_dropped': strip['strip_num_invalid_dropped'],
            }
            tree_edge_by_child[R] = hop_record
            queue.append({'patch': R, 'entry_ij': edge['to_ij'], 'acc': acc2,
                          'hops': hops + 1, 'path': path + [hop_record]})

    agreeing = sorted(votes.keys(), key=lambda k: int(k))
    num_direct = sum(1 for v in voter_details if v['hops'] == 0)
    print(f'\nreached {len(reached)} patch(es); {len(voter_details)} vote(s) '
          f'({num_direct} direct, {len(voter_details) - num_direct} long-range) '
          f'across winding number(s): {", ".join(agreeing) if agreeing else "(none)"}')
    if len(agreeing) > 1:
        print('WARNING: votes disagree on the seed winding number!')

    num_inconsistent_loops = sum(1 for L in loops if L['is_inconsistent'])
    if detect_loops:
        print(f'checked {len(loops)} non-tree relative-edge loop(s): '
              f'{num_inconsistent_loops} with nonzero winding holonomy'
              + (f', {num_loops_unmeasurable} unmeasurable' if num_loops_unmeasurable else ''))
        if num_inconsistent_loops:
            print('WARNING: relative-winding loops do not close (winding inconsistency in the graph)!')

    # Full BFS tree: every reached patch (depth + acc = winding(seed) - winding(entry))
    # and the relative-pcl edge that first reached it. The "votes" only record the
    # subtree of patches leading to absolute anchors; emitting the whole tree lets
    # plot_winding_graph draw the entire graph -- and therefore every detected loop,
    # including loops whose patches no voter passed through.
    tree = {
        'nodes': {pid: {'depth': info['hops'], 'acc': info['acc']}
                  for pid, info in reached.items()},
        'edges': tree_edge_by_child,
    }

    # Per-loop view: each inconsistent loop's full closed cycle (tree path between
    # its two patches + the closing edge), as an ordered patch/pcl stack that
    # plot_winding_graph renders one loop at a time.
    def loop_patch_strip_delta(pid, top_ij, bot_ij):
        """Within-patch winding delta winding(bot) - winding(top) across patch `pid`
        between two of its points, as the integer theta=0 seam transport along a
        within-patch valid-quad strip (None if unmeasurable). Same machinery as the
        BFS's intra_patch_strip_delta, used here for each loop patch's two cycle
        points (which, at the LCA / loop endpoints, are not the BFS tree points)."""
        s = strip_winding_delta(transform, dr, graph_for(pid),
                                np.asarray(top_ij, dtype=np.float32),
                                np.asarray(bot_ij, dtype=np.float32), step_size)
        return None if s is None else s['delta_windings']

    loop_cycles = build_loop_cycles(loops, tree_edge_by_child, loop_patch_strip_delta)

    # Edges lying on the inconsistent fundamental cycles (each cycle's tree-path steps plus
    # its closing edge), keyed exactly as solve_min_edge_fix keys edges. Only these edges
    # can close a broken loop, so the MILP is restricted to them -- edges on only-consistent
    # cycles add gauge freedom that lets a hub patch's potential float to its box bound and
    # produce huge spurious residuals.
    def _edge_key(step):
        return (step['rel_pcl_id'], frozenset((step['from_point_id'], step['to_point_id'])))
    inconsistent_cycle_edge_keys = set()
    for cyc in loop_cycles:
        for step in cyc['steps']:
            inconsistent_cycle_edge_keys.add(_edge_key(step))
        inconsistent_cycle_edge_keys.add(_edge_key(cyc['closing_step']))

    # --- minimal-fix MILP: fewest relative-edge annotations to make the graph close ---
    min_edge_fix = None
    if min_fix and detect_loops:
        print(f'\nsolving minimal-fix MILP (fewest relative edges to change so all loops close); '
              f'restricted to {len(inconsistent_cycle_edge_keys)} edge(s) on inconsistent cycle(s)')
        min_edge_fix = solve_min_edge_fix(reached, rel_adjacency, loop_patch_strip_delta,
                                          allowed_edge_keys=inconsistent_cycle_edge_keys,
                                          time_limit=min_fix_time_limit)
        print(f'minimal fix: change {min_edge_fix["num_edges_changed"]} relative edge(s) '
              f'(of {min_edge_fix["num_edges_considered"]} considered'
              + (f', {min_edge_fix["num_edges_unmeasurable"]} unmeasurable' if min_edge_fix["num_edges_unmeasurable"] else '')
              + f'); solver: {min_edge_fix["solver_status"]}')
        for e in min_edge_fix['edges']:
            if e['action'] == 'detach_or_reattach':
                # 'neither' pcl: no winding number to edit; one of its two points is on the
                # wrong winding -> delete it or re-attach it to the correct patch.
                print(f'  fix same-wind pcl {e["rel_pcl_id"]} ({e["rel_pcl_name"]!r}): delete or '
                      f're-attach point {e["from_point_id"]} (on patch {e["from_patch"]}) or '
                      f'point {e["to_point_id"]} (on patch {e["to_patch"]}); the pcl puts them on '
                      f'the same winding but the graph needs them {e["residual"]:+d} apart')
            else:
                print(f'  change rel pcl {e["rel_pcl_id"]} ({e["rel_pcl_name"]!r}) '
                      f'point {e["from_point_id"]}->{e["to_point_id"]} '
                      f'({e["from_patch"]} -> {e["to_patch"]}): winding delta '
                      f'{e["current_winding_delta"]:+d} -> {e["suggested_winding_delta"]:+d} '
                      f'(residual {e["residual"]:+d})')

    # --- disconnected subgraphs: rel-edge patch clusters the seed BFS never reaches ---
    disconnected_subgraphs = find_disconnected_subgraphs(
        rel_adjacency, patches, patch_id, top_patches_per_subgraph)
    seed_comp = disconnected_subgraphs['seed_component_num_patches']
    comps = disconnected_subgraphs['components']
    print(f'\ndisconnected rel-edge subgraphs (seed\'s component has {seed_comp} patch(es)):')
    if not comps:
        print('  all relative-edge patches are connected to the seed (none disconnected)')
    else:
        print(f'  {len(comps)} component(s) not connected to the seed; largest by patch count:')
        for c in comps[:max_subgraphs]:
            largest = ', '.join(f'{p["patch_id"]} (area {p["area"]:.1f})' for p in c['largest_patches'])
            print(f'  [#{c["rank"]}] {c["num_patches"]} patch(es), {c["num_edges"]} edge(s), '
                  f'area {c["total_area"]:.1f} -> largest: {largest}')
        if len(comps) > max_subgraphs:
            print(f'  ... and {len(comps) - max_subgraphs} more (raise --max-subgraphs to see them)')
    # Keep the count totals (num_disconnected_components) intact, but cap the stored
    # component list to --max-subgraphs to match what was printed.
    disconnected_subgraphs['components'] = comps[:max_subgraphs]

    out = {
        'checkpoint': os.path.abspath(checkpoint),
        'patches_dir': os.path.abspath(patches_dir),
        'umbilicus': os.path.abspath(umbilicus_path),
        'patch_id': patch_id,
        'pcl_paths': [os.path.abspath(p) for p in pcl_paths],
        'model_z_range': [model_z_begin, model_z_end],
        'filter_z_range': [filter_z_begin, filter_z_end],
        'step_size': step_size,
        'medial_weight': medial_weight,
        'max_hops': max_hops,
        'detect_loops': detect_loops,
        'max_subgraphs': max_subgraphs,
        'top_patches_per_subgraph': top_patches_per_subgraph,
        'seed': {
            'ij': seed_ij.tolist(),
            'zyx': seed_zyx_ds.tolist(),
            'xyz': [float(seed_zyx_ds[2]), float(seed_zyx_ds[1]), float(seed_zyx_ds[0])],
        },
        'dr_per_winding': dr_value,
        'model_raw_winding_at_seed': seed_model_winding,
        'num_absolute_pcls': num_abs_pcls,
        'num_relative_pcls': num_rel_pcls,
        'num_neither_pcls': num_neither_pcls,
        'point_collections': pcl_summary,
        'num_relative_cross_patch_edges': num_rel_edges,
        'num_patches_reached': len(reached),
        'num_direct_votes': num_direct,
        'num_long_range_votes': len(voter_details) - num_direct,
        'votes_agree': len(agreeing) <= 1,
        'num_loops_checked': len(loops),
        'num_inconsistent_loops': num_inconsistent_loops,
        'num_loops_unmeasurable': num_loops_unmeasurable,
        'votes': votes,
        'loops': loops,
        'tree': tree,
        'loop_cycles': loop_cycles,
        'min_edge_fix': min_edge_fix,
        'disconnected_subgraphs': disconnected_subgraphs,
    }

    if output is None:
        output = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), f'winding_votes_{patch_id}.json')
    with open(output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {output}')

    if plot:
        # Render straight from the json we just wrote: the full graph (every reached
        # patch + connections) with the max-winding readout. Imported lazily so a
        # normal (no-plot) run never pulls in matplotlib.
        print('\nrendering full winding graph')
        import plot_winding_graph as pwg
        pwg.render(output, output=plot_output, full_graph=True)


if __name__ == '__main__':
    main()
