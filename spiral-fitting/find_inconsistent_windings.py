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

Before constructing graph edges, same-winding fiber attachments are checked
against the first contact with each patch. Transport follows the complete fiber
chain, including linked horizontal/vertical continuations and junction seam
crossings, and is compared with transport along the patch. Later contacts on a
different relative turn are rejected from the analysis graph. The first contact
is defined by the deterministic member-tree traversal; Euler backtracking does
not create another contact. Non-tree fiber links remain cycle constraints.
This preserves genuine within-patch winding changes and rejects a fiber winding
around and accidentally touching another turn of the same patch. Source files
are never changed. Rejections appear in "attachment_return_filter"; a subsequent
independent per-member return audit appears in "attachment_returns".

All attachment connectors include source-to-projected-point seam transport.
Graph point IDs use collection dictionary keys (unique in merged components);
source point/collection IDs and component member files are recorded separately.
An inconsistent graph cycle implicates its constraints; it does not on its own
establish a sheet switch inside a patch or fiber.

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

Alternatively, pass --config params.json with any or all options in a JSON object:

    {
        "checkpoint": "out/checkpoint_fitted.ckpt",
        "patches_dir": "/path/to/dataset/patches",
        "patch_id": "seed-patch",
        "pcl": ["/path/to/abs_windings.json", "/path/to/rel_windings.json"],
        "z_range": "10000,13000",
        "detect_loops": true,
        "plot": false
    }

Keys use option names without leading dashes; hyphens and underscores are both
accepted. Use JSON booleans for toggles, an array for pcl, and an empty string
for fibers to infer the dataset's fibers directory. Relative paths are resolved
from the working directory, as on the CLI. Explicit CLI options override JSON
values; omitted options retain their usual defaults.

An optional JSON-only "linking" object controls analysis attachments (omitting
it preserves checkpoint/legacy linking). For example:

    "linking": {
        "distance": 2.5,
        "window_points": 5,
        "min_points": 3,
        "vertical_max_behind_fraction": 0.2,
        "horizontal_max_front_fraction": 0.2,
        "side_margin_voxels": 0.5,
        "selection": "support_then_distance",
        "fiber_stride": 1,
        "patch_stride": 1,
        "strips": false,
        "strip_max_gap": null,
        "strip_claim_gap": null,
        "strip_rank": "fit",
        "strip_fit_resolution": 0.25
    }

Distance (positive float, full-resolution scan voxels) defaults to the saved
checkpoint tolerance and affects every collection. Patch stride also affects
every collection, but only its link-query geometry; winding analysis uses the
original patches. Coarse cells spanning invalid fine quads are excluded.
Absolute winding annotations attach independently within the distance tolerance;
checkpoint window-support requirements and fiber side rules do not apply to them.

The remaining settings affect fibers only. Stride samples the original trimmed
fiber, replacing checkpoint spacing-based thinning, and always keeps endpoints
and branch junctions. Windows count sampled points within each individual fiber
(before merging fibers), including the center point. window_points defaults to
5 and must be odd; min_points defaults to 1 and is strict even at fiber ends.
The center point must itself be within distance of the candidate patch.

Front points inward along the checkpoint's negative winding gradient in scan
space. Each orientation's wrong-side fraction defaults to null (disabled).
Enabled fractions are in [0,1] and count nearby hits in the local window, with
offsets within side_margin_voxels (default 0.5) treated as on-surface. Undefined
directions do not support a side-filtered link. A fraction equal to the limit
passes. Eligible patches rank by support count, RMS distance, center distance,
then patch ID. selection may instead be "nearest" or "largest_area". Both
strides default to 1.

An optional JSON-only "edge_divergence" object drops same-winding fiber edges
whose two patches diverge away from the hop (see edge_divergence.py):

    "edge_divergence": {
        "radius": 60.0,
        "max_median_offset": 2.0,
        "max_spread": null,
        "min_samples": 6,
        "kinds": ["neither"],
        "report_kept": false
    }

Vertices of each patch within radius voxels of the fiber's hop are projected
onto the other patch and their offsets signed along the umbilicus-inward
direction; a direction with at least min_samples samples is measured, and the
edge is dropped when a measured direction's median absolute offset exceeds
max_median_offset or, with max_spread set (default null: off), its p10-p90
spread exceeds it. Edges measured in
neither direction (long hops) are kept and counted. Settings, counts and the
dropped edges (plus kept ones with report_kept) are saved under
"edge_divergence" in the report.

strips (default false) switches fibers from per-point selection to
strip-level winner-take-all. A strip is one fiber's sampled points hitting one
patch, from the first hit to the last; strip_max_gap (default null, unlimited)
starts a new strip after that many consecutive points without a hit.
window_points is unused and min_points is the minimum number of hits per strip.
strip_rank "fit" (default) ranks strips by RMS hit distance in
strip_fit_resolution voxel bins (default 0.25), then more hits; "hits" ranks by
more hits, then fit bins. Both then prefer larger patch area and patch ID. Each
strip claims every point of its span that no better strip already owns, so a
stretch of fiber never alternates between two patches; strip_claim_gap (default
null: the whole span) limits the claim to the hits and the holes of at most
that many points between them, leaving longer holes open to other patches. What a losing strip keeps is split at the claimed points and at its
gap rule, and fragments below min_points are dropped. Wrong-side fractions are
tested per strip. Resolved settings and counts are saved under "linking" in
the report; candidate rejection counts refer to (sampled point, patch) pairs,
and in strip mode strip_overruled_candidates counts eligible hits that a better
strip claimed.
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
from edge_divergence import EdgeDivergenceConfig, filter_divergent_edges
from point_collection import umbilicus_inward_direction
from winding_linking import AnalysisLinkingConfig, AnalysisPointLinker, model_inward_direction


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


def resolve_fibers_path(patches_dir, fibers_flag):
    """Resolve the fibers directory for --fibers.

    --fibers may be given a path (used verbatim), or passed as a bare flag with no
    value (sentinel ''), in which case we infer the dataset's `fibers/` dir next to
    --patches-dir. Returns None when fibers are not requested."""
    if fibers_flag is None:
        return None
    if fibers_flag:
        path = os.path.abspath(os.path.expanduser(fibers_flag))
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(patches_dir))), 'fibers')
    if not os.path.isdir(path):
        raise SystemExit(f'--fibers directory not found: {path}')
    return path


def _to_py(x):
    """Convert numpy / torch scalars+arrays to plain python for json."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def resolve_base_shape_zyx(checkpoint, patches_dir):
    """The dataset's volume coordinate domain (ZYX shape), or None.

    Fibers exported with coordinate_base_shape_zyx can only be scaled into the
    dataset's domain when that domain is known, so the analysis ScrollSpec must
    carry it like a real fit's does. Prefer the shape the checkpoint was fit
    against; legacy checkpoints predate that field, so fall back to the
    dataset's spiral-scroll.json next to --patches-dir when one exists."""
    shape = checkpoint.get('base_shape_zyx')
    if shape is not None:
        return tuple(int(v) for v in shape)
    from fit_session import SCROLL_SPEC_FILENAME, ScrollSpecError, load_scroll_spec
    dataset_root = os.path.dirname(os.path.abspath(os.path.expanduser(patches_dir)))
    if not os.path.isfile(os.path.join(dataset_root, SCROLL_SPEC_FILENAME)):
        return None
    try:
        return load_scroll_spec(dataset_root).base_shape_zyx
    except ScrollSpecError as e:
        print(f'WARNING: ignoring {SCROLL_SPEC_FILENAME}: {e}')
        return None


def build_fit_inputs(checkpoint, patches_dir, pcl_paths, filter_z_begin, filter_z_end, umbilicus_path,
                     fibers_path=None):
    """Build the explicit FitConfig / ScrollSpec / SpiralInputPaths pointing
    fit_spiral's loaders at the user-supplied patches/pcls/filter-z-range so
    they behave identically to a fit.
    Returns (fit_config, scroll, paths, model_z_begin, model_z_end); the model
    z-range shapes the transform's flow field independently of the filtering
    z-range."""
    from fit_session import PclInputSpec, ScrollSpec, SpiralInputPaths
    from config import filter_known_config_keys

    cfg = fs.Config().as_dict()
    # Analysis needs the saved model/input settings, not an exact match to
    # the training config schema. Backfill new defaults and report old keys.
    cfg.update(filter_known_config_keys(
        checkpoint['cfg'], cfg, label='checkpoint config'))
    # This tool IS the patch graph — a checkpoint trained supervision-free
    # (disable_patches, e.g. the 2026-07-17 normals-only baseline) must not
    # stop the loaders from reading the patches it wants to analyse.
    cfg['input_disable_patches'] = False
    # Analysis needs neither shell supervision nor fiber-direction loss inputs.
    # Disable the shell source too: winding-model checkpoints require it even
    # with both shell loss weights zero.
    cfg['input_use_outer_shell'] = False
    cfg['loss_weight_shell_outer'] = 0.0
    cfg['loss_weight_shell_patch_radius'] = 0.0
    cfg['loss_weight_fiber_directions'] = 0.0
    # The filtering z-range is the optimisation window this analysis uses.
    cfg['z_begin'] = int(filter_z_begin)
    cfg['z_end'] = int(filter_z_end)
    fit_config = fs.FitConfig(cfg)
    scroll = ScrollSpec(
        name='analysis',
        voxel_size_um=9.6,
        spiral_outward_sense=checkpoint.get('spiral_outward_sense') or 'CW',
        base_shape_zyx=resolve_base_shape_zyx(checkpoint, patches_dir),
    )
    # This analysis needs no scroll volume, track store, or outer shell:
    # leaving those paths empty keeps load_host_inputs() from touching the
    # training-only inputs. Legacy role-less PCL specs (role=None) retain the
    # historical abs_winding.json basename inference. Attachment is over the
    # verified patch set only, so the (slow, unrelated) unverified patches
    # are skipped; they don't change the cross-patch / attached pcl set.
    # Fibers are same-winding annotations: when loaded they classify as 'neither'
    # pcls (delta-0 cross-patch edges), exactly like same_windings.json. Off
    # unless a fibers dir is passed.
    paths = SpiralInputPaths(
        umbilicus=umbilicus_path,
        pcls=tuple(PclInputSpec(path=str(path), role=None)
                   for path in pcl_paths),
        fibers=fibers_path or '',
        verified_patches=patches_dir,
    )
    return fit_config, scroll, paths, int(checkpoint['z_begin']), int(checkpoint['z_end'])


def build_transform(checkpoint, cfg, context, model_z_begin, model_z_end, *, device='cuda'):
    """Reconstruct the slice->spiral transform and dr_per_winding from the
    checkpoint, using fit_spiral's own model class. The flow field is shaped by
    the checkpoint's (model) z-range, not the filtering z-range."""
    device = torch.device(device)

    all_zs = np.arange(model_z_begin, model_z_end)
    umbilicus_fn = context.umbilicus_z_to_yx()
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], umbilicus_fn(all_zs)], axis=-1).astype(np.float32)
    ).to(device)

    r = cfg['model_flow_bounds_radius']
    flow_min_corner = torch.tensor([model_z_begin - cfg['model_flow_bounds_z_margin'], -r, -r], dtype=torch.int64, device=device)
    flow_max_corner = torch.tensor([model_z_end + cfg['model_flow_bounds_z_margin'], r, r], dtype=torch.int64, device=device)

    model = fs.SpiralAndTransform(
        flow_integration_steps=cfg['model_num_flow_integration_steps'],
        flow_integration_solver=cfg['model_flow_integration_solver'],
        umbilicus_zyx=umbilicus_zyx,
        flow_min_corner_zyx=flow_min_corner,
        flow_max_corner_zyx=flow_max_corner,
        config=cfg,
        spiral_outward_sense=context.spiral_outward_sense,
    )
    model.to(device)
    model_state = dict(fs.merge_flow_stage_lattices(
        checkpoint)['spiral_and_transform'])
    # Older fits learned a shell-loss winding index. It is not part of the
    # slice->spiral transform and is no longer a model parameter. Keep strict
    # loading for every other key so missing/incompatible geometry still fails.
    model_state.pop('shell_outer_winding_param', None)
    model.load_state_dict(model_state)
    model.eval()

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
    (fit_spiral._unwrap_track_shifted_radii) stitches only genuine theta=0 seam
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
    step_adjustment_windings = (
        get_theta_crossing_step_adjustments(theta, dr) / dr.detach()
    )
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


def _sample_polylines_batch(polylines, step_size):
    """Vectorised _polyline_ijs over many polylines at once.

    `polylines` is a list of (K_k, 2) float32 vertex arrays. Returns
    (ijs, owner): the concatenated samples (N, 2) float32 and, per sample, the
    index of the polyline it belongs to. Each polyline's samples are exactly the
    array _polyline_ijs(vertices, step_size) would produce (same float32
    arithmetic, sample for sample), so the two are interchangeable."""
    if not polylines:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)
    seg_a = np.concatenate([p[:-1] for p in polylines], axis=0).astype(np.float32, copy=False)
    seg_b = np.concatenate([p[1:] for p in polylines], axis=0).astype(np.float32, copy=False)
    seg_owner = np.repeat(np.arange(len(polylines)), [len(p) - 1 for p in polylines])
    seg_first = np.ones(len(seg_a), dtype=bool)
    seg_first[1:] = seg_owner[1:] != seg_owner[:-1]

    # Segment lengths: float32 hypot exactly like np.linalg.norm on a float32
    # 2-vector (dot then sqrt, no fused multiply-add). Only ceil(len / step) is
    # consumed, so a last-bit difference could only matter right at an integer
    # quotient; re-measure those few segments with the scalar reference call.
    d = seg_b - seg_a
    seg_len = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).astype(np.float64)
    quotient = seg_len / step_size
    near_integer = np.flatnonzero(np.abs(quotient - np.round(quotient)) < 1e-4)
    for m in near_integer:
        quotient[m] = float(np.linalg.norm(seg_b[m] - seg_a[m])) / step_size
    counts = np.maximum(1, np.ceil(quotient).astype(np.int64)) + 1  # samples per segment

    # Per-sample segment id and rank within the segment; the duplicated junction
    # vertex (rank 0 of every non-leading segment) is dropped, as in _polyline_ijs.
    sample_seg = np.repeat(np.arange(len(seg_a)), counts)
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(sample_seg)) - seg_start[sample_seg]
    keep = seg_first[sample_seg] | (rank > 0)
    sample_seg = sample_seg[keep]
    rank = rank[keep]

    # np.linspace(0, 1, n, dtype=float32): float64 rank * (1 / (n - 1)), the
    # endpoint pinned to exactly 1.0, then cast.
    div = (counts - 1).astype(np.float64)
    t = (rank.astype(np.float64) * (1.0 / div[sample_seg])).astype(np.float32)
    t[rank == counts[sample_seg] - 1] = 1.0
    t = t[:, None]
    ijs = seg_a[sample_seg] * (1.0 - t) + seg_b[sample_seg] * t
    return ijs.astype(np.float32, copy=False), seg_owner[sample_seg]


def _bilinear_ij_to_zyx_np(zyxs, valid_quad, ijs):
    """numpy twin of tifxyz.Patch.ij_to_zyx (bilinear, not triangulated).

    `zyxs` is the patch's (H, W, 3) float32 vertex grid, `valid_quad` its
    (H-1, W-1) bool quad mask, `ijs` (N, 2) float32. Returns (zyx, valid) with
    invalid samples filled with -1, using the same float32 operation sequence as
    the torch implementation so results agree bit for bit."""
    h, w = zyxs.shape[:2]
    i = ijs[:, 0]
    j = ijs[:, 1]
    i0 = np.floor(i).astype(np.int64)
    j0 = np.floor(j).astype(np.int64)
    in_bounds = (i0 >= 0) & (j0 >= 0) & (i0 < h - 1) & (j0 < w - 1)
    valid = np.zeros(len(ijs), dtype=bool)
    if in_bounds.any():
        valid[in_bounds] = valid_quad[i0[in_bounds], j0[in_bounds]]
    out = np.full((len(ijs), 3), -1.0, dtype=np.float32)
    if valid.any():
        vi0 = i0[valid]
        vj0 = j0[valid]
        di = (i[valid] - vi0.astype(np.float32))[:, None]
        dj = (j[valid] - vj0.astype(np.float32))[:, None]
        tl = zyxs[vi0, vj0]
        tr = zyxs[vi0, vj0 + 1]
        bl = zyxs[vi0 + 1, vj0]
        br = zyxs[vi0 + 1, vj0 + 1]
        top = tl + (tr - tl) * dj
        bottom = bl + (br - bl) * dj
        out[valid] = top + (bottom - top) * di
    return out, valid


class StripMeasurer:
    """Batched, memoised within-patch strip winding measurements.

    Measures exactly what strip_winding_delta measures (that function stays as
    the single-strip reference), but for many strips at once: the fringe-avoiding
    valid-quad routing and polyline sampling run vectorised in numpy, the
    spiral transform runs on one concatenated point cloud per batch instead of
    once per strip, and the theta=0 crossing bookkeeping is done in numpy on the
    transformed thetas. The transform is elementwise per point, so batching does
    not change any sample's spiral-space coordinates; the integer winding
    delta and the raw-winding diagnostics are identical to the reference. The
    unwrapped-residual diagnostic is accumulated sequentially in float32 (the
    reference used a device cumsum), which can differ in the last bits only on
    strips crossing the seam three or more times.

    Every measured strip is memoised by (patch, from_ij, to_ij), so callers may
    `prefetch` every strip a phase might need and then `measure` them one at a
    time in whatever order the algorithm dictates."""

    def __init__(self, transform, dr, patches, medial_weight, step_size, *,
                 max_transform_points=1 << 20):
        self.transform = transform
        self.dr = dr
        self.dr_value = np.float32(dr.detach().float().cpu().item())
        self.patches = patches
        self.medial_weight = medial_weight
        self.step_size = float(step_size)
        self.max_transform_points = int(max_transform_points)
        self._graphs = {}
        self._results = {}
        self.num_measured = 0
        self.num_batches = 0

    def graph_for(self, pid):
        """Lazily-built, memoised valid-quad graph of a patch (with its per-start
        Dijkstra predecessor cache)."""
        g = self._graphs.get(pid)
        if g is None:
            g = cop.build_patch_graph(self.patches[pid], self.medial_weight,
                                      lift_quad_centres=False)
            g._dij_cache = {}
            self._graphs[pid] = g
        return g

    @staticmethod
    def _key(pid, from_ij, to_ij):
        a = np.asarray(from_ij, dtype=np.float32)
        b = np.asarray(to_ij, dtype=np.float32)
        return (pid, float(a[0]), float(a[1]), float(b[0]), float(b[1]))

    def measure(self, pid, from_ij, to_ij):
        """strip_winding_delta(transform, dr, graph_for(pid), from_ij, to_ij,
        step_size), served from the memo (measuring on a miss)."""
        key = self._key(pid, from_ij, to_ij)
        if key not in self._results:
            self.prefetch([(pid, from_ij, to_ij)])
        return self._results[key]

    def prefetch(self, requests):
        """Measure every not-yet-memoised (pid, from_ij, to_ij) in one batch."""
        pending = {}
        for pid, from_ij, to_ij in requests:
            key = self._key(pid, from_ij, to_ij)
            if key not in self._results and key not in pending:
                pending[key] = (pid, np.asarray(from_ij, dtype=np.float32),
                                np.asarray(to_ij, dtype=np.float32))
        if not pending:
            return
        keys = list(pending.keys())
        results = self._measure_batch([pending[k] for k in keys])
        for key, result in zip(keys, results):
            self._results[key] = result
        self.num_measured += len(keys)
        self.num_batches += 1

    def _measure_batch(self, requests):
        n_req = len(requests)
        results = [None] * n_req

        # --- route through valid quads; collect polylines per request ---
        polylines = []       # (K, 2) float32 vertex arrays
        routed = []          # request index per polyline
        num_path_quads = []
        for k, (pid, from_ij, to_ij) in enumerate(requests):
            graph = self.graph_for(pid)
            centres = _valid_quad_path_centres(
                graph,
                _nearest_valid_quad_node(graph, from_ij),
                _nearest_valid_quad_node(graph, to_ij),
            )
            if centres is None:
                continue
            verts = np.concatenate([
                from_ij[None],
                np.array(centres, dtype=np.float32).reshape(-1, 2),
                to_ij[None],
            ], axis=0)
            polylines.append(verts)
            routed.append(k)
            num_path_quads.append(len(centres))
        if not polylines:
            return results
        routed = np.asarray(routed, dtype=np.int64)

        # --- sample every polyline, lift the samples onto their patch ---
        ijs, owner = _sample_polylines_batch(polylines, self.step_size)
        zyx_all = np.empty((len(ijs), 3), dtype=np.float32)
        valid_all = np.zeros(len(ijs), dtype=bool)
        # Group the samples by patch (one stable sort) and lift each group with
        # its patch's grid.
        pids = []
        pid_index = {}
        poly_pid = np.empty(len(polylines), dtype=np.int64)
        for p, k in enumerate(routed):
            pid = requests[k][0]
            if pid not in pid_index:
                pid_index[pid] = len(pids)
                pids.append(pid)
            poly_pid[p] = pid_index[pid]
        sample_pid = poly_pid[owner]
        order = np.argsort(sample_pid, kind='stable')
        bounds = np.searchsorted(sample_pid[order], np.arange(len(pids) + 1))
        for pi, pid in enumerate(pids):
            rows = order[bounds[pi]:bounds[pi + 1]]
            graph = self.graph_for(pid)
            zyxs = graph.patch.zyxs.detach().cpu().numpy()
            zyx_all[rows], valid_all[rows] = _bilinear_ij_to_zyx_np(zyxs, graph.valid_quad, ijs[rows])

        # Per polyline: dropped-invalid count and valid-sample count (strips with
        # fewer than two valid samples are unmeasurable, as in the reference).
        n_poly = len(polylines)
        num_samples = np.bincount(owner, minlength=n_poly)
        num_valid = np.bincount(owner, weights=valid_all, minlength=n_poly).astype(np.int64)
        num_invalid = num_samples - num_valid
        measurable = num_valid >= 2
        keep = measurable[owner] & valid_all
        zyx = zyx_all[keep]
        strip_of_point = owner[keep]  # polyline index per kept point, non-decreasing

        # --- one transform pass over every kept sample ---
        theta = np.empty(len(zyx), dtype=np.float32)
        shifted = np.empty(len(zyx), dtype=np.float32)
        device = self.dr.device
        with torch.no_grad():
            for start in range(0, len(zyx), self.max_transform_points):
                chunk = torch.from_numpy(zyx[start:start + self.max_transform_points]).to(device)
                spiral = self.transform(chunk)
                th, _, sh = get_theta_and_radii(spiral[..., 1:], self.dr)
                theta[start:start + len(chunk)] = th.float().cpu().numpy()
                shifted[start:start + len(chunk)] = sh.float().cpu().numpy()

        # --- theta=0 seam crossings per strip (numpy twin of the torch helpers) ---
        dr = self.dr_value
        strip_start = np.concatenate([[0], np.cumsum(np.bincount(strip_of_point, minlength=n_poly))])
        counts = np.diff(strip_start)
        diffs = theta[1:] - theta[:-1]
        same_strip = strip_of_point[1:] == strip_of_point[:-1]
        pi = np.float32(np.pi)  # torch compares float32 tensors against float32(pi) too
        sign = ((diffs > pi).astype(np.float32) - (diffs < -pi).astype(np.float32))
        sign[~same_strip] = 0.0
        step_adjust = sign * dr                     # get_theta_crossing_step_adjustments
        step_windings = step_adjust / dr           # exactly -1, 0 or +1 per step
        diff_strip = strip_of_point[:-1]
        cumulative_windings = np.bincount(diff_strip, weights=step_windings.astype(np.float64),
                                          minlength=n_poly)  # exact small integers
        # Sequential float32 sum of the (few) nonzero step adjustments per strip:
        # zeros add exactly, so this is the strip's running cumsum end value.
        seam_total = np.zeros(n_poly, dtype=np.float32)
        nonzero = np.flatnonzero(step_adjust != 0.0)
        if len(nonzero):
            nz_strip = diff_strip[nonzero]
            nz_value = step_adjust[nonzero]
            nz_rank = np.arange(len(nonzero)) - np.searchsorted(nz_strip, nz_strip, side='left')
            for r in range(int(nz_rank.max()) + 1):
                sel = nz_rank == r
                seam_total[nz_strip[sel]] = seam_total[nz_strip[sel]] + nz_value[sel]

        first = strip_start[:-1]
        last = strip_start[1:] - 1
        for p in np.flatnonzero(measurable):
            shifted_first = shifted[first[p]]
            shifted_last = shifted[last[p]]
            unwrapped_last = shifted_last + seam_total[p]
            cumulative = float(np.float32(cumulative_windings[p]))
            results[routed[p]] = {
                'delta_windings': int(round(-cumulative)),
                'residual_unwrapped_delta_windings': float((unwrapped_last - shifted_first) / dr),
                'unwrap_adjustment_windings': cumulative,
                'from_raw_winding': float(shifted_first / dr),
                'to_raw_winding': float(shifted_last / dr),
                'strip_num_path_quads': int(num_path_quads[p]),
                'strip_num_points': int(counts[p]),
                'strip_num_invalid_dropped': int(num_invalid[p]),
            }
        return results


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


def attachment_branch_deltas(cross_patch_pcls, patches, transform, dr):
    """Integer branch transport from each source point to its patch endpoint.

    Fiber paths use source coordinates; patch strips use bilinear coordinates.
    Even a sub-voxel projection can cross theta=0. Those short connectors must
    be included to make the cycle a closed geometric path.
    """
    by_patch = {}
    for pcl in cross_patch_pcls.values():
        for point in pcl['points'].values():
            attachment = point.get('on_patch')
            if attachment is not None and attachment['id'] in patches:
                by_patch.setdefault(attachment['id'], {})[id(point)] = point
    result = {}
    for pid, unique in by_patch.items():
        points = list(unique.values())
        patch = patches[pid]
        ijs = np.asarray([p['on_patch']['ij'] for p in points], dtype=np.float32)
        feet, valid = _bilinear_ij_to_zyx_np(
            patch.zyxs.detach().cpu().numpy(),
            patch.valid_quad_mask.detach().cpu().numpy(), ijs)
        # Do not invent transport to an endpoint outside the valid surface.
        # Callers omit attachments absent from the result; never transform the
        # invalid -1 coordinate sentinel.
        selected = np.flatnonzero(valid)
        if not len(selected):
            continue
        raw = np.asarray([points[i]['zyx'] for i in selected], dtype=np.float32)
        pairs = np.stack([raw, feet[selected]], axis=1)
        with torch.no_grad():
            spiral = transform(torch.as_tensor(pairs.reshape(-1, 3), device=dr.device))
            theta, _, _ = get_theta_and_radii(spiral[..., 1:], dr)
            delta = -get_theta_crossing_step_adjustments(theta.reshape(-1, 2), dr) / dr
        for i, value in zip(selected, delta.cpu().numpy().reshape(-1)):
            result[id(points[i])] = int(round(float(value)))
    return result


def build_abs_anchors_by_patch(cross_patch_pcls, patches, attachment_deltas=None):
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
                if attachment_deltas is not None and id(p) not in attachment_deltas:
                    continue
                w = float(p['winding_annotation'])
                if not np.isfinite(w):
                    continue
                anchors.setdefault(patch_id, []).append({
                    'pcl_id': int(pid),
                    'pcl_name': pcl.get('name'),
                    'source_file': pcl.get('source_file'),
                    'point_id': int(p['id']),
                    # Absolute-winding annotations are integer winding numbers.
                    'winding': int(round(w)) + (attachment_deltas or {}).get(id(p), 0),
                    'source_winding': int(round(w)),
                    'attachment_branch_delta': (attachment_deltas or {}).get(id(p), 0),
                    'ij': np.array(p['on_patch']['ij'], dtype=np.float32),
                    'distance': float(p['on_patch'].get('distance', float('nan'))),
                    # Source [z, y, x]; point['p'] is [x, y, z] from the pcl json.
                    'zyx': np.asarray(p['p'], dtype=np.float32)[::-1],
                })
    return anchors


def _tour_unwrap_adjustments(transform, dr, tour):
    """Cumulative theta=0 branch adjustment (winding units) at each tour point."""
    zyx = torch.as_tensor(
        np.stack([p['zyx'] for p in tour], axis=0).astype(np.float32),
        device=dr.device,
    )
    with torch.no_grad():
        spiral = transform(zyx)
        theta, _, _ = get_theta_and_radii(spiral[..., 1:], dr)
        zero_shifted = torch.zeros_like(theta)
        _, adjustments = unwrap_shifted_radii(theta[None], zero_shifted[None], dr)
        return np.round((adjustments[0] / dr).cpu().numpy()).astype(np.int64)


def build_rel_adjacency(cross_patch_pcls, patches, transform, dr, attachment_deltas=None):
    """patch_id -> list of relative-winding edges leaving that patch.

    For each relative-winding pcl we walk its chain-valid point sequence
    (pcl['chain'].iter_chain(): id-sorted order for ordinary pcls, an Euler tour
    of the member tree for merged fiber-link components -- whose id-sorted order is
    not chain-valid across members -- crossing each loop-closing non-tree link
    once too, so link cycles contribute their own patch edges and inconsistent
    holonomy around them is detectable) and connect each *consecutive* pair of
    attached points that lands on two different (loaded) patches. Each such pair
    yields a pair of directed edges, one per direction. An edge records the
    departure point (`from_ij`) on this patch, the arrival point (`to_ij`) on
    the neighbour, and the edge's winding delta = wind_a(arrival) -
    wind_a(departure), corrected for any theta=0 seam crossings accumulated
    along the chain between the two points (through unattached points and, for
    merged components, fiber-hopping junctions). Only consecutive pairs are
    used, so the edge set stays a chain through the pcl's points; more distant
    patches are reached transitively by BFS."""
    adjacency = {}
    for pid, pcl in cross_patch_pcls.items():
        kind = classify_pcl(pcl)
        if kind == 'absolute':
            continue

        def attached(p):
            on_patch = p.get('on_patch')
            return (on_patch is not None and on_patch['id'] in patches
                    and (attachment_deltas is None or id(p) in attachment_deltas)
                    and np.isfinite(float(p['winding_annotation'])))

        # Component points retain member-local IDs. The collection dictionary
        # keys, unlike point['id'], are unique across the merged component.
        # Every downstream edge cache (including divergence rejection) uses
        # these IDs, so using member-local IDs can apply another edge's verdict.
        point_ids = {id(p): int(key) for key, p in pcl['points'].items()}
        tour = list(pcl['chain'].iter_chain())
        tour_adjustments = _tour_unwrap_adjustments(transform, dr, tour)
        attached_tour = [(k, p) for k, p in enumerate(tour) if attached(p)]
        pairs = []
        seen_pairs = set()
        for (ka, pa), (kb, pb) in zip(attached_tour, attached_tour[1:]):
            key = frozenset([id(pa), id(pb)])
            if id(pa) == id(pb) or key in seen_pairs:
                continue  # an Euler tour revisits points; keep each edge once
            seen_pairs.add(key)
            pairs.append((pa, pb, int(tour_adjustments[kb] - tour_adjustments[ka])))
        for pa, pb, unwrap_adjustment in pairs:
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
            branch_delta = -unwrap_adjustment
            connector_delta = ((attachment_deltas or {}).get(id(pb), 0)
                               - (attachment_deltas or {}).get(id(pa), 0))
            dwind = raw_dwind + branch_delta + connector_delta
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
                'from_point_id': point_ids[id(pa)], 'to_point_id': point_ids[id(pb)],
                'from_source_point_id': int(pa['id']), 'to_source_point_id': int(pb['id']),
                'from_source_collection_id': pa.get('collectionId'),
                'to_source_collection_id': pb.get('collectionId'),
                'from_zyx': za, 'to_zyx': zb,
                'winding_delta': dwind,
                'raw_winding_delta': raw_dwind,
                'pcl_unwrap_adjustment': unwrap_adjustment,
                'pcl_branch_delta': branch_delta,
                'attachment_branch_delta': connector_delta,
            })
            adjacency.setdefault(idb, []).append({
                **common, 'neighbor': ida, 'from_ij': ijb, 'to_ij': ija,
                'from_point_id': point_ids[id(pb)], 'to_point_id': point_ids[id(pa)],
                'from_source_point_id': int(pb['id']), 'to_source_point_id': int(pa['id']),
                'from_source_collection_id': pb.get('collectionId'),
                'to_source_collection_id': pa.get('collectionId'),
                'from_zyx': zb, 'to_zyx': za,
                'winding_delta': -dwind,
                'raw_winding_delta': -raw_dwind,
                'pcl_unwrap_adjustment': -unwrap_adjustment,
                'pcl_branch_delta': -branch_delta,
                'attachment_branch_delta': -connector_delta,
            })
    return adjacency


def filter_attachment_returns(pcls, patches, transform, dr, measurer, attachment_deltas):
    """Keep the first sheet assignment for each (fiber component, patch).

    The full tree traversal carries branch transport through member junctions,
    including H->H continuations and reversed members. Euler backtracking does
    not constitute a new contact. Non-tree fiber links remain constraints for
    cycle detection; they must not redefine the first-visit branch frame.

    Compare every later contact with the first one, including transport along
    the patch: a seam crossing or a genuinely multi-turn patch is not itself a
    reason to reject. A nonzero *relative* turn is a wrong-sheet reattachment.
    Only analysis eligibility changes; source point collections are untouched.
    """
    allowed = dict(attachment_deltas)
    candidates = []
    for pid, pcl in pcls.items():
        if classify_pcl(pcl) != 'neither':
            continue
        chain = pcl['chain']
        if hasattr(chain, 'member_sorted'):
            from spiral_helpers import ComponentChain
            chain = ComponentChain(chain.member_sorted, chain.pos_of, chain.tree_parent)
        tour = list(chain.iter_chain())
        if not tour:
            continue
        adjustments = _tour_unwrap_adjustments(transform, dr, tour)
        point_ids = {id(p): int(key) for key, p in pcl['points'].items()}
        seen, first = set(), {}
        for index, point in enumerate(tour):
            identity = id(point)
            if identity in seen or identity not in allowed:
                continue
            seen.add(identity)
            patch_id = point['on_patch']['id']
            if patch_id not in patches:
                continue
            branch = -int(adjustments[index]) + attachment_deltas[identity]
            if patch_id not in first:
                first[patch_id] = point, branch
                continue
            anchor, anchor_branch = first[patch_id]
            candidates.append((pid, pcl, patch_id, anchor, point,
                               point_ids[id(anchor)], point_ids[identity],
                               branch - anchor_branch))
    rejected = []
    unmeasurable = 0
    # Bound sampled-path memory even for millions of attachment candidates.
    for start in range(0, len(candidates), 4096):
        batch = candidates[start:start + 4096]
        requests = [(patch, a['on_patch']['ij'], b['on_patch']['ij'])
                    for _, _, patch, a, b, _, _, _ in batch]
        measurer.prefetch(requests)
        for request, item in zip(requests, batch):
            pid, pcl, patch, a, b, ida, idb, fiber_delta = item
            strip = measurer.measure(*request)
            patch_delta = None if strip is None else strip['delta_windings']
            if patch_delta == fiber_delta:
                continue
            unmeasurable += strip is None
            allowed.pop(id(b), None)
            rejected.append({
                'pcl_id': pid, 'pcl_name': pcl.get('name'), 'patch_id': patch,
                'first_point_id': ida, 'rejected_point_id': idb,
                'first_source_point_id': int(a['id']), 'rejected_source_point_id': int(b['id']),
                'first_source_collection_id': a.get('collectionId'),
                'rejected_source_collection_id': b.get('collectionId'),
                'first_zyx': _to_py(a['zyx']), 'rejected_zyx': _to_py(b['zyx']),
                'fiber_delta': fiber_delta, 'patch_delta': patch_delta,
                'reason': 'unmeasurable_patch_path' if strip is None else 'different_turn',
            })
    return allowed, {'num_checked': len(candidates), 'num_rejected': len(rejected),
                     'num_unmeasurable': unmeasurable, 'rejected': rejected}


def audit_attachment_returns(pcls, patches, transform, dr, measurer, attachment_deltas):
    """Find local certificates: one fiber returns to a different turn of a patch.

    Check each member separately, without any inferred cross-fiber traversal or
    BFS tree. Consecutive visits to the same patch separated by other/unattached
    points must have the same branch transport along the fiber and the surface.
    A disagreement implicates the two attachments or intervening geometry; it
    does not by itself identify a sheet switch in either source.
    """
    requests, candidates = [], []
    for pid, pcl in pcls.items():
        if classify_pcl(pcl) == 'absolute':
            continue
        point_ids = {id(p): int(key) for key, p in pcl['points'].items()}
        chain = pcl['chain']
        members = (chain.member_sorted if hasattr(chain, 'member_sorted')
                   else [list(chain.iter_chain())])
        for member_index, points in enumerate(members):
            if not points:
                continue
            adjustments = _tour_unwrap_adjustments(transform, dr, points)
            previous = {}
            for index, point in enumerate(points):
                on = point.get('on_patch')
                if on is None or on['id'] not in patches:
                    continue
                if id(point) not in attachment_deltas:
                    continue
                patch_id = on['id']
                if patch_id in previous:
                    old_index, old = previous[patch_id]
                    if index > old_index + 1:
                        delta = (int(round(float(point['winding_annotation'])
                                           - float(old['winding_annotation'])))
                                 + int(adjustments[old_index] - adjustments[index])
                                 + attachment_deltas.get(id(point), 0)
                                 - attachment_deltas.get(id(old), 0))
                        request = (patch_id, old['on_patch']['ij'], on['ij'])
                        requests.append(request)
                        candidates.append((pid, pcl, member_index, old, point,
                                           point_ids[id(old)], point_ids[id(point)],
                                           delta, index - old_index))
                previous[patch_id] = index, point
    measurer.prefetch(requests)
    conflicts = []
    unmeasurable = 0
    for request, candidate in zip(requests, candidates):
        strip = measurer.measure(*request)
        if strip is None:
            unmeasurable += 1
            continue
        pid, pcl, member, a, b, ida, idb, delta, gap = candidate
        if delta == strip['delta_windings']:
            continue
        source = (pcl.get('link_members') or [{}])[member if pcl.get('link_members') else 0]
        conflicts.append({
            'pcl_id': pid, 'pcl_name': pcl.get('name'), 'member_index': member,
            'source_file': source.get('source_file', pcl.get('source_file')),
            'source_name': source.get('name', pcl.get('name')),
            'patch_id': request[0], 'from_point_id': ida, 'to_point_id': idb,
            'from_source_point_id': int(a['id']), 'to_source_point_id': int(b['id']),
            'source_collection_id': a.get('collectionId'),
            'from_ij': _to_py(a['on_patch']['ij']), 'to_ij': _to_py(b['on_patch']['ij']),
            'from_zyx': _to_py(a['zyx']), 'to_zyx': _to_py(b['zyx']),
            'from_distance': float(a['on_patch']['distance']),
            'to_distance': float(b['on_patch']['distance']),
            'fiber_delta': delta, 'patch_delta': strip['delta_windings'],
            'winding_residual': delta - strip['delta_windings'],
            'fiber_steps': gap,
        })
    return {'num_checked': len(requests), 'num_unmeasurable': unmeasurable,
            'num_conflicts': len(conflicts), 'conflicts': conflicts}


def edge_provenance(edge, reverse=False):
    """Preserve source IDs when graph IDs belong to a merged collection."""
    result = {'attachment_branch_delta': (-1 if reverse else 1)
              * edge.get('attachment_branch_delta', 0)}
    for side, source in (('from', 'to' if reverse else 'from'),
                         ('to', 'from' if reverse else 'to')):
        for suffix in ('source_point_id', 'source_collection_id'):
            if source + '_' + suffix in edge:
                result[side + '_' + suffix] = edge[source + '_' + suffix]
    return result


def build_loop_cycles(loops, tree_edge_by_child, strip_delta_fn, prefetch_fn=None):
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
    to the BFS that produced the tree.

    `prefetch_fn(requests)`, when given, is handed every ``(patch, top_ij, bot_ij)``
    the cycles need before any `strip_delta_fn` call, so a batched measurer can
    measure them all at once."""
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
            return {**edge_provenance(hop), 'from_patch': a, 'to_patch': b,
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
        return {**edge_provenance(hop, reverse=True), 'from_patch': a, 'to_patch': b,
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
            **edge_provenance(L, reverse=True),
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
        strip_requests = []
        for i in range(m):
            top_ij = closing_step['to_ij'] if i == 0 else steps[i - 1]['to_ij']
            bot_ij = closing_step['from_ij'] if i == m - 1 else steps[i]['from_ij']
            strip_requests.append((patches[i], top_ij, bot_ij))
        cycles.append({
            'closing_rel_pcl_id': L['rel_pcl_id'],
            'closing_rel_pcl_name': L['rel_pcl_name'],
            'loop_winding_delta': L['loop_winding_delta'],
            'loop_winding_residual': L['loop_winding_residual'],
            'from_patch': P, 'to_patch': R, 'lca': lca,
            'patches': patches,
            'steps': steps,
            'patch_strip_deltas': strip_requests,  # measured below
            'closing_step': closing_step,
        })
    if prefetch_fn is not None:
        prefetch_fn([request for cycle in cycles for request in cycle['patch_strip_deltas']])
    for cycle in cycles:
        cycle['patch_strip_deltas'] = [strip_delta_fn(*request) for request in cycle['patch_strip_deltas']]
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


def load_cli_config(ctx, param, value):
    """Supply JSON values as Click defaults, preserving CLI validation/precedence."""
    if value is None or ctx.resilient_parsing:
        return
    try:
        with open(value, encoding='utf-8') as config_file:
            config = json.load(config_file)
    except (OSError, ValueError) as exc:
        raise click.BadParameter(f'cannot read JSON configuration: {exc}', ctx=ctx, param=param) from exc
    if not isinstance(config, dict):
        raise click.BadParameter('must contain a JSON object', ctx=ctx, param=param)

    names = {}
    for option in ctx.command.params:
        if option.name == param.name:
            continue
        names[option.name] = option.name
        for flag in option.opts:
            names[flag.lstrip('-').replace('-', '_')] = option.name

    defaults = {}
    for key, setting in config.items():
        if key in ('linking', 'edge_divergence'):
            try:
                if key == 'linking':
                    ctx.meta['analysis_linking'] = AnalysisLinkingConfig.from_json(setting)
                else:
                    ctx.meta['edge_divergence'] = EdgeDivergenceConfig.from_json(setting)
            except ValueError as exc:
                raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc
            continue
        name = names.get(key.replace('-', '_'))
        if name is None:
            raise click.BadParameter(f'unknown option {key!r}', ctx=ctx, param=param)
        if name in defaults:
            raise click.BadParameter(f'duplicate option {key!r}', ctx=ctx, param=param)
        defaults[name] = setting
    ctx.default_map = {**(ctx.default_map or {}), **defaults}


@click.command()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              is_eager=True, expose_value=False, callback=load_cli_config,
              help='JSON object with option values (hyphen or underscore keys). '
                   'CLI options override JSON; use arrays for pcl and booleans for toggles.')
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to a checkpoint_*.ckpt saved by fit_spiral.')
@click.option('--patches-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory of patch tifxyz folders (the full set, for attachment).')
@click.option('--umbilicus', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Path to umbilicus.json. Defaults to umbilicus.json next to --patches-dir.')
@click.option('--fibers', 'fibers_flag', is_flag=False, flag_value='', default=None,
              help='Include fibers as same-winding (delta-0) edges, like same_windings.json. '
                   'Pass a directory, or use the bare flag to infer the dataset\'s fibers/ dir '
                   'next to --patches-dir. Fibers add same-winding constraints (participating in '
                   'loop/holonomy detection) but contribute no winding votes.')
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
def main(checkpoint, patches_dir, umbilicus, fibers_flag, patch_id, pcl_paths, z_range, step_size, medial_weight,
         max_hops, detect_loops, min_fix, min_fix_time_limit, max_subgraphs, top_patches_per_subgraph, output,
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

    fibers_path = resolve_fibers_path(patches_dir, fibers_flag)
    if fibers_path:
        print(f'including fibers from {fibers_path} (same-winding edges)')

    print(f'loading checkpoint {checkpoint}')
    from checkpoint_io import load_checkpoint_cpu
    ckpt = load_checkpoint_cpu(checkpoint)
    fit_config, scroll, input_paths, model_z_begin, model_z_end = build_fit_inputs(
        ckpt, patches_dir, pcl_paths, filter_z_begin, filter_z_end, umbilicus_path, fibers_path
    )
    print(f'checkpoint (model) z-range: [{model_z_begin}, {model_z_end}); '
          f'filtering z-range: [{filter_z_begin}, {filter_z_end})')
    if not (filter_z_begin >= model_z_begin and filter_z_end <= model_z_end):
        raise SystemExit(
            f'filtering z-range [{filter_z_begin}, {filter_z_end}) extends beyond the checkpoint '
            f'z-range [{model_z_begin}, {model_z_end}); the transform is undefined outside its domain.'
        )

    context = fs.FitContext(fit_config, scroll=scroll, paths=input_paths)
    linking_config = click.get_current_context().meta.get('analysis_linking')
    point_linker = None
    transform = dr = None
    if linking_config is not None:
        try:
            linking_config = linking_config.resolve_distance(fit_config['pcl_link_distance_tolerance'])
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint='linking.distance') from exc
        print('rebuilding spiral transform before linking')
        transform, dr = build_transform(ckpt, fit_config, context, model_z_begin, model_z_end)
        from point_collection import PatchLinkOptions
        point_linker = AnalysisPointLinker(
            linking_config,
            model_inward_direction(transform, dr, device, model_z_begin, model_z_end),
            min_z_fraction=fit_config['pcl_vertical_fiber_min_z_fraction'],
            min_auto_certainty=fit_config['pcl_vertical_fiber_min_auto_certainty'],
            annotation_options=PatchLinkOptions(
                window_points=fit_config['pcl_link_window_points'],
                window_min_points=fit_config['pcl_link_window_min_points']))
    print('loading + z-filtering patches and point-collections')
    # Analysis reads only the loaded patches and the linked, classified point
    # collections; the fit-only caches (sampling caches, device atlases,
    # trusted-geometry index, DT targets) are skipped.
    if point_linker is None:
        context.load_host_inputs(training_caches=False)
    else:
        context.load_host_inputs(point_linker=point_linker, fiber_min_point_spacing=0.0,
                                 training_caches=False)
        print(f'analysis linking: {json.dumps(point_linker.report)}')
    patches = context.verified_patches
    # fit_spiral holds the cross-patch pcls as a list; this tool's
    # vote/edge builders key their reports by pcl id. Per-file collection
    # ids collide across json files, so key by list position instead.
    # Cross-fiber links arrive here as merged 'fibercomp:*' component pcls
    # (load_host_inputs folds link-connected fibers together); build_rel_adjacency
    # walks each component's fiber graph to emit delta-0 edges that hop fibers
    # at the junctions -- same seam machinery as within-fiber / relative edges.
    cross_patch_pcls = dict(enumerate(context.cross_patch_pcls))
    if patch_id not in patches:
        raise SystemExit(
            f'patch id {patch_id!r} not among {len(patches)} loaded patches (after z-filtering); '
            'check --patch-id / --z-range'
        )
    patch = patches[patch_id]

    if transform is None:
        print('rebuilding spiral transform from checkpoint')
        transform, dr = build_transform(ckpt, fit_config, context, model_z_begin, model_z_end)
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
    attachment_deltas = attachment_branch_deltas(cross_patch_pcls, patches, transform, dr)
    measurer = StripMeasurer(transform, dr, patches, medial_weight, step_size)
    print('rejecting different-turn reattachments, including linked fiber continuations')
    attachment_deltas, attachment_return_filter = filter_attachment_returns(
        cross_patch_pcls, patches, transform, dr, measurer, attachment_deltas)
    print(f'attachment return filter: {attachment_return_filter["num_rejected"]} rejected '
          f'of {attachment_return_filter["num_checked"]} later contacts '
          f'({attachment_return_filter["num_unmeasurable"]} unmeasurable)')
    abs_anchors_by_patch = build_abs_anchors_by_patch(cross_patch_pcls, patches, attachment_deltas)
    rel_adjacency = build_rel_adjacency(cross_patch_pcls, patches, transform, dr, attachment_deltas)
    divergence_config = click.get_current_context().meta.get('edge_divergence')
    divergence_report = None
    if divergence_config is not None:
        # Same-winding fiber edges whose two patches diverge away from the hop
        # (a neighbouring winding touching, or a patch crossing another) are
        # dropped before the graph walk; see edge_divergence.py.
        rel_adjacency, divergence_report = filter_divergent_edges(
            rel_adjacency, patches, umbilicus_inward_direction(context.umbilicus_z_to_yx()),
            divergence_config, )
        print(f'edge divergence filter: {json.dumps(divergence_report["counts"])}')
    # Per-pcl classification (absolute / relative / neither) so it is clear what each
    # pcl is doing: absolute pcls anchor windings, relative pcls supply long-range edges
    # with real winding deltas, and 'neither' pcls (fibers / new_same_wind) supply
    # same-winding (delta-0) edges only.
    pcl_summary = []
    for pid, pcl in sorted(cross_patch_pcls.items()):
        num_attached = sum(1 for p in pcl['points'].values() if 'on_patch' in p)
        eligible = [p for p in pcl['points'].values() if id(p) in attachment_deltas]
        pcl_summary.append({
            'id': int(pid),
            'name': pcl.get('name'),
            'kind': classify_pcl(pcl),
            'winding_is_absolute': bool(pcl.get('metadata', {}).get('winding_is_absolute', False)),
            'has_winding_annotations': bool(pcl.get('has_winding_annotations', False)),
            'num_points': len(pcl['points']),
            'num_input_attached_points': num_attached,
            'num_attached_points': len(eligible),
            'num_attached_patches': len({p['on_patch']['id'] for p in eligible}),
            'members': pcl.get('link_members', []),
        })
    num_abs_pcls = sum(1 for s in pcl_summary if s['kind'] == 'absolute')
    num_rel_pcls = sum(1 for s in pcl_summary if s['kind'] == 'relative')
    num_neither_pcls = sum(1 for s in pcl_summary if s['kind'] == 'neither')
    num_rel_edges = sum(len(v) for v in rel_adjacency.values()) // 2
    print(f'{num_abs_pcls} absolute, {num_rel_pcls} relative, {num_neither_pcls} neither '
          f'(fiber/same-wind) pcl(s); absolute anchor {len(abs_anchors_by_patch)} patch(es); '
          f'relative+neither -> {num_rel_edges} cross-patch edge(s) over {len(rel_adjacency)} patch(es)')

    # Batched, memoised strip measurements (see StripMeasurer): the BFS below
    # prefetches every strip a level can ask for, so the transform runs once per
    # level instead of once per strip. It also owns the per-patch valid-quad
    # graphs (built once the first time a strip is needed on that patch).
    def strip_delta(pid, from_ij, to_ij):
        return measurer.measure(pid, from_ij, to_ij)

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
    # The queue is consumed one BFS level (hop depth) at a time: the states of a
    # level are known before any of them is expanded, so every strip the level
    # can request is measured in one batch first. Expansion order within the
    # level -- and therefore which edge first reaches each patch -- is the plain
    # FIFO order the per-state queue produced.
    frontier = [{'patch': patch_id, 'entry_ij': seed_ij, 'acc': 0, 'hops': 0, 'path': []}]

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
            'abs_winding_annotation': anchor['source_winding'],
            'abs_attachment_branch_delta': anchor['attachment_branch_delta'],
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
        return (f'  [{hops} hop(s)] abs pcl {anchor["pcl_id"]} ({anchor["pcl_name"]!r}) '
                f'point {anchor["point_id"]} on patch {voter["abs_patch_id"]}: '
                f'annotation={anchor["winding"]:g} -> expected seed winding={expected:.3f} '
                f'(round {expected_int}){chain}')

    def check_loop(P, q, acc, hops, edge, rstate):
        """Non-tree edge P -> R (R already reached): it closes a cycle. Express
        winding(seed) - winding(e) at the edge's arrival point e two ways and
        difference them:
          * via this edge from P: acc - strip(P, q->d) - edge_delta
          * via R's stored tree path: acc_R - strip(R, entry_R->e)
        The gap is the cycle's winding holonomy (0 == the loop closes). Returns
        the console line to print (or None)."""
        nonlocal num_loops_unmeasurable
        R = edge['neighbor']
        edge_key = (edge['pcl_id'], frozenset((edge['from_point_id'], edge['to_point_id'])))
        if edge_key in loops_seen:
            return None
        strip = strip_delta(P, q, edge['from_ij'])
        rstrip = strip_delta(R, rstate['entry_ij'], edge['to_ij'])
        if strip is None or rstrip is None:
            num_loops_unmeasurable += 1
            return None  # can't measure one side; the reverse direction may still succeed
        acc_arrival_via_edge = acc - strip['delta_windings'] - edge['winding_delta']
        acc_arrival_via_tree = rstate['acc'] - rstrip['delta_windings']
        holonomy = acc_arrival_via_edge - acc_arrival_via_tree
        holonomy_int = int(round(holonomy))
        loops_seen.add(edge_key)
        loops.append({
            **edge_provenance(edge),
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
            return (f'  LOOP inconsistency: rel pcl {edge["pcl_id"]} ({edge["pcl_name"]!r}) '
                    f'reconnects {P} (d{hops}) <-> {R} (d{rstate["hops"]}): winding holonomy '
                    f'{holonomy_int:+d} (residual {holonomy:+.3f})')
        return None

    while frontier:
        # Every strip this level can ask for while voting and growing the tree:
        # anchor -> entry on each state's patch, and entry -> departure point of
        # each of its edges.
        requests = []
        for state in frontier:
            P, q = state['patch'], state['entry_ij']
            for anchor in abs_anchors_by_patch.get(P, []):
                requests.append((P, anchor['ij'], q))
            for edge in rel_adjacency.get(P, []):
                requests.append((P, q, edge['from_ij']))
        measurer.prefetch(requests)

        next_frontier = []
        # Console lines in encounter order; loop checks are filled in below.
        messages = []
        pending_loops = []  # (message slot, P, q, acc, hops, edge, rstate)
        for state in frontier:
            P, q, acc, hops, path = state['patch'], state['entry_ij'], state['acc'], state['hops'], state['path']

            # (a) votes from absolute anchors on this patch: propagate from anchor to
            #     the entry point (strip_delta(P, anchor, q) = winding(q) - winding(anchor)).
            for anchor in abs_anchors_by_patch.get(P, []):
                anchor_strip = strip_delta(P, anchor['ij'], q)
                if anchor_strip is None:
                    messages.append(
                        f'  [{hops} hop(s)] abs pcl {anchor["pcl_id"]} ({anchor["pcl_name"]!r}) '
                        f'point {anchor["point_id"]} on patch {P}: anchor->entry strip had <2 valid '
                        f'points; skipped')
                    continue
                expected = acc + anchor['winding'] + anchor_strip['delta_windings']
                messages.append(record_vote(expected, anchor, hops, acc, anchor_strip, path))

            # (b) relative edges: grow the BFS tree into unreached neighbours, and check
            #     edges that reconnect an already-reached patch for loop holonomy.
            for edge in rel_adjacency.get(P, []):
                R = edge['neighbor']
                edge_key = (edge['pcl_id'], frozenset((edge['from_point_id'], edge['to_point_id'])))

                if R in reached:
                    # Non-tree edge -> a loop check. Its arrival-side strip depends on
                    # R's entry point, which may only have been fixed earlier in this
                    # level, so the checks are batched after the level's tree growth.
                    # Loop checks never change `reached` or the tree, and a check's
                    # outcome depends only on its two strips, so deferring them (in
                    # encounter order) leaves every result unchanged.
                    if (not detect_loops) or edge_key in tree_edges:
                        continue
                    messages.append(None)
                    pending_loops.append((len(messages) - 1, P, q, acc, hops, edge, reached[R]))
                    continue

                # Unreached neighbour: grow the tree into it (bounded by --max-hops).
                if max_hops is not None and hops >= max_hops:
                    continue
                # Propagate within P from the entry point to the edge's departure point.
                strip = strip_delta(P, q, edge['from_ij'])
                if strip is None:
                    continue  # can't carry the winding across this edge; another path may reach R
                acc2 = acc - strip['delta_windings'] - edge['winding_delta']
                tree_edges.add(edge_key)
                reached[R] = {'acc': acc2, 'entry_ij': edge['to_ij'], 'hops': hops + 1}
                hop_record = {
                    **edge_provenance(edge),
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
                next_frontier.append({'patch': R, 'entry_ij': edge['to_ij'], 'acc': acc2,
                                      'hops': hops + 1, 'path': path + [hop_record]})

        # Loop checks of this level: the arrival-side strips are known now.
        measurer.prefetch([(edge['neighbor'], rstate['entry_ij'], edge['to_ij'])
                           for _, _, _, _, _, edge, rstate in pending_loops])
        for slot, P, q, acc, hops, edge, rstate in pending_loops:
            messages[slot] = check_loop(P, q, acc, hops, edge, rstate)
        for message in messages:
            if message is not None:
                print(message)
        frontier = next_frontier

    print(f'measured {measurer.num_measured} within-patch strip(s) in {measurer.num_batches} batch(es) '
          f'over {len(measurer._graphs)} patch graph(s)')

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
        s = strip_delta(pid, np.asarray(top_ij, dtype=np.float32), np.asarray(bot_ij, dtype=np.float32))
        return None if s is None else s['delta_windings']

    loop_cycles = build_loop_cycles(loops, tree_edge_by_child, loop_patch_strip_delta,
                                    prefetch_fn=measurer.prefetch)
    attachment_returns = None
    if detect_loops:
        print('checking individual fiber returns to the same patch')
        attachment_returns = audit_attachment_returns(
            cross_patch_pcls, patches, transform, dr, measurer, attachment_deltas)
        print(f'attachment returns: {attachment_returns["num_conflicts"]} conflict(s) '
              f'in {attachment_returns["num_checked"]} checks; '
              f'{attachment_returns["num_unmeasurable"]} unmeasurable')

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
        'graph_point_id_convention': 'collection_dictionary_key',
        'attachment_returns': attachment_returns,
        'attachment_return_filter': attachment_return_filter,
        'num_attachment_seam_crossings': sum(v != 0 for v in attachment_deltas.values()),
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

    if point_linker is not None:
        out['linking'] = point_linker.report
    if divergence_report is not None:
        out['edge_divergence'] = divergence_report

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
