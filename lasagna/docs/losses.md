\# Losses: adding a new term (checklist)

## Rule: FitResult-only

- Loss code must only consume [`model.FitResult`](../model.py:23).
- Do not re-sample the image or recompute model grids inside the loss.

## 1) Implement the loss

- Create a new module like [`opt_loss_gradmag.py`](../opt_loss_gradmag.py:1) (or add to an existing `opt_loss_*.py`).
- Prefer a `*_loss_map()` function returning `(lm, mask)` and a scalar wrapper using masked mean.

Conventions:

- `lm` is a per-sample loss map (usually `(N,1,H,W)` or `(N,1,H,P)`), float.
- `mask` must match `lm` spatial shape and be derived via `min(...)` of all participating sample masks (never interpolate).

## 2) Wire it into optimization

- Add import in [`optimizer.py`](../optimizer.py:1).
- Add a new key to `lambda_global` in [`optimizer.load_stages()`](../optimizer.py:61).
- Add the term to the `terms` dict in [`optimizer.optimize()`](../optimizer.py:110), mapping name → loss function.

The term name string must match the stage config key.

## 3) Add a default weight in the stages JSON

- Add the term to the `base` map in [`stages_scalespace.json`](../stages_scalespace.json:1).

Notes:

- Some terms may need stage-initialization context computed once (e.g. `mean_pos` captures the current mean position at stage start and penalizes drift).

Optimizer stages can also optionally perform mesh growth + local optimization (see [`docs/model.md`](model.md:1)).

The optimizer status table prints on the first step, final step, and every 100
iterations by default. Override per stage with `global_opt.args.status_interval`
(`0` keeps only first/final prints):

```json
{
  "name": "snap",
  "steps": 1000,
  "args": {
    "status_interval": 25
  }
}
```

### Optional `pred_dt` dense-flow gate

`pred_dt` snapping can be gated per stage via `global_opt.args`:

```json
{
  "name": "snap",
  "steps": 1000,
  "params": ["mesh_ms"],
  "args": {
    "pred_dt_normal_source": "model",
    "pred_dt_flow_gate": {
      "enabled": true,
      "gate_factor": 1.0,
      "backtrack_distance": 10.0,
      "local_boost": 1.0,
      "corr_seed_surface_distance": 6.0,
      "debug_vis_interval": 50,
      "anticipatory_pull": {
        "enabled": true,
        "samples": 8,
        "search_steps": 21,
        "search_angle_degrees": 60.0,
        "inlier_zero": 80.0,
        "inlier_one": 120.0,
        "loss_weight": 1.0,
        "debug_points": [[40, 50], [40, 51]],
        "debug_roi_center_xyz": [15733, 14023, 51588],
        "debug_roi_k": 8,
        "debug_roi_root_min": 0.5,
        "debug_roi_tip_max": 0.5,
        "debug_slice_upsample": 8
      },
      "debug": true
    }
  }
}
```

`pred_dt_normal_source` controls the normal used for the pred-dt sampling
projection and anticipatory pull search. The default is `"model"` for current
mesh normals. Set it to `"gt"` to use sampled GT normals for that stage.

When enabled, the current single-winding `pred_dt` render is median-filtered
with radius 1, thresholded at `110`, routed through `dense_batch_min_cut`, and
sampled at the exact model grid corners. The C++ flow code computes both the
normalized greedy-ascent flow and the local gate. The greedy-ascent flow is
first normalized per source region, then the local gate is that normalized
greedy-ascent flow divided by the dilated local maximum of the same normalized
image. `backtrack_distance` is the dilation radius. `local_boost` blends
between them: `0.0` uses only the per-region normalized greedy-ascent flow,
`1.0` uses the local gate, and values between are linear.
The resulting gate multiplies the `pred_dt` loss map. When multiple accepted
flow sources are present, the C++ flow code normalizes the gate separately per
source region. Disconnected threshold-domain components naturally get separate
regions; connected components are partitioned on the skeleton graph with a
widest-path source assignment, and graph edges where regions meet are split at
their lowest-capacity pixel for normalization only. Sources connected by a
graph bottleneck at least `0.75` of the larger source value share one
normalization region rather than being split.
Correction points are also passed to the flow code as additional source seeds
when `corr_seed_enabled` is not false and the point is within
`corr_seed_surface_distance` full-resolution voxels of the current rendered
surface. The C++ flow graph applies the same source-edge detection and uphill
start traversal to these correction seeds as it does to the primary fit seed.
`gate_factor` controls how much of the regular pred-dt weight comes from the gate:
`weight = gate_factor * gate + (1 - gate_factor)`, so `0.99` keeps a `0.01`
baseline pred-dt loss weight active everywhere. The anticipatory pull uses the
raw gate for activation and is scaled only by `gate_factor`; the baseline term
does not activate anticipatory pulls.
The loss denominator remains the original validity-mask sum; the gate is
intentionally not renormalized.

`backtrack_distance` is measured in the rendered `pred_dt` image pixel units.
It is passed through to the dense grid flow routing and local gate dilation.
`backtrack_distance` and `local_boost` match the C++ debug CLI options:

```bash
./dense_batch_preprocess -i pred.tif --source 240,240 --grid-step 4 --backtrack-distance 10 --local-boost 1.0
```

`anticipatory_pull` is optional and only runs with active flow gating. It scores
all one-step LR neighbor lines before flow weights are known, using subsampled
`pred_dt` values along each line. Each candidate keeps the root fixed and
brute-force searches a tip push/pull along the GT normal sampled at the current
tip position. The default range uses 21 offsets equivalent to `-60` to `+60`
degrees on a flat mesh with the canonical `mesh_step` as reference length.
After flow returns, each candidate whose root gate is higher/nonzero and whose
tip gate is below 1 contributes an independent straight pull to the tip corner,
weighted by root gate and prefix inlier score. The pull is not winner-take-all;
multiple neighbor lines may contribute to the same tip.

The optimizer status table reports the flow-gate strength as fractions and
corner counts for gate weights `>0`, `>0.1`, and `>0.5`, plus the fraction at
`1.0`.

With `debug: true`, flow layer TIFFs are written every `debug_layer_interval`
flow evaluations (default `10`). The service JPG is written every
`debug_vis_interval` evaluations (default `50`; `debug_jpg_interval` remains a
legacy alias) as
`pred_dt_flow_gate_weight_jpg/vis_<iteration>.jpg` and shows the thresholded
flow basis, the retained seed/correction source component mask when available,
`pred_dt`, the dense local-max-ratio gate weight actually used by flow gating,
then the per-region normalized greedy-ascent flow value before the local max
ratio. The
rightmost panel overlays the primary seed in cyan plus correction
points at their nearest rendered-surface pixel; green correction points are
within `corr_seed_surface_distance` and become extra flow seeds, while orange
points are too far and are labeled with their current full-resolution voxel
distance to the rendered mesh. Magenta pixels show the source graph edges that
C++ actually accepted after graph edge detection and uphill source traversal,
and the panel header reports accepted source count, source edge count, and
seeded node count. The layer TIFF also includes `source_components`, a black
and white mask of the final threshold-domain connected component(s) retained
because they contain the primary seed or a valid correction-point source, plus
`graph_source_edges` and
`island_obstacle_factor`, which labels enclosed obstacle islands and annotates
the local loop score without coloring the associated loop pixels. The score is
`min(full_island_dt / representative_point_dt)` over loop pixels associated with
the island, so lower values mean the island is acting more like a real obstacle.
The debug log prints the raw score ingredients for each island: area, bbox,
sample counts, representative point, worst loop point, raw DT values, and
min/mean/max sampled ratios.
As a stop-gap, islands with score greater than `0.5` are filled into the white
domain and the distance transform, rim graph, and dense flow are rebuilt from
that filtered domain before producing the gate weights. The debug layer TIFF
includes `island_removed_mask` to show the filled islands.
The same debug pass also emits experimental island-flow propagation layers:
`island_flow_passability` shows graph edges that can pass flow through compact
island associations, `island_propagated_edge_flow` shows max-product propagated
edge flow over those links, and `island_bonus_edge_flow` shows the extra flow
above the raw graph edge flow. `island_tree_dense_flow_no_backtrack` and
`island_tree_dense_flow_greedy_ascent` then run the same dense no-backtrack and
greedy-ascent visualization path from the propagated edge flow. These layers are
diagnostic only and do not change the gate weights.

When `anticipatory_pull.debug_points` or `debug_roi_center_xyz` is set, every normal flow-gate layer-debug
iteration also writes `pred_dt_flow_gate_<stage>_anticipatory_fit_points.jpg`.
Explicit `debug_points` are LR mesh tip coordinates `(h,w)`. The ROI selector
uses the current root corner position and selects the `debug_roi_k` closest
individual root->tip snap candidates to `debug_roi_center_xyz`, restricted to
directions where the root flow gate is above `debug_roi_root_min` and the tip
flow gate is below `debug_roi_tip_max`. Each tile shows a slice around one
root-tip line, with `pred_dt` scaled from `inlier_zero` to `inlier_one`, the
fitted straight line, per-sample inlier scores, and the complete prefix score.
Tiles are upsampled by `debug_slice_upsample` and packed into an approximately
2:1 mosaic.

## 4) Add visualization output (loss map)

- Import the new loss module in [`vis.py`](../vis.py:1).
- Compute the loss map once in [`vis.save()`](../vis.py:154) (next to the other maps).
- Add an entry to `loss_maps` so a `res_loss_<suffix>_<postfix>.tif` gets written.

## 5) Sanity check

- Run a syntax check:
	- `python -m py_compile <changed_files>`
