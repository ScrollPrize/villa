# Other-winding backup barriers

Prior render work is committed at 75bb38079 and documented in changelog.
Implement weak signed-normal barriers from actual H/V fiber points at the same
phase and V on every other represented winding. Own-winding fitting remains
unchanged. Suppress barriers near existing same-winding attachments.
Independent plan/code review passed. Implemented geometry-only signed barriers,
per-point cumulative-width phase mapping, frozen target-winding support gates,
raw V positions, serialization and restoration. Invalid UV/normals/settings fail.
Small explicit experimental assumptions: 5vx clearance; support gate 0 below100
UVvx and full above300; phase is from initial actual U, not source piece ID.
These are not calibrated anatomical distances. Fixed mapping/gating is approximate
after UV motion and cannot guarantee nonintersection between fiber samples.

Sandbox write was interrupted by full root filesystem; retry succeeded after
space became available. No data removed. Runs used the existing approved runner,
nonexclusive GPU0/8GiB/8CPU manager grants; GPU0-5 restriction preserved.

## Matched results
Configs fiber_surface_barrier_control.json / fiber_surface_barrier_weak.json.
Both800steps, fresh ruled init, geom tangent .1, UVsmooth100, anchor.1, same
1136pieces/1134sources,71733samples,256x823 mesh.646694barrier pairs,503598 with
gate>0;65.4856% normalized weight survives gating. Weight0control versus.01weak.

|Metric|Control|Weak backup|
|---|---:|---:|
|All wrong-side pairs, arc weighted %|21.4608|8.5352|
|Gate>0 wrong-side pairs, arc weighted %|25.6196|9.1102|
|All clearance violations %|21.8981|9.0855|
|Gate>0 clearance violations %|26.0113|9.6440|
|Gate>0 penetration RMS base vx|1668.0088|576.7491|
|H positional RMS base vx|8.2809|8.4361|
|V offset positional RMS base vx|45.7819|45.8636|
|H normal RMS base vx|1.7419|1.7412|
|V offset normal RMS base vx|6.7627|6.3143|
|Opposing-triangle quads, full surface|353|621|
|Opposing-triangle quads, inspection crop|336|371|
|Solve seconds|11.3281|11.9878|
|Total seconds|16.9717|17.6126|

Diagnostic 'active' weights use the gate>0 subset, not gate magnitude. New
'effective' diagnostics also support gate weighting when restored/audited; saved
run summaries predate that diagnostic-only addition. Timings are single shared
GPU trials, not isolated speed measurements. Crossing lock diagnostic max.0031vx
is float32 subtraction rounding; graph uses shared component parameters.

Verdict: major sampled ordering improvement, nearly unchanged attachment fit,
but more local folds. Keep opt-in; not promoted as overall best/default.
Weak coefficient does not bound the absolute loss for large penetrations.

## Rendering and folded maps
Both render_connected_hv_barrier_{control,weak}_v1 contain matching7351x3188
CT JPG, fiber/composite LZW TIFF, inspection.tifxyz.100%CT coverage. Render seconds
control8.89,weak20.63 (different CT cache warmth, not a speed comparison).
User requested folded maps: existing geometry_masks reused, red CT overlay,
binary pixel mask and quad-grid mask, folds.json local/global quad row/column IDs.
Inspection crop columns100:689; full counts therefore differ from crop counts.
folds_only refresh validates original inputs and reuses CT without GPU/fetch.
The diagnostic detects opposing triangle normals, not all possible foldovers or
global intersections. Masks include only valid surface quads and covered CT pixels.

Validation:38 CPU regression tests passed (2.90s), including signs, unequal-width
mapping, support gates, gradient routing, raw V semantics, restore, crop alignment,
last-pixel ownership and invalid vertex masking. No upstream solve changes.
