# H/V surface-density diagnostics from saved100 solutions

Use both CCW/CW saved100 states; no global optimization or inference. Primary
metric exact fiber arc length per million base-voxel squared inferred UV area,
separate H/V surfaces. Secondary intersecting piece count per same area; count
once per piece/bin, profiles independently so V pieces are not counted per heat
cell. Note source splitting biases counts and UV area is not measured3D area.

Three paired versions: enabled only (Pbroken<=.5); all pieces with most-probable
non-broken bucket; all with disabled-only conditional placement recomputed using
fixed enabled MAP neighbors. Reuse canonical solver coordinates, sign/distance/
metric costs, calibrated sign charge and continuation costs. Test all100 buckets
only for disabled pieces against enabled neighbors (O(events*K), notK²). No
disabled-to-disabled inference; no geometry/refit or enabling changes. No enabled
link => explicitly flagged fallback to saved placement. Report confidence gaps,
changed placement counts and constraint support, residual conflicts. Save choices.

Shared bounds and identical area denominators across variants/families. Heatmap
3000vx and U profiles10000vx integrated over the common full V span; clip edge
cell area to common bounding rectangle. Empty cells remain zero, no support
threshold pretending density is absent. Bounds computed across actual placements.
H and V common heatmap color scale across variants; display clipping explicitly
99.5percentile only, numeric exports unmodified. Include length/count profiles,
heatmaps per variant and added-disabled density; distinguish absent observation
from true anatomical absence. Density peaks alone are not kollesis labels.

Implementation reuse Maps, segment_cells, coordinate/cost helpers and save_figure.
CPU batched PyTorch for fixed-neighbor candidate costs; no GPU grant. Independent
plan/code review. Tests length conservation, known uniform density/area, edge
bins, H/V counts, unchanged enabled positions, supported placement switching,
unsupported fallback, costs vs canonical small K² oracle, plot smoke. Run both
with approved benchmark command and inspect figures. Persist hashes/timings.

Spec update: read-only diagnostic, no changed optimized solution or surface mesh.

Review refinement: snap common rectangle to whole heat cells so exact-boundary
fixed-coordinate fibers do not fall into zero/tiny-area cells. Clip only profile
edge widths to this rectangle. Tied costs preserve saved placement; no unary
priors in this explicitly constraints-only diagnostic. Tests compare canonical
costs with prior_weight=0. Include imported helper/config hashes in provenance.
Docs updates: archive prior task log; task/plan/status, results/metrics, README and
changelog; record limitations including earlier-stage rejected fibers absent.
