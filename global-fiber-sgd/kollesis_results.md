# Kollesis diagnostic results — 2026-10-09

Completed on saved20-sweep CCW and CW solutions. No endpoint density, optimizer
changes, overlap fitting or new inference. Definitions and exact commands:
[kollesis_metrics.md](kollesis_metrics.md).

## View

| Artifact | CCW | CW |
|---|---|---|
| Full-height profiles | [SVG](data/kollesis20_v1/ccw/profiles.svg) / [LZW TIFF](data/kollesis20_v1/ccw/profiles.tif) | [SVG](data/kollesis20_v1/cw/profiles.svg) / [LZW TIFF](data/kollesis20_v1/cw/profiles.tif) |
| Close-pair density U/V | [SVG](data/kollesis20_v1/ccw/close_encounter_rate_heatmap.svg) | [SVG](data/kollesis20_v1/cw/close_encounter_rate_heatmap.svg) |
| Distance discrepancy U/V | [SVG](data/kollesis20_v1/ccw/discrepancy_heatmap.svg) | [SVG](data/kollesis20_v1/cw/discrepancy_heatmap.svg) |
| Rejection U/V | [SVG](data/kollesis20_v1/ccw/rejection_heatmap.svg) | [SVG](data/kollesis20_v1/cw/rejection_heatmap.svg) |
| Winding uncertainty U/V | [SVG](data/kollesis20_v1/ccw/uncertainty_heatmap.svg) | [SVG](data/kollesis20_v1/cw/uncertainty_heatmap.svg) |
| Cross-height elevation/support | [SVG](data/kollesis20_v1/ccw/v_coherence.svg) | [SVG](data/kollesis20_v1/cw/v_coherence.svg) |

All figures also have matching LZW TIFFs. Each directory includes five normal
separation-band heatmaps, support maps, CSV profiles, numeric NPZ maps and JSON
provenance/statistics. U/V buckets1000basevx, length quadrature<=250vx. Heatmaps
use equal physical scale in8consecutive U panels. Gray means insufficient support.

## Measurements

| Quantity | CCW | CW |
|---|---:|---:|
| UV pieces |489,663|489,663|
| Evaluated original events |6,606,100|6,606,100|
| Eligible crowding encounters before posterior weighting |1,749,845|1,749,845|
| Grid V×U |81×5,173|81×5,132|
| Expected rejected fiber length |35.3221%|35.4671%|
| Normalized conditional winding entropy |10.4328%|10.4608%|
| Posterior distance discrepancy, turns |0.50662|0.50777|
| Parallel discrepancy, turns |0.41877|0.41980|
| Perpendicular discrepancy, turns |0.98023|0.98271|
| Close0–32vx encounters /1000retained fiber vx |0.49729|0.49570|
| Eight-worker event evaluation |119.86s|101.39s|
| Complete calculation plus original plots |206.36s|182.09s|

Expected rejected length is probability/length-weighted, **not** the earlier
percentage of pieces with rejection probability>0.5. Both conventions conserve
total mapped length679,565,672.0783821basevx exactly at displayed precision.
Deposited event supports equal keep-product sums:2,575,393.677098328CCW and
2,560,736.089045551CW.

## Interpretation

Visual inspection shows variation, but no convincing common set of height-spanning
overlap bands yet. Strong raw U peaks often occur near loss of support at chart
edges; inspecting only those maxima would be misleading. Examples:CCW raw distance
discrepancy maximum at U=-1500 has only5.90weighted events; raw density maximum at
U=3,593,500 has4,061retained fiber voxels.

Descriptive Spearman correlations across U buckets where both metrics have
at least25%of V bins supported:

| Profile pair | CCW | CW |
|---|---:|---:|
| Distance discrepancy / uncertainty |0.441|0.394|
| Distance discrepancy / rejection |0.223|0.176|
| Distance discrepancy / close-pair density |-0.163|-0.165|
| Rejection / close-pair density |-0.069|-0.048|

These do not identify physical joints or exclude their existence. They suggest
the distance-error signal is entangled with assignment uncertainty rather than
simply tracking crowded layers. Current chart deformation/gauge errors may also
spread a physical seam across U. No candidates are promoted to accepted anatomy.

`metrics.json` also lists `peaks_with_25percent_v_support`, separated by>=10000vx
in U. For example CCW close-pair rate1.852 at U=2,574,500 has29.63%V support;
45.83%of supported V bins exceed the global75th-percentile rate. That is **not**
full-height evidence.

## Validation / reproducibility

Independent plan/code review approved.29tests passed across diagnostic, shared
layout, anchor and visualization modules. Tests cover exact all-pairs versus
prefix-sum expectations and threaded versus serial histogram identity. Source
database read-only; database/stage1 hashes match frozen provenance and cached
event IDs match exactly. Saved piece IDs, arcs and families checked. Both
conventions' plot previews inspected; overlapping profile ylabels shortened.

Initial single-worker probe measured~10min per event pass; switched to8CPU
workers with fixed batch/reduction order and reused geometry cache. Above timings
are one run per convention while unrelated global solves remained active, not
isolated performance claims. No GPU reservation for this task.

Regenerate plots from hash-validated accumulation using run_kollesis20_replot.json;
no event evaluation or inference needed. Temporary PNG inspection previews are
only in `/tmp/fiber_kollesis_inspection`; delivered plots are SVG/LZW TIFF.
