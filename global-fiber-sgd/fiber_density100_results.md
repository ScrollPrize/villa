# 100-sweep H/V density results

Completed both conventions using run_density100.json. Outputs:
`data/density100_v1/{ccw,cw}`. Definitions/commands: fiber_density100_metrics.md.
All figures have SVG and LZW TIFF counterparts. Enabled selection Pbroken<=0.5;
all variants include disabled UV pieces with full length at one MAP location.

## Inspect

| Convention | Overlaid length profiles | Enabled heatmap | All saved heatmap | All constraint-replaced heatmap |
|---|---|---|---|---|
| CCW | [profiles](data/density100_v1/ccw/length_profiles.svg) | [enabled](data/density100_v1/ccw/enabled_length_heatmaps.svg) | [all saved](data/density100_v1/ccw/all_saved_length_heatmaps.svg) | [all refit](data/density100_v1/ccw/all_refit_length_heatmaps.svg) |
| CW | [profiles](data/density100_v1/cw/length_profiles.svg) | [enabled](data/density100_v1/cw/enabled_length_heatmaps.svg) | [all saved](data/density100_v1/cw/all_saved_length_heatmaps.svg) | [all refit](data/density100_v1/cw/all_refit_length_heatmaps.svg) |

H/V separate within each plot;3000vx heat cells and10000vx profiles. Additional
plots show only added-disabled density, and piece-count density. Raw grids/areas,
profile CSVs, selected buckets/cost/support arrays and source hashes retained.

## Placement diagnostic and timing

| Metric | CCW | CW |
|---|---:|---:|
| Disabled pieces | 94114 | 94545 |
| With enabled-neighbor support | 94107 | 94543 |
| Unsupported, saved fallback | 7 | 2 |
| Changed chosen bucket | 53768 (57.13%) | 54026 (57.14%) |
| Sign conflicts with enabled neighbors, before | 401634 | 407182 |
| Sign conflicts with enabled neighbors, after | 248298 | 250025 |
| Conditional placement seconds | 8.98 | 8.41 |
| All calculation/provenance seconds | 19.48 | 18.36 |
| Total including figures seconds | 58.30 | 58.14 |

These sign conflicts concern forcibly placed disabled pieces, not the four
remaining violations in either enabled100-sweep solution. Rejected pieces can
remain incompatible with enabled neighbors even at their cheapest placement.
They have NOT been enabled in the solver; no state modified. Geometry is frozen.

## What the plots show

Including disabled pieces increases global H length density by55.60%CCW /56.08%CW
and V density by44.35% /44.43% relative to enabled-only (same area denominator).
This is expected inclusion of omitted length, not evidence for a kollesis.

Recomputed placements change the full-height profiles relatively little: the
sum of absolute per-bin length changes / total included length is2.442%H and
2.605%V CCW;2.546%H and2.582%V CW. This measures redistribution, not a global
density change (total length is conserved). Thus many bucket changes do not
produce a very different coarse density map.

Visual inspection of both profiles and H/V heatmaps: broad U-dependent density
trend and local structure, but no convincing common narrow full-height band
identifying kollesis. Not a negative anatomical finding: inferred UV distortion,
spatial acquisition/extraction completeness and substantial incompatible/disabled
fibers confound density. Earlier-stage rejected fibers remain absent entirely.

## Validation

21tests passed across density, H/V rejection, pooled diagnostics and categorical
costs. Includes exact length conservation, same-area denominators, grid-edge
handling, independent V profile count, unchanged enabled choices, tied/supported
and unsupported placement, canonical cost oracle for both endpoint orientations,
continuation-only support and signed offset geometry. Independent plan/code
review found no blockers; added full helper/config hashes and continuation test
per review. git diff --check clean. Approved runner used, CPU only; no model/GPU
or inference changes. No requested diagnostic deferred.

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_density.py vesuvius/tests/test_fiber_uv_hv_rejection.py vesuvius/tests/test_fiber_uv_kollesis.py vesuvius/tests/test_fiber_uv_categorical.py
```
