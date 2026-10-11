# H/V rejection diagnostics

Saved20-sweep CCW and CW, same inputs as the pooled kollesis study. No solve,
inference, new constraint extraction or change to baked H/V classification.
Profiles use **10000base-vx U buckets**; heatmaps **3000×3000base-vx cells**.

## What is measured

For each family, disabled means **P(broken)>0.5** (strict).

1. Disabled source arc length / total source arc length, percent.
2. Plain number of disabled optimized pieces intersecting the bucket.
3. Disabled pieces / all intersecting pieces, percent.
4. Mean P(broken) over intersecting pieces, percent.
5. Mean P(broken) weighted by within-bucket source arc length, percent.

Each profile has H/V overlays, V-H (percentage points or counts), and V/H (fold
ratio). Each heatmap has separate H, V, V-H and V/H columns, equal physical UV
scale, split into8consecutive U ranges. Support profiles show total H/V piece
counts and lengths. All numeric raw numerator/denominator maps and CSVs saved.

## Integer counts versus probability-weighted placement

- `map/`: put each piece at its highest-probability winding, even if rejected.
  Counts are actual integer counts of optimized pieces intersecting each bucket.
- `posterior/`: spread a piece over all100hypotheses using probabilities
  conditional on not broken. Counts are therefore **weighted piece counts**.
  Rejection is still the same hard>.5decision; soft plots use the original
  P(broken). Placement is not additionally multiplied by retention probability,
  which would erase the rejected fibers we want to inspect.

Geometry uses the existing model's axis-aligned UV piece representations, not a
new flattening of their3Dcurves. Each piece/hypothesis contributes once per cell
it intersects. Source length is clipped exactly into crossed UV cells and
rescaled to the original source arc length (signed float32 offsets can have tiny
roundoff). Zero-length endpoint touches never add counts.

Profiles are calculated independently of heatmaps. A long V piece crossing
many V cells still contributes **one** count to its U bucket, not one count per
V cell. H pieces crossing several U buckets count once in each intersected
bucket. Thus summing counts across U does not recover the global number of pieces;
the separate global summary counts each piece once.

These are optimized pieces, not necessarily whole original fibers: this input
has489663pieces from489197original fibers; only466additional pieces arise from
splitting. Earlier-stage rejected fibers are absent; all reported rejection is
from the global UV stage.

## Support and normalization

- At least5pieces (or piece-equivalents) of a family to show its metric.
- Length-based fractions/probabilities additionally require1000fiber voxels.
- V-H requires both family metrics to have support.
- V/H of a rate requires H>=1%; raw-count V/H requires>=3disabled H pieces.
  No epsilon is added. Undefined/weak ratios are gray/NaN, not zero or infinity.
  V-only rejection where H=0remains visible in the separate V and difference
  columns; it is deliberately not presented as an arbitrary huge finite ratio.
- Heatmap difference and ratio color limits use99th-percentile ranges for
  legibility; numeric arrays/CSV retain the full values. H/V percentage panels
  use the same0–100scale.

Both localization methods remain subject to assignment error. Local V excess
would be compatible with the proposed overlap workaround but does not prove
those fibers are the backing of the inner overlapping sheet. Check physical
normal ordering before anatomical conclusions. Crowding/damage/tracer artifacts
may also produce family-specific rejection.

## Reproduce

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_hv_rejection20.json
```

Output: `data/hv_rejection20_v1/{ccw,cw}/{posterior,map}/`.
`profiles.svg/.tif`, `support_profiles.svg/.tif`, five`*_heatmaps.svg/.tif`,
`profiles.csv`, `heat_raw.npz`, `profile_raw.npz`. Solution/input hashes and
global family counts/rates in each convention's`summary.json`; support-qualified
ratio peak values in`metric_peaks.json`. Delivered TIFFs useLZW. Temporary PNG
inspection previews only under`/tmp/fiber_hv_rejection_inspection`.

## Implementation and verification

`fiber_uv_hv_rejection.py` uses shared solver coordinates and `segment_cells`
from`fiber_uv_kollesis.py`. The existing pooled length-deposition function now
also reuses that exact overlap helper for future runs; its historical saved
250vx-quadrature results remain untouched. Legacy sample_step is recorded as
ignored on future pooled accumulation, rather than silently claiming quadrature.

Tests include reversed direction, exact grid boundaries, nonzero grid origins,
strict rejection threshold, one count per profile bucket despite many V cells,
conditional probability weights, ratio masks and plot/data/LZW smoke tests.
Every full run asserts length and rejected/expected-length conservation for both
families and both placements, and checks integer MAP counts.

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_hv_rejection.py vesuvius/tests/test_fiber_uv_kollesis.py vesuvius/tests/test_fiber_uv_layout.py vesuvius/tests/test_fiber_uv_anchor.py
```
