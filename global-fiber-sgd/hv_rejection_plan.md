# H/V-specific rejection maps

Follow-up to pooled kollesis diagnostics; use the same saved20 CCW/CW states.
CPU only, no solve/inference, no repeated constraint processing. Exclude endpoint
density. User requests10000vx U-profile buckets and3000vx U/V heatmap cells.

## Metrics / localization
- Separate H and V: hard disabled length fraction (Pbroken>0.5), plain disabled
  piece count, disabled-piece fraction, mean rejection probability by piece and
  by length. Count original optimized pieces, not arc samples; disclose that
  the optimization includes a small number of source-fiber splits.
- Primary plots use conditional winding probability for placement, including
  rejected pieces. Counts therefore are posterior-weighted piece counts. Also
  provide MAP placement for genuinely integer plain counts and to assess smear.
- A piece counts once per intersected cell/hypothesis; profiles count once per
  intersected10000vx U bucket, irrespective of how many V cells it spans.
  Compute profiles independently, not by adding UV-cell counts. Length contributes
  exact overlap with each axis-aligned UV grid cell, normalized to source length.
- Plot H/V alongside each other, V-H and V/H for each metric; preserve V-alone
  readable data. Ratio invalid if H denominator iszero/too small; never epsilon
  inflate. Show H/V length and count support. Default>=5piece-equivalents and
  >=1000fiber voxels for fraction maps. H ratios require>=1%H rate, or>=3disabled
  H pieces for raw-count ratio. Report masks and numeric raw arrays.
- Profile10000vx, heatmaps3000vx, equal physical UV scale; consecutive U panels.

## Implementation / tests
Reuse shared solver coordinates and existing Maps/support plotting helpers.
Extract shared exact axis-aligned segment-cell overlap iterator and use it in
existing pooled length deposition as well (replace prior250vx quadrature, disclose
that old saved pooled results remain untouched). No copied geometry formulas.
Tests: exact length conservation including reversed directions and boundaries;
single count per piece/cell and perU bucket; MAP vs conditional localization;
threshold strictly>.5; H/V normalization masks; CSV/SVG/TIFF smoke. Independent
plan review, full generation, inspection and code review via approved runner.

## Spec / documentation updates
New diagnostic only. Archive previous active task log; update task/status/log,
README/changelog and standalone results with support definitions and limitations.
No anatomical labeling or automatic rejection changes. First inspect whether V
is selectively rejected; this alone does not locate an inner sheet backing.
