# 2048 trace-density audit against the original 1024 region

Read-only analysis of saved beam OBJ geometry, trace Zarr `.zattrs` provenance,
saved configs/source patches, and native BP logs. No inference/tracing rerun,
solver change, or resource-limit increase.

Artifacts:
- `/ephemeral/hendrik/fiber_crop_bench/20261002_strict_reverse_crop_v1/beam-isotropic20`
- `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_best2048_v1/beam2048`

## Identical physical region

Intersect each exported polyline segment with XYZ box
`[10240,22016,6144]`–`[11264,23040,7168]` using slab clipping.
Count a trace when its intersection has positive length, not just an interior
vertex. Sum clipped segment lengths. OBJ coordinates are rounded, so reported
lengths are approximate; counts describe these saved visual geometry exports.
The original 1024 export verifies all 1879 polylines lie inside the box.

| Within original 1024 box | Original 1024 run | 2048 run clipped to same box |
|---|---:|---:|
| Trace pieces with positive-length intersection | 1879 | 9015 |
| Connected clipped spans | 1879 | 9089 |
| Entire piece inside box | 1879 | 179 |
| Clipped length, base voxels | 757054.638 | 4438377.515 |
| Pieces adjacent to original seed | 544 | 514 |
| Pieces recovered away from seed after consistency gaps | 1335 | 8501 |
| Recovered pieces whose original seed is outside box | 0 | 6102 |
| Recovered pieces whose original seed is inside box | 1335 | 2399 |

This is 4.80x as many intersecting pieces and 5.86x the traced length in the
same region. The extra geometry is not merely outside the small crop.

## Whole-crop construction counts

| Stage | 1024 | 2048 |
|---|---:|---:|
| Available seed anchors | 668061 | 6077437 |
| Seed candidates actually processed (not discarded speculation) | 1876 | 54316 |
| Distinct seeds contributing retained output | 1598 | 48710 |
| Retained pieces after reverse consistency | 1879 | 74816 |
| Additional retained pieces beyond one per contributing seed | 281 | 26106 |
| Seeds contributing multiple retained pieces | 281 | 24556 |
| Maximum retained pieces from one seed | 2 | 4 |
| BP pieces, current fine scheme | 2799 | 170485 |
| Additional BP subdivision cuts | 920 | 95669 |
| BP pieces, historical coarse/overlapping scheme | 2732 | not run |

Seed identity is the saved coordinate ZYX plus variant. The 281/26106 numbers
are net additional *retained* pieces, not every failed reverse check or gap.
Raw pre-filter polylines were not saved, so do not label contributing-seed counts
as all successful pre-filter traces. Fine BP pieces partition source lines;
their count minus source lines gives subdivision cuts. Historical overlapping
pieces are not a disjoint partition and should not be counted as equivalent cuts.

## Mechanism found in source

Both artifacts record dense beam8, lookahead2, reverse prefix512, iso20,
coverage40/80 and45 degrees. Saved binary hashes differ, but saved source patch
sections for `FiberTrace.cpp`, `FiberletCropTrace.cpp`, and `FiberTraceCli.hpp`
are identical. No evidence of a switch to fiberlet/greedy tracing.

`verifyDenseCandidate` retains successful intervals after failed reverse checks
and records `originalSeedBaseXYZ` for intervals disconnected from the seed.
`traceCropCandidates` checks whether the original seed is still active, then
accepts its recovered pieces. `suppressCoveredAnchors` suppresses nearby future
seeds; it does not test an incoming retained piece for overlap with existing
accepted geometry. Dense tracing does not implement stop-at-covered-anchor.

Consequently an uncovered seed outside the original box can produce a retained
piece inside an already traced region. The 6102 recovered pieces arriving from
outside account for about85% of the net increase in intersecting piece count.
This is a concrete coverage-policy mechanism consistent with the observed
growth, not yet a measurement that all such pieces are geometric duplicates.
Pairwise overlap/direction checks have not been performed. No corrective change
or reduced graph has been tested in this audit.

## BP outcome

2048 graph:170485 pieces,129797934 selected constraints. Orientation BP completed
in1517.99s. Winding support initialization failed its guard:534104915 candidate
states requested versus4000000 allowed. No2048 ladder exists. Raising that guard
does not address the upstream trace-density issue.
