# Anchor coverage plan

1. Archive prior planning; preserve completed short200 result (67failures,1485.621s).
2. Add anchors backend to native replay-set: reuse GT eligibility, bindings, canonical preprocessor/cache, referenceChunkSchedule at max ellipse radius, permits/readiness. Only request anchor chunks, never fiberlet dependencies. Generate complete owner chunks to preserve cache identity; no GT-filtered extraction.
3. Core native metric: spatially indexed closest Euclidean projection onto GT segments, shared acceptance ellipse at projection. Canonical anchor positions, no refinement. Equality accepted; report invalid-normal fallback. Deduplicate stable IDs per piece; unique projected arcs for spacing.
4. Interior gaps in base voxels / actual anchor cell side (currently32basevx). Not literal crossed lattice cells. Histogram edges0,1,2,3,4,5,6,7,8,16,32,infinity; median/P95/max. Report leading/trailing gaps separately, full uncovered length for no-anchor pieces. Additional one-cell-side arclength bins starting at arc0: occupied fraction and empty-run histogram (phase/orientation dependent). Save per-anchor coordinates/projections and gaps.
5. Approved benchmark-full runner handles anchor-specific summary/aggregation, not restart scoring. Retain scroll grouping/shared inference session/GPU release30idle seconds. Python orchestration/analysis only, native extraction/matching. Preserve immutable config/model/source provenance.
6. Tests: straight/bent lines, anisotropic boundaries, missing normals, no/one anchors, duplicate projections, reverse invariance, aggregate histogram. Native integration checks no fiberlet generation. Approved build/tests then run200 and report.

## Spec update
Independent anchor coverage protocol/units/end treatment. Existing extraction/tracing defaults unchanged.

## Docs updates
Anchor guide/results, completed previous comparison, task log/status/changelog.

## Review
Independent plan review before implementation.

Review completed by anchor_plan_review. Preserve existing owner-cell-diagonal
padding through a shared preprocessor helper also used by graph lookup. Use flat
endpoint caps (reject projections outside eligible endpoint planes), nearest
Euclidean projection followed by ellipse, not union-of-ellipses acceptance.
Pool individual gaps for quantiles; no fabricated gaps for zero/one-anchor pieces.
Include final partial arc bin and assign endpoint to last bin. Reversal invariance
applies to interior gaps, not phase-zero occupancy bins.
