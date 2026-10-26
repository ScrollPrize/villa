# Fiber Intersections

Fiber intersection search is split into a straight-distance candidate phase and
a single Ceres refinement per candidate.

## Spatial Index

`searchFiberIntersections` builds a fresh one-shot point R-tree for each run.
The search index is append-only for that run, stores dense samples by fiber id,
and is created with the requested `maxSampleSpacing` before pair search starts.
This keeps phase 2 progress tied to inserting the current fiber snapshot and
avoids hidden rebuilds during pair search.

Pair search precomputes per-fiber candidate lists and then refines requested
source/target pairs through a dynamic worker queue. Phase 3 progress counts
both candidate-list scans and requested pair slots, so progress continues to
advance during candidate preparation as well as Ceres refinement.

`vc::atlas::FiberSpatialIndex` remains available for legacy/incremental
candidate queries. It keeps one committed global point R-tree and two recent
single-fiber point R-trees. Point entries store the fiber id, generation, dense
sample index, segment index, position, and arclength.

Committed entries are generation checked at query time. If a fiber has a recent
single-fiber tree, committed entries for that fiber are skipped so edits can
override stale global entries without rebuilding the whole global tree.

## Generations

Fiber generations come from object-store records. Working snapshots pin both
object and geometry revisions; checked publications reject stale edits.

Each search builds its in-memory index from the fiber snapshot it is given, so
there is no long-lived search index to update.

## Candidate Search

The broad phase uses only voxel-space Euclidean distance between line segments.
No normal, winding, or atlas-space scoring is used before Ceres. Candidate
clustering removes neighboring duplicate segment hits by fiber pair and
arclength proximity while preserving separated local intersections.

When saved fiber control points are available, intersection search is limited
to the active line span between the outermost control points on each fiber.
Dense broad-phase samples outside that span are not indexed or queried, and
Ceres arclength parameters are bounded to the same span. This intentionally
disregards optimized line extensions before the first control point and after
the last control point. Legacy or synthetic fibers with fewer than two finite
control points fall back to the full polyline domain.

`maxDistance` is the R-tree candidate radius in original voxel coordinates, not
a refined score cutoff. Segment pairs farther apart than this are not sent to
Ceres. The default is `2000 vx` for atlas-scale searches; lowering it reduces
work but can miss intersections before refinement has a chance to run.

## Ceres Refinement

Each candidate creates exactly one Ceres problem. The solve is single-threaded
and optimizes two arclength parameters, one on each fiber. The problem contains
point-distance residuals and sign-ambiguous sampled-normal orthogonality
residuals for both fibers when normals are available. No stabilizing prior
residuals are added.

There are no local candidate-window bounds. Parameters are only constrained to
the valid arclength domains of their fibers so interpolation remains defined.
After Ceres, converged results in the same fiber-pair/arclength neighborhood are
deduplicated.

## Cache

`vc::atlas::FiberIntersectionCache` is in-memory only. Keys are unordered fiber
pairs plus both generations, broad-phase options, and Ceres options. Searches
skip pairs that are already covered by a cache hit.

## Live Editing Preparation

The legacy `FiberSpatialIndex` exposes recent-fiber updates separately from
committed global updates, so a transient edited fiber can be queried without
inserting it into the committed global tree. The two recent trees keep the last
edited/updated fibers available for fast reuse when switching between edit
targets.
