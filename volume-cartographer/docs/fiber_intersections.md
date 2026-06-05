# Fiber Intersections

Fiber intersection search is split into a straight-distance candidate phase and
a single Ceres refinement per candidate.

## Spatial Index

`vc::atlas::FiberSpatialIndex` keeps one committed global segment R-tree and two
recent single-fiber R-trees. Segment entries store the fiber id, generation,
segment index, endpoints, arclength range, and segment AABB.

Committed entries are generation checked at query time. If a fiber has a recent
single-fiber tree, committed entries for that fiber are skipped so edits can
override stale global entries without rebuilding the whole global tree.

## Generations

Saved VC3D fiber JSON now carries a `generation` field. Legacy fiber files that
do not have the field load as generation `1`. New fibers start at generation
`1`; saving an existing fiber increments its generation.

Deleting a fiber removes it from the in-memory search index and prunes any cache
entries that mention that fiber. Saving a fiber prunes cache/index entries for
that fiber so the next query uses the new generation.

## Candidate Search

The broad phase uses only voxel-space Euclidean distance between line segments.
No normal, winding, or atlas-space scoring is used before Ceres. Candidate
clustering removes neighboring duplicate segment hits by fiber pair and
arclength proximity while preserving separated local intersections.

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

## Atlas Search Dock

The `Atlas Object Search` dock shows the selected atlas, search controls,
progress/cancel state, and a result table. The first implemented mode searches
all fibers mapped into the selected atlas against all saved fibers, which covers
in-atlas to outside-atlas pairs and in-atlas to in-atlas pairs.

Atlas mappings are used only to decide membership. Intersection geometry is
searched in the original saved fiber volume coordinates. Results are displayed
in the table and are not persisted into atlas metadata.

Ctrl-clicking in the atlas viewer focuses the main volume view at the clicked
volume point and leaves the atlas camera unchanged.

## Live Editing Preparation

The spatial index exposes recent-fiber updates separately from committed global
updates, so a transient edited fiber can be queried without inserting it into
the committed global tree. The two recent trees keep the last edited/updated
fibers available for fast reuse when switching between edit targets.
