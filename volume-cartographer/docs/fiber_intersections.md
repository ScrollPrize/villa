# Fiber Intersections

Fiber intersection search is split into a straight-distance candidate phase and
a single Ceres refinement per candidate.

## Spatial Index

Atlas search builds a fresh one-shot point R-tree for each search run. The
search index is append-only for that run, stores dense samples by fiber id, and
is created with the requested `maxSampleSpacing` before pair search starts.
This keeps phase 2 progress tied to inserting the current fiber snapshot and
avoids hidden rebuilds during pair search.

Pair search precomputes per-fiber candidate lists and then refines requested
source/target pairs through a dynamic worker queue. Phase 3 progress counts
both candidate-list scans and requested pair slots, so the UI continues to
advance during candidate preparation as well as Ceres refinement.

`vc::atlas::FiberSpatialIndex` remains available for legacy/incremental
candidate queries. It keeps one committed global point R-tree and two recent
single-fiber point R-trees. Point entries store the fiber id, generation, dense
sample index, segment index, position, and arclength.

Committed entries are generation checked at query time. If a fiber has a recent
single-fiber tree, committed entries for that fiber are skipped so edits can
override stale global entries without rebuilding the whole global tree.

## Generations

Saved VC3D fiber JSON now carries a `generation` field. Legacy fiber files that
do not have the field load as generation `1`. New fibers start at generation
`1`; saving an existing fiber increments its generation.

Deleting or saving a fiber prunes any atlas-search cache entries that mention
that fiber. Atlas search builds its in-memory index from the current fiber
snapshot on the next run, so there is no long-lived atlas search index to
update.

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

## Intersection Inspection GUI

Opening an intersection creates an MDI inspection layout with the horizontal
display fiber on the left and the vertical display fiber on the right. Manual
H/V tags take precedence over automatic scores when choosing this display
ordering. The stored source/target identities of the intersection result are
not changed by display ordering.

Each side shows previous/current/next cross slices, a connection slice, a
follow slice, and a side-strip line slice. Cross and follow panes do not show
full line projections or generic crossing ellipses. They show the generated
line-position dot from the shared line-annotation generated overlay, nearby
control points from the same shared overlay, and an intersection `X` only when
the intersection point is within the existing 5% visible-viewport distance
threshold of the current slice plane.

Side-strip panes use the shared generated strip overlay. They show the center
line, all control points, the bracketing slice markers, and the current
intersection/focus position as a high-z-order `X` so it remains visible over
control-point markers.

Left-clicking generated cross, follow, or strip panes edits/adds a control
point at the pane's generated line position. Right-clicking uses the shared
generated-view context menu; it offers `Delete control point` and `New line
annotation`. Deletion is not hard-wired into intersection mouse-press handlers.

Follow panes track the mouse position over their side-strip pane until frozen.
Pressing Space freezes or unfreezes the active side's follow slice; when no
active side is known, Space toggles both. Frozen follow slices snap to a nearby
control-point line position using the same snapping tolerance as generated line
annotation views.

## Live Editing Preparation

The legacy `FiberSpatialIndex` exposes recent-fiber updates separately from
committed global updates, so a transient edited fiber can be queried without
inserting it into the committed global tree. The two recent trees keep the last
edited/updated fibers available for fast reuse when switching between edit
targets.
