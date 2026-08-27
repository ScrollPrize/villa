# Plan: Split-piece fiber BP

## Contract

- Use every extracted constraint piece as one BP node. Preserve its source
  trace index and trace-local arc interval for diagnostics.
- Keep admitted dense source traces as a separate solver input: source traces
  select the established central-straight gauge fiber, while every report
  probability, marginal, weight, and diagnostic has piece cardinality. Require
  `constraints.inputTraces == sourceTraces.size()` and complete valid piece
  ownership for every admitted source trace.
- Accept extracted hard-continuity links only when they connect different
  pieces of the same source trace and carry the canonical parallel score 1,
  perpendicular score 0, and winding distance 0. Retain their established
  semantics as strong same-label evidence, not an equality constraint.
- Reject soft same-source links and hard cross-source links.
- Fully validate extracted continuity topology: every source-local consecutive
  piece pair has exactly one canonical link; it joins distinct consecutive
  piece IDs with ordered overlapping or abutting arc intervals, uses the
  overlap midpoint on both arcs, has zero distance and coincident finite points,
  and no missing, duplicate, nonconsecutive, hard/soft-colliding, or cross-source
  hard link is accepted.
- Merge repeated soft cross-source evidence only by unordered piece pair. Hard
  duplicates and hard/soft pair collisions fail. Do not collapse pieces back
  to their source trace.
- Select the primary source trace with the existing central-straight rule,
  then seed the piece of that trace with the smallest exact Euclidean
  crop-center-to-clipped-piece-polyline distance. Tie by global piece index;
  fail if the selected source has no piece. Clamp only that piece to H. This
  preserves the existing source-fiber gauge choice while making its piece
  mapping deterministic.
- Weight optional balance calculations by extracted piece arc length. The
  overlap is intentionally represented because each overlapping piece is an
  independent BP node.
- Charge the Mixed unary once per piece, independent of factor degree. Adjacent
  pieces may independently be Mixed; overlap and Mixed cost therefore both
  scale with the number of extracted pieces by design.
- Emit BP value bands, Mixed bands, CSV rows, consistency summaries, and
  reference confusion per piece. CSV rows retain global piece index, original
  source-trace index, trace-local piece index, and begin/end base-voxel arcs.
  OBJ layers use exact clipped dense source geometry, intentionally duplicate
  overlapping intervals, and partition all pieces. Every piece inherits its
  source trace's direction reference, so confusion and AUROC are piece-weighted.
- Keep the no-split case behavior unchanged: one piece per source trace yields
  the same graph, seed, weights, values, and exact original dense output
  geometry as before.

## Implementation

1. Add a shared constraint-piece-to-polyline helper used by both BP seed
   preparation and the CLI output path. Clip the original dense source
   polyline to each `[beginArc,endArc]`, retaining exact interpolated endpoints
   and original interior vertices; a full-range piece reproduces the source
   points exactly.
2. Change BP graph construction from source-trace nodes to piece nodes and
   validate continuity-link ownership and canonical values.
3. Map the selected central source trace to its closest piece, and construct
   per-piece arc weights.
4. Build the CLI's output geometry, original trace IDs, and diagnostic
   direction references from constraint pieces before solving/reporting.
5. Update public report and console terminology from trace to piece for node
   indices/counts, and report the seed's global piece, original source trace,
   and trace-local piece IDs.

## Spec Update

Replace the one-piece BP restriction with piece-node semantics, continuity
validation, deterministic seed mapping, piece weighting, and per-piece output
requirements.

## Docs Updates

Document finite `--piece-length` BP use, continuity evidence, and per-piece
OBJ/CSV interpretation in `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Add exact tree tests containing two pieces from one source trace, a canonical
  continuity link, and cross-source orientation evidence for min-sum, binary
  sum-product, and Mixed-state sum-product.
- Verify invalid ownership; missing, duplicate, nonconsecutive, or malformed
  continuity; hard cross-source; soft same-source; and hard/soft pair collisions
  fail. Verify repeated soft cross-source measurements still merge.
- Verify the selected seed belongs to the existing central source trace and is
  its exact crop-center-nearest piece, including deterministic overlap ties.
- Verify continuity is finite strong evidence rather than equality, Mixed
  unary is once per piece rather than per factor, and overlapping arc weights
  are counted once for each piece.
- Verify pruning and `--perpendicular-only` retain canonical continuity into
  BP.
- Verify per-piece CSV identity/arc fields and OBJ partitions use exact clipped
  geometry, and no-split output preserves source points and cardinality.
- Verify no-split exact graph, seed, weights, probabilities, and min-marginals
  remain unchanged.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused suite, run a finite-piece 1024 BP command, and run
  `git diff --check`.

## Changelog

Record split-piece BP support and piece-level diagnostics.
