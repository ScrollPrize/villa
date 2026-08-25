# Plan: apply staged Fiberlet filtering before replay

## Semantics and coverage

1. Represent a filter stage independently of CLI and storage layout as a
   positive cubic side plus a globally anchored XYZ offset normalized modulo
   that side.
2. Build complete final-stage analysis boxes whose half-open base-space extents
   intersect the requested replay tube. Clip candidate indices to boxes wholly
   inside the prediction volume; do not create partial boundary boxes.
3. Walk stages backward. For each required later-stage box, expand its read
   extent by the dataset's maximum endpoint reach and select every complete
   preceding-stage box whose write extent intersects that read extent. Union
   and sort boxes deterministically in Z/Y/X order.
4. Use the union of required stage boxes plus endpoint reach as the generation
   support. Keep this support separate from the original replay corridor: the
   former controls source chunk generation, while the latter continues to
   constrain graph seeds and traversal.

## Shared implementation

1. Add reusable core planning helpers for global stage-box selection and
   backward coverage closure. Express all crossings in base coordinates so
   storage chunk side, generation chunk side, and stage side may differ.
2. Add reusable transient staged-overlay orchestration which applies the existing
   `analyzeAndSimplifyFiberletChunkRoutes()` and
   `writeFiberletReductionOverlayBox()` path in stage/box order and exposes the
   final anchor/Fiberlet datasets and caches for replay.
3. Keep `chunk-route-stats` on the same reduction primitives. Do not fork its
   simplification or overlay-write behavior.
4. Extend the cached replay graph source with an explicit traversal-cell
   predicate. Default it to the preprocessor selector for compatibility;
   filtered replay supplies the original tube predicate.

## CLI and replay integration

1. Permit repeatable `--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z`, `--join-angle`,
   `--cost-profile`, and `--max-states` for `fiberlet-replay`. Stages are
   optional and their presence enables filtering; `--mode` remains a
   `chunk-route-stats` diagnostic option.
2. Before constructing the on-demand preprocessor, plan all required boxes and
   replace its generation selectors with the expanded support predicate. Add
   the support contract and ordered stages to source-cache metadata.
3. Enumerate every storage owner intersecting the generation support, schedule
   all anchor dependencies and Fiberlet pairs, and wait for successful
   completion before the first reduction stage.
4. Build transient stage overlays under an invocation-owned directory below
   the replay output, retain them through tracing, and remove them on every
   normal or exceptional exit. Do not persist or reopen filtered overlays.
5. Construct replay over the final overlay while retaining the original tube
   as its traversal predicate. Record the effective filter stages in the replay
   bundle and diagnostics.

## Tests and validation

1. Unit-test global lattice anchoring, negative/large offset normalization,
   half-open intersection, volume-edge exclusion, deterministic ordering, and
   backward expansion across multiple offset stages.
2. Test mismatched storage and stage sides, including one stage box spanning
   several storage chunks and several stage boxes sharing one storage chunk.
3. Test that every final-stage box is complete, all required lower-stage boxes
   and source owners are present, and removing one planned dependency is
   detected before tracing.
4. Test that the replay traversal predicate remains the original tube while
   generation uses expanded support, and that no stages preserves the existing
   graph/source path.
5. Build `vc_fiberlets`, run focused Fiberlet storage/path/replay tests, and run
   `git diff --check`.

## Spec update

Add globally anchored staged replay filtering, complete final-box coverage,
backward endpoint-reach closure, separate generation/traversal predicates,
independent storage/filter grids, transient overlay lifetime, and unchanged
unfiltered defaults to `planning/specs.md`.

## Docs updates

Update `volume-cartographer/docs/fiberlets.md` with replay CLI examples,
global-offset semantics, coverage expansion, transient cache behavior, and the
distinction between storage chunks and filter analysis boxes.

## Changelog update

Add a dated entry for optional staged Fiberlet filtering in cached replay.

## Independent review

Review the implementation against the existing staged-reduction monotonicity,
complete-box, owner-write, cache-identity, and replay-corridor invariants before
final validation.
