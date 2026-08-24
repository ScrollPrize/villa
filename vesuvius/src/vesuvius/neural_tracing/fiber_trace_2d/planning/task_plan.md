# Plan: arbitrary staged Fiberlet graph reduction

## CLI and stage geometry

1. Replace the fixed `--mode two-stage` geometry with `--mode staged` and a
   repeatable `--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z` option. Offsets are
   nonnegative base-voxel offsets relative to `--chunk`. Analysis boxes need
   not align to anchor cells or storage chunks.
2. Keep `--chunk X,Y,Z` and `--region-size N` as the half-open selected bbox.
   For each stage, enumerate all cubic boxes on its offset lattice that are
   completely contained in that bbox. Use deterministic Z/Y/X execution order;
   reject a stage that yields no boxes.
3. Preserve `--mode stats` as the existing read-only single-layout diagnostic.
   Remove the unpublished `two-stage` CLI mode rather than maintain a backward
   compatibility alias.

## Sparse cache layers

1. Add reusable core helpers which expose an anchor or Fiberlet dataset as a
   sparse overlay over a compatible lower cache. A valid upper payload wins;
   when both upper Fiberlet pair members are absent, reads fall through without
   publishing a copy. Partial prefix/route pairs are errors. Require identical
   grid, coordinate, storage profile, quantization, reach, and prediction-scale
   metadata between layers.
2. Give every stage separate temporary anchor and Fiberlet datasets under a
   unique per-invocation root which is never reopened as a completed cache and
   is removed after final reporting. Derive
   their identities from the preceding layer plus the ordered stage index,
   analysis side/offset, exact route-analysis scoring contract, and selected
   bbox. Keep the initial persistent anchor/Fiberlet caches unchanged.
3. Use the initial storage chunk grid in every layer. Stage analysis-box size is
   independent of storage ownership and may cover multiple storage chunks.
4. Add explicit replacement publication for mutable reduction overlays. It is
   only used while one stage is executing, after all graph leases for the old
   view are released. Prefix/routes remain a validated pair. Stage layers are
   temporary experimental data and are removed after reporting.
5. Upper decode/I/O errors never fall through. Fiberlet fallback is allowed
   only when both prefix and route members are absent; a partial pair is fatal.
   Explicit empty payloads shadow every lower layer through arbitrary overlay
   depth.

## Monotone box updates

1. Process boxes strictly serially in canonical Z/Y/X order. Before each box,
   create/read the current stage overlay over the preceding layer so an
   overlapping box sees every earlier removal in the same stage. Exact search
   inside one box may continue to use its configured worker count.
2. Run exact entry-to-first-exit analysis and the existing conservative
   post-analysis simplifier on the current graph.
3. Build the box's actual retained physical set as follows: only a Fiberlet
   whose canonical first endpoint anchor's stored base-space position lies
   inside the half-open box is eligible for removal; storage ownership remains
   derived from that endpoint key. All incident Fiberlets owned outside the box stay. For eligible
   Fiberlets retain only IDs surviving exact route selection and conservative
   directed reachability. Intersect with current contents so no update can
   restore an absent ID.
4. Rewrite each touched Fiberlet owner chunk from its current effective payload,
   filtering only eligible IDs and copying authoritative stored routes without
   numerical conversion or reordered accumulation.
5. Remove an anchor whose stored position is inside the box only after querying
   its complete effective incident-owner reach cube with the proposed Fiberlet
   update applied. Retain it if any surviving effective physical Fiberlet
   references it, including an outside-owned or lower-layer Fiberlet. Rewrite
   every storage chunk geometrically intersected by the box, including an
   unchanged or explicitly empty result; preserve outside-box records
   byte-equivalently at the record level.
6. Before publication, hard-check that every new prefix-ID and anchor-key set is
   a subset of the current effective set and that retained record fields and
   route data are unchanged. Persist empty touched chunks explicitly so empty is distinguishable from
   missing/fall-through. Recreate the stage overlay cache after a box update so
   later overlapping boxes cannot observe stale decoded payloads.
7. Keep macro-Fiberlet merging and deterministic rollouts as exact in-memory
   diagnostics. The existing spec forbids encoding them as ordinary Fiberlet
   records, so this task does not persist macros or use them to replace physical
   IDs.

## Statistics

1. Capture the canonical unique population over the selected bbox from the
   initial graph, after every stage, and after the final stage. Count `all` as
   every Fiberlet incident to the bbox and `interior` as both endpoints inside.
   Use the initial graph's stable endpoint geometry to classify all stages so
   denominators cannot drift after anchor removal.
2. Print one compact table with one row per stage: input, output, stage
   reduction, and cumulative reduction for all and interior populations.
3. Print the joint original/final totals and reduction for both scopes. Also
   print per-stage box counts, touched/rewritten storage chunks, anchors before
   and after, and temporary layer roots under `--stats`.
4. Retain per-box post-simplification diagnostics behind `--stats`; default
   output remains readable progress plus aggregate tables.

## Tests

1. Add a small multi-chunk fixture and reusable overlay tests for upper-hit,
   lower-fallback, explicit empty override, metadata mismatch, and partial
   Fiberlet-pair rejection.
2. Run aligned, half-offset, and whole-bbox stages and verify the equivalent
   two-stage geometry (`256/0`, then `256/128`) is generated for a 512 bbox.
3. Verify later stages and later overlapping boxes see earlier removals, never
   restore IDs, rewrite all affected initial-layout chunks, and preserve records
   outside each box.
4. Verify unused inside anchors are removed without deleting an anchor still
   referenced by an outside-owned or lower-layer surviving Fiberlet.
5. Verify explicit empty chunks shadow lower nonempty chunks through at least
   three layers, missing chunks fall through unchanged, corrupt upper data
   fails, and partial path pairs fail.
6. Verify per-stage and joint all/interior counts use canonical unions and are
   monotone.
7. Add negative replacement tests for attempted restoration, record mutation,
   and route mutation; test a straddling storage chunk and offset margins.
8. Confirm cold source generation remains allowed while staged reduction never
   prunes or rewrites an already materialized initial source chunk.
9. Build `vc_fiberlets`, `test_fiberlet_storage`, and `test_fiberlet_paths` with
   32 threads; run focused tests, `git diff --check`, and the hot Paris4
   512/256/128 staged command.

## Spec update

Replace the fixed two-stage chunk-route experiment with repeatable staged
analysis boxes, same-layout sparse overlay datasets, explicit empty overrides,
monotone interior ownership updates, deterministic overlap order, temporary
layer lifetime, and per-stage plus joint all/interior statistics. State that
sequential box-local pruning is deterministic but does not prove preservation
of a globally optimal replay route. Preserve the existing non-persistence rule
for macro-Fiberlets.

## Docs updates

Update `volume-cartographer/docs/fiberlets.md` with the staged CLI syntax,
geometry examples, layer fallback semantics, ownership/update rules, cache
lifetime, and statistics definitions. Remove the fixed two-stage invocation.

## Changelog update

Record arbitrary staged chunk-route reduction with sparse same-layout cache
overlays and monotone overlapping updates.
