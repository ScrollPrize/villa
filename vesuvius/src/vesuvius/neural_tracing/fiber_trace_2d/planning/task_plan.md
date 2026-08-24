# Plan: two-stage regional Fiberlet graph reduction

## Semantics

1. Keep `--chunk X,Y,Z` as the cubic target-region minimum in base XYZ. Add
   `--region-size N`, defaulting to `--chunk-size`. Derive the globally aligned
   reduction chunks intersecting that target region; the region does not define
   a cache namespace or private coverage domain.
2. Add `--mode stats|two-stage`. `stats` preserves one-pass analysis but runs
   every aligned chunk contained in the selected region. `two-stage` first
   runs that aligned grid, persists the union of retained physical Fiberlets,
   then analyzes the half-chunk-offset grid fully contained in the region.
3. Each stage-one reduction chunk is independently reusable. It stores only
   Fiberlets canonically owned by that chunk. Exact analysis of the same box
   retains every boundary-crossing Fiberlet, while an internal Fiberlet belongs
   to only one non-overlapping box, so no region-specific retained-ID union is
   needed to define chunk contents.
4. A 512/256 region produces eight stage-one boxes and one stage-two box at
   `minimum + 128` on each axis. Larger regions produce all fully contained
   half-offset boxes, in canonical XYZ order.
5. Regional counts are canonical unique physical-ID sets, never sums of
   per-box counts. For the offset grid, `original_before` is the unique source
   population, `stage1_after` is its intersection with the reduced dataset,
   and `stage2_after` is the union of retained IDs from the offset analyses.
   Internal classification uses the original endpoint geometry. Total and
   incremental percentages use `original_before` and `stage1_after`
   respectively.

## Core and storage

1. Extend the exact route report with canonically ordered retained physical
   Fiberlet IDs and exact retained internal counts. Preserve current search,
   tie, cycle, first-exit, and cost semantics.
2. Add a reusable per-chunk reduction writer beside `FiberletChunkDataset`.
   Given one globally aligned analysis box, it writes only prefix/route records
   whose canonical first anchor is owned by the matching reduced-cache chunk.
   It loads authoritative records through ordinary graph/cache APIs and uses
   the existing serializers and pair publication.
3. Give the global reduced dataset strict metadata derived from the source
   dataset plus a reduction descriptor containing reduction chunk size, pass,
   source dataset identity, coordinate/grid contract, edge-cost view, maximum
   join angle, effective local/path scoring configuration, and a canonical
   algorithm contract. Write it below the command output root in a deterministic
   fingerprinted namespace independent of the requested target region. Never
   rewrite the source cache.
4. Keep reduced storage chunks aligned to the analysis grid. Present the
   original anchor cache through a cache-backed rechunked view with the same
   ownership grid; the view aggregates already evaluated source-anchor chunks
   and never regenerates or publishes duplicate anchors. Stage two therefore
   changes only the available Fiberlet graph, not anchor geometry or scoring.
5. Generate only missing reduced chunks selected by the target region and
   reuse already valid chunks byte-for-byte. Stage two reads a bounded view of
   exactly those stage-one owners. Conservative graph requests outside that
   set resolve to ephemeral empty chunks which are never published; they must
   not trigger additional reduction work. A selected generated chunk with no
   retained records is stored as an explicit empty prefix/route pair.
6. Keep decoded data cache-bound. Do not materialize the entire regional graph
   at once beyond the retained-ID set and per-output-chunk records.
7. Reserialize authoritative stored prefix costs and routes with the derived
   dataset codec/fingerprint while preserving pair index alignment. The
   selected sqrt-u16 analysis view is applied only during route analysis and
   never mutates stored records.
8. Validate both members of every requested pair. A hot complete chunk is
   immutable; an incomplete pair is safely regenerated before that chunk can
   be read by stage two.
9. Collect the original offset-grid population IDs independently from the
   immutable source graph. Do not infer the original denominator from the
   already reduced graph or rerun an exact search merely to obtain population
   membership when shared population materialization can expose it.

## CLI and progress

1. Reuse the existing replay cache/preparation progress coordinator for source
   cache generation. Add a regional analysis progress phase with completed/
   total boxes, elapsed time, and ETA.
2. Print a compact stage-one table and a stage-two table. The stage-two table
   contains original-before, stage-one-after, stage-two-after, total reduction,
   and incremental reduction columns for all and internal Fiberlets.
3. Keep `--stats` as optional detailed completed-chunk/search output. Normal
   progress and headline tables must not require it.
4. Require even integral `--chunk-size` for half-offset mode,
   `--region-size >= 2 * --chunk-size`, bounds without integer overflow, and
   centered boxes fully contained by both the selected target region and the
   dataset base-coordinate extent. Region/chunk alignment is not required.

## Tests

1. Extend the deterministic graph fixture to expose retained IDs and verify
   their canonical ordering and internal classification.
2. Add a small 2x2x2 regional fixture where stage-one unions overlap, writes a
   reduced dataset, reopens it through stored caches, and the centered stage-two
   analysis removes an additional interior branch.
3. Assert unique regional ID aggregation and exact original/stage-one/stage-two
   set relationships, including internal classification and percentage
   denominators.
4. Assert source payload bytes and mtimes are unchanged, reduced prefix/route
   tuples remain aligned and are re-encoded under the derived fingerprint,
   empty processed owners are explicit, and a hot reopen preserves derived
   bytes and mtimes without generation.
5. Test missing and partial requested pairs, safe per-chunk regeneration,
   reuse of the same reduced chunk from two overlapping target regions, and CLI
   rejection of odd chunk sizes, undersized regions, overflow, and
   out-of-bounds centered boxes.
6. Build `vc_fiberlets`, `test_fiberlet_storage`, and `test_fiberlet_paths` with
   32 threads; run both suites and `git diff --check`.
7. Run a 512/256 two-stage Paris4 region around the current reference location
   and record exact commands, cold/hot timings, counts, and reduction rates.

## Spec update

Extend the chunk-route diagnostic contract with regional aligned and
half-offset grids, union retention, immutable derived Fiberlet datasets,
lazy reusable reduced chunks, stage-two original-baseline comparison, and exact
internal/all-Fiberlet reporting. State that box-local optimum pruning is not
globally compositional: the two-stage result is an experimental local-pruning
diagnostic, not proof of global replay-route preservation. Preserve all
existing route semantics.

## Docs update

Document `--region-size`, `--mode`, stage-one reduced-cache layout, centered
stage-two behavior, tables, progress, cache reuse, and a 512/256 invocation in
`volume-cartographer/docs/fiberlets.md`.

## Changelog

Record the two-stage regional reduction pipeline and its measured Paris4
stage-one/stage-two reductions. Do not claim that the reduced cache is yet the
default replay graph.
