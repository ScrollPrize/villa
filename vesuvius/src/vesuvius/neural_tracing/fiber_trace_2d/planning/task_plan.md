# Plan: Truly Rolling Shared 3D Tiled Inference

## Current failures

`_RollingZBand` currently creates a sparse file and `np.memmap` shaped as
`(full_output_z, full_output_y, full_output_x)` for every raw channel and weight
array. Advancing `origin_z` and punching old pages does not make this a rolling
mapping: the complete Z stack remains mapped and logically reserved. In the
reported Fiber run, seven raw channels plus weights produce roughly 37.5 TiB
of logical mappings.

Flushing is separately unbounded. It exposes a completed Z band across the
entire XY plane, creates a weight-sized `np.maximum` temporary, and stacks all
raw channels before finalization. For the reported geometry, this reaches tens
to hundreds of GiB and is where interruption showed the process stuck.

The code is also not genuinely shared:

- Fiber calls `_infer_tiled_products_3d`.
- Lasagna calls `_infer_tiled_3d` and owns a separate `_on_z_complete` flush
  implementation in `preprocess_cos_omezarr.py`.

They reuse helpers and types, but duplicate the inference engine.

## Target architecture

### 1. One authoritative tiled runner

Make one runner in `lasagna.tiled_predict3d` authoritative for both Lasagna
and Fiber. After compatibility tests pass, delete `_infer_tiled_3d`, the
current `_RollingZBand`, and Lasagna's caller-owned neural flush loop. Do not
retain a fallback engine.

The shared runner owns:

- global tile and output-chunk lattices;
- crop-to-global coordinate mapping and edge padding;
- tile traversal, input reads, dtype conversion, and missing-input handling;
- blend weights and source-relative per-product downsampling;
- output completeness and resume scheduling;
- circular accumulation, readiness, chunked normalization/finalization,
  atomic writes, slot reuse, cleanup, and interruption behavior;
- shared progress and timing.

Model adapters own model construction/loading, tile preprocessing,
autocast/inference policy, raw-output splitting, activation semantics, and
raw-to-persisted product conversion. Output adapters own coherent product
completeness, metadata, and atomic channel writes. Pyramid creation remains a
shared post-inference operation.

Lasagna's `pred_dt` is not a neural model product and is not derived from
`cos`: it is generated from the separate user-supplied `--pred-dt` prediction
Zarr. Keep it as an independently resumable external-source stage that cannot
schedule neural inference when neural products are already complete.

### 2. Fixed-depth circular mmap accumulator

Implement circular mmap accumulators grouped by distinct source-relative
product scale. Each scale group contains:

- separate raw float32 sum planes for every product/channel at that scale;
- one shared float32 weight-sum ring determined solely by tile geometry,
  blend support, input availability, and scale—not product resume masks;
- a fixed number of physical Z slots;
- the global logical Z represented by each slot or generation metadata;
- a logical origin and completed/flushed frontier.

The mmap shape is `(channels, ring_depth, working_output_y, working_output_x)`, never
`(channels, full_output_z, output_y, output_x)`. The kernel manages resident
pages and writeback of this fixed backing file. There is no application-level
RAM ceiling and no full-volume sparse reservation.

The Y/X extent is the padded crop/output working extent with explicit global
output offsets, not automatically the full-volume XY extent. Tiles and output
chunks remain globally anchored so a crop and full run still produce identical
bytes for a shared complete global chunk.

Determine `ring_depth` from the actual canonical Z positions and actual
nonzero support of each scale's downsampled blend weight. Evaluate the frontier
after each complete Y/X tile row and include output-chunk-aligned flush lag.
Capacity is the maximum span between the oldest not-yet-flushable logical plane
and the greatest write end reached before the next flush opportunity. Do not
approximate this using nominal tile size/overlap alone. Preserve the legacy
common tile lattice alignment across the complete scale set (or prove nested
factors explicitly); do not silently change max-scale alignment to a different
per-scale/LCM lattice. Assert before every write that it cannot overwrite
unflushed data.

When the inference frontier proves that no future tile can contribute to a
logical Z range:

1. round the flush frontier down to complete output Zarr chunks, except for the
   final clipped edge;
2. visit output chunks in canonical Z/Y/X order;
3. take only that chunk's views from the circular raw sums and weights,
   splitting at the ring wrap when necessary;
4. normalize with an in-place divide or reusable chunk-sized scratch—never a
   full-XY `np.maximum` result;
5. finalize only that chunk's raw channels into persisted channels;
6. atomically write the coherent product chunk;
7. clear each product's raw storage after that product chunk is finalized;
8. clear shared weight storage only after every incomplete product using that
   chunk region has been finalized or skipped, then reassign the physical Z
   generation after every XY region for those planes has been cleared.

Where a wrapped chunk has two physical pieces, copy only that output chunk
into a chunk-sized contiguous scratch buffer. No scratch allocation may scale
with full XY, full output Z, or an entire multichannel Z band.

Use a reusable chunk-sized denominator scratch implementing exactly
`maximum(weight, 1e-7)`, including weights strictly between zero and epsilon.
Do not normalize or clear shared weight storage still needed by another
product. Slot clearing must likewise remain chunked; never zero a complete XY
plane in one operation.

Scratch is disposable and should not be flushed or fsynced during normal reuse
or cleanup. Do not depend on Linux hole punching for correctness. Close mmap
objects explicitly before unlinking and keep the lifecycle POSIX-portable for
Ubuntu and macOS.

Every globally anchored model tile is inferred at most once per run. There are
no macroblocks, no boundary re-inference, and no unbounded prediction cache.

### 3. Multiple product scales

The current `OutputProductSpec.scaledown` is effective scale relative to the
base volume, while tile pyrdown needs scale relative to the selected input
array. Represent the source-relative inference scale explicitly in the runner
or product plan; do not accidentally apply the base-relative value to a `/1`
or `/2` input. Validate the relationship with `input_sd` exactly and preserve
manifest levels/base-relative scale.

Create one circular accumulator group and one shared weight ring per distinct
source-relative scale. Products at that scale retain separate raw rings but
share the geometrically defined denominator.
Lasagna therefore uses a fine `cos` ring and a coarser `grad_mag/nx/ny` ring;
Fiber normally uses one full-resolution-relative ring per option. Each ring has
its own depth and flush frontier, while all products share the same single-pass
tile traversal and model result.

Cache blend weights once per distinct scale. Preserve the existing operation
order: multiply model predictions by the full-resolution blend weight, then
apply `_pyrdown3d`, then accumulate float32 values in canonical tile order.

### 4. Resume and readiness

Resume state remains durable output chunks only. Circular mmap files are
disposable scratch and must be cleaned on normal completion or interruption;
they are never treated as completion markers.

Plan neural work per product and scale. A tile may still be needed for one
product while every affected output chunk of another product is complete. Run
the model once when any neural product needs the tile, but allocate/update only
the exact incomplete chunks of each product. Mixed complete/incomplete chunks
within a single tile must produce the same raw sums and weights as a clean run.

Accumulate the scale's shared weight once over the union of incomplete output
regions at that scale. Weight scheduling is product-independent: every
incomplete product chunk must receive denominator contributions from every
input-supported global tile intersecting it, regardless of which product made
the model tile necessary. Existing chunks do not receive raw accumulation and
are not rewritten. Track per-region product liveness/reference counts so the
shared weight region remains valid until the last incomplete consumer has been
finalized or skipped.

Follow the spec-authoritative coherent-product rule: if any required sibling
chunk is missing, the product chunk is incomplete and its complete sibling
bundle is rewritten through atomic temporary paths. Missing only `pred_dt`
schedules only external-source EDT work.

Missing input tiles advance every relevant scale frontier without accumulator
writes. Unsupported output chunks are finalized-as-skipped and their slot
regions cleared. Readiness and reuse account for processed and deliberately
skipped tiles so no product or generation can stall.

## Lasagna/Fiber divergence audit

Before consolidation, characterize both existing paths and assign every
difference either to the shared runner or to a documented adapter hook.

### Mechanics that must become identical shared code

- canonical tile-position generation and global crop anchoring;
- padded input reads and absent-input-chunk checks;
- tile traversal order and output support calculation;
- blend-ramp construction and scaled weight generation;
- tile/output clipping and scale validation;
- product completeness scheduling and skip decisions;
- accumulation order, one geometric weight ring per scale, and product-liveness
  tracking for weight normalization;
- circular Z readiness, chunk flush ordering, and scratch reuse;
- atomic output writes, interruption cleanup, and output-only resume;
- progress counters, ETA calculation, and TTY/non-TTY rendering;
- common pyramid and metadata orchestration where product policy permits.

Neither CLI may retain its own tile loop, accumulator loop, or flush loop.

### Differences that belong in explicit adapters/specification

- Lasagna model/checkpoint construction versus Fiber model/config/snapshot
  construction;
- Lasagna activation policy (`sigmoid` or clamp) versus Fiber's trained output
  semantics;
- Lasagna's existing bfloat16/autocast behavior versus Fiber's configured
  mixed-precision context;
- Fiber per-tile image normalization and validity handling;
- raw output shape/dictionary extraction and product/channel splitting;
- Lasagna fine `cos` and coarse seven-channel normal product versus Fiber's
  seven raw channels per option;
- Lasagna `cos`, scaled `grad_mag`, and compact normal encoding versus Fiber
  presence rounding and compact normal encoding;
- multi-option Fiber completeness and channel naming;
- per-product pyramid policy and manifest channel schema;
- external `pred_dt_zarr` coordinate conversion, halo/chunk alignment, and
  CPU/GPU EDT implementation.

Remove the currently unused/misleading adapter methods that pretend sharing
while the caller still owns accumulation. Keep adapter hooks narrow and named
after semantic operations, not generic escape hatches.

### Behavior that must be explicitly reconciled

- `uint16` input conversion and any product-specific normalization;
- first-tile diagnostics and NaN reporting;
- partial sibling handling: use the coherent rewrite required by specs rather
  than Lasagna's current missing-channel-only shortcut;
- shared-weight liveness across different product/option resume masks;
- independent readiness of products at different scales;
- crop edge chunks and byte identity between full-volume and crop runs;
- source-relative inference scale versus base-relative manifest scale;
- pyramid timing and failure/resume behavior;
- temporary path naming/removal and delayed Ctrl-C inside native NumPy work.
- duplicate Lasagna `_download_one_path`/`_auto_download` logic versus the
  shared download helpers, including independent auto-download of `pred_dt`;
- Fiber-local crop expansion, output alignment, and base-scale resolution
  versus Lasagna geometry setup;
- model lifecycle: Lasagna `gpu_pause_context`, optional InstanceNorm
  calibration, and CUDA cleanup versus Fiber lifecycle;
- `torch.inference_mode`/`no_grad` ownership and scope;
- validity masks, reflect padding, missing-chunk values, and normalization;
- CLI/device/tile/default policy versus engine mechanics;
- stale temporary cleanup and prefix policy;
- `source_to_base`, grad-magnitude factor, and manifest metadata;
- pyramid workers and pyramid failure/resume behavior.

Keep GPU pause and InstanceNorm calibration as explicit Lasagna orchestration
or semantic hooks; consolidation must not silently drop them. Keep `pred_dt`
outside neural product lists/work scheduling, including its separate source
download.

## Progress and observability

Use one shared progress reporter for both CLIs. Report:

- stages: model load, planning, tiled inference, chunk flush/write, external
  derived products, and pyramids;
- tiles processed/skipped and ETA;
- chunks finalized/written/skipped;
- logical Z inference frontier and per-scale flushed frontier;
- ring depth, backing-file size, current logical Z window, and wrap count;
- tile inference time separately from flush/finalization time.

Use carriage-return refresh only for a real TTY. For redirected/captured output,
emit time-throttled newline records. Emit a durable line before and after a
potentially long flush. Chunk-sized NumPy operations keep interrupt latency
bounded; check for interruption between chunks and always clean scratch files.

## Implementation sequence

1. Add characterization tests for both existing inference engines: lattices,
   blend values, scale conversion, operation order, crop identity, resume,
   missing input, output bytes, progress, and adapter-specific semantics.
   Record a before/after call-graph and symbol-ownership table covering every
   divergence above.
2. Add a pure circular-layout planner that computes per-scale ring depth,
   logical-to-physical Z mapping, safe flush frontiers, wrap splits, and backing
   sizes without allocating a volume.
3. Implement and test the fixed-depth circular mmap store, generation safety,
   chunk views/copies across wrap, clearing, cleanup, and interruption.
4. Refactor `_infer_tiled_products_3d` into the sole shared runner, adding
   source-relative per-product scales and chunk-by-chunk finalization.
5. Move all Lasagna model/product semantics into
   `LasagnaCosPredict3DAdapter`; implement external `pred_dt` as an independent
   resumable stage; route `predict3d` through the shared runner.
6. Route Fiber through the same runner using only its semantic adapters.
7. Delete `_infer_tiled_3d`, the old `_RollingZBand`, Lasagna's
   `_on_z_complete`, duplicated imports/helpers, and fake accumulation adapter
   methods. Test that both CLIs call the same exported runner and that the
   legacy runner symbol is absent.
8. Run byte-compatibility, resume, interruption, backing-size, memory-residency,
   and representative throughput tests. Update specs/docs/changelog/task log.

## Testing and acceptance criteria

### Circular accumulator tests

- Compare circular accumulation and chunked finalization to a small dense
  reference across tile sizes, overlaps, borders, chunk sizes, scales, edges,
  crops, and multiple wraps.
- Prove with synthetic huge-Z planning tests that mmap shape/backing size is
  independent of full output Z.
- Assert a logical interval cannot overwrite unflushed slots and stale slot
  generations cannot be read.
- Test chunks wholly inside the ring, chunks split across wrap, final clipped
  chunks, zero-weight regions, missing-input tiles, and interruption cleanup.
- Assert temporary normalization/finalization memory is bounded by one output
  chunk's raw and persisted channels, not full XY or the whole ring band.
- Inspect the created scratch files: their logical size must match the computed
  fixed ring, never the full output stack.
- Compare raw sums, shared per-scale weights, normalized values, and final bytes
  through several wraps, including zero and `0 < weight < 1e-7` cases.
- Test crop-local Y/X allocation and global offset mapping explicitly.
- Test explicit close/unlink lifecycle on Ubuntu and macOS-compatible paths;
  correctness must not require hole punching, `flush`, or `fsync`.

### Shared-runner and compatibility tests

- Assert Lasagna and Fiber entry points resolve and call the same runner;
  assert `_infer_tiled_3d` no longer exists.
- Count model calls and prove each scheduled global tile is inferred once,
  including tiles contributing to multiple products/scales.
- Preserve exact weight-before-pyrdown and float32 accumulation order.
- Characterize and test sigmoid/clamp, autocast, normalization, output splitting,
  first-tile diagnostics, and NaN behavior.
- Compare exact persisted bytes and metadata against the old implementation on
  small fixtures for Lasagna dual-scale output and Fiber single/multi-option
  output.
- Compare a crop run with the corresponding chunks from a full-volume run.
- Test complete chunks, incomplete sibling bundles, missing input, restart
  after interruption, and independently missing external `pred_dt`.
- Test that a product complete at one scale is not accumulated merely because
  another scale still needs the model tile.
- Test mixed complete/incomplete chunks and different option masks inside one
  tile, with a shared per-scale weight ring and clean-run byte equivalence.
- Verify each geometric weight contribution is added once per scale—not once
  per product—and that a weight region is retained until its final incomplete
  product consumer finishes.
- Verify atomic output integrity and scratch cleanup after interruption during
  inference, ring flush, finalization, EDT, and pyramid creation.

### Representative run measurements

For the reported Fiber geometry/config, record:

- computed ring depth and per-ring backing-file size;
- process virtual size, observational RSS, and scratch-file logical/allocated
  sizes over time;
- first-tile and steady tile throughput;
- flush time per chunk and total flush overhead;
- wrap count and progress beyond the first Z row.

Acceptance requires no full-output-Z mmap, no full-XY or full-band flush
temporary, one inference per scheduled tile, bounded interrupt latency at chunk
boundaries, and the same persisted bytes as the characterized behavior.
RSS is reported for diagnosis but is not a deterministic pass/fail threshold.

## Spec update

Update `planning/specs.md` to replace the inaccurate "rolling z-band" language
with normative fixed-depth circular mmap behavior:

- backing shape and logical reservation are independent of full output Z;
- ring depth derives from live tile support and output-chunk alignment;
- the kernel manages mmap residency; no application RAM budget is imposed;
- completed data is finalized one output chunk at a time before slot reuse;
- products at one source-relative scale share one geometrically accumulated
  weight ring; resume masks affect raw products and weight-region liveness, not
  the denominator definition;
- scratch temporaries cannot scale with full XY/full Z/full channel bands;
- each model tile is inferred at most once;
- both CLIs use one runner with explicit source-relative inference scales and
  base-relative output scales;
- output-only resume, coherent sibling rewriting, crop byte identity, progress,
  interruption, and external `pred_dt` requirements remain normative.

Preserve all existing model/output compatibility requirements.

## Documentation updates

Update `docs/code_structure.md` with the sole-runner call graph, circular mmap
layout, logical/physical Z mapping, safe slot lifecycle, per-scale rings,
chunked wrap-aware flush, adapter boundaries, external `pred_dt`, progress, and
cleanup. Include a small data-flow diagram and backing-size example. Remove all
claims that the old full-Z mapping was rolling or that Lasagna/Fiber already
shared the complete engine.

## Changelog and task records

When implemented, add a dated `planning/changelog.md` entry. Replace
`planning/task_log.md` with decisions, deviations, commands, test results,
byte comparisons, ring/backing measurements, RSS observations, and throughput.
Keep `planning/status.md` current.

## Risks and non-goals

- The fixed ring can still have a large backing file because it spans full XY
  for the live Z window and all raw channels. This is intentional: storage is
  Z-bounded and mmap lets the kernel manage residency. Report its computed size
  before inference.
- Chunk-at-a-time flushing adds indexing and write overhead. Measure it, reuse
  chunk-sized buffers, and optimize only without changing accumulation order.
- OS/filesystem behavior for sparse allocation, page eviction, hole punching,
  and mmap cleanup differs. Correctness must use portable mmap/close/unlink
  behavior on Ubuntu and macOS; Linux-specific page advice may be an optional
  optimization but normal reuse must not require flush/fsync/hole punching.
- This task does not alter models, checkpoints, output resolution, prediction
  precision, normalization, or numerical semantics.
