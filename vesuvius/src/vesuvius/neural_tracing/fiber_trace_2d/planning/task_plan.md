# Plan: Fiber Scale-2 Output, Sparse Accumulator Activity, and 64³ Chunks

## Current behavior and terminology

- The requested control is Fiber whole-volume inference's
  `--inference-scaledown-power`, not the separate `scaledown` stored in the
  tracer/model config. The config value is unrelated and must not be renamed,
  reinterpreted, defaulted, or migrated by this task.
- `--inference-scaledown-power=2` means factor `2**2 == 4`, or `0.25x` linear
  output resolution relative to the selected inference input array. The shared
  runner needs that literal factor for tensor pyrdown and ring geometry, so
  convert power to factor at the Fiber whole-volume inference boundary; do not
  store an exponent in `OutputProductSpec.inference_scaledown`.
- Lasagna currently uses literal factors (`cos_scaledown=2`,
  `scaledown=4`). Preserve those meanings. Only its OME chunk default changes.
- Fiber and Lasagna CLI/function defaults are currently `ome_chunk=32` even
  though one adapter constructor happens to default to 64. The caller-provided
  value wins, so defaults must be made consistent at every public boundary.
- The shared flush currently revisits every chunk in a completed full-XY Z
  range. It separately guesses support with `_chunk_supported`, then clears
  raw and weight mmap regions unconditionally. This writes zeros into untouched
  sparse mmap pages and is the observed first-flush stall.

## 1. Default Fiber `--inference-scaledown-power` to 2

Define the whole-volume inference control as a non-negative power:

```text
inference_scaledown_power = 2                  # default
inference_factor = 1 << inference_scaledown_power  # 4
effective_base_factor = input_base_factor * inference_factor
```

Use the factor for crop/output geometry, `_pyrdown3d`,
`OutputProductSpec.inference_scaledown`, and base-relative manifest metadata.
Use the effective base factor to derive the OME level and array shapes. Log
both unambiguously, for example `inference_scaledown_power=2 inference_factor=4
effective_base_factor=16`.

The numerical downscale operation is the shared Lasagna `_pyrdown3d` pyramid
operation, not direct `::4` sampling or interpolation. Power 2 performs two
successive separable 5-tap low-pass blur plus 2x decimation stages, preserving
the existing kernel, reflect-padding, sampling phase, float32 operation order,
and prediction-times-blend-before-pyrdown behavior for both raw products and
their geometric weights. Fiber and Lasagna must call the same implementation.
This is characterization of the pre-unification Lasagna implementation as
well: it already used the same `[1,4,6,4,1]/16` separable kernel and operation
order, so this task must reuse it rather than introduce a new blur policy.
Fiber inference must call this exact shared path with no Fiber-local filter,
border crop, resampling, or weight construction. Given identical prediction
tensors, tile geometry, scale factor, and crop phase, Lasagna and Fiber must
produce identical downscaled weighted accumulations and denominators.

Preserve the historical tile-border semantics and document their interaction
with pyramid filtering precisely. `border` first zeroes the blend weight on
every face of the full-resolution model tile, so predictions made inside that
raw border never enter the numerator. The configured `overlap` is total
input-tile overlap, and before filtering the nominal blend-ramp width remaining
after both neighboring zeroed borders is:

```text
nominal_blend_width = max(0, overlap - 2 * border)
```

For downscaled outputs, however, that is not the final support. The runner
multiplies prediction by the full-resolution weight, then filters both the
weighted numerator and weight denominator with `_pyrdown3d`. One pyramid step
has radius 2 in input voxels. Power `p` has effective input-space radius
`2 * (2**p - 1)`; power 2 therefore has radius 6 and a 13-voxel effective
kernel. Positive denominator support can extend by up to six input voxels into
the nominally zero border. The raw border predictions remain excluded, but
downscaled samples in that fringe are reconstructed from retained core
predictions and normalized by the identically filtered weight.

Consequently, do not claim that `overlap < 2*border` necessarily creates an
uncovered downscaled gap, and do not derive coverage from nominal widths alone.
Validate the actual filtered weight lattice and sampling phase. For
`border=32`, `overlap=96`, the pre-filter nominal blend width is 32 and the
filtered transition is wider/smoother. For `border=32`, `overlap=64`, the
pre-filter cores merely meet, but the factor-4 Gaussian still spreads weight
across their boundary. This historical behavior must remain unchanged unless
a separate future task requests a strict post-filter output border.

At physical volume edges, read-side reflect padding places real edge voxels
after the padded border so they retain core coverage. `_pyrdown3d` also uses
reflect padding at the tile tensor edge, but with a border wider than the
filter radius that reflected area is already zero-weight. Do not crop a
separately scaled border by integer division.

Keep an explicit `--inference-scaledown-power 0` full-resolution override.
Reject negative powers and impractically large shifts before geometry
allocation. Use the same explicit name in the Python API. Do not read or alter
the tracer/model config's `scaledown`, and do not change Lasagna's
factor-valued arguments.

Validate divisibility requirements after conversion. With a factor of 4,
Fiber tile size, stride, and border must satisfy the shared runner's alignment
rules; error messages should name both the power and derived factor.

Validate the selected input's scale from all three axes, allowing the normal
ceil-divided pyramid edge shapes. Require the effective base factor to be an
exact isotropic power of two before assigning an OME level; do not use a
Z-only rounded ratio or rounded `log2` to manufacture metadata.

## 2. Track actual accumulator activity lazily

Replace flush-time full-grid support discovery with an activity ledger built
while scheduling and accumulating real work.

For each source-relative scale and output chunk origin, track:

- whether its exact mapped source/output footprint is supported by locally
  present input Zarr chunks;
- which incomplete products still need it;
- whether raw sums and the shared geometric weight actually received a
  contribution in the current logical generation.

Compute/cache source support lazily when a model tile intersects an incomplete
output chunk. Do not pre-scan the complete volume. The support test must use
the exact source-space footprint represented by that output chunk, clipped to
the input/crop, rather than asking whether some much larger overlapping model
tile contains any input somewhere. This prevents a central scroll chunk from
making unrelated masked edge chunks appear supported. Preserve missing-chunk
semantics; do not read/decompress every input voxel merely to predict sparsity.

This direct-footprint rule is an intentional sparse-output contract change at
masked boundaries: an output chunk with no stored input in its own footprint
remains absent even if a neighboring stored chunk lies inside the model or
pyrdown filter halo and could numerically influence it. Specify the global
output-to-selected-input coordinate phase exactly for every power, crop, and
odd edge. The support footprint intentionally excludes model and Gaussian
filter halos. Compatibility claims apply only to chunks supported under both
the old and new policies; tests must compare output chunk-key sets as well as
bytes.

The initial implementation may detect sparsity from absent local Zarr-v2 chunk
keys (`.` or `/` dimension separators). A physically stored all-zero chunk is
still considered present; do not claim value-mask discovery. Document and
fail clearly for unsupported stores/layouts rather than silently treating them
as empty.

Only schedule a model tile if at least one incomplete, supported output chunk
needs it. Run that tile once, then:

- add raw values only to its supported, incomplete product chunks;
- add the shared weight once over the union of those product regions;
- mark a chunk dirty only when a nonzero blend-weight region was actually
  added;
- make no mmap write for unsupported or already-complete chunks.

The dirty/activity ledger, not a second `_chunk_supported` scan during flush,
is authoritative. Represent each `(scale, logical generation, chunk origin)`
with one `weight_dirty` state and a set of `dirty_products`. This keeps resumed
products independent while adding geometric weight once over their union. The
generation key prevents circular slot reuse from confusing old and new chunks.

## 3. Flush, release, and resume sparse regions correctly

At each chunk-aligned Z frontier, iterate the active/dirty chunks for that
frontier rather than the full XY chunk grid.

For a dirty chunk:

1. confirm at least one incomplete product remains;
2. read the chunk-sized shared weight and verify it contains positive support;
3. normalize/finalize/write only products that received raw contributions;
4. atomically write the coherent channel bundle;
5. clear only the raw and weight regions that were dirtied;
6. remove its activity record and release its generation.

For an unsupported, untouched, resumed-complete, or zero-weight chunk:

- do not read accumulator storage;
- do not normalize or finalize;
- do not create output Zarr chunk files;
- do not assign zero into mmap storage;
- advance/release frontier bookkeeping without materializing sparse pages.

New scratch files start logically zero. A physical circular region that was
previously dirty must be cleared after its last consumer and before reuse;
untouched regions need no clearing. Retain generation assertions so a dirty
region can never be overwritten. Shared weights remain live until all dirty
product consumers for that chunk complete.

Prove safe reuse at XY-chunk granularity even though the ring currently tracks
whole-Z-plane generations: every previously dirty rectangle must be cleared
before reassignment, while disjoint untouched rectangles remain unmaterialized.

Resume remains output-chunk-only. Existing coherent product chunks suppress
new raw/weight contributions and are never rewritten. Partially present
sibling bundles remain incomplete and are coherently regenerated. Empty
regions have no output chunk and are rediscovered as unsupported on a later
run; absence is not confused with a durable completion marker.

## 4. Default Lasagna and Fiber OME chunks to 64³

Set `ome_chunk=64` consistently in:

- Fiber inference CLI and Python entry point;
- Fiber adapter construction defaults;
- Lasagna `predict3d` CLI and `run_preprocess_3d` entry point;
- shared group/pyramid creation call sites whose default is user-visible;
- examples and tests that assert defaults.

Keep `--ome-chunk` as an override. Do not alter unrelated 2D preprocessing
chunk flags (`--chunk-z`, `--chunk-yx`, EDT work chunks) unless they directly
represent these output OME-Zarr chunks.

Because chunk size changes output storage layout and flush alignment, verify
metadata, crop rounding, final edge chunks, resume checks, pyramid invalidation,
and ring-depth planning at 64. Persisted voxel values must remain unchanged;
only Fiber's requested default output scale and both pipelines' default chunk
layout change.

## 5. Shared ownership and observability

Implement activity tracking, support mapping, dirty accumulation, sparse flush,
clearing, resume, and counters once in
`lasagna.tiled_predict3d.run_tiled_inference_3d`. Neither Fiber nor Lasagna may
grow a private accumulator or flush loop.

Extend the common progress output with:

- inferred and skipped model tiles;
- dirty/active chunks;
- chunks written, resume-skipped, and unsupported-skipped;
- dirty bytes touched/cleared versus logical ring backing size;
- per-flush chunk progress and elapsed time.

Skipped tiles and sparse flushes must update progress. A long flush must never
sit indefinitely at only `flush z=[...]` without chunk counters.
Count unsupported chunks uniquely per scale/generation; distinguish absent
input, all-output-complete, and no-supported-target tile skips so ETA and
diagnostics are meaningful.

## Implementation sequence

1. Characterize the current whole-volume inference scale behavior, separately
   assert that tracer/model config `scaledown` is untouched, and characterize
   current 32³ defaults.
2. Wire/test `--inference-scaledown-power`, convert it to a literal runner
   factor, and make power 2 the default.
3. Add a pure, cached output-chunk-to-source-support mapper with crop/edge and
   sparse-input fixtures.
4. Add per-scale, per-generation activity/dirty records and route tile
   scheduling plus accumulation through them.
5. Replace full-XY flush iteration and `_chunk_supported` rescans with active
   chunk iteration; clear only dirty regions and release untouched generations
   without mmap writes.
6. Change Lasagna and Fiber OME defaults to 64 at all public boundaries.
7. Add shared progress counters/timing for sparse scheduling and flushes.
8. Run compatibility, sparse-file allocation, resume, crop, wrap, and
   representative-volume measurements, including exception/KeyboardInterrupt
   cleanup; then update specs/docs/changelog/status and the task log.

## Testing and acceptance criteria

### Scale and chunk defaults

- Fiber default `--inference-scaledown-power=2` produces source-relative factor
  4, shapes equal to ceil-divide-by-4, and correct effective base
  scale/OME level/manifest.
- Fiber power overrides 0, 1, and 3 produce factors 1, 2, and 8.
- Power 2 output and blend weights exactly match two successive shared
  blur-plus-decimate stages and differ from naive stride-4 subsampling on a
  high-frequency fixture.
- Impulse/ramp fixtures verify the 5-tap kernel, reflect edge behavior, global
  sampling phase, and identical raw-product/weight downscale geometry.
- A cross-adapter fixture feeds identical synthetic predictions through
  Lasagna and Fiber adapters and asserts identical weighted numerators,
  denominators, border support, and normalized values from the shared runner.
- Border fixtures verify zero raw-prediction contribution from tile faces,
  physical-edge reflect padding, nominal
  `blend_width=overlap-2*border`, effective Gaussian support radius 2/6/14 for
  powers 1/2/3, and denominator coverage at the exact sampling phase for
  `border=32` with total overlaps 64 and 96.
- Changing inference output power does not read, mutate, or reinterpret the
  separate tracer/model config `scaledown` value.
- Odd ceil-divided pyramid edges validate correctly; anisotropic or
  non-power-of-two input/base relationships fail before output creation.
- Lasagna factor-valued `cos_scaledown`/`scaledown` behavior is unchanged.
- Both inference CLIs and Python entry points default to `ome_chunk=64`; an
  explicit value such as 32 still works.
- Created `.zarray` metadata uses `(64,64,64)` chunks at the prediction level,
  including clipped array edges, and pyramids/resume use the same geometry.

### Sparse activity and output

- A synthetic central island of input chunks surrounded by absent chunks
  creates output chunks only for supported regions.
- A model tile overlapping supported and unsupported output chunks updates
  only the supported chunk records and mmap regions.
- Boundary fixtures prove the intentional direct-footprint policy when a
  neighboring present chunk lies inside the model/filter halo of an absent
  output footprint; compare both chunk keys and supported-chunk bytes.
- Support mapping covers powers 0/1/2/3, nonzero crops, odd edges, global
  sampling phase, border padding, and both Zarr-v2 dimension separators.
- Stored all-zero chunks count as present while absent chunk keys count as
  unsupported; unsupported store types fail clearly.
- Fully skipped Z/Y/X areas cause zero accumulator writes, zero clearing, and
  zero output writes while frontiers continue advancing.
- Spy/instrument mmap assignments and prove untouched regions are never
  assigned zero during flush/release.
- Dirty chunks are cleared exactly once after their final product consumer;
  wrapped generations cannot observe stale values.
- Disjoint and overlapping XY chunks across two logical generations sharing
  physical Z slots show no stale leakage; inspect physical allocation through
  `st_blocks`, not only logical `st_size`.
- Zero blend-weight intersections remain absent even if bookkeeping considered
  the chunk; positive weights below `1e-7` retain existing normalization
  semantics.
- Resume with complete, incomplete-sibling, unsupported, and mixed-product
  chunks remains byte-identical to a clean run for every written chunk.
- Model call counts prove one inference per needed global tile and no calls for
  tiles serving only unsupported or complete output chunks.
- Exceptions and KeyboardInterrupt remove disposable scratch files through
  `finally`; no scratch state is required for resume.

### Representative validation

On the reported masked Fiber volume, record the exact command, input scale,
crop, tile/overlap/border, output level/factor, and 64³ chunking. Report:

- old versus new scheduled/inferred/skipped tiles;
- logical ring size versus physically allocated scratch bytes;
- dirty versus unsupported output chunk counts;
- first-flush and steady flush duration;
- total wall time and peak RSS;
- output chunk count and total output bytes.

Use two correctness references: the new default must match an explicit power-2
run, and supported chunks must match old explicit factor-4 inference. Record
chunk-key sets separately because deliberately omitted masked-boundary chunks
are not byte-compatible with the old support policy.

Acceptance requires output at 0.25x by default, no output chunks for unsupported
regions, no mmap writes/clears for untouched regions, bounded visible flush
progress, and unchanged bytes for supported chunks relative to an equivalent
explicit-scale reference run.

## Spec update

Update `planning/specs.md` to specify:

- Fiber whole-volume inference's `--inference-scaledown-power` defaults to 2;
  the shared runner's `inference_scaledown` remains a literal source-relative
  factor, and tracer/model config `scaledown` remains separate and unchanged.
- Inference downscaling uses repeated shared 5-tap low-pass-blur plus 2x
  decimation stages; direct striding or independent Fiber filtering is not
  allowed.
- Lasagna's existing scaledown arguments remain literal factors.
- Lasagna predict3d and Fiber inference default to 64³ OME-Zarr chunks.
- accumulator/output activity is contribution-driven and lazy;
- unsupported/untouched chunks produce neither output files nor scratch mmap
  writes, including zero-clearing;
- only dirty regions are normalized, finalized, cleared, and released;
- progress includes active/written/unsupported/resume-skipped chunks and flush
  timing.

Remove the current statement that unsupported output chunks have their slot
regions cleared; untouched regions must instead be released without touching
their mmap pages.

## Docs updates

Update `docs/code_structure.md` with:

- the Fiber inference-power-to-factor-to-effective-base-scale mapping;
- the repeated blur-plus-decimate kernel, phase, padding, and why power 2 is
  filtered 0.25x output rather than direct stride-4 sampling;
- examples for default power 2 and explicit full-resolution power 0;
- an explicit warning that tracer/model config `scaledown` is a separate
  setting;
- the lazy source-support cache and per-generation dirty chunk ledger;
- dirty-only flush/release lifecycle and why sparse mmap pages remain sparse;
- the 64³ defaults and override behavior;
- new progress fields and interpretation.

Update Fiber/Lasagna user-facing inference examples or README sections that
need `--inference-scaledown-power` or still describe 32³ default chunks.

## Changelog and task log

After implementation, add a changelog entry covering Fiber's new default scale,
64³ output chunks, and contribution-driven sparse accumulator flushing. Record
all validation commands, measurements, deviations, and unsupported cases in
`planning/task_log.md`.
