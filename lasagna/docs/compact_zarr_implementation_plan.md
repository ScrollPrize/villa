# Compact prediction Zarr: implementation plan

Status: implemented, with CPU/native regression validation. Branch `las-compact-zarr`,
based on `origin/main` `e2254ebe0`. Initial format proposal: `d39a5abaa`.
The later user request explicitly deferred whole-volume mmap/flush layout
optimizations and added default Fiber invalid-channel blanking and uint8
base-voxel width export. Those supersede the original optimisation step below.

## Current implementation decisions

- New `compact_zarr.py` owns packed array creation, whole-chunk publication,
  resume checks, and mixed-policy pyramid generation. Logical products and raw
  sums remain separate until a single spatial publication combines them.
- Whole-volume mmap rings and raw-slab assembly are unchanged. Demand retains
  its existing packed chunk records/shared denominator; no second arena exists.
- Full-volume runners and manager launches default to compact; explicit legacy
  output and existing sidecar resume remain supported.
- Fiber defaults blank quantized-zero-presence locations and export annotated
  model width as rounded/clamped uint8 base voxels, with opt-outs. Float32
  auxiliary channels require lossless common-array float32 promotion.
- The native local tracing reader shares one decoded packed payload and exposes
  zero-copy channel views. Existing native interpolation is reused. Generic
  legacy 3D behavior remains unchanged.
- TTA and mixed-model reference views alias channel slots without splitting or
  copying packed data. New cache identities include format and export semantics.
- The prior invalid-width filtered-stop fix, still uncommitted in the training
  worktree rather than on main, was carried over because new invalid/blank
  samples must not crash width-aware tracing.
- The optional Lasagna prediction-distance stage is prepared before neural
  publication so its channel can join the same packed chunk without partial
  read/modify/write publication. Its staging array retains the existing EDT
  computation and is not an authoritative second output. Its resumable staging
  array is removed after successful packed publication and pyramid completion.

## Validation recorded so far

- 536 Python tests passed (five skips, two expected failures); legacy-default
  fixture assertions were updated to explicitly request legacy output.
- Native interpolation, packed payload sharing, base-width conversion, and
  invalid-width filtered-stop regression targets passed after rebuilding.
- CPU integration covers full-volume Fiber in both layouts, exact Lasagna
  channel/pyramid equality, demand publication and warm resume, native tracing,
  and TTA/mixed channel aliases. Opt-outs and resume-encoding checks are covered.
- Publication microbenchmark: one CPU caller, 64-cubed, three seeded
  random uint8 channels, Blosc-Zstd, three warmups + 20 measured writes.
  Legacy mean/p50/p95: 1.619/1.582/1.893 ms; compact: 1.227/1.112/1.665 ms
  (approximately 24% lower mean publication latency).
  A subsequent full-suite run measured legacy 1.582/1.583/1.630 ms and compact
  1.326/1.270/1.642 ms (16% lower mean; p95 essentially unchanged). Do not
  interpret the isolated mean improvement as an end-to-end speed guarantee.
  This is publication latency, not overall tracing throughput. GPU work and
  production training have not been used or modified.
- Validation uses `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests`
  with its local control file selecting this checkout, the compact pytest suite,
  existing inference/demand/provenance/manager suites, and native CTest targets
  `test_fiber_width_scoring`, `test_lasagna_normal_sampler`, and
  `test_chunked_plane_sampler`. The native build is Release, GCC 13. No GPU
  throughput claim is made; an end-to-end GPU benchmark remains outstanding.

The following sections retain the original code-based assessment and sequence.
The later request defers **both** mmap-layout changes and raw flush-copy
optimizations, including those identified below; neither was implemented.

## Target and constraints

Make [Lasagna schema v3](volume_schema_v3.md) the default for new Lasagna
normal and Fiber volume inference: one Zarr store, embedded manifest,
resolution-named groups, and chunks spanning all channels at that resolution.
Declare `grad_mag` or `presence` as the validity channel, respectively.

Retain legacy manifest/layout handling and resume of existing inference runs.
Do not convert an existing output in place, silently change encodings, alter
precision, reorder floating-point additions, or change inference sampling,
blend weights, resolutions, normal reconstruction, or pyramid reduction math.
No training restart or modification is needed for this work.

Subproject playbook: Lasagna is a Python/PyTorch inference and volumetric I/O
pipeline, with shared inference infrastructure consumed by Vesuvius Fiber
inference. Use the existing environment and pytest suites; no new dependency
is planned. CPU format, writer, resume, and accumulation tests come first.
Any real GPU smoke run uses the established approved launcher and manager
allocation, restricted to the GPUs authorized at that time.

## What the current code already does

- `tiled_predict3d.py` describes logical products separately from output
  channels, groups products by inference scale, and shares a blend-weight
  accumulator within each scale group.
- Whole-volume accumulation uses channel-bearing circular bands per product.
  Demand inference already has owner-local chunk records with shared weights
  and product/channel layouts, backed by `ChunkAccumulatorArena`.
- Normal products contain intermediate raw quantities whose count can differ
  from their final output-channel count. Their custom finalization is essential.
- `OmeZarrOutputAdapter` nevertheless checks and writes separate 3D channel
  stores. Pyramid generation and provenance inventory also assume that layout.
- Both inference CLIs and manager launchers currently construct `.lasagna.json`
  output paths. Those assumptions must change together with the default.

### Concrete accumulation assessment

The relevant paths have been inspected, including the native add kernel:

| Path | Current implementation | Consequence for this change |
|---|---|---|
| `_execute_accumulate_process_task` in `tiled_predict3d.py` | Adds the shared weight once, then loops over products and raw channels, invoking `_add_accumulator_view` per channel. | Channels are already processed under one spatial task/owner. Packing final output channels does not remove accumulation tasks or repeated weight updates: the latter are already shared. |
| `accumulator_add.cpp::add_inplace` | Accepts exactly 3D arrays, supports strided Z/Y and contiguous X, releases the GIL, and uses AVX-512/F16C where available. | Existing support cannot take a CZYX array in one call. A batched native API would need implementation, not merely changing an argument. It would save call/validation overhead, not additions or bandwidth. |
| `_CircularZBand` | Allocates a separate mmap per raw channel. | Concatenating those mappings just to invoke one add would introduce a copy. Retain this layout for now rather than proposing an unmeasured replacement. |
| `_read_mmap_band_chunk` and `_execute_flush_process_task` | Allocates/fills one float32 array per raw channel, then `np.stack` allocates and copies them into a second CZYX array. | There is a specific avoidable full-buffer copy at flush. Fill a preallocated CZYX buffer channel-by-channel through destination views instead. |
| `demand_predict3d.py::_publish_record` | Converts each raw channel to float32 separately and then stacks those arrays. The arena record already groups raw channels and shared weights. | Fill the same preallocated CZYX buffer directly from record views; no arena redesign is needed. |
| `OmeZarrOutputAdapter.write_product_chunk` | Finalizes logical products, then encodes/publishes each persisted channel separately. | Packed storage removes repeated encode/publication setup and output-file operations. This is the primary grouping benefit, separate from accumulation arithmetic. |

Both Lasagna's normal product and the standard Fiber direction/presence
product accumulate **seven raw channels** before producing three encoded
channels. Packing those three outputs cannot reduce seven raw sums to three:
the joint normal/direction finalizer requires those intermediate quantities.

For the default float16 raw accumulator and float32 input, each raw-channel
add logically reads 4 source bytes plus 2 destination bytes and writes 2
destination bytes per contributing voxel. Grouping does not reduce those
8 bytes or the shared float32 weight update. Cache behavior affects actual
memory traffic; this is a data-flow count, not a bandwidth measurement.

For seven raw channels, eliminating the extra float32 stack copy removes a
`7 * 4 * V`-byte temporary and approximately `2 * 7 * 4 * V` bytes of copy
read/write traffic at each flush, for `V` chunk voxels. At a 128-cubed chunk
that is a 56 MiB temporary and 112 MiB of copy traffic. This applies to the
raw-slab assembly, not every input-tile contribution, and does not eliminate
the finalizer's own allocations.

Expected result: worthwhile output/publication overhead reduction and a
concrete flush-copy reduction; **no basis for promising a large speedup of
the core accumulation loop just from grouping**. Do not rebuild the
accumulator architecture as part of the storage schema change.

## Implementation sequence

### 1. Manifest model and output-layout resolution

In `lasagna_volume.py`, add explicit v3 embedded serialization/validation and
loading of root metadata while retaining v1/v2 sidecar loading unchanged.
Keep one shared manifest representation; do not duplicate all field handling.
Validate group paths, C-axis order, base shape/scales, and validity-channel
existence. Preserve extension fields and auxiliary references.

Separate schema-version selection from the legacy save method that currently
unconditionally writes version 2. New output is compact by default; an explicit
legacy format option can retain v2 output. Existing outputs select their
recorded layout for resume; conflicting requested layouts fail without writes.
Do not infer a format solely from an ambiguous directory suffix.

### 2. Shared compact output writer and restart-safe publication

In `tiled_predict3d.py`, add a shared resolution-group layout mapping from
logical product channels to packed CZYX channel slices. Keep product
finalization separate from storage layout. Create `sd<N>/<L>` arrays with
full-C chunks and subgroup OME metadata; embed the root manifest.

Publish one encoded chunk only after all contributing products at that spatial
location have been finalized. Use the existing atomic chunk-publication
mechanism, generalized to 4D chunk keys. Do not independently read/modify/write
channel slices of one compressed chunk from multiple workers.

Completion/resume checks refer to the packed chunk, not the presence of one
product's old channel files. Preserve explicit completed-zero-chunk behavior
so zero fill is not confused with unfinished work. Validate shape, channel
order, dtype, codec, and layout before reusing output. Invalidate dependent
coarser chunks consistently when a finest-level chunk is replaced.

### 3. Packed pyramid generation

Generalize the existing `omezarr_pyramid.py` / shared tiled-inference helpers
to access channel views inside packed chunks. Reuse scalar reduction,
`grad_mag` zero-overrides, and joint normal second-moment reduction.
Assemble all output channels before publishing each destination chunk once.
Do not replace mixed channel semantics with a generic mean over CZYX.

Keep each native-resolution group's channel set constant throughout its
pyramid; do not merge the coarse copies of a higher-resolution group into a
different group. Retain the existing number and absolute indices of levels.

### 4. Inference entrypoints and manager integration

Update `preprocess_cos_omezarr.py` and
`vesuvius/.../fiber_trace_3d/infer.py` to use the common compact writer by
default, passing the correct validity-channel name. Cover whole-volume and
demand output paths without forking the numerical inference implementations.

Update `manager/runs.py` to launch new compact outputs and record the store
root as the manifest location, while retaining old run-record interpretation.
Update `inference_provenance.py` to inventory the embedded root manifest,
resolution groups, channel order, and CZYX metadata without walking data chunks.
Keep model identifier, snapshot hash, and source-volume provenance intact.
Place generated auxiliary artifacts and scratch paths deliberately relative
to the output/run directory; rebase references rather than dropping them.

Audit manager bundle validation/upload and catalogue ingestion for sidecar-only
assumptions. Do not mislabel v3 as v2 or publish an unreadable artifact as a
legacy channel volume. If the external Atlas consumer needs separate support,
report that dependency explicitly before enabling compact publication.

### 5. Deferred accumulation efficiency work (not implemented)

Keep the existing spatial task ownership, shared weight, channel accumulation
order, raw-channel count, accumulator dtype, mmap rings, and demand arena.
There is no proposed regrouping of the raw accumulation buffers.

A later optimization could add a destination-buffer form of the existing mmap chunk reader. Allocate
one float32 `(raw_channels, z, y, x)` slab in the flush path and fill its
channel views directly, retaining zero initialization for clipped/missing
regions. Do the equivalent conversion directly into slab views during demand
record publication. Reuse shared slab-assembly behavior rather than copying
the algorithm into both publishers. Preserve normalization and finalization
order exactly. This removes the identified `np.stack` copy without touching
the hot accumulation layout.

Benchmark add, raw-slab assembly, finalization, and packed output publication
separately. A batched native add API is **not** part of the initial change:
the current API is 3D-only, and the code inspection provides no evidence that
its call overhead dominates the existing native memory loop. Revisit only if
profiling actually identifies that overhead. Never concatenate per-channel
buffers just to reduce the number of calls.

## Verification and acceptance

- Round-trip v3 metadata, required validity declaration, unknown extensions,
  relative references, multiple scales, and one-channel CZYX groups.
- Legacy v1/v2 reader behavior and existing-run resume remain unchanged.
- Deterministic synthetic tiles through both old and new storage layouts:
  decoded arrays, blend sums, normal results, sparse coverage, all pyramid
  levels, and original-volume coordinate mapping agree. Include odd extents.
- Interrupted publication, absent/explicit-zero chunks, concurrent flushes,
  crop resume, incompatible output rejection, and mixed logical products in
  one packed chunk. No partial-channel chunk may be considered complete.
- Exercise normal validity through `grad_mag` and Fiber validity through
  `presence`; no hardcoded channel-index assumption.
- Extend existing tests in `lasagna/tests` (live/demand tiled inference,
  compression, pyramids, provenance, manager) and the Vesuvius Fiber inference
  tests; add focused compact-format tests where existing fixtures are 3D-only.
- Use the existing preapproved test/experiment launchers. CPU tests need no
  GPU reservation; any CUDA integration test obtains manager resources first.
- Benchmark identical input/model/tiles, precision, worker counts, codec, and
  spatial chunk sizes for legacy versus compact: accumulation time, encode/write
  time, files/requests, total bytes, RAM/scratch peaks, and clean/resume latency.
  Record commands, input, iterations, mean, and median/p95. Do not promise an
  end-to-end speedup before measuring.

## Deferred work and rollout limitation

The local accelerated tracing reader is included because demand tracing needs
it. Remote catalogue/publication consumers, VC3D display, and Fiberlet discovery
are separate integration work. New v3 stores cannot be used by those old
consumers until they are updated. The explicit legacy output option remains
the compatibility route during that transition. No existing published volume
is rewritten.
