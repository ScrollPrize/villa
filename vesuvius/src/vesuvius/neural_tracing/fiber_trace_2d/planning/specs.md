# 2D Fiber Trace Initial Loader Specs

## Managed whole-volume Fiberlet inputs

- `las_manager fiberlet run <completed-fiber-run>` resolves the unique regular
  Lasagna normal representation for the same sample and source volume from the
  public catalogue. Regular normals require `grad_mag`, `nx`, and `ny` from the
  exact Atlas model schema; Fiber `presence`/`nx`/`ny` outputs are not normals.
- Explicit `atlas:<model-id>@L<source-level>` and completed local Lasagna-run
  overrides remain supported. Fiber and normal source levels may differ. Both
  must be uncropped, have numeric source-to-base mappings, and identify base
  shapes differing by at most one voxel per axis.
- Published normal manifests use the exact VC3D open-data cache contract under
  `<cache_dir>/open_data/lasagna/<sample>/<volume>/<identity>/`. The VC identity
  is derived from the canonical remote artifact URL and Atlas coordinate/model
  metadata, never from the manifest content SHA. Native access is a persistent
  lazy chunk cache shared with VC3D; manager completion is cache-only and
  performs no network access. Resume binds the exact manifest URL and SHA-256
  recorded at initial launch rather than repeating automatic selection.
- Published dependency identity uses the exact remote artifact/manifest URL,
  Atlas model, and level. Manifest SHA-256 is an integrity check only; it is not
  synthesized into a run UUID, source locator, or cache address.

## Shared 3D Tiled Inference

- `lasagna.tiled_predict3d.run_tiled_inference_3d` is the sole neural tiled
  inference runner used by both Lasagna predict3d and Fiber 3D inference. It owns:
  canonical global tile/output-chunk lattices, crop bounds, S3 auto-download
  from `_download` metadata, circular Z scratch, output-chunk-only resume,
  temp cleanup, progress formatting, and atomic Zarr chunk writes.
- Product-specific adapters own model semantics, raw output splitting, output
  channel schema, tile preprocessing, raw-to-persisted finalization, and
  completeness semantics. Shared predict3d helpers own OME-Zarr group
  creation, Lasagna manifest writing, standard scalar/normal pyramid building,
  and chunk writing/resume mechanics.
- `preprocess_cos_omezarr.py predict3d` remains the compatibility wrapper for
  Lasagna cos/normal inference. Its CLI, output values, `.lasagna.json`
  manifest, scale handling, optional `pred_dt`, and OME-Zarr pyramid behavior
  must remain compatible with the pre-extraction implementation.
- Resume state is durable output chunks only. Done markers are not allowed.
  Scratch mmap/temporary files are not resume state and may be deleted on
  startup/resume or finish.
- Neural accumulation uses a fixed-depth circular mmap per raw product,
  float32 by default and optionally float16 for memory-constrained experiments, and one
  float32 geometric weight mmap per distinct source-relative inference scale.
  Flush reads widen bounded product chunks to float32 before normalization and
  finalization. FP16 assumes finite, model-bounded raw products; unbounded
  custom adapters must select float32 to avoid overflow.
  Ring depth is derived from the actual canonical Z tile positions, nonzero
  tile support, flush opportunities, and output chunk alignment. Mmap shape and
  logical file size must be independent of full output Z.
- Ring planning keeps the chunk-aligned `flushed` frontier separate from the
  physical ring origin and follows runtime write-before-post-row-flush order.
  The initial prefix from logical plane zero remains live when a computed
  frontier merely equals `output_begin`; only a strictly advancing runtime
  flush releases it.
- Flush overlaps inference through the same circular mmap, enlarged only for
  the exact maximum span produced by one frozen finalized interval plus the
  following active canonical Z row. There is one runner-wide flush future,
  including across inference scales. No live overlap, finalized band, or full
  mmap region is copied to another mmap or a band-sized RAM buffer.
- Submitted/frozen, completed/written, and released/origin frontiers are
  distinct. A combined chunk-aligned, possibly empty, multi-scale flush batch
  has immutable plans and disjoint queued, inflight, and completed task IDs.
  The coordinator drains acknowledgements continuously and, immediately after
  the complete batch succeeds, clears its exact dirty rectangles, releases its
  generations, and advances user-visible finalized Z. Before submitting the
  next combined interval it waits only if the preceding batch is still active.
- The flush worker receives immutable chunk descriptors and reads frozen mmap
  regions one output chunk at a time. Temporary denominator, stacked raw, and
  finalized channel arrays remain bounded by one output chunk. A failed flush
  never clears or reuses its frozen slots, and all exit paths wait for the
  non-cancellable reader thread before mmap cleanup.
- Completed output is normalized, finalized, written, and cleared one globally
  anchored output chunk at a time. Denominator and wrap scratch are bounded by
  one output chunk; no full-XY or full-band normalization/finalization
  temporary is allowed. Circular slots may be reused only after every live raw
  product region sharing their geometric weight has finished.
- Resume masks suppress accumulation for already complete product chunks.
  Weight contributions are accumulated once over the union of incomplete
  product regions at a scale, never once per product. Each scheduled global
  model tile is inferred at most once even when it feeds several scales or
  products.
- `OutputProductSpec.scaledown` is base-relative output metadata. The
  internal product `inference_scaledown` value is only runner geometry state
  for tile downsampling and ring layout; it is not serialized into Lasagna
  manifests.
- Fiber inference manifests must not add redundant trace-scale aliases such as
  `trace_to_base_scale`, `prediction_to_base_scale`,
  `prediction_spacing_in_trace_voxels`, `inference_scaledown_factor`, or
  per-group `inference_scaledown`. Native consumers require an explicit
  numeric manifest `source_to_base` and derive persisted prediction sample
  scale as `source_to_base * 2**group.scaledown`. Native precomputed tracing
  receives the missing inference-output scaledown relative to trace coordinates
  as `--inference-scaledown-power` (default `2`), then derives
  `trace_to_base = prediction_to_base / 2**power` and
  `prediction_spacing_in_trace_voxels = 2**power`.
- Native precomputed Trace2CP search must match the Python tracer's beam
  semantics: pruning is ordered by `cumulative_loss + depth * 1e-12` with
  original tensor/generation order preserved on ties, spatial pruning uses the
  squared-distance `>= distance**2` keep rule, and reached-target selection
  chooses the first reached state with minimum cumulative loss only. Native
  compact normal interpolation must choose the principal tensor axis with a
  symmetric eigensolver and then apply the same hint/no-hint sign convention as
  Python. The active `candidate_substeps=1` candidate loss is the Python
  all-pairs direction product plus the configured smoothness terms.
- Native precomputed Trace2CP may score independent beam candidates in
  parallel only when the prediction source and, if present, the Lasagna normal
  sampler explicitly advertise concurrent sampling support. Parallel scoring
  must build candidate tasks in deterministic beam/candidate order, keep
  persisted Zarr/cache access chunky by preparing interpolation requests as a
  batch, and rebuild the next frontier serially in original candidate order,
  so pruning, reached-state selection, and trace output remain deterministic.
  Persisted sources may decode and score each candidate directly while its
  pinned corners are hot, provided scores are written at their original global
  candidate indices. Static scoring ranges may be submitted as one indexed
  worker batch to avoid per-range futures, but every index must run exactly
  once and worker exceptions must be rethrown after batch completion.
  Candidate task metadata and point coordinates may use separate compact
  arrays, provided their shared index remains the original deterministic
  beam/candidate order. `--threads 0` is the default and uses the available
  worker pool; `--threads 1` must force serial candidate scoring.
- Native precomputed Trace2CP persisted sampling must use one long-lived VC3D
  decoded chunk cache per physical scalar Zarr volume. Each candidate batch
  must fetch the ordered eight integer voxel corners through blocking
  requested-level nearest-neighbor coordinate sampling, with dependencies
  deduplicated and chunks pinned before candidate access. The tracer must use
  the shared corner visitor to interpolate scalar channels and decode/score
  compact channels without candidate-sized corner arrays. The materializing
  corner API must use that same visitor rather than duplicate cache/layout
  behavior. Compact `nx/ny` corners must be decoded as paired ambiguous axes
  and interpolated through the weighted orientation tensor; independently
  interpolating encoded `nx` and `ny` is not allowed. Candidate points sharing
  one integer voxel cube must reuse one gathered ordered-corner record per
  physical scalar volume; per-point fractions and callback indices remain
  distinct. Concrete single-point prediction sampling must resolve each cube's
  unique chunk keys once rather than probing the shared chunk cache per corner.
- Native precomputed Trace2CP final lookahead orders intermediate parents by
  nonnegative cumulative-loss lower bound and original parent index. With
  `--lookahead-parent-cap 0`, it must expand parents lazily until the next lower
  bound is strictly greater than the established reached loss or complete
  spatial-beam threshold; equal bounds must remain observable, producing the
  same result as exhaustive expansion. The production default intentionally
  caps this ordered expansion at 32 parents. This is an accepted approximate
  search-semantic change measured at 7 restarts on the representative
  87-segment workload. Original global child indices remain required for ties.
  `--exhaustive-lookahead` bypasses both lazy stopping and the parent cap.
  `--lookahead-retry-parent-cap` is an explicit deterministic failed-segment
  retry cap; `0` disables it and remains the default. A retry result replaces
  the original only when it succeeds.
- Native fiber-trace internal geometry, direction, interpolation, beam, and
  loss math may use float. Public persisted coordinates may remain double at
  API boundaries. Candidate generation order, pruning tie order, and output
  determinism remain required, and performance changes must report the
  representative whole-fiber restart metric so numeric changes cannot silently
  degrade trace quality.
- Multi-GPU tiled inference exists only in the shared runner and is therefore
  identical for Lasagna predict3d and Fiber 3D inference. It uses one
  persistent spawn-context model worker per selected CUDA device, without DDP
  or GPU collectives.
- The canonical tile stream materializes only independent Z/Y/X axis lattices
  (O(Z+Y+X)), never their Cartesian product. Multi-device execution owns one
  explicit Z band at a time and lazily generates that band's Y/X traversal into
  fixed bounded descriptor/read windows; it never retains a whole band. Reads,
  GPU execution, and accumulation do not cross the band completion barrier.
- Input reads happen outside GPU workers. The default local Zarr-v2 backend is
  one asynchronously polled TensorStore driver/context created only after all
  spawned GPU/flush workers start. `python-zarr` is an explicit fallback.
  Resume/skipped work is rejected before read submission. The lazy prefetch
  window is `prefetch_tiles_per_gpu * selected_device_count` (CPU/single-device
  counts as one) and is independent of GPU shared-memory slots.
- Every outstanding TensorStore read may own a full padded input tile. Input
  memory is bounded by that window times tile bytes, plus the separately bounded
  TensorStore cache, existing input/result shared memory, and request/cache
  overhead. A ready tile enters shared memory only when input/result slots and a
  GPU queue are available. Existing clipped bounds, source dtype, uint16
  conversion timing, fully-outside uint8 behavior, and NumPy reflect-padding
  semantics are preserved exactly.
- Input slots cannot be reused until H2D completion, and results cannot be
  published until D2H completion. The coordinator alone unlinks shared memory;
  workers only attach and close it.
- CUDA input transfer preserves compact uint8/uint16 source dtype. UInt16 is
  converted through int32 floor division by 257; normalization and adapter
  preprocessing are CUDA FP32. CPU fallback preserves historical NumPy
  conversion. Fiber model autocast follows checkpoint training-policy metadata
  by default, with explicit precision override and all-device validation;
  shared product arithmetic, filtering, D2H, and accumulation remain FP32.
- Opt-in multi-device pipeline profiling uses bounded streaming aggregates,
  preserves the disabled worker/message path, distinguishes summed concurrent
  service from wall span, and directly reports reader throughput and effective
  outstanding-request concurrency, queue delays, CPU conversion,
  CUDA/transfer/model/output, and commit stages.
- GPU results may finish out of order, but accumulator task submission remains
  in canonical tile order and is nonblocking under worker-queue pressure. Each
  shared input/result slot has exactly one `(band, sequence, stage)` owner and
  must be returned before the Z-band transition. A band completes only after
  all of its lazy events, including skips, commit and all reader/GPU/accumulator
  work is acknowledged. The coordinator solely owns scheduling, circular-ring
  frontiers, resume state, and progress; flush workers execute distinct
  coordinator-planned output chunk writes.
- GPU admission computes the effective accumulation frontier after its
  contiguous `done_skip` prefix. While that frontier still needs GPU work, one
  input/result slot pair is reserved for it; later ready tiles may consume only
  the remaining pairs. This prevents later out-of-order GPU results from owning
  every result slot and blocking the canonical tile required to release them,
  without changing canonical accumulator submission or numerical order.
- After draining every completion source, unresolved state with no tracked live
  reader, GPU, accumulator, or flush producer is an immediate invariant error
  with band/event/slot diagnostics. Legitimate tracked work is waited on at its
  actual stage; no generic GPU poll or elapsed-time watchdog diagnoses it as a
  process-limit failure.
- Output adapters must permit completeness checks for future, disjoint chunks
  while the flush worker writes a frozen chunk. Model adapters must permit
  inference to overlap their stateless raw-product finalization callback.
- Sparse/resume-complete work is rejected before prefetch. Workers calculate
  only the union of raw products required by incomplete output chunks; the
  coordinator retains chunk masks and adds shared geometric weight once.
- Worker exceptions, hard exits, CUDA failures, interrupts, and coordinator
  errors cancel prefetch, stop workers, and close shared resources rather than
  waiting indefinitely.
- `--devices all` selects all visible CUDA devices and a comma-separated list
  selects a subset. Existing singular `--device` and CPU behavior remain
  supported; conflicting or invalid selections fail before model construction.
- Automatic OME-Zarr download uses `--download-workers` independently of
  inference prefetch, GPU slots, and pyramid workers. It defaults to 64 and
  must be positive even when automatic download is disabled.
- Fiber and Lasagna expose the same opt-in `--live-fetch` selected-level disk
  cache through the shared runner. It is valid only for full, non-cropped
  inference on a local numeric OME-Zarr-v2 level backed by S3 `_download`
  metadata, and conflicts with `--no-download`. The disk target defaults to
  10240 GiB (10 TiB), the materialization window defaults to 10,000 canonical
  tile descriptors, and `--download-workers` controls raw chunk transfers.
  This descriptor window is independent of the smaller TensorStore full-tile
  read window and must not materialize a global Cartesian job list.
- When live fetch is disabled, supported-tile discovery and bounded input-read
  submission are one atomic scheduler transition. Live fetch adds only the
  upstream materialization state; completed live descriptors enter the same
  bounded reader/GPU/accumulator/flush stages and canonical commit order.
- GPU and accumulator descriptor queues remain bounded. The coordinator tracks
  the exact sequence/task owner for every queued worker item and the exact
  owner/stage of every shared input/result slot; mismatched, foreign, duplicate,
  or dead-worker ownership is a fatal invariant error.
- Live source support is authoritative per active remote Z-chunk inventory;
  transient local absence and advisory `.noremote` data cannot suppress valid
  inference. Completed output work is rejected before remote materialization.
  Downloads use unique temporary files and atomic replacement, and terminal
  list/GET/write failure is fatal rather than reclassified as masked fill.
- Live cache accounting and deletion apply only to valid chunks in the exact
  selected scale. On each completely committed canonical model Z row, the
  runner advances a safe input frontier. If actual completed resident bytes
  exceed the target, it removes oldest whole cached Z-chunk planes whose ends
  are at or before that frontier, repeating until under target or no safe plane
  remains. It never evicts by Y/X, LRU, or ahead of inference; insufficient
  obsolete data permits a reported temporary overshoot.
- Cooperating ordinary inference readers hold a shared selected-level advisory
  lock, while live mutation and bulk download hold the exclusive lock. Lock
  files live below `.dl_cache`; non-cooperating external readers are outside
  this protection. Live mode initially supports only the primary input source;
  separately remote Lasagna `pred_dt` is rejected while local `pred_dt` remains
  supported.
- `.dl_cache/<level>.noremote.json` is advisory only. Missing, unreadable,
  malformed, or schema-invalid cache data warns and behaves as an empty set;
  it must never abort inference or suppress remote validation. Saves snapshot
  Stats under lock, include empty sets, and use same-directory unique temporary
  files plus atomic replace. Save failures warn, retain the previous valid
  target when possible, clean temporary files, and do not fail the download.
- Fiber whole-volume inference's `--inference-scaledown-power` defaults to 2
  (factor 4 relative to selected input). It is converted to the runner's
  literal factor and does not read or reinterpret tracer config `scaledown`.
- Model tensor downsampling retains floor-sized interpolation geometry, but
  persisted OME-Zarr level shapes and exclusive output-region endpoints use
  ceil division. Odd selected-input dimensions must therefore write their
  final valid output plane and output chunk rather than leaving the ceil-sized
  OME edge unwritten.
- Scaled output uses the shared repeated separable `[1,4,6,4,1]/16`
  blur-plus-2x-decimation path for weighted predictions and weights. Fiber has
  no private resampling, blending, or border implementation.
- Lasagna predict3d and Fiber inference default to 64x64x64 OME-Zarr chunks.
- `las_manager` Bash and Zsh completion is generated from the same command
  registry used for prefix dispatch. Dynamic snapshot and catalog candidates
  use cached indexes only; inference candidates read durable records; live-run
  candidates may query tmux but never reconcile or mutate records.
- A manager-launched inference is complete only when the child exits zero and
  its portable `artifacts/inference.json` reports `completed` with an artifact
  inventory. A zero exit without that contract is recorded as failed with a
  diagnostic completion error.
- Portable artifact inventory paths must be relative, remain inside the bundle,
  and resolve after the complete `artifacts/` directory is moved. This bounded
  validation does not recursively enumerate Zarr chunks.
- Pyramid multiprocessing may use the automatic available-CPU process count,
  but every pyramid worker must run native BLAS/OpenMP libraries with one
  thread. The same constraint applies to serial pyramid execution, and parent
  environment/native limits must be restored on success and failure.
- Accumulator activity is contribution-driven. Unsupported, resume-complete,
  and untouched chunks produce neither output chunks nor mmap zero-writes.
  Only dirty product and shared-weight regions are flushed and cleared before
  circular-slot reuse.
- Sparse source support is evaluated on the global output lattice: each global
  output-chunk footprint is clipped to the product's full output shape and only
  then mapped into selected-input coordinates. Crop-local padded ring
  dimensions must never clip this global footprint. Products sharing an
  inference-scale accumulator must share the same full output shape.
- Output products are independently resumable. For Lasagna, missing `pred_dt`
  chunks schedule only derived distance-transform generation; they must not
  schedule neural model inference when `cos` and `grad_mag/nx/ny` chunks are
  complete. Missing one sibling of `grad_mag/nx/ny` makes only the coarse
  normal bundle incomplete.
- Output chunks and model tile origins are anchored to a global full-volume
  lattice. A crop only selects which global output chunks to produce; it must
  not shift the tile support used for a shared chunk. Overlapping or separate
  crop runs therefore produce the same bytes for the same complete global
  output chunk.
- Every output chunk write uses a unique temporary path on the target
  filesystem followed by atomic `os.replace`. If any channel in a coherent
  product is missing, the product is incomplete and the next run rewrites the
  missing product chunk through the same atomic path.
- Fiber 3D inference is exposed by
  `python -m vesuvius.neural_tracing.fiber_trace_3d.infer`. It uses the shared
  tiled runner with the common arguments `--input`, `--output`,
  `--checkpoint`, `--tile-size`, `--overlap`, `--border`, `--scaledown`,
  `--crop`, `--device`, `--no-download`, `--levels`, and `--ome-chunk`.
  It also accepts `--pyramid-workers` to pass worker count to the shared
  pyramid builders.
  Fiber-specific arguments are the positional training/inference config,
  `--recurrent-steps`, `--base-ref`, and `--base-scale`.
- Fiber 3D inference `--output` is a `.lasagna.json` manifest path. The
  manifest is the authoritative output description and points to per-channel
  OME-Zarr groups derived from the manifest stem.
- Fiber 3D inference must use the existing 3D fiber model/config/checkpoint
  stack: `build_fiber_trace_3d_model(...)`, training snapshot loading, the
  configured tile image normalization, mixed-precision/autocast helpers, and
  Lasagna 3x2 direction encoding helpers.
- When a 3D fiber inference/tracing checkpoint contains a saved training
  `config`, that checkpoint config is authoritative for model construction,
  option count, and tile preprocessing. Runtime configs still provide the
  dataset/CLI context, but must not silently build a different architecture
  than the snapshot. Older checkpoints without embedded config may infer a
  minimal legacy free-branch model layout from
  `net.decoder.final_seg_layer.weight` when possible.
- Lasagna 3x2 normal estimation and compact `nx/ny` byte encoding live in the
  package-safe shared `lasagna.normal_encoding` module. Lasagna predict3d,
  fiber whole-volume inference, and live fiber prediction paths must import
  that helper directly, not private functions from
  `preprocess_cos_omezarr.py`.
- Fiber model output has seven raw channels per option internally:
  `dir0_z`, `dir1_z`, `dir0_y`, `dir1_y`, `dir0_x`, `dir1_x`, and
  `presence`. These raw channels are accumulated in the shared circular Z ring
  and are never persisted as output channels.
- Fiber persisted output is only `presence`, `nx`, and `ny` per option.
  Presence is fixed-point uint8 with `0 == 0.0` and `255 == 1.0`. `nx/ny` use
  Lasagna's compact ambiguous hemisphere encoding: estimate the 3D axis from
  raw 3x2 direction channels, flip the equivalent sign to `z >= 0`, then write
  `round(component * 127 + 128)` clipped to uint8.
- Product completeness for a fiber option requires all three persisted sibling
  chunks: `presence`, `nx`, and `ny`.
- Multi-branch legacy outputs and conditioned recurrent outputs must be
  preserved as separate coherent fiber options. Inference must not collapse
  them to branch 0, min/max, average, or any other summary unless a separate
  explicit postprocessing mode is added.
- Fiber inference must build coarser OME-Zarr pyramids: scalar mean-pool
  pyramids for `presence` and paired normal pyramids for `nx/ny`, using the
  existing Lasagna pyramid helpers.
- Fiber inference must not keep legacy V0 output compatibility shims: no
  `fiber_trace_3d_inference.json`, no raw seven-channel persisted bundle, no
  directory-style `--output`, no duplicate fiber output adapter, and no public
  exports for removed V0 symbols.
- Shared multi-device Fiber/Lasagna inference accumulates output chunks in
  persistent spawned CPU processes. A stable integer mapping gives each
  `(scale, chunk_z, chunk_y, chunk_x)` exactly one FIFO owner, so overlapping
  tile updates need no locks and retain canonical per-chunk order. GPU result
  slots remain live until every referenced accumulation task acknowledges.
- Accumulation queues are bounded and a new Z row cannot reserve circular-ring
  generations until the preceding row is committed. Flush frontiers therefore
  observe only acknowledged tasks and retain the rolling-memory bound.
- Product rings default to float16 while the shared weight ring remains
  float32. Each product update widens the stored half to float32, adds the
  float32 tile contribution, and rounds to nearest-even back to binary16.
  This reduced-precision accumulation is explicitly not bitwise equivalent to
  float32; `--product-accumulator-dtype float32` restores float32 accumulation.
- The optional native accumulator extension must retain a portable scalar
  implementation. On supported x86 GCC/Clang builds it may runtime-dispatch to
  an isolated AVX-512F+F16C kernel; the package must not require AVX-512
  globally and unsupported CPUs/platforms must continue through the fallback.

## 3D CP-Centered Fiber Model Variant

- The 3D CP model lives in a sibling package,
  `vesuvius.neural_tracing.fiber_trace_3d`. It must not replace or reinterpret
  `fiber_trace_2d` configs, strip geometry, Trace2CP tooling, or 2D training.
- 3D training samples ordinary CP-centered ZYX volume blocks from the selected
  base-volume Zarr level. It must not build fiber-aligned 3D strips or slices
  for input loading.
- Dataset entries use `base_volume_path`, `base_volume_scale`, and
  `fiber_paths`/`fiber_glob`. `lasagna_manifest_path` is optional for 3D
  training; when omitted, the loader derives `base_shape_zyx` from the raw
  OME-Zarr volume and validates the selected scale against the configured
  `base_volume_scale`.
- JSON and NML fiber parsing, control-point exactness, and optional
  dataset-level XYZ affine transforms follow the existing
  `fiber_trace_2d.fiber_json` semantics.
- `base_volume_scale` selects both the Zarr level read by the 3D loader and the
  voxel scale at which CP-centered patches are sampled.
- The 3D sample stream is deterministic pseudo-random by configured `seed` and
  covers every configured control point once per pass before repeating. Changing
  batch size or step count may truncate/extend the consumed prefix, but must not
  reshuffle earlier samples.
- 3D training uses a strict stream/data index split. `stream_index` is the
  unbounded deterministic stream position; `data_index` is the bounded
  dataset-selection index after applying `training.max_sample_index` /
  `sample_index_limit`. `training.max_sample_index` limits CP/data sample
  selection only. Every deterministic random source and augmentation parameter
  must be keyed by `stream_index`, never `data_index`, so reusing a bounded
  CP/data prefix cannot replay the same augmentation transforms on each repeat.
- Public 3D loader compatibility calls may still accept an argument named
  `sample_index`, but it is semantically `stream_index` and must be normalized
  to that name internally. Batch/sample data structures must carry explicit
  `stream_index`/`stream_indices` and `data_index`/`data_indices` fields.
  `data_index` is only for dataset lookup/CP selection and debug reporting.
- With `training.max_steps = 0`, 3D training repeats the deterministic training
  stream indefinitely until interrupted. Positive `max_steps` values are
  absolute target training steps, including for resumed runs.
- 3D geometric augmentation is represented by explicit coordinate maps before
  the final volume patch is materialized. `backward_source_zyx` maps output
  voxels to selected-level source-volume coordinates for image sampling.
  Source fiber points are mapped to output-patch coordinates with the matching
  analytic forward transform built from the same augmentation parameters. The
  image, transformed fiber line, transformed CP, and direction targets must
  therefore see the same geometry without dense inverse search.
- V0 3D geometric augmentations support CP-local shift, isotropic scale,
  arbitrary 3D rotation, and independent axis flips. Non-zero 3D shear/skew and
  ringing artifact keys are rejected until their semantics are specified.
- 3D smooth displacement is opt-in through
  `augment_smooth_displacement_mode` (`none`, `1d`, `2d`, `3d`),
  `augment_smooth_displacement_amplitude_zyx`,
  `augment_smooth_displacement_control_spacing_zyx`, and
  `augment_smooth_displacement_probability`. Smooth modes must use explicit
  paired map construction, matching the 2D fused-map contract. Runtime paths
  must not invert one map direction into the other by search, brute-force
  nearest lookup, iterative solving, or formula re-evaluation. The current 3D
  mode uses explicitly invertible 1D/2D offsets and 3D triangular coupling
  stages.
- The 3D loader must sample the final regular 3D patch through explicit
  coordinates using the VC3D blocking coordinate sampler. It must not load an
  oversized axis-aligned zarr crop and then resample that crop with torch
  `grid_sample` for normal training. Array-backed tests may use the
  NumPy/trilinear fallback sampler. Value-only augmentations happen after
  sampling as torch tensor operations.
- VC3D blocking coordinate sampling means strict requested-level sampling:
  every required requested-level chunk is fetched/decoded and locally pinned
  before sampling starts, scale fallback is disabled, and returned stats report
  `requested_level_only: true`, `fallback_levels: 0`, and `missing_chunks` for
  genuinely absent requested-level chunks. Only those truly missing
  requested-level chunks may render black. Chunk I/O/decode errors must fail
  loudly. The returned sampler `valid_mask` is only geometry/sample coverage;
  it must not be treated as proof that requested-level data was used.
- V0 3D value augmentations support normalization, brightness, contrast, gamma,
  noise, and separable isotropic Gaussian blur. Opt-in anisotropic blur is
  configured with `augment_anisotropic_blur_probability`,
  `augment_anisotropic_blur_sigma_along`,
  `augment_anisotropic_blur_sigma_across`,
  `augment_anisotropic_blur_orientation`, and
  `augment_anisotropic_blur_roll_degrees`. It is a value augmentation after
  coordinate sampling, not a geometric transform.
- The active 3D multi-direction experiment uses
  `model_3d.conditioned_decoder_enabled: true`. A shared spatial 3D U-Net
  emits a latent feature volume whose width is
  `model_3d.conditioned_latent_channels` and defaults to `64`, independently
  of `unet_base_channels`. A separate conditioned decoder receives the latent
  channels plus a six-channel Lasagna 3x2 query direction at each voxel and
  emits seven sigmoid channels: six direction channels and one sheet/fiber
  presence channel.
- The conditioned decoder head must be pointwise only: `1x1x1` convolutions or
  equivalent per-voxel linear layers. It must not add spatial 3D kernels after
  the shared U-Net latent.
- The all-zero six-channel query is a reserved off-manifold unconditioned
  token. It must not be decoded or interpreted as a real direction.
- `FiberTrace3DNet.forward(volume)` remains a zero-query compatibility path
  returning `B,7,D,H,W`. `forward_recurrent_grouped(volume, steps=2)` returns
  branch-shaped grouped outputs where the first seven channels are zero-query
  output and the next seven channels are decoded using the first prediction's
  encoded direction as the query.
- Legacy/free-branch configs remain supported when
  `conditioned_decoder_enabled` is false. In that mode the output layout is
  grouped by direction branch, each branch has seven channels, and branch 0
  preserves the legacy channel positions (`0:6` direction, `6` presence).
- Each seven-channel prediction's six direction channels use Lasagna's double-angle projection layout:
  `dir0_z,dir1_z` for `(tx,ty)`, `dir0_y,dir1_y` for `(tx,tz)`, and
  `dir0_x,dir1_x` for `(ty,tz)`.
- Direction supervision is computed from the transformed 3D line tangent and is
  masked to positive fiber-neighborhood voxels. Projection-magnitude weighting
  may downweight channels whose projection is nearly degenerate.
- In conditioned mode, sparse positive supervision evaluates two positive
  queries per supervised point with equal weight: the zero/unconditioned query,
  and one deterministic query sampled from the plane perpendicular to the GT
  direction with configurable jitter
  (`training.conditioned_perpendicular_jitter_degrees`, default `45.0`). Both
  positive queries predict positive presence and the same GT direction.
- Conditioned positive-query randomness is deterministic from `stream_index`
  and the local sparse point coordinate, not from loader order or global RNG.
- In conditioned mode, dense negative presence supervision decodes both the
  zero query and one deterministic random direction query per patch over all
  `presence_mask` locations, including positive pixels by design. The weak
  dense negative BCE at positive pixels is intentional; under the configured
  positive/negative component weights it is equivalent to a softened positive
  target, not contradictory masked supervision.
- Conditioned presence BCE is normalized by patch/query group before
  aggregation. The group-normalized positive and negative components are
  multiplied by `training.conditioned_positive_query_weight` and
  `training.conditioned_negative_query_weight`, both defaulting to `1.0`.
- Legacy/free-branch positive supervision still chooses one branch per sparse
  positive point by detached
  `argmax(abs(dot(decoded_predicted_axis, target_axis)) * predicted_presence)`.
  Legacy two-branch training keeps the deterministic `2x2x2` anti-collapse
  repair and legacy global-negative branch BCE semantics for old configs.
- 3D training defaults `training.direction_weight` to `10.0` and
  `training.presence_weight` to `1.0`, so direction loss is 10x stronger than
  presence loss unless the config overrides it.
- 3D target generation is source-format dependent. NML fibers use dense
  supervision along all fiber-line segments that overlap the patch. The
  transformed output-space segments are clipped/rasterized directly into the
  patch target volume; they must not be generated by a full voxel-by-segment
  nearest search or by inverting sampled image coordinates. Non-NML fiber
  sources supervise only the sampled CP neighborhood for direction and
  presence.
- The first 3D model used branch-routed direction plus presence losses; the
  active multi-direction experiment uses the conditioned query loss above.
  Contrastive embedding remains unsupported by default in the 3D V0 path.
- The 3D fiber model defaults to `BatchNorm3d`; configured `batch_size` is the
  actual BatchNorm batch because the trainer has no internal micro-batching.
  `model_3d.normalization: "none"` remains supported for explicit ablations.
- `batch_size` is the actual CP-patch batch passed through the 3D model in one
  forward/backward call. The 3D trainer does not support internal
  micro-batching; any BatchNorm statistics must come from the real configured
  batch.
- `training.mixed_precision` controls trainer autocast only and must not
  introduce internal micro-batching. Supported modes are `off`, `bf16`, `fp16`,
  and `auto`. BF16 uses autocast without a scaler; FP16 uses AMP with
  `GradScaler` and snapshots include scaler state when present. Dense test
  loss, benchmark forward loss, TensorBoard sample-sheet inference, and
  Trace2CP metric/visual inference use the same configured autocast mode.
- The active S1A 3D configs use `training.mixed_precision: "bf16"` to reduce
  activation memory while preserving the configured BatchNorm batch.
- The S1A NML 3D training config uses `patch_shape_zyx: [192,192,192]`,
  `augment_shift_zyx: [48,48,48]`, and a fixed six-stage U-Net depth
  (`[16,32,64,128,256,512]`) so the deepest feature map remains appropriate
  for 192-voxel patches.
- `train_s1a_nml_all_64_sd2.json` and
  `train_s1a_nml_all_128_sd2.json` are experimental S1A NML configs at
  `base_volume_scale: 2` for 64- and 128-voxel patches. The 64 sd2 config uses
  the active conditioned decoder path. The 128 sd2 config is currently a
  regular single-output legacy config with one seven-channel branch
  (`direction_branch_count: 1`, `output_channels: 7`) for direct comparison
  against the conditioned experiment. Both keep the same implemented
  augmentation families enabled at magnitudes appropriate for their patch
  sizes: affine shift/rotation/scale/flip, value
  brightness/contrast/gamma/noise, isotropic blur, smooth displacement, and
  anisotropic blur. Shear/skew and ringing remain unsupported and must not
  appear as enabled keys in these configs.
- 3D training TensorBoard visualization logs CP-centered slice sheets at
  `training.sample_vis_interval`. By default, up to four batch samples are
  shown; `training.sample_vis_count` / `train_sample_vis_count` and
  `training.test_sample_vis_count` control the side-by-side train/test sample
  counts. Each sample block has five rows: the `yx`, `zx`, and `zy` principal
  planes, a longitudinal slice containing the GT CP tangent, and a
  perpendicular/cross slice whose plane normal is the GT CP tangent.
  Single-output rows have five columns: volume image with projected GT line and
  model-predicted/fitted CP direction overlay where applicable,
  target/context presence, raw predicted presence, predicted presence weighted
  by `abs(dot(pred_axis, slice_normal))`, and predicted presence weighted by
  `abs(dot(pred_axis, GT_tangent))`. Multi-output rows keep the branch summary
  layout: image, target/context presence, first prediction presence, second
  prediction presence, prediction presence for the output whose decoded
  direction is closer to the slice normal by `abs(dot(axis, normal))`, the other
  prediction presence, max prediction presence, min prediction presence, and
  average prediction presence. In conditioned mode the first prediction is the
  zero-query output and the second prediction is the recurrent output
  conditioned on the first decoded direction; in legacy branch mode these are
  branch outputs. The target/context
  presence panel must visualize the carried transformed fiber-line segment
  metadata even for JSON/non-NML CP-only samples where loss supervision remains
  CP-only. The two oblique rows must project/rasterize transformed line
  segments into their oblique slice frame for both image overlay and
  target/context presence. Dense-line/NML samples must carry the transformed CP
  tangent so the GT-tangent and perpendicular rows are constructed from the
  actual local target tangent. The GT line overlay includes target-line
  portions within 2 voxels of the displayed principal slice plane or oblique
  slice plane. The sparse direction angular-error panel
  is intentionally not shown because it is too sparse to be useful for routine
  inspection. The predicted/fitted CP direction overlay is drawn as a thin
  anti-aliased line whose length is scaled by the in-slice projection magnitude,
  so out-of-slice directions are visibly shorter.
- The 3D target-presence panel in TensorBoard is display-only max-pooled with a
  `3x3x3` kernel before slicing. This must not modify `presence_target` used by
  training or test loss.
- 3D training/test loss logging reports average supervised prediction direction
  angular error in degrees as `train/angle_mean_deg` and
  `test/angle_mean_deg`. The scalar is computed over sparse supervised
  direction samples with Lasagna 3x2 analytic decoding and unoriented
  `abs(dot)` agreement. Legacy branch routing diagnostics include branch usage
  fractions and selected score means; conditioned mode reports fixed equal
  query fractions for its two positive query groups and the mean
  `abs(dot) * presence` over those positive query outputs.
- Conditioned 3D presence loss uses equal default component weight when
  positive-query and dense negative-query supervision are both present:
  `mean(positive-query BCE) + mean(dense negative-query BCE)`, multiplied by
  the configured conditioned positive/negative component weights. Legacy branch
  mode keeps its existing selected-positive plus branch-negative BCE contract.
- When `training.test_interval > 0`, 3D training runs the configured test
  evaluation at step 0 before the first optimizer step and logs the same
  TensorBoard scalars/stdout as interval tests.
- 3D snapshots are evaluation-only. `training.checkpoint_interval` must be a
  positive multiple of `training.test_interval`, and
  `training.kept_snapshot_interval` must be `0` or a multiple of
  `training.test_interval`. `current.pt` and retained numbered snapshots are
  written only on aligned evaluation steps; an otherwise unscheduled final
  training step must not write a snapshot.
- The 3D `best.pt` snapshot is selected exclusively by the lowest observed
  dense `test/loss_total`. Training loss must never be compared with the test
  metric or trigger a best snapshot.
- Dense 3D test loaders do not inherit train augmentations by default.
  `training.test_augment_enabled: true` is the explicit opt-in for augmented
  dense tests.
- Dense 3D tests default to evaluating every configured held-out CP once in the
  deterministic pseudo-random test stream from sample index zero.
  `training.test_control_points: 0` is the explicit full-test sentinel with
  the same behavior. Positive values keep the fixed deterministic random test
  range beginning at `test_start_sample_index`.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.train` is the 3D training
  entrypoint. It supports normal training, `--benchmark`, `--load-only`, and
  `--prefetch`.
- Multi-process 3D training is enabled only by the standard `torchrun`
  environment (`WORLD_SIZE > 1` with `RANK` and `LOCAL_RANK`); no DDP config
  keys are required. A typical launch is
  `torchrun --standalone --nproc_per_node=N -m vesuvius.neural_tracing.fiber_trace_3d.train <config.json>`.
  `--benchmark`, `--prefetch`, and `--trace2cp-vis` are single-process-only
  modes and must fail clearly when launched with `WORLD_SIZE > 1`.
- In DDP training, configured `batch_size` remains the per-rank local batch size.
  The effective global optimizer-step batch is `batch_size * WORLD_SIZE`.
  Training samples are partitioned by rank as disjoint deterministic stream
  batches, and `training.max_steps` remains the number of optimizer steps.
- CUDA DDP training must convert `BatchNorm3d` modules to `SyncBatchNorm`
  before DDP wrapping so BatchNorm statistics are computed across ranks.
  Ordinary single-process training keeps literal `BatchNorm3d` modules.
  Checkpoints are saved by rank 0 from the unwrapped model so snapshot keys do
  not receive a DDP `module.` prefix.
- DDP side effects are rank-0-only: TensorBoard, stdout progress, checkpoints,
  Trace2CP metrics, and train/test visualization. Dense test model evaluation
  is deterministically distributed across all ranks; scalar training losses
  are averaged across ranks before rank-0 logging.
- Dense DDP tests partition global batch IDs as `rank, rank + WORLD_SIZE, ...`.
  The configured test start is a literal sample offset, including when it is
  not batch-aligned; every global batch is evaluated exactly once and only the
  final global batch is sliced. Per-batch float metric rows are gathered to
  rank 0, restored to global batch order, and combined with the historical
  unweighted Python per-batch mean so metric and best-snapshot semantics remain
  unchanged. Ranks with no assigned batch still enter the gather.
- Each rank retains a separate persistent test DataLoader worker pool using the
  same worker count, prefetch factor, worker device, and multiprocessing context
  as training. Rank 0 reuses its already evaluated global batch zero for the
  test sample sheet instead of synchronously loading it again. The train and
  test pools coexist and are both released before distributed teardown.
- Normal 3D training also supports `--resume <snapshot.pt>`. The CLI path
  overrides config resume keys, restores model and optimizer state, writes a
  fresh timestamped run directory, and records the effective resume path in
  TensorBoard config text. After restoring checkpoint optimizer state, training
  must reapply the current config optimizer hyperparameters supported by the
  trainer, currently `training.learning_rate` and `training.weight_decay`, to
  every optimizer param group while preserving loaded AdamW moment buffers and
  step counters. If finite `training.max_steps` is not greater than the
  checkpoint step, training must fail clearly.
- 3D training and `--benchmark --load-only` runtime loading use
  `torch.utils.data.DataLoader` worker processes when
  `training.loader_workers > 0`. Each DataLoader item is one complete
  `FiberTrace3DBatch`, not an individual CP patch, and PyTorch default
  collation is bypassed so the custom dataclass is not nested or reshaped.
- Each 3D DataLoader worker lazily constructs its own `FiberTrace3DLoader` and
  VC3D sampler state in the worker process. Worker outputs are CPU
  `FiberTrace3DBatch` objects; the main training process transfers the whole
  batch to `training.device` immediately before forward/backward. The old
  thread-backed `_OrderedBatchLoadPipeline` is not a supported 3D loading path.
- In 3D configs, omitted or `null` `volume_cache_memory_mib` means a
  Python-side default of 512 MiB per VC3D sampler/loader/worker, not VC3D's
  internal 8 GiB default. Explicit positive values override this cap. The
  generated 2D Trace2CP geometry loader used by 3D evaluation receives the same
  default when the 3D raw config leaves the key unset or `null`.
- 3D DataLoader workers must not materialize full dense direction/presence
  target tensors. Worker batches carry image/valid tensors plus compact target
  descriptors: CP-only samples carry local CP/tangent metadata plus
  visualization-only transformed line segments, and NML dense-line samples carry
  transformed output-space line segments with precomputed patch bboxes for
  supervision. Dense `presence_target` and `presence_mask` are created by
  `fiber_trace_3d.targets.materialize_targets(...)` in the main training process
  on the training device. Direction supervision is represented sparsely as
  `direction_indices_bzyx`, `direction_target_sparse`, and
  `direction_weight_sparse`; normal training must gather predictions at those
  supervised line/CP voxels instead of creating full-patch dense six-channel
  direction targets.
- For JSON/non-NML 3D samples, `target_segment_*` metadata is visualization
  context only. The materializer must filter dense line rasterization by
  `_TARGET_MODE_DENSE_LINE`, so CP-only JSON segments do not create dense
  presence or direction supervision. Their direction target is the transformed
  CP tangent applied only to the CP neighborhood. TensorBoard visualization may
  draw those visualization-only segments in the target/context presence panel,
  but that display-only raster must not be fed back into loss materialization.
- The GPU target materializer must preserve the existing label semantics:
  NML sources supervise direction/presence by drawing the overlapping clipped
  fiber centerline voxels only, without a radius-expanded distance-to-segment
  tube. Non-NML sources supervise only the sampled CP neighborhood using
  `presence_radius_voxels`; that radius does not apply to NML centerline
  targets. Presence edge masking applies only to CP-only samples; NML dense-line
  samples supervise presence over the full valid patch. Lasagna 3x2 direction
  encoding uses the shared NumPy/torch-compatible helper semantics.
- `training.loader_workers` controls 3D DataLoader worker process count.
  Under DDP this count is per rank. `0` is the explicit serial/debug path.
  `training.loader_prefetch_factor` maps directly to PyTorch DataLoader prefetch
  factor for worker processes.
  `training.loader_worker_device` defaults to `"cpu"`. CPU worker processes
  use a guarded `forkserver` multiprocessing context where available, falling
  back to `fork` only when needed; CUDA worker devices select `spawn`.
- 3D `--benchmark --load-only` timing output separates main-process
  `wait_ms` from `to_device_ms`. It also reports worker-side profiling columns
  for loader construction, descriptor lookup, augmentation parameters,
  geometry-map creation, coordinate conversion, valid-mask generation, VC3D
  sampling, tensor conversion, value augmentation, compact target-spec
  generation, batch stacking, worker wall time, and worker CPU time. Dense
  target work is reported separately as main-process GPU target materialization
  timings (`target_ms`, `gpu_ms`, `line_idx`, `cp_idx`, `scatter`, `dir_enc`,
  `gpu_mask`, `linePts`, `dirPts`, and `posK`). With worker processes, the
  first `loader_workers` benchmark rows can include worker-local loader
  construction and should not be used as steady-state throughput.
- 3D prefetch computes chunk dependencies from a CP-centered selected-level
  augmentation-envelope bbox and asks VC3D to convert that bbox to authoritative
  chunk dependency metadata. It follows the 2D step-count
  sentinel rules: omitted `--prefetch-steps` uses `training.max_steps`;
  positive values override config; explicit `--prefetch-steps 0` means every
  selected training CP once; negative values fail clearly. A positive
  `training.max_sample_index` bounds the prefetched training prefix, and
  full/config-driven prefetch also covers held-out test CPs once in flat order
  when `test_datasets` is configured.
- VC3D dependency collection exposes both coordinate-surface metadata
  (`collect_coords_dependencies`) for 2D strip/top-slice surfaces and selected
  ZYX bbox metadata (`collect_bbox_dependencies`) for regular 3D prefetch
  envelopes. Python must preserve VC3D-returned metadata rather than
  reconstructing cache paths or remote chunk keys.
- 3D prefetch must follow the same streaming dependency/download state machine
  as 2D prefetch: bounded dependency producers controlled by
  `prefetch_sampler_workers`, bounded download workers controlled by
  `prefetch_workers`, deterministic raw-sample-order producer consumption,
  global chunk de-duplication, cache-hit / `.empty` classification before
  downloads, earliest-raw-sample download priority, safe-prefix `idx`
  tracking, live dependency and download progress, sample skip accounting,
  fatal cancellation of queued futures, temporary PyTorch CPU intra-op thread
  pinning, and the shared Python atomic download helper.
- The only intentional 3D differences from 2D prefetch are that one 3D sample
  produces one CP-centered 3D augmentation-envelope dependency volume, valid
  counts are voxels, and there is no strip-z offset loop or top-view branch.
- The 3D augmentation-envelope dependency volume is
  augmentation-sample-independent, not augmentation-config-independent:
  configured augmentation extrema define the conservative source range, but
  one deterministic random augmentation draw must not decide which chunks are
  prefetched.
- 3D prefetch dependency generation must not call `_sample_augment_params` or
  otherwise sample concrete augmentation parameters. It must generate
  a selected-level bbox from the configured envelope and call VC3D bbox
  dependency discovery without `sample_coords`, coordinate materialization,
  image decoding, normalization, or target construction.
- V0 3D prefetch uses VC3D chunk dependency metadata and the shared Python
  prefetch writer with atomic cache-file renames and `.empty` marker handling.
  It does not prefetch Lasagna manifest channels.
- The 3D-to-2D evaluation bridge in `fiber_trace_3d.trace2cp_bridge` samples
  dense 3D model outputs at explicit 2D Trace2CP strip coordinates, projects
  six-channel Lasagna 3x2 direction predictions into the requested 2D strip
  frame, carries presence through, and reuses the existing 2D Trace2CP scorer.
  This bridge is metric/debug tooling only and does not change 3D input
  loading into strip loading.
- 3D Trace2CP projection must decode the six Lasagna 3x2 direction channels
  analytically: each two-channel projection is decoded with
  `theta = atan2(sin2theta, cos2theta) / 2`, then the three projection planes
  are reconstructed/sign-aligned with the Lasagna three-plane logic. Unit-sphere
  candidate tables, binned direction lookup, or grid-search decoding are not
  allowed for 3D Trace2CP projection.
- 3D training test evaluation may reuse the 2D `FiberStrip2DLoader` only to
  construct Trace2CP segment geometry. It must keep normal 3D training samples
  as CP-centered volume blocks. For Trace2CP evaluation, dense 3D inference is
  run over tiled axis-aligned blocks covering the requested 2D strip
  coordinates plus configured context, then sampled/projected back to 2D.
- Hang diagnostics are disabled by default. Setting the JSON boolean
  `training.test_hang_diagnostics_enabled: true` or passing
  `--test-hang-diagnostics` during normal training enables append-only per-rank
  diagnostic logs, `SIGUSR2` manual dumps, detailed test phase markers, and a
  rank-0 pre-NCCL-timeout watchdog. The watchdog defaults to 480 seconds through
  `training.test_watchdog_seconds`, which must be positive and below the
  600-second process-group timeout when diagnostics are enabled. Disabled mode
  creates no files or handlers, arms no timer, polls no resources, and performs
  no diagnostic CUDA synchronization. The CLI flag is invalid in auxiliary
  prefetch, benchmark, and Trace2CP visualization modes.
- Rank 0 prints `test_timing step=... total_seconds=...` for the complete test
  routine through distributed dense evaluation, visualization, and Trace2CP,
  excluding the subsequent TensorBoard flush, and logs the same duration as
  `timing/test_total_seconds`. Rank-0-only post-dense phases must remain below
  the process-group timeout while other ranks wait at the result broadcast.
- When a 3D config defines `test_datasets` and `test_trace2cp_enabled` is
  false, test evaluation runs ordinary 3D sparse direction/presence loss on the
  held-out CP-centered 3D samples. It must not require Trace2CP geometry or
  trace loss.
- Configured dense 3D tests log `test_sample_3d/principal_slices` with the same
  principal-slice sheet layout as training at step 0 and interval test runs.
  The TensorBoard writer is flushed after configured test logging so initial
  test scalars and images are visible promptly.
- The `train_s1a_nml_all_64_sd2.json` 3D config includes the same held-out 2D
  fiber JSON `test_datasets` block as the full S1A NML 3D config, so step-0 and
  interval dense 3D test loss run for the fast 64-scale training setup.
- When `training.test_trace2cp_enabled` is true, 3D training logs
  `test/trace2cp_error`, raw y-error, valid segment count, and skipped segment
  count. Trace2CP metrics are diagnostic and must not replace dense
  `test/loss_total` as the snapshot metric. `best.pt`, `current.pt`, and
  retained numbered snapshots store `metric_name: test/loss_total`.
- `training.test_trace2cp_control_points: 0` means the full held-out Trace2CP
  CP set in flat order. Positive values use the deterministic random held-out
  range beginning at `training.test_trace2cp_start_sample_index` or, when that
  key is omitted, `training.test_start_sample_index`.
- The 3D Trace2CP metric path performs no training augmentations. Required 2D
  metric geometry must be explicit through `training.test_trace2cp_loader_config`
  or the 3D config keys `test_trace2cp_patch_shape_hw`,
  `test_trace2cp_strip_z_offset_count`, and `test_trace2cp_strip_z_offset_step`;
  missing required geometry must fail loudly.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.train --trace2cp-vis`
  runs the same 3D projection/scoring path for one sample or a whole
  `--fiber-json`, prints `trace2cp_error=...` or `trace2cp_error_mean=...`, and
  exports `trace2cp_3d_vis.jpg`.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool` is a
  separate native 3D Trace2CP inspection tool. It must not replace the
  projected `test/trace2cp_error` diagnostic or affect best-checkpoint
  selection, which is based only on dense `test/loss_total`.
- Native 3D Trace2CP is metric-only by default. It always prints native metric
  lines and writes `trace2cp_native_3d_summary.json`; JPG visualization and
  partial image updates are opt-in and run only when `--vis` is supplied.
- Native 3D Trace2CP accepts multiple `--fiber-json` paths in one invocation.
  Multi-fiber mode is whole-fiber only: sample-index and explicit CP selectors
  are rejected because the accumulated score is defined over complete fibers.
  The tool traces the fibers sequentially with one shared loaded model, writes
  one per-fiber summary as `trace2cp_native_3d_000_summary.json`,
  `trace2cp_native_3d_001_summary.json`, etc., writes indexed JPGs with the
  same stems when `--vis` is supplied, writes
  `trace2cp_native_3d_summary_all.json`, and reports the accumulated
  restart-rate score from summed restarts divided by summed original-line
  reference length.
- Native 3D whole-fiber JPG visualization must never write a JPEG whose width
  reaches the format limit. When `--vis` is enabled, completed restart spans
  are packed into split pages with a target width of `32000` pixels so page
  breaks prefer restart boundaries. A single very long no-restart span is split
  internally before it reaches the JPEG dimension cap; split pages keep the
  base output path for page zero and write additional pages with numbered
  suffixes.
- Native 3D whole-fiber visualization must mark CP positions without covering
  the point with distance text. CP labels are drawn at the bottom edge of each
  rendered strip and include the CP index plus the distance/miss state, e.g.
  `cp=17 d=3.2`, so the same index can be used with explicit CP selection
  arguments.
- Dedicated native 3D Trace2CP metric configs may require the JSON fiber to be
  supplied by `--fiber-json`. In that mode the config `datasets` entry is only
  a volume/scale/manifest template and must not carry a config-local
  `fiber_glob` or `fiber_paths` list. It should contain only metric/runtime
  fields and must not carry unrelated NML training datasets, affine transforms,
  train/test duplicate dataset blocks, augmentation settings, prefetch
  settings, loss weights, TensorBoard settings, or training-loop/run/checkpoint
  settings.
- Native 3D Trace2CP selection supports both the existing
  `--sample-index`/`--target-offset` mode and explicit fiber segment mode:
  `--fiber-json <path> --start-cp-index A --target-cp-index B`. Explicit CP
  index mode requires `--fiber-json`, requires both CP indices, uses flat
  single-fiber CP ordering, and must reuse the existing 2D Trace2CP segment
  source builder with `target_control_point_index`.
- When `--fiber-json <path>` is supplied without explicit CP indices, native
  3D Trace2CP defaults to whole-fiber mode. Whole-fiber mode traces
  consecutive CP pairs from CP `0` to the last CP by default. Supplying
  `--whole-fiber-start-cp-index N` starts whole-fiber tracing at CP `N` and
  measures the restart-rate denominator along the original line from CP `N` to
  the final CP. The chosen start CP must leave at least one target segment.
  Supplying both explicit CP indices keeps the single-segment debug mode;
  supplying only one CP index must fail loudly.
- The native 3D tool traces in selected-level ZYX voxel coordinates. It loads
  the same dataset/test-dataset CP pair as the visualization geometry loader,
  decodes six Lasagna 3x2 direction channels analytically, treats predicted
  axes as sign-ambiguous, and aligns sampled directions to the current trace
  direction before scoring.
- For conditioned 3D models, native Trace2CP inferred-block caching must store
  grouped recurrent outputs: zero-query output first, then output conditioned
  on the first decoded direction. Existing branch-aware candidate scoring then
  chooses between strongest and recurrent secondary predictions; these grouped
  slots are not free branch heads.
- Native 3D Trace2CP inference uses overlapped axis-aligned model-output
  blocks. Each block has a full input patch and a cropped trusted core; point
  lookups must route to a block whose trusted core contains the queried point.
  The tool must not silently score candidates from cropped-away model-output
  borders.
- When the native 3D Trace2CP checkpoint output has grouped
  direction/presence branches (`7*K` channels), inferred block sampling decodes
  all `K` Lasagna 3x2 direction branches plus their branch-local presence
  values. Branch 0 remains the compatibility layout for single-branch callers,
  but native tracing must not be branch-0-only for grouped outputs.
- Native 3D Trace2CP cached inferred blocks are device-resident on the tracing
  device and bounded by the existing LRU byte budget. CUDA tracing must sample
  cached model-output blocks without copying every resident block back from CPU
  for each lookup. Long whole-fiber traces still must not retain every
  historical block until process exit; `--max-cached-inference-gib` bounds the
  resident inferred field cache and eviction may cause re-inference.
- Native 3D Trace2CP field lookup must keep query points, block-origin
  calculation, per-block grouping, trusted-core masks, and sampled field
  tensors on `cache.device` for resident lookups. CPU transfer is limited to
  the unique missing/resident block origins required by the VC3D/model-output
  block cache. The lookup must not convert every candidate point batch to
  NumPy for `np.unique`/`flatnonzero` routing.
- Broad reference-line or corridor model-block prefetch must not be enabled by
  default. It may only be added when it is proven metric-equivalent to the
  incremental inference order, because changing model-block materialization
  order/batching has been observed to alter native Trace2CP decisions for the
  current checkpoint.
- Native 3D Trace2CP supports `--inference-scaledown-power N` for opt-in
  lower-resolution tracing over the raw model-output field. The scaledown
  factor is `2 ** N`: `0` is the default no-op, `1` samples a half-resolution
  field, and `2` samples a quarter-resolution field. The model input patch is
  unchanged; after inference, every raw product tensor is downscaled with the
  same Gaussian pyramid helper Lasagna predict3d uses (`_pyrdown3d`, repeated
  `[1,4,6,4,1]/16` separable filtering plus `::2` subsampling) before the
  field cache stores it. The valid mask remains a conservative support mask:
  it is reduced with the same factor and a scaled output voxel is valid only if
  all source voxels in the cell were valid. The inference patch shape and
  `--core-margin-voxels` must be evenly divisible by the factor, and invalid
  combinations must fail before tracing starts. The trusted core is still
  defined in selected-level voxel coordinates, while cached output lookups
  convert points to the scaled field with the same factor.
- Native 3D Trace2CP supports `--inference-blur-sigma-voxels` for opt-in 3D
  Gaussian blur over the inferred direction/presence fields. The blur runs
  after model inference and after optional `--inference-scaledown-power`
  pyramid filtering, but before trusted-core margin cropping into the field cache.
  The configured sigma is measured in unscaled selected-level inference voxels;
  internally the scaled field uses `sigma / inference_scaledown_factor`, so
  changing scaledown does not change the selected-level blur size. The default
  `0.0` preserves unblurred behavior, and negative sigma values must fail
  before tracing.
- Native 3D Trace2CP inferred blocks must be bounded by resident cache bytes by default.
  The native CLI uses an LRU byte budget exposed as
  `--max-cached-inference-gib`, defaults to 8 GiB, and reports total inferred,
  resident, evicted, and resident byte/GiB counts. Eviction may cause
  re-inference, but long whole-fiber runs must not retain every historical
  block until process exit. Cached model-output blocks retain only the trusted
  core plus the one-voxel upper interpolation halo needed to preserve
  trilinear point sampling inside the trusted core; full margin outputs must
  not remain in the resident cache after block inference.
- Native 3D Trace2CP inference blocks are regular axis-aligned selected-level
  regions and must be sampled through `CoordinateSampler.sample_block_zyx(...)`.
  This path is backed by VC3D requested-level chunk-cache reads and must not
  materialize dense `[D,H,W,3]` coordinate grids or call generic
  `sample_coord_batch(...)` for these blocks. Real configured volumes must not
  be read by direct zarr/raw block slicing in Python. The block sampler must use
  strict requested-level VC3D blocking semantics, report
  `requested_level_only=true` and `fallback_levels=0`, reject chunk errors, and
  mark out-of-volume voxels invalid/zero. Known-missing requested-level chunks
  follow the existing strict VC3D rendering semantics: black covered pixels and
  a non-zero `missing_chunks` stat.
- Generic `CoordinateSampler.sample_coords(...)` and
  `sample_coord_batch(...)` remain the correct boundary for arbitrary
  coordinate surfaces such as side/top strips, TTA surfaces, and strip
  visualization.
- Native 3D Trace2CP applies the configured 3D model-input normalization before
  inference. Exported native strip volume panels must display that same
  normalized input domain so the visualization shows what inference sees. For
  `image_normalization: "zscore"`, display maps a fixed normalized `[-3, 3]`
  window to `0..255`; for `minmax`, display maps normalized `0..1`; for
  raw/none modes, display clips raw `0..255`. Per-panel percentile display
  scaling is not allowed for native Trace2CP volume panels because it hides
  loading and brightness problems.
- Trace2CP strip rendering must reject non-blocking coordinate samplers and
  VC3D sampler results that do not report strict requested-level blocking
  semantics. Scale fallback, unresolved requested chunks, or chunk errors must
  fail loudly for debugging renders instead of being shown as valid strips.
- Native 3D Trace2CP defaults to `--inference-patch-shape-zyx 128 128 128`
  and `--core-margin-voxels 48`, matching the current trained checkpoint setup
  and the observed artifact margin. Other patch shapes remain explicit CLI
  overrides.
- Native 3D Trace2CP may batch missing axis-aligned inference blocks before
  model forwarding. `--inference-block-batch-size` controls the maximum number
  of newly materialized blocks in one forward and defaults to `2` to limit
  transient 128-cube GPU memory.
- Native 3D Trace2CP ordinary single-sample CLI mode defaults to sample index
  13 when no explicit `--sample-index` is provided. Bare `--fiber-json`
  without sample/CP selectors remains whole-fiber mode and must not be turned
  into sample-index mode by this default.
- Native 3D Trace2CP does not default to a fixed large step count. The default
  trace guard is distance-derived:
  `ceil(max_step_factor * cp_distance_voxels / step_voxels)`, with
  `--max-step-factor 3.0`. `--max-steps N` is only an optional additional
  safety cap.
- Native 3D candidate stepping samples deterministic tangent-plane angular
  offsets around the current inferred 3D direction. The default cone is
  `--cone-angle-degrees 25.0` with `--cone-angle-step-degrees 5.0`, keeping
  offsets inside the cone disk and always including the center direction. This
  produces 81 candidates at the default settings. The legacy square-grid
  generator is used only when `--cone-angle-step-degrees <= 0`, in which case
  `--cone-grid-size` controls the grid. Ring/azimuth candidate generation is
  not supported.
- Native 3D Trace2CP uses beam search by default. `--beam-width 8` keeps
  multiple cumulative candidate histories, `--beam-prune-distance-voxels 1.0`
  merges near-duplicate live beam states, and `--beam-lookahead-steps 2`
  expands short future trees before pruning. Pruning happens after the
  configured lookahead expansion, not after every single candidate step.
  `--beam-width 1` preserves the previous greedy one-step-commit control flow
  and bypasses lookahead. When target-plane candidates are found, the reached
  beam with the lowest cumulative score is selected; if no beam reaches the
  target plane before the step guard, the best live state is returned with the
  same failure reason semantics as greedy tracing.
- Native 3D beam-mode candidate selection is vectorized across the active
  beam/frontier states and their candidate directions for each lookahead
  depth. Candidate directions are generated as torch tensors on `cache.device`;
  current-point branch selection, candidate scoring, target-plane crossing, and
  pruning operate on tensors. Candidate points are then grouped by trusted
  inference block, sampled with batched `grid_sample`, decoded with the
  analytic Lasagna 3x2 torch decoder, and scored as tensors. The bounded
  inferred-block cache keeps sampled model-output tensors on `cache.device`;
  cache-miss source-block construction may still involve CPU/VC3D reads, but
  resident candidate point routing and trusted-core grouping stay tensorized.
  For multi-branch outputs, every candidate evaluates every branch at the
  candidate point and uses the branch with the best score. Candidate selection
  minimizes a cost. By default, the direction score uses all-pairs product
  scoring over four signed/aligned directions: previous step direction,
  current-point sampled direction, candidate step direction, and candidate-point
  sampled direction. Candidate-sampled axes are sign-aligned to the candidate
  step direction, pairwise dots are clamped to `[0, 1]`, and the score is
  `presence * product(six pairwise dots)`. `--no-all-pairs-direction-product`
  restores the older two-dot score
  `dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * presence`.
  `--candidate-substeps 1` is the default and preserves endpoint-only candidate
  scoring. With `--candidate-substeps S` for `S > 1`, candidate scoring samples
  the segment at `t = 1/S, 2/S, ..., 1`, evaluates all branches at every
  substep, takes the best branch score per substep, averages those substep
  scores, and then applies the current-point direction gate when legacy
  two-dot scoring is enabled. A multi-substep candidate is valid only when
  every substep has at least one valid branch. Search smoothness defaults to
  normal-aware split smoothness in the native 3D CLI. Candidate Lasagna normals
  default to sparse Lasagna corner/tensor sampling through
  `--normal-sampler sparse-corner-principal`: sample `grad_mag` at candidate
  points, sample compact `nx/ny` only at the eight channel-grid corners, decode
  corners, blend the sign-invariant tensor/hint, and recover the axis with the
  same principal-axis helper as the baseline path. The established
  `fiber_trace_2d` geometry-loader path `_lasagna_normals_at_zyx_batch` remains
  available as `--normal-sampler baseline` and as the fail-fast comparison
  reference. Compact `nx`/`ny` normals must not be interpolated directly, and the
  tracer must not call `FitData3D.normal_3d` on interpolated compact normals,
  perform grid search, or interpolate normals by reference-line progress.
  Once the sign-invariant local tensor is built from decoded compact-normal
  corners, the principal axis may be recovered with a batched symmetric
  eigensolve or another measured tensor principal-axis method. The native 3D
  default is `--normal-principal-axis-method config`, which resolves to
  `native_trace2cp.normal_principal_axis_method` when present and otherwise to
  `eigh`. The explicit `analytic` method is an experimental closed-form
  symmetric-tensor principal-axis decoder. It must remain opt-in unless the
  approved whole-fiber benchmark matches the `eigh` restart metric and improves
  timing. With a valid
  candidate normal axis,
  smoothness is split into tangent-plane turn and normal-tilt turn using the
  vector-normal projection equations from the pre-acceleration tracer:
  tangent-plane turn is the angle between previous and candidate step
  directions after subtracting their signed normal-axis components, while
  normal-tilt turn compares their signed `asin(dot(direction, normal))`
  elevations. Both components use
  `max(0, angle - smoothness_free_angle)^2`, in radians, and the native 3D
  CLI default for `smoothness_free_angle` is `0` degrees so all measured
  turns are penalized unless explicitly overridden. The Lasagna normal
  sign ambiguity must not affect this penalty. The CLI flags
  `--smoothness-tangent-weight` and `--smoothness-normal-weight` override the
  component weights independently; their native 3D CLI defaults are `10.0` for
  tangent-plane turn and `0.1` for normal-tilt turn. Native Trace2CP must fail
  before tracing when these normal-aware terms, or the cumulative tangent term,
  are active and no Lasagna normal sampler is available. If a Lasagna normal
  sampler exists but returns an invalid normal for one candidate, that
  candidate falls back to the previous isotropic smoothness term
  `smoothness_weight * max(0, angle(previous_step_dir, step_dir) - free_angle)^2`.
  Native 3D Trace2CP also adds cumulative tangent-only smoothness over a
  short history direction so several small tangent-plane turns cannot compound
  into a large tangent-plane bend. This cumulative term is additive
  smoothness, not a direction/presence gate. It uses
  `--cumulative-smoothness-steps` to update a running trace heading and
  `--cumulative-smoothness-tangent-weight` to penalize the tangent-plane angle
  between that heading and the candidate step. It never penalizes
  normal/elevation change. Missing Lasagna normal sampling is a hard error when
  this term has positive weight. If a sampled candidate normal is invalid or the
  tangent projection is degenerate, the cumulative term is zero for that
  candidate.
  The optional `--debug-compare-normal-sampler` mode is diagnostic only. It
  wraps the production geometry-loader sampler, runs one or more accelerated
  sparse Lasagna samplers on the same candidate points, returns the production
  normals to the tracer, and raises immediately on valid-mask mismatch or an
  angular difference above `--debug-normal-angle-threshold-degrees`. This mode
  must not be used as production scoring.
  The native 3D tool does not expose additive direction/presence
  candidate-selection weights.
- The native 3D Trace2CP start direction is sampled from the model at the
  start CP. The adjacent CP-local fiber-line tangent toward the target CP's
  line index is only a reference used to sign-align and choose the start
  direction branch. It must not use the straight CP-to-CP chord. For
  multi-branch outputs, the start branch is the valid branch with the highest
  directional agreement to that CP-local tangent; start-branch selection is
  not weighted by branch presence. The selected sampled direction becomes both
  the current direction and previous/history direction for the first candidate
  step, so direction scoring and normal-aware/cumulative smoothness apply to
  the first step exactly as they do to later steps. Later steps sample the
  model direction at the current trace point, sign-aligned to the previous
  accepted step, and keep the full direction gate plus normal-aware smoothness.
  For ordinary current-point lookup after the start CP, the branch is chosen by
  best `dot(branch_dir, previous_step_dir) * branch_presence`.
- The native 3D CLI prints live progress bars for forward and backward tracing.
  Progress is measured from remaining Euclidean distance to the target CP, not
  from a CP-to-CP chord target-plane normal. It includes step count, ETA, and
  inferred-block count.
- Native 3D Trace2CP always reports final metric lines plus total trace
  wall/CPU time. Detailed per-stage profiling is opt-in through `--profile`,
  because profiling uses instrumentation and some CUDA synchronizations that
  should not slow ordinary metric-only runs.
- When `--vis` is supplied, native 3D strip visualization prints live progress
  for rendering stages and
  for side/top presence-strip sampling. Presence progress must report
  processed inference blocks, total unique inference blocks, sampled strip
  points, valid output points, newly inferred blocks, cached blocks, and total
  cache block count. Regular trace candidate sampling remains quiet unless a
  caller explicitly supplies a progress label.
- When `--vis` is supplied, native 3D strip visualization progressively
  overwrites the regular
  `trace2cp_native_3d_vis.jpg` output at render start, stage start/end, and as
  panels are rendered and added to the sheet. Before the first panel is
  available, the file must contain a status canvas rather than being absent.
  There must not be separate partial snapshot filenames; the same output path
  should always show the latest available status, partial sheet, or final sheet.
- `--trace-step-limit N` is a debug-only cap on accepted trace steps per
  direction. When set, native tracing can intentionally return a partial trace
  with `reason=trace_step_limit`; this is distinct from the safety guard
  `--max-steps`.
- Native 3D tracing must not use the straight CP-to-CP chord as a target-plane
  normal. Each one-way trace targets explicit target-local planes through the
  target CP: the plane normal from the target CP line point to the next fiber
  line point when available, the plane normal from the target CP line point to
  the previous fiber line point when available, and the sampled model
  direction at the target CP sign-aligned to the local target tangent. The
  trace continues until all configured target-local planes have been crossed
  and the selected crossing is within the caller's endpoint threshold, or
  until its step budget is exhausted. It selects the crossed plane with the
  smallest in-plane CP error. Later
  crossings of the same target plane replace earlier crossings when their
  in-plane CP error is lower. The shared C++ tracer used by VC3D and the native
  metric CLI must preserve this state independently for every beam and compact
  lazy-lookahead frontier. Its fixed-capacity representation supports the three
  derived target-local planes and must reject larger explicit sets. In Python
  and C++ whole-fiber tracing, all target-local planes being crossed is necessary
  but not sufficient for segment acceptance:
  the selected best crossing error must also be at or below the configured
  whole-fiber threshold; otherwise tracing continues until the budget or another
  failure condition ends the segment. CP-pair tracing must retain the complete
  unsnapped stepped path for post-trace meeting search, including samples after
  an early out-of-threshold crossing. Whole-fiber tracing likewise retains the
  actual stepped endpoint and uses the selected crossing only for acceptance
  and error reporting. Missing required target planes are reported in the
  failure reason.
- In native 3D whole-fiber mode, `--fiber-json <path>` without sample or CP
  selectors traces the entire fiber. `--fiber-json <path> --sample-index N`
  remains single-segment inspection using deterministic flat sample selection,
  and explicit `--start-cp-index/--target-cp-index` remains explicit
  single-segment inspection. `--whole-fiber-start-cp-index N` is only valid in
  whole-fiber mode and traces CP `N` through the final CP.
- In native 3D whole-fiber mode, each segment targets the next CP using the
  same target-local plane set described above. A segment succeeds only when
  all configured target-local planes are crossed within the segment's step
  budget and the selected smallest in-plane error to the target CP is at most
  `--whole-fiber-error-threshold-base-voxels` (default `20`) after converting
  selected-volume distances with `volume_spacing_base`. Successful segments
  do not restart, resample CP-start direction, or reset smoothing history. The
  selected crossing is only the metric/checkpoint location; live tracing
  continues from the actual stepped trace point with previous direction,
  sampled-current direction when cached, and smoothing-history direction
  preserved. Failed segments count one restart and resume tracing from the
  failed target CP with a fresh CP-local fiber tangent.
- Shared C++ CP-pair fusion must preserve each trace's traced order and search
  the complete forward and reverse paths even when either trace exhausts its
  endpoint-plane budget. It resamples both traces at a deterministic frequent
  interval, moves a locally tangent plane along each trace, and intersects the
  other trace's segments with that plane. The symmetric search also includes
  qualifying target-CP endpoint-plane crossings. The selected candidate has
  the smallest raw 3D/in-plane meeting error; exact ties prefer more balanced
  progress, then greater combined progress and stable trace indices.
  Acceptance requires positive combined partial traced length and
  `meeting_error / combined_partial_trace_length <= 0.10` by default. An
  acceptable moving-plane meeting does not require either one-way trace to
  have reached all endpoint planes. The selected pair midpoint is the fusion
  meeting point: the forward start-to-meeting and reverse target-to-meeting
  partial traces are warped to that midpoint by traced arc-length fraction,
  concatenated, then arc-length-resampled as the CP-to-CP fused line. Original
  CP endpoints are restored exactly.
- Native 3D Trace2CP reports tool-local debug metrics:
  `native_trace2cp_plane_error` and
  `native_trace2cp_closest_target_error`, plus fusion diagnostics such as
  selected diagnostic progress, raw gap, considered pair score, and center
  penalty. For pairwise traced-arc fusion the center penalty is fixed to `1.0`.
  These are not the public 2D `trace2cp_error`.
- Native 3D whole-fiber mode reports its tool-local human stdout metric as
  compact error-rate fields: `err/kvx=...` and, when physical units are
  available, `err/m=... (N.Nmm)`, where the parenthesized value is the mean
  successful traced run length between restarts in millimeters. Human
  stdout/progress should use one digit after the decimal for `err/kvx`,
  `err/m`, and the millimeter run length, and must not include physical unit or
  reference length fields. Live whole-fiber progress must update one terminal
  line with carriage returns; it must not print a fresh line for every segment.
  It should emit a newline when the restart counter first increases to a new
  value and when progress reaches the terminal state, so restart events remain
  visible in persisted terminal logs.
  The metric is `restart_count / (reference_length_voxels / 1000)`, where
  `reference_length_voxels` is measured along the original loaded fiber line
  between CP0 and the final CP in selected-level voxels. Physical units are
  reported only when the VC3D sampler exposes
  `record.sampler.volume.metadata["voxelsize"]` as a finite positive value in
  micrometers. In that case the tool converts it with `voxelsize * 1e-6`.
  If that exact VC3D metadata path is unavailable or invalid, the per-meter
  field is omitted from stdout and null in JSON. The
  VC3D remote volume loader must normalize public Vesuvius
  `scan/tomo/acquisition/detector/samplePixelSize` metadata into
  `metadata["voxelsize"]` whenever no explicit positive `voxelsize` exists,
  independent of the remote volume base-scale mode. The
  fiber code must not parse Zarr/OME JSON directly, inspect dataset config or
  record metadata for physical units, accept alternate keys, or infer voxel
  size from filenames. The JSON summary stores per-segment status, reason,
  reached-plane flag, in-plane error, step count, restart point, reference arc
  distance at the last successful CP plane, full-precision reference lengths,
  full-precision `native_trace2cp_fiber_restarts_per_kvx` and optional
  `native_trace2cp_fiber_restarts_per_meter`, and the old segment-normalized
  fraction only as `restart_fraction_per_segment`.
- VC3D native GUI fiber tracing is a segment-local port of the 3D Trace2CP
  behavior. It consumes a precomputed fiber inference `.lasagna.json` dataset
  from the project and never runs PyTorch/model inference inside VC3D.
- VC3D projects store normal and fiber inference manifests in the canonical
  `lasagna_datasets` collection. The reserved `vc-lasagna-fiber` entry tag
  identifies fiber inference data; an entry without that tag is regular
  Lasagna data. `selected_lasagna_dataset` and
  `selected_fiber_inference_dataset` select the two roles independently. The
  old `fiber_inference_datasets` project field is accepted on read, migrated
  into tagged canonical entries, and is not written. A native GUI trace
  requires both a normal Lasagna dataset for geometry normals and a selected
  fiber inference dataset for persisted `presence`/`nx`/`ny` prediction fields.
- Native GUI fiber prediction must decode persisted fiber inference channels
  with the shared `vc_lasagna` compact-channel helper. It must not copy private
  normal-sampler logic, raw-interpolate compact `nx`/`ny` values as ordinary
  directions, or invent a separate remote-cache path.
- Native GUI fiber prediction must resolve the selected fiber inference
  manifest's trace scale from `source_to_base` and the persisted prediction
  sampling scale from prediction group `scaledown` before opening prediction
  channels. The
  persisted lines and control points remain in base coordinates even when the
  derived trace scale differs. The GUI must convert base points to trace space,
  open separate prediction and normal samplers with that trace-to-base scale,
  run tracing in trace voxels, then convert accepted points back to base space.
  Original segment endpoints must be restored exactly. Endpoint errors are
  converted to base voxels before acceptance; the ordinary line sampler
  remains in base space for final reconstruction.
- Ctrl-right-click on a generated line annotation span exposes a checked
  `Interpolation goal` submenu with `Global`, `Cubic spline`, `Lasagna`, and
  `Fiber trace`. The action changes only the CP-owned goal for that span, runs
  the shared grouped interpolation coordinator in a background task, blocks
  line edits while it runs, and commits geometry and segment descriptors
  atomically. Original CP coordinates must remain exact.
- Native GUI segment optimization runs both directions until their configured
  target-local planes are reached within `20` base voxels or their step budgets
  are exhausted, then applies the symmetric moving-plane meeting search. At
  the default sd2 trace scale the endpoint threshold is `5` trace voxels. A
  fused span is accepted when the selected meeting error in base voxels is at
  most `max(10, 10% of its combined partial traced length)`. The ratio remains
  a stored/displayed diagnostic. `Volume::voxelSize()` is used only to add a
  micrometer diagnostic when it is finite and positive; unavailable physical
  metadata must not block tracing.
- `vc3d_fiber` version 3 stores every control point as an object with a finite
  `position`. CP `i` owns the required `segment_to_next` descriptor for its
  following span; every non-final CP must contain one and the final CP must not.
  Every new non-final CP
  persists `interp_goal` (`global`, `cspline`, `lasagna`, or `trace`) and
  `interp_mode` (`cspline`, `lasagna`, or `trace`). The goal is policy; the mode
  identifies the algorithm that produced the stored dense geometry and must
  never be inferred from the fiber-wide mode after loading.
- A version-3 descriptor also stores compact `msg` and optional `metric`.
  `trace` metric is minimum meeting-plane error in base voxels, `lasagna`
  metric is maximum final normal-alignment error in degrees, and `cspline`
  has no metric. Trace configuration/meeting/failure diagnostics and Lasagna
  failure code/detail remain mode-specific. Stale fields are cleared when a
  later retry changes actual mode. Unknown enums, malformed metadata, or a
  descriptor on the final CP are hard errors in every strict reader.
- `normal_manifest` is the Lasagna manifest identity used for interpolation;
  `fiber_manifest` is the fiber-inference manifest identity. Direct Lasagna
  records the Lasagna identity only. Trace records both because the tracer also
  samples Lasagna normals. Direct/short cubic spline records neither. A
  fallback retains every identity actually consulted by its rejected attempts.
  Ordinary project datasets use their configured local or remote manifest
  location. Catalogue-backed Lasagna uses the exact public remote manifest URL
  reconstructed from its artifact URL and root manifest filename, never the
  cache path. The catalogue has no artifact UUID; sample ID, volume ID,
  coordinate level, optional model ID, and artifact index are auxiliary
  catalogue identity components.
- Legacy version-1 fibers remain readable and their numeric CP-to-CP spans load
  as goal `global`, actual `lasagna`. Version 3 is the only supported object-CP
  format and the only format carrying segment descriptors. The unpublished
  top-level file version 2 and metadata/tracer schemas `(1, 1)` and `(2, 2)`
  are rejected; version 3's current descriptor schema `(3, 2)` remains valid.
  Writers always emit version 3. Fiber coordinate scaling preserves goals,
  actual modes, and diagnostics while scaling base-voxel trace quantities.
- Every v3 reader validates the complete descriptor sequence and the required
  top-level `optimization_mode` before consuming geometry, including VC3D,
  native CLI tools, Atlas/Lasagna/Spiral consumers, Python loaders, and sync.
  Missing or malformed v3 metadata is a hard format error: readers must not
  synthesize defaults, normalize fields, tag the file for reoptimization, or
  rewrite it. Multi-file tools may report and skip the invalid file unchanged.
  Legacy v1 alone may omit `optimization_mode`, which defaults to `lasagna`,
  and receives in-memory global/Lasagna span state.
- Segment descriptors and dense geometry are applied atomically. CP movement
  dirties both adjacent spans while preserving their goals. Insertion copies
  the split owner's goal to both new spans. Interior deletion leaves the left
  owner's goal on the merged span. Unaffected spans remain protected. Overlay
  refresh and reload must repopulate status directly from the persisted
  descriptors.
- Fiber-aware sync treats each version-3 CP-to-CP dense line slice and its
  CP-owned descriptor as one atomic three-way-merge value. A one-sided change
  is retained verbatim; identical two-sided changes are accepted. Different
  changes to the same run are conflicts. Local-only and remote-only changed
  runs may coexist only when at least one complete base span between them is
  unchanged on both sides. Adjacent changes, overlapping CP topology edits,
  missing ordered CP/line anchors, or inexact selected-run joins are manual
  conflicts. Sync must never select a segment from the higher generation,
  combine descriptor fields from different results, or replace version-3
  geometry with a CP-only placeholder. `optimization_mode` is merged
  base-aware: an isolated change wins, equal changes converge, and different
  two-sided changes conflict. Version-1 merge behavior is unchanged; version 2
  is invalid and is sent to manual conflict handling.
- Each VC3D fiber has a persisted top-level `optimization_mode`: `lasagna` or
  `native_fiber_trace3d`. It is required for version 3. Missing mode metadata
  on existing version-1 files defaults to `lasagna`; unknown values are errors.
  The mode is the fiber-wide extrapolation policy and resolves only `global`
  CP-to-CP goals.
  New interactive fibers default to `native_fiber_trace3d`; this controls the
  initial line geometry as well as the GUI and persisted mode. When a selected
  or uniquely attached fiber-inference dataset is configured, seed placement
  must use the Lasagna seed solve only as an internal reference-line/tangent
  baseline and then replace both open tails through the existing single-CP
  native extrapolator. The intermediate Lasagna line must not be displayed or
  saved. With no selected or uniquely attached inference dataset, initial
  creation remains Lasagna and must not force a file picker. This must not
  change the Lasagna compatibility default used while loading older files.
- Goal resolution starts with the explicit goal or the fiber-wide mode for
  `global`. A global Lasagna/trace span whose Euclidean endpoint distance is
  below 100 base voxels uses `cspline` directly; exactly 100 still attempts the
  resolved global mode. Explicit goals are exempt. Fallback order is
  `trace -> lasagna -> cspline` and `lasagna -> cspline`; every relevant
  reoptimization starts again from the goal so prior fallback is not sticky.
- Trace attempts and Lasagna initialization success are decided independently
  per span. Successful trace spans are stitched and protected. Each failed
  trace enters Lasagna, and only a Lasagna span with no usable initializer is
  demoted to cubic spline. Connected usable Lasagna geometry is then jointly
  refined with all trace, cubic-spline, and unrelated explicit spans protected.
  At every protected-span-adjacent CP, the tangent is
  `normalize(first_distinct_native_point - CP)`, walking inward from that CP;
  the Lasagna geometry on the opposite side must leave the CP along the
  negated tangent. VC3D materializes one adjacent proxy point on that tangent
  and fixes both the CP and proxy in the ordinary Ceres solve. No custom
  manifold or weighted direction penalty is used. The fiber direction is the
  only rollout candidate when it is available at one endpoint; if both
  endpoints have native directions, one rollout candidate is generated from
  each endpoint. The old Lasagna span and its endpoint directions are never
  reinitialization candidates. A direction propagated from an already solved
  neighboring span has the same precedence over generic CP/chord rollout
  initialization. This applies independently to all native spans, to both ends
  of a Lasagna span bracketed by native spans, and to a retained Lasagna tail
  after native extrapolation failure.
  Per-span solves and the final shared global solve preserve the fixed proxy
  while the existing Lasagna smoothness terms optimize the remaining points.
  Native samples remain fixed. A successful native span without a finite,
  distinct endpoint-neighbor sample is an error.
- Adjacent `cspline` spans form one run. The shared core interpolator uses
  chord-length piecewise cubic Hermite geometry with exact CPs, one shared
  tangent at every internal CP, and hard boundary directions from neighboring
  stored spans. With only two CPs and no boundary directions it is exactly
  straight. Handles are bounded and deterministically reduced until every span
  is finite, forward-progressing, and locally bounded. Geometry does not
  consult normal or prediction data and is resampled at base annotation
  spacing.
- A global-mode change dirties every `global` span and initially protects
  explicit spans. A goal change dirties its span. CP edits dirty adjacent spans.
  Dirty cubic-spline spans expand through their connected spline run; untouched
  explicit spans stay fixed and provide boundary directions.
- Generated-strip status is derived from each descriptor. Prefix labels with
  `C`, `L`, or `T` for actual mode and show persisted metric and `msg`. Show a
  label whenever any of its span intersects the viewport, clamp it into the
  visible interval, preserve fiber order while packing labels in viewport
  pixels, and use a deterministic second row only when one row cannot fit.
  Detailed failure fields remain available in the tooltip.
- The VC3D generated line-annotation layout has a separate fixed-height,
  full-width schematic overview map plus two volume-rendered strip viewers:
  `lineSurface` followed by `lineSideSlice`. The overview compresses the whole
  line into its width and is not a replacement for either rendered strip. Both
  rendered strips remain ordinary interactive viewers with independent pan,
  zoom, scrolling, camera persistence, control-point interaction, and the
  per-span mode/metric/message labels above. They must not be converted to the
  obsolete fixed-height, fit-to-width, non-interactive rendered top strip.
  During in-place surface replacement, each rendered strip retains its prior
  overlays until that strip displays a frame for the new surface geometry.
- Lasagna direction transport must remain in the sampled-normal tangent plane.
  If removing a direction's normal component is degenerate, transport chooses
  a deterministic perpendicular tangent; it must never return the original
  normal-parallel direction.
- The line-annotation extrapolation control is measured in base voxels and
  applies beyond both outer CPs. Native mode converts that distance to trace
  voxels and derives a hard nominal generation budget from it. Extrapolation
  has no target planes and does not multiply its budget by `max_step_factor`;
  it takes `ceil(distance / step)` generations, with the final generation using
  the remaining nominal distance. Completing those generations is success;
  accumulated measured arc length is not a termination or acceptance input.
  Successful completion replaces the corresponding Lasagna tail. If the next candidate generation has no
  valid prediction directions, the one-way tracer retains its last valid path
  and VC3D uses that path as a native tail truncated at the data edge. A failure
  before producing one outward step retains the Lasagna tail. Whenever VC3D
  retains that fallback, it emits a command-line warning with the tail side,
  the full tracer or exception reason, the returned trace-point count, and the
  reason source. Completed length-based extrapolation and accepted data-edge
  truncation do not warn. This extrapolation-only rule does not change CP-pair
  or whole-fiber target-plane acceptance. A zero distance trims the line to the
  outer CPs.
- A fiber with only its initial seed CP still rebuilds both open tails when
  switching modes or changing the extrapolation distance. Changing the
  distance marks the line unoptimized and, when Auto-reoptimize is enabled,
  immediately rebuilds both tails in the active fiber mode.
- `vc_fiber_trace_metric <fiber.lasagna.json> <fiber.json>` is the native
  no-visualization full-fiber metric runner for precomputed 3D fiber inference
  output. It loads one `vc3d_fiber` JSON, requires exact control-point matches
  in `line_points`, traces one-sided from CP to next CP plane, continues from
  the reached point after success, restarts from the failed CP after failure,
  and reports restart rate per 1000 trace voxels as `err/kvx`.
  The runner requires an explicit numeric manifest `source_to_base` and a
  command-line `--inference-scaledown-power`, defaulting to 2. Persisted
  prediction channel scale is validated as
  `source_to_base * 2**group.scaledown`; trace coordinate scale is derived as
  `prediction_to_base / 2**inference_scaledown_power`, and prediction spacing
  in trace voxels is `2**inference_scaledown_power`. All `presence`/`nx`/`ny`
  channels used by the tracer, including prefixed multi-output channel sets,
  must agree on prediction scale. Missing prediction channels or
  scale-mismatched prediction channels are hard errors. The JSON fiber is
  assumed to already be in the manifest base coordinate system; no command-line
  fiber rescaling is currently exposed. Default trace-control parameters are
  `--step-voxels 4.0`,
  `--cone-angle-degrees 25.0`, `--cone-angle-step-degrees 5.0`,
  `--cone-grid-size 25`, `--beam-width 8`,
  `--beam-prune-distance-voxels 1.0`, `--beam-lookahead-steps 2`,
  `--smoothness-weight 2.0`, `--smoothness-free-angle-degrees 0.0`,
  `--smoothness-normal-weight 0.1`, `--smoothness-tangent-weight 10.0`,
  `--cumulative-smoothness-steps 4`, and
  `--cumulative-smoothness-tangent-weight 2.0`, matching the regular Python
  Trace2CP defaults except for inference-only options.
  Endpoint acceptance uses `--error-threshold-base-voxels 20`; trace-grid
  endpoint errors are multiplied by the derived `trace_to_base` before this
  comparison.
  `vc_fiber_trace_metric` requires an explicit `--normal-manifest` pointing to
  the Lasagna normal manifest used for tangent/normal smoothness. It must not
  try to derive normals from the fiber prediction manifest because the
  precomputed fiber products do not persist Lasagna normal channels. Physical
  `err/m` is reported only when the caller provides an explicit
  positive `--voxel-size-um`; the runner must not invent physical units from
  filenames or parse unrelated metadata. The CLI is a thin wrapper over
  `vc_fiber_tracer`, `vc_lasagna` dataset opening, and the required
  `LasagnaNormalSampler`.
- The native Lasagna dataset opener supports local `.lasagna.json` manifests,
  local manifests with an adjacent `lasagna-remote.json` read-through marker,
  and direct remote `s3://`, `s3+REGION://`, `http://`, or `https://`
  manifests when the caller supplies an explicit remote cache root. Direct
  remote manifest JSON is materialized through the shared exact-byte remote
  file cache and reused across runs. The CLI and VC3D therefore share durable
  manifest and referenced-Zarr caching. Remote
  manifest fetch failures must include the original location, redacted resolved
  request URL, HTTP status or no-response marker, response metadata and body
  excerpt when available, plus S3 region and credential-loaded status for S3
  requests.
- The shared remote-file cache stores one arbitrary file/object per request. It
  supports HTTP, HTTPS, S3, and region-qualified S3 transports plus a custom
  fetch-to-file callback. Cache-first and explicit refresh policies validate a
  canonical query-free source location, byte size, and accounting class from a
  sidecar. Readable cache placement mirrors validated
  `remote_sources/<scheme>/<authority>/<path>` components; traversal and
  platform-invalid components are rejected. Publication is atomic,
  concurrent in-process fetches are
  coalesced, failed refresh preserves the previous entry, and diagnostics must
  redact URL queries. Managed payloads participate in the persistent cache
  budget; unmanaged control files such as Lasagna manifests do not. Recursive
  remote-directory caching and automatic TTL/ETag refresh are not implied.
- Lasagna manifest group `zarr` paths are location strings. Relative paths
  resolve against the containing manifest location: the parent directory for
  local manifests or the parent URL for direct remote manifests. Absolute local
  paths starting at `/` are opened as local Zarr groups. Absolute remote
  `s3://`, `s3+REGION://`, `http://`, or `https://` group paths are opened as
  independent remote read-through Zarr roots. Relative remote-backed group
  paths that escape their manifest/artifact parent are rejected; absolute paths
  are explicit and are not rewritten.
- Attaching either a local or remote Lasagna manifest to a VC3D project also
  attaches every referenced group/channel as a flat ordinary 3D project
  volume. Every group must name exactly one channel and reference an actual ZYX
  array. Flat channel-first CZYX arrays are older Lasagna preprocessing/fit
  intermediates and must be converted to per-channel 3D OME-Zarr before VC3D
  attachment. Generic `Volume`, remote-volume loading, project storage, and
  VC3D remain strictly 3D. Each derived volume stores the actual resolved local
  or remote Zarr source as `location`; cache paths and synthetic identifiers
  are not project source fields. A `vc-lasagna-derived:<manifest location>`
  ownership tag drives deduplication, reload reconciliation, role changes, and
  detach cleanup. Group, channel, spacing, dtype, and shape remain authoritative
  in the manifest/Zarr descriptor and must not be duplicated in project tags. An
  independently attached primary volume is reused without an ownership tag and
  survives manifest detach only when its resolved source, 3D shape, dtype,
  fill value, base/present level layout, per-level shapes/chunks, and voxel
  spacing match the manifest-prepared volume. Runtime UUID equality is not
  required. Missing or incompatible runtime backing rejects and rolls back the
  manifest attachment. Manifest entry, derived volume entries, and
  selected-role updates
  are committed together or rolled back together; detaching removes a derived
  volume only when no remaining manifest owns it.
- `vc_fiber_trace_metric` exposes `--remote-cache-dir PATH` and opens both the
  fiber inference manifest and `--normal-manifest` through the shared
  location-aware Lasagna opener. If either manifest argument is remote and no
  remote cache directory is supplied, the CLI fails before tracing.
- Native 3D single-pair visualization first builds the initial side/top strip
  source from the existing 2D Trace2CP geometry loader for the input CP pair.
  In single-pair mode, the configured cross-strip height is a maximum cap: the
  rendered cross height is the odd centered size needed to cover the projected
  forward, backward, and fused traces with 50% extra margin, capped by that
  configured maximum. This adaptive render height is visualization-only and
  must not affect tracing or metric values.
- Trace2CP segment-source construction may trim extra line-window margin to the
  valid compact-geometry interval that contains both start and target control
  points. It must not synthesize missing normals. If the actual CP-to-CP line
  range crosses an invalid compact-geometry gap, source construction must fail
  loudly.
- Native 3D visualization includes side/top strip panels of the inferred 3D
  presence signal sampled on the displayed side/top strip coordinates from the
  native inference cache. Presence sampling should batch strip coordinates per
  strip rather than call model inference per pixel.
- Refined/fused/regenerated native 3D presence panels are visualization-only
  presence multiplied by the sampled ambiguous fiber direction's alignment to
  the displayed strip plane. The alignment must be sign-invariant and compare
  against the plane spanned by the strip column tangent and row axis, not a
  signed dot product against a single tangent vector. Original/input presence
  panels remain raw presence for comparison.
- Native 3D whole-fiber visualization uses eight stitched panel rows: initial
  side volume, initial side 3D presence, initial top volume, initial top 3D
  presence, regenerated/fused side volume, regenerated/fused side 3D presence,
  regenerated/fused top volume, and regenerated/fused top 3D presence.
  Whole-fiber mode renders restart-delimited continuous long strips instead of
  one visual column per CP segment. Each visual span starts at the first CP
  after a restart and ends at the latest traced target CP in that span. Failed
  segment overlays are cut before they overlap the next CP region, then the
  displayed trace resumes from the restart CP in the next visual span.
  Whole-fiber visualization always uses a fixed 64 px cross-strip width; this
  width is visualization-only, and a traced path leaving the 64 px strip must
  only clip the drawn overlay, not invalidate tracing, metric calculation, or
  3D sampling. The regenerated/fused rows rebuild side/top strip geometry from
  explicit traced volume-space XYZ points using the same low-level
  `FiberStripLineWindow` / `build_side_strip_patch_grid_tensor_from_line_window`
  construction as original CP-pair strips. Regenerated/fused strips must sample
  fresh Lasagna normals at the traced line points and must not recover their
  line from `source.grid.coords_xyz`, `source.grid.offset_axis_xyz`, or
  `source.grid.side_axis_xyz`. Traced points that are outside the original
  source strip valid area are not fatal and are not clipped for regenerated
  strip construction; only non-finite or degenerate traced line points may be
  discarded before construction. The
  regular `trace2cp_native_3d_vis.jpg` path must be overwritten
  after every completed segment so long whole-fiber runs show partial visual
  progress. Completed whole-fiber spans are retained as one composed sheet, not
  as a growing list of individual 8-panel image tuples.
  The control points covered by each visual span must be projected into the
  displayed initial and regenerated side/top strip frames and drawn as markers
  on all eight rows. Each marker
  must also show that CP's native trace distance at the CP plane: the span
  start CP is `d=0.0`, reached target planes show the segment's
  `in_plane_error_voxels`, and unreached target planes show `miss`.
- Native forward, reverse, and fused 3D traces are projected onto the initial
  side and top strip coordinate systems for overlay. The same visualization
  also rebuilds side/top strip geometry from the fused native 3D line and
  renders fused-line volume/presence panels with only the fused line overlaid
  thinly. The fused-line panels are debug visualization only; they do not define
  a new scoring path.

- The initial implementation loads batches of fiber-strip patches around random control points from the fiber dataset.
- Fiber source parsing accepts existing VC3D fiber JSON files and Knossos /
  WebKnossos `.nml` files. VC3D JSON parsing follows
  `vesuvius.neural_tracing.fiber_trace.fiber_json`.
- NML parsing orders nodes by edges, not XML order. Each usable open simple
  path component becomes one normalized `Vc3dFiber`; branch components, closed
  loops, disconnected singleton nodes, or malformed components are skipped or
  rejected with diagnostics rather than guessed through.
- NML line points and control points initially use the same ordered node
  coordinates unless a later explicit control-point convention is added.
- Each selected control point must be an exact member of `line_points`; otherwise the fiber JSON is rejected as inconsistent.
- The loader works on 2D sampled fiber side-strip patches.
- Neighboring strip-z context is represented as separate 2D patches.
- The default strip-z offset settings are `strip_z_offset_count=16` and `strip_z_offset_step=1.0`, generating `-7..8` selected-scale offsets and giving 16 patches per selected control point.
- Lasagna normals are used where needed to construct aligned strip frames.
- At loader startup, the loader builds one shared compact in-RAM fiber-line
  geometry store for all configured records. It computes the line-index ranges
  that can affect configured CP source windows and consecutive CP-to-CP
  Trace2CP spans, samples Lasagna normals only for those required ranges,
  builds valid contiguous frame intervals, and keeps compact per-line/frame
  arrays read-only for the rest of the process.
- A requested Trace2CP CP-to-CP span must not fail because an interior
  centerline point was omitted from compact-geometry preload. If compact
  geometry reports an invalid unsampled point inside such a span, that is a
  diagnostic bug path and the error must say so explicitly with a direct
  Lasagna value probe.
- Startup compact geometry construction may parallelize across independent
  records with process workers controlled by `loader_workers`.
  `loader_workers=0` means all logical CPU cores, and `loader_workers=1` is
  the serial startup/debug path. Each process opens its own base-volume and
  Lasagna channel handles, builds compact geometry for assigned records, and
  returns only compact arrays/metadata to the parent. Parallel startup may
  complete records out of order internally, but the final parent-owned store
  must be indexed by original record order.
- Startup Lasagna normal sampling may use batched/vectorized channel reads and
  normal decoding, but it must preserve Lasagna `_decode_normals`, ambiguous
  normal principal-axis handling, and strict invalid-data semantics.
- The compact geometry store is assembled and owned by the parent process, then
  shared by all threaded loader workers and cloned loaders in that process.
  Startup process workers must not remain as runtime geometry owners.
  `fiber_trace_2d` training does not currently use DDP or
  `torch.distributed`; this task must not introduce per-worker duplicated
  compact geometry.
- Runtime side/top source-grid construction looks up the record/control-point
  entry directly, evaluates only the requested source columns from the compact
  frame arrays, and broadcasts rows by frame axes. It must not write/read a
  dense per-CP coordinate cache.
- If a line point required by a CP source window cannot be sampled from the
  Lasagna manifest channels, the loader must not fabricate or propagate a
  replacement normal. That CP is invalid and is skipped by training/prefetch in
  deterministic stream order.
- During prefetch and training batch assembly, invalid CP-local samples caused by Lasagna channel data such as missing samples or in-bounds `grad_mag == 0` are skipped and reported, then the deterministic sample stream advances to the next sample.
- Fatal prefetch/training errors are infrastructure or programming failures such as missing APIs, broken bindings, interrupts, memory errors, or unexpected internal exceptions; those should stop the run rather than being hidden as data skips.
- VC3D side-strip/surface/segment sampling semantics define patch coordinates.
- Strip centerlines are sampled from all `line_points` with cubic Hermite interpolation over arc length; control points only select the strip anchor.
- The coordinate construction must be equivalent to VC3D side strips; flat planar patch simplifications are not acceptable except where they match the VC3D algorithm for that case.
- The implementation should reuse/export VC3D side-strip coordinate APIs when possible, or port the same algorithm with only small rounding/interpolation differences.
- Source-strip coordinate generation may use torch vectorization for Hermite
  interpolation and normal interpolation, but it must preserve the existing
  VC3D/Lasagna frame construction semantics.
- Dense source-strip coordinates, strip-z offset coordinates, geometric coordinate augmentation, and transformed line/control-point coordinates stay as torch tensors on the configured augmentation device until an explicit NumPy consumer boundary.
- The explicit NumPy boundaries are VC3D coordinate sampling, runner/PIL visualization/export, and sample metadata arrays. The loader must not repeatedly convert coordinates between NumPy and torch inside one source/augmentation path.
- The augment-vis source/patch path is the canonical loader path for runner exports, training batch loading, and prefetch coordinate generation.
- Augmentation visualization, training, runner batch loading, and prefetch must share the same CP-local source-strip and final-coordinate generation implementation.
- Normal training, benchmark/profile/load-only, augment-vis, line-trace-vis,
  dir-vis, and direct runner/debug center-patch loading use the configured
  `augment_device` for torch coordinate generation. With the example config,
  `augment_device: "auto"` uses CUDA when available. Prefetch dependency
  generation is the exception and stays CPU-pinned.
- The loader builds source geometry from the compact in-RAM store for each
  selected CP and reuses that source across augmentation variants and strip-z
  offsets as appropriate. Source-space line pixel coordinates remain the full
  per-column centerline used by training visualization, while volume sampling
  coordinates come from compact frame interpolation.
- Image loading samples base-volume Zarr values from explicit coordinates.
- Training/export coordinate sampling uses the VC3D blocking coordinate sampler: required chunks are collected and fetched/decoded before sampling, so a cold cache miss must not become an invalid output pixel.
- Interactive/progressive VC3D `tryGetChunk` semantics are not acceptable for this loader path because they can queue I/O and return an all-invalid first sample.
- `base_volume_scale` selects both the Zarr level to read and the sampling pixel scale: by default, one output patch pixel advances by one voxel at that selected level.
- Internally, fiber control-point coordinates remain in base-volume coordinates, so a selected level `s` uses a patch pixel spacing of `2**s` base voxels before coordinates are divided for reading level `/s`.
- The loader must not use the existing neural-tracing crop-loading path for image loading.
- Training's multiple strip-z offsets are derived from one CP-local source geometry by offsetting along the strip normal/frame direction, not by rebuilding a separate coordinate-generation path.
- Runtime geometric augmentation work should be batched across strip-z offsets
  and patches where tensor shapes are compatible. The implementation should
  avoid many tiny per-patch GPU calls when the same operation can be expressed
  as one batched tensor operation without changing deterministic sample order
  or augmentation semantics.
- Runtime image sampling should batch each CP sample's strip-z coordinate stack
  through `CoordinateSampler.sample_coord_batch`. If no native sampler batch API
  is available, flattening `[strip_z,H,W,3]` to one larger coordinate image is
  functionally valid because every output pixel samples from explicit 3D
  coordinates; only request traversal and cache/chunk locality may change.
- Dataset and loader settings are specified in Vesuvius-style JSON.
- Config keys include `datasets`, `batch_size`, `patch_shape_hw`,
  `strip_z_offset_count`, `strip_z_offset_step`, `seed`, `loader_workers`,
  `prefetch_workers`, `prefetch_sampler_workers`, `volume_cache_dir`, optional
  `volume_cache_memory_mib`, optional `volume_io_threads`, and optional volume
  cache settings. `strip_coord_cache_dir` has been removed and must be rejected
  if present.
- Augmentation config keys include `augment_enabled`, `augment_device`, `augment_seed`, `augment_shift_x`, `augment_shift_y`, `augment_rotation_degrees`, `augment_shear_x`, `augment_shear_y`, `augment_scale_min`, `augment_scale_max`, `augment_smooth_offset`, `augment_smooth_offset_stride`, `augment_brightness`, `augment_contrast_min`, `augment_contrast_max`, `augment_gamma_min`, `augment_gamma_max`, `augment_noise_std`, and `augment_blur_sigma`.
- Default augmentation extrema are `+-patch_width/4` px horizontal offset, `+-patch_height/4` px vertical offset, `+-180` degree rotation, `+-1` px/px shear, `sqrt(0.5)x..sqrt(2.0)x` scale, smooth curve offset up to `+-8` px with 16 px control stride, `+-0.25` valid-range brightness offset, `0.5x..2.0x` contrast around the valid patch center, `0.5..2.0` gamma, valid-range-relative noise std up to `0.125`, and Gaussian blur sigma up to `2.0`.
- Geometric strip augmentations operate on strip coordinates before image sampling.
- Training, augment-vis, line-trace, Trace2CP, labels, TTA, and all core
  loader paths must never do geometric augmentations as image-space operations.
  No helper, function, or API on those paths may geometrically warp, rotate,
  scale, shear, translate, resize, or flip an already sampled image/tensor.
  Such geometric changes must be represented as coordinate manipulation before
  sampling/slicing the patch.
- The only image-space geometric exception is the `--dir-vis` diagnostic probe:
  it may apply pixel-perfect identity, flips, and 90-degree rotations to an
  already sampled center patch to inspect checkpoint robustness. That exception
  must not be reused for training data, labels, augment-vis, line tracing,
  Trace2CP, or TTA.
- Image-space operations after sampling are allowed only for value-only changes
  such as brightness, contrast, gamma, noise, and blur.
- Geometric augmentation builds an oversized strip-coordinate source area, maps output patch pixels into that source, and samples the volume once at the final augmented coordinates to avoid edge and image reinterpolation artifacts.
- Geometric augmentation map handling is centralized in one paired fused map
  object. In this spec, "fused map" means actual precomputed coordinate map
  tensors, not a shared bundle of transform formulas.
- The paired transform must be constructed once for the specific source shape,
  output shape, augmentation parameters, and torch device. It must store:
  `backward_map_xy` for output pixel -> source pixel sampling, and
  `forward_map_xy` for source pixel -> output pixel line/control-point lookup.
- Both geometric map directions must be built explicitly as paired concrete
  map tensors during augmentation construction. Runtime consumers must never
  invert one map direction into the other after the fact, including by
  nearest-neighbor searches, brute-force distance scans, iterative solvers, or
  analytic/formula re-evaluation. If a runner, loader, target, or visualization
  path needs the opposite direction, it must receive and sample the prebuilt
  opposite map.
- Every geometric augmentation stage is baked into those map tensors at
  construction time: translation, flips, scale, shear, rotation, and smooth
  offset. No geometric augmentation may invert coordinates with rasterized
  masks, image warps, nearest-neighbor searches over the output grid,
  brute-force distance scans, iterative solvers, or runtime analytic inverse
  formulas.
- Smooth offset augmentation is a direct paired vertical map in source-strip
  coordinates. The output-to-source map applies the smooth offset as
  `source_y += f(source_x)`, and the source-to-output point map applies the
  inverse as `source_y -= f(source_x)` before the affine forward map. It must
  not require iterative solving or dense nearest-grid inversion. Smooth control
  generation/interpolation is allowed only while constructing the fused map
  tensors; it must not run during line/control-point lookup.
- Affine geometric shift is an output-space translation applied after scale/flip, not a source-space translation before scale. Combined shift+scale must keep image sampling, transformed line coordinates, and transformed control-point coordinates under that same composition.
- Training line targets and debug line overlays are geometric coordinate products, not raster images. The line must be represented by strip/output pixel coordinates after the same geometric coordinate transform used for image sampling.
- Transformed line/control-point coordinates are computed from cached
  source-space line/control-point coordinates by bilinear lookup/interpolation
  against the precomputed `forward_map_xy`. Smooth-offset line/control-point
  mapping must not run smooth interpolation, evaluate affine transform formulas,
  or invert the patch by dense output-grid nearest-neighbor search.
- Line points and the control point for a patch must be transformed together in
  one vectorized lookup call through the fused map object, then split back into
  line and CP outputs.
- Sparse line/control-point mapping must use direct bilinear gather against
  `forward_map_xy`. Tiny `grid_sample` calls for sparse point lists are not the
  intended implementation because their fixed launch overhead dominates the
  small amount of point data.
- When multiple patches from one loader sample need transformed line/control
  point coordinates, their `forward_map_xy` tensors and source point lists
  should be stacked and processed as a batched sparse lookup where shapes are
  compatible.
- Coordinate augmentation should stack `backward_map_xy` tensors and run
  batched dense sampling for compatible strip-z offset patches.
- When multiple strip-z offsets share the same CP source geometry and the same
  augmentation parameters, transformed line/control-point coordinates must be
  computed once and reused across those offsets.
- The line must never be transformed by resampling a raster line mask. No geometric augmentation may be implemented as an image-space transform of a previously rasterized line, mask, or image patch.
- Debug visualization may rasterize the transformed line coordinates only as the final drawing step, with fixed screen-space thickness/opacity, so line thickness and sharpness are not affected by scale, rotation, shear, or interpolation artifacts.
- Any future training target derived from the fiber line must use the same transformed output pixel coordinates as the sampled image, so labels and image pixels remain aligned exactly.
- Image/value augmentations after Zarr loading run as torch tensor operations on the configured device.
- Value augmentations after VC3D image loading should run as batched tensor
  operations where possible. Variable per-patch Gaussian blur uses grouped
  batched convolutions instead of one CUDA convolution loop per patch.
  Per-patch operations remain acceptable only when required to preserve behavior,
  such as deterministic per-patch noise streams.
- VC3D coordinate sampling remains the explicit image I/O boundary unless the
  sampler exposes a true batched coordinate API. The loader must not add a
  separate image sampling path just to batch around that boundary.
- Augment visualization uses raw clipped image values and must not apply percentile or per-cell normalization.
- The augment visualization mode renders a three-row JPG contact sheet: lower-limit examples, upper-limit examples, and random combined training-style examples.
- Augment visualization prints no timing diagnostics by default.
- `--augment-vis --augment-profile` enables timing diagnostics. Profile mode runs the same sample and augmentation entries twice, prints a pass 1 table for cold/first-use costs and a pass 2 table for warmed costs, and each table includes per-entry rows plus full total/average-per-patch summaries and `total/no-first` plus `avg/no-first` summaries that exclude the first unaugmented row.
- Augment contact sheets draw the transformed fiber-line coordinates at 50 percent opacity with fixed drawing thickness.
- Augment contact sheets draw a final visualization-only thin vertical marker at the transformed control-point coordinate for each patch, leaving a small gap around the CP pixel itself.
- Augment contact-sheet cells include a top label band naming the shown augmentation; labels must not overlay image pixels.
- Dataset entries include `fiber_paths` or `fiber_glob`, `base_volume_path`,
  and `base_volume_scale`; `lasagna_manifest_path` is required for 2D strip
  training/Trace2CP normals but optional for 3D CP-volume training.
- Dataset entries may define a fiber-coordinate affine transform using one of
  `fiber_transform_json` / `fiber_transform_json_path` for Vesuvius
  registration `transform.json`, inline `fiber_transform`, or Lasagna-compatible
  inline `transform`. Inline matrices are XYZ 3x4 or homogeneous 4x4. The
  matrix direction is source/moving fiber XYZ to current/fixed base-volume XYZ;
  `fiber_transform_invert` or `transform_invert` inverts it before use.
- Fiber-coordinate transforms are applied once immediately after JSON/NML
  parsing and before bounds checks, sample ordering, strip-coordinate cache
  identity, prefetch, training, and Trace2CP tooling. Lasagna manifest normals
  are still sampled from the current manifest after transformation.
- Optional top-level `test_datasets` uses the same dataset-entry schema as `datasets`; when present it defines a separate deterministic test loader while reusing the rest of the loader configuration.
- Strip-frame normals are sampled only through the Lasagna manifest `grad_mag`, `nx`, and `ny` channels.
- Trace2CP segment samples carry the line-window original line indices and the
  signed Lasagna normals used to build the segment strip, plus the actual
  start/target strip row-axis vectors after VC3D-style frame construction.
  Single-pair Trace2CP may choose either valid sign, but whole-fiber Trace2CP
  must align each later pair-local row axis to any already accepted shared-CP
  row-axis reference before sampling image data. If the initially built grid's
  row-axis disagrees with that reference, the loader flips the local Lasagna
  normal sequence and rebuilds the grid. This prevents adjacent segment images
  from flipping in y solely because Lasagna normals are sign-ambiguous.
- Normal batch loading samples control points in deterministic pseudo-random order from the configured seed.
- The deterministic pseudo-random training order covers every configured control point exactly once per dataset pass before repeating.
- Changing training step counts, batch size, or control points per step must only truncate or extend the consumed prefix of that deterministic sample stream; it must not reshuffle earlier samples.
- `load_batch` may parallelize CP-sample construction with `loader_workers`,
  defaulting to the machine logical CPU count. `loader_workers=0` explicitly
  requests all logical CPU cores. Parallel workers may evaluate candidate
  samples out of order, but accepted output samples and skip handling must
  follow the same deterministic sample-index order as serial loading.
  `loader_workers=1` is the serial no-thread debug path. When
  `loader_workers > 1`, the loader reuses a persistent CP-level executor across
  batches instead of constructing a new thread pool per step.
- The same `loader_workers` setting also controls startup compact-geometry
  record construction. Startup uses process workers, while warm
  `load_batch` CP-sample construction uses the persistent in-process thread
  executor above. No separate startup worker-count key exists.
- Parallel loader workers must not serialize on deterministic random-order
  locks during the warm path. Random dataset-pass orders are built once per
  pass, cached by pass index, and prewarmed for the attempted batch window
  before CP workers are submitted.
- The tester/runner loads a batch from a specified deterministic control-point sample index.
- Prefetch uses the same shared source-strip implementation as training and augment-vis.
- Prefetch remains CPU-pinned, but it still uses the same torch-native source-grid and strip-offset path, converting to NumPy only once for VC3D dependency discovery.
- Prefetch is independent of any one random augmentation draw: for each selected CP and strip-z offset it covers the configured maximum augmentation envelope represented by the oversized source-strip coordinates. In 3D this same rule means one CP-centered volume envelope per sample, not one concrete sampled 3D augmentation.
- Prefetch may conservatively cover more chunks than one concrete augmented training sample, but it should avoid misses for later random augmentations within the configured extrema.
- Prefetch must use dependency-only chunk discovery for the base-volume sampler. For VC3D this means `collect_coords_dependencies` over the same conservative source-envelope coordinates, without `sample_coords`, image-value sampling, or discarded sampled pixels.
- VC3D dependency discovery must return explicit per-chunk metadata for Python prefetch: remote chunk key/URL, final persistent cache data path, `.empty` marker path, persistent extension, and cache payload format.
- Python prefetch must not reconstruct VC3D persistent cache paths or remote chunk keys; it consumes the metadata returned by VC3D dependency discovery.
- Python prefetch currently supports only direct-source uncompressed chunks whose remote payload is exactly the persistent `.bin` payload VC3D expects. Compressed, filtered, sharded, byte-swapped, or otherwise non-direct payloads must fail clearly until explicit codec support is added.
- Normal image loading still uses the VC3D blocking coordinate sampler, so training/export samples are decoded before image sampling returns.
- Prefetch classifies VC3D persistent-cache files in Python: existing data files are cache hits, existing `<cache>/level_<level>/<iz>/<iy>/<ix>.empty` files are known-missing hits, and definitive missing chunks write that same zero-byte `.empty` marker.
- Prefetch data downloads are written to unique temporary files in the final cache directory and then atomically renamed to the VC3D-provided final cache path.
- VC3D also writes persistent-cache `.empty` markers as zero-byte files and reads them by existence.
- Prefetch performs global chunk deduplication by store identity and chunk key before network work.
- Python Zarr prefetch must preserve store-relative v2 chunk keys and raw chunk
  bytes under both supported Zarr 2 and Zarr 3 APIs; version-specific key and
  store access belongs in one shared helper rather than individual loaders.
- Prefetch runs parallel dependency producers plus bounded chunk download workers; download worker count is controlled by `prefetch_workers` without an additional hard-coded cap, and dependency/sampler producer count is controlled by `prefetch_sampler_workers`.
- During prefetch, PyTorch CPU intra-op threads are temporarily forced to `1`
  while dependency producers run, then restored. This prevents each producer
  from fanning out over the full machine and makes `prefetch_sampler_workers`
  the practical CPU-side source/dependency generation limit.
- Prefetch may generate dependency requests with parallel producers, but producer
  results are consumed in raw deterministic sample-index order before chunks are
  classified or enqueued for download.
- Prefetch schedules not-yet-submitted chunk downloads by the earliest raw
  deterministic sample index that requested the chunk. This keeps downloads as
  close as practical to `idx` order and avoids burying earlier-sample chunks
  behind a large later-sample executor backlog. Transfers already active are
  not cancelled or restarted for reprioritization.
- Prefetch reports sample/dependency progress only while dependency generation is incomplete; once all requested samples have been processed or skipped, live progress reports only download progress. The live progress includes unique chunks, cache hits, known-missing chunks, downloaded chunks, queued download futures, configured transfer worker count, configured sampler/dependency producer count, skipped samples, errors, and MiB/s. The download denominator is the number of chunks that were not cache hits or pre-existing `.empty` markers and therefore needed fetch/missing resolution. While dependency generation is incomplete, download ETA extrapolates from observed chunks per sample and observed cache-hit/known-missing/download-needed ratios.
- Prefetch reports skipped invalid samples separately from download errors and includes the first skip reason.
- If prefetch hits a fatal producer error, queued producer/download futures are cancelled so shutdown does not wait on a large stale download backlog.
- Prefetch remains base-volume-only; Lasagna manifest channels are not prefetched by the VC3D base-volume prefetch path.
- V0 training is provided by `python -m vesuvius.neural_tracing.fiber_trace_2d.train`.
- `train.py --prefetch` runs training-oriented chunk prefetch only and exits before model, optimizer, TensorBoard, run-directory, or snapshot setup.
- `train.py --benchmark` runs 100 training batches and exits without test evaluation, TensorBoard, run-directory creation, or snapshot setup. It reports throughput in CNN image patches per second, where one patch is one flattened strip-z image sent through the 2D model.
- `train.py --profile` runs the same 100-batch benchmark path with per-batch timing rows and a final average milliseconds-per-patch summary. Profiled stages are aggregate coordinate generation, descriptor lookup, strip-coordinate cache load, source geometry generation, line-coordinate generation, coordinate augmentation, base-volume Zarr read/sampling, torch image/value augmentation, forward plus loss, and backward plus optimizer step.
- When loader CP parallelism is enabled, profile rows report both loader wall
  time and summed loader worker time. The threading factor is
  `worker_time / wall_time`; stage columns such as descriptor/cache/load remain
  summed worker timings and must not be interpreted as wall time under
  parallel loading.
- Profile rows also report whole-process CPU time for the benchmark batch. The
  process CPU factor is `process_cpu_time / batch_wall_time`; compare this
  value against system CPU-utilization monitors when checking whether loader
  work is actually keeping CPU cores busy.
- `train.py --load-only` runs the same 100-batch benchmark loader path and exits without test evaluation, TensorBoard, run-directory creation, snapshots, image/value augmentation, image normalization, supervision building, model forward, backward, or optimizer work. It still performs deterministic sample selection, CP-local source construction, coordinate augmentation, and base-volume sampling so loading bottlenecks can be isolated. When `training.pipeline_enabled` is true, load-only benchmarks use the bounded whole-batch queue so loader parallelism can be measured without model work.
- Training and training prefetch use the same deterministic pseudo-random CP sample-index sequence: each pass visits all configured CPs once in seeded random order and wraps at dataset end.
- With `training.max_steps = 0`, training repeats the full training dataset indefinitely.
- `training.max_sample_index` is an optional positive exclusive deterministic
  `data_index` limit. The default `0` means no limit. When positive, training
  maps every unbounded `stream_index` through
  `stream_index % training.max_sample_index` only for CP/data selection, so
  long runs reuse that deterministic CP/data prefix independently of
  `training.max_steps`. The limit does not bound any random source: geometric
  and value/image augmentation draws, branch-choice grid offsets, jitter, and
  noise are keyed by the unbounded `stream_index`, so repeated use of the same
  bounded `data_index` gets fresh deterministic random parameters instead of
  replaying the same transform.
- Explicit positive `--prefetch-steps N` overrides `training.max_steps` and prefetches exactly `N * training.control_points_per_step` CP samples from the deterministic random training stream.
- Explicit `--prefetch-steps 0` overrides `training.max_steps` and prefetches every configured training-dataset CP once, independent of `control_points_per_step`; if `training.max_sample_index` is positive it prefetches that bounded deterministic prefix instead. When `test_datasets` is configured, it also prefetches every held-out test CP once.
- If `--prefetch-steps` is omitted, prefetch uses `training.max_steps`; if that configured value is `0`, omitted prefetch also means every configured training/test CP once.
- Negative `--prefetch-steps` values are invalid.
- The V0 trainer uses `FiberStrip2DLoader` batches directly; it must not use the neural-tracing 3D crop loader or a separate image sampling path.
- In training and benchmark modes, top-level `batch_size` is the number of CP
  samples loaded per step and must match `training.control_points_per_step`.
  Non-default flattened CNN patch counts are valid and must not warn; the
  flattened CNN patch count is `batch_size * strip_z_offset_count`.
- Training geometric augmentations are the same coordinate-space augmentations used by augment-vis. Value augmentations run through the existing torch augmentation functions after Zarr sampling.
- On CUDA training runs, `training.pipeline_enabled` may overlap future batch loading with current-batch model work through a bounded deterministic producer/consumer queue. The same queue is used by `--load-only` benchmarks to measure batch-loader parallelism directly. `training.pipeline_depth` controls the number of submitted whole-batch futures and defaults to `16`. `training.pipeline_workers` controls how many whole-batch `load_batch` calls may run concurrently and defaults to `8`; `0` means use `pipeline_depth`. Whole batches are still consumed strictly by step number, so the deterministic CP sample stream is unchanged.
- `training.pipeline_isolated_loaders` defaults to `false`. The normal
  pipeline shares the base loader, parsed fiber/Lasagna metadata, deterministic
  sample-order cache, and VC3D sampler/cache across whole-batch futures.
  Setting it to `true` creates worker-local loader clones with fresh VC3D
  samplers but shared parsed records and deterministic order cache.
- `volume_cache_memory_mib` is an optional VC3D sampler cache budget. `null` or
  omission leaves VC3D's default cache behavior intact. Positive values cap
  each VC3D sampler cache, which is mainly useful when
  `training.pipeline_isolated_loaders=true` duplicates VC3D samplers.
- `volume_io_threads` optionally forwards a positive VC3D I/O thread count to
  each VC3D sampler when the installed binding exposes that control.
- The training pipeline keeps strip-coordinate generation and geometric coordinate augmentation on the configured `augment_device`. It does not move those torch coordinate operations to CPU. The only unavoidable CPU/NumPy boundary remains the VC3D coordinate sampler call after final coordinates are generated.
- The pipeline uses the shared `FiberStrip2DLoader.load_batch` path with image/value augmentation deferred. Loaded batches carry the deterministic per-patch augmentation parameters.
- CUDA training prepares loaded batches on a separate CUDA stream when `training.pipeline_enabled` is true. A background preparation executor submits deferred torch image/value augmentation, image normalization, and direction-supervision tensor construction for loaded batches, then records a CUDA event. The main training stream waits for that event immediately before forward pass.
- Normal CUDA training must keep prepared augmented image tensors on CUDA and must not round-trip deferred value augmentation through NumPy. Runner/debug APIs may still return NumPy batches for export and tests.
- Training timing logs include CPU batch load time, load-pipeline wait, preparation enqueue time, measured CUDA preparation time, preparation wait time, and preparation submit time for queuing future prepared batches. `prep_submit_ms` is main-thread queue-refill overhead; the full preparation work is represented by `prep_ms`/`prep_gpu_ms` and runs in the background preparation executor on CUDA pipeline runs. Profile mode also reports an `outside` aggregate for work outside the forward/backward/optimizer critical path.
- Normal training prints the effective CUDA pipeline enable flag, queue depth,
  whole-batch loader workers, and loader CP-worker count once at startup.
- Whole-batch loading may use multiple concurrent producers. Zarr cache tracing is thread-local, and whole-batch outputs remain ordered at consumption time.
- CPU training runs default to the synchronous path even when `training.pipeline_enabled` is true; the automatic pipeline is a CUDA training optimization.
- A training step samples `training.control_points_per_step` deterministic control-point samples and every configured strip-z offset. The default is four control points and 16 strip-z offsets, giving 64 2D strip patches.
- If a deterministic training sample is invalid because its CP-local Lasagna normal window is invalid, batch loading skips it and continues with following deterministic sample indices until the requested number of control-point samples is loaded. If too many consecutive samples are invalid, batch loading fails with a clear error.
- Training flattens control-point and strip-z dimensions into a patch batch before the 2D model forward pass.
- The default V0 direction model is a 10-block residual CNN with 64 hidden channels. It uses a 3x3 input projection, constant-width residual blocks, BatchNorm2d normalization, a final 1x1 direction projection, an optional 1x1 sheet/fiber-presence projection, and an optional 1x1 embedding projection for explicit contrastive experiments.
- Standard training uses direction supervision plus sheet/fiber presence when
  `training.presence_enabled` is true. Contrastive embedding training is
  disabled by default and must be explicitly enabled.
- V0 model output always starts with exactly two per-pixel direction channels in the Lasagna ambiguous two-cos-channel encoding. When `training.presence_enabled` is true, one sigmoid sheet/fiber-presence channel follows the direction channels. When `training.contrastive_enabled` is true and `training.contrastive_embedding_channels > 0`, raw embedding channels are appended after direction and any presence channel. A disabled contrastive config must instantiate no embedding head, even if a stale positive `contrastive_embedding_channels` value is present. Consumers must use explicit output-slicing helpers instead of hard-coded embedding offsets.
- When `training.top_view_enabled` is true, training jointly instantiates a
  second V0 model for top-view strip slices. This top-view model outputs the
  same two Lasagna ambiguous direction channels plus one sigmoid scalar channel
  interpreted as a fiber-center distance transform, not sheet/fiber presence.
  The side model output layout is unchanged.
- For strip-image tangent angle `theta`, target channels are `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`.
- Sheet/fiber presence training is enabled by `training.presence_enabled`.
  It supervises each loaded strip patch's rounded transformed CP pixel as
  presence `1`. Valid non-CP pixels inside the same shift-reachable
  CP-neighborhood rectangle used for contrastive negatives are supervised as
  presence `0`; unreachable patch edges are ignored so the network does not
  learn that CPs can never occur there. The positive-pixel BCE mean and the
  negative-pixel BCE mean have equal aggregate weight, and the combined loss is
  multiplied by `training.presence_weight`.
- Contrastive embedding training is experimental opt-in, enabled by
  `training.contrastive_enabled`.
  It requires `training.contrastive_embedding_channels > 0` and
  `training.control_points_per_step` divisible by
  `training.contrastive_control_points_per_fiber`.
- In contrastive mode, each training step loads deterministic same-fiber CP
  groups: every group contains `contrastive_control_points_per_fiber` CPs from
  one fiber, and consecutive groups are concatenated to fill
  `control_points_per_step`. Group ordering is deterministic and covers the
  effective CP set by shuffled fiber-local CP groups before repeating.
- Same-fiber CP patches keep independent geometric augmentation draws through
  unique stream indices. Value/image augmentation draws are synchronized
  within each same-fiber group so the embedding objective does not treat
  value-only appearance jitter as identity evidence.
- The contrastive embedding loss uses cosine similarity on each loaded strip
  patch's rounded transformed CP pixel. Positive terms are z-search-aware:
  for every anchor CP sample/strip-z offset, candidates are only other CP
  samples from the same fiber, across their loaded strip-z offsets. The
  already most-similar candidate is selected and trained toward cosine
  similarity `1`; same-CP offsets are not used as positives. Negative terms
  compare each CP embedding sample with one deterministic valid non-CP pixel
  from the batch and penalize cosine similarity above
  `training.contrastive_negative_margin`. Negative candidates are restricted to
  the CP-neighborhood reachable rectangle implied by the configured
  output-space `augment_shift_x/y` bounds; unreachable patch edges are ignored,
  not supervised as negatives. CP embeddings from other fibers are not used as
  negative samples. Positive and negative means are averaged equally, then
  multiplied by `training.contrastive_weight`.
- Contrastive embedding training also includes a similarity-image sparsity
  term. For each supervised CP embedding, the embedding similarity image
  against that CP is computed in normalized visualization space
  `0.5 + 0.5 * cosine_similarity`; its valid-pixel mean over the same
  shift-reachable CP area used for contrastive pixel negatives is trained
  toward the fixed target `0.1` with MSE. This term is added to the balanced
  positive/negative pair loss before applying `training.contrastive_weight`, so
  CP-similar embeddings are encouraged to stay spatially sparse without using
  unreachable patch edges as evidence.
- Presence visualization writes TensorBoard presence-probability maps when the
  presence head is enabled. Contrastive embedding visualization writes
  TensorBoard similarity maps:
  per-pixel cosine similarity against the selected patch's CP embedding is
  mapped from `[-1, 1]` to `[0, 255]` with invalid pixels black.
- Equivalent implementation formulas are `cos2theta=(dx^2-dy^2)/(dx^2+dy^2+eps)`, `sin2theta=2*dx*dy/(dx^2+dy^2+eps)`, `dir0=0.5+0.5*cos2theta`, and `dir1=0.5+0.5*(cos2theta-sin2theta)/sqrt(2)`.
- Lasagna two-channel direction decoding must use the analytic inverse:
  `cos2theta=2*d0-1`,
  `sin2theta=cos2theta-sqrt(2)*(2*d1-1)`, and
  `theta=atan2(sin2theta, cos2theta)/2`. Binned or candidate-angle lookup
  decoders must not exist anywhere in `fiber_trace_2d`.
- Forward/backward ambiguity comes from the double-angle encoding itself; `(dx,dy)` and `(-dx,-dy)` must encode identically.
- Direction targets are derived from the transformed output-pixel line coordinates produced by the same augmentation path as the image. They must not be derived from unaugmented line points for augmented patches.
- Each loaded strip sample carries the transformed control-point output-pixel coordinate. V0 direction supervision is limited to the eight neighboring pixels around that rounded transformed control-point location, filtered by image validity and patch bounds.
- The V0 loss compares predicted and target encoded channels directly with MSE over those CP-local samples; raw signed `(dx,dy)` regression and `abs(dot)` losses are not the V0 training representation. Training additionally reports folded unoriented angular error in degrees over the same supervised pixels, with `0` degrees perfect and `90` degrees maximally wrong.
- Top-view training loads one top-strip patch per loaded CP sample, using the
  same deterministic CP ordering and the same geometric/value augmentation
  parameters as that CP's center side-strip sample. The top strip is sampled
  with the VC3D-style `lineSurface`/top-strip coordinate construction already
  used by Trace2CP visualization: columns follow the fiber line and rows follow
  the side/cross-fiber axis derived from Lasagna normals.
- Top-view training uses the same cached, vectorized, batched loader mechanics
  as side-view training. The only source-coordinate difference is the grid
  builder: top view uses the top-strip builder, side view uses the side-strip
  builder. Top-view coordinate augmentation stacks maps and tensors through the
  same batched coordinate-resampling helper, and top-view image loading is
  grouped through `CoordinateSampler.sample_coord_batch`.
- Because a top patch and its center side-strip patch use the same source/output
  pixel frame and the same geometric augmentation parameters, the transformed
  fiber line and CP pixel coordinates are identical. Top-view batch loading
  must reuse the already computed side-sample line/CP coordinates instead of
  running a second line-coordinate lookup for the top patch.
- Top-view direction supervision uses the transformed top-strip line tangent
  and the same Lasagna ambiguous two-channel MSE objective as side strips.
  Top-view distance-transform supervision uses only the rounded normal
  cross-section through the transformed CP. Its target is `1.0` at the CP,
  falls linearly to `0.0` at `training.top_view_dt_radius_px` pixels
  (default `30.0`), and remains explicitly supervised as `0.0` for valid
  rounded-line pixels beyond that radius. The top direction and DT losses are
  multiplied by `training.top_view_direction_weight` and
  `training.top_view_dt_weight`.
- Training creates a run directory from `training.run_path` and `training.run_name` plus a date string. Passing `--resume <snapshot.pt>` creates and names a fresh run directory the same way, restores model and optimizer state from the snapshot, starts from `checkpoint_step + 1`, and keeps `training.max_steps` as the absolute target step. To continue past a finished run, increase `training.max_steps` before resuming. If two runs start in the same second, a numeric suffix is added to avoid a run-directory collision.
- Training config keys include `max_sample_index` for bounded deterministic-prefix reuse, `pipeline_enabled`, `pipeline_depth`, `pipeline_workers`, and `pipeline_isolated_loaders` for CUDA training load/model overlap, and `test_interval`, `test_control_points`, `test_start_sample_index`, `test_trace2cp_step_px`, and `test_trace2cp_rf_margin_px` for deterministic test evaluation when `test_datasets` is configured.
- Test evaluation runs at step 1, every `training.test_interval`, and the final step when `test_datasets` is configured. Positive `training.test_control_points` values load the fixed deterministic random range starting at `training.test_start_sample_index`, so the same held-out CP samples are compared across time. `training.test_control_points: 0` is the full-test sentinel: it evaluates every configured held-out CP sample once in flat CP order starting at zero, ignoring `training.test_start_sample_index`, so whole-fiber test metrics can be compared directly against `--trace2cp-vis --fiber-json` on the same held-out fiber apart from pair-alignment details. In addition to fixed-batch direction loss, the test path evaluates the public Trace2CP metric by tracing each selected held-out CP to its next CP segment and averaging valid `trace2cp_error` values.
- TensorBoard logging writes the training config JSON as text, direction-loss scalars, angular-error degree scalars, timing/cache diagnostics, and batch direction overlay images at configured intervals. Batch direction overlays show the transformed fiber centerline as context and one short network-predicted direction segment at the transformed CP; they do not draw CP-neighborhood supervision boxes or extra CP markers. Overlay contact sheets select examples across loaded control-point samples first, preferring each CP's strip-z offset closest to zero before showing additional offsets. When `test_datasets` is configured, TensorBoard also logs `test/loss_direction`, `test/angle_error_mean_deg`, `test/supervision_samples`, test cache diagnostics, and a `test/batch_direction_overlay` image at test evaluation steps.
- When top-view training is enabled, TensorBoard also logs top-view
  direction/angle/DT scalars and writes top-view image summaries: a GT-line
  plus predicted-direction overlay and a fixed `0..1` DT scalar map for train
  and test batches.
- Console training progress prints every one of the first 100 training steps,
  then falls back to `training.scalar_log_interval`.
- Prefetch progress includes `idx=<exclusive-index>` showing the largest
  contiguous exclusive deterministic training-stream prefix whose required
  chunks are cache-complete. This index is counted through the seeded shuffled
  CP stream used by training, before mapping a stream position to its original
  flat fiber/CP id. Operators can use that value as
  `training.max_sample_index` to train on the same prefetched random-prefix
  stream. A stream sample is cache-complete only after every dependency request
  for that sample has been classified and each required chunk is a cache hit, a
  known/new missing marker, or a completed successful download. Dependency
  generation alone must not advance `idx` while downloads are still pending.
  When `training.top_view_enabled` is true, prefetch dependency generation
  includes the top-view strip envelope in addition to all side-strip z-offset
  envelopes, and both views must be complete before that sample can advance the
  prefix.
- Training writes snapshots under `<run_dir>/snapshots/current.pt` and `<run_dir>/snapshots/best.pt`. With `test_datasets`, current snapshots are written at the test evaluation cadence and best is selected by lowest observed averaged `test/trace2cp_error`. Without `test_datasets`, current snapshots use `training.checkpoint_interval` and best is selected by lowest observed training loss. `training.kept_snapshot_interval` defaults to `10000` and writes retained numbered snapshots named `step_<iteration>.pt`; `0` disables retained numbered snapshots. A resumed run writes its own fresh `current.pt`, `best.pt`, and retained numbered snapshots under the newly created resumed run directory.
- The runner is `python -m vesuvius.neural_tracing.fiber_trace_2d.runner`.
- Augment contact sheets are exported with `--augment-vis --export-dir <dir>`. Add `--augment-profile` to print cold and warm augment timing tables.
- Direction-field inspection is exported with `--dir-vis --checkpoint <snapshot> --export-dir <dir>`.
- Direction-field inspection uses the same deterministic `--sample-index` ordering as training, prefetch, augment-vis, and line-trace-vis. It loads the center side-strip patch, center-crops it to the largest native square when needed, applies pixel-perfect image-space identity, flip-x, flip-y, rot90, rot180, and rot270 variants, runs the checkpointed direction model on each native-resolution variant, decodes the Lasagna ambiguous two-cos-channel output, nearest-neighbor scales each augmented patch image by 4x for visualization only, and draws short direction line segments on top.
- Direction-field inspection draws only every second source pixel in x and y, so each drawn sample corresponds to an 8x8 display-pixel cell in the 4x visualization. It draws anti-aliased 6-display-pixel direction segments, skips invalid image pixels and invalid/non-finite decoded directions for arrow placement only, writes the augmented variants as one natural-size horizontal `dir_vis.jpg` strip with a single top label band, and writes sample/checkpoint/per-augmentation drawn-count metadata to `dir_vis_summary.txt`. The valid mask gates model normalization and arrow placement, but does not black out display pixels.
- `--dir-vis --dbg-dirs` adds a second row to `dir_vis.jpg`. Column 1 is the raw unaugmented patch without direction arrows. The remaining columns copy the unaugmented center crop whose side is half the center-patch image side into the center of each transformed patch, run inference on those pasted variants, and render their direction overlays.
- V0.1 patch line-tracing inspection is exported with `--line-trace-vis --checkpoint <snapshot> --export-dir <dir>`.
- Line-tracing inspection uses the same deterministic `--sample-index` ordering as training, prefetch, and augment-vis, loads the center side-strip patch, runs the checkpointed direction model, decodes the Lasagna ambiguous two-cos-channel output, and traces from the transformed CP in both directions.
- The line tracer bilinearly samples the decoded per-pixel direction field, flips sampled directions as needed to maintain forward/backward sign continuity, and steps in strip-pixel coordinates.
- The line tracer stops when the next point would enter the configured receptive-field border margin, when the sampled direction is invalid, or when image validity around the bilinear sample is insufficient. By default the receptive-field margin is `model_depth`; `--line-trace-rf-margin` can override it for inspection.
- The default line-trace step is `4.0` strip-image pixels and can be overridden with `--line-trace-step`.
- Line-tracing inspection writes `line_trace_vis.jpg` as a two-column image by default: the first column is the original transformed strip line plus the unaugmented direction-traced line, and the second column is the same original patch with a flock of traces from random combined geometric test-time augmentations mapped back through TTA output-to-reference coordinate grids.
- Line-trace test-time augmentations are deterministic per `sample_index` and are sampled from the regular training geometric augmentation ranges: shift, rotation, shear, scale, smooth offset, and flips. Value-only augmentations are not applied for line tracing.
- `--line-trace-tta-count` controls the number of random geometric TTA variants and defaults to `100`. `--med-tta-count` is accepted as a compatibility alias for the same count.
- Line-trace TTA constructs augmented coordinate grids from the base patch coordinates, samples the volume at those coordinates, runs the model and tracer in augmented patch coordinates, then uses the TTA output-to-reference coordinate grid to map traced points back into original patch coordinates before drawing. It writes per-TTA trace counts to `line_trace_summary.txt`.
- `--line-trace-vis --med-tta` adds a third `line_trace_vis.jpg` column for
  median test-time augmentation tracing. Median TTA traces in the unaugmented
  reference patch space; at each trace step it transforms the current reference
  point into the reference/TTA direction fields, samples decoded Lasagna
  ambiguous directions there, transforms orientations back to reference space,
  keeps only the ambiguous sign aligned within 90 degrees of the previous
  reference-space step direction, then takes and normalizes the component-wise
  median direction before stepping. The median trace uses the same random TTA
  field list as the flock column. `line_trace_summary.txt` records
  `med_tta=true`, `line_trace_tta_count`, and the median trace point count.
- Trace-to-control-point inspection is exported with `--trace2cp-vis
  --checkpoint <snapshot> --export-dir <dir>`.
- Trace2CP inspection uses the same deterministic `--sample-index` ordering as
  training, prefetch, augment-vis, dir-vis, and line-trace-vis. The sampled
  control point is the start CP. The default target is the next control point
  in the same fiber; `--trace2cp-target-offset` changes the relative target and
  `--trace2cp-target-cp-index` selects an absolute target CP index in the same
  fiber. The target CP must be in range and different from the start CP.
- `--trace2cp-vis --fiber-json <path>` runs whole-fiber Trace2CP visualization
  for an explicit fiber JSON. In this mode the runner narrows a single-dataset
  config to `fiber_paths=[<path>]` before constructing the loader, so it loads
  only that fiber while reusing the configured Lasagna manifest, volume scale,
  cache, and sampler context. It must not require the fiber to match the
  configured dataset glob/list, and it must not introduce a separate
  manifest-less fiber loading path. Whole-fiber mode uses all in-range CP pairs
  for the non-zero `--trace2cp-target-offset`; the default offset `1` evaluates
  adjacent pairs `(0,1), (1,2), ...`. It cannot be combined with
  `--trace2cp-target-cp-index`.
- Whole-fiber Trace2CP must continue past CP pairs whose segment cannot be
  constructed or traced because of invalid local data such as zero
  Lasagna `grad_mag` normal samples. Skipped pairs are reported to stdout and
  listed in `trace2cp_fiber_summary.txt`; the command fails only if every
  requested pair is skipped.
- Trace2CP loading constructs a side-strip segment that spans the start and
  target CPs plus receptive-field/visualization margin. The segment strip
  height is eight times the configured patch height so traces have more vertical
  room before entering the RF margin. It uses the same
  Lasagna manifest normal sampling and VC3D-equivalent side-strip coordinate
  construction as CP-local patches, but anchors the start CP at an explicit
  strip x-coordinate so the target CP lies in the same image at its arc-length
  column. It does not use the neural-tracing 3D crop loader.
- Trace2CP samples the center strip-z image only, runs the checkpointed
  direction model, decodes the Lasagna ambiguous two-cos-channel output, and
  traces the same selected segment in both directions: start CP to target CP,
  and target CP back to start CP on the same segment strip. Each directional
  trace uses `--line-trace-step` and the line-trace receptive-field margin
  default `model_depth`, overrideable with `--line-trace-rf-margin`.
- Each Trace2CP direction initializes its ambiguous-direction sign from the
  vector pointing from that direction's start CP to its target CP. The reverse
  trace therefore starts at the second CP and is seeded toward the first CP.
- Per-direction Trace2CP endpoint diagnostics still evaluate the traced y
  coordinate at that direction's target CP x-column. If the trace crosses that
  x-column, the y coordinate is linearly interpolated between bracketing trace
  points. These endpoint scores remain in the summary as diagnostics, but they
  are not the public `trace2cp_score`.
- The public Trace2CP metric is `trace2cp_error`: the mean target-column y
  error divided by the horizontal start-to-target CP span. The forward trace is
  linearly interpolated where it reaches the target CP x-column and compared
  to the target CP y. The reverse trace is linearly interpolated where it
  reaches the start CP x-column and compared to the start CP y. The two raw y
  errors are averaged before division by horizontal span.
- Target-directed Trace2CP traces must normally stop by reaching the opposite
  CP x-column. When a step crosses that column, the returned trace must append
  an exact linearly interpolated point at the target column before terminating
  with reason `target_column`.
- If a trace explicitly stops before the opposite CP x-column because it hits
  the RF margin, invalid sampled data, or an invalid predicted direction, that
  direction uses the default maximum y error for the segment: vertical distance
  from the CP centerline y to the nearest usable vertical strip edge after
  RF-margin exclusion. The same maximum y error caps pathological endpoint y
  errors. This intentionally treats exact early/late edge intersection as noise
  for now.
- If a target-directed Trace2CP trace terminates by exhausting `max_steps`, that
  is an internal budget failure and must raise a visible error. It must not be
  scored through the missing-target-column maximum-y fallback, because that can
  hide traces that stop far before the opposite CP column.
- The previous center-biased closest-approach value remains available only as a
  refinement/visualization diagnostic named `refine_score`. It must not be used
  as the public Trace2CP metric or as the training best-checkpoint criterion.
- Trace2CP builds a CP-to-CP initialized segment from the two closest-approach
  partial traces. Each CP stays fixed. Each partial trace is corrected only in
  y, with zero correction at its CP and linearly increasing correction toward
  the closest x position. At the closest x position both traces are warped to
  the midpoint between their original y values. The two warped partial traces
  are fused and resampled by arc length using `--line-trace-step`.
- Trace2CP also runs a small deterministic refinement of the fused line that
  reduces local direction mismatch against the sampled direction field while
  discouraging uneven segment spacing. The refined line keeps the two CP
  endpoints fixed.
- `--trace2cp-refine-iterations N` enables iterative Trace2CP refinement after
  the initial pass. Iteration `0` is the normal Trace2CP evaluation. Each extra
  iteration smooths the previous selected fused CP-to-CP trace, keeps the CP
  endpoints and x columns fixed during smoothing, converts the smoothed
  patch-space `(x,y,z)` trace back through the previous segment source to
  volume coordinates, builds a fresh side-strip segment from that volume-space
  curve, and reruns the same Trace2CP scoring mode. `N=0` preserves the current
  single-pass behavior.
- Iterative Trace2CP refinement must resample the volume from the refined
  curve geometry. It must not geometrically warp, bend, rotate, or otherwise
  reuse the previous strip image as the next pass input.
- A refined pass must be equivalent to running Trace2CP on an independent
  line source: after converting the previous fused trace to volume-space line
  points, the loader must build a fresh segment source with endpoint context
  before the start CP and after the target CP. Both forward and reverse traces
  must therefore have the same valid local neighborhood at their start points
  as they do for an original fiber-json line.
- `--trace2cp-refine-smooth-window` controls the finite Gaussian smoothing
  window used between iterations and defaults to `5`. Even values are rounded
  up to the next odd window. The smoothing keeps x columns and both CP
  endpoints fixed.
- Single-pair refinement outputs keep the initial pass as `trace2cp_vis.jpg`
  and `trace2cp_summary.txt`. Extra passes write `trace2cp_vis_it1.jpg`,
  `trace2cp_summary_it1.txt`, then `it2`, etc. If z-layer TIFF export is also
  enabled, extra passes write `trace2cp_z_layers_it1.tif`, etc.
- Trace2CP CLI runs print a compact stage timing table after the metric/debug
  summary. Single-pair mode prints `trace2cp timings`; whole-fiber mode prints
  `trace2cp fiber timings` aggregated across valid pairs. Rows are grouped by
  stage and include count, total milliseconds, mean milliseconds, and max
  milliseconds, covering inference, source sampling, tracing, debug rendering,
  and file output stages where applicable.
- Slow Trace2CP dynamic-programming solves print live progress before the final
  timing table. DP progress is opt-in at CLI call sites and uses rows
  `trace2cp dp start`, `trace2cp dp progress`, `trace2cp dp done`, or
  `trace2cp dp failed`. Progress rows include the DP label, solved columns,
  elapsed seconds, and `eta_s` on progress rows. The low-level DP helper is
  quiet by default for unit tests and internal direct calls.
- Trace2CP CLI side, side-z, and top-model DP solves use a torch-vectorized
  backend on the active model device. The backend keeps the DP column
  recurrence sequential but vectorizes per-column work across z layers, rows,
  sampled transition columns, and move chunks. The existing NumPy/Python DP
  remains the fallback when no torch device is supplied.
- Side-z DP must not infer or optimize unreachable z layers. Since the path is
  anchored at the center layer at both CP columns and transitions can move only
  one z layer per DP column, the effective layer bound is capped by the number
  of horizontal transitions. The side DP vertical move lattice may use a broad
  compute search band, but that band must be independent of the configured
  candidate-angle limit.
- Trace2CP uses `--med-tta` to determine whether TTA is used. Without
  `--med-tta`, it traces and scores both directions on the base strip
  direction field. With `--med-tta`, it builds deterministic random geometric
  TTA direction fields using `--line-trace-tta-count`, default `100`, and
  traces both median-TTA directions in the reference segment strip.
- Trace2CP supports an optional inspection/refinement mode enabled by
  `--trace2cp-combined` or `--trace2cp-use-presence`. In this mode the selected
  trace uses the regular stepwise candidate-fan tracer by default, scoring side
  direction at both the current/last point and the candidate point, plus
  optional presence. The non-combined reference tracer remains the public
  target-column `trace2cp_error` path. The monotone-x dynamic-programming
  backend is experimental and must only run when `--trace2cp-dp` is explicitly
  supplied.
- The side-strip DP state is `(side_z_layer, y, prev_dy, prev_dz)`. It uses
  fixed 4 px horizontal transitions, plus the exact target column, and
  integrates angle-space direction alignment cost across every crossed pixel
  column. The sampled alignment is frame-ambiguous:
  `theta = degrees(acos(abs(dot(path_tangent, layer_direction))))`, and the
  cost is
  `(theta / 10)^2 * (1 + max(theta - knee, 0) / knee)` before applying
  `direction_weight`. Transition samples use fractional bilinear
  interpolation in strip row and z-layer coordinates, not rounded nearest
  lookup. Because decoded Lasagna directions are sign-ambiguous, all
  interpolated direction-vector corners are sign-aligned to the candidate
  transition tangent before blending. Invalid or missing direction pixels add a
  fixed penalty instead of breaking the path. Side-strip DP does not apply a
  default per-step z movement penalty; its default z regularization is
  second-order dz smoothness, currently `0.5 * (dz_current - dz_previous)^2`,
  so steady z motion is allowed while abrupt z-step changes are discouraged.
- The side-strip DP still uses `--line-trace-step` only for resampling the
  selected fused output trace and for the public trace visualization density.
  It must not use `--line-trace-step` as the DP transition length.
- The side-strip DP uses the existing Trace2CP candidate-angle setting as the
  local angle-excess knee in that penalty. With the default 25 degree knee,
  10 degrees costs roughly 1, 20 degrees roughly 4, and 45 degrees roughly
  36. This setting must not cap global horizontal slope or vertical moves,
  because valid local fiber directions can be steeper than 45 degrees.
- The default side-strip DP dy smoothness penalty is zero. The default
  side-strip DP dz smoothness penalty is nonzero as described above and should
  discourage lateral/z jitter without penalizing total z travel.
- `--trace2cp-combined-mode direction` is the only active combined mode.
  `--trace2cp-combined-mode embedding`, `--trace2cp-use-embedding`,
  `--trace2cp-combined-mode image`, and `--trace2cp-use-image` are removed from
  the active tracer and must fail clearly if requested. Legacy helper
  functions may remain as inactive implementation experiments, but runner
  Trace2CP selection must not route through embedding or image similarity.
- `--trace2cp-use-presence` adds an orthogonal sheet/fiber-presence score to
  the active combined tracer. It samples the sigmoid presence probability from
  the same selected layer as the direction field and adds
  `trace2cp_combined_presence_weight * (1 - presence_probability)` to the
  candidate/transition cost. This requires a checkpoint/model output with a
  presence channel and fails clearly if the channel is absent. Visualization appends
  fixed-scale presence debug output when presence scoring is active: single-pair
  `trace2cp_vis.jpg` gets a presence column, whole-fiber
  `trace2cp_fiber_vis.jpg` gets a presence row, `0` renders black, `1` renders
  white, invalid pixels are black, and the fiber line, CPs, and selected traces
  are overlaid. When z-search is enabled, the z debug visualization must also
  show forward, reverse, and fused z-corrected presence maps selected
  column-by-column from the same trace z layers as the z-corrected image.
  Whole-fiber presence visualization must use the fused z-corrected presence
  when it is available.
- Trace2CP z-search uses raw per-layer side-presence by default. Adding
  `--trace2cp-presence-blur` makes Trace2CP use a cache-level
  Gaussian-smoothed side-presence view for presence scoring and presence
  display. The smoothing is weighted by valid pixels and runs over side-z plus
  side-image x/y. The side-z pass uses radius 21. The side-image pass uses a
  per-pixel anisotropic Gaussian rotated around the side-z axis to align with
  the local predicted side direction: radius 5 along the direction and radius 1
  across it. The kernel is symmetric, so direction sign ambiguity does not
  change the result. Non-z Trace2CP presence scoring remains unblurred because
  there is no side-z stack.
- Trace2CP visualization also appends VC3D-style top-strip output sampled from
  volume coordinates, not warped from rendered side-strip pixels. Single-pair
  `trace2cp_vis.jpg` gets a top-strip debug column. Whole-fiber
  `trace2cp_fiber_vis.jpg` gets top-strip rows stitched into the same global CP
  x-coordinate system as the side-strip rows. The original/init comparison top
  strip uses the same pair-local line window and Lasagna/VC3D frame
  construction as the side-strip segment, but rows are offset along
  `frame.side` as in VC3D `lineSurface`. Visualizations must also include a
  traced fused top strip projected to the central z slice: for each output
  column, interpolate the fused trace, sample the segment coordinate grid and
  Lasagna row-normal axis at that traced side-strip point, derive the
  top-strip side axis from traced tangent and row normal, then sample rows
  along that side axis with zero side-z offset. When z-search is active, the
  visualization additionally appends a traced fused z-corrected top strip using
  the fused trace's selected-scale `z_voxels` value as an out-of-plane side-z
  offset before the top-strip side-axis offset. This is visualization-only and
  must not change Trace2CP scoring.
- When z-search is active and the side model exposes a sheet/fiber-presence
  head, Trace2CP top-strip visualization also appends fixed-scale side-presence
  z-pillar rows below the regular top-strip slices. For each output column `x`,
  each pillar row samples one inferred side-slice layer at `(x, trace_y(x))`;
  the image height is `2 * trace2cp_z_max_layer + 1`, so `+/-40` produces an
  81 px tall z-pillar image. Separate z-pillar panels may be shown for the
  original/init trace, the traced fused central-z line, and the z-search fused
  line. For the z-search fused line, each column is shifted by that column's
  selected z value (`round(z_voxels / z_step_voxels)`), so the center row
  represents relative z=0 at the layer actually used by the trace. These rows
  are side-z-stack projections rather than true top-strip surface predictions;
  if the side presence field is broad or similar across shifted layers they
  can resemble a narrow side-presence slice. They are visualization-only, do
  not use the optional top-view model, and must not affect Trace2CP scoring,
  z-search, or training.
- `--trace2cp-top-model-dir-vis` requires a checkpoint with
  `top_model_state_dict`. It samples a fixed top-strip normal-offset stack
  around the traced fused top strip using offsets `-4..+4` selected-scale
  voxels in one-voxel steps, runs the jointly trained top-view model on every
  layer, and appends sparse direction indicators from an aligned median
  direction field. Per pixel, only valid layer directions within 45 degrees of
  image-horizontal are considered; each Lasagna-ambiguous direction is
  normalized and sign-aligned before taking the median so opposite signs cannot
  cancel. If a z-corrected fused trace is available, that trace is used as the
  stack center; otherwise the central-z fused trace is used. The same fused
  top-direction field is traced from each CP along the top-strip center row
  until the opposite CP x-column, invalid direction, edge, or max-step guard,
  and those two traces are drawn with equal visual weight on the debug panel.
  The panel also draws a monotone-x dynamic-programming path connecting the two
  CP columns on the top-strip center row. That DP path's state is
  `(top_offset_layer, y, prev_dy, prev_dz)`, so it may transition between
  neighboring top-offset layers with a fixed z-transition penalty of
  `0.1 * abs(delta_layer)` while also preferring smooth step sequences. The
  default second-order penalties are zero; the first transition has no
  smoothing cost because no previous step exists. There is no default
  absolute-y row penalty because that would bias the path toward a row rather
  than smoothing its slope. It uses fixed 8 px horizontal transitions, plus the
  exact target column, and integrates direction alignment
  cost `1 - abs(dot(path_tangent, layer_direction))` across every pixel column
  crossed by each transition, using fractional row/z interpolation from the
  direction field. The vertical transition band scales with the horizontal
  step, and start/target rows and layers are exact at the CPs. Invalid or
  missing direction pixels in the selected layer add a fixed penalty instead of
  blocking the path, so the diagnostic path still connects the CPs while
  preferring valid pixels where available.
  The visualization also appends optimized-line diagnostics derived from that
  DP path: a top strip resliced around the DP top-row path and selected
  top-offset layers, a side slice reconstructed column-wise from the same
  optimized side displacement, and matching top z-pillar plus side-column
  presence panels when side-model presence is available. The optimized side
  displacement is the sum of the selected top-offset layer and the DP
  top-row offset from the old center row. In the optimized top-strip panel,
  the optimized path is the slice center and is drawn as a straight centerline,
  not as the pre-reslice curved path. Side-slice and presence diagnostics must
  build a visualization z-plane cache whose bounds cover this combined
  optimized side displacement; using only the raw selected top-offset layer
  range can incorrectly turn out-of-cache columns black. These panels use the
  optimized line only for visualization and do not feed back into Trace2CP
  scoring.
- Z-corrected side-image and side-presence visualization helpers must infer
  requested cache layers on demand with the z-plane cache API. They must not
  treat `plane_cache.layers` as a complete layer set, because visualization
  caches may start with only the center layer populated.
  During top trace integration, ambiguous direction signs must be resolved
  before bilinear interpolation by flipping each of the four neighboring pixel
  direction samples, if needed, so it agrees with the current trace direction;
  otherwise opposite signs from the Lasagna two-cos encoding can cancel or
  flip the sampled direction. This is visualization-only and must not change
  Trace2CP scoring or z-search layer selection.
- `--trace2cp-side-top-z-experiment` is an opt-in single-pair diagnostic. It
  is exclusive: when set, the runner writes only the side/top-z experiment
  artifacts and does not run the normal Trace2CP overlay/refinement chain,
  public `trace2cp_error` export, training metric, or best-checkpoint
  selection. The
  experiment runs regular stepwise side-strip traces from both CPs while also
  carrying a selected-scale z/offset state. Side x/y stepping must use the same
  candidate fan scoring semantics as the normal forward/backward combined
  tracer: interpolate side direction at the current/last point and at each side
  candidate point from the side prediction for the current z layer, using
  ambiguity-aware two-cos direction interpolation, and include optional side
  presence scoring when `--trace2cp-use-presence` is active. It must not score
  embedding/image similarity or run DP in this diagnostic. Top
  inference must not run for all side candidates. After the side candidate is
  selected, the experiment builds one local top patch centered at that accepted
  side point and runs the checkpoint's top-view model on that patch to update
  only the carried z/offset state. The local top patch x axis is derived from the
  sampled side-view direction: the side-strip tangent is tilted within the side
  tangent/normal plane according to the side direction. The top patch keeps the
  side-strip lateral axis as the second in-plane axis, so this experiment only
  corrects angle relative to the side-view normal and does not optimize roll or
  arbitrary rotation around the fiber line. The top direction used for the
  offset update is an ambiguity-aligned weighted median over a normal
  neighborhood, default radius 20 px. The experiment writes separate
  `trace2cp_side_top_z_experiment.jpg` and
  `trace2cp_side_top_z_summary.txt` artifacts. The JPG must stay compact and
  diagnostic-specific: forward side trace with z-corrected image, backward side
  trace with z-corrected image, forward z-corrected presence, backward
  z-corrected presence, original top strip, forward traced top strip with
  z-correction, and backward traced top strip with z-correction. It must not
  draw per-step top-direction ticks there or reuse the full Trace2CP overlay
  rows for
  fused/reference/similarity/DP debug.
  The experiment additionally writes every local top slice actually used for
  z-update inference to `trace2cp_side_top_z_top_slices/` and a matching
  native-resolution direction overlay to `trace2cp_side_top_z_top_overlays/`;
  filenames are prefixed `fw_` or `bw_` by trace direction. These generated
  directories clear stale JPGs before each export. XYZ trace positions and z
  offsets are subpixel/floating-point throughout stepping; rounding is limited
  to side z-layer prediction lookup and column-wise display reconstruction.
  Because this diagnostic repeatedly samples local top patches and runs top
  model inference, it prints throttled `trace2cp side_top_z progress` rows for
  the forward and backward traces. Each row includes a small progress bar,
  accepted steps versus expected horizontal steps, top-patch and invalid counts,
  current z offset, elapsed time, ETA, and the final termination reason.
- Combined Trace2CP is an inspection/refinement path. It does not replace the
  public `trace2cp_error` definition, the direction-only tracer, training loss,
  or best-checkpoint selection unless explicitly enabled by the command-line
  flag. `--med-tta` is supported only by the stepwise combined tracer, not by
  the explicit `--trace2cp-dp` backend.
- `--trace2cp-vis --trace2cp-combined --trace2cp-z-search` enables an
  experimental side-strip z-search mode. It requires combined tracing and
  cannot be combined with `--med-tta`. By default this is the regular stepwise
  candidate-fan z-search that existed before the DP experiment: each accepted
  side step may choose the current or neighboring z layer. Existing Trace2CP
  commands without `--trace2cp-z-search` keep the center strip-z image-only
  behavior. Adding `--trace2cp-dp` switches this z-search to the experimental
  monotone DP backend.
- Trace2CP z-search derives additional segment-strip planes from one accepted
  center segment source. The center source is built once from the CP-to-CP
  line window and Lasagna normals, including the row-axis sign alignment used
  for whole-fiber Trace2CP. Side-strip axes are explicit: image x follows the
  fiber tangent/arc direction, image y follows the Lasagna mesh-normal row
  axis, and z-search layers move along the remaining out-of-plane side axis
  aligned with the VC3D frame side direction, approximately
  `mesh_normal x tangent`. State layer `k` represents
  `z_voxels = k * --trace2cp-z-step-voxels` along that axis. Volume/model
  inference must run at no finer than one selected-scale voxel spacing: when
  `--trace2cp-z-step-voxels >= 1.0`, the requested state layer is sampled
  directly by adding `side_axis_zyx[y,x] * (z_voxels * volume_spacing_base)`
  to every center coordinate before volume sampling; when
  `--trace2cp-z-step-voxels < 1.0`, only the bracketing integer
  selected-scale side-z voxel offsets are sampled and inferred. Direction and
  sheet/fiber-presence fields for sub-voxel state layers are interpolated from
  those integer layers, with ambiguous direction vectors sign-aligned before
  interpolation and normalized afterward. It must not use the side-strip
  image-y/row axis, a global normal, a row-coordinate approximation, an
  image-space shift, or an unrelated rebuilt plane. The default
  `--trace2cp-z-step-voxels 1.0` means layer `k` is offset by `k`
  selected-scale voxels along the segment strip side-z axis.
  `--trace2cp-z-max-layer` bounds lazy expansion and defaults to `4`.
- Default z-search lazily samples side-strip layers as the stepwise candidate
  tracer requests the current and neighboring z layers. Inference is
  deterministic and stores each layer's sampled image, valid mask, decoded
  direction field, and optional presence field. For sub-voxel z steps, lazy
  sampling stores both requested state layers and the integer inferred layers
  used to build them. Each selected path point carries `x`, `y`, and
  selected-scale `z_voxels`; direction and presence costs are sampled from the
  selected state layer, which may be interpolated from neighboring inferred
  integer side-z layers. If presence is used, those state-layer presence fields
  are first smoothed over side-z and strip x by the cache-level presence blur.
- With explicit `--trace2cp-dp`, z-search infers the bounded reachable layer
  stack for the pair before running the side DP. The DP may transition between
  neighboring z layers without an absolute z movement penalty, while its
  default dz smoothness term discourages abrupt z-step changes. DP output is
  already a fused CP-to-CP path, so the optimized visualization row uses that
  joint path directly.
- Z-search does not change the public `trace2cp_error`, training test metric,
  or best-checkpoint selection. Those remain target-column y error per
  horizontal CP span.
- Single-pair z-search visualization adds a z-corrected column. It contains
  separate forward and reverse views because each trace direction can choose a
  different z layer per x column. It also contains a fused z-corrected view and
  a fused z-layer map row so the selected layer per output column is visible
  even when neighboring sampled planes look similar. Each z-corrected image is
  assembled column-by-column by rounding the trace/fused z value to the nearest
  z-search state layer and copying that state's sampled image column. For
  sub-voxel z steps, interpolated states reuse the nearest integer-inferred
  side-z image. It must not re-sample the volume and must not interpolate image
  values between z layers; columns without a trace/fused z value and columns
  whose rounded layer is missing render black and are counted in summary/debug
  output.
- `--trace2cp-vis --trace2cp-z-search --trace2cp-z-layers-tif` exports the
  already inferred z-search layer cache as TIFF debug stacks. Single-pair mode
  writes `trace2cp_z_layers.tif`; whole-fiber mode writes one pair-local TIFF
  per valid pair under `trace2cp_z_layers/` because segment strips can have
  different shapes. Pages are uint8 and non-interleaved: all sampled slice
  images first in sorted inferred z-layer order, then all available
  sheet/fiber-presence maps in the same sorted inferred z-layer order. For
  sub-voxel z steps, the TIFF stack exports only the actually inferred integer
  side-z layers, not every interpolated DP/search state. The export must use
  the existing z-search cache, must not re-sample the volume, and must not
  interpolate image values between z layers.
- `--trace2cp-vis --trace2cp-obj` is an opt-in single-pair diagnostic export.
  It writes vertex-colored OBJ meshes under `trace2cp_obj/` plus a manifest.
  OBJ geometry must come from the same sampled Trace2CP coordinate grids used
  for image loading: center side strip, z-search selected side-strip columns,
  original top strip, traced fused top strip, and z-corrected traced top strip
  when those surfaces exist. Vertex colors are grayscale scalar values from the
  corresponding volume image (`0..255`) or side-model sheet/fiber presence
  (`0..1` for raw presence, `0..255` for z-corrected debug presence). Quad
  faces are emitted only where all four vertices are valid. The flag is not
  currently supported by whole-fiber `--fiber-json` Trace2CP output.
- Single-pair `trace2cp_vis.jpg` includes an additional embedding-debug column
  when the checkpoint exposes embedding channels. The column renders cosine
  similarity maps for the start CP embedding, target CP embedding, same-fiber
  CP-bank mean similarity when the combined Trace2CP bank is available,
  forward trace-progress last-point columns, and reverse trace-progress
  last-point columns. For the forward/reverse panels, each newly placed trace
  point paints the vertical column band around itself using the previous
  accepted trace point's embedding as the similarity reference; the band radius
  is `ceil(step_px / 2)`, and small overwrites are allowed.
  These maps are fixed-scale cosine displays (`-1..1` mapped to
  `0..255`) and are visualization-only; they must not affect tracing,
  refinement, metrics, or best-checkpoint selection.
- Trace2CP TTA samples from the regular training geometric augmentation ranges
  but forces y-shift to zero and scale to one for long-strip target-column
  semantics. Each TTA field is built by transforming the segment coordinate
  grid first, then sampling the volume at those coordinates. It must not warp
  the already sampled base segment image.
- Trace2CP TTA output canvases are sized so the transformed base segment-strip
  corner footprint fits in the TTA image. Pixels that map outside the base
  coordinate strip or volume stay invalid/black.
- The Trace2CP median trace is stepped in the reference segment strip by
  sampling the reference and TTA direction fields, mapping each current
  reference trace point into each TTA field through the prebuilt
  reference-to-TTA map, mapping TTA directions back to reference coordinates
  through each TTA output-to-reference coordinate grid, resolving ambiguous
  signs against the previous step, and using the normalized component-wise
  median direction. It must not locate reference points in TTA fields by
  scanning the dense output-to-reference grid.
- `--trace2cp-vis --med-tta --vis-tta` writes `trace2cp_tta/reference.jpg`,
  one `trace2cp_tta/random_NNN.jpg` per generated TTA field, and a contact
  sheet. Each TTA debug image shows the sampled TTA slice with the transformed
  base-strip corner outline and start/target CP markers.
- Trace2CP writes `trace2cp_vis.jpg`, writes `trace2cp_summary.txt`, and prints
  a dedicated public-metric stdout line beginning with `trace2cp_error=...`.
  Additional stdout lines are diagnostics and must not duplicate the selected
  public metric label. The summary includes sample index, fiber path,
  start/target CP indices, trace mode, public `trace2cp_error`,
  target-column metric raw y error in pixels, horizontal CP span, refinement
  diagnostic score, endpoint diagnostic scores, per-direction raw errors,
  target x-columns, reach statuses, termination reasons, and trace point
  counts. The JPG is a labeled vertical stack with rows for full bidirectional
  traces, partial traces up to the closest point, the fused CP-to-CP line, and
  the optimized refinement. Without `--med-tta`, this stack is the
  reference-only inference result. With `--med-tta`, the JPG has two columns:
  the selected median-TTA result first, and a second reference-only inference
  column using the base direction field without TTA. It does not draw score
  text over image pixels.
- With `--trace2cp-refine-iterations`, the base `trace2cp_vis.jpg` remains the
  initial pass for compatibility; extra pass visualizations use the `itN`
  suffix. Each `itN` pass uses the same drawing structure and public
  `trace2cp_error` reporting semantics as the initial pass.
- Whole-fiber Trace2CP mode writes `trace2cp_fiber_vis.jpg` and
  `trace2cp_fiber_summary.txt`, and `trace2cp_fiber_debug.txt`. Each CP pair
  is loaded, traced, and measured with the same pair-local Trace2CP path as the
  single-pair command. The final
  visualization is composed afterward by mapping each pair-local segment image,
  centerline, CP markers, selected traces, and optimized line into a shared
  arc-length x coordinate system for the selected fiber. The mapping uses each
  pair's local start/target CP image columns and the corresponding global
  start/target CP arc-length columns. Pair-local y orientation is fixed before
  image sampling by shared-CP row-axis alignment, not by guessing after
  composition. The debug file and stdout include per-pair start/target CP strip
  coordinates, strip-space CP deltas, start/target row axes, frame vectors, and
  3D CP deltas projected into the start frame. The image layer uses dense rectangular valid-mask averaging of
  the already sampled segment images; it must not use sparse per-pixel
  splatting that can introduce display holes. Metric errors and traces are
  still computed pair by pair. The JPG uses the same four-row Trace2CP structure as
  single-pair output: full bidirectional traces, partial closest-approach
  traces, fused CP-to-CP line, and optimized CP-to-CP line. Skipped-pair counts
  and reasons are included in the summary. Whole-fiber metric output is the
  average public `trace2cp_error` over all valid CP-pair segments and is
  printed on its own stdout line as `trace2cp_error_mean=<value>`.
- With `--trace2cp-refine-iterations`, whole-fiber mode writes additional
  aggregate iteration images and summaries as `trace2cp_fiber_vis_it1.jpg` and
  `trace2cp_fiber_summary_it1.txt`, then `it2`, etc. The unsuffixed whole-fiber
  outputs remain the initial pass.
- Trace2CP target-column crossing takes precedence over RF-margin rejection for
  the next step in each direction. If a step crosses that direction's target
  x-column and would also enter the RF margin, the trace is considered to have
  reached the target column, an exact interpolated target-column point is
  appended to the trace, and the score is computed at that point. RF-margin
  stop reasons should identify whether the x margin, y margin, or both were
  hit. `max_steps` exhaustion is not a valid scored stop reason for
  target-directed Trace2CP traces and must raise instead.
- Tests use fake/local arrays and monkeypatched readers where possible and must not require network access.
- `docs/code_structure.md` documents the current implemented module structure, data flow, config shape, runner outputs, and local workflow caveats; `planning/specs.md` remains the normative behavior source.
- Future changes that affect public config, data flow, sampling, caching, augmentation, runner outputs, tests, or local workflow must update both the relevant specs and code docs.
# Lasagna inference manager foundations

- The installed `las_manager` CLI owns one backend-neutral configuration,
  catalog, snapshot, run, provenance, and publication model for Fiber 3D and
  Lasagna inference. Command and entity tokens use exact match first, then only
  unambiguous prefixes; ordinals printed by listings are not stable selectors.
- Global configuration lives at
  `${XDG_CONFIG_HOME:-~/.config}/las_manager/config.toml`, with
  `LAS_MANAGER_CONFIG` as an automation override. It contains the catalog URL,
  public bucket, snapshot directory list, cache/output/venv/Atlas directories,
  staging S3 origin, and catalog maximum age. It never contains AWS
  credentials. Relative paths resolve from the config file.
- Global `params` is a string-token array passed to both inference backends.
  Initialized configs default to `--tile-size 512 --border 32 --overlap 96
  --devices all`. Per-run arguments after `--` follow configured tokens and
  override them; explicit singular/plural device selection removes the
  configured mutually exclusive device selector. Existing configs without the
  key receive the default without migration failure.
- The open-data catalog is cached as validated JSON plus a sidecar containing
  its source URL, SHA-256, ETag/Last-Modified, fetch/validation times, and last
  refresh error. `fetch` always revalidates; dependent commands refresh when
  the cache is missing or at least one hour old. Refresh failure may use a
  previously valid cache with a warning; invalid new data must not replace it.
- A volume record preserves the complete catalog identity needed for portable
  provenance and Atlas ingestion: sample/volume IDs and long ID, shape, voxel
  size/format/license, the original volume DataEntry, all OME origins/access
  roots, selected public S3 origin, and catalog hash/fetch metadata. Stable
  selectors are `sample_id/long_id`, globally unique `long_id`, and globally
  unique volume ID.
- Human `volume ls` renders a deterministic, aligned table with one header,
  groups records by sample/scroll, prints the scroll and first volume together,
  and puts branches for additional volumes beneath it in the `SCROLL` column.
  A single-volume scroll has no branch. The redundant `ID`
  column is omitted because the long volume name begins with that ID. Three-D
  shapes retain depth/height/width order and use space-padded widths 6/5/5.
  Its `PREFETCHED` column contains numerically
  sorted local OME groups only when `.zarray` and at least one non-metadata
  chunk exist; advertised or metadata-only groups remain absent. UTF-capable
  output uses `├─`/`└─`, otherwise `|-`/`\-`. Empty results are header-only.
  `volume ls --json` retains the backend-neutral record schema for machines and
  is unaffected by human rendering.
- Snapshot roots may be a run collection, one run, or `snapshots/`. Listing
  deduplicates canonical paths, reads checkpoints with CPU mapping, mmap and
  `weights_only=True`, and caches metadata by path/size/mtime. The stable
  selector is backend/run/checkpoint. Records include hash, step/test metric,
  patch/architecture/output options, precision, task/process/code revision,
  and optional Atlas model identity; absent legacy metadata remains explicit.
- Shell completion is generated by the same command registry and may use only
  valid local catalog/checkpoint caches. It must not refresh the network, open
  an uncached checkpoint, download data, or mutate state.
- `completion install [bash]` atomically installs a canonical loader in the
  standard XDG user Bash-completion directory plus one digest-isolated provider
  per canonical console-script executable. At completion time the loader uses
  the external `las_manager` selected by `PATH`, obtains its canonical provider
  identity through a config-free command, and dispatches only to that provider.
  Providers from multiple venvs coexist; missing venvs are inert. Installation
  support is Bash-only, while `completion bash|zsh` remains available for
  generated setup.
- A final literal `help` before any `--` prints argparse help for the longest
  exact or uniquely abbreviated command prefix. Unrecognized suffixes fall
  back to the understood parent; an unrecognized first token remains an error.
  A `help` token after `--` is forwarded to the inference backend unchanged.
- Bash and Zsh completion delegate full word context to one shared resolver,
  understand unique command abbreviations, and complete valid flags, static
  option values, cached selectors, samples, formats, and locally evidenced OME
  scale indices. Scale discovery reads only local `.zattrs` multiscale dataset
  paths and numeric groups with `.zarray`; it does not invent unknown remote
  levels or perform network access.
- Nullable optional catalog collections are normalized before iteration.
  Specifically, `properties.shape = null` is indexed and displayed as unknown
  rather than preventing other volumes from being listed or completed.
- `volume prefetch <volume> <scale>` calls the existing OME-Zarr downloader for
  exactly that numbered group and stores the OME root at
  `<cache_dir>/volumes/<sample_id>/<long_id>`. It preserves downloader metadata
  and accepts an explicit worker count; no manager-specific transfer path is
  permitted.
- A Lasagna install built from the monorepo includes the sibling Vesuvius
  namespace, Fiber inference packages, their config data, and canonical
  `lasagna.*` modules. The configured venv must run Fiber inference without an
  ambient `PYTHONPATH`; both editable installs and monorepo-built wheels obey
  this contract.
- `inference run <snapshot> <volume> <scale>` atomically reserves a run,
  launches a detached manager-prefixed tmux session, prints its path, and
  returns immediately. By default the tmux runner invokes the shared downloader
  before inference and passes backend `--no-download`; `--no-prefetch` skips the
  manager phase and omits backend `--no-download`, retaining normal on-demand
  crop-aware fetching during the inference lifecycle. On an empty cache the
  manager initializes only local `_download` source metadata, with no remote
  scan or chunk transfer. `--download-workers` applies to both modes; explicit
  backend arguments after `--`, including `--no-download`, retain precedence.
  The serialized, versioned request
  preserves source, destination, group, workers, anonymous access, and remote
  inventory behavior. Additional inference
  argv is preserved without shell interpolation. The positional scale selects
  the input OME group and does not alter the backend's output-scaledown default.
- Every launch has an immutable UUID and atomically reserved output directory
  containing private `metadata.json`, `command.json`, `run.log`, and a portable
  `artifacts/` subtree. The backend-neutral record pins the complete source and
  checkpoint identity and tracks prefetch, inference, staging-upload,
  Atlas-ingest, and Atlas-publication state independently. New directory labels
  use `<sample>-<acquisition>-las-sd<group>-<uuid8>`; this is a concise human
  label, not Atlas canonical identity. Full volume, model, backend, source
  group, command, time, revision, and UUID identity remains structured metadata.
- A manager wrapper owns the `created -> running -> completed|failed|interrupted`
  transition, combined prefetch/inference logging, real child exit status, and
  signal forwarding. Prefetch tracks pending/running/completed/failed/skipped,
  timestamps and error; failure/interruption prevents inference from starting.
  Inference stdout/stderr is teed byte-for-byte to both `run.log` and the tmux
  pane so attaching shows live carriage-return progress without weakening the
  durable log or exit-code contract.
  Stale active records are reconciled against both tmux and PID/start-time
  identity without deleting artifacts. `inference ls` is the durable view;
  `run ls` contains only live tmux sessions.
- `tmux attach` attaches normally outside tmux. Inside tmux it links the run
  window immediately after the current window and selects it, never nesting a
  client or renaming the source window. Creation atomically captures tmux's
  stable `window_id`, tags the window with the immutable run UUID, and validates
  both before live/attach decisions. Numeric indices are never durable
  identity. Window names use `inf-<sample>-<uuid4>` and are only short display
  labels. A surviving orphan inference process is durable-running but not
  attachable when its wrapper window is gone.
- Subsequent ordered phases add direct portable provenance,
  shared Zstd output, Atlas Lasagna-bundle publication, and the Lasagna backend without
  introducing a second manager workflow or discarding the identities above.
- Current Fiber checkpoints are self-configuring for inference: their embedded
  config is authoritative and the CLI positional config is optional. A
  positional config remains supported only as the explicit fallback for a
  legacy checkpoint without embedded config; conflicting explicit config does
  not override a current checkpoint.
- Fiber inference directly and atomically maintains bundle-relative
  `inference.json` with schema version, status, source/catalog identity,
  requested and observed input scale, effective output scale/levels/crop,
  numerical settings, checkpoint/config identity, repository/runtime identity,
  and a bounded structural artifact inventory. It records failed/interrupted
  status after artifact initialization and never recursively inventories Zarr
  chunks.
- The shared Fiber/Lasagna inference metadata writer records the checked-out
  Villa revision as `inference.code_commit`; this applies to direct and managed
  invocation. Repository dirtiness remains explicit. Packaged deployments may
  supply the build commit when no Git checkout is present; otherwise the value
  is null. Git provenance is not copied into Atlas model metadata.
- The manager passes portable catalog/model/run context through an explicit
  private context file. Fiber inference, not the manager, authors inference
  facts. Private absolute paths, command, hostname/user, logs, and tmux identity
  remain outside `artifacts/`.
- Lasagna manifests preserve a relative `provenance` reference and unknown
  forward-compatible top-level fields across load/save.
- Newly created Fiber and Lasagna inference OME-Zarr arrays use the shared
  exact Zarr-v2 compressor `{id: blosc, cname: zstd, clevel: 3, shuffle: 1,
  blocksize: 0}` at every generated level. Both direct CLIs expose the same
  `--ome-compressor` compatibility override. Resume never rewrites an existing
  array's codec; requested/actual mismatches are reported, while provenance
  inventories the codec actually persisted per level.

## Atlas staging and ingestion

- `las_manager open-data validate/upload` accepts completed portable Fiber and
  Lasagna bundles through one backend-neutral implementation. Both require a
  CC BY-NC 4.0 source license. Model identity is resolved automatically from a
  carried ID or a freshly rehashed checkpoint in configured snapshot roots;
  there is no normal upload-time model-ID argument.
- Staging uses the immutable run UUID and a fixed local file inventory.
  `_INCOMPLETE` is the manager-side transaction guard: it is written before
  rclone starts and removed only after rclone succeeds; failed staging never
  invokes Atlas ingestion. Retry invokes rclone again and relies on its
  configured comparison behavior. Completed run UUID/bundle contents are
  immutable because `rclone copy --size-only` neither detects changed same-size
  objects nor deletes stale destination objects. No manager upload manifest or
  per-Zarr-chunk transaction hash is stored remotely.
- Atlas maps both Fiber and Lasagna output to the existing `lasagna` copy-first
  artifact. Canonical identity is volume, canonical model ID, and source input
  level. The data entry contains only its private origin and existing
  `model_id`/`level` parameters; portable provenance remains in `inference.json`
  and is not copied into Atlas `creation_info`. The origin path is relative to
  its access-root URL and Atlas joins them to resolve the full data-sync source.
- Missing models are registered automatically using an Atlas UTC datetime ID.
  Fiber uses the minimal Lasagna model record with architecture `fiber3d/unet`,
  task `lasagna`, process `model_training`, snapshot-root-relative checkpoint
  path, and snapshot SHA-256. Data entries keep numeric model references.
  Byte-identical checkpoint aliases are normalized; incompatible hashes,
  metadata, tasks, or canonical-ID collisions are rejected before staging.
- Public publication remains an operator-controlled Atlas data-sync action.

## Lasagna manager backend

- Snapshot discovery classifies current Fiber checkpoints by their embedded
  Fiber model config and Lasagna checkpoints by their Lasagna architecture
  metadata/state-dict wrapper. It never infers a backend from a filename.
  Selectors are namespaced as `fiber3d/run/checkpoint` and
  `lasagna/run/checkpoint`; `--backend` is needed only for an ambiguous legacy
  shorthand.
- Lasagna launches use `preprocess_cos_omezarr predict3d` but otherwise share
  the manager's volume prefetch, immutable run record, tmux runner, completion,
  lifecycle, portable artifact bundle, staging, and Atlas ingestion paths.
- Direct and managed Lasagna inference write the shared `inference.json`
  envelope with `artifact_kind = "lasagna"`. Its product metadata preserves
  the `.lasagna.json` source-to-base mapping, gradient encoding scale/factor,
  crops, groups, channels, Zarr paths, and output scaledowns. The manifest
  points to `inference.json` by a bundle-relative path.

# Process-parallel 3D flush

- Fiber and Lasagna inference use the same shared rolling-mmap flush engine.
- Positive `flush_workers` use persistent spawn processes with bounded
  descriptor-only queues. Workers read frozen mmap regions directly and own
  distinct scale/chunk writes; ndarray payloads never cross IPC.
- Exactly one immutable combined flush batch may overlap the following
  inference band. Zero-task and fully acknowledged batches finalize immediately
  through the same release/accounting routine as synchronous flush. Ring reuse
  and `final_z` advancement require successful completion of the entire batch;
  failed batches never clear or release frozen generations.
- Each worker limits native CPU libraries and its Zarr executor to one thread
  and allocates at most one chunk's denominator/raw/finalized arrays. This
  prevents the process pool from multiplying into CPU-count-sized nested teams.
- `flush_workers=0` is the synchronous baseline and uses immediate ring release;
  both inference CLIs default to the available CPU count capped at 64 process
  workers.

# Fiberlet anchor extraction

- The initial C++ fiberlet stage consumes exactly the canonical 3D `uint8`
  `presence/nx/ny` triplet from a Fiber Lasagna manifest. Each prediction-grid
  voxel supplies one presence and one compact unoriented 3D direction. Extra
  groups are not alternative observations. The channels must share shape and
  effective spacing but may use different chunk layouts.
- The manifest must contain an explicit positive numeric `source_to_base`.
  Anchor extraction uses the existing local/remote Lasagna opener, compact-axis
  decoder, persistent remote cache, and decoded chunk cache. It must not create
  a reference direction merely to use the trace sampling API.
- Stored prediction indices are voxel centres. Cubic cells of integer side 2
  through 8 are anchored at prediction-grid origin zero and own half-open,
  globally fixed ZYX index ranges. CLI spatial coordinates are always in base
  voxels. The half-open `--crop` base box maps its two boundaries to point
  indices with scale-aware `ceil(base/prediction_to_base_scale)`, then selects
  intersecting complete global cells without recentering or truncating them.
  A non-empty box containing no prediction sample and any selection outside the
  global prediction volume are errors; there is no implicit clipping.
- Every non-empty cell attempts two independent, unoriented, potentially
  non-orthogonal direction components. Exclusive assignment maximizes squared
  alignment, and each fixed-assignment update is the principal eigenvector of
  `sum g_i p_i d_i d_i^T` for that component. Deterministic multistart
  assignment/PCA supplies only initial directions. A single covariance's first
  two orthogonal eigenvectors are not a valid replacement. Initial component
  positions are the centre of the clipped owned-cell voxel range.
- Direction and a provisional position are then refined from a bounded halo. For
  anchor `(p_k,u_k)`, the transverse Gaussian is recentered at `p_k`, uses the
  distance to the line through `p_k` along `u_k`, and is truncated at three
  sigma. Evidence is also restricted to a symmetric axial slab about the plane
  through the fixed cell pivot normal to `u_k`. Invalid axes and presence below
  the inclusive floor contribute no numerator. Direction-mode assignment uses
  `g_ik p_i abs(d_i dot u_k)^2`; stable component index breaks exact ties and
  zero-evidence sites remain unassigned.
- Within each assigned component, projective residual
  `1-abs(d_i dot u_k)^2` is summarized with a deterministic 256-bin weighted
  median/MAD histogram using `g_ik p_i` mass. The cutoff is the larger of
  median plus the configured MAD multiplier and the configured angular-noise
  floor expressed as `sin(angle)^2`. It may trim no more than the configured
  evidence-mass fraction, at most 0.20; coherent data retains all observations
  and complete cutoff bins are retained. The defaults are 0.20 maximum trim,
  multiplier 3, and 5 degrees.
- Each component direction update is installed directly as the principal axis
  of retained `g_ik p_i d_i d_i^T`. There is no angular interpolation or
  angular line search. A retained tensor without the existing unique principal
  axis removes only that component while preserving stable diagnostic ancestry.
  Supported close components are not merged before refinement.
- Production compact cutoff materialization also records retained logical
  indices per component in the same order. The immediately following centroid
  traverses those sparse indices rather than rescanning the complete support;
  expanded/public fitting retains the full defensive scan. Worst-case index
  storage is included in worker memory admission.
- Production compact observations cache the complete configured direction and
  presence eligibility predicate once per unique sampled voxel. All compact
  fitting consumers reuse that bit; arbitrary expanded/public observations
  continue to validate direction, presence, and validity on each use. The bit
  occupies existing trailing record padding and does not enlarge the compact
  observation.
- Its position target is the retained assigned
  `g_ik p_i (d_i dot u_k)^2` centroid projected onto the plane through the cell
  pivot normal to the updated direction. Deterministic position-only
  backtracking evaluates `1, 1/2, ...` through the first
  displacement at or below the peak-grid step and accepts the first strict
  objective improvement. Direction remains installed when position does not
  improve. Transverse displacement is clamped to the local window and
  prediction grid.
- The spatial objective and final aligned support use retained evidence in the
  numerator and `sum g_ik` over every sampled lattice site for each active
  component in the denominator. The denominator is independent of site
  presence, direction, assignment, and trim state, so rejected positive sites
  cannot create normalization holes that attract or repel the fit. Coherence
  divides by retained assigned `sum g_ik p_i`. Empty, degenerate, and
  below-threshold components are discarded independently.
- Fixed-direction spatial objectives are implemented in a separate private
  objective module. Production indexed compact observations remain float32
  through Gaussian, alignment, numerator, denominator, and final ratio
  arithmetic; their directions are already normalized by compact-observation
  construction. Logical assignment/membership indices address the per-cell
  index stream, while observations address the indexed tile storage. The
  direct public fitter uses the same float32 observation, component, and
  accumulator representation. Persistent component state, acceptance
  tolerances, final aligned-support evaluation, diagnostics, and serialized
  numeric values are float32; direct and indexed fitting differ only in storage
  and indexing.
- Robust assignment, trimming, direct direction update, and position update run
  for at most `maximum_iterations`, default 1. This is a bounded alternating
  refinement budget, not convergence to exact equality of hard assignment or
  histogram membership sets. An earlier exit is allowed when projective
  direction change is within tolerance and accepted position movement is no
  larger than the peak-grid step. The later peak stage owns finer positioning.
- After direction refinement, final position is selected independently for each
  surviving direction by deterministic steepest ascent on a narrower
  direction-conditioned response in the final normal plane. The response is
  `sum(G_peak p_i abs(d_i dot u_k)^2) / sum(G_peak)`, where `G_peak` is an
  anisotropic Gaussian about the candidate line. Its default transverse sigma
  is `1.5` prediction voxels and its default along-direction sigma is
  `1.5 * cell_size` (`6` prediction voxels for default cells). Consequently a
  straight fiber one cell from the pivot retains about `0.80` axial weight and
  the ends of a centered three-cell span retain about `0.61`.
- When `peak_gradient_weight > 0`, every outer cell job samples one additional
  lattice voxel around that peak support and computes presence-only gradients
  with separable 3D Sobel correlation. For example,
  `grad_x=(P[x+1]-P[x-1])/2` after unit-sum `[1,2,1]/4` smoothing on Y and Z.
  All 27 stencil presences must be finite and decoded; compact-direction
  validity and the presence floor do not affect gradient validity. Global-edge
  or otherwise incomplete stencils provide no gradient vote. Weight zero skips
  both gradient construction and the extra halo exactly.
- For candidate `a` and observation `x`, the presence gradient and `a-x` are
  projected into the fitted anchor's normal plane. Zero projected vectors are
  skipped. Their cosine is positive for an inward-pointing gradient and
  negative for outward. Inward/outward vote mass is the anisotropic Gaussian
  times squared fiber-direction alignment, transverse gradient magnitude,
  `peak_sigma`, and squared radial cosine. Thus it is dimensionless in
  prediction-grid units and weakly radial gradients contribute less.
- The signed term is `(inward-outward)/(inward+outward+epsilon)`. It is gated by
  valid-gradient Gaussian/alignment coverage and by
  `R/(R+peak_gradient_reliability_scale)`, where `R` is normalized radial vote
  magnitude. The default reliability scale is `0.05`; flat, tangential,
  incomplete, or otherwise unsupported gradient neighborhoods fall back to the
  presence response. The gated signed value is added with default weight
  `1.0`. `--gradient-weight` changes or disables it.
- Peak support is the intersection of independent three-sigma transverse and
  axial bounds. Positive peak signal is restricted to the final retained
  assignment for that component; trimmed or competing observations cannot
  re-enter. Its denominator includes every sampled in-volume site inside both
  bounds independent of presence, direction, assignment, or trim state. The
  broad direction fit
  continues to use its separate fixed axial half-width.
- After the exact transverse and axial support cutoffs, peak search evaluates
  Gaussian weights with the standard exponential, matching robust-direction
  and final-support calculations.
- The peak grid is anchored at the fixed cell pivot and uses canonical
  transverse basis construction, configured prediction-voxel step, a circular
  local window, and continuous voxel-Voronoi ownership
  `[cell_begin-0.5,cell_end-0.5)` clipped to `[0,shape-1]` at global edges.
  Search starts at the grid node nearest the provisional position and repeatedly
  chooses the highest strictly improving 8-neighbor, with canonical integer-grid
  tie breaking. It stops on a plateau and cannot cross a response valley to a
  stronger distant fiber. The grid radius is limited to 128 steps so malformed
  parameters cannot create unbounded enumeration.
- Production subpixel placement uses `separable_1d`: each normal-plane
  coordinate independently fits the center and its two axial neighbors.
  Missing neighbors, non-finite curvature, or non-negative curvature leave only
  that coordinate at zero. Each finite stationary offset is clamped to
  `[-0.5,0.5]`; the combined point passes one owner/window and directly
  evaluated non-decreasing-response guard or falls back wholly to discrete.
  Broad-kernel support and coherence are reevaluated at this production point
  for ordinary filtering and NMS ranking. A cell emits zero, one, or two
  anchors.
- For matched benchmarking, when the complete feasible 3x3 neighborhood exists,
  all nine response samples
  are fit by deterministic least squares to
  `a + bx + cy + dx^2 + exy + fy^2` in grid-step coordinates. The stationary
  point is usable only when all samples and coefficients are finite, the
  symmetric Hessian's largest eigenvalue is strictly below
  `-1e-12 * max(1,max(abs(samples)))`, its determinant exceeds the square of
  that tolerance, and both offsets lie in the closed interval `[-0.5,0.5]`.
  An exterior point is rejected rather than clamped. The fitted 3D point is
  retained only when it remains in the owner/window domain and its directly
  evaluated normalized response does not decrease within the existing
  scale-aware response tolerance. Missing neighbors, unsuitable curvature, or
  either final guard leave the joint comparison equal to the discrete maximum.
- Refined-only benchmark extraction retains the converged grid maximum as
  transient diagnostic provenance paired with two independently accepted
  normal-plane candidates. `separable_1d` reproduces the previous estimator:
  each coordinate uses its center and two axial neighbors; missing neighbors,
  non-finite curvature, or non-negative curvature leave that coordinate at
  zero; finite stationary offsets are clamped independently to `[-0.5,0.5]`;
  the combined point passes one owner/window and real-response guard or falls
  back wholly to discrete. `joint_2d` uses the complete quadratic rule above
  and is benchmark-only.
  All three stored transient positions are final estimator outputs, so a
  rejected estimator equals the discrete point exactly. These values are not
  serialized in either final or any diagnostic-stage version-1 anchor artifact.
  Both estimators optimize two normal-plane coordinates from a response that
  integrates 3D volumetric evidence; neither permits axial motion.
- There is no pre-refinement same-direction merge. Supported close components
  remain independently fitted; the ordinary cross-cell NMS below handles true
  duplicates. Merge fields remain readable and serializable for compatibility
  but are inert for newly fitted cells. Robust anchor artifacts use schema
  version 2; the loader accepts strict legacy version 1 artifacts with their
  original parameter set.
- After all local fits, deterministic local-maximum NMS removes cross-cell
  duplicates with compatible unoriented directions. Two anchors interact only
  when they come from different cells and
  within the configured angular, transverse, and longitudinal limits measured
  around their sign-aligned average axis. Transverse and longitudinal defaults
  are independent of the refinement window and cell size: 2 and 1 stored
  prediction voxels respectively. Ranking is aligned support, coherence, then
  cell/component identity. Every decision compares against the original
  candidate set, not only surviving candidates. Crossing directions and
  sequential anchors along a fiber therefore remain distinct. Cropped runs fit
  the exact external pivot cells that can directly compete with a selected
  anchor, conservatively including the full refinement displacement of both
  anchors before the NMS reach, but serialize and count only selected cells.
- Reductions, seeds, assignment ties, convergence, component ordering, and
  serialization sign are deterministic. Parallel work cannot change a
  within-cell reduction. After serial cell selection, extraction forms the
  selected cells plus every conservative direct-NMS-context cell, partitions
  them into deterministic six-cell-per-axis spatial tiles, and samples one
  dense ZYX box per tile with the complete fitting/peak/gradient halo. Cell
  fits traverse indexed tile storage in the same canonical ZYX order as an
  independent halo. Each physical tile voxel has one compact float32 position,
  pre-normalized direction, presence, and gradient record. A cell owns only
  32-bit tile indices and cell-local gradient-validity bytes, avoiding expanded
  observation copies across overlapping support regions. The public expanded
  observation API and production indexed path share one templated fitter. Tile
  halos may overlap, but the decoded chunk cache avoids repeated source-chunk
  reads. Complete cells with a full volume halo expand one immutable ordered
  `(z, y, xBegin, xEnd)` owned-or-radius support stencil through their tile's
  actual strides. This preserves canonical ZYX order and logical observation/
  gradient populations. Crop and tile boundaries do not affect eligibility;
  partial cells or cells clipped by a volume boundary retain the scalar sample-
  cube construction. When gradients are enabled, the extra halo voxel makes
  every retained stencil site gradient-eligible, while sampled tile-gradient
  validity remains authoritative. Deterministic tile splitting and the worker
  count keep coordinate vectors, decoded samples, and cell scratch under the
  aggregate sample-memory budget; lower-level sampling uses one thread. Tiles
  are paired for overlap reuse only when the staged pair fits that budget. Once
  a tile's sampling, gradient, and compact-observation construction finish, its
  cells enter one cooperative worker queue. Any extraction worker may fit a
  ready cell, but the tile owner retains the immutable observations and waits
  for every dependent cell before releasing them or advancing overlap reuse.
  Sampling groups therefore remain the deterministic memory-ownership unit
  while cell fitting is work-balanced independently.
  Results and worker failures are stored by canonical cell index, retain
  predicates and diagnostic aggregation run serially, and progress callbacks
  are serialized. The lowest-index cell failure is reported after all workers
  join.
  Initial seed fitting uses a separate exact owned-cell view over the same
  dense tile. Its constant-time layout validation requires monotonic cell
  bounds, containment by the tile sample box, matching tile observation
  cardinality, and matching clipped owned cardinality. It then visits dense
  rows in canonical Z/Y/X order without an owned-index allocation. Robust
  refinement continues to use the larger support-index range. The public
  expanded-observation API retains stable input-order filtering and its
  historical count-only owned-coverage check; it does not sort, deduplicate,
  or require lattice coordinates.
  Production compact observations also keep robust-proposal positions,
  directions, component state, Gaussian/alignment values, masses, residual
  histograms, and retained direction tensors in float32 throughout the repeated
  per-observation loop. Fixed residual histograms, retained tensors, robust
  cutoff selection, the shared float principal-axis solver, persistent
  component state, diagnostics, and serialized output remain float32. The
  public direct-observation fitter uses this same precision path.
  Machine output and OBJ store spatial positions only in base-volume XYZ
  coordinates. Prediction shape/scale, cell indices, cell size, direction
  falloff, transverse/axial peak sigmas, peak grid step, cutoff, local window,
  broad axial slab, convergence, and independent NMS limits remain explicit lattice
  metadata and parameters. The orientation-independent sampling halo covers the
  larger of the broad direction kernel and anisotropic peak kernel, including
  the local-window displacement. Runtime timing and worker count are not
  scientific output, so artifacts remain byte-identical across worker counts.
- The anchor command itself does not connect anchors. Its strict version-1
  JSON is the authoritative input to the separate integer-DP path stage. It
  requires the refinement/NMS and merge parameters, aggregate diagnostics,
  per-anchor refinement fields, and auditable per-cell merge evaluations where
  applicable. Positions are validated against the prediction grid, continuous
  owner cell, rotating pivot plane, and local window. Older experimental
  artifacts must be
  regenerated. `anchors.obj` contains all retained anchors, while
  `anchors_0.obj` and `anchors_1.obj` contain deterministic post-sort
  per-cell component slots. These slots are not global H/V classes.
  Strict final and diagnostic-stage parameters include
  `peak_gradient_weight`, `peak_gradient_reliability_scale`, and
  `nms_transverse_radius_prediction_voxels`; missing fields are rejected
  without repair.

# Refined-anchor localization benchmark

- `vc_fiberlets anchor-benchmark <fiber-manifest> <fiber.json>` measures the
  geometric anchor positions immediately after direction/position refinement.
  It reads the strict fiber's dense `line_points` in base-volume XYZ
  coordinates and never substitutes control points or later retained anchors.
- Benchmark cells are the canonical prediction-grid anchor cells whose closed
  continuous ownership boxes intersect an exact reference-polyline segment.
  A segment on a shared cell face therefore selects both cells. Selection is
  unique, sorted, grid-clipped, and uses radius zero; no sampled-vertex
  approximation is permitted.
- Extraction stops after the `refined` diagnostic stage. Support rejection,
  retain predicates, external NMS context, NMS ranking, and final artifact
  population cannot affect the benchmark. Refined prediction-coordinate
  positions are converted to base coordinates before exact point-to-polyline-
  segment distances are computed. The command reports `discrete`,
  `separable_1d`, and `joint_2d` from the stored matched positions. All three
  reports use the exact same geometric refined records and therefore have equal
  populations.
- The fixed inclusive thresholds are 4 and 8 base voxels. For each threshold,
  the command reports anchor hits over all geometric refined anchors and cell
  hits over all reference cells. A cell hits when at least one of its refined
  anchors is within threshold; a cell with no refined anchor is a miss.
- Distance output contains count, minimum, mean, median, p95, and maximum in
  base voxels. Quantiles use linear interpolation at zero-based rank
  `q*(n-1)`. Empty anchor populations report `n/a` rates/statistics rather than
  changing denominators, and an empty reference-cell selection is an error.
  Population, distance, and threshold output records carry
  `stage=discrete|separable_1d|joint_2d`; extraction timing is shared. No
  artifacts are written.

# Curved-domain DP fiberlet paths

- `vc_fiberlets paths` consumes a strict `vc_fiberlet_anchors` version-1 JSON,
  its matching canonical fiber manifest, and a separate regular Lasagna normal
  manifest. Format/version, retained component structure, finite unit axes,
  cell ownership/order, grid shape, prediction-to-base scale, and materialized
  fiber-manifest content hash must match. Malformed artifacts are rejected
  without repair. The stored source locator is informational, so identical
  manifest content may be relocated. Fiber prediction `nx/ny` must never be
  used as Lasagna surface normals.
- Candidate target cells lie in the deterministic filled integer neighborhood
  `0 < norm(offset) < radius+margin`, initially radius four and margin 0.5.
  Offsets are symmetric and lexicographically ordered. All shorter distances
  through the outer half-open bound are therefore searched, including
  non-integer radial shells. All retained component pairs are considered once
  under canonical `source_id < target_id` ordering. Both unoriented endpoint
  axes must align to the chord within the configured bound, initially 45
  degrees. This stage does not impose degree, mutual-best, or overlap limits.
- Candidate generation completes before scoring. A fixed worker pool prepares
  each searchable candidate exactly once into its canonical slot: one Hermite
  domain, one corridor enumeration, checked packed local keys, and mapped
  positions. DP reuses this representation and never reconstructs the domain,
  corridor, or local nodes. Local corridor admission uses float32 continuous
  point-to-segment distance against only the two centerline segments incident
  to the node's curved-domain layer. Points strictly inside the layer center's
  transverse-radius circle are admitted directly. This is a layer-local tube,
  not a union over distant candidate segments, so separate bends cannot admit
  shortcut nodes into one another's layers.
- Preparation accumulates positive-weight native corners in worker-local sets.
  The authoritative interpolation-cell decomposition is shared with weighted
  interpolation. When all eight positive-weight corners occupy one `16^3`
  bitmap page, preparation resolves that page once and sets the eight local
  bits directly; lower-cardinality and page-crossing cells use the general
  per-corner insertion path. Sorted worker vectors are reduced by deterministic
  pairwise unique merges to
  one complete stored-ZYX ordered global union. Its exact contents and size are
  invariant under worker count and `--batch`, including shared corners and
  selection-boundary corners.
- `--batch`, initially 65536, limits consecutive global unique coordinates per
  sampler call. All prediction ranges complete before any normal range. After
  all reads, prepared scoring voxels and their immutable paged index are
  retained through parallel DP. Exact endpoints are interpolated eagerly;
  interior-node corners and weights are re-derived from canonical stored
  positions only on a candidate-local lazy-cache miss. Prediction
  and normal samplers each receive every global coordinate exactly once; only
  their call count changes with `--batch`.
- Every parallel stage writes canonical slots. Workers continue after
  independent candidate errors and the lowest candidate-index error is rethrown
  after the stage. A deterministic serial pass derives aggregate diagnostics.
  Batch size, worker count, sampled-corner counts, and phase timings are absent
  from artifacts; changing them preserves path and graph artifact bytes.
  Benchmark timing separates candidate generation, parallel preparation,
  corner merge, prediction reads, normal reads, score materialization, and DP.
  Every phase reports wall time, process CPU seconds, and effective cores
  (`CPU/wall`), plus unique requests, calls, and checked owned-payload estimates.
- The optional core progress callback emits phase-labelled initial, periodic,
  and terminal states under a serialization mutex. `vc_fiberlets` writes
  completed/total, percentage, elapsed time, rate, and ETA to stderr. Empty
  stages report `0/0`. Progress is operational and cannot affect artifacts.
- Every candidate has a deterministic curved local coordinate domain. A cubic
  Hermite centerline uses exact anchor positions, chord-oriented fitted axes,
  and endpoint-distance derivative magnitudes. It is tabulated with
  `max(64,ceil(16*endpoint_distance))` segments and deterministically inverted
  by linear interpolation of cumulative arclength. Planes are placed every 2
  prediction voxels of centerline arclength with the exact target inserted once.
  A nonfinite or degenerate curve/tangent/frame rejects the candidate.
- Each plane normal is the normalized Hermite derivative. The initial
  transverse axis is the projection of the least-aligned canonical world axis;
  deterministic ties use axis order. Later frames use minimal-rotation parallel
  transport followed by re-orthogonalization. A local key `(layer,u,v)` maps to
  `center + 0.5*u*U + 0.5*v*V` in prediction XYZ. Finite transverse bounds are
  derived from corridor radius, then mapped points are filtered by inclusive
  volume bounds, corridor membership, and any replay tube predicate. An
  interior mapped point is narrowed once to three `float32` coordinates before
  any of those admission tests; that stored position is authoritative for
  sampling, DP geometry, and output. Exact endpoint anchors use the same
  float32 representation as interior nodes.
  `(layer,u,v)` is checked row-major packed into one `uint32_t` using the
  per-candidate transverse width; non-finite or overflowing lattices fail.
- The exact start `(s=0,u=v=0)` and target `(s=L,u=v=0)` are source and sink.
  Interior edges advance exactly one layer and change each transverse index by
  at most one. Their actual mapped XYZ vectors and Euclidean lengths drive all
  feasibility and scoring. The final interval may be shorter than 2 prediction
  voxels; exact multiples and curves shorter than 2 do not create duplicate
  planes. States 0 through 8 encode the incoming `(du,dv)` transition; state 9
  is source-only. The predecessor node is derived from the checked packed key
  as `(layer-1,u-du,v-dv)`, while reconstruction stores only the predecessor's
  state index. Cumulative costs are float32 and exist only for the current and
  next interior layers; one predecessor-state byte per global node/state
  remains until reconstruction. The exact source and sink stay outside this
  rolling interior representation. Strict cost ties retain the first canonical
  predecessor.
- Arbitrary-position presence is the trilinear weighted sum of all positive-weight
  native corners. Unoriented prediction axes are normalized and combined as
  `T=sum(w*d*d^T)` without presence weighting; the deterministic shared
  symmetric eigensolver supplies the unique principal axis. Every positive-
  weight corner must have finite valid prediction data and the tensor must have
  a unique principal eigenvalue. Antipodal axes therefore interpolate without
  cancellation, while an equal orthogonal mixture is invalid. Normals use the
  same sign-invariant interpolation, but an invalid corner or ambiguous tensor
  produces an invalid normal and retains isotropic curvature fallback.
  Materialized interior prediction and normal axes are re-encoded in the
  Lasagna +Z compact `nx/ny` byte convention; presence is a clamped rounded
  byte and validity uses independent bits. This intentional second
  quantization is part of the experimental DP objective. The persistent
  16-byte geometry node stores only key and float position; compact scoring is
  converted immediately into the candidate-local prepared cache and no normal
  reason or interpolation stencil is retained. The
  global sparse union includes all required native corners even when a corner
  lies outside the floating-node tube predicate. Evaluated nodes retain integer
  candidate and local keys rather than floating-point hash identity. Each
  candidate owns a direct node-to-cache map and append-only prepared scoring
  cache. First access interpolates, compact-quantizes, decodes, and normalizes
  that node; later accesses reuse its cache index. Mutable page lookup state is
  candidate-local, while the underlying prepared scoring voxels and page index
  are shared read-only. Both local structures are released after that candidate
  and are not added to persistent prepared geometry.
- Every nonzero DP edge must have an unoriented angle strictly below 25 degrees
  to the interpolated dense fiber-prediction axis. Equivalently, for normalized step
  `d` and fiber axis `f`, `abs(d dot f) > cos(25 degrees)` without a boundary
  epsilon. Interior moves use the interpolated prediction at their mapped
  destination. Exact source/sink transitions are exempt from this sampled gate
  and remain constrained by the fitted endpoint-axis rule. A missing, invalid,
  nonfinite, or degenerate required interior prediction rejects that edge before
  scoring. The independent Lasagna surface normal does not participate in this
  hard gate. A reached node resolves at most nine outgoing neighbor lookups,
  mapped directions/lengths, and hard-gate results once and reuses those edge
  descriptors across every reached incoming state. Unreached nodes do not
  pre-generate edges.
- Valid-data scoring shares the regular native tracer's exact ordered float
  multiplicative alignment loss. For incoming step `a`, outgoing step `b`,
  current prediction `c` sign-aligned to `a`, candidate prediction `d` sign-
  aligned to `b`, and candidate presence `p`, `score` is `clamp01(p)` times the
  six positive-clamped dots `a.b`, `a.c`, `a.d`, `c.b`, `c.d`, and `b.d` in
  that order. Alignment cost is `(1-score)*actual_edge_length`; edge-length weighting
  keeps differently directed local transitions comparable. The former lattice
  direction floor and independent presence/direction weights do not exist.
  Fiberlet DP calls the shared prepared-input scoring path with cached unit
  vectors; the regular validating scoring API normalizes and delegates to the
  same arithmetic. Interior DP may evaluate the four pair-dependent alignment
  dots for its valid outgoing slots as one compact structure-of-arrays batch.
  Valid lanes remain in ascending transition-slot order. Smoothness, complete
  cost assembly, accumulated-state addition, strict-less relaxation decisions,
  and backpointer writes remain scalar in the existing order.
- The source transition uses the fitted start axis as both `a` and `c`, its
  actual mapped direction as `b`, and the dense destination prediction as `d`.
  The sink transition uses the current dense prediction and incoming step, its
  actual mapped direction as `b`, and the fitted target axis at presence one as
  `d`. The first and last transition directions must satisfy the configured
  endpoint-axis bound. Interior destinations alone use the strict sampled-
  prediction gate.
- Invalid fiber samples cannot be destinations of nonzero DP edges under the
  hard sampled-direction gate. The pre-existing invalid-prediction cost remains
  serialized as part of the experimental objective configuration, but it does
  not turn an invalid destination into a traversable bridge.
- Direct local curvature shares the native 3D tracer's exact float
  normal-aware equations. With a valid normal it emits tangent-plane and
  normal-tilt costs; otherwise it emits only the isotropic fallback. Fiberlet
  defaults are isotropic/normal/tangent weights `2/0.1/10` and a zero-degree
  free angle, matching greedy tracing. The per-turn value is divided by
  `max(1,(previous_edge_length+candidate_edge_length)/2)`. Cumulative history
  smoothness is not part of this DP.
- `fiberlets.json` stores explicit source paths/hashes, base-volume coordinates,
  parameters, diagnostics, every considered endpoint pair and reason, component
  cost totals, and successful base-coordinate paths. Every successful scored
  path also stores its base-voxel arc length, total loss divided by prediction-
  voxel arc length, and report-relative visual quality.
  Smoothness terms remain included in the density even though their existing
  per-turn integration differs from the edge-integrated data terms. These are
  reporting values only and cannot change path selection or acceptance.
  Parameters include nominal longitudinal and transverse spacing in prediction
  voxels and their derived base-voxel values. Successful candidates retain the
  exact interpolated endpoint prediction and normal samples needed for graph
  joins. All path positions remain base XYZ.
- Relative visual quality is computed over exactly the successful scored paths
  in canonical report order as inverse min-max normalized loss density: the
  lowest density is one and the highest is zero. Equal densities all map to one;
  an empty population has null bounds. This is an artifact-relative display
  value, not absolute confidence. Display color is napari runtime state.
- `fiberlets.obj` contains only successful base-coordinate line groups and
  report comments declare the density population, unit, bounds, and formula;
  each group records total loss, loss density, and relative quality. Path MTL,
  materials, and serialized RGB do not exist, and publication removes a stale
  `fiberlets.mtl`. JSON and OBJ use individual atomic replacements; the pair is
  not transactionally atomic. Timing and worker count are not serialized, so
  repeated identical runs are byte-identical.
- A candidate has independent searched, score-valid, and accepted state. A
  score becomes valid only when DP finds a sink path with a finite total;
  feasibility failures never serialize a zero placeholder cost. `paths
  --stats` reports retained anchors, candidate/search/unscored/accepted counts,
  min/mean/max integrated total loss for all scored paths and the accepted
  subset, and min/mean/max accepted loss density, using `n/a` for an empty
  population. Until quality filtering exists, every scored path is accepted
  and the two total-loss ranges match.
- OBJ output uses one group per accepted fiberlet and one explicit two-index
  line element per adjacent path edge. Napari is the supported path viewer. It
  strictly validates OBJ report/group metrics, rejects obsolete material
  records, crop-filters geometry and properties together, and exposes total
  loss, density, and relative quality as Shapes features. Quality is mapped
  over fixed `[0,1]`; a dock selector defaults to a custom red-yellow-green map
  and offers napari's available colormaps in deterministic order. Selection is
  display-only. Anchor OBJ parsing remains unchanged.
- By default, `paths` writes separate `fiber_presence_{xy,xz,yz}` OBJ/MTL/PNG
  bundles. Each OBJ is one base-coordinate textured quad on the lower central
  prediction voxel of the whole-cell-expanded selected region and can be loaded
  independently. Quad bounds extend half a stored prediction voxel beyond the
  first and last sample centers on each varying axis. Four UVs map the
  minimum/minimum corner to `(0,1)`, maximum/minimum to `(1,1)`,
  maximum/maximum to `(1,0)`, and minimum/maximum to `(0,0)`.
- Each grayscale PNG samples the canonical stored presence channel directly,
  one pixel per prediction voxel, with `round(255*clamp(p,0,1))`; missing
  presence is black and direction validity is irrelevant. Total texture pixels
  are overflow-checked and capped at one million. Per-file replacement is
  atomic and each OBJ is published after its PNG and MTL, but the nine-file
  collection is not transactionally atomic. `--no-slices` skips sampling and
  removes all bundles without changing path or JSON results.
- `fiberlet_graph.json` is generated from every successful scored fiberlet.
  Anchor identity, base-coordinate position, exact interpolated dense
  prediction, and exact interpolated Lasagna normal define a node; repeated
  candidate references must agree. Each canonical
  fiberlet is one edge with deterministic array-index ID, exact dense polyline,
  candidate ID, additive loss, and prediction-grid length, and supplies two
  directed arc IDs: `2*edge` forward and `2*edge+1` exact reverse. Endpoint
  tangents use the first/last distinct dense point in traversal direction.
  At a shared anchor an incoming arc may transition to another outgoing arc
  only when the angle between its tangent pointing into the anchor and the
  outgoing tangent pointing away is strictly below 45 degrees and the anchor
  prediction is valid. Every transition stores the shared local metric's
  alignment/isotropic/tangent/normal components. It uses the same anchor sample
  as current and candidate prediction, outgoing segment length for alignment,
  and `max(1,(incoming_length+outgoing_length)/2)` for smoothness. Invalid
  normals use isotropic fallback. Join cost is additive to the two fiberlet
  endpoint-proxy costs and adds no route length. Transitions and adjacency are
  canonically ordered. Graph source metadata binds the exact manifest and
  anchor hashes used for `fiberlets.json`.
- Learned path-quality rejection, overlap
  deduplication, extension, H/V and winding labels, CUDA batching, and
  production radius selection remain out of scope.

# C++ dense-fiber failure replay

- `vc_fiberlets fiberlet-replay` consumes a fiber-prediction Lasagna manifest,
  strict VC3D fiber JSON, required regular-normal Lasagna manifest, and output
  directory. All spatial arguments and artifacts use base-volume XYZ/base
  voxels. Defaults are `--fail 20`, `--radius 64`, `--match-refine 1`,
  `--beam 16`, `--lookahead-distance 384`, and `--batch 65536` native
  coordinates. The independent beam-front step defaults to 48 base voxels and
  is overridden by `--beam-step-distance`.
- The evaluated interval begins at the first control point's dense-line arc and
  ends at the final dense reference point by default. Replay-only `--length N`
  selects at most `N` positive finite base voxels from that point and clamps an
  oversized request at the reference end. One shared effective begin/end pair
  bounds anchor/path extraction, both evaluators, failure fractions, artifacts,
  and visualizations. Anchor extraction is local to cells intersecting the
  exact radius tube around that selected interval. The normal replay path
  exposes one logical graph through sparse anchor and fiberlet chunks; it must
  not materialize the whole corridor graph before evaluation. `--eager-graph`
  retains the complete graph path only as a diagnostic equivalence
  implementation. Spatially close but arclength-distant candidates remain
  eligible in either path.
- Eager replay uses the same global staged path extraction contract:
  prepare every curve once in parallel, merge one global unique corner union,
  batch only consecutive coordinate ranges, materialize all scores, then solve
  all curves in parallel. Required corners remain present outside the tube
  predicate. Batch and worker counts cannot change serialized path/graph
  results or the unique request population.
- Lazy replay stores anchors and fiberlets in separate local Zarr-v2-layout
  roots. Their arrays contain explicitly identified custom version-2 object
  payloads rather than tensor samples. Prefix and route arrays are separate so
  adjacency and beam ranking do not retain unused route geometry. A fiberlet
  owner chunk blocks only on the complete bounded halo of anchor chunks needed
  by its endpoint search.
- Lazy replay schedules the exact reference tube at storage-chunk resolution;
  it must not enumerate the complete corridor's anchor-cell population before
  cache workers start. Each requested anchor chunk enumerates its owned cells
  in canonical Z/Y/X order and uses the canonical exact segment-to-cell test.
  The post-refinement anchor-position predicate, fiberlet-interior point
  predicate, and cross-chunk NMS dependency context remain mandatory. Cache
  identity binds the complete clipped reference geometry, radius, source and
  algorithm metadata, and corridor-selector version, not a serialized cell
  list.
- Generated anchor and fiberlet payloads use separate `ChunkCache` scheduler
  lanes and one shared decoded-byte budget. Stable coordinate-plus-variant
  anchor IDs and canonical endpoint-pair edge IDs are the only graph state held
  by beam/frontier records. Serialized chunks are decoded exactly once when
  entering the existing `ChunkCache` LRU; there is no graph-private LRU and no
  resident serialized duplicate. Anchor and fiberlet caches remain separate
  participants in the shared budget. A decoded prefix payload builds and
  charges a deterministic two-endpoint incident index once. Incident queries
  batch-prefetch every possible owner chunk, lease those indexed payloads, and
  release the leases after the query. Float-cache anchors persist the exact
  endpoint prediction direction, presence, validity flags, Lasagna normal, and
  normal validity produced by canonical DP interpolation. Float-cache prefixes
  persist all five additive path-cost components, authoritative float path
  length, and exact base-space first/final nonzero steps. Beam ranking and
  transition scoring consume those records directly; replay must not resample
  source volumes or reconstruct endpoint scoring geometry.
  Lookahead requests route payloads for decoded segment densities and
  route-lattice segment lengths, but retains only stable IDs and scalar scores
  after each query. The current provisional best is additionally reconstructed
  for reference-error evaluation and final output. Cache eviction and reload
  cannot invalidate graph identity or alter tie order.
  Committed route reconstruction applies the same adjacent-point epsilon
  suppression as eager DP finalization. For the same extracted graph, cold and
  warm float-cache replay artifacts must be byte-identical to eager replay.
  Compact storage is intentionally quantized and excluded from this identity
  contract.
  Active payload leases may temporarily exceed the nominal cache budget, but
  unleased connectivity and geometry remain LRU-bound.
- Sparse payloads are strict and unpublished: no version repair or compatibility
  reader exists. Each payload binds kind, profile, chunk coordinate, dataset
  fingerprint, scalar widths, source scale, field directory, and checksum.
  Fields are deterministic little-endian structure-of-arrays blocks compressed
  independently with Zstd when smaller. Local publication uses a same-directory
  temporary file, file `fsync`, atomic rename, and parent-directory `fsync`.
  Prefix and route members are immutable and become readable only when both
  payload files are present, decode against the expected dataset/chunk codec,
  and have matching record counts. There are no per-chunk completion markers.
  A partial pair is incomplete and resumable; empty complete pairs are explicit
  valid payloads.
- Greedy replay uses the regular native one-way candidate generation,
  prediction loss, Lasagna-normal curvature, and validity rules. Graph replay
  uses a deterministic persistent beam over the complete immutable graph with
  the shared edge/join objective.
  The greedy evaluator and on-demand graph evaluator run concurrently;
  neither evaluator changes the other one's state.
- Graph replay has separate finite positive checkpoint-step, lookahead, and
  intermediate-prune distances in base voxels. It keeps up to the configured
  positive beam width of complete route
  histories and a shared logical checkpoint `C` that is no greater than the
  shortest retained history. Exact cost-bounded lookahead is the default, with
  a 384-base-voxel lookahead and 48-base-voxel checkpoint step. It expands
  without an intermediate width approximation, then retains at most the final
  configured beam width, which defaults to 16 and has no fixed upper policy
  limit. The existing exact cycle rules and one-million-state per-decision bound
  remain mandatory. Positive `--search-width N` explicitly selects
  approximate intermediate pruning at the configured `--prune-distance`,
  which defaults to 48 base voxels and is ignored in exact mode.
- Every successful fiberlet route stores one quantized total-cost density per
  emitted geometry segment. Route geometry and cost samples use independent
  offsets. Densities are per prediction voxel and use the fixed global
  `sqrt(density / 256)` uint16 codec; this unpublished route schema has no
  repair or compatibility reader. Forward/reverse arcs reverse geometry
  lengths and density samples together. Anchor cache identity is unchanged;
  the route-cost schema is part of fiberlet cache identity.
- Replay defaults to aggregate `fiberlet` cost mode. It ranks from the segment
  seed through the common horizon with the stored whole-edge and join costs,
  prorating only the horizon-crossing edge by length. The checkpoint is a
  commitment boundary, not a scoring boundary; an entering join is charged in
  full whenever any of its following edge lies before the horizon. Aggregate
  mode does not request route cost profiles. Its exact relaxed cost-to-go and
  intermediate-pruned search use the same aggregate objective.
- Explicit `stepped` cost mode integrates fiberlet cost profiles on one regular distance grid rooted
  at the current checkpoint. Its finite positive integration spacing is in
  base voxels. Arbitrary edge/grid/horizon boundaries linearly interpolate the
  piecewise-linear cumulative cost implied by piecewise-constant segment
  densities. Before geometric weighting, replay linearly blends each decoded
  subsegment density with the decoded-profile average of its complete
  fiberlet: `effective = (1-A)*average + A*subsegment`, where finite
  `0 <= A <= 1`. The average is
  `sum(decoded_density * segment_length) / fiberlet_length`; it is not derived
  from or normalized to the separately stored whole-edge cost. `A=0` is a
  constant density within each fiberlet, `A=1` is the stored profile, and the
  stepped-mode default is 1. The blend is replay-only and does not participate in anchor or
  fiberlet cache identity. Each interval cost is multiplied by `W^s`, where `0 < W <= 1`
  is the configured per-base-voxel weight and `s` is the interval midpoint's
  base distance from the checkpoint after the configured finite nonnegative
  delay `L`: the exponent is `max(0, s-L)`, so weight remains one through the
  delay. Delay defaults to zero, preserving immediate decay; half the
  lookahead is the initial opt-in experiment. `W=1` exactly conserves the decoded
  profile total. Joins before the checkpoint are excluded, a join exactly at
  it is included with weight one, and a join exactly at the horizon is
  excluded. Weighted ranking remains a scalar separate from unweighted
  five-component committed diagnostics.
- Stepped-only weight, delay, integration-step, and profile-blend options are
  rejected unless stepped mode is selected. Replay artifacts always store
  `cost_mode`; aggregate artifacts omit inactive stepped fields. Cost mode and
  stepped replay controls do not participate in cache identity.
- Exact search seeds one cost-ordered frontier from every retained live route
  and maintains one global set of the best configured-beam-width completions at
  `C+H`. Completion
  order is checkpoint-relative weighted exact-horizon loss followed by canonical persistent
  logical-route order; loss per voxel is diagnostic only. Completions are
  deduplicated only by seed plus their complete logical route, so any number of
  winning routes may share the same route through `C+D` and diverge later.
  Distinct physical expansion states are not merged merely because they have
  the same logical identity.
- The exact cutoff does not exist until the configured beam width of distinct
  full-route completions is available. Thereafter the worst retained weighted total
  is the one shared cutoff
  for every source route. A pending state is pruned only when its admissible
  lower bound is strictly greater than the cutoff; equal bounds remain eligible
  for canonical tie ordering. Fixed-size expansion batches are independent of
  worker count. All entries admitted under the batch-start cutoff count as
  expanded, workers publish ordered successor arrays, and the coordinator
  merges/account them in popped-route and successor order. Only that canonical
  merge mutates the decision-wide state budget.
- Exact winners remain live through their complete `C+H` lookahead routes when
  `C` advances by `D`. At the next decision, a retained terminal fiberlet that
  already covers the new horizon is immediately rescored by integrating its
  segment profile through the horizon; otherwise expansion resumes only at the retained
  physical endpoint. The globally best winner's prefix through the complete
  fiberlet containing `C+D` is the only route reference-evaluated and committed
  at that decision. Its future suffix is not reference-evaluated early.
- Bounded search first advances to the next checkpoint `C+D`, then to fronts at
  most `P` apart until the exact `C+H` horizon. All fronts are clamped at the
  selected route end; a final nondivisible interval is shorter. The final
  fiberlet stays whole in graph state and may overshoot a front.
- Between fronts, search is a deterministic uniform-cost label search. A label
  state is `(logical incoming directed fiberlet, absolute front-offset bin)`;
  front-offset bins are 0.5 prediction voxels. When histories reconverge at the
  same state, only the lowest accumulated-cost history remains, with ordered
  logical arc IDs breaking exact-cost ties. Its visited-node state becomes the
  state's cycle state; alternatives with different visited histories are
  intentionally discarded. Crossing labels integrate the exact segment
  profile through the terminal edge. Once `K` exact completions exist, expansion stops only when the
  next accumulated-cost lower bound is strictly greater than the `K`th exact
  completion, so equal-cost ties remain eligible. Queue exhaustion returns all
  available completions when fewer than `K` exist.
- A route's ranking score is the sum of two terms. The prefix from the segment
  seed through the current checkpoint uses the authoritative unweighted
  whole-edge and transition costs. The forward interval from the checkpoint to
  the exact logical front integrates the decoded segment-density profile with
  the configured checkpoint-local delay and geometric weights. Decoded profile
  values are used directly: they are never normalized or rescaled to force
  agreement with the separately stored whole-edge cost. The same integration
  algorithm is used at every weight, including `W=1`; changing the integration
  spacing can therefore retain small interpolation and accumulation-rounding
  effects.
- Edge intervals are half open for score ownership. An edge portion before the
  checkpoint belongs to the authoritative prefix, while the portion beginning
  at the checkpoint belongs to the weighted forward term. An entering join
  before the checkpoint belongs to the prefix, a join exactly at the checkpoint
  belongs to the forward term with weight one, and a join exactly at the horizon
  is excluded. When the front lies inside the terminal fiberlet, only the
  traversed profile interval contributes. When the front is exactly an anchor,
  the edge ending there and its entering join contribute, but no outgoing edge
  or join does. The complete terminal geometry and visited-node state remain
  attached to the candidate.
- At an intermediate front, identical complete logical routes are deduplicated
  and ranked by exact-front loss and stable logical IDs. The best continuation
  for every represented stable prefix is retained first, up to working width;
  remaining slots are filled by the globally best unselected routes. The
  stable prefix is the segment seed plus ordered logical arc IDs through
  `C+D`, never a history pointer. At the final front, only the best
  continuation for each actual next-checkpoint prefix is eligible for the
  globally best 16. Pruning a prefix that would become better by `C+H` is the
  explicit approximation boundary.
- Each expansion worker retains bounded local global candidates and stable-
  prefix representatives. Dominated labels and stale queued labels are counted
  separately from exact front completions and cost-bound pruning. Worker
  candidate limits are the sufficient local bound: one representative for the
  worker's stable prefix plus only the number of globally unoccupied fill
  slots. The first `C+D` front uses the full target width because its stable
  prefixes do not exist until that front is reached. Worker results merge in
  canonical input order and are reranked by deterministic
  logical IDs, so 1-thread and multi-thread executions produce identical
  output. One generated-state limit covers the complete decision and aborts it
  without retaining partial worker output.
- In bounded search, after pruning `C` advances by `D` and each winner is
  retained through the complete fiberlet containing the new checkpoint.
  Depending on its existing overshoot, a beam may add no fiberlet, one
  fiberlet, or several fiberlets in that step. Exact search instead retains the
  full lookahead route described above. Both preserve
  `C <= min(retained_history_length)`.
- Compact-cost replay consumes the decoded authoritative segment-density
  profile and join costs. Logical-front scoring integrates the profile through
  the terminal edge;
  final reference-end or failure materialization may additionally clip output
  geometry without altering persistent graph state. Exceeding the deterministic
  per-decision one-million-state bound is an explicit error. A route that
  exhausts before the common horizon is excluded; if all routes do, normal
  graph exhaustion applies.
- Persistent histories use shared immutable parent records and retain IDs,
  cumulative costs, path length, incoming transition, and visited-node state,
  not duplicate route geometry. A reference failure closes the chosen history,
  resets the segment origin, and reseeds the full population. The reference
  never removes graph candidates during ranking.
- Persistent search bookkeeping must not scale with the already committed
  prefix. Logical routes use exact canonical parent/arc identity plus exact
  ancestor/first-divergence ordering; physical candidates remain separate.
  Decision-score initialization starts from cumulative scalar edge and join
  cost at the history immediately before the checkpoint and visits only the
  checkpoint-to-horizon suffix. Logical-route interning cleanup advances a
  bounded stable cursor through the ordered registry; it never performs a
  whole-registry sweep at every checkpoint.
  Exact queue ordering, completion deduplication, and cutoff maintenance use
  scalar cumulative state and those persistent identities only; they never
  materialize logical arc or route-point vectors.
  Cycle membership uses an immutable exact-key Patricia trie and is never
  compacted by copying the accumulated prefix. Selected-route reference
  matching resumes from the nearest evaluated physical-history ancestor and
  evaluates each newly selected suffix once. Diagnostic indices are allocated
  from that same newly selected suffix in their historical order. Full route
  vectors, matches, steps, and consumed-node output are assembled once when a
  segment terminates. Explicit decision diagnostics may additionally build the
  complete route payloads they serialize. These are implementation constraints
  only and must not change ranking, costs, failure locations, cache identity,
  or replay JSON.
- The `fiberlet-replay` CLI runs the classic greedy and graph evaluators at the
  same time. Its positive `--threads N` setting is one trace-evaluator worker
  budget split deterministically between those two nested searches, not `N`
  workers assigned independently to each evaluator. This scheduling rule does
  not change graph ordering or numerical accumulation.
- Ordinary graph-arc queries expose connectivity and aggregate edge metadata
  only. Decoded cost profiles are requested explicitly by the scorer and kept
  in a bounded decision-local cache, so queue expansion does not reconstruct or
  copy route payloads on every adjacency lookup. Exact-search relaxed
  cost-to-go memoization is decision-local, scalar, and conservative; its state
  count, cache hits, and zero-bound fallbacks are diagnostic output and do not
  affect ordering.
- Beam-step, lookahead, prune distance, search width, geometric cost weight,
  geometric cost delay, and cost integration spacing are replay-only
  metadata. They must not affect anchor or fiberlet cache identity, corridor
  selection, generation settings, chunk payloads, or prefetch scheduling.
  On-demand traversal may populate a missing chunk, but repeating against a hot
  cache must not rewrite cache files.
- Default cached replay progress exposes independent `cache/prep` and `trace`
  terminal bars while preprocessing and tracing overlap. Cache progress is
  `(resolvedAnchors + 16*resolvedPrefixes) /
  (expectedAnchors + 16*expectedPrefixes)` over deterministic scheduled
  prefetch keys. Trace progress is exactly
  `min(greedy_reference_fraction, fiberlet_reference_fraction)`, with no
  weighted combination between the two. Resolution counts generated and
  persisted chunks identically and deduplicates reloads. Eager replay exposes
  only trace progress. Cache progress excludes data-dependent neighbor-prefix
  and committed-route reads, which occur during tracing. All fractions are
  monotone. Non-finite tracer values are
  ignored and stale or restart-local callbacks cannot move either tracer
  fraction backward. A 250-millisecond ticker refreshes elapsed time even
  without worker callbacks. The compact line contains only one elapsed field;
  cache and trace retain independent ETAs while overlapping, and completed
  scheduled cache progress is removed from the line. Trace additionally shows
  a current-speed ETA over a rolling ten-second fraction window, reporting
  `n/a` when that window has no positive progress. Requested visualization and durable publication use a separate
  output phase after tracing, terminate their line before result/error output,
  and never alter or masquerade as reference progress.
- Each completed bounded fiberlet decision publishes search diagnostics with
  its progress callback. The rollout expansion count is the total number of
  states whose successors were enumerated across all intermediate fronts. For
  each maximum-lookahead-front input where the existing strict cutoff actually
  stops expansion, the diagnostic subtracts that input route's loss at the
  front start from the cumulative cutoff and divides by the front length in
  prediction voxels; it publishes the minimum such local loss density. This
  normalization is diagnostic only: search continues to compare the unchanged
  cumulative raw-loss cutoff against queued lower bounds. The CLI retains the
  last published values until the next completed fiberlet decision. Exact mode
  omits both diagnostics, and a bounded decision omits the cutoff if it never
  binds.
- `--stats` replaces the concise bar with detailed machine-readable stage,
  chunk, restart, evaluator, cache, visualization, and publication diagnostics.
  It does not change scheduling, generated payloads, tracing, or artifacts.
- Repeated `--decision-window BEGIN,END` options require `--stats` and retain
  complete decision records only when the selected matched reference arc is in
  an inclusive window. They alter diagnostic materialization only, not search,
  matching, restart history, cache identity, or generated chunks. Serialized
  route geometry begins at the checkpoint and ends at the lookahead; it must
  not rebuild the already committed prefix.
- `fiberlet-replay --arc BEGIN` selects an absolute base-voxel reference start
  for both greedy and graph replay; `--length` then limits that focused
  interval. Greedy starts from the sampled point and tangent at `BEGIN`, and
  graph replay seeds at the same reference arc. A focused run retains the full
  first-control-point corridor's cache identity and containment predicate but
  schedules only the focused interval. Missing chunks therefore have the same
  complete contents as full replay chunks; focused geometry must never be
  persisted under a full-corridor identity.
- On-demand chunk diagnostics bind every generated anchor/fiberlet chunk to a
  stable full-interval schedule index and nearest reference arc. Generated
  counts are monotone, while spatial chunks may complete out of schedule order.
  Fiberlet generation exposes candidate generation, preparation, sampling,
  materialization, and search phases with their own completed/total counts.
- Anchor ready-cell completion and its condition predicate are synchronized by
  the same mutex. Cache dependency waits do not poll, emit heartbeats, or use
  timeout recovery. A failed generated dependency preserves the exact dataset
  stage, chunk key, cache status, and underlying error.
- Fiberlet candidates may be generated concurrently by canonical source
  anchor. Per-source output is merged in source order, and completed prepared
  geometry may be released by its owning search worker. Neither operation may
  change candidate order, per-candidate arithmetic, graph identity, or encoded
  payload bytes.
- Anchor and Fiberlet metadata include one shared producer-generation contract
  in their algorithm identity. Any implementation change that can alter an
  authoritative anchor, prefix, or route record for otherwise identical
  effective inputs must increment that contract before writing payloads. The
  current unpublished contract is version 3. Compiler identity/version and
  build configuration are diagnostic metadata only; they are excluded from
  algorithm and dataset fingerprints and do not make otherwise equivalent
  caches incompatible. Default cache discovery may reuse an earlier version-3
  toolchain-specific namespace after comparing the remaining structured
  identity. Version-2 directories are not migrated, repaired, or read through;
  explicitly selecting any scientifically incompatible root is an error.
- Cached anchor prediction, presence, and Lasagna-normal fields are not a
  generation consistency boundary. Fiberlet generation uses freshly sampled
  endpoint and interior evidence for every metric and decision and does not
  compare it with those cached fields. Replay transition scoring likewise uses
  a single-flight derived anchor view which resamples that evidence at each
  effective anchor position. Cached stable IDs, fitted geometry, and positions
  remain authoritative. A scheduled failure preserves the owner key, terminal
  cache status, and original nested generator message.

## Staged Fiberlet replay filtering

- Cache-backed `vc_fiberlets fiberlet-replay` accepts the ordered repeatable
  `--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z`, `--join-angle`, `--cost-profile`,
  and `--max-states` reduction options. No stage preserves ordinary replay
  behavior. Filtered replay is incompatible with eager graph extraction.
- Replay filter stages are cubic lattices in global base-volume XYZ. Each
  offset is normalized modulo its positive side. Selected boxes are complete,
  half-open boxes within the prediction volume and execute in deterministic
  Z/Y/X order. They are never re-anchored to a replay interval, storage chunk,
  generation cell, or processing-cache chunk.
- The last stage includes every complete global box whose corridor-expanded
  extent intersects the selected reference polyline. Planning proceeds
  backward: every preceding stage includes all complete boxes intersecting a
  required later box expanded by the dataset's declared maximum endpoint reach.
  Base anchor/Fiberlet generation covers the union of required boxes expanded
  by that reach and clipped to the volume.
- Stage boxes, anchor-generation cells, and persistent Fiberlet storage chunks
  are independent globally anchored layouts and may have different sizes and
  boundaries. Coverage crosses layouts only through half-open base-space
  extents and the canonical storage-owner halo calculation.
- Filtering reuses the canonical staged analysis, simplification, overlay, and
  write-back implementation. Derived layers are invocation-local and are never
  published as a persistent filtered cache. Canonical unfiltered on-demand
  chunks retain their normal persistent cache behavior.
- Expanded support exists only to make every selected final-stage box complete.
  Replay seed and traversal predicates remain the original selected reference
  corridor, so support expansion cannot make out-of-corridor graph nodes
  eligible. Replay result metadata records the ordered stage and filter policy.

## Chunk-local optimal Fiberlet-route diagnostic

- `vc_fiberlets chunk-route-stats` is a read-only graph diagnostic using the
  regular on-demand anchor and fiberlet cache infrastructure. Its analysis
  chunk is an axis-aligned, half-open base-coordinate box `[minimum, maximum)`
  selected by a base-XYZ minimum and a base-voxel side. The analysis box is
  independent of persistent cache chunk ownership and replay lookahead; route
  traversal continues until first exit even when it exceeds either side or the
  regular replay horizon.
- An anchor belongs to the analysis chunk according to its stored base-space
  position. The analyzed physical-fiberlet population contains every fiberlet
  with at least one endpoint inside. A directed outside-to-inside arc is an
  entry and an inside-to-outside arc is an exit. A route starts with its entry
  arc and terminates at its first exit; an outside/outside fiberlet is not an
  internal graph decision and is excluded.
- Routes are simple. Entry initialization marks both the outside source and
  inside target visited, and every later target is rejected when that anchor
  already occurs in the route. Consequently neither an anchor nor a physical
  fiberlet can be revisited. A route may curve back spatially and leave through
  any face only through a sequence of ordinary valid joins; this never permits
  a cycle or an unconstrained turn.
- Every transition uses the same strict maximum anchor-join angle, prediction
  validity check, and normal/tangent-aware local join objective as regular
  cached replay. Entry, internal, and first-exit edge costs and intervening
  joins are each charged exactly once. The selected stored-float or fixed
  sqrt-`uint16` edge-cost view is explicit; join costs remain float.
- Each entry is solved independently by exact nonnegative additive-cost search.
  All exactly equal minimum-loss completions contribute to the retained union.
  The diagnostic reports reachable entries, tied optima, route length/loss
  distributions, and the union of inside anchors and physical fiberlets used
  by any optimum. Internal-Fiberlet before/after counts and reduction are
  reported independently of boundary-crossing entry/exit Fiberlets. Exceeding
  the per-entry state bound, which defaults to 5,000,000 generated states,
  fails the complete diagnostic rather than publishing partial pruning
  statistics.
- Missing anchor or fiberlet chunks are generated and persisted through the
  same `FiberletOnDemandPreprocessor`, separate cache participants, dependency
  halo, serialization, and shared decoded-byte LRU used by tracing. Generated
  payloads are canonical full storage chunks whose identity does not depend on
  the analysis box. Existing compatible chunks are reused; a hot pass does not
  rewrite them. Incompatible metadata, generation/decode failure, invalid
  costs, or invalid geometry fail the complete diagnostic. The command never
  rewrites or prunes existing graph payloads and never marks a partial cache
  complete.
- Staged regional mode accepts an ordered, repeatable sequence of cubic
  analysis sides and global base-XYZ offsets plus a selected half-open bbox.
  Offsets are normalized modulo their side. Each stage enumerates every
  complete global-lattice box contained by the selected bbox in canonical
  Z/Y/X order. Analysis boxes need not align with anchor cells or storage
  chunks. Every derived stage uses separate temporary anchor and Fiberlet
  overlays with exactly the initial datasets' storage layout and encoding; the
  initial persistent caches are never rewritten.
- Invocation-local stage overlays use one shared, bounded write-back LRU of
  canonical serialized chunk bytes. Dirty anchor chunks and atomic
  prefix/route owner pairs remain in memory and are read there by later boxes
  and stages. They are not written merely to move between stages. When the
  shared `--cache-gib` allowance is exhausted, deterministic LRU victims spill
  through bounded asynchronous atomic writes; resident and queued/writing
  bytes remain charged until their storage is released, and the ordinary
  decoded-chunk allowance is reduced by the same amount. A prefix/route pair is
  one eviction, visibility, and failure unit. Unspilled temporary bytes are
  discarded only after all caches stop and outstanding spills drain.
- Detailed stage payload hashes cover the logical union of metadata, spilled
  files, and current memory/pending chunks without forcing a flush. Newer
  in-memory bytes shadow an older spilled file at the same relative path. This
  logical hash must equal the hash of fully durable identical payloads.
- A missing upper chunk falls through unchanged to the preceding layer. An
  explicit empty payload shadows lower data through arbitrary layer depth.
  Fiberlet fallback requires both prefix and route members to be absent;
  partial pairs, corrupt upper payloads, and incompatible coordinate/layout,
  storage, quantization, source, or processing contracts are errors.
- Boxes within a stage execute serially, and an overlapping later box reads
  earlier writes from that same stage. Only a Fiberlet whose canonical first
  endpoint position is inside the current box is eligible for removal. Every
  affected initial-layout owner chunk is rewritten from its effective current
  payload, including a canonical owner outside the geometrically intersected
  chunks. Publication must be a strict record subset: retained anchor,
  prefix, and route fields remain exactly unchanged, and removed IDs can never
  be restored by a later box or stage. Explicit empty touched chunks are
  persisted in the temporary layer.
- An inside anchor is removed only if the effective graph after the proposed
  Fiberlet update contains no incident surviving Fiberlet. Outside-owned and
  lower-layer incident Fiberlets therefore retain it. Derived stage layers are
  invocation-local and removed after final reporting; persistent/completed
  reduction layers require a later publication protocol.
- Each stage reports only geometry in the union of its complete analysis
  boxes: inside anchors, the canonical union of all incident Fiberlets, and the
  canonical union of Fiberlets with both endpoints inside one complete stage
  box. A Fiberlet crossing between adjacent stage boxes is therefore `all`,
  not `interior`. Original, inherited input, and output populations use that
  identical stage-local domain, so offset stages never count untouched
  selected-bbox geometry. The joint report separately compares original and
  final anchors, all incident Fiberlets, and interior Fiberlets over the
  complete selected bbox. Sequential local pruning is deterministic and
  monotone but does not prove preservation of a globally optimal replay route
  for arbitrary later boundaries.
- Exact analysis and simplification consume the same immutable materialized
  local graph for a box. Materialization loads every required anchor, prefix,
  and route owner once, applies the configured anchor view once, and constructs
  arcs and transitions without point cache queries. Transition construction,
  entry searches, serialization, and independent owner-chunk writes may
  execute through reusable worker pools and canonical index slots, with
  deterministic lowest-index failure selection. Exact entry-search workers
  read the immutable local graph through fixed, disjoint index partitions and
  own all queue, ancestry, metric, and terminal scratch; their hot search loops
  perform no scheduling or synchronization. Every replacement is prepared from
  the old graph before Fiberlet prefix/route pairs are published; anchor chunks
  publish only after all Fiberlet pairs. Box and stage order remains serial and
  canonical.
- Each processed box additionally produces an exact in-memory simplification
  report. A directed state is proven dead only when it is absent
  from either entry-forward or exit-backward reachability. The analysis is
  conservative under the simple-route no-revisit constraint: an uncertain
  state may remain, but a valid route must not be removed. A physical Fiberlet
  remains when either direction is live, and every later adjacency and macro
  operation must retain the explicit per-direction live mask.
- Every anchor unreferenced by the live physical graph is removed. Surviving
  outside endpoints are explicit boundary portals with their original stable
  anchor identity. Macro application must validate all hidden target anchors
  against route history atomically before adding any of them.
- A physical macro may cross only an interior anchor with exactly two live
  incident physical Fiberlets and admissible mutual joins in both directions.
  It stores the ordered original directed Fiberlet IDs, full anchor sequence,
  edge losses, join losses, and lengths. Authoritative scoring replays those
  scalars in the original order and association. Branches, portals, one-way
  continuations, and cycles stop physical contraction.
- Directed one-successor states are reported separately. Disjoint directed
  contraction additionally requires a unique predecessor; maximal forced
  continuation descriptors may overlap at convergence so arbitrary reached
  states can use them without changing graph choices. Cycles remain explicit.
- A physical Fiberlet ID is already the canonical exact endpoint-key pair.
  Exact same-endpoint duplicates are invalid input. Distinct endpoint variants
  must not be removed as cost-dominated because doing so changes the valid route
  set under path-dependent visited-anchor histories. Macro representations are
  in-memory references to original Fiberlets and must not be serialized as an
  ordinary single-Fiberlet lattice route.

- Dense-reference matching is monotone and local. Greedy supplies its nominal
  step and graph replay each actual dense fiberlet edge length. Failure is the
  first Lasagna-oriented threshold error strictly above `--fail T`; equality is
  accepted. The already selected Euclidean reference match is not changed.
  At that matched reference point, the evaluator samples and normalizes the
  local Lasagna surface normal in the sampler's declared working scale, then
  decomposes the base-coordinate error into normal magnitude `dn` and full 2D
  tangent-plane magnitude `dt`. The threshold error is
  `sqrt(dn^2 + (dt/4)^2)`, giving ellipsoid radii `T` normal and `4T`
  tangential. Tangential means the plane orthogonal to the Lasagna surface
  normal, not the learned fiber tangent. A missing, invalid, non-finite, or
  zero-length normal conservatively uses the old Euclidean error with no
  relaxation. At `T=0`, exact zero is accepted and every nonzero error fails.
  Native
  termination, graph exhaustion, and lack of an admissible graph seed are typed
  failures rather than successful termination.
- Fiberlet seed selection first rejects Euclidean distances above `4T`, then
  applies the exact same Lasagna-oriented evaluator inclusively and orders
  usable seeds by reference arc, threshold error ratio, and node ID. The seed's
  first stored match contains the same complete measurement as later route
  points.
- Each failure terminates only the current evaluator segment. Reset advances by
  at least its nominal step, samples the exact authoritative reference point and
  fitted forward nonzero-edge tangent, clears graph incoming-join history, and
  starts the next segment. Reset count is bounded by reference length. Graph
  replay finishes a failure-containing fiberlet before reseeding, while the
  immutable first failing dense point remains the event location. At the
  selected end, threshold failure takes precedence over completion and later
  samples are not evaluated. An end inside a fiberlet retains the selected
  edge's full identity/cost accounting but only route samples through the bound;
  it has no stop anchor and is marked `terminal_partial_edge`. Reset jumps are
  distinct segments and must never be emitted as connecting line geometry.
- Every evaluated match stores explicit Euclidean, optional normal, optional
  tangential, threshold-equivalent, ratio, and local-normal-valid fields. A
  distance failure copies the exact terminal match measurement. Graph exhaustion
  retains the existing last-match diagnostics; failures without an evaluated
  point use null diagnostics. Strict publication validates finiteness,
  non-negativity, component reconstruction, the ellipsoid formula, ratio,
  invalid-normal fallback, geometry distance, and failure/match identity.
  Failure records also contain tracer-local index, typed reason, matched reference
  arc and evaluated-interval fraction, optional evaluator point/index,
  segment index, and segment-local point index. Missing-seed or termination
  events still carry an exact reference reset point. Each callback prints the
  current independent greedy/fiberlet counters. Callback arrival order is only
  diagnostic; persisted visualization order is canonical by reference arc,
  tracer enum, and tracer-local index. Both async evaluations are joined before
  either exception is propagated.
- The authoritative strict version-2 root stores source/scale/config bindings,
  requested/effective interval metadata, exact selected reference geometry,
  complete segmented evaluator results and
  matches, failure arrays/counts, trace OBJ/JSON descriptors, and an ordered
  visualization index. Runs publish an immutable `runs/<content-hash>`
  generation before atomically replacing `fiber_replay.json`. Each visualization
  also has an atomically replaced stable top-level alias named
  `fiber_replay_visualization.<tracer>.<failure-index>.json`; its artifact paths
  reference the current immutable generation. Strict version-1
  single-visualization artifacts remain readable and normalize their trace and
  optional graph route into the current segmented display representation.
  One authoritative threshold descriptor records the ellipsoid shape, normal
  radius, fixed tangential factor/radius, strict comparison, and invalid-normal
  policy; nested greedy and fiberlet descriptors are generated from the same
  values. The unpublished ambiguous `error_base_voxels` and `error_ratio` replay
  keys are not emitted and have no repair path.
- Visualization is disabled by default. `--vis` creates one diagnostic tube per
  failure using exact reference bounds `[failure_arc-along,
  failure_arc+along]` clipped to the evaluated interval and the configured
  radius. These local anchors, stages, paths, and graph are diagnostics only;
  the evaluators are not rerun. The local manifest also references cropped
  full-run reference and both segmented evaluator traces plus its failure
  marker. `--along` is visualization-only and does not limit evaluation.
- Every newly generated failure visualization also contains exactly three
  self-contained sheet-aligned OBJ/MTL/TIFF triples for the reference, greedy,
  and fiberlet traces. `--vis` requires `--volume` to name one concrete local
  uint8 OME-Zarr dataset array/group, not the pyramid root. The group must be
  advertised by its parent `multiscales` metadata. The producer maps
  base-volume trace coordinates into that group with the declared dataset/base
  transforms before calling the shared renderer.
  Geometry is produced directly by `buildLineViewSurfaces()` with the unchanged
  default `LineViewConfig`: the exact trace points are the longitudinal samples,
  the normal-aligned surface has 21 cross samples, and its existing inferred
  width and transported frame are retained. Each surface is rendered through
  the same extracted helper used by `vc_lasagna_line_probe`, including blocking
  dependency prefetch and `sampleCoordsFineToCoarse()`. Before that unchanged
  renderer is called at scale one, each component's coordinate grid is
  endpoint-preservingly resampled from its maximum row/column arc extent in the
  selected group's voxel coordinates. The resulting grid has at least two
  samples per axis and no more than one group voxel of arc per texel. Replay has
  no independent render-scale option.
  No replay-specific surface builder, mask renderer, uint16 path, or PNG path
  exists. Evaluator resets and clipped trace boundaries remain disconnected
  surface components. Their standard VC3D textured meshes are packed into one
  grayscale atlas per trace type with a replicated one-pixel border and an
  affine transform of their existing UVs. Empty trace types use an empty OBJ,
  valid MTL, and 1x1 TIFF. The manifest records the selected group path, its
  actual base-to-group scale/offset transform and shape, native-grid contract, and
  artifact hashes.
- Independently of the failure-local artifacts, every replay run with `--vis`
  publishes indexed full-selected-interval direct-inspection JPEGs. It builds
  the reference `LineModel` and one line model per at-least-two-point fiberlet
  replay segment through one shared batched normal-sampling/default-
  `buildLineViewSurfaces()` helper. Both the reference- and fiberlet-centered
  top surfaces and side slices use the same shared fine-to-coarse CT renderer
  as the failure strips. The renderer first constructs
  the native selected-group coordinate grid and passes render scale `8` to that
  shared renderer, producing eight samples per native-grid interval without a
  post-render resize. The per-failure OBJ/MTL/TIFF strips retain their existing
  native selected-group sampling. Every wrapped block contains reference top,
  reference side, fiberlet top, and fiberlet side in that order. The reference
  centerline, greedy trace, and fiberlet trace are overlaid in yellow, red, and
  cyan respectively. Greedy and fiberlet
  points project through their strict stored matched absolute reference arcs;
  the selected interval begin is subtracted before mapping into the returned
  surfaces' authoritative raw coordinate grids. Every non-seed greedy point
  and every fiberlet point must have contiguous, monotonic, in-range match
  metadata, and reset segments remain disconnected polylines. In fiberlet-
  centered strips, stored matched reference points are projected at their
  corresponding route-point columns, greedy points are interpolated only
  through the component's stored monotonic match arcs, and the cyan fiberlet is
  the actual surface centerline. Samples outside a component's covered arc are
  omitted; samples covered by overlapping reset-component intervals appear in
  each component. Rendered components are concatenated in source-segment order
  with eight black columns between them. The root records source segment,
  matched arc interval, top/side row and column ranges, and separator semantics.
  No nearest-point rematching or alternate frame construction is permitted.
  Every failure is
  marked at its stored pre-reset error arc by a three-pixel full-strip-height
  vertical band: red for greedy and cyan for fiberlet. Coincident marker pixels
  are magenta. The later reset seed is not marked.
- The overview vertically stacks complete four-strip blocks and left-aligns
  unequal widths on black. When any 8x strip exceeds `32000` columns, all four
  are partitioned into the same minimum number of equal-progress-fraction
  blocks. Each strip independently maps those fractions to exact half-open
  column ranges; every rasterized column is copied exactly once without
  resizing. As many complete blocks as fit are stacked in one JPEG. A block is
  never split across files; remaining blocks continue in the next indexed JPEG.
  Every output dimension must be at most `65000` pixels, and one complete block
  that cannot fit is rejected before allocation. It
  covers exactly the selected replay geometry, so `--length` also bounds this image. It is written before
  generation hashing as `replay/full_strip.NNNNNN.jpg`, described by the version-2
  root with selected arcs/reference-point count, source group transform and
  shape, full unwrapped top/side dimensions, render scale, marker semantics,
  exact page/block ranges, component placements, colors, paths, and content
  hashes, and copied to stable top-level `fiber_replay.NNNNNN.jpg` aliases. A
  successful shorter or non-visual run removes stale indexed aliases and the
  unpublished singular alias. This root overview is
  separate from the per-failure OBJ/MTL/TIFF napari artifacts and is not a
  direct visualization manifest.
- The napari viewer takes one direct visualization manifest through `--replay`;
  there is no index argument. An aggregate root is discovery/reporting data and
  is rejected with either a directly usable manifest path or a request to rerun
  with `--vis`. The viewer rejects manual crop/anchor/path arguments, path
  escapes, hash/geometry mismatches, and malformed strict state. Reload rereads
  the same stable direct alias and does not reload the external presence Zarr.
- Replay strip artifacts are an optional all-three-or-none part of a
  visualization. When present, the viewer strictly resolves each OBJ's local
  MTL and each MTL's local hashed TIFF, validates the standard textured-mesh
  topology, atlas padding and transformed UVs, reads the
  stored finite grayscale texels, bilinearly tessellates the validated coarse
  OBJ surface to one displayed vertex per native-grid texel, derives
  one shared p1/p99 display range, and creates independent hidden grayscale
  napari Surface layers. It neither accepts a CT-volume argument nor opens the
  recorded provenance path. Empty meshes remain valid empty layers. Reload
  reparses geometry, faces, UVs, materials, and values from the nine artifacts and updates the
  existing Surface layers without changing their display state. There is no
  reader or repair path for the discarded replay-specific strip formats.
- `vc_fiberlets benchmark` runs the exact shared local tube anchor/path
  extraction without artifacts. Its interval starts at the first control point
  and, when `--along` is omitted, extends through the reference end. An explicit
  positive `--along` limits the interval to that many base voxels and remains
  clamped at the reference end. Replay's separate visualization `--along`
  remains a 128-base-voxel arclength half-window around each failure. The final
  row reports
  exact interval/radius, populations and rates, coordinate-call/unique-voxel/
  owned-memory counts, DP nodes/rate, and per-stage wall/CPU/effective-core
  timings.
  Benchmark comparisons must retain identical inputs, parameters, build type,
  and interval.
- Benchmark and full replay extraction emit the same versioned
  `fiberlet_extraction_profile version=26` key/value schema. The profile exposes
  deterministic workload counters and finer anchor/fiberlet phase timings.
  Enclosing phase fields are wall time, `_work_seconds` fields are summed
  worker/candidate time, and CPU fields are process CPU time. Corner insertion
  attempts count every positive-weight attempt; the existing sampled-voxel
  count remains the deterministic globally unique union. Profiled phase sums
  and residual elapsed time make uncovered overhead visible. Progress callback
  cost remains in its enclosing phase. Diagnostics must not change sampling,
  fitting, candidate generation, DP math, ordering, serialized artifacts, or
  determinism.
- Version 22 retains the version-21 split of robust direction proposals into
  axis-producing and final membership-only calls. For each kind, logical visits count the complete
  candidate range considered by the first pass, eligible visits count entries
  passing the immutable validity/presence/direction predicate, indexed visits
  count entries physically traversed by proposal scoring, and cutoff visits
  count entries physically traversed while materializing retained membership.
  The legacy local-tensor visit count remains two times complete logical
  cardinality per proposal for comparison across implementations.
- Proposal-buffer counters report complete-cardinality assignment/retained
  storage initializations plus actual bytes initialized and copied into
  returned evaluation state. They are diagnostics and do not change fitting
  behavior.
- Version 25 materializes compact robust-proposal evidence once per cell in
  increasing logical-observation order. Each record stores float position,
  already-normalized direction, presence, and its original logical destination;
  both axis proposals and final membership reuse the same immutable records.
  Invalid positions remain represented and naturally contribute zero Gaussian
  mass. Full-support centroid, objective, peak, and final-support scans retain
  their shared indexed observation ranges. The profile reports prepared record
  count/size and summed preparation work, and bounded worker admission replaces
  the former eligible-index scratch with the full record bytes.
- Version 20 replaces pair-local tile sampling groups with bounded exact-union
  partitions. Tiles remain in canonical order. Each partition merges exact X
  intervals for structured `(z,y)` rows, samples every union coordinate once
  in deterministic bounded batches, and joins sampling. Version 24 then builds
  one compact observation and one presence gradient per partition-union voxel.
  Tiles retain only dense uint32 local-to-union maps. Cell support and owned-
  observation traversal remain in canonical tile-local Z/Y/X order, and a
  shared gradient remains usable only where it was interior to that tile's
  original sample bounds. Large extractions stream through multiple partitions
  rather than requiring the whole union to remain resident.
  Sampling and fitting worker counts are admitted independently against
  `maximumConcurrentSampleBytes`; shared samples, row metadata, sampler scratch,
  batch/error control, ready-cell queue storage, timing storage, tile buffers,
  gradients, compact observations, and per-cell scratch are included in the
  reported maximum live-byte ceiling. Every sampler call receives one lower-
  level thread. A failed batch is selected by batch order and assigned to its
  partition cells so final failure propagation remains canonical by cell.
  Partition owners retain immutable compact observations until all published
  cells finish. Version 20 reports partition count and duration quantiles, shared
  batch count/maximum size, shared-sampling wall/CPU, shared/accounted bytes,
  while version 24 separately reports shared-observation construction and tile
  index-map worker time. Submitted voxels and shared-observation voxels count
  partition unions; reused
  voxels count tile occurrences not submitted. The exact whole-extraction tile
  union remains a diagnostic and may be lower than submissions when bounded
  partition boundaries repeat overlap.
- Version 2 divides anchor fitting into exclusive summed-worker setup, seed
  generation, seed-pair refinement, initialized-component finalization, local
  direction/position refinement including backtracking, direction-conditioned
  peak search, and final-evaluation phases. Their explicit profiled sum and
  residual reconcile against total fitting work. It separately counts fitter
  invocations/nonempty cells, seeds/pairs/pair iterations, local attempts and
  accepted steps, backtracking evaluations, peak components, and observation
  visits for every repeated assignment, tensor, objective, centroid,
  refined-state, peak-response, and final-evaluation scan. Peak response
  requests include grid-cache hits; computed-grid responses are cache misses;
  acceptance responses are uncached subpixel evaluations. Seed-pair iterations
  exclude their final reassignment/objective pass, and local attempts exclude
  the initial refined-state evaluation. The legacy `anchor_fit_iterations`
  field counts accepted cell-level refinement once per output component;
  `anchor_fit_local_refinement_accepted_steps` is the canonical cell-level
  value.
- Version-2 observation-visit fields are logical operation counts. Exact broad
  phases may avoid detailed kernel calculations while retaining these counts,
  so before/after workloads and fitting decisions remain directly comparable.
- Version 3 partitions local-refinement worker time into tensor proposals,
  centroid proposals, and refined-state evaluations. The remaining parent time
  is reported as local control work and includes setup, state interpolation,
  acceptance/convergence decisions, and profiling overhead. These four values
  reconcile to total local-refinement work. Refined-state evaluation includes
  the initial state and every backtracking candidate and computes spatial
  Gaussian support, directional evidence, assignments, and the normalized
  objective.
- Version 4 adds robust no-outlier/trimmed/non-unique/iteration-limit component
  counts, candidate/actual trimmed and retained evidence mass, and spatial
  candidate counts by halving depth. Tensor proposal time includes competitive
  assignment, weighted histogram cutoff selection, and retained sampled-axis
  aggregation. State-evaluation time contains fixed-direction spatial
  objectives. Robust component and mass counters accumulate proposals across
  bounded passes and are not final-population counts.
- Version 6 separately reports logical gradient attempts and physical
  tile-gradient computations. Tile gradients are constructed once and reused
  by overlapping cell halos. Robust histogram/tensor accumulation, paired
  spatial objectives, and transverse peak evaluation may regroup otherwise
  equivalent floating-point operations; small numeric differences from
  version 4 are accepted for this anchor fitter.
- Version 13 adds solve-local prepared-node/index/state byte maxima, prepared
  node count, reached-node and generated/valid/reused-edge counts, and separate
  node-preparation worker time. DP visit/lookup counters exclude outgoing work
  from the final interior layer because that layer transitions directly to the
  sink. State-memory accounting is the global predecessor bytes plus the two
  largest adjacent rolling cost layers; adjacent-layer populations are counted
  during parallel node generation rather than by a separate serial pass.
- Version 14 changes `interpolatedScoringPoints` to actual endpoint plus unique
  lazy-node interpolations. It separately reports endpoint interpolations, lazy
  requests, unique materializations, cache hits, maximum node-to-cache index
  bytes, and shared prepared-scoring/page-index bytes retained during search.
  Sparse interpolation timing samples use a fixed hash of canonical candidate
  index and packed node key; exact resolution/materialization counts remain
  complete and deterministic.
- Version 15 adds `anchor_support_stencil_cells` and
  `anchor_clipped_support_cells`. Their sum equals `anchor_work_cells`.
  `anchor_candidate_observations` retains the represented sample-cube count,
  while retained-observation and gradient counters retain their logical
  populations when the support stencil avoids rejected-site tests.
- Version 16 adds `anchor_fit_owned_discovery_observation_visits`,
  `anchor_fit_owned_initialization_observation_visits`, and
  `anchor_fit_avoided_owned_support_observation_visits`. Public fitting reports
  coordinate-discovery and stable-filter visits. Production reports direct
  owned visits and the visits avoided relative to its former two complete
  support-range scans; existing logical observation populations are unchanged.
- Version 17 splits direction-conditioned peak data into 16-byte float32 hot
  response records, parallel 32-bit evidence indices, and 16-byte retained
  evidence records. Invalid-gradient retained observations remain evidence
  because their aligned weight contributes to gradient coverage. Prepared
  response/evidence counters report populations; response-observation visits
  count every hot scan, while evidence-observation visits count indexed records
  actually loaded after radial rejection. Record sizes and maximum temporary
  peak-observation storage are reported explicitly.
- Version 23 moves peak signal into sparse evidence. Hot response records are
  12-byte float32 geometry records; sparse evidence records are 20 bytes and
  contain signal plus direction/gradient evidence. Every radial survivor still
  contributes to the denominator. Numerator and gradient terms are evaluated
  only for indexed evidence; nonzero numerator contribution order and all peak
  equations remain unchanged. Nonzero signal implies evidence, but valid
  positive-alignment evidence may contain zero signal.
- Version 29 removes the parallel dense peak-evidence index. Dense 12-byte
  response records are traversed only for the denominator; self-contained
  32-byte sparse evidence records carry transverse/axial geometry, signal, and
  gradient state for a second traversal. Denominator and evidence accumulator
  orders remain the original observation order. The obsolete evidence-index
  record-size profile field is removed.
- Peak response implementations may regroup equivalent accumulation work;
  exact floating-point identity and fixed accumulation order are not required.
  Deterministic repeatability, anchor axis/position distributions, populations,
  replay metrics, failures, and visual quality remain the acceptance gates.
- Direction-conditioned peak observations, transverse response math,
  persistent anchor state, accepted positions, diagnostics, and serialized
  output use float32. Peak ties and downstream DP counts need not be
  numerically identical, but retained populations and replay quality must be
  checked on the canonical workload.
- Stored-prediction samplers may retain their shared double-valued interface,
  but anchor extraction must normalize and range-check each returned sample
  once into a float32 tile. Reused tile halos, gradients, compact observations,
  and fitting must never retain the shared double sample. Float convergence,
  matrix, geometry, and response-comparison tolerances must be representable
  and effective at the magnitude where they are applied.
- Float32 anchor/fiberlet grids must have nonzero extents no larger than
  `2^24` on every axis, preserving exact integer voxel identity. Base-space
  serialization and OBJ output must fail instead of emitting coordinates that
  overflow float32.
- The existing double principal-axis API retains its historical names. The
  float32 API uses explicit `*F` names so brace-initialized double callers do
  not become ambiguous.
- Final refined-anchor support evaluation uses one float32 reduction for both
  compact production observations and direct public observations. All fields
  are consumed directly and public directions are normalized safely in
  float32. All finite-position sites contribute to the Gaussian
  denominator independently of direction/presence usability or robust
  membership. Aligned support, directional coherence, and combined objective
  remain float32 in persistent anchor state. Exact numeric identity is not
  required; deterministic support
  classification and canonical replay quality are the contract.
- The bounded direction-conditioned peak grid uses a checked row-major layout.
  Physical grid points and feasibility are constructed once in canonical
  first/second-index order; computed response values use direct cache slots
  with a separate computed-state byte. Hill-climb traversal, lexicographic
  tie-breaking, response arithmetic, separate float response coordinates, the
  no-feasible-slot center fallback, and uncached subpixel acceptance semantics
  remain unchanged. Existing profile request, computed-grid, and acceptance
  response counters retain their definitions.
- The hot replay-tube filter snapshots authoritative clipped source segments in
  prediction coordinates as float32 and uses a packed Boost.Geometry R-tree of
  radius-expanded segment AABBs. A point is inside when any candidate's
  continuous float32 point-to-segment distance is within the float32 radius.
  Candidate order and the linear projection function's `1e-12` tie behavior are
  intentionally irrelevant to this boolean geometric-union predicate. The
  snapshot owns its tree and segment storage, is immutable after construction,
  and supports concurrent const queries. Public replay-tube distance methods
  retain the existing double-precision linear projection for diagnostics.
- Fiberlet performance changes are measurement-first. Exact legacy numeric
  identity is not required for the robust anchor fitter or the float32
  replay-tube/local-corridor filters. Robust fitting is accepted through
  deterministic repeatability, matched anchor axis/position distributions,
  anchor/fiberlet/graph populations, downstream replay metrics, and visual
  inspection. Float32 containment comparisons still quantify boundary changes.
  A comparison records
  the command, input identities, commit, build
  type and flags, host, thread count, decoded/disk/OS cache state, warmup policy,
  repeated-run distribution, and exact output hashes or bytes in addition to
  aggregate correctness metrics.

# Anchor pipeline stage diagnostics

- Anchor extraction assigns stable per-cell diagnostic IDs to the two fitted
  attempts before robust component removal, compaction, or component sorting.
  Both attempts are emitted for every selected cell; unavailable or robustly
  degenerate directions use null geometry. No merge successor is created.
- Five strict `vc_fiberlet_anchor_stage` version-1 JSON files capture the real
  boundaries `initialized`, `refined`, `support`, `selection`, and `nms`.
  External NMS-context cells may be referenced as suppressors but never enter
  the selected stage populations.
- Per-record transitions distinguish empty/degenerate initialization or robust
  direction removal, refined empty or below-support rejection,
  outside-selection rejection, and
  NMS suppression. Threshold decisions store their tested value and threshold;
  NMS rejection stores the actual higher-ranked suppressor used by the existing
  pass, its ranking fields, and whether it is external context.
- Stage files emitted by one extraction bind the same source hash, grid/scale,
  complete selected-cell set, extraction parameters, and glyph length. The
  napari visualization reader treats extraction parameters as opaque optional
  producer metadata: parameter additions, removals, value changes, or absence
  cannot reject otherwise compatible geometry. Cross-stage visualization
  checks bind only the source, coordinates, selection, and glyph length. They
  still validate canonical identities/order, merge lineage, survivor subsets,
  unchanged geometry/metrics across filter-only stages, and equality between
  final NMS geometry and `anchors.obj`.
- Stage capture copies existing computations and must not change fitting,
  filtering, NMS ranking, final `anchors.json`/OBJ bytes, or fiberlet paths.
  `anchors.json` remains the only authoritative path-stage input.
- Napari displays all five anchor diagnostic stages by default, with one Shapes
  layer per nonempty stage using stable colors, transition/metric features,
  common clipping, and width controls. Only NMS is initially visible while the
  duplicate final-anchor OBJ layer starts hidden. `--no-anchor-stages` is the
  explicit fast-start opt-out and displays only final centre/direction glyphs.
- Failed-replay viewing provides an independent `Anchor radius` in base voxels.
  Each final/stage glyph uses its symmetric center, each cell-center point uses
  itself, and each refinement offset uses its anchor target. Exact Euclidean
  point-to-segment distance is measured against the union of the reference
  fiber and complete failed trace; a glyph remains rendered when its distance
  is at most the radius. Anchor radius defaults to 32 base voxels independently
  of the extraction-tube radius. The presence EDT display mask is independently
  controlled and also defaults to 32 base voxels.
- Anchor distance filtering physically subsets each final/stage glyph Shapes
  layer, the cell-center Points layer, and the refinement-offset Shapes layer.
  Out-of-radius geometry must be absent from the displayed layer rather than
  transparent or merely recolored, because hidden geometry must not participate
  in depth rendering. The controller retains defensive full-population copies
  of geometry, every aligned stage-feature column, exact distances, and source
  order, then reconstructs the visible subset on each radius change. Filtering
  preserves layer identity/order, widths/sizes, colors, clipping, layer
  visibility, and full artifact counts in layer names; item selections are
  cleared when displayed indices change. It does not affect the independently
  controlled fiberlet-path or presence-EDT filters.
- Failed-replay viewing provides an independent `Fiberlet radius`, defaulting
  to 16 base voxels. Each fiberlet's distance is the exact minimum 3D
  segment-to-segment distance between its complete rendered polyline and the
  union of the reference fiber and failed trace. Degenerate segments are valid.
  A path is retained inclusively when that distance is at most the radius, so
  only fiberlets completely outside the threshold are physically absent from
  the Shapes layer. The controller retains full geometry and aligned quality
  features so widening restores source order, coloring, and widths.
- Failed-replay mode creates stable typed layers for final anchors, cell
  centers, refinement offsets, and fiberlets even when an artifact population
  is empty, plus all five anchor stages unless diagnostic loading was explicitly
  disabled. The
  `Reload artifacts` command always rereads the original direct visualization
  manifest, follows its newly published immutable hashed generation, and passes
  it through the startup strict loaders. Version-1 inputs reread their original
  single-visualization path.
- In-process artifact reload requires the same failed-replay artifact contract,
  fiber-prediction manifest content hash, prediction shape/scale, displayed
  Zarr level transform, base crop, extraction-tube radius, and the same optional
  stage-layer topology.
  Counts may cross zero; geometry, metrics, reference/trace/failure data, and
  generation paths may change. An incompatible replacement leaves the current
  display unchanged and reports that restart is required.
- Reload never resolves or opens the presence Zarr and retains the exact lazy
  source crop object. It recomputes only the derived reference/trace EDT, exact
  anchor distances, and exact fiberlet-polyline distances; creates a new lazy
  mask graph over that same source; and reapplies the current independent
  presence, anchor, and fiberlet radius values.
- Reload preparation completes strict parsing, compatibility, features, names,
  and distance calculations before layer mutation. Commit updates existing
  layers under blocked events, clears stale item selections, and preserves
  layer identity/order, visibility, clipping, widths/sizes, path colormap,
  volume rendering, crop controls, and radius controls. A commit error rolls all
  artifact layers, full-population filter sources, and derived controller state
  back before it is reported. Reload reapplies the current anchor and fiberlet
  radii to their replacement full sources, including transitions between empty
  and nonempty displayed populations; later widening uses replacement full
  sources rather than subsets visible during reload.

# Fiberlet storage quantization experiment

- `vc_fiberlets quantization-benchmark` runs one float baseline and one selected
  quantized scenario through persistent, fingerprinted on-demand caches. It
  never materializes the complete corridor graph.
  The default scenario is `combined_q4_axis_cost_u8`; the command requires an
  output directory so interrupted and warm runs reuse complete chunks.
  The standard matrix has 19 entries: one baseline and 18 non-baseline rows in
  deterministic order. The unpublished `combined_q1_axis_cost_u8` row is
  replaced by `position_q1_8_compact_axis_cost_sqrt_u16_max256`; no alias is
  retained.
- One canonical float anchor dataset belongs to each exact source, extraction,
  corridor, grid, and chunk-layout identity. Baseline and every geometry or cost
  scenario reuse that dataset. Quantization never changes serialized anchors or
  reruns anchor extraction.
- Evaluation position quantum is a finite positive spacing in base voxels;
  zero selects exact float positions. Fractional values are permitted and must
  survive scenario state, cache fingerprints, JSON, and CLI output without
  integer truncation. Chunk side divided by the quantum must be an integral,
  representable bin count. This evaluation type does not change the unrelated
  compact physical storage header's integer quantum.
- Quantized endpoints are globally nearest-rounded in base coordinates before
  conversion back to prediction coordinates and DP. Positive half steps round
  upward. The regular DP
  constructs the resulting Hermite domain and chooses a new interior route; a
  baseline `(u,v)` route must not be transplanted onto changed endpoint planes.
- Stable anchor IDs remain source cell coordinate plus component `0/1`.
  Rounded positions are geometry, not identity: anchors from adjacent cells may
  legitimately round to the same base coordinate and must not be merged.
- Duplicate/unresolvable source-cell keys, out-of-volume endpoints, or
  unsupported local-position/delta widths invalidate the complete scenario.
  Ordinary candidate rejection or no-path after otherwise valid quantization is
  a measured result and remains visible in the scenario graph population.
- Raw-total cost offset and scale are float32 per first-endpoint spatial chunk. Encoding
  nearest-rounds onto the complete unsigned range with an exact-maximum case;
  decoding evaluates float32 `offset + scale * code`. Raw-total scenarios use
  the decoded scalar as the authoritative edge total. Fixed nonlinear density
  scenario instead uses the fixed constant `Cmax=256` and encodes
  `round(65535 * sqrt(clamp((total / path_length) / 256, 0, 1)))`. Decoding uses
  `256 * (code / 65535)^2 * path_length`. No chunk contents or observed values
  influence this mapping. The same geometry's stored positive float32 path
  length is used throughout. Entering join costs remain unquantized float
  components and are added exactly once.
- Each geometry scenario derives a chunk-local endpoint view from canonical
  anchors and runs fresh candidate generation, Hermite construction, dense
  sampling, and DP. Compact direction changes only the fitted axis and retains
  canonical scoring. Position quantization also resamples prediction direction,
  presence, validity, and Lasagna normal fields at every rounded endpoint.
  It reuses no float candidate, interior route, endpoint step, path length, or
  cost. Source cell/component IDs address evaluation-cache data and define
  canonical graph ordering; quantized positions affect geometry only.
- Derived anchor chunks use single-flight construction and a bounded LRU. The
  same derived chunk view supplies candidate DP, replay seeds, individual
  endpoint lookup, arcs, route reconstruction, transitions, and compact-cost
  ownership; no consumer may mix canonical and transformed endpoint geometry.
- Reports include explicit scalar settings, logical-key collisions, failure
  counts, completed fractions, bounded decoded residency, wall/CPU time,
  symmetric Euclidean/normal/tangential line-distance distributions, and both
  replay-to-reference distributions. Anchor/candidate and point-index identity
  are not tracing-quality metrics.
- Batch replay teardown cancels and drains only its own speculative cache task
  groups. Fiberlet work drains before anchor work because an active fiberlet
  producer may synchronously request anchor dependencies. Issued persistent
  writes finish before the replay context is released, and each completed
  machine-readable result row is flushed immediately.
- Geometry-cache identity separates generated fiberlets from the replay cost
  view. Its generation settings contain only endpoint position quantum and
  fitted-direction encoding. Float, raw `uint8`, raw `uint16`, and fixed-sqrt
  `uint16` cost view for one
  geometry share the same fiberlet prefixes, routes, path lengths, endpoint
  steps, and float component costs. All of them share the one canonical anchor
  cache; cost-only views also reuse baseline fiberlets.
- `compact_axis`, `compact_axis_cost_u8`, `compact_axis_cost_u16`, and
  `compact_axis_cost_sqrt_u16_max256` all use float positions and the
  identical compact-direction geometry cache. Their selected graph costs are
  respectively float, raw-total `uint8`, raw-total `uint16`, and fixed-sqrt
  density `uint16`; the
  geometry namespace's opaque historical u8 compatibility tag does not select
  graph cost precision and never changes persisted prefixes or routes.
- Cache-backed fiberlet replay defaults to
  `compact_axis_cost_sqrt_u16_max256`: exact float positions, compact fitted
  directions, fixed sqrt-density `uint16` edge costs with ceiling 256, float
  path lengths, and float join costs. The same profile must select both the
  compact-direction fiberlet cache namespace and the graph cost view. Canonical
  anchors remain exact float. The all-float correctness oracle is selected
  explicitly and cannot inherit this production default.
- Replay bundles record the selected evaluation profile. Eager graph replay is
  the explicit exact-float oracle and must identify itself as such; it is not a
  compact-cache parity mode.
- This default does not change the unpublished `CompactQuantized` payload
  schema. On-demand evaluation cache payloads remain `Float32Cache`; the
  accepted profile is the baseline for subsequent compact resident/persistent
  representation work.
- Compact costs use one affine range per first-endpoint storage chunk. Building
  that stable range may complete missing on-demand chunks in the shared
  geometry cache, but it must not create a cost-specific cache namespace or
  rewrite existing geometry payloads.
- Fixed-sqrt density cost has no spatial owner or adaptive range. Its domain and
  ceiling are replay-view state only: they are absent from geometry
  fingerprints, cache roots, dataset metadata, and persisted payloads. A warm
  fixed-sqrt replay reads the existing float total and path length and performs
  no range scan, anchor extraction, DP/fiberlet generation, or payload rewrite.
- For compatibility with already completed experiments, every non-float
  geometry cache uses the historical internal `cost_bits=8` namespace tag.
  This tag is opaque cache identity data and cannot reach preprocessing. The
  replay graph alone receives the selected cost precision. Cache chunk side is
  still shared physical layout metadata and remains part of cache identity.
- `--scenario NAME` selects one exact matrix scenario plus the baseline. The
  focused `combined_q4_axis_cost_u8` scenario means a 4-base-voxel endpoint
  position quantum, the existing compact two-byte fitted-direction encoding,
  and per-chunk 8-bit total cost. An unknown scenario name is an error.
- `position_q1_8_compact_axis_cost_sqrt_u16_max256` means a 0.125-base-voxel
  endpoint quantum, compact fitted directions, and fixed-sqrt density `uint16`
  cost with ceiling 256. At prediction-to-base scale 8 this is 0.015625
  prediction voxel, not 0.125 prediction voxel. All endpoint scoring fields are
  resampled at the rounded position, fresh fiberlet geometry and DP are built,
  and float join costs are added exactly once. It shares canonical float
  anchors but owns a distinct position-plus-direction fiberlet cache; all cost
  views with the same geometry reopen that cache.
- `--scenario all` runs the baseline once, then emits 18 comparison rows for
  every non-baseline standard scenario in deterministic matrix order. Each
  geometry group is generated at most once and later cost views reopen it.
- Maximum line distance treats replay restart segments as disconnected. It
  samples both directions at no more than one base voxel spacing and projects
  onto the other replay's actual segments. Normal/tangential components use the
  existing replay measurement with the Lasagna normal at the projected point.
- Baseline and scenario noise-floor summaries are directed from each replay to
  the annotated reference. Replay segments are sampled at no more than one base
  voxel spacing and projected onto actual reference segments. Euclidean,
  Lasagna-normal, and Lasagna-tangential summaries contain count, minimum, mean,
  median, and maximum in base voxels. Invalid reference normals are counted and
  excluded only from the component summaries. Long extraction, DP, line-to-line,
  and reference-distance stages report initial, periodic, and terminal progress.
- Every quantization comparison atomically writes its complete baseline and
  scenario `vc_fiberlet_graph_replay` JSON files and prints a deterministic
  failure-window row. The window begins at the owning segment's reset-search
  arc, extends through the complete failure-containing edge, and carries the
  segment's graph seed key. `--arc` and `--length` select that base-voxel
  interval; `--seed-key` forces the original first seed so prior consumed-node
  state cannot change the focused replay. Focused diagnostics retain the full
  corridor cache identity, disable prefetch, and request only chunks touched by
  seed selection and beam traversal. They persist every final ranked beam
  frontier with logical routes, geometry, decomposed edge/join costs, length,
  and density, plus a comparison artifact containing the first selected-route
  difference, cross-ranks, and maximum paired-route distance.
- Every full or focused comparison also records one lightweight cost entry for
  each actually committed fiberlet; lookahead alternatives are not counted.
  `route-cost-statistics-<scenario>.json` reports complete-route and
  failure-excluded min/mean/median/max/sum distributions for combined, edge,
  transition, and decomposed objective loss per prediction voxel. Every value
  in these distributions is the committed occurrence's component cost divided
  by its edge path length; raw per-fiberlet cost distributions are not exposed
  because replay does not compare them across unequal lengths. Aggregate raw
  loss and aggregate path length remain only to verify the length-weighted
  whole-route density. Failure exclusion uses
  the baseline replay's failure arcs for both baseline and scenario statistics
  so they describe the same reference neighborhoods. An occurrence is excluded
  exactly when its matched-reference interval intersects a closed
  `[failure_arc-margin, failure_arc+margin]` interval.
  `--route-stats-failure-margin` controls this distance in base voxels and
  defaults to 128. Repeated commitments of one logical fiberlet remain separate
  observations. Each observation owns the selected edge's stored component
  costs and the incoming join from the immediately preceding committed edge in
  the same segment; the first edge after every reset owns no transition.

## Whole-volume fiberlet preprocessing

- `vc_fiberlets preprocess-volume FIBER_MANIFEST OUTPUT_ZARR
  --normal-manifest NORMAL_MANIFEST` materializes an entire prediction volume
  without a reference-fiber corridor.
- The command has two durable outputs: a float intermediate anchor cache and a
  combined final Zarr containing anchors, edge prefixes, and routes. The anchor
  cache defaults beside the final output and is not deleted after completion.
- Sparse eligibility is based only on canonical stored presence chunks. Missing
  and decoded-all-zero chunks are inactive; every final output chunk that
  overlaps a decoded-nonzero presence chunk is active. Direction channels never
  participate in sparse eligibility.
- The sorted candidate output-chunk set is reconstructed from a fresh canonical
  presence scan on every preprocessing invocation and kept only in memory. It
  schedules work and resume checks; it is not part of the stored data contract.
  No active index, per-chunk marker, or whole-dataset completion marker is
  persisted.
- A present combined chunk is valid only when its anchor, prefix, and route
  files all exist, decode against their expected codecs/fingerprint, and have a
  matching prefix/route record count. A wholly absent tuple is an empty sparse
  chunk under ordinary Zarr semantics; corrupt, partial, or conflicting
  present tuples fail validation.
- Stored graph readers consume the combined dataset directly. They synthesize
  canonical empty anchor/prefix/route payloads for absent tuples and never
  require the original Fiber prediction manifest or a reconstructed expected
  set.
- Crop graph materialization may inspect the complete prefix/route owner halo,
  but it must restrict that population to fiberlets with at least one actual
  in-crop endpoint before requesting endpoint anchor chunks. Partial tuples
  needed by a retained fiberlet remain errors; unrelated partial tuples beyond
  the crop cannot invalidate an otherwise materializable crop.
- Preprocessing does not publish intermediate anchor chunks with no retained
  anchors or final combined tuples with no generated Fiberlets. Missing and
  decoded-all-zero input presence chunks are never scheduled in the first
  place. Canonical decoded empty results remain available to the current
  process, while both durable Zarr datasets stay sparse.
- Intermediate anchors include the exact dependency halo required by each
  active fiberlet owner. Halo-only chunks remain in the intermediate cache and
  are absent from the final active set.
- Every required intermediate anchor is checked directly: a present payload
  must decode against its expected codec/fingerprint, a missing payload is
  generated, and a conflicting payload fails. Extra cached halo chunks remain
  reusable.
- Final payloads use float positions, compact directions, and fixed sqrt-density
  `uint16` costs with ceiling 256. The intermediate cache remains float.
- Resume rescans source presence, anchor dependencies, and final tuples. One
  global `--threads` chunk-worker budget dispatches all ready current-Z
  fiberlets before filling remaining capacity with anchor dependencies ordered
  by their earliest dependent output. Anchor work may look ahead; final-output
  dispatch cannot cross the current Z slab. Each whole-volume chunk extraction
  is single-threaded so nested teams cannot exceed that global budget.
- The pipeline exposes one live progress line refreshed about once per second,
  with a persistent newline every minute and at completion. It reports the Z
  frontier plus separate anchor/output completed counts, rates, ETAs, and
  current/projected compressed payload bytes. The projection is the visited
  expected-chunk mean and excludes Zarr metadata.
- Every payload file uses same-directory temporary publication, file `fsync`,
  atomic rename, and parent-directory `fsync`. The final three-file tuple is a
  logical completeness unit, not one filesystem transaction. Preprocessing
  exclusively locks both roots, removes exact abandoned atomic-write temporary
  files before resume and after workers stop, and removes legacy activity and
  completion bookkeeping.
- `--presence-floor` is an inclusive owned-observation gate. Cells with no
  usable owned observation at or above it return before seed generation and
  refinement. `--minimum-support` remains a post-fit threshold.

## Managed Fiberlet preprocessing and publication

- `las_manager fiberlet run FIBER_RUN NORMAL_RUN` launches native whole-volume
  processing through the shared managed-job runner. Both inputs must be
  completed, uncropped runs for the same volume and base coordinate frame;
  roles and required products are strict.
- `fiberlet_threads` is a positive global config value (portable default 32)
  passed as `--threads`. Per-run `--threads` may override it. The manager owns
  all input/output/context/cache arguments.
- The durable float anchor cache is stored below the run's private `cache/`.
  Only `artifacts/fiberlets.json` and the final combined
  `artifacts/fiberlets.zarr` form the portable bundle.
- Fiberlet metadata schema version 2 is self-describing and path-independent.
  Algorithm identity contains all scientific processing/coordinate/layout/
  storage/codec settings; dataset identity adds stable source/model/level/run
  identities and manifest hashes. Runtime paths and executable/host audit data
  do not participate.
- Fiberlet upload uses the common marker/lifecycle path but streams its sparse
  tree through rclone under `fiberlets/<run_uuid>` without materializing a full
  file list.
- Atlas stores Fiberlets as a `fiberlets` volume data entry depending on
  existing Fiber and normal Lasagna model/level entries. Publication is copy
  only: one canonical copy and one CC-licensed public copy under the volume's
  `representations/fiberlets/` directory; it creates no model and no derive
  action.

## Anchor-seeded Fiberlet crop tracing

- `vc_fiber_trace_chunk` consumes one combined Fiberlet Zarr and a structurally
  compatible regular Lasagna normal manifest. Manifest path and byte identity
  are provenance only. Compatibility requires the Fiberlet whole-volume base
  frame, ceil-downsampled base shape, consistent prediction-to-base scale,
  valid `nx`, `ny`, and `grad_mag` channels, matching normal-component shape
  and base spacing, and base-shape coverage with at most the established
  one-chunk array padding. It does not require or read the original dense Fiber
  prediction manifest. Missing sparse tuples are empty; partially present
  tuples are invalid.
- The public crop is a half-open base-volume XYZ box. Every stored anchor
  variant in intersecting cells is a deterministic seed candidate, ordered by
  descending prediction presence and then storage key.
- Each uncovered seed traces both signs of its fitted axial direction. The
  initial pair must use distinct physical Fiberlets and pass the ordinary join
  constraint at the seed. Later joins use the stored Fiberlet transition
  metric, cycles are rejected, and the first crop exit clips and terminates a
  side.
- Seed attempts may be capped independently from accepted output fibers.
  Covered seeds do not consume attempts; selecting an active seed does,
  including when it has no usable initial edge. Zero means unlimited for both
  limits, and neither limit changes descending-presence/storage-key ordering.
- Before crop tracing, the stored chunk graph is bulk-materialized into an
  immutable replay graph over an internal search box. That box expands every
  face of the requested crop by exactly the configured base-voxel lookahead
  distance. The preparation loads anchors, physical Fiberlets, complete route
  and cost-profile data, outside endpoints needed by crossing routes, and
  ordinary stored join transitions for that expanded box. One lookahead
  distance is sufficient: every physical Fiberlet incident to an anchor in the
  search box is retained in full, including a final horizon-crossing edge whose
  endpoint lies outside the box. No additional maximum-Fiberlet-length padding
  is used. Chunk requests are batched through the existing cache workers, and
  each retained route is reconstructed once.
- Canonical starting seeds remain only the anchors in the requested half-open
  crop. Speculative lookahead clips and terminates at the expanded search-box
  boundary; committed trace geometry still clips and terminates at the
  requested crop boundary. The requested crop remains authoritative for seed
  ownership, anisotropic coverage, stored artifact bounds, CT faces, and all
  downstream output. The halo may change a near-boundary route choice by
  exposing its full lookahead evidence, but it must not create halo starts or
  emit halo geometry.
- Seed workers query only the immutable materialized graph. They perform no
  cache I/O, waits, or shared graph mutation. Crop tracing assigns dense
  strongest-first tickets continuously and commits completed candidates only
  at the ordered ticket frontier. The submitted-but-uncommitted window is
  bounded to the worker count plus one eighth (at least one) as queue headroom.
  Integration rechecks seed activity; a speculative result whose seed was
  covered by an earlier accepted line is discarded and consumes no attempt.
  Attempt/fiber limits and errors take effect only when their canonical ticket
  reaches the frontier, so later speculative results and failures are ignored
  after the equivalent serial stop point. Counters, output order, geometry,
  and the first propagated error are identical to one-thread execution.
- The immutable graph stores sorted contiguous anchors, physical edges, joins,
  and flat directed adjacency. Crop traversal consumes forward/reverse views of
  adjacency and route geometry and materializes clipped points only for the
  selected edge. Immutable views borrow stable storage without ownership
  traffic. Other graph sources may return one shared owner per complete query
  result; compact cached routes use one reconstructed owned buffer rather than
  reference-counting individual elements or retaining leases in search state.
- Crop lookahead keeps the committed trace's visited-anchor set immutable and
  represents only rollout-local ancestry as parent-linked route nodes. Cycle
  rejection checks both sets and includes the current rollout node. Complete
  arc sequences are materialized only at the existing intermediate
  lexicographic-pruning boundary. Terminal candidates retain route-node
  indices; the selected minimum uses the same cost-density and full
  lexicographic route ordering as the former sorted completion vectors. This
  representation must not change arithmetic, generated-state cutoff behavior,
  traversal order, or selected Fiberlets.
- `--threads` defaults to the host CPU count and controls both batched graph
  preparation and immutable seed tracing. There is no separate low trace-worker
  cap.
- Accepted lines suppress only later anchor seeds. Coverage uses the shared
  replay threshold measurement with a default 20-base-voxel normal radius and
  four-times-wider Lasagna tangent-plane radius. The threshold comparison is
  strict, and invalid normals use the existing Euclidean fallback.
- Coverage additionally requires inclusive unoriented fitted-axis agreement
  with the projected line tangent within 25 degrees. Crossing directions are
  retained. This version performs no line-to-line Fiber deduplication.
- The authoritative crop output is a sparse Fiberlet Zarr dataset of kind
  `traces`, not an OBJ. It uses a trace-only `float64_traces` profile and
  crop-local base-coordinate chunks aligned to the source Fiberlet spatial
  chunk side. Each accepted line is owned by its seed chunk and stores its
  deterministic result ordinal, seed base position and presence, total metric
  cost, prediction-space traced length, and exact float64 base-XYZ polyline.
  Complete crop traces are not encoded as short transverse-lattice Fiberlets.
- Stored total metric cost includes every committed selected edge and internal
  join once. A crop-clipped edge's cost and prediction length use its retained
  fraction. A bidirectional line includes the central join once when that
  transition exists; the existing independent-side fallback has no central
  transition to add. Speculative lookahead cost is never stored.
- Trace publication writes and strictly reopens a unique sibling temporary
  root before atomically renaming it to a previously absent output path.
  Metadata inventories every populated chunk and the global record count.
  Loading rejects missing or unexpected chunks, wrong seed ownership,
  malformed records, and duplicate or incomplete ordinals.
- Line visualization is derived only from a reopened trace dataset. `trace`
  mode writes the dataset and may also write line OBJ artifacts; `visualize`
  mode regenerates them without source Fiberlets, normals, or CT data.
  Optional crop faces remain trace-time artifacts and use one concrete uint8
  CT OME-Zarr group with the existing VC3D fine-to-coarse coordinate renderer.
- After tracing, every nonzero consecutive accepted-line step contributes its
  normalized unoriented axis and base-voxel length to a deterministic
  two-direction fit. The axial PCA tensor is initialization only. The fitted
  directions independently maximize
  `sum(length*max((step dot d1)^2,(step dot d2)^2))` and are not constrained to
  be orthogonal. Seed pairs, equal-alignment assignment, fit ties, axis signs,
  and final direction labels have fixed canonical ordering.
- For line grouping, let `q=(d1 dot d2)^2`. Each unit local step `u` contributes
  `step_length*clamp(((u dot di)^2-q)/(1-q),0,1)` support to direction `i`.
  These are independent affinities and must not be normalized by their sum;
  with non-orthogonal axes their sum can exceed the step length. Axis dots are
  clamped before squaring. When `1-q <= 1e-8`, direction 1 instead receives
  ordinary squared-alignment support and direction 2 receives zero.
- A line is direction-1- or direction-2-dominant when the larger accumulated
  support divided by the line's actual valid arc length reaches the selected
  dominance fraction; an exact support tie prefers direction 1. Otherwise it
  is mixed. Hard fit assignments remain only for the existing deterministic
  axis ordering and do not classify lines. The default is 0.75.
  `--direction-dominance F` is accepted only by `trace`, `visualize`,
  `direction-diagnostic`, and `direction-ablation`, must be finite in
  `(0.5,1]`, and is inclusive at `F`; constraint and consensus modes reject it.
  A line without a nonzero step
  is mixed. Only accepted lines
  contribute, and analysis uses a fixed serial reduction order. The selected
  fraction is reported with the classification.
- Direction dominance changes only the direction-group line and anchor
  partition. It cannot change tracing, coverage, stored trace data, fitted
  axes, per-direction supports, or quality classification. `visualize` rewrites
  the complete all/direction/anchor/quality family for its output basename.
- The requested line OBJ remains the complete accepted set. Sibling `_dir1`,
  `_dir2`, and `_mixed` OBJs partition those polylines without changing names,
  geometry, or order. Matching `_anchors`, `_dir1_anchors`, `_dir2_anchors`,
  and `_mixed_anchors` OBJs contain one point primitive at each trace's actual
  seed anchor. Empty groups still produce valid OBJ artifacts.
- Trace quality is total metric cost divided by prediction-space traced length;
  lower is better. Visualization stably orders by quality then stored ordinal
  and partitions rank `r` among `N` with `min(9,floor(10*r/N))`. Ten sibling
  `_quality_00_10` through `_quality_90_100` OBJs partition every trace exactly
  once. A `_quality_histogram.csv` and console table report count and
  min/mean/max total cost and cost density per bin; empty bins retain valid
  empty OBJ files and blank numeric CSV fields.
- Stored-artifact consumers accept `--quality-fraction F` for finite
  `0 < F <= 1`. Rank by the exact visualization quality and stored ordinal,
  retain `ceil(F*N)` entries (at least one for nonempty input), then restore
  stored-ordinal order before visualization, direction fitting, splitting,
  constraints, consensus, or BP. The full artifact is still read and strictly
  validated; the option reduces downstream work, not artifact I/O. Crop bounds
  and metadata remain unchanged, and diagnostic original-trace IDs must compose
  through the retained ordinal map. Report input/retained counts, effective
  fraction, and the worst retained density. The cutoff is diagnostic because
  ordinal tie-breaking may split equal-density traces. Trace generation rejects
  this artifact-input option.

## Stored crop-trace H/V constraint diagnostics

- `vc_fiber_trace_chunk constraints TRACE.zarr --normal-manifest MANIFEST`
  operates on durable `float64_traces` crop output and emits statistics plus
  diagnostic connector OBJs, solves five-state piece labels, and emits one OBJ
  for each label. Structured constraint persistence remains outside this
  stage.
- `vc_fiber_trace_chunk direction-diagnostic TRACE.zarr --normal-manifest
  MANIFEST` first applies the same gradual direction grouping as stored-trace
  visualization to every source trace. It writes the unfiltered direction
  family beneath `<base>_initial.obj`, then removes all mixed traces before
  piece splitting, spatial constraint extraction, optional strength pruning,
  and optimization. Retained traces preserve an explicit mapping to their
  original stored-trace indices.
- Direction diagnostics always use the ordinary discrete H/V-plus-broken
  HiGHS MILP with parity disabled. They reject explicit `--hv-only`, LP
  relaxation/backend controls, exact-perpendicular MILP, and labeling-only
  link exclusion. Extraction settings, finite winding cutoff or
  `--no-winding-cutoff`, `--constraints-per-fiber`, broken cost, MIP gap,
  threads, and cache size retain their canonical meanings.
- Optimized H/V labels are compared to initial direction groups only after
  resolving the arbitrary H/V gauge independently in every connected
  component of active pieces. Components use the exact post-pruning constraint
  graph passed to the solver, including hard same-trace continuity, with edges
  incident to a final Broken piece omitted. Each component is flipped only
  when that strictly reduces disagreement; a tie keeps direction 1 mapped to
  H. Broken pieces are errors independently of orientation.
- The diagnostic prints raw H/V/broken totals, the number of active and flipped
  components, a gauge-aligned confusion table, orientation and broken errors,
  piece and represented-source-trace error rates, and a stable per-error table
  carrying filtered and original trace IDs plus the trace-local arc interval.
  The piece denominator is every retained extracted piece. The trace
  denominator is every retained source trace represented by at least one
  piece. An all-mixed input succeeds and writes valid empty constraint and
  label families in addition to the complete initial direction family.
- `vc_fiber_trace_chunk direction-ablation TRACE.zarr --normal-manifest
  MANIFEST` classifies once, preserves above-threshold direction-1/direction-2
  traces as a trusted cohort, and ranks Mixed traces by descending
  `max(direction_support)/valid_arc_length` followed by original trace index.
  Confidence controls admission order only. Every Mixed trace is a Defect
  reference and is expected to optimize to Broken; it never receives a
  tentative H/V reference. Zero-valid-length traces rank last.
- Ablation checkpoint zero contains only trusted traces. Each later checkpoint
  cumulatively admits `--ablation-step N` ranked Mixed traces, default 5; the
  optional `--ablation-limit N` restricts admission to the first N ranked
  Mixed traces for repeatable parameter sweeps, while omission admits all; the
  final remainder is always included. Ranking selects membership only:
  the retained input vector is reconstructed in original stored-trace order at
  every checkpoint. Pieces, constraints, configured pruning, and the ordinary
  discrete H/V-only-plus-broken MILP are recomputed independently without a
  solution warm start. Solver status and gap remain explicit.
- Every checkpoint also solves the matching H/V-only LP relaxation. Continuous
  activity and H/V values are independently thresholded at 0.5; inactive maps
  to Broken and active maps to H or V. MILP and thresholded-LP errors and solve
  times are reported separately, and their final label OBJ basenames remain
  distinct.
- Ablation gauge selection uses only trusted active pieces in each exact
  post-pruning active constraint component. A strict reduction in trusted
  disagreement is required to flip; a tie or component without a trusted
  active piece keeps direction 1 mapped to H. That one gauge evaluates trusted,
  admitted, and combined cohorts. A trusted H/V mismatch is an orientation
  error, a trusted Broken piece is a broken error, and an active admitted
  Defect piece is a defect-active error; Broken is correct for a Defect.
  Each cohort reports piece errors and their union per represented source trace
  with its own denominators. Raw H/V/Broken totals remain global.
- Ablation writes the complete `<base>_initial` direction family once and the
  existing constraint and label OBJ families only for the final selected
  checkpoint. Intermediate checkpoints emit statistics only. The default base
  is sibling `<trace-stem>_direction_ablation`. With no Mixed traces,
  checkpoint zero is final; all-Mixed checkpoint zero is a valid empty solve.
- `--perpendicular-only` is an opt-in labeling filter accepted by constraints,
  direction diagnostic, and direction ablation. It retains measured links only
  when `perpendicular_score > 0.5`; the exact ambiguous boundary is excluded.
  Hard same-trace continuity links remain unchanged as their existing strong
  evidence but do not become equality constraints. Filtering precedes degree,
  adjacency, gauge, triangle, row, and objective construction and applies
  identically to MILP and LP. The extracted constraint OBJ family remains
  complete, while solver reports include retained and excluded counts. The
  option conflicts with the redundant parallel-separate-winding exclusion.
- Direction ablation optionally accepts `--post-iterations N` with
  `--post-influence I`, where `N>0` requires `--perpendicular-only` and
  finite `I` lies in `(0,1]`. The final selected checkpoint must contain
  exactly one MILP piece for each represented trace, with unique contiguous
  trace indices; split, missing, duplicate, and non-represented traces are not
  repaired or synthesized. H initializes to 1, V to 0, and Broken to 0.5.
- Post-filter adjacency is the unique source-trace graph induced by the exact
  retained solver-link indices. Hard same-source links are ignored and every
  cross-source link must have `perpendicular_score > 0.5`; a neighbor therefore
  contributes `1-v`. Its confidence weight is
  `clamp((abs(v-0.5)-0.5*(1-I))/(0.5*I),0,1)`. Each iteration synchronously
  replaces a value by the weighted neighbor mean; zero total weight preserves
  the previous value. This never changes MILP labels, costs, or errors.
- Final values own ten complete-trace OBJ layers `<base>_p0.obj` through
  `<base>_p9.obj`. Bands are fixed value intervals `[0,0.1)`, ..., `[0.8,0.9)`,
  `[0.9,1]`; exact internal boundaries use the higher band and equal values
  remain together. Only represented checkpoint traces occur, every trace
  occurs exactly once, and empty bands are valid files. The accompanying table
  stratifies the existing gauge-aligned comparison errors into trusted H,
  trusted V, and admitted Mixed cohorts; it defines no new error threshold.
- Every distance and arc value is in base voxels. Defaults are a 32-vx common
  sample pitch, 512-vx target piece length, 128-vx overlap, 128-vx neighbor
  radius, 32-vx centered tangent window, and 8-vx winding integration step.
- A nondegenerate trace is divided into the minimum number of equal overlapping
  arc intervals whose span is no greater than the target. Each interval stores
  exact endpoints in addition to regular samples. Wholly degenerate traces are
  skipped and counted; zero-length internal steps do not contribute arclength.
- A point R-tree supplies a cubic broad phase for distinct-trace piece pairs.
  Candidates must also pass the exact Euclidean radius. Only the minimum pair
  of sampled points for each unordered piece pair is measured, with stable
  sample-order tie breaking. Same-trace pieces never enter this search.
- Consecutive pieces of one trace receive a hard link at their overlap midpoint
  with parallel score 1, perpendicular score 0, winding distance 0, and closest
  distance 0. Nonconsecutive pieces receive no implicit same-trace link.
- A measured pair is oriented by the signed dot of centered secants at its
  closest samples. The closest sampled pair remains the unchanged walk seed,
  reported closest connector, and first parallel-winding sample. The default
  `distance` correspondence walks both pieces forward and backward at the
  common pitch and retains a strictly closer anti-correlated phase shift of one
  twentieth pitch, bounded to that magnitude. The optional
  `perpendicular-grid` correspondence instead walks incrementally from the last
  accepted pair. At each step, an independent deterministic 2D grid varies each
  advance in increments of one twentieth pitch, bounded to one quarter pitch.
  Grid candidates minimize configurable nonnegative weights of both squared
  target-step-normalized advance residuals and the squared dot of the unit
  connector with each centered tangent. Optional nonnegative terms penalize
  connector-direction change and squared target-step-normalized connector-
  length change from the previous accepted pair; both default to zero.
  Zero-length connectors are invalid; absolute connector length is not part of
  the objective. Ties
  prefer smaller step residual, smaller total absolute offset, then
  lexicographic offsets. The grid limit must remain below one pitch so every
  accepted step advances. The raw parallel score is the clamped `[0,1]` mean
  of consistently oriented tangent dots. Raw perpendicular score is
  `1 - abs(initial tangent dot)`. Division by their sum produces complementary
  normalized scores.
- Grid step, range, and all four objective weights are explicit CLI controls.
  The distance default ignores them. Each measured link records diagnostic-only
  sample count, advance residual, connector/tangent residual, connector-length
  change, connector-direction change, and limit-hit fraction; winding-factor
  CSV output exposes those values when explicitly enabled. Collection defaults
  off and does not change BP inputs.
- Normal-aligned winding uses the existing Lasagna connector integral and
  trapezoidal sampling. Each endpoint winding-density sample is multiplied by
  `abs(dot(unit connector, decoded unit normal))`. Missing required normal or
  density samples reject the measured link. The ordinary winding API retains
  its previous unmodulated behavior.
- A finite measured aligned winding distance must be strictly less than `1.5`.
  Values greater than or equal to `1.5` are discarded and counted separately
  from invalid/non-finite winding samples. Hard same-trace continuity links
  remain at zero and do not pass through measured-link rejection.
  `--winding-cutoff N` selects another finite positive exclusive cutoff.
- `--no-winding-cutoff` disables only that finite `1.5` rejection for
  H/V-only constraint diagnostics. Every finite winding measurement is
  retained; invalid and non-finite measurements remain rejected. The flag
  requires `--hv-only`; joint parity labeling retains the finite `<1.5`
  invariant. Default cutoff behavior is unchanged.
- The normal dataset is opened in base-coordinate working space, must have the
  valid 3D `nx`, `ny`, and `grad_mag` structure used by Fiberlets, and its
  declared base shape must cover the complete trace crop. Manifest path, bytes,
  and parsed identity are not compatibility gates; parsed equality with stored
  provenance is diagnostic only.
- Candidate scoring may run in parallel, defaulting to host CPU count, but uses
  deterministic candidate slots and stable final ordering. The report includes
  configuration, population and rejection counts, phase timing, and deciles
  for closest distance, both normalized orientation scores, and aligned
  winding distance.
- Accepted measured connectors use bounded float-coordinate grouped corner
  sampling for aligned winding. Each batch lays out and fetches `grad_mag`,
  `nx`, and `ny` dependencies collectively, materializes samples in parallel,
  and integrates connectors independently; it performs no per-point channel
  cache queries. A distinct compatible density grid is sampled separately from
  the jointly sampled `nx`/`ny` grid. Orientation, winding, and total score
  timing are reported separately.
- Constraints mode writes four connector-line OBJ diagnostics. `--output PATH`
  is a basename whose final extension is removed; when omitted for `TRACE.zarr`,
  the basename is sibling `TRACE_constraints`. Literal suffixes are
  `_perpendicular_same_winding.obj`,
  `_perpendicular_separate_winding.obj`, `_parallel_same_winding.obj`, and
  `_parallel_separate_winding.obj`.
- An OBJ connector joins the measured closest points and is named
  `constraint_piece_A_B` from ascending global piece IDs. Hard continuity
  links are excluded. Perpendicular requires score `>0.5`; winding `<0.5`
  selects same-winding and winding `>=0.5` selects separate-winding. Parallel
  uses the same split after requiring score `>0.5`. All four outputs are
  disjoint and exact threshold values have defined ownership.
- Every piece receives exactly one state: H/even, H/odd, V/even, V/odd, or
  broken. The HiGHS MILP uses active, H/V, and parity binaries per piece, with
  H/V and parity constrained to zero while inactive, and pair-active plus gated
  H/V-XOR and parity-XOR bounded auxiliaries per retained link. Their linear
  envelopes force binary values from binary endpoints without declaring the
  link-local columns integer.
- For parallel score `p`, an active link costs `1-p` for matching H/V and `p`
  for differing H/V. For winding `d`, matching parity costs `d` and differing
  parity costs `abs(1-d)`. The terms are additive. A broken endpoint disables
  both terms. Each broken piece costs `broken_cost_per_link * degree`, where
  the default coefficient is `0.5`, degree includes hard and measured retained
  links, and the coefficient must be finite and nonnegative.
- HiGHS must return an optimal MIP solution before label artifacts are written.
  Default relative and absolute gaps are `1e-4` and `1e-6`; `--mip-gap` changes
  the finite nonnegative relative tolerance, including zero for an exact proof,
  and reports the achieved gap. The solver seed and model ordering are fixed. Each active connected
  component is canonicalized through objective-preserving global flips so its
  lowest piece ID is H/even; zero-degree pieces are canonically broken.
- Label OBJ suffixes are `_h_even.obj`, `_h_odd.obj`, `_v_even.obj`,
  `_v_odd.obj`, and `_broken.obj`. Each object is the sampled piece polyline and
  carries stable piece, source-trace, and trace-local piece IDs. Reports include
  objective, orientation/winding/broken decomposition, model dimensions, MIP
  nodes and gap, solve time, and all five label counts.
- `--lp-relaxation` makes the three piece columns continuous on `[0,1]` and
  writes raw active, H/V, and parity values to
  `<output-stem>_values.csv`; it must not threshold, repair, or emit the
  five MILP label OBJs. It additionally emits diagnostic threshold layers with
  five ordinary label suffixes: V and odd own exact `0.5`, and
  active owns values at or above the mean activity. Values below the mean are
  shown as broken. These layers must be identified as visualization rather
  than an integer solution. Every link-local H/V and parity difference has one
  stable meaning: zero for an inactive link, otherwise zero for matching
  endpoint labels and one for differing endpoint labels. Its full gated XOR
  hull remains exact whenever piece values are binary, including signed direct
  objective coefficients.
- The lowest piece in each input graph component has only its H/V and parity
  upper bounds fixed to zero; its active value remains free. This removes the
  global flip when the root is active without changing solutions in which it
  is broken. LP mode enumerates every graph triangle and adds four gated
  cut-polytope inequalities separately for H/V and parity. All-active triangles
  obey ordinary XOR consistency; with a broken vertex, the opposite active
  edge remains free to differ. Reports include gauge-root, triangle, and
  triangle-row counts. Materializing every triangle is a diagnostic and may
  become impractical on dense crops; no silent subset or cap is permitted.
- LP backend selection is explicit and diagnostic-only. `--lp-parallel`
  requests HiGHS parallel LP execution and `--lp-solver` accepts `choose`,
  `simplex`, `hipo`, or `ipm`; both require `--lp-relaxation`. The default
  remains HiGHS automatic solver and parallel selection (`choose`). Reports
  identify the requested solver, requested parallel mode, and thread count,
  not the backend HiGHS ultimately selected. These options do not alter the
  integer labeling path or the LP formulation.
  A requested backend that is unavailable in the linked HiGHS build must fail
  explicitly; it must not silently substitute a different solver.
- `--exclude-parallel-separate-winding` is an opt-in labeling-only ablation.
  It removes exactly non-hard links with `parallel_score > 0.5` and
  `winding_distance >= 0.5` before degree, adjacency, gauge, triangle, row, and
  objective construction. Hard continuity links are retained regardless of
  scores, the complete extracted report remains the input to constraint OBJ
  visualization, and retained/excluded counts are reported. Default-off model
  behavior is unchanged.
- `--perpendicular-only` removes every non-hard measured link with
  `perpendicular_score <= 0.5` before the same model construction stages.
  Hard continuity remains retained with its existing objective semantics.
  It conflicts with `--exclude-parallel-separate-winding`, reports its own
  exclusion count, and leaves default-off behavior unchanged.
- `--hv-only` is an independent opt-in labeling ablation. It retains the same
  filtered graph, degrees, broken penalties, active variables, H/V variables,
  pair-active variables, H/V gated-XOR variables, H/V gauges, and H/V triangle
  cuts, while constructing no parity variables, parity gated-XOR variables,
  parity gauges, parity triangle cuts, or winding objective terms. With `N`
  pieces, `E` retained links, and `T` relaxation triangles, the model has
  exactly `2N + 2E` columns, `N + 8E + 4T` rows, `2N` MILP integer columns, and
  `4T` reported triangle rows. Winding cost is exactly zero.
- H/V-only LP and MILP reports must contain exactly `N` parity values equal to
  zero. Active MILP pieces decode only to H/even or V/even. The stable values
  CSV and five label OBJ paths do not change; H/odd and V/odd are valid empty
  OBJ files. Solver-objective validation uses only broken plus orientation
  cost in this mode. The default joint model and its output are unchanged.
- `--exact-perpendicular-milp` is an H/V-only diagnostic and is mutually
  exclusive with `--lp-relaxation` and LP backend controls. Piece activity is
  binary, piece H/V is continuous on `[0,1]`, and parity is absent/fixed zero.
  Pair activity and H/V difference remain bounded continuous auxiliaries.
- For parallel score `p`, every active edge pays
  `(1-p) + (2*p-1)*abs(h_a-h_b)`. Edges with `p <= 0.5` receive one binary
  endpoint-order variable and pair-gated big-M-2 upper bounds so their
  difference equals the actual absolute endpoint difference. Edges with
  `p > 0.5` obtain the same equality through the positive difference objective
  coefficient and the existing lower envelope. Inactive edges have zero
  difference and must not constrain the active endpoint.
- Exact-perpendicular mode must not enumerate or add triangle cuts: all edge
  relations derive from shared scalar piece values. With `N` pieces, `E`
  retained links, and `P` links having `p <= 0.5`, it has exactly
  `2N + 2E + P` columns, `N + P` integer columns, `N + 8E + 2P` rows, and zero
  triangle rows. It emits continuous-value CSV and threshold OBJ diagnostics,
  with binary activity and continuous H/V, rather than discrete label OBJs.
- `vc_fiber_trace_chunk consensus` is a separate non-HiGHS H/V diagnostic that
  reuses canonical constraint extraction and its exclusive winding cutoff. It
  groups pieces by original trace index, discards same-trace links, and counts
  each retained cross-trace piece-pair constraint once. Accumulation order is
  stable by neighbor trace and piece IDs.
- `--constraints-per-fiber K` is an opt-in shared constraint-pruning stage for
  `constraints` and `consensus`; omission preserves the extracted graph
  exactly. It runs after canonical extraction and winding-cutoff rejection but
  before constraint OBJ output, consensus, or HiGHS labeling. The later
  labeling-only `--exclude-parallel-separate-winding` ablation, when requested,
  therefore runs after strength pruning and reports its additional exclusions
  separately.
- Every non-hard cross-trace piece-pair link has strength
  `abs(2*p-1) * max(0, 1-d/D)`, where `p` is normalized parallel score, `d` is
  closest distance in base voxels, and `D` is the exact positive extraction
  maximum distance. Zero-strength links are excluded. Each source trace ranks
  its incident positive-strength links by descending strength, ascending
  distance, then normalized ascending piece IDs. A link survives only if both
  endpoint traces rank that individual piece-pair link in their top `K`.
  Multiple piece-pair links to the same neighboring source trace consume
  multiple slots. This mutual graph is only the initial sparse proposal.
- Connectivity recovery considers every discarded link with exact positive
  strength and adds only links whose endpoint source traces occupy different
  current components. A strength-ordered first pass accepts bridges only while
  both post-add degrees remain at most `K`. A fallback dynamically minimizes
  post-add overflow
  `max(0,degree_a+1-K)+max(0,degree_b+1-K)`, then prefers greater strength,
  smaller distance, and normalized piece IDs. Zero-strength links never bridge.
  Canonical DSU roots use the minimum source-trace index, and final constraint
  order is canonical and independent of input order.
- For each original positive-strength component containing `c` mutual
  components, recovery adds exactly `c-1` links and no cycle edge. Globally,
  accepted recovery links equal mutual components minus original positive-
  graph components, and the final partition exactly matches the original.
  Therefore `K` is a target rather than a guaranteed cap. Reported fallback
  overflow is a result of the greedy policy, not proof that another
  degree-bounded spanning forest cannot exist.
- Hard continuity links must connect pieces of the same source trace, remain
  unchanged, and do not consume top-K slots or contribute to cross-trace graph
  degree. Soft same-trace links, hard cross-trace links, duplicate canonical
  piece pairs, invalid references, non-finite/out-of-range evidence, and a
  nonpositive `D` or `K` fail explicitly.
- Pruning diagnostics are separate from extraction diagnostics. Input/mutual/final
  graph statistics count source traces represented by at least one extracted
  piece, individual retained cross-trace constraints as degree, and connected
  components in the cross-trace graph; every isolated valid trace is one
  component. They report total/cross-link counts, min/mean/median/max degree,
  isolates, and components, with mean degree equal to `2E/V`. They also report
  recovery candidates, expected/actual bridges, cap-respecting versus fallback
  bridges, and final traces above `K`. Constraint OBJ, consensus, and HiGHS
  receive the same recovered constraint report. A later labeling-only ablation
  may independently fragment the HiGHS model and is outside this guarantee.
- Crop bounds for consensus are the stored trace-artifact bounds and must be
  finite with positive XYZ extents. The nominal crop side is
  `min(maximum_base_xyz - minimum_base_xyz)` and its center is the arithmetic
  XYZ midpoint. The primary H seed must have finite positive arc length
  strictly greater than half the nominal side. Eligible traces are ranked by
  descending endpoint-chord/arc-length straightness, ascending exact minimum
  3D Euclidean crop-center-to-polyline-segment distance in base voxels,
  descending arc length, then ascending trace index. These comparisons use no
  tolerance. If no valid trace passes the strict primary cutoff, consensus
  fails explicitly. Lines without two distinct points are immediately broken
  and do not count as assignments.
  When active assigned evidence exists, the next trace maximizes
  `constraint_count / mean_closest_distance_base_voxels`; zero distance is
  infinite, then ties prefer greater count, smaller mean distance, and lower
  trace index. Links to broken traces do not provide evidence. Exhausting
  active evidence starts another H seed using the same geometric ranking but
  without the primary length cutoff.
- A candidate H/V assignment sums `1-p` on equal labels and `p` on differing
  labels for every current active evidence link with parallel score `p`.
  Broken costs `broken_cost_per_link * current_active_evidence_count`. Strict
  minimum wins with H, V, broken tie order. Costs are irreversible incremental
  choices: later links to a broken trace remain disabled and do not
  retroactively alter its cost. Winding and orientation confidence do not
  affect candidate priority.
- Consensus output contains final full-trace `_h.obj`, `_v.obj`, and
  `_broken.obj` layers plus `_step_N_h.obj`, `_step_N_v.obj`, and
  `_step_N_broken.obj` snapshots for every ten total valid assignments through
  100 and every 100 thereafter. A snapshot includes the decision at assignment
  `N`. Seeds and broken choices count toward `N`; degenerate traces do not and
  occur in none of the layers. Each triplet partitions the assigned valid
  traces, so its counts sum to `N`; every file is written even when empty.
  Final paths never alias milestone paths, and an explicit output basename
  owns and overwrites all of these consensus layers. HiGHS-only flags are
  invalid in consensus mode. Console output
  includes detailed rows for the first 100 assignments and ends with the full
  assignment, label-count, and objective summary.

### Binary BP constraint-consistency diagnostic

- `direction-ablation --bp-only` processes only the final admitted cohort and
  must not call the HiGHS MILP or LP solvers. Every extracted constraint piece
  is one BP node. Optional strength pruning precedes the shared
  labeling constraint selector; BP receives the selector's retained constraints
  in their original order. `--perpendicular-only` optionally restricts the
  graph; without it, complementary parallel and perpendicular evidence in
  `[0,1]` is accepted. Canonical same-source continuity is exact and edge-local
  by default: two active endpoints must share H/V, while either endpoint may be
  Mixed/Defect and neutralize that edge. Explicit finite compatibility mode
  restores its parallel-score-1 same-label evidence. Missing, duplicate,
  nonconsecutive, malformed, or cross-source hard continuity; soft same-source
  links; hard/soft piece-pair collisions; invalid ownership; and
  non-complementary evidence fail.
- Admitted dense source traces remain the input to the established
  central-straight seed selection. The selected source's exact dense clipped
  piece with minimum Euclidean crop-center distance becomes the sole hard-H
  seed; exact ties use ascending global piece ID. A selected source without a
  valid piece fails. Optional balance weights are piece arc spans, including
  overlap once for every piece.
- BP merges soft measurements by unordered represented-piece pair. A merged
  factor's same cost is `sum(1-p_k)`, different cost is `sum(p_k)`, strength is
  `abs(same-different)`, and the lower-cost relation may be same or different.
  Before inference, subtract `min(same,different)` from both oriented costs, so
  only decisiveness remains. An exact tie is omitted from the effective graph,
  degree, components, and mismatch accounting; neutral merged-factor and raw-
  measurement counts are reported separately. Near-ties retain their exact
  sign and nonzero strength without an epsilon. Degree counts effective unique
  factors while incident-measurement count retains effective raw measurements.
- Consistency diagnostics resolve `h<=0.25` as V and `h>=0.75` as H, inclusively.
  Hard mismatch count is the number of resolved incident factors whose observed
  equal/different relation disagrees with the factor's lower-cost relation. Its
  unweighted denominator is resolved degree and its weighted denominator is
  resolved strength. Incident factors with either endpoint unresolved
  contribute instead to unresolved degree and strength. Zero denominators are
  undefined, written as `NA`, and excluded from summaries.
- The binary soft mismatch proxy is the strength-weighted independent-endpoint
  probability of violating the preferred relation: `1-q_same` for a
  same-preferring factor and `q_same` for a different-preferring factor, where
  `q_same=h_i*h_j+(1-h_i)*(1-h_j)`. It is not a calibrated pair marginal.
  Neighbor support maps the neighbor H/V evidence through the preferred same
  or different relation. Neighbor support balance is
  `2*min(sum(w*(1-h_j)),sum(w*h_j))/sum(w)` and must be interpreted with the
  separately reported mean neighbor certainty
  `sum(w*abs(2*h_j-1))/sum(w)`.
- `<base>_consistency.csv` has stable global piece, original source
  trace, source-local piece, and begin/end base-arc identities, initial
  reference group, BP status and thresholds, horizontalness,
  degree/measurement/strength partitions, hard mismatch fields, and the three
  smooth diagnostics. It is replaced on each successful run. Console summaries
  are equal-per-piece count/min/mean/median/p90/max by Direction1, Direction2,
  and Mixed, excluding undefined values. Tie-aware AUROC compares Mixed against
  both trusted groups, with the printed direction defining whether higher or
  lower values predict Mixed.
- `--bp-inference sum-product` is valid only with `--bp-only` and selects
  binary sum-product on exactly the same merged factor graph and central
  straight hard-H seed. Omission retains min-sum. For factor costs
  `E_same,E_diff` and positive temperature `T`, the log potentials are
  `-E_same/T,-E_diff/T`; subtracting a common per-factor offset is allowed
  because it must not alter any message ratio.
- A directed sum-product message is
  `ell_i->j=log m_i->j(H_j)-log m_i->j(V_j)`. Given cavity log odds `r`, its
  raw update is
  `logsumexp(-E_diff/T,r-E_same/T)-logsumexp(-E_same/T,r-E_diff/T)`.
  The hard-H seed emits `(E_diff-E_same)/T`. All messages start at zero and
  update synchronously as `old+damping*(raw-old)`, including seed messages;
  convergence uses the maximum post-damping change.
- Sum-product horizontalness is the normalized node marginal
  `P(H)=sigmoid(sum incoming log messages)`, with the seed exactly one.
  Unseeded components retain H/V gauge symmetry and therefore report `0.5`.
  Marginals are exact on trees and approximate loopy-BP/Bethe marginals on
  cyclic components after convergence. `message_limit` publishes the last
  finite iterate with an explicitly nonconverged status.
- Sum-product rejects all population-balance modes and balance-only target,
  strength, iteration, and tolerance controls. Temperature scales factor
  energies in sum-product; it remains only a post-hoc min-marginal display
  scale in min-sum. BP owns `<base>_orientation_p0.obj` through `_p9.obj` and
  `<base>_consistency.csv`, which records its inference name, temperature, and
  status. A later run replaces the current-result artifacts. The soft mismatch
  value remains an endpoint-independence proxy rather than a pairwise marginal.
- `--bp-inference sum-product-mixed` is a separate experimental categorical
  V/Mixed/H solver over the same merged orientation graph. Mixed denotes an
  orientation defect, not a third direction. Oriented pairs use the normalized
  `E_same`/`E_diff`; every pairwise energy with at least one Mixed endpoint is
  zero. Normalization ensures at least one oriented relation also has zero
  energy, so a common raw factor offset cannot favor Mixed. Each non-seed node
  instead has unary energies `U(V)=U(H)=0` and
  `U(Mixed)=bp_mixed_cost * incident_measurements`. Incidence sums the raw
  measurement count of each retained non-neutral merged factor at that node;
  measurements in omitted neutral factors count zero. The scaled unary is
  still applied once at the node, so Mixed cannot propagate transitively.
  `--bp-mixed-cost` defaults to `1`, must be finite and nonnegative, and is
  invalid outside this inference mode.
- Ternary directed messages contain V/Mixed/H log values. Raw and post-damping
  messages subtract their log-sum-exp gauge, and convergence uses the maximum
  normalized post-damping change. The 3x3 update is
  `m_i->j(t)=logsumexp_s(-U_i(s)/T + sum_{k!=j}m_k->i(s)
  -E(s,t)/T)` followed by normalization. The final marginal uses the same
  unary plus all incoming messages. A cavity clamped to Mixed sends a uniform
  factor message; a soft cavity may still transmit evidence from its residual
  V/H mass. Merged oriented energies subtract their common minimum before the
  3x3 potential is formed; Mixed entries remain zero.
- The central seed is exactly H. An isolated unseeded node has uniform
  marginals only when `bp_mixed_cost=0`; otherwise its V/Mixed/H probabilities
  are proportional to `(1,exp(-bp_mixed_cost/T),1)`. Any unseeded connected
  component retains exact V/H gauge symmetry but can have nonuniform Mixed
  probability. Exact argmax ties are reported separately rather than assigned
  to Mixed.
- Ternary reports normalized `p_v,p_mixed,p_h`. Its legacy scalar orientation
  projection is `p_h+0.5*p_mixed`; it may drive orientation bands and
  explicitly named heuristic consistency output but must not be called `P(H)`
  or treated as a calibrated binary marginal. Mixed discrimination uses the
  gauge-invariant `p_mixed` directly.
- Ternary soft mismatch uses explicit state marginals. For a same-preferring
  factor its violation probability is `pV_i*pH_j+pH_i*pV_j`; for a different-
  preferring factor it is `pV_i*pV_j+pH_i*pH_j`. Terms involving Mixed are zero.
  Neighbor support similarly maps only explicit neighbor V/H mass through the
  preferred relation, and ternary neighbor certainty is `abs(pH-pV)`.
- Ternary owns `<base>_orientation_p0..p9.obj` projection bands,
  `<base>_error_probability_p0..p9.obj` Mixed/error-probability bands, and
  `<base>_consistency.csv` containing the three
  probabilities, projection, inference temperature, Mixed unary cost, and
  status.
  It additionally owns the mutually exclusive argmax layers `<base>_v.obj`,
  `<base>_err.obj`, `<base>_h.obj`, and `<base>_tie.obj`; every
  represented fiber occurs in exactly one layer and exact ties are never
  silently assigned to Mixed.

## BP-aligned Lasagna normal samples

- Normal sign alignment consumes only regular manifest-backed Lasagna
  `grad_mag`/`nx`/`ny` samples. It must validate the normal dataset structure,
  open base-coordinate callers with `workingToBaseScale=1`, and must not use
  `NormalGridVolume` or rewrite the source Lasagna Zarr.
- Binary pairwise sum-product is one shared core implementation for fiber BP
  and normal alignment. Extracting or extending it must preserve fiber factor
  and message order, central-H seed semantics, disconnected unseeded
  probability `0.5`, probabilities, log odds, iteration count, residual, and
  convergence status.
- For normalized ambiguous normal axes with signed dot `d` clamped to
  `[-1,1]`, normal-alignment costs are exactly `E_same=(1-d)/2` and
  `E_different=(1+d)/2`. Exact neutral factors are omitted. Temperature is
  applied only by BP. Every normal-alignment connected component fixes its
  lowest retained node to unflipped state zero; this per-component gauge rule
  does not apply to fiber BP.
- Standalone sampling uses finite half-open base-voxel XYZ bounds and a
  globally anchored isotropic lattice. Axis indices run from
  `ceil(min/spacing)` up to but excluding `ceil(max/spacing)`. Invalid samples
  are compacted in stable Z/Y/X order while retaining lattice identity.
- The standalone graph connects each retained pair within a positive
  Chebyshev lattice radius once, with no distance weight; radius one is the
  default 26-neighborhood. This topology is diagnostic and must not silently
  become the future H/V graph because it can couple nearby parallel sheets and
  has no `grad_mag` confidence weighting.
- A posterior flip probability strictly greater than `0.5` flips the sampled
  axis; ties remain unflipped. A finite message-limit iterate remains usable
  and explicitly nonconverged. Invalid data/configuration is a hard failure.
- Paired standalone OBJ diagnostics use the exact same retained centers and
  sample order. Each normal emits two crossed undirected base strokes and one
  directed normal stroke in distinct OBJ groups. The aligned normal set is a
  reusable core result for later H/V optimization; OBJ is diagnostic output,
  not its persistence format.
- Binary BP parallelism is explicit and defaults to one worker for reusable
  core callers. Parallel node totals use contiguous CSR incoming-message slots
  in original factor order, factor updates write disjoint slots, and every
  totals/update/swap/stop phase is synchronized. Worker count may change only
  execution time, effective-worker and timing diagnostics; it must not change
  factors, synchronous Jacobi semantics, convergence, probabilities, or log
  odds. GCC uses OpenMP while the supported Clang/MSVC configurations fall
  back to one worker through the project shim.
- Binary BP and crop normal alignment expose optional observational progress.
  Binary BP emits one serialized event after each complete synchronous message
  sweep and one success-only terminal event. Callback time is excluded from BP
  phase timings; callback exceptions leave the OpenMP region before being
  rethrown. Normal alignment reports sampling candidates at opaque batch
  boundaries, lattice sites scanned while building factors, the sum of normal
  normalization items plus factor-adjacency items plus component-visited nodes,
  complete message sweeps, and retained nodes finalized. Preparation/finalize
  callbacks are bounded to phase boundaries plus one per 65,536 work items.
  Progress must not reorder factor or component traversal or change any
  numerical report field.

## Interleaved-lattice signed crop winding BP

- Crop BP must construct one globally anchored aligned-normal lattice over the
  half-open crop plus one effective normal-channel spacing on every available
  side. Sampling uses the already open manifest sampler/cache. A nearest
  aligned lookup is usable only when A, midpoint, and B are present within
  `sqrt(3)/2 * spacing` and share one normal-alignment component.
- The finite nonnegative `windingDistance` is the closest-connector observation
  for the perpendicular hypothesis. For ordered pieces `A -> B`, its optional
  sign is `sign(dot(B-A, aligned_normal_midpoint))` and its meaning is exactly
  `winding(B)-winding(A)`. Canonical endpoint reversal negates this target.
- Parallel constraints have an independent `parallelWindingDistance`. Reuse
  exactly the matched connector pairs that contributed tangent-valid samples
  to the parallel score in both walk directions, with the closest pair included
  exactly once. Integrate all connectors in one existing batched Lasagna read.
  The closest value is mandatory; omit nonfinite additional values. For an
  unoriented report, use the sorted unsigned median, averaging the middle pair
  for an even count.
- For an oriented report, sign each parallel sample from its connector and
  aligned midpoint normal. Its endpoint and midpoint normal samples must all
  belong to the closest connector's normal-alignment component; omit samples
  from another independently gauged component. The signed median is the
  authoritative parallel target and its magnitude is
  `parallelWindingDistance`. If the closest connector cannot be signed, expose
  no signed parallel target. Missing aligned sign removes a nonzero parallel
  winding term, not its H/V factor; exact zero distance remains sign-free.
- The raw extraction cutoff is exclusive and applies to both the closest
  perpendicular distance and final parallel median magnitude. Either reaching
  the cutoff rejects the complete measured link as one winding-cutoff
  rejection. Invalid additional walk samples do not reject the link.
- H/V-aware interleaved winding BP converts every admitted nonnegative raw
  `parallelWindingDistance` to the nearest signed integer target for its
  parallel component:
  `[0,0.5) -> 0`, `[0.5,1.5) -> 1`, and so on. Its loss is
  `parallel_weight*abs(latent_delta-signed_integer_target)`, independent of
  measurement gain/scale. It separately converts each
  available raw signed observation to a signed half-integer interval center
  for its perpendicular component: `(0,1] -> 0.5`, `(1,2] -> 1.5`, and so on.
  Exact signed zero and hard continuity stay zero. The raw exclusive cutoff
  runs before conversion and is not re-applied to either effective value. The independent
  integer-only winding diagnostic remains raw because it has no H/V half
  ladder. Alternating and joint-grid inference consume the same converted
  target. Fixed phase `0.5` and scale `1.0` define the exact-step experiment;
  adaptive calibration remains an explicit comparison mode.
- H/V diagnostic and winding-BP CLI paths default to the exclusive raw winding
  cutoff `4.0`; the shared extraction config and legacy parity-labeling path
  retain their `<1.5` default. Independently multiply each component by
  `2^-floor(abs(effective_target))`: parallel distances `0`, `1`, `2`, and `3`
  and perpendicular targets `0.5`, `1.5`, `2.5`, and `3.5` each use
  `1`, `0.5`, `0.25`, and `0.125`, respectively. Each measured
  constraint retains only its dominant hypothesis: parallel dominance uses the
  integer-distance term and perpendicular dominance uses the signed-offset
  term. The selected term and decay apply consistently in continuous
  initialization, calibration, and discrete inference. Hard continuity remains
  a parallel zero-distance factor. The formula continues safely for larger bins
  admitted by explicit cutoff overrides.
- `--parallel-winding-cutoff N` is a finite positive exclusive cutoff on the
  quantized integer parallel distance in H/V-aware winding solves. `0.5` keeps
  only same-winding distance zero. Filtering zeros the winding weight of a
  parallel-dominant factor while retaining its H/V orientation score,
  extracted/stored constraint, and reference diagnostic. Perpendicular-dominant
  factors are unaffected. Filtered orientation-only factors do not join winding
  gauge components.
  The default is no additional parallel cutoff.
- In `sum-product-mixed`, orientation and winding are one joint state per split
  piece. Every piece remains a distinct variable. By default, a same-trace
  continuation edge accepts either state when at least one endpoint is Defect;
  if both endpoints are active, their H/V class and integer winding must be
  identical. Defect therefore splits a source into independent active runs.
  Enforce this in the orientation prepass, winding pair potentials, and
  deterministic final decoding. A finite compatibility mode may restore the previous
  parallel-score-1, zero-distance pair factor. Other BP modes retain the
  independent integer-winding diagnostic.
- Each factor/message component has one local class A/B gauge, while each
  disconnected subgraph with effective winding evidence has its own integer
  offset gauge. The crop-central class gauge is fixed to `(A,0)`. Additional
  integer gauges fix `k=0` while retaining A/B choice, so filtering a winding
  component never removes the same factor's orientation evidence. A has latent coordinate `k`; B has
  `k+sign_c*phase`, where `k` is integer, shared phase magnitude is in
  `[0,0.5]`, and deterministic component sign `sign_c` accounts for otherwise
  incomparable class swaps. Absolute `k` values across components are not
  physically comparable.
- Before a BP-only cohort enters its final winding solve, retain only the
  largest component of the exact effective-winding graph. Component formation
  must reuse the solver's hard-continuity, merged-measurement, signed-target,
  quantized multiplier, parallel-cutoff, and fixed-Mixed rules. For fixed
  orientation, run the orientation prepass, remove every smaller winding
  component, remap source traces/pieces/constraints, rebuild topology, and
  repeat monotonically until one component remains; reuse the stable final
  prepass. A removed interior piece splits the retained pieces of that source
  trace into separate contiguous represented runs; an arc gap must not be
  relabeled as hard continuity. Equal-size components prefer the crop-central piece, then the lowest
  original piece index. Reference fibers never affect this selection. Print
  one cumulative before/after trace, piece, and constraint summary per solve.
- `sum-product-mixed` exposes `joint-grid` and `alternating` winding solvers.
  `joint-grid` is the default. `alternating` is the established comparison
  implementation described below and retains its orientation pre-pass,
  multi-start calibration, numerical defaults, and output conventions.
- Both winding solvers optionally support fixed-prepass orientation. The
  ordinary Mixed-state BP runs once, and each piece's unique H/Mixed/V MAP is
  retained as its pre-pass class; an exact MAP tie becomes Mixed. The
  prepass uses `--bp-mixed-cost`. Its exact fixed assignment is written as
  `<base>_prepass_h.obj`, `<base>_prepass_v.obj`,
  `<base>_prepass_err.obj`, and the shared empty
  `<base>_prepass_tie.obj` artifact. The
  subsequent solver cannot switch H to V or V to H, but may declare either
  directional piece Mixed/Defect when winding evidence is incompatible. A
  pre-pass H/V piece has one fixed-direction state per candidate integer plus
  one winding-free Defect state. A pre-pass Defect piece has only that single
  Defect state. Constraints incident to a pre-pass Defect are removed before
  continuous initialization, component/gauge construction, and discrete BP.
  A late Defect assignment makes every incident pair factor neutral.
- Hard split continuity is enabled by default. An active/Defect or
  Defect/Defect continuation edge is neutral. A continuation edge with two
  active endpoints requires identical H/V and winding state. The orientation
  prepass and final winding decode must apply the same edge invariant because
  independent nodewise marginal MAP values do not guarantee pairwise
  feasibility. Deterministic projection preserves the seed or winding gauge,
  otherwise disables the lower-confidence endpoint, and uses the larger node
  index as the exact tie-break. In explicit finite compatibility mode, the nonnegative
  `--piece-break-cost` defaults to `0` and is charged exactly once when exactly
  one continuity endpoint is active. Do not charge active/active,
  Defect/Defect, or measured cross-trace pairs. Multiple measurements merged
  into one prepared edge must not multiply the cost. Divide it by orientation
  temperature and preserve it in summary and CSV output.
- A newly available late-Defect state has log unary
  `-winding_defect_cost * incident_measurements / orientation_temperature`.
  In fixed-prepass mode, incidence counts only retained measurements with an
  active winding term or hard-sign restriction after filtering. The finite
  nonnegative `--winding-defect-cost` defaults independently to `1`; changing it must not
  change the prepass unary. Fixed-prepass winding factors
  do not repeat the H/V same/different orientation energy; they use only
  winding evidence, so the completed orientation pre-pass is not counted
  twice. Alternating calibration and component-sign updates exclude pair mass
  whenever either final endpoint state is Defect.
  Calibration, component sign, integer support, and winding potentials retain
  the selected solver's existing behavior. Reports preserve the soft pre-pass
  marginals and separately record `fixed-prepass`, the pre-pass class, and the
  final H/Defect/V posterior per piece. Winding diagnostics record the separate
  winding Defect cost. Final OBJ layers use the final class.
  Joint orientation inference remains the default when the option is absent.
- `joint-grid` has no H/V/Mixed pre-pass. Each piece has `(A,k)` and `(B,k)`
  states plus one winding-free Defect state, with the crop-central piece of each connected constraint
  component fixed to `(A,0)`. Each component has a binary ladder-order sign,
  while one explicit calibration variable over `(log gain, phase)` is shared
  by every component. Aligned-normal sign resolution remains separate and is
  never inferred by this variable.
- Without fixed orientation, joint-grid uses `--winding-defect-cost` as its
  sole Defect unary. Non-fixed alternating uses the orientation prepass
  posterior as its orientation prior and does not charge this unary again.
- A joint non-Mixed factor charges orientation and winding evidence once.
  Same classes cost the perpendicular score, different classes cost the
  parallel score, and every measurement additionally contributes
  `parallel*abs(delta-signed_integer_target)` plus
  `perpendicular*abs(gain*delta-signed_target)` when its signed target exists.
  Orientation and the Mixed unary retain `--bp-temperature`; winding retains
  its established temperature `0.25`. A Defect endpoint makes the complete
  pair factor neutral, so it transmits neither orientation nor winding
  evidence and cannot couple winding ladders through a crossing fiber.
- Each admitted enabled nonzero dominant sign is hard when global sign mode is
  hard or its raw absolute connector alignment with the aligned Lasagna normal
  reaches the configured inclusive threshold. The default threshold is
  `cos(30 degrees)`. Perpendicular uses the closest connector alignment;
  parallel uses the deterministic median over admitted signed connector
  samples. This gate runs after dominant-hypothesis, sign-enable, nonzero-target,
  and parallel-cutoff admission and before confidence transforms. Missing
  alignment is not promoted; disabling the gate leaves finite-sign behavior.
  For a hard sign, an active-active state is admissible only when
  `signed_target*predicted_delta > 0`; zero and reversed deltas have exactly zero
  probability. A Defect endpoint remains neutral. Solver preparation,
  diagnostics, reference inference, and final feasibility decoding must use the
  identical promotion rule.
- Exact zero-probability messages use finite-log sums plus negative-infinity
  counts for cavity construction. Damping changes finite/impossible support
  explicitly and must never evaluate `-inf - -inf` or replace the exclusion
  with a finite sentinel penalty.
- Joint gain uses an absolute lattice in `log(gain)` centered initially on
  gain 1; phase uses a fixed absolute lattice spanning `[0,0.5]`. Support
  changes are considered only after both message and calibration posteriors
  settle. One-sided boundary pressure shifts the gain window by one physical
  cell only when the leaving boundary is negligible. Otherwise the window
  grows within the explicit resource guard. Retained physical cells preserve
  their messages and newly exposed cells receive neutral incoming messages.
  Integer support similarly expands from settled orientation-marginalized
  boundary pressure without restarting BP.
- Joint convergence requires message residual convergence, calibration
  posterior stability, resolved support pressure, and consecutive stable
  iterations. Reports include solver mode, calibration MAP and posterior
  means, entropy, absolute gain bounds, boundary masses, grid shifts,
  component-sign posterior/MAP, state count, and convergence. No cost-based
  post-fit or automatic fallback to `alternating` is permitted. A final
  deterministic feasibility decode may remove hard-sign violations by changing
  active pieces to Defect. It then resolves each mismatched active-active hard
  continuation edge by changing one endpoint to Defect; existing Defects split
  the chain. It repeats sign/continuity projection once. It never changes phase, scale, or component
  sign. The authoritative published state must contain neither a hard-sign
  violation nor a hard-continuity mismatch.
- Joint-grid may instead use fixed calibration when both a phase in `[0,0.5]`
  and a finite positive measurement scale are supplied. Fixed calibration is
  a distinct reported mode, not a one-cell latent calibration posterior: pair
  factors consume scalar phase and reciprocal gain directly, no calibration
  messages or marginals exist, gain support cannot move, and calibration does
  not enter resource accounting or convergence. Piece, component-sign, and
  adaptive integer-support inference remain unchanged. Integer support changes
  are gated only by settled piece/sign messages. Fixed mode reports the exact
  supplied phase and scale as both MAP and mean, one reporting cell, zero
  calibration entropy/residual/boundary mass/shifts/iterations, and calibration
  convergence equal to message convergence. Supplying only one fixed value or
  combining fixed values with explicit adaptive-grid controls is invalid.
- In `alternating`, the established Mixed-state orientation BP runs first. Its
  normalized A/Mixed/B posterior is the alternating winding stage's soft node
  prior. The
  winding stage must not repeat the same/different factor or the Mixed unary.
- For positive measurement scale `s`, raw perpendicular and parallel winding
  observations are first transformed by `s`. Perpendicular targets are then
  signed-half-integer quantized and parallel targets are coherently signed- or
  unsigned-integer quantized. Canonical class, power-of-two distance decay,
  parallel cutoff, and sign activation are derived only after that transform.
  For latent coordinate difference `delta`, a parallel-dominant measurement
  contributes `p*abs(delta-integer(raw_parallel*s))`; a
  perpendicular-dominant measurement contributes
  `q*abs(delta-half_integer(raw_perpendicular*s))`. `p` and `q` are its
  parallel and perpendicular scores, but the losing hypothesis contributes
  zero winding energy. The H/V relation potential continues to use the two
  scores as alternative same/different assignment costs, preserving confidence
  through their difference rather than treating both as winding evidence.
  The residual is expressed in latent winding units. Repeated
  endpoint pairs sum complete measurement energies. A component may consume
  signed evidence from only one aligned-normal component.
- Mixed remains a piece-local error state with one unary cost and no integer
  winding coordinate. If either endpoint is Mixed, the complete pairwise
  potential is neutral. It therefore disables parallel, perpendicular,
  orientation, calibration, and component-sign evidence for that assignment.
- For fixed phase and positive scale, run stable synchronous damped sum-product
  over `(A,k)`, `(B,k)`, and one Defect state. Sum boundary probability over
  active classes and normalize by total active mass before adaptive integer
  support expansion; zero active mass cannot request expansion. The single
  Defect message slot is remapped by identity while active slots are remapped
  by integer coordinate. Gauges contain only `(A,0)` in ordinary joint mode.
  Expansion continues until resolved or the explicit total-state guard throws.
- Alternating calibration computes full pair beliefs. Oriented pair mass forms
  an expected squared-residual proposal in inverse-scale coordinates
  `g=1/scale` and `h=g*phase`. The bounded fit uses the exact wedge induced by
  phase `[0,0.5]` and scale `[0.5,2]`; a rank-deficient normal matrix retains the
  previous unidentifiable values. Backtracking accepts the proposal only when
  the authoritative scale-first, re-quantized fixed-belief expected L1 winding
  energy does not increase.
  Both component-sign selection and phase/scale backtracking reject candidates
  that reverse any contributing nonzero signed perpendicular observation.
  Mixed pair mass does not calibrate phase or scale.
- Use deterministic phase/scale initializations. Rank converged starts before
  nonconverged starts, then select the lowest complete energy of the per-node
  joint-marginal argmax assignment; this is a decoded loopy-BP assignment, not
  an exact MAP claim. Stable ties use initialization order. A message-limit
  result stops that start's calibration and remains a labeled finite candidate.
- Reports expose joint A/Mixed/B marginals and explicit winding validity.
  Integer MAP/posterior values are authoritative only for active pieces;
  Defect rows use `NA` in CSV winding fields and are excluded from `w_N` OBJ
  layers. Reports additionally expose
  latent coordinate, phase, scale, component sign, calibration rank/status,
  initialization, decoded energy, hard-sign projected-Defect count, support,
  convergence, and timings. Published
  `w_N` layers group integer `k` after a nonnegative display offset. The CSV
  retains relative `k` and component identity. Finite iteration-limit results
  remain usable but must be labeled nonconverged.
- Winding reports also expose the authoritative post-projection MAP
  H/V/Defect class, finite MAP latent coordinate for active pieces, and the
  independently gauged effective-winding subgraph ID for every piece. A
  hard-sign-projected Defect has no latent coordinate. Consumers must use these
  fields rather than reconstructing MAP state from node marginals or the
  broader H/V factor-component ID.
- Winding factor diagnostics preserve raw original-order and raw
  canonical-order signed observations, separately expose the effective
  half-integer canonical target, its winding-weight multiplier, and its
  effective parallel/perpendicular winding weights, and derive any calibrated
  latent target from that effective value.
- Every completed winding solve must report final agreement as separate items:
  continuity; perpendicular H/V, signed winding value `0.5`/`1.5+`, and extra
  sign hardness; parallel H/V, signed winding value `0`/`1`/`2+`, and extra
  sign hardness; plus their sum. Structural extraction determines which items
  are listed even when a tested solver weight is zero. Factors with a Defect
  endpoint neutralize every applicable item independently. Active winding-value
  items compare the signed canonical target, while sign items independently
  check ordering. The reported percentage is `infringed/evaluated`; a zero
  denominator is `NA`.
- The reusable interleaved solver exposes an optional synchronous progress
  callback. Events identify preparation, the one-based multi-start
  initialization, calibration iteration, adaptive-support round, BP message
  iteration and residual, candidate-state count, current phase/scale, elapsed
  time, and terminal completion. Events are emitted only after synchronized
  message updates and are observational: enabling them must not change solver
  ordering, arithmetic, convergence, or results. The CLI throttles repeated
  message events but forces stage transitions. It must not print a global
  percentage because convergence, support expansion, and per-state work leave
  no valid fixed denominator. After one calibration completes, the CLI may
  report `eta_est` using mean calibration duration and all remaining maximum
  calibration slots, labeled `eta_basis=calibration_max`. After one full
  initialization completes it switches to mean initialization duration,
  labeled `eta_basis=initialization`. Both estimates subtract elapsed work in
  the active unit, are empirical rather than conservative, and are recomputed
  after each matching completion.

## Winding fiber Napari visualization

- `vc_fiber_trace_chunk direction-ablation` optionally accepts the paired
  `--reference-fiber-dir DIR --reference-fiber-tag TAG` diagnostic options.
  It scans regular `.json` files immediately inside `DIR` in lexicographic
  path order, selects an exact case-sensitive tag match, and strictly parses
  each selected document with the shared VC3D fiber parser. Missing `tags`
  skip a document; present tags must be an array of strings. Malformed JSON,
  malformed selected fibers, selected fibers with fewer than two dense line
  points, or zero matches are errors. The reference data never changes
  splitting, constraints, BP, or published winding labels.
- A successful tagged selection writes `<base>_reference.obj` with the exact
  preamble `VC3D tagged reference fibers` and one uniquely named ordered
  polyline per retained in-crop run. The authoritative bounds are the input
  trace artifact's finite, positive-extent, half-open base-XYZ crop; they are
  not derived from its storage grid. Point membership is `minimum <= p <
  maximum`. A computed entry/exit intersection may equal a maximum face only
  as the closure endpoint of a retained segment. Maximum-face-only and
  point/tangential contacts are discarded. Exit/re-entry produces separate
  source-ordered runs with stable source and run ordinals. Matching fibers
  with no nondegenerate crop intersection are an error. The CLI owns this
  sibling: a successful
  `direction-ablation` run without reference options removes an older sibling,
  and any unsuccessful selection, validation, or clipping does not leave a
  stale artifact.
- When tagged reference fibers are loaded, the CLI must reuse those exact
  cropped runs and invoke the canonical `extractFiberTraceConstraints` path
  exactly once, outside any ablation checkpoint loop. It must use the active
  resampling, piece length/overlap, distance, tangent-window, winding-step,
  winding-cutoff, thread, and batched normal-aligned Lasagna sampling settings.
  Downstream pruning, constraints-per-fiber limits, perpendicular-only
  selection, and labeling filters must not alter this raw diagnostic set.
  Generated hard continuity links and measured links whose runs came from the
  same selected source fiber must be excluded. Lexicographic selected-source
  path order assigns virtual winding `W_i = 0.5*i` before crop retention; an
  empty source retains its slot and all its runs share that value. Each
  remaining measured link is presentation-oriented from `min(i,j)` to
  `max(i,j)` and is emitted exactly once under the lower source. Extractor
  order is preserved within each output group. The report is retained until a
  matching reference-to-BP benchmark has fitted its global sign. Each BP
  execution formats its own copy of these tables with that sign; no
  uncalibrated duplicate is emitted.
- Reference constraint output has one source section in source order, with the
  source name and virtual winding only in its header. Each section prints a
  perpendicular table followed by a parallel table; an exact orientation-score
  tie belongs to perpendicular. Both tables have exactly `target_winding`,
  `raw_step`, `calibrated_step`, `canonical_step`, `gt_step`, and
  `calibrated_minus_gt` columns, or `(none)` when empty. `raw_step` is the
  source-oriented signed measurement before global-sign calibration. When a
  signed measurement exists, `calibrated_step` is
  `global_sign*raw_step`; an unsigned fallback remains a nonnegative magnitude
  because calibration cannot manufacture its direction. `canonical_step`
  quantizes `calibrated_step`, using the signed half-integer winding-solver rule
  in perpendicular tables and signed nearest-integer rule in parallel tables.
  `calibrated_minus_gt` is `calibrated_step` minus
  `0.5*(target_source-owner_source)`. The additive per-gauge calibration offset
  is not applied to a difference because it cancels between endpoints. Run, piece,
  distance, score, and extractor-index metadata are not printed. After all
  source tables, print one `reference constraint canonical summary` with
  `correct`, `false`, and `total`. Increment it while emitting each displayed
  measured piece-pair row; do not deduplicate repeated rows for the same source
  pair. A row is correct exactly when its calibrated, table-specific canonical step equals
  its virtual GT step. Empty diagnostics print `0 0 0`, and always satisfy
  `correct + false = total = displayed row count`. This output is diagnostic
  only and must not change main constraints, BP, or published labels.
- Each calibrated reference/reference table set must be preceded by a separate
  unweighted `reference raw signed step distributions` table. Every signed row
  is grouped by its dominant perpendicular/parallel hypothesis, source and
  target H/V class under the selected reference phase gauge, and virtual
  separation. Opposite-parity bands are `0.5`, `1.5`, and `2.5+`; same-parity
  bands are `1`, `2`, and `3+`. Each nonempty group reports count, raw signed
  minimum, mean, median, and maximum. This table must not use solver admission
  or effective weights and must not canonicalize the measurements.
- The raw distribution must be followed by `reference constraint phase
  calibration`. It uses every signed, dominant perpendicular, opposite-parity
  reference row with unit weight and enumerates winding direction plus the
  even-reference H/V assignment. For source index `i`, parity `p=i mod 2`,
  direction `d`, and phase `phi`, the two gauges are
  `d*(floor(i/2)+p*phi)` and `d*(ceil(i/2)-p*phi)`. At the run's fixed positive
  measurement scale it minimizes
  `sum(abs(predicted_delta/scale-raw_signed_delta))` exactly over `[0,0.5]` by
  testing all L1 breakpoints and both bounds. Sign penalties and all production
  factor weights are excluded. Same-parity perpendicular and both parallel
  parity classes are counted but do not identify phase. This diagnostic must
  not change solver calibration, state, or artifacts.
- Each calibrated reference/reference table set must be preceded by a separate
  `reference constraint measurement-scale calibration` table. It must use the
  matching BP benchmark global sign and the exact effective magnitude weights
  produced by the shared winding-factor preparation path: dominant hypothesis,
  canonical target, parallel-cutoff admission, class multiplier,
  power-of-two distance decay, decision confidence, and normal confidence.
  It must not duplicate those rules in the CLI. Its magnitude objective is
  `sum(w*abs(gt/scale-target))`; it must be evaluated separately for the
  globally signed continuous raw measurement and for the canonical quantized
  target using identical effective weights. The raw result estimates
  integration bias before target snapping. The canonical result describes the
  target consumed by current inference and may be pinned exactly at one by
  quantized target/GT ratios without implying exact physical calibration.
  Scale-independent hard and finite sign costs
  are excluded, while negative calibrated targets remain signed in the L1
  residual. Over the fixed diagnostic interval `[0.5,2.0]`, reciprocal scale
  `g=1/scale` is the deterministic bounded weighted median of locations
  `target/gt` with weights `w*abs(gt)`. If the weighted median has an interval
  of minimizers, prefer reciprocal scale one when it lies in that interval,
  otherwise the closest endpoint; then clamp to the diagnostic interval.
  Rows with zero GT add constant loss but do not identify scale.
- The scale table must report raw and canonical rows for `perpendicular_all`,
  `parallel_all`, `all_constraints`, and all five canonical groups
  (`perp_0.5`, `perp_1.5+`, `parallel_0`, `parallel_1`, `parallel_2+`) with total,
  admitted, and informative counts, `sum(w)`, `sum(w*abs(gt))`, scale-one loss,
  fitted scale/loss, reduction percentage, and bound status. The combined
  perpendicular row is solver-compatible. Perpendicular group rows are
  per-class diagnostics. The parallel and all-constraint aggregates and all
  parallel group rows are labeled counterfactual because the current solver's
  integer parallel magnitude bypasses measurement scale.
  Scale below one means the selected raw or canonical targets exceed known
  latent separation; scale above one means they are smaller. The existing detailed constraint tables retain
  exactly their specified six columns and computing the scale fit must not
  alter solver state or output artifacts.
- `vc_fiber_trace_chunk direction-ablation` defaults joint-grid winding
  calibration to fixed phase `0.5` and measurement scale `0.822`, derived from
  the combined continuous reference/reference fit. Explicit fixed phase and
  scale flags override both defaults together. `--winding-adaptive-calibration`
  restores latent phase/scale inference and may be combined with adaptive-grid
  controls, but not explicit fixed calibration.
- For each represented `sum-product-mixed` BP cohort, tagged references enable
  a second diagnostic extraction between canonical reference pieces and the
  exact BP piece lines. Reference runs reuse the pieces already produced by the
  reference/reference extractor; the cross extractor preserves each supplied
  line as one piece and filters distinct-trace candidates to reference/BP pairs
  before scoring. It runs against the final retained component for each
  balance/solver execution. Reference/reference,
  BP/BP, and hard-continuity links are absent. Accepted repeated piece-pair
  measurements are not deduplicated by source fiber.
- Cross constraints are evaluated only after BP and never enter its graph.
  Exact parallel/perpendicular score ties are perpendicular. For signed
  perpendicular raw target `d`, measurement scale `s`, and BP MAP latent `z`,
  first compute the signed half-integer target `q=half_integer(s*d)`. A
  reference at endpoint A of `A -> B` is inferred as `z-q`; at endpoint B it
  is `z+q`. A parallel constraint similarly scales before integer
  canonicalization; a canonical nonnegative integer distance `n` infers
  the endpoint-ordered signed candidate; an unsigned fallback retains both
  `z-n` and `z+n`. `n=0` is `parallel_same`, otherwise it is
  `parallel_other`. A Defect/invalid BP endpoint or unsigned perpendicular
  observation has no inferred candidate. Constraints whose BP endpoint is
  final Defect/Mixed or otherwise winding-invalid are excluded from both
  calibration and all benchmark totals. An unsigned perpendicular observation
  from an otherwise active endpoint remains a non-candidate and is likewise
  excluded.
- Reference source `i` retains virtual winding `0.5*i`. Before calibration,
  infer one raw half-integer winding per `(reference source, effective-winding
  gauge)` using the same dominant-factor scorer as the final `all` row, with an
  identity gauge mapping. Raw inference uses admitted evidence but must not read
  the virtual reference winding, reporting tolerance, or calibration state.
  Every raw estimate is one calibration vote regardless of constraint degree.
  Fit one global sign and one half-integer offset per gauge from candidate
  offsets `raw - sign*virtual`. For each sign and gauge, maximize literal exact
  estimate matches, then minimize summed absolute estimate residual; remaining
  offset ties prefer smaller absolute offset and then smaller signed offset.
  Across signs maximize total exact matches, then minimize total residual, then
  prefer sign `+1`. Gauge-offset magnitude must not choose global sign. Gauges
  with no admitted raw estimate are absent. The later inclusive `0.5` tolerance
  applies only to constraint accuracy and estimated-winding support; it cannot
  affect calibration. Candidate-bearing observations from calibrated gauges
  remain in those final accuracy totals even when their coefficient was
  suppressed, preserving the broader diagnostic population.
  Print balance mode, solver/status, per-gauge
  offset/exact-match/estimate-vote rows, then one
  row-oriented `reference constraint groups finite weighted L1` diagnostic per
  original reference source and evidence group. Groups are selected from the
  same dominant-hypothesis projection as the reference accuracy benchmark:
  canonical perpendicular distance `0.5`, canonical perpendicular distance
  `1.5+`, canonical parallel distance `0`, canonical parallel distance `1`,
  and canonical parallel distance `2+`. Canonical distances are absolute and
  are assigned only after half-integer or integer target quantization.
  Each group row reports valid observation count, raw finite-L1 coefficient,
  coefficient admitted after the active parallel-distance cutoff, total and
  admitted-coefficient-normalized loss at the calibrated true winding, the
  half-integer winding preferred by that group alone, and its total and
  normalized loss. Each categorized constraint contributes exactly its
  dominant BP winding term: the winning hypothesis score times its
  `2^-floor(abs(canonical_step))` distance multiplier. Perpendicular and
  parallel loss both use latent-coordinate error directly against their
  scale-first canonical target. Parallel cutoff
  suppression matches factor preparation. Inferred candidates are mapped from each independent gauge
  with `globalSign * (candidate - gaugeOffset)`, allowing one reference source
  to combine evidence from multiple gauges without another calibration.
  Preferred winding first minimizes violated hard perpendicular signed-order
  constraints and then minimizes admitted weighted L1 on the half-integer lattice;
  exact ties prefer lower signed winding.
  Empty and zero-admitted-coefficient groups print `NA` loss/inference values.
  Print hard-violation counts at truth and at the preferred winding. Add an
  `all` row containing every categorized constraint for that reference, and
  use exactly its preferred winding as the following compact row's `est_w`.
  This replaces support-count/squared-residual reference inference. BP makes a
  hard-sign-violating active state impossible and may select Defect; because
  the reference diagnostic is forced active, contradictory signs instead use
  the minimum-violation lexicographic fallback before finite factor energy.
  Then print one compact row per original selected reference source in source order, identified
  only by its virtual winding `0.5*i`. Each source row
  contains `right`, `wrong`, and `right_fraction` for perpendicular winding,
  perpendicular sign hardness, parallel-same winding, parallel-other winding,
  parallel sign hardness, and sum; zero-total fractions are `NA`.
  Follow this with aggregate `right`, `wrong`, `total`, and
  `right_percent` rows for
  those same five classes and `sum`. Empty classes
  print `NA` percent, and right plus wrong always equals total. Per-source rows
  reuse the single global gauge calibration, aggregate all runs and pieces of
  one source, include explicit zero-observation sources, and sum exactly to the
  aggregate `sum` row.
- The compact reference winding table must include `parity_ok` beside `est_w`.
  For source `i`, normalize the modulo-two parity of integer half-step index
  `round(2*est_w)` and compare it with `i mod 2`. Missing estimates print `NA`;
  a finite estimate outside the half-step lattice is an invariant error. No H/V
  naming gauge is applied. Whole-winding errors preserve parity, while
  half-step errors change it.
- Before the compact winding table, benchmark solved H/V endpoint consistency.
  Carry final endpoint active state, published H/V class, and BP component in
  every reference cross observation independently of winding-target admission.
  Exclude only final Mixed/Defect or otherwise inactive endpoints; unsigned,
  cutoff-suppressed, zero-weight, and candidate-free winding observations still
  count when their endpoint is active. Reference source parity is `i mod 2`.
  Parallel-dominant constraints expect the same class and perpendicular-
  dominant constraints expect the opposite class; exact score ties are
  perpendicular. Fit the binary mapping from even/odd reference parity to H/V
  independently per BP component by maximizing correct observations, choosing
  even-to-H on an exact tie. Print component mapping vote totals plus per-source
  and aggregate perpendicular, parallel, and sum right/wrong/fraction rows.
  Repeated cross measurements count separately, zero-observation source rows
  remain present, and source sums must equal aggregates. This measures
  consistency between solved class and dominant geometry; it does not identify
  which one caused a mismatch.
- Before each reference-to-BP benchmark, print a final-state cohort table for
  the BP execution. Capture the central cohort before component filtering as
  pieces whose original local `pieceIndex` is one; carry that bit through all
  subset remapping. The other row contains every remaining retained piece.
  Count post-projection winding-valid H and V separately and as active; final
  Mixed/Defect and any winding-invalid piece count as Defect. Include total
  pieces and Defect percentage, using `NA` for an empty cohort. Central plus
  non-central must exactly equal the total row. Without reference input, emit
  the same table with the deferred BP diagnostics.
- Immediately after that state table, print admitted winding evidence for the
  same central, non-central, and total cohorts. Use the winding solver's
  authoritative factor diagnostics after fixed-orientation exclusion,
  quantization, distance-cutoff suppression, and coefficient downweighting.
  Report a unique admitted-measurement row followed by continuity,
  perpendicular-winding, perpendicular-sign, parallel-same-winding,
  parallel-other-winding, and parallel-sign term rows. Winding rows report the
  ordinary signed-value coefficient. Sign rows report the additional finite or
  hard ordering rule separately.
  A measured constraint contributes to exactly one perpendicular or parallel
  row according to its dominant hypothesis, matching the solver. Each endpoint is one incidence: an internal
  constraint contributes twice to one cohort and a cross-cohort constraint
  once to each. The measurement row counts admitted winding measurements; it
  can be smaller than the Defect-unary degree when the parallel cutoff suppresses
  a parallel-dominant winding term. Report incidence and coefficient totals, incidence and
  coefficient per final active and Defect piece. For the sign row, report hard
  signed-order incidence and incidence per final active and Defect piece.
  Active and Defect numerators are endpoint-stratified before division. A
  Defect endpoint neutralizes its realized pair energy, so its reported
  coefficient is admitted evidence that selecting Defect evades, not realized
  final energy. The table is descriptive association, not proof of causality.
- Buffer the reference/reference tables and every reference-to-BP benchmark
  generated across checkpoints and balance modes. Emit them in execution order
  after all ordinary `direction-ablation` summaries and immediately before the
  command returns.
- The winding viewer consumes current published state artifacts matching
  `<base>_w_N_{h,v,err,tie}.obj`. Aggregate `<base>_w_N.obj`, CSV reports, and
  unrelated siblings are not state layers. At least one matching artifact is
  required, but state files and whole winding labels may be absent. Every file
  that is present still receives strict header and geometry validation.
- OBJ input follows the shared strict ordered-polyline contract. Container
  names are unique, indices are global and one-based but must reference only
  vertices owned by the current container, and every nonempty container owns
  exactly one ordered nonbranching path. Orphan/cross-container vertices,
  branching, cycles, disconnected chains, unsupported records, and singleton
  winding fibers are errors. The existing fiber-presence reader uses the same
  parser while retaining its own header, metadata, and crop validation.
- Every H/V/error/tie slot at every integer winding from the lowest through the
  highest observed artifact label becomes an independent managed 3D Napari
  path layer. Present nonempty artifacts supply geometry; empty or absent
  artifacts and completely absent intermediate windings become empty layers.
  H and V at one winding `N` have exactly the same bright opaque color derived
  from `N`; error and tie remain distinct from H/V and each other.
  Viewer category `Broken` contains both Mixed/error argmax (`err`) and exact
  argmax ties (`tie`). H, V, Broken, All, and None presets act across every
  winding. Full-size Previous/Next buttons snapshot and circularly rotate the
  complete managed H/V/error/tie visibility mask over that contiguous winding
  range. Next moves source slot `i` to `(i+1) mod N`; Previous moves it to
  `(i-1) mod N`, preserving state. Hidden bits move exactly like visible bits,
  so arbitrary sparse selections and missing/empty windings rotate bijectively
  without being replaced by a preset. One observed winding is a no-op. The independent reference layer
  and unmanaged Napari layers remain unchanged. The displayed winding label
  summarizes all currently visible managed winding labels and follows presets,
  rotations, and manual layer-visibility changes.
- When `<base>_reference.obj` is present, the viewer validates its dedicated
  header and ordered-polylines contract, converts base XYZ to Napari ZYX, and
  adds one visible bright `Reference fibers` layer. Winding category and
  previous/next controls do not change this independent comparison layer.
- Published `N` is the solver's nonnegative display-offset integer label.
  Absolute winding and physical H/V identity are not comparable across
  independently gauged components; visualization must not imply otherwise.
- Winding BP accepts one finite nonnegative multiplier for each canonical
  dominant factor class in tuple order `perp_0.5`, `perp_1.5+`, `parallel_0`,
  `parallel_1`, `parallel_2+`. The multiplier composes with the existing
  canonical-distance decay. Signed parallel targets are authoritative for
  integer class selection. These multipliers affect ordinary signed
  winding-value loss and
  its diagnostics only; they never scale orientation loss, hard perpendicular
  order, hard continuity, Defect unary cost, or piece-break cost.
- `--winding-hard-signs none|perpendicular|parallel|both` independently enables
  signed ordering for the dominant perpendicular and parallel hypotheses. The
  default enables both. A zero winding-value weight removes that ordinary
  signed-value term. If that dominant observation has an enabled nonzero sign,
  its separately weighted extra sign rule may remain in winding connectivity,
  Defect incidence, BP, projection, and reference inference. Zero/same-
  winding and unsigned observations never impose hard signs, and parallel
  signs obey the parallel winding cutoff. Canonical class weights do not alter
  same-trace hard continuity.
- `--winding-decision-confidence legacy|linear|cosine` scales only the selected
  dominant winding factor. For selected normalized score `s` in `[0.5,1]`,
  `legacy` uses `s`, `linear` uses `2s-1`, and `cosine` uses
  `(1-cos(pi*(2s-1)))/2`. The default is `cosine`.
- `--winding-normal-confidence none|linear|cosine` additionally scales that
  factor from the connector's absolute alignment with the aligned normal.
  `none` uses one; `linear` uses `1-2*acos(abs_dot)/pi`; `cosine` uses
  `abs_dot`. Perpendicular evidence uses the closest connector alignment.
  Parallel evidence uses the deterministic median over every admitted signed
  connector sample, averaging the central pair for an even count. Missing,
  invalid, or component-incompatible alignment is neutral under `none` and
  zero under weighted modes. The default is `linear`.
- `--winding-sign-weights PERP,PARALLEL` supplies finite nonnegative multipliers
  for the additional perpendicular and parallel sign-hardness rules, default
  `1,0.5`. Zero disables that relation's extra finite penalty and aligned hard
  promotion but does not remove the ordinary signed winding-value residual.
- `--winding-sign-cost` defaults to finite cost `44`. A wrong-sign or exactly
  zero predicted delta adds
  `sign_cost * relation_sign_weight * decision_confidence *
  normal_confidence` per measurement. Winding class weights and distance decay
  do not scale this term. Zero cost
  or zero confidence removes the sign factor from energy, connectivity, and
  Defect incidence. The literal value `hard` restores strict rejection
  regardless of confidence. Independently, `--winding-hard-sign-angle` defaults
  to 30 degrees and promotes admitted signs whose raw absolute normal alignment
  reaches `cos(30 degrees)` to strict rejection; `off` disables promotion.
  Solver, decoded energy, factor diagnostics, and reference inference must use
  identical coefficients and promotion. These controls never alter H/V
  orientation evidence, same-trace hard continuity, or the discrete Defect
  unary.
- Standard fixed-orientation crop evaluation defaults to orientation BP
  temperature `1.25` and winding-stage Defect cost `100`. The underlying
  winding factor temperature remains `0.25`. The default sign multipliers are
  perpendicular one and parallel one-half; finite sign cost remains `44`, decision
  confidence `cosine`, and normal confidence `linear`.
- Shared and CLI H/V-aware winding defaults are `0,0,0.5,4,1`. An explicit
  `1,1,1,1,1` tuple restores neutral class weighting. The standalone
  raw-integer solver retains unscaled measurements because it does not
  quantize the H/V ladder targets used for canonical class selection.
- `--winding-weights` runs one explicit five-value winding tuple and
  `--winding-sign-weights` runs one explicit two-value sign tuple.
  `--winding-weight-search` takes one deduplicated positive value list and runs
  its complete seven-dimensional Cartesian product, capped at 100,000
  scenarios. Search requires reference fibers and mixed BP, conflicts with a
  fixed tuple, reuses all pre-winding geometry/orientation work, isolates failed
  scenarios, and uses the selected tuple for every final artifact and
  diagnostic. Converged scenarios rank before nonconverged scenarios, followed
  by exact calibrated source estimates on the fixed reference-source
  denominator, fewer missing and incorrect estimates, more right and evaluated
  constraints, fewer wrong constraints, residual, and lexicographic tuple.
- `--winding-weight-search-local` requires an explicit `--winding-weights`
  start and is mutually exclusive with exhaustive search. It searches all five
  winding plus two sign coordinates. Each coordinate has
  a canonical tagged state: zero or an integer power-of-two exponent relative
  to its immutable positive starting base; an initially zero coordinate uses
  base one. In dimension order `0..6`, a positive coordinate proposes zero,
  `/2`, and `*2`, while zero proposes `base/2`, `base`, and `base*2`. The search
  deduplicates these at most 21 tuples, caches successes and failures by exact
  tagged tuple, moves only on a strict benchmark-quality improvement, and
  repeats to a local optimum. Residual and lexicographic tuple ordering may
  select deterministically among improving neighbors but cannot turn a quality
  tie into a move. Positive exponents remain in `[-16,16]`; out-of-range
  candidates are omitted, zero remains available at either bound, and the
  printed progress denominator is the deduplicated in-range candidate count.
  An iteration-limit exit is an error rather than a local optimum.
- A cutoff-based seven-coordinate experiment was rejected because a `0.5`
  parallel cutoff excluded parallel-other and parallel-sign benchmark items.
  The corrected no-cutoff 1024 refinement evaluated 58 scenarios from winding
  `0.5,2,1,2,1`, sign `0,1` and selected winding `0.5,0,1,2,1`, sign `0.5,1`.
  It converged with `8/8` exact reference windings and `4835/5446` correct
  fixed-denominator reference items (88.781%). All five reference classes must
  be populated during this tuning protocol. No parallel cutoff is enabled by
  default. A zero selected winding-class weight suppresses that energy term but
  does not remove its structural diagnostic or benchmark item.
- The subsequent zero-aware 1024 refinement against the complete 26-fiber
  `2026-09-01_fiber_stack2` reference set evaluated 129 scenarios. It moved
  from winding `0.5,0,1,2,1`, sign `0.5,1`, with 13 exact, 12 wrong, and 1
  missing inferred winding to winding `0,0,0.5,4,1`, sign `1,0.5`, with 16
  exact, 9 wrong, and 1 missing. The selected tuple is the shared and CLI
  default.
- Tagged VC3D reference fibers are diagnostic inputs independent of the traced
  crop. Reference OBJ export, reference-to-reference extraction, and
  reference-to-BP benchmarking must use each selected JSON fiber's complete
  dense line exactly once in deterministic filename order; they must not clip
  or split that line at the stored trace artifact bounds.
