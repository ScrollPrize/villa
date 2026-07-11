# 2D Fiber Trace Initial Loader Specs

- The initial implementation loads batches of fiber-strip patches around random control points from the fiber dataset.
- Fiber JSON parsing follows the existing VC3D fiber parsing semantics from `vesuvius.neural_tracing.fiber_trace.fiber_json`.
- Each selected control point must be an exact member of `line_points`; otherwise the fiber JSON is rejected as inconsistent.
- The loader works on 2D sampled fiber side-strip patches.
- Neighboring strip-z context is represented as separate 2D patches.
- The default strip-z offset settings are `strip_z_offset_count=16` and `strip_z_offset_step=1.0`, generating `-7..8` selected-scale offsets and giving 16 patches per selected control point.
- Lasagna normals are used where needed to construct aligned strip frames.
- For a selected control point, the loader computes the CP-local line window that can affect the requested strip width plus interpolation/frame margin, and samples Lasagna normals only for that local window.
- Fiber endpoints or other distant line points outside the CP-local window must not be touched while loading that CP-local patch.
- If a line point inside the CP-local window cannot be sampled from the Lasagna manifest channels, the loader must not fabricate or propagate a replacement normal.
- During prefetch and training batch assembly, invalid CP-local samples caused by Lasagna channel data such as missing samples or in-bounds `grad_mag == 0` are skipped and reported, then the deterministic sample stream advances to the next sample.
- Fatal prefetch/training errors are infrastructure or programming failures such as missing APIs, broken bindings, interrupts, memory errors, or unexpected internal exceptions; those should stop the run rather than being hidden as data skips.
- VC3D side-strip/surface/segment sampling semantics define patch coordinates.
- Strip centerlines are sampled from all `line_points` with cubic Hermite interpolation over arc length; control points only select the strip anchor.
- The coordinate construction must be equivalent to VC3D side strips; flat planar patch simplifications are not acceptable except where they match the VC3D algorithm for that case.
- The implementation should reuse/export VC3D side-strip coordinate APIs when possible, or port the same algorithm with only small rounding/interpolation differences.
- Dense side-strip coordinate generation may use torch vectorization for per-pixel Hermite interpolation and normal interpolation, but it must preserve the existing VC3D/Lasagna frame construction semantics.
- Dense source-strip coordinates, strip-z offset coordinates, geometric coordinate augmentation, and transformed line/control-point coordinates stay as torch tensors on the configured augmentation device until an explicit NumPy consumer boundary.
- The explicit NumPy boundaries are VC3D coordinate sampling, runner/PIL visualization/export, and sample metadata arrays. The loader must not repeatedly convert coordinates between NumPy and torch inside one source/augmentation path.
- The augment-vis source/patch path is the canonical loader path for runner exports, training batch loading, and prefetch coordinate generation.
- Augmentation visualization, training, runner batch loading, and prefetch must share the same CP-local source-strip and final-coordinate generation implementation.
- Normal training, benchmark/profile/load-only, augment-vis, line-trace-vis,
  dir-vis, and direct runner/debug center-patch loading use the configured
  `augment_device` for torch coordinate generation. With the example config,
  `augment_device: "auto"` uses CUDA when available. Prefetch dependency
  generation is the exception and stays CPU-pinned.
- The loader builds CP-local source geometry once per selected control point and reuses it across augmentation variants or strip-z offsets as appropriate.
- When `strip_coord_cache_dir` is configured, CP-local source geometry is cached
  before strip-z offsets, coordinate augmentation, image sampling, or value
  augmentation are applied. Training, augment-vis, runner center/line/dir
  visualizations, and prefetch all use this cache through the shared
  source-strip path.
- Strip-coordinate cache entries include source-space line pixel coordinates
  and source-space control-point pixel coordinates. Unaugmented patches reuse
  those cached line/CP coordinates directly; augmented patches still compute
  transformed output line/CP coordinates per patch because they depend on the
  augmentation draw.
- Strip-coordinate cache identity is based on the base-volume path, selected
  scale and pixel spacing, strip-z offset step/center, control-point 3D
  coordinate, and fiber-line identity. The requested source size is stored in
  entry metadata rather than isolated into incompatible files so larger source
  entries can satisfy smaller later requests for the same identity.
- A strip-coordinate cache entry with height and width greater than or equal to
  the requested source size is a hit. The loader center-crops the cached source
  grid to the requested size. If a larger source is generated later, it
  atomically replaces the previous smaller entry.
- Strip-coordinate cache writes use unique temporary files in the cache
  directory followed by atomic rename. Corrupt or incompatible entries are
  treated as misses and regenerated through the normal Lasagna/strip-coordinate
  path. The current payload stores the zyx source coordinates, valid mask,
  zyx offset axis, source-space line/CP pixels, and strip frame; xyz arrays are
  derived from zyx when needed instead of stored redundantly. The cache key
  remains compatible with the previous source-cache identity so existing
  entries are reused where their payload version is supported.
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
- Config keys include `datasets`, `batch_size`, `patch_shape_hw`, `strip_z_offset_count`, `strip_z_offset_step`, `seed`, `loader_workers`, `prefetch_workers`, `prefetch_sampler_workers`, `volume_cache_dir`, optional `volume_cache_memory_mib`, optional `volume_io_threads`, optional `strip_coord_cache_dir`, and optional cache settings.
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
- Dataset entries include `fiber_paths` or `fiber_glob`, `base_volume_path`, `base_volume_scale`, and required `lasagna_manifest_path`.
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
  defaulting to the machine logical CPU count. Parallel workers may evaluate
  candidate samples out of order, but accepted output samples and skip handling
  must follow the same deterministic sample-index order as serial loading.
  `loader_workers=1` is the serial no-thread debug path. When
  `loader_workers > 1`, the loader reuses a persistent CP-level executor across
  batches instead of constructing a new thread pool per step.
- Parallel loader workers must not serialize on deterministic random-order
  locks during the warm path. Random dataset-pass orders are built once per
  pass, cached by pass index, and prewarmed for the attempted batch window
  before CP workers are submitted.
- The tester/runner loads a batch from a specified deterministic control-point sample index.
- Prefetch uses the same shared source-strip implementation as training and augment-vis.
- Prefetch remains CPU-pinned, but it still uses the same torch-native source-grid and strip-offset path, converting to NumPy only once for VC3D dependency discovery.
- Prefetch is independent of any one random augmentation draw: for each selected CP and strip-z offset it covers the configured maximum augmentation envelope represented by the oversized source-strip coordinates.
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
- Prefetch runs parallel dependency producers plus bounded chunk download workers; download worker count is controlled by `prefetch_workers` without an additional hard-coded cap, and dependency/sampler producer count is controlled by `prefetch_sampler_workers`.
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
- `training.max_sample_index` is an optional positive exclusive deterministic sample-index limit. The default `0` means no limit. When positive, training wraps every global sample position with `sample_index % training.max_sample_index`, so long runs reuse that deterministic prefix independently of `training.max_steps`.
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
- The default V0 direction model is a 10-block residual CNN with 64 hidden channels. It uses a 3x3 input projection, constant-width residual blocks, BatchNorm2d normalization, a final 1x1 direction projection, and an optional 1x1 embedding projection.
- V0 model output always starts with exactly two per-pixel direction channels in the Lasagna ambiguous two-cos-channel encoding. When `training.contrastive_embedding_channels > 0`, raw embedding channels are appended after those first two direction channels. Direction consumers must explicitly slice the first two channels.
- For strip-image tangent angle `theta`, target channels are `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`.
- Contrastive embedding training is enabled by `training.contrastive_enabled`.
  It requires `training.contrastive_embedding_channels > 0` and
  `training.control_points_per_step` divisible by
  `training.contrastive_control_points_per_fiber`.
- In contrastive mode, each training step loads a deterministic same-fiber CP
  group: `contrastive_control_points_per_fiber` CPs from one fiber are repeated
  to fill `control_points_per_step`. Group ordering is deterministic and covers
  the effective CP set by shuffled fiber-local CP groups before repeating.
- Repeated same-fiber CP patches keep independent geometric augmentation draws
  through unique raw sample indices. Value/image augmentation draws are
  synchronized across the repeated group so the embedding objective does not
  treat value-only appearance jitter as identity evidence.
- The contrastive embedding loss uses cosine similarity. Positive terms compare
  CP-neighborhood embedding samples from the same fiber and target cosine
  similarity `1`. Negative terms compare each CP-neighborhood embedding sample
  with one deterministic valid non-CP pixel from the batch and penalize cosine
  similarity above `training.contrastive_negative_margin`. Positive and
  negative means are averaged equally, then multiplied by
  `training.contrastive_weight`.
- Contrastive embedding visualization writes TensorBoard similarity maps:
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
- Training creates a run directory from `training.run_path` and `training.run_name` plus a date string. Passing `--resume <snapshot.pt>` creates and names a fresh run directory the same way, restores model and optimizer state from the snapshot, starts from `checkpoint_step + 1`, and keeps `training.max_steps` as the absolute target step. To continue past a finished run, increase `training.max_steps` before resuming. If two runs start in the same second, a numeric suffix is added to avoid a run-directory collision.
- Training config keys include `max_sample_index` for bounded deterministic-prefix reuse, `pipeline_enabled`, `pipeline_depth`, `pipeline_workers`, and `pipeline_isolated_loaders` for CUDA training load/model overlap, and `test_interval`, `test_control_points`, `test_start_sample_index`, `test_trace2cp_step_px`, and `test_trace2cp_rf_margin_px` for deterministic test evaluation when `test_datasets` is configured.
- Test evaluation runs at step 1, every `training.test_interval`, and the final step when `test_datasets` is configured. Positive `training.test_control_points` values load the fixed deterministic random range starting at `training.test_start_sample_index`, so the same held-out CP samples are compared across time. `training.test_control_points: 0` is the full-test sentinel: it evaluates every configured held-out CP sample once in flat CP order starting at zero, ignoring `training.test_start_sample_index`, so whole-fiber test metrics can be compared directly against `--trace2cp-vis --fiber-json` on the same held-out fiber apart from pair-alignment details. In addition to fixed-batch direction loss, the test path evaluates the public Trace2CP metric by tracing each selected held-out CP to its next CP segment and averaging valid `trace2cp_error` values.
- TensorBoard logging writes the training config JSON as text, direction-loss scalars, angular-error degree scalars, timing/cache diagnostics, and batch direction overlay images at configured intervals. Batch direction overlays show the transformed fiber centerline as context and one short network-predicted direction segment at the transformed CP; they do not draw CP-neighborhood supervision boxes or extra CP markers. Overlay contact sheets select examples across loaded control-point samples first, preferring each CP's strip-z offset closest to zero before showing additional offsets. When `test_datasets` is configured, TensorBoard also logs `test/loss_direction`, `test/angle_error_mean_deg`, `test/supervision_samples`, test cache diagnostics, and a `test/batch_direction_overlay` image at test evaluation steps.
- Console training progress prints every step covering the first 100 deterministic control-point sample indices, then falls back to `training.scalar_log_interval`.
- Prefetch progress includes `idx=<exclusive-index>` showing the largest contiguous exclusive bounded deterministic sample index whose required chunks are cache-complete: each required chunk is a cache hit, a known/new missing marker, or a completed successful download. Dependency generation alone must not advance `idx` while downloads are still pending. Operators can use that value as `training.max_sample_index` to train on the prefetched deterministic prefix.
- Training writes snapshots under `<run_dir>/snapshots/current.pt` and `<run_dir>/snapshots/best.pt`. With `test_datasets`, current snapshots are written at the test evaluation cadence and best is selected by lowest observed averaged `test/trace2cp_error`. Without `test_datasets`, current snapshots use `training.checkpoint_interval` and best is selected by lowest observed training loss. A resumed run writes its own fresh `current.pt` and `best.pt` under the newly created resumed run directory.
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
  height is twice the configured patch height so traces have more vertical room
  before entering the RF margin. It uses the same
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
- The public Trace2CP metric is `trace2cp_error`: the actual closest vertical
  separation between opposing traces divided by the horizontal start-to-target
  CP span. Both traces are resampled at one point per horizontal pixel over
  their overlapping x span. Trace2CP chooses the candidate x with the smallest
  actual y separation; ties choose the candidate closest to the midpoint only
  as a deterministic tie-breaker. No center-focus penalty is applied to this
  public metric.
- If the two opposing traces have no usable overlap, the public metric uses the
  default maximum y error for that segment: vertical distance from the CP
  centerline y to the nearest usable vertical strip edge after RF-margin
  exclusion, divided by the horizontal CP span. The same maximum y error caps
  pathological closest-gap values. This intentionally treats exact early/late
  edge intersection as noise for now.
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
- Trace2CP uses `--med-tta` to determine whether TTA is used. Without
  `--med-tta`, it traces and scores both directions on the base strip
  direction field. With `--med-tta`, it builds deterministic random geometric
  TTA direction fields using `--line-trace-tta-count`, default `100`, and
  traces both median-TTA directions in the reference segment strip.
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
  a concise stdout line with sample index, fiber path, start/target CP indices,
  trace mode, public `trace2cp_error`, metric raw y separation in
  pixels, horizontal CP span, refinement diagnostic score, endpoint diagnostic
  scores, per-direction raw errors, target x-columns, reach statuses,
  termination reasons, and trace point counts. The JPG is a labeled vertical
  stack with rows for full bidirectional traces, partial traces up to the
  closest point, the fused CP-to-CP line, and the optimized refinement. Without
  `--med-tta`, this stack is the reference-only inference result. With
  `--med-tta`, the JPG has two columns: the selected median-TTA result first,
  and a second reference-only inference column using the base direction field
  without TTA. It does not draw score text over image pixels.
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
- Trace2CP target-column crossing takes precedence over RF-margin rejection for
  the next step in each direction. If a step crosses that direction's target
  x-column and would also enter the RF margin, the trace is considered to have
  reached the target column and the score is computed by interpolation at the
  target column. RF-margin stop reasons should identify whether the x margin,
  y margin, or both were hit.
- Tests use fake/local arrays and monkeypatched readers where possible and must not require network access.
- `docs/code_structure.md` documents the current implemented module structure, data flow, config shape, runner outputs, and local workflow caveats; `planning/specs.md` remains the normative behavior source.
- Future changes that affect public config, data flow, sampling, caching, augmentation, runner outputs, tests, or local workflow must update both the relevant specs and code docs.
