# Changelog

## 2026-07-13

- Restored regular stepwise combined Trace2CP and stepwise z-search as the
  defaults, added explicit `--trace2cp-dp` routing for the monotone-DP backend,
  and made `--trace2cp-side-top-z-experiment` an exclusive export mode.
- Changed Trace2CP DP direction alignment to an angle-space penalty where the
  existing candidate max angle is the excess knee.
- Changed side/joint Trace2CP DP z regularization to use dz smoothness instead
  of a per-step z transition penalty.
- Added throttled progress-bar output for the forward/backward
  `--trace2cp-side-top-z-experiment` traces.
- Kept side/top-z per-step top-direction overlays in
  `trace2cp_side_top_z_top_overlays/` while leaving the compact experiment JPG
  tick-free.
- Added opt-in single-pair `--trace2cp-side-top-z-experiment`, which keeps
  regular Trace2CP as the default, uses regular side candidate scoring for
  side x/y motion, and writes separate side/top z-offset diagnostic artifacts
  using one top-model inference per accepted trace point.
- Added side/top-z experiment top-slice debug exports:
  `trace2cp_side_top_z_top_slices/` and
  `trace2cp_side_top_z_top_overlays/`.
- Added a torch-vectorized Trace2CP monotone-DP backend for CLI side/z/top
  solves, with side-z reachable-layer pruning.
- Changed side-strip Trace2CP joint DP to use fixed 4 px transitions, removed
  candidate-angle-derived vertical move pruning, disabled default second-order
  smoothing, and switched transition scoring to fractional row/z interpolation
  with ambiguous direction sign alignment.
- Added Trace2CP stage timing tables for single-pair and whole-fiber
  visualization commands, and changed side-strip joint DP tracing to use a
  local candidate-angle excess penalty rather than a global slope cap.
- Added time-throttled `trace2cp dp ...` progress rows with ETA for slow
  Trace2CP side/z/top dynamic-programming solves.
- Replaced active Trace2CP combined tracing with a joint side-strip monotone DP
  path over direction plus optional presence, including z-layer DP search.
- Added second-order `dy/dz` smoothness to the visualization-only
  `--trace2cp-top-model-dir-vis` yellow DP path, with debug labels for the
  smoothing weights.
- Changed the `--trace2cp-top-model-dir-vis` yellow DP path to use
  `(top_offset_layer, y)` state over raw per-layer top-model directions, with a
  small z-transition penalty and layer-range debug output.
- Added a visualization-only monotone-x dynamic-programming CP-to-CP path over
  the fused `--trace2cp-top-model-dir-vis` direction field, with summary debug
  for top-trace stop reasons and a soft penalty for invalid direction pixels.
- Updated that top-model DP path to use fixed 8 px horizontal transitions and
  integrated direction error over each transition to reduce late vertical
  jumps.
- Changed `--trace2cp-top-model-dir-vis` top-direction fusion to use a
  sign-aligned median of offset-layer directions within 45 degrees of
  image-horizontal, and made the reverse top trace equally visible.
- Updated `--trace2cp-top-model-dir-vis` to infer a `-4..+4` selected-voxel
  top-strip offset stack and add two visualization-only top traces through the
  fused top-direction field.

## 2026-07-12

- Added `--trace2cp-top-model-dir-vis` to append top-view model direction
  indicators over the traced fused top strip in Trace2CP visualizations.
- Added opt-in iterative Trace2CP refinement with
  `--trace2cp-refine-iterations`: extra passes smooth the previous fused trace,
  resample a fresh volume-backed side strip from that curve, and export `itN`
  visualizations/summaries.
- Changed Trace2CP refinement smoothing from a moving-average kernel to a
  finite Gaussian kernel while preserving x columns and CP endpoints exactly.
- Fixed refined Trace2CP pass source construction so the synthetic line keeps
  endpoint context before and after the CP pair; reverse traces in `it1+` now
  start from the target CP with a valid local direction neighborhood.
- Added `--trace2cp-z-layers-tif` for Trace2CP z-search runs, exporting the
  already inferred z-layer cache as non-interleaved multilayer TIFF stacks:
  sorted sampled slices first, then sorted presence maps.
- Increased Trace2CP segment strip height from four to eight times the
  configured patch height, giving visualizations/traces more vertical room
  before RF-margin or edge stops.
- Changed training augmentation seeding so bounded-prefix runs still select CPs
  through `training.max_sample_index`, but geometric and value/image
  augmentations are seeded by the unbounded deterministic training stream index
  and therefore do not repeat when the bounded CP prefix wraps.
- Fixed `fiber_trace_2d` prefetch `idx` accounting so it advances only after
  all side/top dependency requests for a deterministic shuffled-stream sample
  are classified and resolved; docs now clarify that `idx` is the same random
  stream prefix consumed by `training.max_sample_index`, not a flat CP id.
- Refactored top-view batch loading to match side-view loader accelerations:
  top source grids now use separate strip-coordinate cache entries, top coord
  augmentation uses the same batched map/tensor path, top images are grouped
  through `sample_coord_batch`, and top patches reuse the already transformed
  side-sample line/CP pixel coordinates.
- Changed `fiber_trace_2d` prefetch scheduling so missing chunk downloads that
  are not yet active are prioritized by the earliest raw deterministic sample
  index that needs them, helping the reported cached-prefix `idx` advance
  sooner during top-view-inclusive prefetch; prefetch also temporarily caps
  PyTorch CPU intra-op generation fanout so `prefetch_sampler_workers` is the
  practical source/dependency generation limit.
- Added optional joint top-view training for `fiber_trace_2d`: a second
  direction model consumes VC3D-style top-strip slices and trains a sigmoid
  distance-transform channel along the CP normal line, with TensorBoard
  overlays/maps and prefetch dependency inclusion.
- Changed standard `fiber_trace_2d` training to direction plus sheet/fiber
  presence only: contrastive embedding is now explicit opt-in, and disabled
  contrastive configs instantiate no embedding head.
- Added Trace2CP top-strip visualization sampled as VC3D `lineSurface`,
  including original/init comparison, traced fused central-z output, and fused
  z-corrected output when z-search is active.
- Changed combined Trace2CP CLI/scoring so candidate fan tracing defaults to
  direction-only, embedding and image similarity are explicit opt-in modes, and
  `--trace2cp-use-presence` can add sheet/fiber presence maximization to any
  combined mode.
- Added optional sheet/fiber presence training for 2D fiber strips: model
  outputs can now include one sigmoid presence channel after the two direction
  channels, trained with balanced CP-positive and reachable non-CP negative BCE
  while ignoring unreachable shift-margin edges.
- Changed contrastive embedding positives to be z-search-aware CP matches:
  each anchor CP sample/strip-z offset now trains only against the already
  most-similar other CP from the same fiber across loaded offsets, instead of
  all same-fiber CP-neighborhood pairs.
- Fixed Trace2CP target-directed traces stopping short under the old diagonal
  step budget: target-column crossings now append an exact interpolated point,
  `max_steps` exhaustion raises visibly instead of being scored, and
  single-pair stdout prints the selected public `trace2cp_error` once on its
  own line.
- Added experimental `--trace2cp-combined-mode image` for Trace2CP
  visualization: candidate scoring can now compare oriented, fiber-axis blurred
  image descriptors against the start CP, target CP, and previous trace point,
  including z-search support without requiring embedding channels.
- Changed regular non-z combined Trace2CP direction scoring to match z-search:
  candidates now average current-point direction agreement and candidate-point
  direction agreement before applying the existing embedding terms.
- Refined experimental Trace2CP z-search so lazy layers are derived from one
  center segment source by per-pixel strip offset axes, changed the z-step
  default to one selected-scale voxel, and made candidate scoring use both the
  current-point and candidate-point direction fields.
- Added experimental `--trace2cp-z-search` for combined Trace2CP inspection:
  lazy center/neighbor strip-offset plane inference, z-aware combined
  candidate scoring, y+z closest-approach fusion, and single-pair
  forward/reverse/fused z-corrected visualization columns plus a fused z-layer
  map row.
- Changed the Trace2CP embedding-debug forward/reverse last-similarity panels
  to paint per-trace-step column bands from the previous accepted point's
  embedding instead of showing one full-image map against the final trace
  embedding.
- Removed the cross-fiber CP negative component from contrastive embedding
  training; negatives are again only deterministic valid non-CP pixels, while
  same-fiber positives and the reachable-area similarity-mean sparsity term
  remain active.

## 2026-07-11

- Added a contrastive embedding sparsity term that trains each CP's normalized
  shift-reachable similarity-image mean toward `0.1`, encouraging CP-similar
  embeddings to occupy fewer reachable pixels.
- Added cross-fiber CP embedding negatives to contrastive training and changed
  contrastive batches to concatenate same-fiber CP groups; CP-vs-other-fiber-CP
  negatives now share the aggregate negative branch with existing valid-pixel
  negatives when multiple fibers are present.
- Added an optional embedding-similarity debug column to single-pair
  `trace2cp_vis.jpg`, showing fixed-scale cosine maps for both CPs,
  same-fiber/global CP-bank similarity, and forward/reverse trace-last
  embeddings when embedding outputs are available.
- Increased Trace2CP segment strip height to four times the configured patch
  height so tracing has more vertical room before RF-margin or edge stops.
- Switched public Trace2CP `trace2cp_error` back to target-column y error per
  horizontal CP span for all trace modes while keeping closest-approach logic
  for fusion/refinement diagnostics.
- Added optional `--trace2cp-combined` visualization mode that greedily scores
  angular trace candidates with direction plus contrastive embedding terms,
  including previous-step, enclosing-CP, and same-fiber CP-bank cosine losses.
- Added optional cosine contrastive embedding training for 2D fiber strips:
  same-fiber grouped CP batches, appended embedding output channels, balanced
  positive/negative embedding loss, and TensorBoard CP-similarity maps.
- Restricted contrastive embedding negative candidates to the CP-neighborhood
  region reachable under configured shift augmentation so unreachable patch
  edges are ignored instead of learned as permanent negatives.
- Made `training.test_control_points: 0` evaluate all configured held-out CP
  samples once in flat order, so training `test/trace2cp_error` can cover the
  same segment set as whole-fiber Trace2CP visualization.
- Added `--resume <snapshot.pt>` to continue fiber_trace_2d training from an
  existing model/optimizer snapshot into a fresh timestamped run directory, and
  made whole-fiber Trace2CP stdout print the public metric on its own
  `trace2cp_error_mean=...` line.
- Changed public Trace2CP reporting to `trace2cp_error`, the closest actual
  vertical trace gap divided by horizontal CP span; the center-biased score is
  now only a refinement/visual diagnostic, and test-dataset training selects
  `best.pt` by averaged `test/trace2cp_error`.
- Added `--trace2cp-vis --fiber-json <path>` to run Trace2CP over all
  configured CP pairs for one explicit fiber and compose a long-strip
  visualization without requiring that fiber to match the configured
  `fiber_glob`; invalid CP-pair segments are skipped and listed in the
  summary, adjacent pair strips align actual strip row axes at shared CPs to
  avoid random y flips, and `trace2cp_fiber_debug.txt` records per-pair strip
  CP vectors.
- Added a reference-only comparison column to `--trace2cp-vis --med-tta`
  output.
- Added a center-bias penalty to Trace2CP closest-approach scoring so candidate
  gaps at either CP x-coordinate count twice as much as gaps at the midpoint.
- Changed Trace2CP's public score to the closest vertical approach between
  opposing traces and added fused/optimized CP-to-CP refinement rows to the
  visualization.
- Switched the default 2D fiber-strip direction model normalization from
  GroupNorm to BatchNorm2d.
- Replaced Trace2CP/line median-TTA reference-point lookup with direct sampling
  of the prebuilt reference-to-TTA map, removing dense nearest-grid scans over
  TTA coordinate images during tracing.
- Replaced binned Lasagna two-channel direction decoding with the analytic
  inverse, removing the large `pixels * bins` temporary allocation in tracing
  and visualization paths.
- Removed image-space geometric TTA warps from `fiber_trace_2d`, switched
  line/Trace2CP TTA to coordinate-sampled volume patches, and added
  `--trace2cp-vis --med-tta --vis-tta` per-TTA slice debug exports.
- Made `--trace2cp-vis` trace selected CP segments in both directions, draw both
  traces, and report the average of the forward/reverse normalized scores.

## 2026-07-10

- Tuned the 2D fiber-strip load-only pipeline to use shared-loader whole-batch
  futures plus bounded CP-prep workers, added optional VC3D sampler cache/I/O
  controls, and documented measured loader throughput variants.
- Added `--dbg-dirs` for 2D `--dir-vis`, adding a half-image pasted-center
  debug row to probe local direction evidence against transformed patch context.
- Added explicit image-space flip/90-degree-rotation panels to 2D `--dir-vis`
  and writes them as one labeled direction-overlay contact sheet.
- Switched 2D `--dir-vis` direction overlays to 8x8 display-pixel cells with
  6-pixel anti-aliased direction segments.
- Added real process CPU timing to the 2D fiber-trace benchmark/profile table
  so loader summed-worker timing can be compared against actual CPU usage.
- Parallelized the 2D CUDA training load+prepare path by submitting exact training steps to concurrent workers, added one-offset strip-cache and CP-tangent/no-coordinate-retention fast paths, and documented measured throughput limits.
- Removed the 2D training default patch-count warning, added a real
  `batch_size`/`control_points_per_step` validation error, and switched
  variable-sigma value-augmentation blur to the measured unfold-based batched
  implementation.
- Set the measured default CUDA training pipeline queue to 8 batches, default
  whole-batch loader workers to 4, added a startup print for effective
  pipeline settings, and removed invalid-sample skip reason spam from hot
  training loads.
- Added configurable concurrent whole-batch CUDA training loaders and moved
  CUDA preparation submission into a background preparation executor.
- Added CUDA side-stream training preparation for deferred image/value
  augmentation, normalization, and supervision tensors, plus prep/outside
  timing diagnostics.
- Added `--trace2cp-vis` runner inspection for CP-pair segment tracing,
  optional median-TTA scoring, normalized trace-to-target score output, and
  single-panel JPG/summary export.
- Added CUDA training batch pipelining with bounded deterministic whole-batch
  futures, deferred torch value augmentation, and profile wait timing.
- Removed avoidable warm-path loader threading overhead by prewarming cached
  deterministic random pass orders and reusing a persistent CP worker executor
  across `load_batch` calls.
- Slimmed strip-coordinate cache reads/writes by using zyx source coordinates
  and deriving xyz tensors on load while preserving the existing cache key
  identity for supported entries.
- Batched 2D strip-z image loading through `CoordinateSampler.sample_coord_batch`
  and added `loader_workers` CP-level parallel `load_batch` construction with
  deterministic accepted-sample ordering.
- Added 2D loader profile wall/work/threading-factor reporting so parallel
  loader timings are not confused with summed worker timings.
- Batched 2D strip training sample augmentation across strip-z offsets:
  sparse line/CP lookup now uses direct batched bilinear gather, coordinate
  augmentation stacks fused maps, and post-load value augmentation runs on the
  loaded image stack.
- Replaced formula-based 2D strip point mapping with concrete cached
  `backward_map_xy`/`forward_map_xy` tensors so coordinate augmentation and
  line/CP mapping use the same fused geometric augmentation maps.
- Cached fused 2D strip augmentation transform constants/smooth controls,
  batched line+CP mapping, and reused transform/line mappings across matching
  strip-z offset patches.
- Added paired forward/backward strip augmentation transforms and routed
  smooth line/CP mapping through direct source-coordinate forward/backward
  point transforms instead of iterative or dense nearest-grid inversion.
- Cached source-space line/CP coordinates in strip-coordinate cache v2 and split coordinate profiling into descriptor/cache/source/line columns.
- Added configurable CP-local strip-coordinate caching shared by training, visualization, and prefetch source-grid construction.
- Kept 2D fiber-strip source grids, strip-z offsets, coordinate augmentation, and line/control-point transforms torch-native until explicit VC3D/export NumPy boundaries.
- Added opt-in `--med-tta` runner line tracing that uses per-step median directions across reference and fixed TTA direction fields.
- Added 100-batch `train.py --benchmark`, `--profile`, and `--load-only` modes for patch-throughput, per-stage timing, and loader-only diagnostics.
- Switched the default 2D fiber-strip direction model to a 10-block, 64-channel residual CNN.
- Switched the default 2D direction ResNet normalization to 8-group GroupNorm.
- Added `--dir-vis` runner direction-field inspection for checkpointed per-pixel direction predictions.
- Added the V0.1 runner line-tracing inspection mode for tracing checkpointed direction predictions on one deterministic side-strip patch.
- Added fixed test-time augmentation flock visualization to the V0.1 line-trace runner output.
- Added folded unoriented direction angle-error reporting in degrees for 2D fiber-strip train/test output.
- Fixed prefetch `idx` progress semantics so it reports the cache-complete safe sample prefix rather than dependency-generation progress.
- Added `prefetch_sampler_workers` to tune dependency producer concurrency separately from download worker concurrency.
- Added `training.max_sample_index` for bounded deterministic-prefix reuse and prefetch `idx` progress reporting.
- Restored 2D fiber-strip training/prefetch to deterministic pseudo-random full-dataset CP passes instead of flat sequential CP order.
- Switched 2D fiber-strip training/prefetch to flat CP order and made explicit `--prefetch-steps` override configured `training.max_steps`; `--prefetch-steps 0` now covers every configured training/test CP once.
- Added 2D fiber-strip `test_datasets` evaluation with deterministic held-out batches and test-loss current/best snapshot cadence.
- Added `--augment-vis` contact-sheet CP crosshairs and separate label bands so labels no longer cover image pixels.
- Fixed 2D fiber-strip affine augmentation composition so shift is applied in output/scaled space and image sampling stays aligned with line/control-point coordinates.

## 2026-07-09

- Tightened 2D fiber-strip prefetch so VC3D only reports dependency metadata while Python handles direct-source uncompressed chunk downloads, atomic cache writes, zero-byte `.empty` markers, and retry/progress behavior.
- Replaced 2D fiber-strip prefetch sampling/discarding with dependency-only chunk discovery, Python-side VC3D cache classification, `.empty` missing markers, bounded parallel fetching, and compact progress output.
- Unified 2D fiber-strip training, augment-vis, and prefetch around the shared torch-vectorized source-strip path; prefetch covers the configured augmentation envelope through dependency-only chunk discovery.
- Added training-oriented 2D fiber-strip prefetch mode with `--prefetch`, `--prefetch-steps N`, and `--prefetch-steps 0` for all configured steps.
- Added the V0 2D fiber-strip direction training path with Lasagna ambiguous two-cos-channel targets, CP-local supervision, TensorBoard logging, and current/best snapshots.
