# 2026-08-13: fiberlet graph replay

- Replaced the quantized world-axis half-grid DP with a deterministic curved
  cubic-Hermite domain using 2-voxel arclength planes, parallel-transported
  frames, 0.5-voxel transverse offsets, and floating-point interpolation.
- Removed the fiberlet lattice's 45-degree smoothness dead zone and scored
  graph joins with the same shared local alignment and Lasagna-normal
  tangent/normal objective, including exactly-once replay accounting.
- Expanded integer-DP candidate generation from the radius-four shell to every
  shorter cell offset within the same outer bound and added a strict 25-degree
  sampled-fiber-direction feasibility constraint.
- Added deterministic bidirectional fiberlet graphs with strict 45-degree
  anchor joins and loss-density beam routing with anchor-level lookahead.
- Added `fiberlet-replay`, shared variable-step reference matching, strict
  graph/route artifacts, and an independent reloadable napari route layer.
- Unified reference, greedy, and fiberlet longitudinal comparison scope under
  `--along`; greedy postroll and exact display cropping now derive from it.
- Fiberlet graph replay now completes the edge containing its first distance
  failure and continues for anchor-bounded `--along` postroll, with explicit
  complete/truncated distances, overshoot/shortfall, and whole-edge identities.

# 2026-08-12: narrower independent anchor NMS

- Decoupled transverse anchor NMS from the peak-refinement window and reduced
  defaults to 2 transverse and 1 longitudinal prediction voxels.
- Added the transverse radius to strict experimental anchor artifacts and path
  loading while retaining conservative external crop context.

# 2026-08-12: fiberlet replay distance filtering

- Added an independent napari fiberlet-radius control that physically removes
  paths wholly outside the exact reference/failed-trace polyline distance.
- Set display defaults to 32 base voxels for presence and anchors and 16 for
  fiberlets, with all three values preserved across artifact reload.

# 2026-08-12: parameter-independent anchor visualization

- Decoupled napari anchor-stage visualization compatibility from extractor
  parameters. Old, absent, or extended parameter metadata now renders without
  affecting coordinate, geometry, lineage, or final-output consistency checks.

# 2026-08-12: full 2D anchor subpixel experiment

- Replaced independent 1D peak parabolas with a complete 3x3 least-squares 2D
  quadratic including cross-coupled curvature and conservative Hessian,
  half-step, owner-domain, and real-response guards.
- On the fixed David Paris4 benchmark, the 2D fit reproduced the discrete
  population exactly but scored below the 1D baseline: 29.07%/43.90% versus
  29.78%/44.88% anchor/cell hits at 4 vx, and 54.09%/80.42% versus
  54.59%/81.06% at 8 vx.
- Extending the signed-gradient sweep above its former maximum found a
  4-voxel optimum near weight `1.0` and an 8-voxel optimum near `1.3-1.5`;
  weight `1.2` gave the best equal-weight aggregate for joint-2D.
- Added matched `discrete`, `separable_1d`, and `joint_2d` benchmark positions
  from one extraction. The old `0.2` 1D result reproduced exactly. Across the
  complete `0.0-2.0` sweep, separable-1D beat joint-2D on every 4/8 anchor/cell
  hit rate; separable weight `1.1` scored 31.14%/46.89% at 4 vx and
  55.97%/82.64% at 8 vx.
- Selected separable-1D for production anchor placement and gradient weight
  `1.0` as the default. Joint-2D remains transient benchmark provenance only.

# 2026-08-12: discrete-versus-subpixel anchor benchmark

- Split refined-anchor localization output into equal-population `discrete`
  and `subpixel` reports while keeping discrete provenance transient and strict
  version-1 artifacts unchanged.
- Selected `0.2` as the default signed gradient weight from the fixed David
  Paris4 sweep. Explicit CLI and artifact values remain authoritative.
- At `0.2`, subpixel fitting improved 4-base-voxel anchor/cell hit rates from
  27.79%/41.94% to 29.78%/44.88% and 8-base-voxel rates from 53.76%/79.86% to
  54.59%/81.06% relative to the discrete peaks.

# 2026-08-12: signed presence-gradient anchor centering

- Added deterministic presence-only 3D Sobel gradients and reliability-gated
  inward/outward normal-plane voting to the refined-anchor peak objective.
- Added strict gradient weight/reliability artifact parameters and
  `--gradient-weight`, with exact weight-zero fallback.
- On the supplied David Paris4 reference, the default raised 4-base-voxel
  anchor/cell hits from 1,002/982 to 1,021/1,000 and 8-base-voxel hits from
  1,884/1,823 to 1,896/1,832 while reducing mean, median, and p95 distance.

# 2026-08-12: outer-parallel anchor extraction

- Parallelized anchor sampling and fitting over canonical cell jobs while
  forcing every job's lower-level prediction sampler to one thread.
- Added aggregate concurrent-halo memory bounding, deterministic indexed result
  and error assembly, and serialized progress/retain handling.

# 2026-08-12: refined-anchor localization benchmark

- Added `vc_fiberlets anchor-benchmark` to extract only geometric refined
  anchors in exact reference-intersecting cells and measure their exact
  base-coordinate distance to a strict fiber's dense line.
- Added stable distribution output and inclusive 4/8-base-voxel anchor and
  reference-cell hit rates, with empty refined populations kept explicit.
- Shared exact polyline-to-cell selection with failed-trace replay while
  preserving replay's positive-radius tube behavior.

# 2026-08-12: physical replay anchor filtering

- Changed the Napari replay anchor-radius cutoff to physically subset final and
  staged anchors, cell centers, and refinement offsets instead of leaving
  transparent depth-occluding geometry in the rendered layers.
- Retained defensive full geometry/features for reversible slider updates and
  transactional artifact reload/rollback.

# 2026-08-12: local-maximum fiberlet anchors

- Added deterministic direction-conditioned normal-plane peak search after the
  existing non-orthogonal anchor direction fit.
- Integrated the peak response with a `1.5`-voxel transverse Gaussian and a
  longer `1.5`-cell-side along-direction Gaussian, with complete rotated-kernel
  halo sizing.
- Added bounded non-decreasing subvoxel peak fitting, continuous owner-cell
  constraints, normalized edge response, strict transverse/axial peak fields,
  and base-voxel CLI controls.

# 2026-08-12: faster replay viewer startup

- Kept detailed anchor-stage layers enabled by default so replay diagnostics are
  immediately available; `--no-anchor-stages` provides an explicit fast-start
  opt-out and avoids hashing unused stage artifacts.
- Preserved strict stage schemas and selected-stage topology across artifact
  reloads.

# 2026-08-11: dense-fiber failure replay

- Added C++ greedy native replay against dense VC3D fiber geometry with bounded
  monotone matching, typed failure statuses, and exact postroll handling.
- Added exact tube-selected anchors, tube-constrained integer fiberlets, sparse
  replay scoring preload, and content-addressed atomic diagnostic bundles.
- Added strict napari replay loading with external-Zarr validation and separate
  reference, trace, failure, anchor, and quality-colored fiberlet layers.
- Kept replay failure-marker styling compatible across napari Points API
  versions by avoiding the renamed outline-color keyword.
- Added a replay presence-tube mask using a one-time reference rasterization and
  base-voxel distance transform, with a runtime radius slider defaulted from the
  anchor/fiberlet extraction tube.
- Added replay stage timing plus per-second anchor and fiberlet progress/ETA;
  anchor progress distinguishes selected-cell fitting from NMS context.
- Expanded the replay presence mask to the union of the reference fiber and the
  complete failed greedy trace, including postroll, for direct comparison.
- Added `anchor_cells.obj` with every selected cell center and retained-anchor
  displacement lines, exposed as separate clipped napari diagnostic layers.
- Added strict initialized/refined/support/selection/NMS anchor-stage JSON
  diagnostics with stable merge lineage, exact filter causes, and actual NMS
  suppressors, plus independently toggleable napari comparison layers.
- Added an independent base-voxel anchor-radius slider that filters every replay
  anchor diagnostic against the exact reference/failed-trace union distance.
- Added transactional in-process replay-artifact reload with stable empty
  layers, strict prediction/crop compatibility, derived-distance refresh, and
  no presence-Zarr reopening.

# 2026-08-11: fiber-centered anchor refinement

- Replaced fixed cell-centroid anchor positions with bounded halo-backed joint
  direction/position refinement on a rotating plane through each cell pivot.
- Added deterministic direction-compatible local-maximum NMS across cells,
  exact crop-boundary context, strict refinement metadata, and focused
  convergence, locality, determinism, and artifact validation.

# 2026-08-11: fiberlet loss/quality visualization

- Added length-normalized per-fiberlet trace loss, report-relative visual
  quality, strict napari metric parsing, per-shape features, a runtime napari
  colormap selector, and density CLI statistics. Path MTL/material output was
  removed in favor of napari-owned display color.

# 2026-08-10: integer-DP fiberlet paths

- Added strict anchor-artifact loading, radius-four cell-shell pairing, and
  deterministic integer prediction-voxel DP with exact virtual endpoints.
- Added the native tracer's shared multiplicative local alignment loss with
  lattice-edge integration, finite invalid-prediction bridges, and shared
  native Lasagna-aware direct curvature.
- Added `vc_fiberlets paths`, versioned JSON/OBJ output, focused regression
  tests, and real-crop deterministic validation. Global graph construction,
  deduplication, extension, and H/V/winding assignment remain deferred.
- Made base-volume coordinates the only spatial contract exposed by the
  unshipped fiberlet CLI and JSON/OBJ artifacts; no compatibility aliases or
  legacy coordinate fields are retained.
- Added `paths --stats`, explicit score-presence semantics, and two-index OBJ
  path edges.
- Added independently loadable central XY/XZ/YZ fiber-presence textured-quad
  OBJ/MTL/PNG context, with direct presence sampling and `--no-slices` opt-out.
- Materialized each small crop's fiber/normal scoring volume once and moved
  exact independent DP searches to a deterministic fixed worker pool, removing
  repeated sampling and nested thread teams.
- Added monotonic rate-limited path progress with throughput and ETA on stderr.

# 2026-08-10: C++ fiberlet cell anchors

- Added deterministic cell-based extraction of zero, one, or two
  non-orthogonal unoriented anchors from canonical Fiber Lasagna
  `presence/nx/ny`, plus the cache-aware `vc_fiberlets anchors` command and
  sparse JSON/base-coordinate OBJ artifacts. Connection and path stages remain
  deferred.
- Merged near-duplicate fitted directions using an angle-plus-objective test
  and a joint PCA refit, with strict diagnostics and configurable thresholds.
- Kept the complete anchor OBJ and added separate deterministic component-zero
  and component-one OBJ layers for inspection.

# 2026-08-03

- Added experimental float16 shared raw-product accumulator rings while
  retaining the measured-faster float32 default, float32 weights, and float32
  flush arithmetic.
- Moved shared inference integer-to-FP32 normalization onto CUDA after compact
  H2D and made Fiber model autocast default to checkpoint training policy.
- Added opt-in shared Fiber/Lasagna multi-device loader and worker stage
  profiling with direct reader-concurrency diagnostics.
- Overlapped shared Lasagna/Fiber output flushing with inference using one
  enlarged bounded circular mmap and one runner-wide asynchronous flush,
  without band-sized RAM snapshots or overlap copies.
- Made downloader negative-remote caches tolerant and atomic, and exposed the
  automatic S3 transfer worker count in Fiber and Lasagna inference CLIs.
- Added shared bounded streaming multi-GPU whole-volume inference for Lasagna
  and Fiber 3D, with separate CPU/Zarr prefetch, persistent per-device workers,
  shared-memory input/results, and canonical single-writer accumulation.
- Made Fiber 3D per-rank hang logs, manual dumps, test watchdog, resource
  polling, and diagnostic CUDA synchronization explicitly opt-in through config
  or CLI.

# 2026-08-02

- Fixed cropped shared 3D inference treating globally positioned output chunks
  as unsupported when their origins exceeded the crop-local accumulator shape.
- Fixed circular accumulator depth underplanning across initial chunk-aligned
  no-op flushes in cropped, downscaled shared inference.
- Distributed dense Fiber 3D tests deterministically across DDP ranks with
  persistent process-worker prefetch, exact ordered metric reconstruction, and
  stdout/TensorBoard total-test timing.

# 2026-08-01

- Added persistent per-rank Fiber 3D training diagnostics and an eight-minute
  rank-0 test watchdog with per-batch phase/resource markers and manual
  `SIGUSR2` stack dumps.
- Fixed Vesuvius Python CI collection by exposing the monorepo's shared
  `lasagna` source namespace and triggering the workflow for Lasagna changes;
  repaired the Zarr 3.2.1 matrix with explicit cross-version v2 fixtures and
  shared Zarr 2/3 chunk-key and raw-store access.

# 2026-07-31

- Repaired Atlas pred-snap tests for strict per-channel 3D Lasagna arrays,
  made Fiber 3D inference include ceil-sized odd OME edge planes, and made VC3D
  reject incompatible independently attached Lasagna source volumes.
- Made native mode effective during new-fiber seed creation: configured fiber
  inference now chains the internal Lasagna reference solve directly into the
  existing single-CP native extrapolator without displaying or saving the
  intermediate Lasagna geometry.
- Made every v3 fiber reader reject missing or malformed mode/segment metadata
  without repair, completed v3 geometry consumption in Atlas, Lasagna, and
  Spiral, made sync route invalid v3 inputs to manual conflict handling, fixed
  NML/direct Python construction, and made Lasagna probe optimization outputs
  carry coherent regenerated v3 span metadata.
- Merged the main VC3D line-annotation toolbar, tag, schematic-overview,
  in-place refresh, and pane-lifecycle work while retaining both interactive
  rendered fiber strips and their per-span mode/metric/message labels.
- Removed the unpublished top-level `vc3d_fiber` version 2 and its obsolete
  descriptor migrations from VC3D, shared C++ readers, Python training input,
  and sync validation. Legacy file version 1 and current version 3 remain;
  version 3's `tracer_version: 2` is unchanged.
- Made fiber-aware sync merge version-3 dense span geometry and CP-owned
  descriptors atomically, preserve separated local/remote edits across an
  unchanged span, merge the global mode base-aware, and route every adjacent,
  overlapping, or unalignable result to the existing manual conflict workflow.
- Persisted the actual Lasagna and fiber-inference manifest identities consulted
  per interpolation span, using public manifest URLs instead of local cache
  paths for open-data catalogue Lasagna.
- Changed newly created VC3D fibers to default their global interpolation and
  extrapolation mode to the native fiber tracer while retaining Lasagna for
  older files with no persisted mode.
- Added `vc3d_fiber` v3 per-span interpolation goals and actual modes, grouped
  cubic-spline interpolation, trace-to-Lasagna-to-spline fallback, the
  global-only 100-base-voxel shortcut, persisted metrics/messages, a checked
  Ctrl-right-click goal selector, and viewport-packed `C`/`L`/`T` span labels.
- Restricted persisted meeting diagnostics to accepted native spans; fallback
  records now retain only failure code/detail, and all readers ignore stale
  fallback meeting values so earlier project files load cleanly.
- Made native VC3D fiber extrapolation completion depend only on its nominal
  `ceil(distance / step)` generation budget, without target planes,
  `max_step_factor`, or measured arc-length acceptance.
- Added a 10-base-voxel floor to native CP-pair meeting acceptance, which now
  uses `max(10 base voxels, 10% of combined partial traced length)`.
- Added VC3D terminal warnings whenever native line-annotation tail
  extrapolation retains its Lasagna fallback, including the side, full
  trace/exception reason, returned point count, and failure source.

# 2026-07-30

- Preserved native span error labels through Reoptimize/branch refresh and made
  native extrapolation stop at the last valid prediction before a volume edge
  instead of restoring the Lasagna tail.
- Replaced VC3D's endpoint-only native CP-pair acceptance with symmetric
  moving-plane trace intersection, 10%-of-traced-length acceptance, and
  arc-length-warped fusion; persisted accepted/fallback outcomes now restore
  meeting errors or stable failure labels after reload.
- Hard-constrained every Lasagna fallback span and retained tail adjacent to
  native fiber geometry to continue the fitted dense native endpoint tangent,
  using fixed proxy points in the regular Ceres solve, independent of normals,
  candidate selection, seed choice, or solve order.
- Removed previous Lasagna geometry and endpoint directions from full
  reinitialization candidates, made solved-neighbor directions exclusive
  rollout sources, and made degenerate normal-plane transport choose a
  perpendicular tangent instead of preserving a normal-parallel direction.
- Fixed Fiber-model mode and extrapolation-distance changes to rebuild both
  tails on newly seeded one-control-point fibers when Auto-reoptimize is active.
- Added persisted fiber-global Lasagna/native modes to VC3D line annotation,
  including full rebuilds on mode changes, invalid-span native retracing with
  per-span Lasagna fallback, trained-neighbor continuation directions, and
  configurable native/Lasagna tail extrapolation.
- Ported Python's target-local multi-plane intersection behavior to the shared
  C++ tracer used by VC3D and `vc_fiber_trace_metric`, including per-beam
  recrossing state, all-plane termination, threshold-aware whole-fiber
  continuation, and selected-crossing diagnostics.
- Further reduced the approved native precomputed whole-fiber workload from
  1.869s wall / 8.222s CPU to a three-run median of 0.986s / 5.134s with
  deterministic partial parent ordering, compact lazy frontiers, unique-cube
  corner reuse, unit-vector invariant math, unique-key start sampling, indexed
  worker batches, and compact candidate task storage. All final runs retained
  7 restarts and the exact 6,910,839-candidate / 4,318-generation workload.
- Reduced the approved native precomputed whole-fiber workload from 21.155s
  to 1.869s with exact lower-bound lookahead ordering, a measured default
  32-parent final-lookahead cap, and fused pinned-corner decode/scoring. The
  retained result has 7 restarts versus the 8-restart baseline; exact lazy and
  exhaustive controls remain available.
- Added shared requested-level VC3D eight-corner batch sampling for native
  fiber prediction and Lasagna normal volumes, including mixed physical chunk
  grids, one decoded cache per physical scalar volume, boundary-aware retained
  chunk lookup, and caller-side orientation-tensor normal interpolation.
- Fused persisted prediction/normal corner decoding into native candidate
  scoring, eliminating full decoded candidate arrays and reducing the approved
  warm-cache whole-fiber workload from 37.277s to 27.291s wall time. The
  measured 8-restart post-float result is unchanged.
- Reduced that workload to 21.155s wall / 619.366s CPU with compact
  reconstruct-on-selection frontier records and corner-batch improvements.
  The accepted post-float result remained at 8 restarts.
- Converted native fiber tracing's internal geometry and candidate math to
  float while retaining double persisted/public coordinate boundaries.
- Added persistent CP-owned native fiber segment metadata in strict
  `vc3d_fiber` v2 files, Lasagna protection for traced spans, scoped CP-edit
  invalidation, finalized trace auto-save, and Ctrl-right-click transactional
  reversion of traced spans to Lasagna optimization.
- Standardized Python CLI, native C++ CLI, and VC3D fiber endpoint acceptance
  on a `20` base-voxel threshold; physical voxel size is now optional and used
  only for physical-unit reporting.
- Replaced opaque Lasagna project/cache identities with actual local or remote
  Zarr source paths, readable remote-source cache paths, and path-bearing
  manifest ownership tags without duplicating manifest-derived group, channel,
  or spacing metadata.
- Corrected VC3D native segment tracing to keep stored fibers in base
  coordinates while prediction and normal sampling run on the derived sd2
  trace grid with correctly scaled physical units.
- Added generic exact-byte arbitrary remote-file caching, persistent direct
  remote Lasagna manifests, and VC3D local/remote manifest attachment with
  canonical role tags and automatically reconciled ordinary 3D project
  volumes.
- Restricted VC3D Lasagna attachment and sampling to per-channel 3D ZYX arrays;
  older flat CZYX preprocessing/fit artifacts now require conversion to
  per-channel OME-Zarr.
- Reduced native `vc_fiber_trace_metric` warm-cache runtime on the remote
  fiber manifest workload from roughly 314s to roughly 86s with profiled
  direct chunk-resolution sampling, bounded deterministic beam pruning, and
  lightweight final-lookahead frontier records while preserving the measured
  5-restart result.
- Added opt-in native C++ Trace2CP parallel candidate scoring through batched
  persisted prediction materialization, batched Lasagna normals, and
  `vc_fiber_trace_metric --threads`, while preserving deterministic beam output
  order.
- Reduced native Trace2CP beam expansion overhead by keeping internal trace
  paths parent-linked and caching per-trace cone offsets instead of rebuilding
  them per beam.
- Matched native C++ Trace2CP beam pruning/reached-target selection and compact
  normal principal-axis decoding more closely to the Python tracer, with
  focused `test_fiber_trace3d` regression coverage for beam order, reached
  ties, normal eigensolver behavior, and all-pairs candidate loss.

# 2026-07-29

- Native 3D Trace2CP refined/fused/regenerated presence panels now display
  presence modulated by predicted direction alignment to the strip plane, while
  original presence panels remain raw presence.
- Python native 3D Trace2CP whole-fiber tracing now keeps tracing after early
  far target-plane crossings until the best selected crossing is within the
  configured error threshold or the trace budget is exhausted.
- Python native 3D Trace2CP whole-fiber tracing now preserves live trace point,
  previous direction, sampled-current direction, and smoothing history across
  successful CP crossings instead of reinitializing at each accepted CP.
- Native 3D Trace2CP target-plane termination no longer uses the CP-to-CP
  chord. Python and VC3D native tracers now require target-local line-neighbor
  and inferred-direction planes, wait until all configured planes are crossed,
  and score the lowest in-plane CP crossing error.
- Added native `vc_fiber_trace_metric --inference-scaledown-power`, defaulting
  to 2, so existing fiber prediction manifests derive trace scale from the
  persisted prediction scale without adding manifest fields.
- Brought the native C++ `vc_fiber_tracer`/`vc_fiber_trace_metric` Trace2CP
  search controls into parity with the Python native tracer for persisted
  inference products: circular default cone candidates, presence-weighted
  current branch choice, angle-squared split smoothness, cumulative tangent
  smoothness, target-plane crossing interpolation, spatial beam pruning, and
  matching CLI flags/defaults.
- Python native 3D Trace2CP now accepts multiple `--fiber-json` paths, runs
  them sequentially with a shared loaded model, writes indexed per-fiber
  summaries/visualizations, and reports an accumulated restart-rate score.
- Python native 3D whole-fiber Trace2CP visualization now page-splits wide
  JPG output around restart boundaries and before the JPEG dimension limit.
- Python native 3D whole-fiber Trace2CP CP labels now render at the bottom of
  each strip and include CP indices for explicit trace selection.
- Python native 3D whole-fiber Trace2CP can now start metric tracing at a
  selected CP with `--whole-fiber-start-cp-index`.
- Improved remote Lasagna manifest fetch errors with resolved URL, HTTP
  response metadata/body excerpts, and S3 region/auth diagnostics.
- Required Lasagna normal samplers for native Trace2CP normal-aware smoothing
  and made `vc_fiber_trace_metric` require explicit `--normal-manifest`.
- Changed Python native Trace2CP and the native C++ fiber metric default beam
  lookahead to 2 so their relevant trace-control defaults match.
- Changed native `vc_fiber_trace_metric` to infer its tracer working scale from
  the precomputed fiber inference manifest channels instead of a CLI scale
  argument.
- Extended native VC3D Lasagna dataset opening so `vc_fiber_trace_metric` can
  stream precomputed fiber inference manifests directly from HTTP/S3 locations
  with an explicit remote cache directory, including relative and absolute
  group Zarr paths.
- Prevented pyramid multiprocessing from multiplying every worker by a full
  OpenBLAS/OpenMP thread pool while retaining automatic process parallelism.
- Added the first native VC3D 3D fiber Trace2CP segment tracer: shared
  Lasagna compact-channel helpers, project-level fiber inference dataset
  selection, a Qt-free `vc_fiber_tracer` core target, and a Ctrl-right-click
  generated-line segment optimization action in the line annotation GUI.
- Added native `vc_fiber_trace_metric`, a no-visualization C++ whole-fiber
  restart-rate runner for precomputed 3D fiber inference `.lasagna.json`
  products and `vc3d_fiber` JSON files.

# 2026-07-28

- Defaulted Fiber whole-volume output to filtered 0.25x inference, changed
  Fiber and Lasagna OME chunks to 64 cubed, and made the shared circular
  runner touch, flush, and clear only supported contribution-dirty chunks.
- Native 3D Trace2CP inference blocks now use a shared VC3D-backed
  requested-level axis-aligned block-read API and batch missing block forwards,
  avoiding dense coordinate-grid sampling for regular inference cubes.
- Native 3D Trace2CP live inference now uses the shared fiber 3D inference
  adapter plus shared Lasagna normal encoding and raw fiber prediction decode
  helpers.
- Native 3D Trace2CP now defaults to metric-only output; `--vis` explicitly
  enables JPG rendering and partial image updates.
- Native 3D whole-fiber Trace2CP now reports restarts per 1000 reference
  voxels, with optional restarts per meter in summaries and progress output
  when VC3D `volume.metadata["voxelsize"]` is available.
- Native 3D whole-fiber Trace2CP human metric output now uses one decimal
  place and includes mean successful run length in millimeters beside `err/m`
  when physical units are available.
- Native 3D Trace2CP can now box-downsample raw inferred fields with
  `--inference-scaledown-power` before direction/presence sampling.
- VC3D remote volume metadata normalization now always discovers public
  `scan/tomo/acquisition/detector/samplePixelSize` as `voxelsize` when no
  explicit positive `voxelsize` exists.
- Replaced full-Z predict3d scratch mappings with fixed-depth circular mmap
  rings and consolidated Lasagna/Fiber neural inference onto one multi-scale,
  chunk-flushing runner.
# 2026-08-03: process-parallel shared inference flush

- Replaced the single Python flush thread with bounded persistent spawn workers
  that read frozen rolling-accumulator mmaps by absolute path.
- Added shared `flush_workers` control and `--flush-workers` to Fiber and
  Lasagna inference (automatic CPU-count default capped at 64, synchronous
  baseline 0).
- Added process failure/hard-exit cleanup, overlap, multi-process execution,
  and CLI forwarding coverage.
- Motivation: the prior threaded implementation regressed the representative
  eight-GPU inference phase from 178.8 s to 305.4 s.
- Reused the pyramid pool's pre-spawn native-runtime guard so NumPy/OpenBLAS is
  single-threaded during child module import, including for GPU workers.
# 2026-08-03: TensorStore whole-volume inference prefetch

- Shared Fiber/Lasagna inference now defaults to asynchronous TensorStore Zarr
  bbox reads with read-ahead capacity independent of GPU/result slots.
- Added bounded cache/I/O/copy and per-GPU prefetch controls, single-device
  read-ahead, Python-Zarr fallback, exact reader-equivalence tests, and input
  starvation/high-water diagnostics.

# 2026-08-03: process-parallel native accumulation

- Added deterministic process-owned chunk accumulation shared by Fiber and
  Lasagna multi-device inference, with bounded queues and retained result-slot
  lifetimes.
- Added a portable native accumulator extension with runtime AVX-512F+F16C
  dispatch and restored float16 product rings as the default.
- Added `--accumulator-workers`, backend/throughput diagnostics, native
  numerical coverage, and process-vs-synchronous output coverage.
# 2026-08-13: half-voxel fiberlet DP

- Halved anchor-to-anchor fiberlet DP spacing to 0.5 stored prediction voxels,
  with cached sign-invariant prediction/normal interpolation and unchanged
  base-coordinate artifacts.
