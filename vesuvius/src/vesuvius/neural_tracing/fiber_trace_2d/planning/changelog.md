# 2026-07-30

- Added shared requested-level VC3D eight-corner batch sampling for native
  fiber prediction and Lasagna normal volumes, including mixed physical chunk
  grids, one decoded cache per physical scalar volume, boundary-aware retained
  chunk lookup, and caller-side orientation-tensor normal interpolation.
- Converted native fiber tracing's internal geometry and candidate math to
  float while retaining double persisted/public coordinate boundaries. The
  representative quality check currently regresses from 5 to 8 restarts and
  remains an open acceptance issue.
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
