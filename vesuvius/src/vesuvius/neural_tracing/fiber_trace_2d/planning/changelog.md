# 2026-07-28

- Native 3D Trace2CP detailed stage profiling is now opt-in with `--profile`;
  ordinary metric runs still print final metric and total wall/CPU timing
  without per-stage instrumentation overhead. Added an explicit experimental
  `--normal-principal-axis-method analytic` path for sparse normal
  reconstruction while keeping `eigh` as the quality-matching default.
- Accelerated native 3D whole-fiber Trace2CP hot-path lookup/scoring while
  preserving the reference benchmark metric: point lookup now routes candidate
  batches with torch block-origin/grouping, duplicate current-point field
  sampling is carried in beam state, beam pruning avoids per-selection syncs,
  sparse normal tensor principal axes use batched symmetric eigensolve, and
  inferred block coordinate metadata is cached on device.
- Restored native 3D Trace2CP candidate Lasagna normals to the
  pre-acceleration geometry-loader sampler and restored vector-normal
  smoothness scoring.
- Removed the intermediate sparse-normal Trace2CP scoring path and its torch
  closed-form normal helpers; live inferred prediction blocks still stay
  device-resident under the existing LRU cache budget for point lookup.
- Added a debug-only native 3D Trace2CP normal comparison mode that runs sparse
  Lasagna normal sampling beside the production geometry-loader sampler and
  fails fast on significant differences.
- Native 3D Trace2CP now defaults to the sparse corner/tensor Lasagna normal
  sampler, with the geometry-loader sampler retained as `--normal-sampler
  baseline` and as the debug comparison reference.
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
- Native 3D Trace2CP can now downscale raw inferred fields with Lasagna
  Gaussian pyramid filtering through `--inference-scaledown-power`.
- Native 3D Trace2CP can now apply an opt-in 3D Gaussian blur to raw inferred
  fields after scaledown and before trusted-core caching.
- VC3D remote volume metadata normalization now always discovers public
  `scan/tomo/acquisition/detector/samplePixelSize` as `voxelsize` when no
  explicit positive `voxelsize` exists.
- Replaced full-Z predict3d scratch mappings with fixed-depth circular mmap
  rings and consolidated Lasagna/Fiber neural inference onto one multi-scale,
  chunk-flushing runner.
