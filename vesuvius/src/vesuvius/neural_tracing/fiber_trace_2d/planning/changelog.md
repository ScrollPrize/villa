# 2026-07-28

- Native 3D Trace2CP candidate normal sampling now uses Lasagna streaming
  sparse GPU sampling for `grad_mag`, `nx`, and `ny` on CUDA, converts compact
  normals to Lasagna-style second-moment tensors before interpolation, and
  keeps live inferred blocks device-resident under the existing LRU cache
  budget for point lookup.
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
