# Task: Fiber Scale-2 Output, Sparse Accumulator Activity, and 64³ Chunks

Plan changes to shared Lasagna/Fiber 3D tiled inference so that:

- Fiber whole-volume inference defaults `--inference-scaledown-power` to 2,
  meaning a source-relative factor of `2**2 == 4` and therefore `0.25x` output
  in each spatial dimension. Downscaling includes the matching low-pass blur,
  not direct stride-4 subsampling. The separate tracer/model-config
  `scaledown` setting is out of scope and must remain unchanged.
- The circular accumulator touches, normalizes, clears, and writes only output
  chunks that actually receive supported model contributions. Masked or absent
  outer volume regions must remain sparse: no output Zarr chunk and no scratch
  mmap page should be created merely because a flush frontier crossed them.
- Lasagna predict3d and Fiber inference default to cubic OME-Zarr chunks of
  `64x64x64`, while retaining explicit CLI/API overrides.

Keep the single shared tiled runner and preserve global lattice, crop, resume,
atomic-write, numerical, manifest-scale, and one-model-call-per-tile semantics.
This task is planning-only until implementation is explicitly requested.
