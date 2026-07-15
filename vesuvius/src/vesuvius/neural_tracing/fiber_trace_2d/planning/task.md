# 3D Fiber Loader Performance Rewrite

The current 3D CP-centered training path has abysmal loader performance. The
observed training output shows `load_ms` around 10-15 seconds for a single
192^3 patch while forward/backward is around 0.5 seconds.

Required behavior:

- Establish a current load-only benchmark baseline before changing the loader.
- Stop using the custom `_read_raw_block` zarr crop path plus torch
  `grid_sample` as the normal 3D training input loader.
- Build explicit 3D augmentation coordinate maps in the same spirit as the 2D
  fiber loader: geometry is represented as concrete output-to-source and
  source-to-output maps, with no search, brute-force inversion, or iterative
  inverse solving.
- Use the VC3D coordinate sampler for 3D patch loading so chunk discovery,
  blocking chunk fetch/decode, cache use, and sampling go through the same
  production machinery as the 2D fiber loader.
- Use sensible runtime parallelization inspired by the 2D fiber loader. Do not
  rely on `ThreadPoolExecutor` for CPU-heavy Python work that is effectively
  serialized by the GIL.
- Ground-truth creation must draw/project the transformed fiber segments and
  segment mask directly from the coordinate maps. Preprocessing/loading must not
  require nearest-segment searches over full dense patches, inverse solving, or
  other search-based geometry recovery.
- Preserve deterministic random sample ordering: changing batch size or max
  step count may truncate/extend the deterministic prefix, but must not reshuffle
  prior samples.
