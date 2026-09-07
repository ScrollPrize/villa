# Task log: native fused Fiber3D coordinate generation

- 2026-09-07: User authorized replacing full-resolution PyTorch coordinate
  operations with a custom C++ generator. Approximate rather than bit-identical
  coordinates are acceptable; GPUs remain unavailable.
- 2026-09-07: Selected a standalone native coordinate-generation primitive,
  leaving the shared VC3D volume sampler unchanged. Direct per-coordinate
  evaluation is the first implementation; approximate low-resolution 3D
  precomposition is conditional on measured benefit.
- 2026-09-07: Independent review approved direct lookup and required explicit
  supersession of the former implicit/bitwise plan, numerical pass thresholds,
  a complete array contract, and unambiguous auto/native/eager selection.
- 2026-09-07: Implemented a portable scalar C++ binding which releases the GIL
  and directly emits base coordinates and validity. Added auto/native/eager
  loader selection and focused contract/reference/sampled-value tests.
- 2026-09-07: Release GCC 13 one-worker S1a results improved raw throughput
  from 3.04 to 4.72 samples/s and coordinate generation from 5.63 to 17.83
  maps/s. The 128-loader run reached 227.33 samples/s, versus 116.53 for the
  previous cache-local fused Torch path. Maximum reference coordinate error was
  3.82e-6 base voxels. The final dedicated native/backend selection test run
  passed (8 tests); the broader native-related selection passed 110 tests.
- 2026-09-07: Did not add approximate low-resolution 3D precomposition: direct
  low-resolution control lookup already exceeds the 221 samples/s global target,
  so another representation and interpolation layer is not justified by the
  measurements. It remains an option only if later profiles identify smooth
  control lookup as a bottleneck.
