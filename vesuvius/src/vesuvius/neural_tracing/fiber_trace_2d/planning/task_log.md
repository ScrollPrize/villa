# Task log: mixed-integer crop-fiber labeling

## Initial decisions

- The user requested five mutually exclusive states but described three binary
  labels. The MILP uses active, H/V, and parity binaries with H/V and parity
  forced to zero while inactive, giving exactly one canonical broken state.
- Winding cutoff is exclusive: measured links with distance `>= 1.5` are
  discarded. Same-trace continuity links remain at distance zero.
- `0.5 * degree` is the default broken penalty, directly following the user's
  proposed constraint-count scaling. Degree includes every retained incident
  link, including same-trace continuity.
- Orientation and parity terms are added with equal weight. No additional
  distance weighting is introduced because it was not requested.
- Existing three constraint diagnostic OBJs remain; five label OBJs are added
  from the same command and basename.

## Environment

- HiGHS `1.15.1` is installed as `/usr/lib/libhighs.so` with CMake target
  `highs::highs` and C++ headers under `/usr/include/highs`.

## Real-graph formulation finding

- Declaring the three link-local auxiliaries integer created about 169,000
  integer columns on the 55k-link representative graph. The exact run was
  stopped after 189.66 s at 1,233,844 KiB peak RSS. The auxiliaries are now
  continuous because their AND/XOR envelopes force integral values from binary
  endpoints; only 3,894 piece columns remain integer.
- Even the reduced formulation did not prove zero gap within 146.76 s and used
  1,353,360 KiB peak RSS. The default now uses standard practical MIP
  tolerances (`1e-4` relative, `1e-6` absolute), reports the achieved gap, and
  exposes `--mip-gap 0` for explicit exact solves.
- Rewriting each pair energy as a nonnegative minimum base cost plus an
  indicator for only the more expensive agreement/difference relation reduced
  the exact hull from 13 to at most 7 rows per link. The requested active-label
  objective is algebraically identical. The representative default-radius
  solve still did not finish within 183.66 s (893,380 KiB peak RSS), so dense
  graph solve time remains an explicit limitation of this initial MILP.

## Validation

- GCC Release build and focused test command:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace -j32 && volume-cartographer/build/bin/test_fiberlet_crop_trace`.
- Focused tests cover the exclusive winding cutoff, separate invalid/cutoff
  counts, exact orientation/parity costs, broken-link disabling, isolated-piece
  canonicalization, invalid solver coefficients, and five OBJ classes.
- Clang validation and a completed representative default-radius solve were not
  run before the user requested wrap-up.

## Independent plan review

- Added per-component H/even canonicalization for the two unavoidable global
  binary symmetries and canonical broken labels for isolated pieces. HiGHS will
  also run with deterministic settings. A fully lexicographic choice among all
  mathematically equivalent broken/active cuts would require a potentially
  prohibitive sequence of MILP solves and is not part of this diagnostic.
- Added a dedicated winding-cutoff rejection counter instead of conflating it
  with invalid/non-finite winding samples.
- Expanded the plan to provision and link HiGHS on Ubuntu, macOS, and Windows,
  validate the broken coefficient, and exercise user-visible output behavior.
