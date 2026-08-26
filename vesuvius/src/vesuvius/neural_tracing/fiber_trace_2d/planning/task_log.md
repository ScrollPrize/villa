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

## LP relaxation follow-up

- A centered 512-base-voxel crop produced 426 traces in 8.02 seconds, but the
  MILP was stopped after 197.63 seconds without a solution report at
  3,532,912 KiB peak RSS.
- A centered 256-base-voxel crop produced 63 traces in 1.21 seconds. Its MILP
  likewise did not return interactively, motivating an explicit continuous
  relaxation of the three piece variables.
- `--lp-relaxation` retains the same model rows and objective but makes active,
  vertical, and odd continuous on `[0,1]`. Raw values are emitted without
  thresholding or repair so relaxation quality can be inspected directly.
- The untightened 1024-crop LP solved in 0.943 seconds with 83,100 variables
  and 187,410 rows. Of 1,298 pieces, 1,293 were active; 1,292 H/V and parity
  values were exactly `0.5`. The local edge envelopes let every edge select its
  cheaper same/different relation independently, without enforcing XOR parity
  around graph cycles.

## Triangle tightening results

- The independent review's gauge concern was resolved by fixing only the root
  H/V and parity upper bounds, never active. If the root breaks, split active
  components retain harmless flip symmetry. Triangle edge inequalities use an
  activity-sum big-M form; the focused test includes fractional activity and
  confirms a broken/fractional vertex does not force the opposite edge equal.
- Centered 256 crop: 63 pieces, 638 links, 3,250 triangles, 34,420 rows. The LP
  solved in 1.659 seconds (1.72 seconds command wall, 122,320 KiB peak RSS).
  Active values span `0.7..1`, H/V `0..0.7`, and parity `0..0.65`; the prior
  all-half solution is removed.
- Centered 384 crop: 190 pieces, 3,353 links, 27,615 triangles, 264,889 rows.
  The LP solved in 112.875 seconds (113.03 seconds command wall, 507,592 KiB
  peak RSS). Active spans `0.2..1`, H/V `0..0.7125`, and parity `0..0.6`.
- Centered 512 crop: stopped after 232.36 seconds without a result at 5,875,616
  KiB peak RSS. The 1024 crop was stopped after 273.57 seconds without a result
  at 2,892,536 KiB peak RSS. All-triangle construction/solution therefore
  becomes impractical between 256 and 384/512; per user instruction no further
  optimization was attempted.
- Review requests for a triangle cap and a crop-selection CLI were not added:
  silently omitting cuts would change the diagnostic, while the benchmark uses
  explicit trace artifacts generated at documented centered 256/384/512 bounds.
- HiGHS received `threads=32`, but the measured tightened LP run used one
  effective core (`112.875` CPU seconds for `112.875` solve-wall seconds).
  Installed HiGHS 1.15.1 exposes `parallel=on` and the `hipo` solver.
- LP output now includes five threshold visualization OBJs: H/V and parity
  split at `0.5`, while activity splits at its solution mean. Exact threshold
  values remain active, V, and odd. The layers are diagnostics, not repaired
  integer labels.

## Explicit LP backend experiment

- Added relaxation-only `--lp-parallel` and `--lp-solver` controls. The latter
  accepts `choose`, `simplex`, `hipo`, and `ipm`; both are rejected without
  `--lp-relaxation`.
- The options only configure HiGHS execution. Model variables, constraints,
  objective, deterministic ordering, and the MILP path are unchanged.
- Centered 384 crop with automatic solver plus `parallel=on`: objective
  `1358.9023`, solve `113.134` seconds, command wall `113.28` seconds, CPU
  `113.32` seconds (`100%`), and peak RSS `508,284` KiB. This is effectively
  identical to the serial automatic baseline, so HiGHS still selected a serial
  algorithm. The five relaxation visualization OBJs were written under
  `/tmp/crop_constraints_central_384_parallel_relaxation_*.obj`.
- Parallel HiPO could not start. A direct HiGHS 1.15.1 option probe reported:
  `The HiPO solver was requested ... features are unavailable: amd, blas,
  metis, rcm`. The installed executable, headers, and `libhighs.so.1.15.1`
  belong to the same Arch package. The wrapper rejects this error and does not
  fall back silently.

Build: `volume-cartographer/build`, CMake `Release`, HiGHS 1.15.1, one run per
configuration. Input artifact: `/tmp/crop_traces_central_384.zarr`; normal
manifest: `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json`.

| Requested backend | Status | Objective | Solve | Wall | CPU | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| solver=`choose`, parallel=`choose` | Optimal | 1358.9023 | 112.875 s | 113.03 s | 112.875 s | 507,592 KiB |
| solver=`choose`, parallel=`on` | Optimal | 1358.9023 | 113.134 s | 113.28 s | 113.32 s | 508,284 KiB |
| solver=`hipo`, parallel=`on` | unavailable | n/a | n/a | 0.08 s | 0.48 s | 83,672 KiB |

Parallel automatic command:

```sh
/usr/bin/time -v volume-cartographer/build/bin/vc_fiber_trace_chunk constraints /tmp/crop_traces_central_384.zarr --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" --output /tmp/crop_constraints_central_384_parallel --lp-relaxation --lp-parallel
```

HiPO attempt: the same command with output basename
`/tmp/crop_constraints_central_384_hipo` and additional `--lp-solver hipo`.
The complete initial-fiber, constraint, relaxation-label, and CSV visualization
set was collected under `$VES/data/workdir3/384/` with concise
`384_<label>` filenames.

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
