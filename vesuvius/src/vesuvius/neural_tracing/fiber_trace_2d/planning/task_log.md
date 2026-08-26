# Task log: parallel constraint scoring and OBJ diagnostics

## Baseline

- Baseline commit: `d4db87721` (`Extract constraints from stored crop traces`).
- The scoring loop already uses deterministic OpenMP slots and averaged about
  28 effective cores, but its 55,170 independent calls discover and acquire
  Lasagna channel chunks at fine granularity. The representative run spent
  11.2827 s in scoring, with 49.27 s user and 271.08 s system CPU time overall.
- The spatial R-tree phase took only 0.1478 s and is not the target.

## Decisions

- The user explicitly confirmed that double-coordinate normal sampling is not
  required. The implementation will reuse the existing float-coordinate grouped
  corner sampler and will measure its difference from scalar-double winding.
- OBJ lines connect the two closest sampled base-coordinate points. Hard
  continuity links are excluded because their endpoints coincide.
- Winding `0.5` belongs to the separate-winding view, leaving the parallel
  classes disjoint. Perpendicular winding uses the requested strict `>0.3`.

## Independent plan review

- Review correctly rejected the initial prefetch-only design because it would
  retain fine-grained scalar samples and would not pin a whole working set.
  After user clarification, the plan now uses the actual grouped corner sampler.
- Review required exact output-basename behavior and stable OBJ identities.
  The plan now strips a supplied final extension, defines the `.zarr` default,
  and names objects from ascending global piece IDs.

## Deviations and deferred work

- The reviewed exact-double grouped API was replaced before implementation after
  the user explicitly stated double precision was unnecessary. Batched winding
  uses the established float-coordinate grouped corner infrastructure; tangent
  scoring and stored closest-point geometry remain double precision.
- Constraint records remain report-only in memory. The three OBJ files are
  diagnostics, not the future discrete-optimization interchange format.

## Validation

- GCC build:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace test_lasagna_normal_sampler -j32`.
- Clang build:
  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiber_trace_chunk test_fiberlet_crop_trace test_lasagna_normal_sampler -j32`.
- Both compilers passed `test_fiberlet_crop_trace` (20 cases) and
  `test_lasagna_normal_sampler` (12 cases). Tests cover strict OBJ thresholds,
  hard-link exclusion, extension-independent filenames, stable object names,
  scalar/batch integration, varying-field float tolerance, and exact one- versus
  four-thread batch parity.
- Representative command:
  `volume-cartographer/build/bin/vc_fiber_trace_chunk constraints /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/crop_traces.zarr --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --output /tmp/constraint_batched.obj`.
- Dataset/build: the same 500-trace artifact and Release build as baseline;
  55,170 measured links and 798 hard links, three iterations.
- External wall seconds: `0.50, 0.48, 0.48`; mean `0.487`, median `0.48`,
  min/max `0.48/0.50`. Internal total wall seconds: `0.360, 0.338, 0.343`;
  mean `0.347`, median `0.343`, min/max `0.338/0.360`.
- Batched winding seconds: `0.0839, 0.0706, 0.0778`; median `0.0778`, versus
  baseline scalar winding/scoring `11.2827` seconds. Median winding speedup is
  about `145x`; internal total speedup versus baseline `11.4466` seconds is
  about `33x`.
- Peak RSS was 127.7-128.4 MiB. Outputs were stable at 33,808 perpendicular,
  5,667 parallel same-winding, and 15,000 parallel separate-winding lines.
- Batch-versus-scalar representative deciles differed only in the final few
  decimal places; focused varying-field tests enforce relative tolerance
  `2e-5`. `git diff --check` passed.
