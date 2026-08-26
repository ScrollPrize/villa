# Task log: H/V constraints from stored crop traces

## Discovery

- The durable crop artifact already stores complete float64 base-XYZ polylines,
  stable ordinals, seed data, and trace cost. Constraint extraction therefore
  does not need the source Fiberlet graph.
- `LasagnaNormalSampler::windingDistance` is the Atlas winding integral used by
  the existing fiber-intersection search. It integrates decoded `grad_mag`
  winding density along a straight connector with trapezoidal sampling.
- Existing Atlas intersection broad-phase code has a general fiber/Ceres data
  model, but this task explicitly uses uniformly resampled trace pieces and a
  point-only closest-pair search. Reusing its full Ceres search would violate
  that requested scoring contract.
- The separate `fiber-chunk-hvopt` branch contains the earlier NML encounter
  inspection tool. It remains intentionally separate; this task consumes crop
  trace Zarr artifacts and will not merge that NML CLI.

## Decisions

- The experimental closest-distance default is `128` base voxels and remains a
  CLI parameter because the user left the threshold open.
- Piece splitting uses the minimum count that keeps equal piece spans no longer
  than `512` after adding `128` overlap. This is the standard deterministic
  interpretation of an evenly split overlapping window when the line length is
  not an exact fit.
- Only consecutive pieces from one original trace receive hard links. Linking
  every same-trace piece pair would create false nonlocal continuity edges.
- R-tree cube queries are only a broad phase. Every hit is checked against the
  exact Euclidean radius before it can become a piece-pair candidate.
- Degenerate consecutive points do not contribute arclength; wholly degenerate
  traces are skipped and counted because they cannot provide tangent evidence.
- A supplied manifest is validated structurally in the stored trace's base
  frame. Parsed manifest equality is diagnostic only: locator and byte identity
  are provenance, not geometric compatibility.

## Independent plan review

- Review found and the plan now addresses: exact spherical filtering after
  R-tree cube lookup, infeasible manifest-identity gating, degenerate trace
  handling, and a missing trace-specific normal frame/coverage validator.

## Deviations and deferred work

- Constraint serialization, graph optimization, H/V label assignment, winding
  index assignment, and conflict handling are explicitly deferred by the task.
  This first stage reports in-memory constraint statistics only.

## Validation

- GCC build:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace test_lasagna_normal_sampler -j32`.
- Clang build:
  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiber_trace_chunk test_fiberlet_crop_trace test_lasagna_normal_sampler -j32`.
- Both GCC and Clang passed `test_fiberlet_crop_trace` (19 cases) and
  `test_lasagna_normal_sampler` (11 cases).
- `git diff --check` passed.
- Representative Release command:
  `volume-cartographer/build/bin/vc_fiber_trace_chunk constraints /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/crop_traces.zarr --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json`.
- The artifact contained 500 traces. Extraction produced 1,298 pieces, 19,820
  resampled points, 55,170 measured links, and 798 hard continuity links with no
  tangent or winding rejection. Internal phase timings were 0.0013 s prepare,
  0.1478 s spatial search, 11.2827 s scoring, and 11.4466 s total wall time.
  `/usr/bin/time` reported 11.60 s wall, 49.27 s user, 271.08 s system, and
  86,752 KiB maximum RSS.
- The dominant cost is parallel Lasagna channel sampling during aligned winding
  scoring, not the point R-tree search. This task records that baseline but does
  not alter sampling arithmetic or caching behavior.
