# Plan: H/V constraints from stored crop traces

## Contract and data model

1. Add a `FiberTraceConstraintConfig` expressed entirely in base voxels. Initial
   defaults are: resample spacing `32`, target piece length `512`, overlap `128`,
   closest-pair threshold `128`, tangent secant window `32`, phase-refinement
   per-fiber increment/limit `spacing/20` (a combined relative shift of
   `spacing/10`), and Lasagna
   winding integration step `8`.
2. Represent every piece by original trace ordinal, piece ordinal, original-line
   arc interval, uniformly sampled points/arcs, and exact endpoint samples.
   Choose the minimum number of pieces whose equal arc span is at most the target
   length after accounting for fixed overlap; short traces remain one piece.
   Existing zero-length consecutive edges are ignored by arc sampling. A trace
   with no nonzero edge is skipped and counted rather than aborting extraction.
3. Represent every constraint by stable piece identities, closest points/arcs,
   closest Euclidean distance, normalized parallel/perpendicular scores, the
   normal-modulated winding distance, and whether it is a hard same-trace link.
4. Keep this report-only format in memory. Constraint persistence and discrete
   optimization are explicitly outside this task.

## Spatial candidates and scoring

5. Build one Boost.Geometry point R-tree over resampled piece points. Query every
   point with the configured distance cube, reject same-original-trace pairs,
   apply an exact Euclidean `distance <= max_distance` test to every hit, and
   retain only the minimum-distance point pair for each unordered piece pair.
   Resolve equal distances by stable point ordinals so results are deterministic.
6. Do not insert same-trace pair candidates. Emit one hard continuity constraint
   between each consecutive pair of pieces using the midpoint of their overlap,
   parallel score `1`, perpendicular score `0`, winding distance `0`, and
   Euclidean distance `0`.
7. At a measured closest pair, use centered secant tangents and the sign of their
   dot product to orient both piece walks consistently. Walk in both arc
   directions at the resample spacing. At every step, compare the current phase
   with plus/minus `spacing/20` counter-shifts, retain only a strictly closer
   phase, and cap each fiber's persistent phase at `spacing/20`.
8. Average signed, consistently oriented tangent dots across all usable walk
   samples, then clamp the mean into `[0,1]` for raw parallel evidence. Use
   `1 - abs(initial tangent dot)` for raw
   perpendicular evidence. Normalize the two raw values by their sum; reject a
   pair only when tangents/evidence are unusable.
9. Extract the existing Lasagna normal-manifest structural validation into a
   shared Lasagna helper used by both Fiberlet preprocessing compatibility and
   this stage. The trace-specific validator additionally requires base-coordinate
   working scale `1`, positive preserved prediction scale, and a declared base
   shape covering the complete stored crop. It does not require manifest path,
   byte, or parsed-JSON identity; it reports whether parsed content matches the
   trace provenance for diagnostics only.
10. Refactor Lasagna winding integration through one shared internal integrator.
   Preserve the existing `windingDistance` behavior exactly and expose a second
   `normalAlignedWindingDistance` operation that multiplies winding density at
   each trapezoid endpoint by the absolute connector/normal alignment. Missing
   density or required normal samples yield a non-finite result.
11. Score independent measured candidates in deterministic index slots using
    the configured thread count; zero means host CPU count. Sort final measured
    and hard constraints by stable piece identities.

## CLI and reporting

12. Extend `vc_fiber_trace_chunk` with:
    `constraints TRACE.zarr --normal-manifest PATH [options]`. The mode reopens
    the trace artifact, opens normals in base-coordinate working space, and
    structurally validates their frame and crop coverage.
13. Expose `--sample-step`, `--piece-length`, `--piece-overlap`,
    `--max-distance`, `--tangent-window`, and `--winding-step`; retain the
    existing host-CPU and cache defaults. Reject trace/visualization-only flags.
14. Print configuration, input trace/piece/sample counts, candidate/accepted/
    rejected/hard counts, phase timings, and total CPU/wall time. Print a compact
    `q0..q100` decile table for closest distance, normalized parallel score,
    perpendicular score, and aligned winding distance.

## Tests

15. Unit-test even overlapping splitting for short, exact, and remainder
    lengths; verify endpoint inclusion, duplicate-point handling, wholly
    degenerate trace skipping, and adjacent hard constraints.
16. Unit-test nearest-point candidate deduplication, exact spherical rejection
    of R-tree cube-corner hits, same-trace exclusion,
    deterministic ties, parallel lines, perpendicular lines, reversed line
    orientation, bounded phase refinement, and multi-thread parity.
17. Extend Lasagna sampler tests with constant synthetic density/normals to prove
    the aligned integral equals the ordinary integral for aligned connectors and
    approaches zero for perpendicular connectors while the ordinary method is
    unchanged.
18. Test trace-specific normal compatibility for wrong working scale, a crop
    outside `base_shape_zyx`, malformed channel layouts, and a structurally
    compatible manifest whose parsed provenance differs.
19. Build GCC Release and Clang targets, run the focused crop and Lasagna sampler
    suites, run `git diff --check`, then time the command on a representative
    stored crop artifact and record its exact command/data/result.

## Spec update

Add the stored-trace constraint extraction contract: base-coordinate defaults,
piece splitting, nearest-point broad phase, hard same-trace continuity links,
parallel/perpendicular scoring, aligned Lasagna winding definition,
deterministic parallel evaluation, and report-only first-stage scope.

## Documentation updates

Extend `volume-cartographer/docs/fiber_chunk_tracing.md` with the constraint
command, parameter units/defaults, piece/link semantics, score definitions,
normal-manifest identity requirement, and console report schema.

## Changelog

Add one entry for extracting H/V and winding constraints from durable crop-trace
artifacts after implementation and validation.
