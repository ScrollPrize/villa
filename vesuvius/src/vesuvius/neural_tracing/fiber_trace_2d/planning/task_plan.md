# Native Fiber Trace VC3D Corner-Batch Sampling Plan

## Scope

- Reuse VC3D's `ChunkCache` and blocking requested-level coordinate sampler for
  persisted fiber prediction and Lasagna normal volumes.
- Maintain one long-lived decoded cache per physical scalar Zarr volume.
- Fetch the eight integer voxel corners for each candidate with nearest-neighbor
  sampling, then interpolate in fiber/normal code.
- Convert native fiber tracing internals to float while preserving public
  double-based file and GUI boundaries where changing those interfaces would
  expand the task unnecessarily.

## Implementation

1. Export a shared VC3D helper that wraps an already-open scalar Zarr array in a
   one-level `ChunkCache`; do not duplicate the private Zarr fetcher.
2. Add a shared Lasagna channel corner-batch helper that:
   - prepares float source-grid coordinates and interpolation fractions;
   - sends all eight integer corners through VC3D blocking nearest-neighbor
     coordinate sampling;
   - returns ordered uint8 corner values and validity;
   - retains one cache for the channel binding's physical Zarr volume.
3. Port persisted fiber prediction batching and Lasagna normal batching to the
   corner helper. Scalar channels use float trilinear interpolation. Compact
   `nx/ny` pairs decode each corresponding corner as one ambiguous axis,
   accumulate the weighted orientation tensor, and select its principal axis.
4. Fuse candidate prediction, normal decoding, and loss evaluation as far as
   practical without changing candidate generation/pruning order.
5. Convert internal fiber-trace vectors, interpolation weights, directions,
   losses, beam state, and geometry calculations to float. Convert at the
   existing public API boundaries.
6. Remove or make unreachable the superseded direct per-thread resolver paths
   from fiber tracing while retaining shared legacy channel APIs needed by
   other Lasagna callers.

## Correctness And Performance

- Add focused tests for ordered eight-corner sampling, cross-chunk corners,
  missing/error handling, scalar interpolation, and orientation-aware normal
  interpolation.
- Run native fiber and Lasagna sampler tests.
- Build `vc_fiber_trace_metric` in the existing build tree.
- Run only the approved representative command for end-to-end performance and
  quality comparison. Record wall/CPU time, profile stages, restart count, and
  segment count against the current approximately 86s / 5-restart result.
- Confirm deterministic result stability with a repeated representative run if
  runtime permits.

## Spec Update

- Change native precomputed Trace2CP sampling requirements to mandate one VC3D
  decoded cache per physical scalar volume, batched nearest-neighbor reads of
  the eight interpolation corners, and caller-side orientation-aware compact
  normal interpolation.
- Allow float internal native fiber-trace math while preserving deterministic
  candidate order and requiring metric-based quality validation.

## Docs Updates

- Update the planning changelog with the shared VC3D batch reader and measured
  performance/quality result.
- Record implementation attempts, measurements, and deviations in
  `planning/task_log.md`.

## Review

- Review the final diff against `planning/specs.md`, this task, and the
  monorepo portability/determinism requirements.
- The workflow requests an independent agent plan review, but higher-level
  agent policy forbids spawning an agent without explicit user authorization;
  record this as a process deviation and perform a direct consistency review.
