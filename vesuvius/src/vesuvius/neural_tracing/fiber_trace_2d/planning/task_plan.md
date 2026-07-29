# Native C++ Trace2CP Parity Fix Plan

## Goal

Bring `vc_fiber_trace_metric` and the shared `vc_fiber_tracer` implementation
back into conformance with the existing native 3D Trace2CP spec, excluding the
known persisted-output quantization/compression gap. The target behavior is
the Python native Trace2CP scoring/search contract for the same persisted
presence/normal products.

## Current Mismatches To Fix

- Candidate directions are generated as a square angular grid in C++; the spec
  and Python path use a circular cone disk at `--cone-angle-step-degrees`.
- C++ current-point branch selection uses direction agreement only; Python and
  the spec use `abs(dot(branch_dir, previous_step_dir)) * branch_presence` for
  non-start points.
- C++ smoothness uses non-angle proxies (`1 - dot` and raw normal-component
  deltas); Python/spec use radians angle-squared penalties after free-angle
  subtraction.
- C++ has no cumulative tangent smoothness state or CLI controls.
- C++ detects target-plane crossing but keeps the overshooting candidate
  endpoint; Python interpolates the exact target-plane crossing and continues
  the whole-fiber trace from that crossing point.
- C++ beam pruning sorts and truncates without the Python/spec spatial
  diversity merge controlled by `--beam-prune-distance-voxels`.
- C++ CLI/config does not expose all controls needed for parity:
  `--beam-prune-distance-voxels`, `--smoothness-free-angle-degrees`,
  `--cumulative-smoothness-steps`, `--cumulative-smoothness-tangent-weight`,
  and the legacy-only `--cone-grid-size`.

## Implementation Plan

1. Add the missing trace config fields and CLI flags.
   - Extend `FiberTraceConfig` with `beamPruneDistanceVoxels`,
     `smoothnessFreeAngleDegrees`, `cumulativeSmoothnessSteps`, and
     `cumulativeSmoothnessTangentWeight`.
   - Use defaults matching the Python/native spec:
     `1.0`, `0.0`, `4`, and `2.0`.
   - Add `coneGridSize` with Python's legacy fallback default `25` so
     `coneAngleStepDegrees <= 0` has the same explicit behavior as Python.
   - Validate finite/non-negative values and require a Lasagna normal sampler
     when any normal-aware or cumulative tangent smoothness term is active.

2. Replace default candidate generation with the circular cone-disk contract.
   - For `coneAngleStepDegrees > 0`, generate tangent-plane offsets where
     `u^2 + v^2 <= coneAngleDegrees^2`, always including the center.
   - Keep the existing square/grid behavior only for the explicit legacy path
     when `coneAngleStepDegrees <= 0`, controlled by `coneGridSize`.
   - Keep deterministic ordering compatible with Python:
     `np.lexsort((v_deg, u_deg, radius2))`, meaning lower angular radius first,
     then stable offset ordering.

3. Split start-branch and current-point branch selection.
   - Start CP branch selection remains pure angular agreement to the CP-local
     tangent and ignores presence.
   - All later current-point sampling chooses the branch by
     `abs(dot(branch_dir, previous_step_dir)) * branch_presence`.
   - Candidate endpoints continue to evaluate all branches and reduce to the
     best branch score for that candidate.

4. Port the smoothness math exactly.
   - Convert free angle from degrees to radians.
   - Isotropic fallback:
     `smoothness_weight * max(0, angle(prev_step, step) - free_angle)^2`.
   - With a valid Lasagna normal:
     tangent turn is the angle between previous and candidate step after
     subtracting signed normal components and normalizing the tangent-plane
     projections.
   - Normal/elevation turn is
     `asin(dot(step, normal)) - asin(dot(prev_step, normal))`, sign ambiguity
     handled by choosing the normal sign consistently for the comparison.
   - Apply component weights independently:
     `tangent_weight * tangent_loss + normal_weight * normal_loss`.
   - Keep `smoothness_weight` as the isotropic fallback weight and as the
     component default only if tangent/normal weights are absent.
   - If the sampler exists but a candidate normal is invalid or projections
     are degenerate, fall back per candidate to isotropic smoothness.

5. Add cumulative tangent smoothness to beam state.
   - Carry a history heading per beam, initialized to the selected sampled
     start direction.
   - Update it with the configured history length after accepting/extending a
     candidate, matching Python’s running-heading semantics.
   - Add only a tangent-plane angle penalty against the history heading, using
     the candidate Lasagna normal and
     `cumulativeSmoothnessTangentWeight`.
   - If the candidate normal is invalid or tangent projection degenerates, the
     cumulative term is zero for that candidate.

6. Interpolate target-plane crossings.
   - When a candidate step crosses the target plane, compute the linear
     crossing parameter from previous and candidate signed distances.
   - Store the interpolated crossing point in the returned trace instead of the
     overshooting candidate endpoint.
   - Use that crossing point as the continuation seed for the next whole-fiber
     segment.
   - Preserve existing in-plane error calculation, but base it on the crossing
     endpoint.

7. Add beam spatial-diversity pruning.
   - After each configured lookahead expansion, sort by reached-state and
     cumulative score as today.
   - Keep candidates whose endpoint is at least
     `beamPruneDistanceVoxels` away from already-kept endpoints, until
     `beamWidth` is reached.
   - If pruning distance is `0`, preserve pure score truncation.

8. Add focused C++ regression tests.
   - Candidate generation default returns 81 candidates at `25/5` degrees and
     excludes square corners.
   - Start branch selection ignores presence, while later branch selection uses
     direction agreement times presence.
   - Smoothness numerics match small Python-derived fixtures for isotropic,
     tangent-plane, and normal/elevation cases.
   - Target crossing returns the interpolated point, not the overshoot point.
   - Beam pruning keeps spatially distinct lower-ranked alternatives when
     near-duplicate high-ranked beams exist.
   - Whole-fiber continuation starts the next segment from the prior crossing
     point.

9. Add an apples-to-apples validation command.
   - Run the user command and record `restarts`, `err/kvx`, wall time, and
     config echo.
   - Compare against Python on the same persisted manifest when practical, or
     against Python fixture-level score/candidate parity when full Python
     persisted tracing is not available.
   - Do not compare this task’s result against raw-checkpoint Python inference
     as a strict parity test, because that is the excluded item 1.

## Spec Update

The existing spec already states the intended behavior for candidate cones,
presence-weighted current-branch selection, angle-squared smoothness,
cumulative tangent smoothness, target-plane continuation semantics, beam
pruning, required Lasagna normals, and native metric defaults. The spec update
for this task should be narrow:

- Clarify that the C++ `vc_fiber_tracer` implementation must implement these
  same rules directly and may not use simplified proxy formulas.
- Clarify that whole-fiber continuation uses the interpolated target-plane
  crossing point, not the candidate endpoint beyond the plane.
- Note that persisted-output quantization/compression parity is intentionally
  out of scope for this task.

## Docs Updates

- Update `docs/code_structure.md` only if names/flags/defaults change from the
  current documented contract.
- Add a short native tracer parity/debug note describing the expected
  `vc_fiber_trace_metric` command, the persisted-manifest caveat, and the
  regression tests used to protect Python/C++ behavior.

## Testing

- Build the affected C++ target with the existing CMake build directory.
- Run the new/affected C++ unit tests for `vc_fiber_tracer`.
- Run `vc_fiber_trace_metric` on the provided S1 command:

```bash
volume-cartographer/build/bin/vc_fiber_trace_metric \
  s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json \
  $VES/data/train_fibers/fibers_test_paul_4/kb_202606*01.json \
  --normal-manifest $VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
  --remote-cache-dir $VES/vesuvius_fiber_trace_zarr_cache
```

- If the Python persisted-manifest comparison path exists or can be reached
  without new model inference, run it too and compare segment/restart
  decisions.
- Re-run a compile check for the touched C++ app/library.

## Expected Outcome

After these fixes, the remaining systematic gap should mainly be the excluded
raw-model-output versus persisted-product difference. If C++ still has extra
restarts, the next task should add a per-segment candidate-score dump to compare
the same persisted samples between Python and C++.
