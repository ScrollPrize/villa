# Native Fiber Trace Meeting Search And Persisted Diagnostics

## User Request

- Make CP-to-CP native tracing fall back to Lasagna substantially less often.
- Run both one-way tracers until they reach the opposite endpoint's local
  target planes within the configured endpoint threshold or exhaust their
  maximum step budgets.
- After tracing, move locally tangent planes along both traces and intersect
  the opposite trace at frequent intervals.
- Select the endpoint-plane or moving-plane meeting with the smallest spatial
  error and accept it when that error is at most 10% of the selected traced
  path length.
- Construct the final CP-to-CP line with the existing Python fusion-point
  arc-length lerping behavior translated to shared C++.
- Below the generated strip, show native meeting error in base voxels for
  accepted native spans and the native failure reason for Lasagna fallbacks,
  instead of a Lasagna normal-alignment error.
- Persist that diagnostic per segment on the existing owning CP so it remains
  visible after saving and reloading.

## Required Semantics

- Preserve the current target-local multi-plane construction; do not introduce
  a straight CP-chord target plane.
- Endpoint-plane success uses the existing 20-base-voxel threshold and tracing
  continues after an out-of-threshold crossing while budget remains.
- Moving-plane error is the Euclidean distance within the plane between the
  plane's source-trace sample and the interpolated opposite-trace crossing. It
  is not signed point-to-plane distance.
- Search symmetrically by moving planes along both traces.
- The 10% acceptance denominator is the selected forward partial arc length
  plus selected reverse partial arc length. Store the scale-independent ratio
  and the raw error converted to base voxels.
- The smallest raw error wins. Deterministic tie-breaking may prefer a more
  balanced/later meeting but must not replace raw error as the primary key.
- Accepted geometry retains exact CP endpoints.
- A failed native attempt stores its reason on the same CP-owned segment record
  but does not protect the Lasagna fallback geometry as native.

## Scope

- Shared C++ native CP-pair trace termination, meeting search, fusion, result
  diagnostics, and tests.
- VC3D mixed native/Lasagna optimization and direct segment action.
- VC3D CP-owned segment metadata, JSON persistence, generated-strip labels,
  and reload behavior.
- Strict C++/Python/script readers of the versioned segment metadata.
- Specifications, implementation documentation, changelog, and workflow
  records.

## Out Of Scope

- Changing whole-fiber restart/continuation semantics.
- Changing the model, prediction scoring, normal-aware smoothness, or beam
  search numerics.
- Adding actual 4D volume handling.
- Changing ordinary Lasagna normal-alignment metrics for spans without a
  native trace attempt.
