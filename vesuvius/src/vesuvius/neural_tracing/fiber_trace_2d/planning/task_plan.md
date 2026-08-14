# Plan: straightened fiberlet DP and scored graph joins

## Implementation

1. Replace global integer half-grid states with deterministic candidate-local
   `(longitudinal_layer, transverse_u, transverse_v)` states. Fit the existing
   cubic-Hermite centerline from both anchor positions and oriented directions,
   tabulate it by arclength, and place planes at approximately 2-prediction-
   voxel arclength intervals with an exact final endpoint plane.
2. Use the normalized Hermite derivative as each plane normal. Construct the
   first transverse basis from a deterministic least-aligned world axis, then
   propagate it by minimal-rotation parallel transport and re-orthogonalize.
   Map each state to actual prediction XYZ as
   `center(s) + 0.5*u*axis_u(s) + 0.5*v*axis_v(s)`.
3. Connect each ordinary layer only to the next layer and permit transverse
   index changes in `{-1,0,1}` on each axis. Keep the exact start anchor as the
   source, use the exact target anchor as the sink, and permit a shorter final
   longitudinal remainder without rounding either endpoint. Retain corridor,
   volume, tube, endpoint-angle, and sampled-prediction-angle checks in actual
   prediction XYZ coordinates. Retain incoming predecessor identity in every
   DP state because alignment and curvature are second-order. Score every edge
   using its actual mapped XYZ direction and Euclidean length, not nominal grid
   spacing. Enumerate finite transverse indices from the corridor radius and
   reject degenerate/nonfinite Hermite frames deterministically.
4. Refactor the native-corner interpolation so arbitrary finite prediction XYZ
   points use the existing sign-invariant axis-tensor direction interpolation,
   trilinear presence, strict invalid-prediction handling, and invalid-normal
   isotropic fallback. Preload and deduplicate only the native integer corners
   required by the candidate-local nodes and exact endpoints; do not retain a
   global derived half-grid cache.
5. Record the effective longitudinal and transverse grid spacing in the
   experimental fiberlet artifact and remove the obsolete world-axis half-grid
   parameter and implementation without compatibility handling.
6. Extract the complete local path metric wrapper from `FiberPaths.cpp` into
   shared fiber-tracer code. It owns prediction-axis sign alignment, presence,
   invalid prediction cost, alignment length integration, Lasagna-normal
   tangent/normal splitting, free angle, and effective-length normalization.
   Both the DP and graph joins call this wrapper. Set the common fiberlet/join
   free angle to zero; the previous 45-degree dead zone existed only for the
   coarse world-axis lattice and would erase all join smoothness below the hard
   join limit.
7. Retain exact interpolated dense prediction and Lasagna-normal samples at each
   anchor for graph construction; fitted anchor support is not point presence.
   Repeated references to one anchor must agree. Invalid anchor predictions
   admit no join; invalid normals use the shared isotropic fallback. For every
   admissible incoming/outgoing arc pair, calculate and store a transition cost
   from the shared fiberlet local alignment and tangent/normal smoothness
   wrapper using the incoming and outgoing dense endpoint directions and
   physical step lengths. Alignment uses the outgoing segment length;
   smoothness uses `max(1,(incoming_length+outgoing_length)/2)`. This cost is
   additive to existing per-fiberlet endpoint proxy costs and adds no route
   length.
8. Keep the strict `<45 degree` graph join as a hard feasibility bound. Add the
   transition cost to lookahead expansion, pruning density, committed replay
   totals exactly once, and graph/replay JSON component diagnostics. The first
   arc has no preceding join cost; edge-only physical length remains the route
   density denominator.

## Tests

1. Replace half-grid assertions with a curved endpoint-direction fixture proving
   that interior points lie on the Hermite-normal 2-by-0.5-by-0.5 local grid,
   contain non-world-half-grid coordinates, retain exact endpoints, follow the
   endpoint tangents, and remain monotonically layered.
2. Cover arbitrary-point interpolation, antipodal direction interpolation,
   invalid required corners, boundary points, endpoint remainders, strict
   25-degree prediction rejection, near-multiple centerline lengths,
   deterministic frame ties, and deterministic reruns. Tube predicates apply
   to floating evaluation points while every positive-weight native corner is
   globally deduplicated and preloaded even outside the tube.
3. Add graph tests proving a lower-loss join wins over an otherwise preferable
   edge sequence, transition components are serialized, the 45-degree bound is
   still strict, reverse arcs use the correct endpoint sample, and receding-
   horizon replay totals include each committed join once without replanning
   double counts. Compare DP and graph outputs from the shared metric helper for
   identical local inputs.
4. Build `vc_fiberlets`, `test_fiberlet_paths`, and `test_fiber_replay` with
   `-j32`; run the fiber-focused CTest set.
5. In the existing build configuration, run the supplied Paris4
   `fiberlet-replay --along 512` command with 32 workers once for the initial
   expensive comparison. Compare against the recorded same-build half-grid
   baseline using wall/user time, peak RSS, globally deduplicated native-corner
   preload count, total candidate-local DP nodes, searched/accepted fiberlets,
   graph size, route progress, maximum/final route error, and stop reason.

## Spec Update

Replace the world-axis half-grid definition with the Hermite-centered,
parallel-transported layered domain and exact floating-point sampling rule.
Specify graph-transition loss as part of the route objective in addition to the
hard join-angle constraint.

## Docs Update

Document the Hermite centerline, parallel-transported coordinate frame,
2-by-0.5-by-0.5 search resolution, endpoint remainder behavior,
arbitrary-point interpolation, graph join loss, and performance/curvature
tradeoffs.

## Changelog

Record the candidate-local fiberlet DP and scored graph joins.
