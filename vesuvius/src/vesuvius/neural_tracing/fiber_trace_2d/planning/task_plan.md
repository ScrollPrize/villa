# Native 3D Trace2CP Tool Plan

## Current Findings

- The current 3D Trace2CP path is a projection bridge. It builds a 2D
  Trace2CP segment strip with `FiberStrip2DLoader`, tiles 3D inference over
  that strip, projects decoded 3D directions into strip x/y, and then calls the
  existing 2D Trace2CP scorer.
- The 2D combined Trace2CP scorer samples a candidate fan around the local
  ambiguous direction, scores direction agreement at the current and candidate
  positions, optionally adds `1 - presence`, and chooses the lowest score.
- Native 3D tracing needs a different stop condition: crossing the target CP
  plane in 3D, not crossing a 2D x-column.
- Existing strip builders can still be reused for visualization if we feed them
  a traced 3D line with proper Lasagna/VC3D frame construction. The adapter
  should preserve current side/top strip semantics, not replace them with
  planar debug slices.

## Implementation Plan

1. Add a separate native 3D Trace2CP CLI.
   - Add a new module such as
     `vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool`.
   - Keep the existing `fiber_trace_3d.train --trace2cp-vis` projected bridge
     unchanged unless a later migration is explicitly requested.
   - CLI inputs:
     - config path;
     - checkpoint path;
     - `--sample-index` or `--fiber-json`;
     - `--export-dir`;
     - step size, cone angle/spacing, direction/presence weights;
     - inference patch shape and trusted-core crop margin.

2. Define native 3D trace coordinate conventions.
   - Trace internally in selected Zarr-level voxel coordinates, because that is
     the model-output coordinate system.
   - Convert CPs and traced points to/from base XYZ only at loader/strip
     boundaries.
   - Decode direction with the existing analytic Lasagna 3x2 decoder.
   - Treat direction as sign-ambiguous and align every sampled direction to the
     previous step direction before scoring.

3. Add an inferred-block cache/router.
   - Build on-demand model inference blocks from the base volume.
   - Each block has:
     - input origin and input shape;
     - output tensor;
     - trusted core bounds after cropping away `core_margin` voxels from the
       output patch border;
     - volume-valid mask for the trusted core.
   - The block grid stride is the trusted core size, so adjacent input patches
     overlap by `2 * core_margin`.
   - Point lookup routes to the block whose trusted core contains the point.
     If the block is missing, infer it lazily.
   - Querying a point outside all trusted cores after block construction is an
     explicit error, not a silent fallback to edge output.
   - Candidate scoring samples direction and presence from this structure with
     trilinear interpolation on model outputs.

4. Implement 3D cone candidate stepping.
   - At the current point, sample the decoded 3D direction and align to the
     previous step.
   - Construct an orthonormal basis around that direction.
   - Generate candidates inside the cone with a deterministic ring/azimuth
     pattern, including the center direction.
   - For each candidate:
     - compute next point as `current + candidate * step_voxels`;
     - reject invalid/out-of-volume candidates;
     - sample candidate-point direction and align to the candidate vector;
     - compute direction loss as the mean of current and candidate endpoint
       angular agreement, using `1 - dot(...)` like the 2D combined scorer;
     - compute presence loss as `1 - presence(candidate)`;
     - total loss is weighted direction plus weighted presence.
   - Select the lowest total score, tie-breaking by smaller cone angle and
     stable candidate order.

5. Stop on target-plane crossing.
   - For forward tracing, use the plane through the target CP with normal
     `normalize(target_cp - start_cp)`.
   - Detect crossing by a sign change of the signed plane distance between the
     current and next point.
   - Append the exact linear interpolation point on the target plane.
   - Run the same procedure in reverse with the opposite CP order.
   - Keep a generous `max_steps` guard and make max-step termination visible in
     stdout/summary.

6. Build metric/debug results.
   - Return forward and reverse 3D polylines, reasons, per-step mean direction
     and presence losses, total candidate count, rejected candidate count, and
     target-plane crossing status.
   - Initially report a 3D endpoint-plane/closest-approach metric for the
     native tool, separate from the existing public 2D `trace2cp_error`.
   - Do not replace training best-model selection with this native metric until
     it is validated.

7. Reuse strip creation for visualization.
   - Convert traced selected-level points back to base XYZ.
   - Add a local adapter that creates a Trace2CP-style side/top strip source
     directly from an arbitrary traced 3D line:
     - sample Lasagna normals at traced line coordinates;
     - build VC3D-equivalent side and top strip grids from the traced line;
     - sample the volume with the existing VC3D sampler.
   - The original 2D Trace2CP source strip must not be used as a hard spatial
     domain for native 3D trace visualization. It may only supply record/CP
     metadata and visual sizing defaults.
   - This should be an adapter around existing strip geometry semantics and
     should not change current 2D Trace2CP behavior.
   - Render side strip, top strip, projected forward/reverse/fused traces, CPs,
     presence/debug panels, and a concise summary text.

8. Keep projected 3D Trace2CP as a separate baseline.
   - The existing bridge remains useful as a comparison because it uses the
     mature 2D scorer.
   - The native tool output should clearly label itself as native 3D tracing.

## Spec Update

- Add a native 3D Trace2CP tool spec alongside the existing projected 3D bridge.
- Document the inferred-block cache/router:
  model input patch, trusted core crop, overlap stride, trilinear lookup, and
  no edge-output fallback.
- Document 3D cone candidate selection and direction/presence scoring.
- Document target-plane crossing as the native 3D stop condition.
- Document that native 3D Trace2CP does not replace the existing public 2D
  `trace2cp_error` or best-checkpoint selection until explicitly enabled.
- Document that native 3D strip visualization reuses VC3D/Lasagna strip
  semantics from traced 3D coordinates and is not clipped to the original
  Trace2CP source strip.

## Docs Updates

- Update `docs/code_structure.md` with the new 3D Trace2CP tool module,
  inferred-block cache, tracing loop, and strip-output adapter.
- Add the intended command form and artifact names.
- Clearly distinguish:
  - projected 3D-to-2D Trace2CP bridge;
  - native 3D cone/plane Trace2CP tool.

## Testing Plan

- Unit-test target-plane crossing and interpolation.
- Unit-test 3D direction ambiguity alignment.
- Unit-test cone candidate generation for deterministic ordering and bounded
  angular radius.
- Unit-test inferred-block routing:
  - a point in a trusted core routes to that block;
  - a point near an output border routes to the neighboring overlapped block;
  - no lookup samples from cropped-away output margin.
- Unit-test candidate scoring with synthetic fields:
  - constant direction reaches the target plane;
  - high-presence off-axis candidate can win only according to configured
    weights;
  - invalid candidates are rejected.
- Smoke-test the CLI on a tiny/fake model or a small configured sample and
  verify output files and stdout labels.
- Keep existing focused 3D and 2D tests passing:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

- Add a 2026-07-16 planning entry for native 3D Trace2CP cone tracing with
  overlapped inference-block lookup and target-plane stopping.

## Deviations Or Deferrals

- No implementation in this planning step.
- The native 3D metric is planned as tool-local debug output first. It should
  not replace the existing projected Trace2CP test metric or best-checkpoint
  selection in the first implementation.
