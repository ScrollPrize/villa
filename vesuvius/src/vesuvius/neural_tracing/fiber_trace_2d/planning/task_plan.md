# Native VC3D 3D Fiber Tracer Plan

## Scope

- Implement a native volume-cartographer C++ Trace2CP segment tracer for 3D
  fiber inference outputs.
- Consume precomputed fiber inference OME-Zarr/Lasagna-style datasets. Do not
  run PyTorch/model inference in VC3D for this task.
- Reuse existing VC3D/Lasagna local and remote volume access, chunk caching,
  metadata, and normal sampling facilities.
- Add first GUI integration as a Ctrl-right-click segment optimization action
  in the line annotation window.
- Keep this task segment-local. Whole-fiber tracing and visualization are out
  of scope for this first native port.

## Reference behavior to port

- Match the Python command-line reference behavior used with:
  `--beam-lookahead-steps 2 --beam-width 8 --smoothness-normal-weight 0.1
  --smoothness-tangent-weight 10.0 --core-margin-voxels 48
  --inference-patch-shape-zyx 128 128 128 --inference-scaledown-power 2`.
- Port the effective tracing behavior, not the debug/visualization code.
- Use CP-local fiber tangent toward the target CP as the initial reference
  direction, not the straight CP-to-target chord.
- Choose the first model direction branch by alignment with that CP-local
  tangent.
- Run bidirectional CP-to-CP tracing and then use the Python center-fusion
  behavior for the accepted segment.
- Preserve multi-direction semantics: direction and presence channels for a
  branch are evaluated as one option.

## Implementation steps

1. Pin the exact Python reference semantics
   - Inspect the current `fiber_trace_3d.trace2cp_tool` scoring, bidirectional
     stop condition, endpoint error calculation, and fusion implementation.
   - Record the exact defaults and units in the new C++ config.
   - Add small Python fixture outputs if needed so the C++ implementation can
     be checked against known candidate/fusion behavior.

2. Add project-level fiber inference dataset storage
   - Extend the VC3D project/session dataset model in the same style as
     Lasagna datasets.
   - Store fiber inference dataset manifest/path/URL, remote cache information,
     and any display name needed to reopen it later.
   - Do not hardcode the fiber inference location in the tracer or GUI.
   - Keep local and remote paths flowing through the existing VC3D volume and
     Lasagna dataset facilities.

3. Add a separate native tracer core library
   - Add a new built-in CMake target, for example `vc_fiber_tracer`.
   - Keep Qt/UI code out of this library.
   - Public types should include:
     - `FiberTraceConfig`
     - `FiberPredictionField`
     - `FiberTraceSegmentRequest`
     - `FiberTraceSegmentResult`
     - progress and cancellation callback interfaces.
   - Link to existing VC3D core/Lasagna libraries for volume access and normal
     sampling.

3a. Extract shared Lasagna compact-channel helpers
   - Move the currently private Lasagna compact `nx/ny` decode, tensor
     reconstruction, trilinear scalar sampling, and chunk-cache mechanics used
     by `LasagnaNormalSampler.cpp` into an exported `vc_lasagna` helper.
   - Port `LasagnaNormalSampler` to call the shared helper.
   - The new fiber tracer must call this helper instead of copying the private
     implementation into the tracer library.
   - This extraction is a required first implementation step because copied
     compact-normal sampling logic is not acceptable.

4. Implement fiber prediction field access
   - Open the precomputed fiber inference dataset through existing
     `Volume`/Lasagna manifest paths.
   - Read persisted `presence`, `nx`, and `ny` products per option.
   - Decode presence as fixed-point `uint8 / 255.0`.
   - Decode direction with the same Lasagna compact ambiguous hemisphere
     encoding used by the Python fiber inference output. Do not raw-interpolate
     ambiguous direction vectors as ordinary vectors.
   - Use the shared compact-channel helper from step 3a for both scalar
     presence sampling and tensor-based compact direction sampling.
   - Support local and remote datasets through the existing chunk cache and
     remote volume code.

5. Implement native CP-to-CP bidirectional search
   - Convert line annotation coordinates to the inference dataset coordinate
     space explicitly and keep conversions localized.
   - Generate the cone candidate set around the current sampled direction using
     the reference angular spacing and cone limit.
   - Score candidates with the same ingredients as the Python tracer:
     - last step direction;
     - current sampled model direction;
     - candidate step direction;
     - candidate sampled model direction;
     - candidate presence;
     - tangent/normal smoothness using Lasagna normals sampled at candidate
       points;
     - cumulative tangent smoothness where enabled by the reference config.
   - Run the beam/lookahead search in both directions with progress callbacks.
   - Stop each direction at the target CP plane using the same target-plane
     convention as the Python reference.

6. Apply endpoint threshold and fusion
   - Compute endpoint in-plane errors in base-volume coordinates.
   - Convert endpoint errors to um using only VC3D volume metadata
     (`voxelsize`/`Volume::voxelSize()`). If a positive voxel size is
     unavailable, fail explicitly and do not apply the optimization.
   - If either endpoint error exceeds 50 um, leave the fiber unchanged.
   - If accepted, find the forward/reverse fusion pair with the current Python
     score: `2 * closest_point_gap + forward_length + reverse_length`.
   - Build the fused center-lerped segment from that pair.
   - Replace only internal line points between the two CPs. Original CP
     coordinates must remain exact.

7. Persist tracer-optimized segment metadata
   - Extend stored fiber/session data with per-segment tracer metadata:
     - adjacent CP indices;
     - endpoint coordinates/signature;
     - optimized line-span signature;
     - maximum endpoint error in um;
     - tracer config/version.
   - Save/load this metadata with normal fiber JSON.
   - On CP move/delete or CP insertion inside the segment, invalidate the
     metadata and restore regular line-annotation behavior.

8. Protect optimized segments from regular re-optimization
   - Extend the existing line optimization path so unchanged tracer-optimized
     segments are treated as protected/fixed ranges.
   - Use the stored segment signature to decide whether the optimized segment
     is still unchanged.
   - If the signature no longer matches, discard the tracer metadata before
     running regular line optimization.

9. Add line annotation GUI integration
   - Add a Ctrl-right-click segment context menu action in generated line
     annotation views.
   - Resolve the clicked segment as the CP span around the clicked line
     position.
   - Dispatch the native tracer through `LineAnnotationController` as a
     background task.
   - While the task runs, block CP/line modifications and show a progress
     overlay/status.
   - On success, splice the optimized segment into the fiber, store tracer
     metadata, refresh views, and mark the fiber dirty for save.
   - On failure or threshold rejection, leave the fiber unchanged and report the
     reason.

10. Validation
    - Add core tests for direction decoding, branch selection, candidate
      scoring, endpoint thresholding, CP preservation, and fusion.
    - Add fiber JSON load/save tests for tracer metadata.
    - Add GUI/context-menu tests for the Ctrl-right-click action and edit
      blocking while tracing is active.
    - Run a small parity fixture against the Python reference before testing on
      real project data.
    - Run the relevant VC3D C++ tests after rebuilding.

## Spec update

- Add a VC3D native 3D fiber tracer section to `planning/specs.md`.
- Specify that native tracing consumes precomputed fiber inference datasets and
  does not run neural inference.
- Specify that fiber inference datasets are stored in the VC3D project in the
  same style as Lasagna datasets.
- Specify that remote/local volume access must reuse existing VC3D/Lasagna
  facilities.
- Specify CP-to-CP bidirectional search, endpoint rejection at 50 um,
  center-fusion-lerp application, and CP preservation.
- Specify tracer-optimized segment metadata, invalidation rules, and regular
  re-optimization protection.

## Docs updates

- Document how to configure/select a fiber inference dataset in a VC3D project.
- Document the Ctrl-right-click segment optimization workflow.
- Document accepted/rejected segment behavior and the meaning of the stored
  endpoint error.
- Document how native tracing relates to the Python reference command.

## Changelog update

- After implementation, add one changelog entry for the native VC3D 3D fiber
  Trace2CP segment tracer and GUI segment action.

## Risks and constraints

- The C++ implementation must not invent alternate direction decoding,
  interpolation, or remote-cache semantics.
- Endpoint thresholding requires valid voxel-size metadata. Missing physical
  metadata is an explicit non-apply error, not a guessed fallback.
- The exact fiber inference channel layout must be read from the current
  preprocessing output/manifest before implementation.
- The regular optimizer protection must be implemented through focused fixed
  segment/range handling, not by silently disabling unrelated optimization.
