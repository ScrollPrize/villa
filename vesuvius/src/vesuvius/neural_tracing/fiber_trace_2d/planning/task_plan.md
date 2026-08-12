# Plan: reduce fiber-anchor NMS radii

## Implementation

1. Add an explicit `nmsTransverseRadiusPredictionVoxels` anchor setting with a
   default of 2. Set the longitudinal default to 1 prediction voxel.
   Remove the CLI resolver's cell-size-derived longitudinal assignment; neither
   NMS radius may derive from cell size or the refinement window.
2. Use the explicit transverse radius for NMS neighbor classification, spatial
   binning, and external-context enumeration. Leave peak refinement and its
   local-window radius unchanged. External-context pivot reach remains
   conservative as refinement window plus the new NMS ellipsoid radius.
3. Serialize the transverse NMS radius in final and diagnostic anchor JSON and
   require it in the strict C++ anchor reader used by fiberlet path tracing.
   Do not add compatibility handling for older experimental artifacts.
4. Report both NMS radii in CLI output as base voxels and document that values
   are prediction-volume voxels internally/artifacts. Do not add NMS CLI
   overrides in this default-only change; change `--window` wording to
   refinement only.

## Tests

1. Extend focused NMS tests to distinguish the transverse NMS radius from the
   larger refinement window and verify inclusive transverse/longitudinal
   boundaries.
   Preserve a cropped external suppressor that can move through the full
   refinement window, and verify defaults do not vary with cell size.
2. Update strict artifact and path-reader fixtures for the new field and verify
   missing/incorrect schema rejection remains strict.
3. Build affected C++ targets with `-j32` and run focused anchor/path/replay
   tests.
4. Run the Python viewer tests because it consumes diagnostic artifacts, while
   confirming its visualization-only parameter handling remains decoupled.

## Spec Update

Specify independent default NMS radii of 2 transverse and 1 longitudinal
prediction voxels and remove wording that couples transverse NMS to refinement.

## Docs Update

Update fiberlet command/algorithm documentation and code-structure notes.

## Changelog

Record the narrower independent NMS defaults.
