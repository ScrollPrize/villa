# Plan: Color direct BP state OBJ layers

## Contract

- Add optional uniform per-vertex RGB output to the shared polyline OBJ writer.
- Use MeshLab-compatible `v x y z r g b` records without companion MTL files.
- Color only the four direct BP argmax layers with normalized RGB values:
  H `(1.00, 0.35, 0.05)`, V `(0.05, 0.80, 1.00)`, Mixed
  `(1.00, 0.10, 0.75)`, and exact ties `(0.60, 1.00, 0.10)`.
- Preserve coordinate, object, point, and line records and leave every existing
  caller uncolored unless it explicitly supplies a color.
- Preserve the existing writer call contract by adding a colored overload whose
  color follows the existing comment argument. Reject non-finite or out-of-range
  normalized color components.

## Implementation

1. Add a small RGB value type and optional color argument to the shared writer.
2. Assign the four state colors in the ternary-state artifact writer.
3. Verify colored state files and legacy uncolored output in focused tests.

## Spec Update

Specify the exact normalized palette and MeshLab-compatible vertex RGB
representation for the four direct BP state layers.

## Docs Updates

Document the colors and that MeshLab loads them from OBJ vertex records without
an MTL sidecar.

## Testing

- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`.
- Run `test_fiberlet_crop_trace`.
- Parse vertex token counts: colored records contain exactly XYZRGB and default
  records exactly XYZ; preserve object, line, point, and index records.
- Run `git diff --check`.

## Changelog

This is a small visualization tweak; record it in the current task log only.
