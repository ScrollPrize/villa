# Plan: Bright material colors for BP state lines

## Contract

- Remove the optional per-vertex RGB API introduced for direct BP state files.
- Extract the existing private OBJ token validation and MTL serialization from
  the textured-mesh implementation into a shared OBJ-material helper. Make the
  existing texture material writer and the new material-backed polyline writer
  use that helper; do not add a second MTL implementation.
- Emit coordinate-only `v x y z` records, `mtllib` and `usemtl` records in the
  OBJ, and a sibling MTL defining the material used by every `l` or `p`
  primitive in that file.
- Use saturated normalized material colors: H orange `(1.00,0.45,0.00)`, V
  cyan `(0.00,1.00,1.00)`, Mixed magenta `(1.00,0.00,1.00)`, and exact ties
  lime `(0.55,1.00,0.00)`. Set ambient and diffuse to the full color, specular
  to zero, opacity `d 1`, and illumination model 1 so MeshLab does not reduce
  these to near-black vertex attributes.
- Preserve the existing unmaterialed polyline writer and every existing caller.
- Reject invalid colors and material tokens in the shared API.
- Name the sidecar `<obj-stem>.mtl`, reference only its local filename, and
  return both paths from the shared writer. Atomically publish the MTL first and
  the OBJ last so the OBJ is the bundle publication marker.

## Implementation

1. Extract shared OBJ token/reference and MTL serialization from the existing
   textured-mesh helper without changing existing texture artifacts.
2. Replace the colored-vertex overload with a clearly named shared
   material-backed writer that emits OBJ and MTL siblings.
3. Use it for the four mutually exclusive direct BP state layers.
4. Regenerate the centered-1024 output in its existing main directory.

## Spec Update

Replace the vertex-RGB statement with material-backed line colors, exact
palette, and sibling artifact semantics.

## Docs Updates

Document that each direct state OBJ requires and automatically references its
sibling MTL file.

## Testing

- Verify colored OBJs retain exactly XYZ vertex fields and unchanged line/point
  indices, reference the expected MTL, and select its material before geometry.
- Verify MTL material name, ambient/diffuse/specular values, and illumination.
- Verify the legacy writer creates no material records or sidecar.
- Cover empty state groups, local-basename references, MTL-before-geometry
  ordering, atomic bundle paths, and rejection of malformed output tokens.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused test suite, inspect regenerated artifacts, and run
  `git diff --check`.
- Ask the user to confirm the regenerated lines render with the four material
  colors in MeshLab; record that visual check separately from automated tests.

## Changelog

Record the correction in the current task log; no durable changelog entry is
needed for this unshipped visualization iteration.
