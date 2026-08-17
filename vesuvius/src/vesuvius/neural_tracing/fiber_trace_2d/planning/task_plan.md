# Plan: existing VC3D strips for replay

## Implementation

1. Restore `LineViewBuilder` and its tests to their pre-replay form. Convert
   every replay component's trace points and sampled normals into a `LineModel`
   and take `buildLineViewSurfaces(line).lineSurface` using the exact default
   `LineViewConfig` used by the line probe: 21 cross rows and inferred
   half-width. Do not preserve replay-specific densification, tube-width strip
   geometry, normal repair, or alternate failure handling; an exception from
   the existing builder aborts visualization publication unchanged.
2. Mechanically extract the existing `vc_lasagna_line_probe` surface texture
   helper into `vc_core`. Preserve its raw-surface coordinate resizing, uint8
   requirement, trilinear sampling, blocking dependency preparation over the
   selected and coarser levels, and `sampleCoordsFineToCoarse()` call. Make the
   line probe and replay call that one helper; do not copy or alter its logic.
3. Require `--volume` to name a concrete OME-Zarr dataset array/group rather
   than the pyramid root; do not add a separate level selector. Resolve its
   base-to-group transform from the parent `multiscales` metadata, transform a
   sampling-coordinate copy, and pass the selected group to the unchanged
   renderer as a one-level volume. Persist the group path, transform, shape,
   and automatic native-grid contract.
4. Delete the complete alternate path: `buildLineSurfaceRibbon`, its changed
   caller/test, `renderFiberReplayStripCt`, fixed-level validation, replay-only
   raw ribbon/intensity state not needed by the shared helpers, the
   `render_surface_image()`/mask path, uint16 support, and PNG contract. No
   dormant compatibility route remains.
5. Mechanically extract the existing line-probe `TexturedMesh` construction,
   OBJ/MTL conventions, and uncompressed TIFF writer alongside its renderer.
   Assemble rendered disconnected components into one atlas per trace kind by
   applying only an affine atlas transform to each component's existing 0..1
   UVs. Publish standard OBJ/MTL/TIFF triples. This atlas combination is the
   sole new packaging step; per-component geometry and surface-to-texture
   mapping remain the existing implementation.
6. Update the strict napari reader for TIFF atlases and scale-aware UVs. Since
   napari does not render OBJ UV textures, tessellate the validated surface to
   one displayed vertex per stored texel. It continues to read only hashed
   self-contained artifacts and never opens the CT source.

## Tests

1. C++: capture the pre-extraction line-probe surface geometry/OBJ/MTL/TIFF
   conventions in fixtures, then prove the shared helper preserves their exact
   coordinates, texture samples, and serialized output. Prove replay stores the
   returned default line surface, resolves and persists the explicitly selected
   group's OME transform, samples group voxels at transformed coordinates, and
   publishes all three OBJ/MTL/TIFF sets.
2. Python: prove strict TIFF/MTL/UV loading, disconnected and empty layers,
   malformed artifact rejection, and reload behavior.
3. Build affected targets with `-j32`; run focused C++ and Python tests, Ruff,
   and `git diff --check`.
4. Regenerate a bounded Paris4 visualization with `--volume <ct.ome.zarr>/2`, strictly
   load it, and verify that the three non-empty textures contain CT values.

## Spec Update

- Replace the erroneous legacy-renderer/PNG contract with the existing VC3D
  strip geometry and fine-to-coarse texture-rendering contract.
- Require and persist the selected OME-Zarr group path, transform, and native-grid contract.
- Retain strict self-contained disconnected trace artifacts.
- Replace the contradictory PNG/uint16/level-zero/ribbon-helper/mask section
  before treating the updated spec as the validation authority.

## Documentation Updates

- Document the concrete OME-Zarr group-path requirement and a Paris4 `/2`
  example in `volume-cartographer/docs/fiberlets.md`.
- Correct the current changelog, status, and task log; explicitly record that
  no renderer, sampler, cache, or pyramid behavior changed.

# Plan: native-resolution replay strip textures

1. Derive each disconnected strip component's texture width and height from
   its arc extent in the explicitly selected Zarr group's voxel coordinates.
   Resample only the coordinate grid to at most one group voxel per texel, then
   call the unchanged shared fine-to-coarse renderer at scale one.
2. Remove the fixed replay `--strip-render-scale` control and metadata. Permit
   different native heights for disconnected components and retain the
   standard OBJ/MTL/TIFF atlas with one-pixel replicated tile borders.
3. Keep the persisted OBJ as the standard default `buildLineViewSurfaces()`
   mesh. In the napari reader, validate its UV atlas and bilinearly tessellate
   the mesh to the stored native texture grid so every CT texel becomes one
   displayed surface value instead of discarding intermediate texels.
4. Add C++ coverage for group-resolution sizing and variable-size atlas
   publication, Python coverage for strict atlas inference and dense display
   geometry, then rebuild and run the focused suites with `-j32`.
