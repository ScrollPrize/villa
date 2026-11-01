# Shared 3D canvas

`vc3d::scene3d::Scene3DCanvas` is a display-only Qt OpenGL 3.3 widget. It
supports CT planes, volumetric CT slabs, and a surface mesh without a fiber
scene or annotation controller. `fibers3d::Fiber3DCanvas` extends it with fiber
geometry, selection, ribbons, filtering, and annotation overlays. Existing
fiber panels use the same shared camera and CT/surface rendering implementation.

Controllers supply immutable snapshots using `setCTSlices()` and
`setSurfaceMesh()`. Sampling and I/O stay outside the canvas. The existing
`slices::sample()` and `surface::build()` helpers can prepare CT data and a
mesh from a tifxyz surface's coordinate matrix. Their implementations and
snapshot types remain in `fibers3d/` for source compatibility; the shared
canvas exposes `scene3d::slices` and `scene3d::surface` namespace aliases.
All geometry uses level-0 CT voxel coordinates. Call `fitAll()` after replacing
content when a new camera framing is desired. Empty slices and a null mesh
remove those layers independently.

The camera provides orbit, pan, zoom, orthographic/perspective projection,
`focusPoint()`, and `viewScaleChanged()`. Controllers can handle
`positionClicked()`, `emptyContextRequested()`, and `clearSelectionRequested()`.
A subclass can extend painting using `beginFrame()`, `drawSurface()`, and
`drawCTSlices()` (the fiber view inserts its geometry before translucent CT).
GL resources must be released with their context current, before shared context
cleanup; see the fiber subclass for context destruction and recreation handling.
`Scene3DCanvasState.hpp` is internal to the rendering implementations.

This change does not add a new dock or a whole-volume streaming loader. The CT
renderer still accepts the existing bounded slabs and transfer settings; a
controller decides which region and pyramid level to sample.

Build and run from `volume-cartographer/` with existing dependencies:

```sh
AGENTS_AGENT_MODE=1 ninja -C build VC3D test_fiber_3d_canvas
build/bin/VC3D
build/bin/test_fiber_3d_canvas
```

The Qt tests require a display with OpenGL 3.3. They cover standalone slices,
volumetric slabs, surface visibility/removal, and existing fiber rendering and
asynchronous scene replacement. These rendering fixtures are regression checks,
not a substitute for visual acceptance on real scroll data.
