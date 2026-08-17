# Task: reuse the existing VC3D strip infrastructure for replay

Remove the replay-specific strip generation and rendering paths and use the
existing VC3D/line-probe infrastructure without changing its behavior.

- Build every disconnected trace strip through
  `buildLineViewSurfaces(...).lineSurface`.
- Reuse the existing line-probe fine-to-coarse surface texture renderer.
- Do not use `render_surface_image()`, generate a mask, add a new sampler, or
  change the volume/cache/pyramid implementations.
- Require `--volume` to name the concrete OME-Zarr array/group to render, so
  users can choose a fully present stored scale without a separate level option.
- Keep three self-contained napari layers for reference, greedy, and fiberlet
  traces, with disconnected segments in each layer.
- If implementation requires any behavioral deviation from the existing strip
  infrastructure, stop and ask before proceeding.
