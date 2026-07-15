# Process-Level Startup Fiber Geometry Preload

The loader must preload compact fiber-line geometry at startup by splitting
fibers/records across independent worker processes. Each process should do its
assigned fiber work independently, including opening the needed volume/Lasagna
handles inside that process. The parent process must collect the compact
geometry results, store them in deterministic record order, and keep one shared
in-memory geometry store for subsequent training/prefetch/runner use.

This replaces the incorrect thread-local zarr/VC3D handle direction. Do not
lazy-build geometry during the first training pass, and do not use per-thread
record handle caches as the solution for startup preload.

Keep these existing requirements:

- Build the compact geometry store during `FiberStrip2DLoader` construction.
- Show startup progress while building the store.
- `loader_workers=0` means all logical CPU cores.
- `loader_workers=1` remains the serial/debug path.
- Final geometry order, deterministic sample order, skip behavior, and compact
  geometry contents must match the serial path.
- The compact geometry store must not be duplicated after startup; worker
  processes return compact results to the parent, and the parent owns the
  shared store.
- Parallel startup must not share opened zarr/VC3D/Lasagna objects between
  workers.
