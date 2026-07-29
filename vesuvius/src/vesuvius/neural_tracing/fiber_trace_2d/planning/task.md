# Native VC3D 3D Fiber Tracer Continuation

Continue the native VC3D 3D fiber tracer work. The first pass landed the
shared Lasagna compact-channel helpers, native segment tracer core, project
dataset storage, and GUI segment action, but the task is not complete.

Add a C++ command-line runner for the metric workflow:

- input is a preprocessed fiber inference `.lasagna.json` manifest and a
  `vc3d_fiber` JSON fiber;
- no visualization and no PyTorch/model inference;
- trace a single whole fiber in the same one-sided consecutive-CP mode used by
  the Python native 3D metric path;
- read persisted fiber inference `presence`/`nx`/`ny` channels from the
  manifest;
- use Lasagna normals from the manifest when available;
- report restart count, segment count, restarts per kvx, optional restarts per
  meter when an explicit voxel size is provided, and wall/CPU timing;
- implement configured multi-step beam lookahead in the native core so the
  metric runner can use the same lookahead knob as the Python benchmark
  command;
- keep the implementation reusable in `vc_fiber_tracer`; the CLI should be a
  thin wrapper and must not copy private helper implementations.

The existing known gaps remain visible until implemented: persisted
tracer-optimized segment metadata/invalidation, optimizer protection for
unchanged traced segments, numeric GUI progress, and real-data parity
validation against the Python reference.
