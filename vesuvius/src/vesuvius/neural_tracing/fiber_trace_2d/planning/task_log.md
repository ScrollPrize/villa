# Task Log: restore aggregate fiberlet replay as the default

## Correction

- The first implementation attempt changed `costProfileWeight` from `1` to
  `0`. That was not the requested evaluator: profile weight zero still loads
  decoded subsegment profiles and walks the checkpoint-rooted integration grid.
- The desired baseline uses the stored aggregate `edge.cost` and transition
  cost directly, prorating only a partial boundary fiberlet. It therefore
  avoids route-profile reads and stepped integration entirely.
- The incorrect default-only edit was not committed. This task replaces it
  with an explicit evaluator mode and keeps stepped scoring opt-in.

## Validation

- Independent review corrected the scoring boundary: aggregate loss is from
  the segment seed through the common horizon. The checkpoint only controls
  commitment; it does not truncate aggregate scoring. The implementation and
  tests cover this for exact and intermediate-pruned search.
- Added `FiberletGraphReplayCostMode::{Fiberlet,Stepped}`. `Fiberlet` is the
  API/CLI default and uses authoritative whole-edge/join costs without calling
  `costProfile`. `Stepped` retains the prior profile/grid evaluator and its
  `A=1` default.
- Core and CLI reject stepped-only settings in aggregate mode. Replay JSON
  records `cost_mode` and omits inactive stepped fields for aggregate output.
- Built with 32 jobs:
  `cmake --build volume-cartographer/build --target vc_fiberlets test_fiberlet_paths test_fiber_trace3d test_fiber_replay test_fiberlet_storage -j32`.
- Passing suites: `test_fiber_trace3d` (55), `test_fiber_replay` (12), and
  `test_fiberlet_storage` (17).
- `test_fiberlet_paths` has the same 298 pre-existing failures at lines 414 and
  1026-1028; no new failure remains. New coverage includes default-versus-
  stepped selection, profile-free exact/bounded search, partial horizon/join
  ownership, inactive-option rejection, and JSON mode fields.
- CLI smoke checks confirmed `[fiberlet]`, rejected `--cost-weight` without
  `--cost-mode stepped`, and accepted it with explicit stepped mode.
- Hot-cache comparison used the Paris4 fiber prediction, David reference fiber,
  Lasagna normals, 5,000 base voxels, radius 64, default exact search, and the
  existing `volume-cartographer/build` build. Aggregate: 0.35 s wall, 1.79 s
  user, 0.73 s system, 94,108 KiB peak RSS. Stepped `W=1,A=1`: 0.35 s wall,
  1.71 s user, 0.52 s system, 95,284 KiB peak RSS. Both had zero fiberlet
  failures. This short graph is below useful timing resolution; it validates
  shared-cache behavior but does not supersede the wide-corridor measurements.
- `git diff --check` passes.
