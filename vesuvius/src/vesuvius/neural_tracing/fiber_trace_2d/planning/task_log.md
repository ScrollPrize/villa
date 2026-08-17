# Task log: Fiberlet-centered replay strips and indexed overview JPEGs

## Findings

- The existing full overview builds only one reference-centered
  `buildLineViewSurfaces()` pair and projects both tracers onto it.
- Long strips are currently divided into 32,000-column ranges and stacked into
  one JPEG. Width is bounded, but sufficiently many panels can exceed the JPEG
  height limit.
- Fiberlet replay segments already retain one monotonic matched reference arc
  per route point, which is sufficient to project reference/greedy comparison
  geometry and failure arcs into a fiberlet-centered strip without rematching.
- The user clarified that long-strip ranges should remain vertically wrapped in
  one image. Each wrapped range must contain all four strips; indexed spill
  files are needed only after complete blocks would exceed the image limit.
- Independent review required retaining the existing 32,000-column proportional
  ranges, packing only complete blocks across pages, recording component/page
  placement explicitly, and defining match-arc behavior across fiberlet reset
  components. The detailed plan now includes those contracts.

## Deviations

- None.

## Validation

- Built the affected targets with all 32 requested jobs:

  `cmake --build volume-cartographer/build --target test_fiber_replay vc_fiberlets -j32`

- Ran `volume-cartographer/build/bin/test_fiber_replay`: all 9 test cases
  passed. Coverage includes four-strip composition, exact half-open column
  copying, complete-block page spilling under an artificial row bound, strict
  component metadata, indexed immutable/stable publication, deterministic
  rewrites, and stale indexed/legacy alias removal.
- Ran `git diff --check`: passed.
- Ran a bounded Paris4 visual replay over 512 base voxels with the rebuilt
  `vc_fiberlets`, `fiber_s1_002.lasagna.json`, David fiber
  `dj_20260805T025256484_000003.json`, Lasagna normals `las_008`, and concrete
  CT group `20260411134726-2.400um-0.2m-78keV-masked.zarr/2`.
- The run completed with zero greedy and zero fiberlet failures and published
  `/tmp/fiber-replay-four-strip-validation/fiber_replay.000000.jpg` at
  1144x2110 pixels. The strict root records one fiberlet component and four
  unwrapped shapes: reference top 336x1056, reference side 336x1048, fiberlet
  top 648x1128, and fiberlet side 648x1144. Direct inspection confirmed all
  four CT strips and yellow/red/cyan overlays are visible.
