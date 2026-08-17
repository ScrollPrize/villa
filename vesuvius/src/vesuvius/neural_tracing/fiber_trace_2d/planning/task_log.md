# Task Log: detailed replay overview and restart markers

## Findings

- The overview currently renders the native selected-group coordinate grid at
  scale one. The existing shared renderer already supports endpoint-preserving
  coordinate interpolation and subvoxel CT sampling through its render-scale
  parameter, so 8x does not require a new sampler.
- Full Paris4 is roughly 11,500 native group pixels longitudinally and would be
  roughly 92,000 pixels at 8x, above JPEG's 65,500-pixel dimension limit.
  Deterministic longitudinal wrapping is required to retain all requested
  pixels in one JPG.
- Replay failure records already contain the absolute matched reference arc
  which caused each reset. This is the correct focus position; the subsequent
  segment seed is intentionally later.
- Independent review required JPEG-safe marker bands, explicit overlap
  semantics, strict pre-render failure validation, fraction-aligned independent
  top/side panel ranges, typed panel descriptors, and a direct compositor test;
  the plan was updated accordingly.

## Deviations

- None.

## Implementation

- The full replay overview now passes render scale `8` to the existing shared
  fine-to-coarse renderer. Its native surface coordinate grid remains the
  authoritative grid; there is no image resize and no change to per-failure
  OBJ/MTL/TIFF sampling.
- Greedy and fiberlet failures are projected from their strict stored absolute
  reference arcs and drawn as three-pixel, full-strip-height red and cyan
  bands. Intersections are magenta. The marker identifies the failing sample
  before reset, and no marker is added for the later reset seed.
- Long overview rasters are split into the minimum number of equal-reference-
  fraction panels needed to keep each source range at or below 32,000 columns.
  Top and side use independent exact half-open column ranges, and all panels
  remain in one vertically stacked JPEG. Typed panel descriptors are validated
  before publication and recorded in the root manifest.
- The compositor is exposed only to the focused test target through the
  existing `VC_TESTING` convention.

## Validation

- Built with all 32 requested jobs:
  `cmake --build volume-cartographer/build --target test_fiber_replay vc_fiberlets -j32`.
- `volume-cartographer/build/bin/test_fiber_replay`: 9 test cases passed. The
  suite verifies exact 8x dimensions, full-height red/cyan/magenta bands after
  JPEG encoding, strict failure validation, exact two-panel source coverage,
  boundary preservation, black panel gaps, and publication metadata.
- Ran a Paris4 replay for a 512-base-voxel interval with `--vis` and CT group
  `/2`. It completed with no failures and produced top `336x1056`, side
  `336x1048`, combined `1056x766`; the former overview sizes were approximately
  top `42x132` and side `42x131`.
- Inspected `/tmp/vc_fiber_replay_8x.gdiT1s/fiber_replay.jpg`; CT detail and all
  three polylines render correctly. This bounded replay had no restart events,
  so it correctly contains no restart bands.
