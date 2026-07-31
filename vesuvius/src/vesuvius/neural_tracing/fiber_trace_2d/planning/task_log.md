# Persist Per-Segment Manifest Identities Task Log

## Findings

- Segment metadata already serializes `normal_manifest` and `fiber_manifest`,
  but direct Lasagna/cubic paths did not consistently populate or clear them.
- Catalogue Lasagna is opened from a local cache path. Its project tags retain
  public artifact URL, sample ID, volume ID, coordinate level, and optional
  model ID. The catalogue provides no standalone artifact UUID.
- Catalogue preparation guarantees exactly one root `.lasagna.json`, allowing
  the public manifest URL to be reconstructed deterministically.

## Implementation

- Added a source manifest location to resolved catalogue Lasagna results. The
  shared resolver reconstructs the public root `.lasagna.json` URL from the
  artifact URL tag and cached manifest filename; ordinary entries return their
  configured project location.
- Line-annotation sessions now keep Lasagna/fiber source identities separate
  from runtime cache/opening locations and pass the source identities to the
  segment coordinator.
- Direct Lasagna interpolation writes `normal_manifest` and clears stale
  `fiber_manifest`. Direct/short cubic spline clears both. Trace results and
  trace fallbacks keep both; later Lasagna-to-spline fallback retains the
  identities of the attempted datasets.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_line_annotation_generated_views test_open_data_manifest -j32`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`
  passed 56 cases.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_open_data_manifest`
  passed 32 cases. Expected fixture warnings covered unavailable test URLs and
  deliberately invalid TIFF placeholders.
- `cmake --build volume-cartographer/build --target VC3D -j32` completed.

## Deviations

- None.
