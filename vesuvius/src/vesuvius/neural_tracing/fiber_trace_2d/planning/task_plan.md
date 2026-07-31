# Plan: Persist Per-Segment Manifest Identities

## Identity Flow

1. Keep runtime opening locations separate from persisted source identities.
2. For ordinary project entries, use the configured local or remote manifest
   location.
3. For catalogue-backed Lasagna entries, reconstruct the exact public manifest
   URL from the existing artifact URL tag and discovered root manifest name.
   The catalogue has no artifact UUID; sample/volume/level/model/index remain
   auxiliary catalogue metadata.

## Segment Metadata

1. Continue using the existing v3 `normal_manifest` field as the Lasagna
   manifest identity and `fiber_manifest` as the fiber-inference identity.
2. Direct Lasagna records set the Lasagna identity and clear stale fiber
   identity.
3. Successful trace and trace-to-Lasagna fallback records retain both
   identities because both datasets were consulted.
4. Direct/short cubic-spline records clear both identities. A spline reached
   after failed Lasagna/trace attempts retains the identities used by those
   attempts together with the fallback diagnostics.

## Tests

1. Extend open-data resolution tests for the public manifest URL.
2. Extend coordinator/schema tests for mode-dependent identity persistence.
3. Run focused C++ tests and rebuild `VC3D` with `-j32`.

## Spec Update

- Specify source-identity semantics and catalogue URL behavior.

## Docs Updates

- Document the two segment fields and the available catalogue identifiers.

## Changelog

- Record manifest identity persistence.
