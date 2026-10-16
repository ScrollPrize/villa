# Metadata-only frame provenance preflight — evidence staging

> Status: sanitized local staging package. Nothing in this directory has been published by this preparation phase.

## Practical workflow problem

Before a TIFXYZ render, the selected catalog entry and the requested CT target volume may refer to the same coordinate frame, to different frames connected by an explicit transform, or to a relationship that the available metadata cannot establish safely. Treating these cases as equivalent can lead a user to proceed as though no frame conversion were needed.

The C02 contribution adds a conservative, machine-readable preflight based only on explicit catalog metadata. It reports one of three independent frame decisions:

- `SAME_FRAME`: an explicit frame match is established.
- `TRANSFORM_REQUIRED`: the selected entry and requested volume differ, and an explicit directed transform is available.
- `UNKNOWN`: the metadata is missing, conflicting, or otherwise insufficient for a safe decision.

Alternative discovery is reported separately as `SAME_FRAME_ALTERNATIVE_AVAILABLE`, `NONE_CONFIRMED`, or `UNKNOWN`. Alternatives are enumerated without ranking.

## Concrete P1 example

For P1, the selected TIFXYZ entry is explicitly associated with volume `20250728140407`, while the requested target is `20260102150214`. The catalog contains a directed transform from the selected-entry frame to the requested frame, so the result is `TRANSFORM_REQUIRED` with application direction `FORWARD`. The same segment also has an explicitly target-bound alternative, reported as `SAME_FRAME_ALTERNATIVE_AVAILABLE`.

This diagnosis can warn a user before rendering that the selected entry must not be treated as already sharing the requested target frame. It does not apply the transform, select the alternative, or claim that either mesh is better.

## Evidence package

- Commit: `917de185c102c045af1b6523ea11e68736a905e2`
- Pull request: <https://github.com/ScrollPrize/villa/pull/1944>
- Python: 3.14.8
- Tests: 20/20 frame-preflight and 30/30 surface-preflight tests passed
- Real-data catalog cases: P1, N1, and U1, each reproduced twice with byte-identical output
- Catalog SHA-256: `40517076dc5f2128599868b337e25a17922ebe70a22b238b990e109b6f82af0f`

See `REPRODUCIBILITY.md` for exact commands, `EVIDENCE_SUMMARY.md` for the validation record, and `provenance/public_manifest.json` for machine-readable provenance.

## Scope and non-goals

The preflight does not read CT voxels, TIFXYZ coordinate rasters, or Zarr chunks. It does not perform bounds checks, assess mesh quality, rank alternatives, apply transforms, or integrate automatically with the renderer. `SAME_FRAME` is not a quality judgment, and `TRANSFORM_REQUIRED` is not a defect judgment.
## Integrity verification

Run checksum verification from the root of this evidence package—the directory containing this README:

```text
sha256sum --check provenance/SHA256SUMS.txt
```

The checksum paths are relative to the public package root. `SHA256SUMS.txt` intentionally excludes itself to avoid circular hashing.

