# Reproducibility

## Purpose

`frame_preflight.py` checks explicit catalog provenance between a selected TIFXYZ entry and a requested target volume before rendering. It produces a frame decision, a separate alternative-entry decision, machine-readable reason codes, supporting evidence, contradictions, and transform direction when applicable.

## Supported inputs

The CLI accepts:

- a local catalog JSON file; or
- an explicit HTTP(S) catalog URL.

The validated evidence run used a local snapshot so that all six case executions consumed identical bytes. No CT data, TIFXYZ coordinate rasters, or Zarr chunks are required.

## General prerequisites

- A Villa checkout containing commit `917de185c102c045af1b6523ea11e68736a905e2`.
- The `vesuvius` subproject environment.
- Python compatible with the project declaration (`>=3.14,<3.15`).

The concrete reproduced environment used Python 3.14.8 and pytest 9.1.1 on Windows. That environment is an evidence record, not a claim that other supported systems cannot run the CLI.

## Catalog snapshot

- Public source: <https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json>
- Evidence snapshot filename: `official_metadata_2026-10-09.json`
- Size: 17,537,378 bytes
- SHA-256: `40517076dc5f2128599868b337e25a17922ebe70a22b238b990e109b6f82af0f`

The catalog snapshot is not bundled in this staging package. Reproduction of the recorded outputs requires a local file with exactly this size and hash.

## Working directory

Run the following commands from the `vesuvius/` subproject root. The commands below are portable forms of the recorded C02-EVIDENCE-001B invocations; only machine-specific absolute paths were replaced by relative paths.

## P1 — documented directed transform and same-frame alternative

```text
python src/vesuvius/frame_preflight.py --catalog ./official_metadata_2026-10-09.json --sample-id PHerc0139 --segment-id 20250108000004-w029_2025010827 --entry-origin-path PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20250728140407-9.362um.tifxyz/ --target-volume-id 20260102150214 --output ./p1_output.json
```

Expected process exit code: `2`.

Expected decision fields:

- `frame_relation`: `TRANSFORM_REQUIRED`
- `frame_reason`: `EXPLICIT_DIFFERENT_TARGET_AND_DIRECTED_VOLUME_TRANSFORM`
- `transform.application_direction`: `FORWARD`
- `transform.inverse_required`: `false`
- `alternative_relation`: `SAME_FRAME_ALTERNATIVE_AVAILABLE`

## N1 — explicit same-frame negative control

```text
python src/vesuvius/frame_preflight.py --catalog ./official_metadata_2026-10-09.json --sample-id PHerc1447 --segment-id 20250703034159-auto_grown_20250703034159599 --entry-origin-path PHerc1447/segments/20250703034159-auto_grown_20250703034159599/mesh/20250703034159-on-20250521151220-8.64um.tifxyz/ --target-volume-id 20250521151220 --output ./n1_output.json
```

Expected process exit code: `0`.

Expected decision fields:

- `frame_relation`: `SAME_FRAME`
- `frame_reason`: `EXPLICIT_ENTRY_TARGET_MATCHES_REQUESTED_VOLUME`
- `alternative_relation`: `NONE_CONFIRMED`

## U1 — insufficient selected-entry frame evidence

```text
python src/vesuvius/frame_preflight.py --catalog ./official_metadata_2026-10-09.json --sample-id PHerc1447 --segment-id 20250502183138-auto_grown_20250502162038685 --entry-origin-path PHerc1447/segments/20250502183138-auto_grown_20250502162038685/mesh/intermediate/tifxyz_original/ --target-volume-id 20250521151220 --output ./u1_output.json
```

Expected process exit code: `2`.

Expected decision fields:

- `frame_relation`: `UNKNOWN`
- `frame_reason`: `MISSING_EFFECTIVE_ENTRY_FRAME_ID`
- `alternative_relation`: `SAME_FRAME_ALTERNATIVE_AVAILABLE`

## Status interpretation

Exit code `0` is produced by the reproduced N1 same-frame case. Exit code `2` is the expected diagnostic outcome for the reproduced P1 and U1 cases; it does not indicate that the CLI failed to produce a result.

The frame and alternative axes are independent. For example, U1 remains `UNKNOWN` for the selected entry while an explicitly target-bound alternative is available.

`NONE_CONFIRMED` means that complete enumeration of the relevant segment entries found no strong matching alternative in the evaluated catalog snapshot. It does not predict future catalog contents.

## Transform direction

For P1, the selected entry is explicitly associated with volume `20250728140407`, and the requested target is `20260102150214`. The catalog edge runs from the former to the latter. Therefore the stored matrix is reported as `FORWARD`, with `inverse_required=false`. The preflight reports this relationship but does not transform geometry.

## Limits of the decision

The result is based only on recognized explicit catalog fields. It does not infer frame relationships from filenames, resolution, timestamps, segment names, or spatial fit. Missing, conflicting, or ambiguous strong evidence produces `UNKNOWN`.

The decision does not validate geometry, bounds, CT support, TIFXYZ contents, Zarr chunks, render quality, or mesh quality. The three recorded cases are targeted evidence cases, not a prevalence estimate.
