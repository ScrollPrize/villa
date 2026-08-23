# Plan: catalogue-backed Fiberlet normal inputs

## Catalogue and resolution

1. Add a typed Lasagna-prediction record/index in `lasagna.manager.catalog`.
   Classify an entry from its exact Atlas model's declared `output_channels`,
   not merely from `type = lasagna`: regular normals require `grad_mag`, `nx`,
   and `ny` and must not advertise `presence`; Fiber predictions advertise
   `presence`.
2. Preserve the data entry, model ID, requested level, origin, catalogue hash,
   and source sample/volume identity. Select public S3 origins with the same
   origin rules used for CT volumes.
3. Resolve the published bundle's one root `.lasagna.json` manifest by reusing
   a public paginated S3-prefix helper extracted from the downloader. Convert
   the public S3 origin with VC3D's canonical URL rules, cache that small
   manifest under the manager cache, and reject zero or multiple manifests.

## Managed Fiberlet interface

1. Change the CLI to
   `fiberlet run <fiber-inference> [normal-input]`. With no normal input,
   refresh/use the catalogue through the existing one-hour policy and select
   the unique regular-normal representation matching sample and volume.
2. Keep an explicit completed local Lasagna run accepted as `normal-input`.
   Also accept a published normal selector when multiple catalogue candidates
   require an explicit choice.
3. Use an explicit published selector grammar
   `atlas:<model-id>@L<source-level>` when an override is needed; local run
   names retain precedence unless the `atlas:` prefix is present.
4. Materialize only the remote manifest through the existing native Lasagna
   persistent lazy read-through-cache contract. Use VC3D's exact
   `open_data/lasagna/<sample>/<volume>/<identity>` path and marker fields so
   prediction chunks are fetched on demand and reused across manager and VC3D
   jobs; do not build a second addressing or chunk-cache format. The VC identity
   uses the canonical remote URL plus model/coordinate metadata, not manifest
   SHA. The persistent disk cache is not described as size-bounded.
5. Apply the existing whole-volume and coordinate compatibility checks to both
   local and remote inputs before launch: same sample/volume, base shapes equal
   within one voxel per axis, numeric `source_to_base`, uncropped coverage, and
   valid channel geometry. Normal and Fiber source levels may differ (published
   PHerc0332/PHerc1299 pairs are Fiber L1 and normals L2).
6. Record a discriminated portable remote dependency identity (kind, synthetic
   stable source ID, catalogue/model/level/origin/exact manifest URL and hash),
   not local cache paths. `--remote-cache-dir` is manager-owned.
7. Keep resume stable by persisting the resolved remote manifest URL and
   re-resolving/revalidating the same dependency without silently changing to a
   newer catalogue candidate.

## Completion and docs

1. Complete the optional second argument from both compatible local regular
   Lasagna runs and published regular-normal selectors. The default one-argument
   workflow needs no second completion.
2. Completion reads only the cached catalogue; it never performs network I/O.
3. Update manager docs/examples to show the one-argument command, on-demand
   normal caching, explicit override syntax, and clear failure behavior.

## Tests

1. Add catalogue fixtures covering regular Lasagna versus Fiber Lasagna,
   exact model lookup, missing/null fields, multiple origins, and selector
   resolution.
2. Add manager tests for unique auto-selection, explicit local override,
   explicit remote selection, no match, ambiguity, mismatched level/frame,
   remote-cache CLI wiring, portable provenance, resume, and completion.
3. Run focused Lasagna manager/catalog/completion tests and the existing native
   Fiberlet CLI/storage smoke tests. No scientific/numerical behavior changes.
4. Cover the VC3D-compatible URL, identity hash, directory, marker, cache-hit,
   and exact-hash resume behavior with a fixed golden cache-path test.

## Spec update

Add the manager contract that whole-volume Fiberlet preprocessing consumes one
local completed Fiber prediction plus a compatible regular Lasagna normal
prediction, resolved by default from the public catalogue and read through the
native persistent lazy remote cache. Document role classification, base-frame
compatibility, and ambiguity rules.

## Docs update

Update the Lasagna manager/Fiberlet workflow docs and CLI help. Explain that
published prediction payloads are fetched lazily, independently from CT-volume
prefetching.

## Changelog

Record catalogue-backed automatic Lasagna-normal resolution for managed
Fiberlet jobs.
