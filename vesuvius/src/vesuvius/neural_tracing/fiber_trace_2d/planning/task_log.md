# Task log: minimize manager staging and Atlas inference ingestion

## Planning findings

- `upload-manifest.json` and `_INCOMPLETE` were introduced by Villa manager
  commit `3379ea7a7`; they are not pre-existing Atlas formats.
- The upload manifest is unnecessary because rclone already resumes/skips
  transferred objects. `_INCOMPLETE` remains useful as the sole publication
  guard.
- Atlas data-sync reconstructs a source URI by joining
  `AccessRoot.url.rstrip('/')` and `DataOrigin.path.lstrip('/')`; the relative
  origin is therefore existing Atlas behavior, not a new unresolved path.
- Public Atlas export filters origins to the target usage. The internal
  `private-s3` staging origin is excluded from public metadata after data-sync
  adds a `public-read` origin.
- Portable inference provenance was unnecessarily duplicated into the new
  Atlas entry's `creation_info`. The approved cleanup leaves provenance in
  `inference.json` and reuses the lean existing `lasagna` entry shape.
- Independent review confirmed relative-origin resolution and identified that
  `_INCOMPLETE` is manager-enforced rather than Atlas-enforced. The plan now
  requires failed staging never to invoke ingestion and keeps the marker name
  reserved in local bundles.
- Marker-only `rclone copy --size-only` intentionally gives up manifest-based
  same-UUID content collision detection and does not delete stale objects. The
  plan makes the immutable run UUID/bundle contract explicit and removes the
  misleading `uploaded` boolean.
- Cleanup must report, but not silently delete, any transaction manifest that
  has already reached a canonical/public destination.

## Deviations

- The user elected to delete existing remote `upload-manifest.json` objects;
  implementation performed no S3 mutation.
- Public/canonical destinations were not mutated. The user confirmed the
  observed manifest was in private staging and owns its removal.

## Validation

- Villa: `62 passed` across manager, open-data staging, and portable provenance
  tests; manager/provenance compile smoke passed with bytecode cache in `/tmp`.
- Atlas: `5 passed` across inference-bundle and AtlasConfig tests; package
  compile smoke passed with bytecode cache in `/tmp`.
- Both completed local runs validate with the new manager path: PHerc0332 has
  58,138 files and PHerc1299 has 83,679 files; both resolve existing model
  `20260801084232-lasagna` with input level 1.
- One-off local cleanup atomically removed upload manifest/digest/boolean fields
  from both manager records. Their staging URL and completed lifecycle remain.
- The two Atlas volume entries contain only type, private origin, numeric model
  ID, and level; neither contains `creation_info`.
- Independent implementation review found one stale cross-check constructing
  the abandoned `fiber3d-prediction`/`creation_info` entry. It was removed as
  duplicate of the authoritative Atlas adapter tests, and the failed-transfer
  test was strengthened to exercise the bulk rclone path.
- Source commits: Villa `ef09251e7` (marker-only staging) and Atlas `75a407e`
  (lean entries plus the already registered model and two inference origins).
