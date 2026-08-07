# Plan: minimize manager staging and Atlas inference ingestion

## Scope and invariants

1. Preserve the existing Atlas copy-first `lasagna` flow. The internal Atlas
   entry stores a private S3 origin; data-sync resolves it, copies to the
   canonical public Lasagna path, adds the public origin, and public metadata
   export retains only `public-read` origins.
2. Preserve the existing origin representation. `DataOrigin.path` is relative
   to its `AccessRoot.url`; Atlas joins them as
   `access_root.url.rstrip('/') + '/' + origin.path.lstrip('/')`. For the
   current run this resolves
   `s3://philodemos` plus
   `hendrik/lasagna/inference/<run_uuid>/` to the complete private S3 URI.
3. Keep the approved UUID staging layout, `_INCOMPLETE`, portable
   `inference.json`, automatic model registration, `fiber3d/unet` model
   architecture, snapshot path/SHA-256, numeric model references, and the
   Lasagna 3--4 product validation requiring `nx` and `ny`.
4. Remove only two unneeded extensions:
   - remote `upload-manifest.json` and its manager bookkeeping;
   - duplicated portable provenance in Atlas `DataEntry.creation_info`.

## Marker-only rclone staging

5. Keep the fixed local bundle inventory used for `rclone --files-from-raw`,
   but reduce it to the information needed to name files. Do not recursively
   hash every Zarr chunk for upload transaction metadata.
6. Simplify manager staging to:
   - put `<prefix>/_INCOMPLETE` before starting transfer;
   - invoke the configured `rclone copy` over the fixed inventory;
   - retain `_INCOMPLETE` on every failure or interruption;
   - delete `_INCOMPLETE` only after rclone exits successfully.
   `_INCOMPLETE` is a manager-side transaction/commit guard. Atlas data-sync
   does not inspect it, so failed staging must never call the Atlas ingester.
7. Let rclone provide retry/resume/idempotent transfer behavior. A repeated
   manager upload runs rclone again; already transferred files are skipped by
   the configured comparison flags. Do not add a second remote completion
   record.
8. Remove `UPLOAD_MANIFEST`, manifest put/get/verification, bundle-digest
   calculation, per-file upload SHA-256, same-UUID manifest collision checks,
   and manifest-specific reserved-name handling. Keep `_INCOMPLETE` reserved in
   local bundles so artifact content cannot collide with the manager marker.
   Retain checkpoint SHA-256 and
   portable inference validation because those identify the model and bundle,
   not the upload transaction.
9. Remove `bundle_digest`, `manifest`, and the ambiguous `uploaded` boolean from
   newly persisted manager upload metadata and from human/JSON validation
   output. Keep staging URL and upload lifecycle; every upload command invokes
   a resumable rclone transfer attempt.
10. Document the deliberate marker-only limitation: `rclone copy --size-only`
    neither detects changed same-size objects nor deletes stale destination
    objects. This is safe only under the existing immutable run UUID/bundle
    contract. Validate that completed bundle identity/content is immutable and
    never reuse a run UUID for changed artifacts.

## Lean Atlas ingestion

11. Parse and validate `inference.json` as before, but do not construct or pass
    Atlas `creation_info` from schema version, run UUID, timestamps, catalog
    hash, code commit, inference settings, or artifact inventory.
12. Ingest exactly the existing `lasagna` data-entry shape:

    ```json
    {
      "type": "lasagna",
      "origins": [{
        "path": "hendrik/lasagna/inference/<run_uuid>/",
        "access_roots": [{
          "type": "s3",
          "url": "s3://philodemos",
          "usage": "private-s3"
        }]
      }],
      "parameters": {
        "model_id": "<numeric-model-id>",
        "level": "<input-OME-group>"
      }
    }
    ```

13. Keep detailed provenance solely in staged `inference.json`. Keep the
    separate approved minimal Atlas model record unchanged.
14. Preserve strict idempotency for Atlas entries: an existing entry with the
    same type and parameters must match the lean origin entry; conflicting
    origins remain an error. Do not silently preserve obsolete
    `creation_info` during re-ingestion.

## Existing-state cleanup

15. Discover every manager run whose local record references
    `upload-manifest.json` and every configured staging prefix that contains
    that sidecar. Report the exact set before mutation.
16. Also inspect/report whether a transaction manifest was already copied to a
    canonical or public destination. Do not mutate published data as part of
    this cleanup without separate user direction.
17. For private staged bundles that are otherwise complete, delete only the remote
    `upload-manifest.json`; do not touch `inference.json`, Lasagna manifests,
    OME-Zarr objects, or public destinations. `_INCOMPLETE` must remain absent
    for completed uploads.
18. Remove obsolete local `upload.manifest`, `upload.bundle_digest`, and
    `upload.uploaded` fields
    from affected run metadata atomically, preserving lifecycle and staging
    URL.
19. Remove only the newly copied `creation_info` from already ingested Fiber
    Lasagna entries. Preserve type, origins, parameters, all legacy Lasagna
    entries, models, and copy/publication information. Perform this as a
    reported one-off cleanup rather than shipping a permanent migration CLI.

## Tests and validation

20. Manager unit tests must cover successful marker creation/removal, marker
    retention on rclone failure, safe rerun through rclone, absence of manifest
    reads/writes and digest fields, exact fixed file-list transfer, reserved
    local `_INCOMPLETE`, failed staging never invoking Atlas ingestion, the
    immutable run UUID/bundle invariant, and both Fiber and direct Lasagna
    bundles.
21. Atlas tests must assert that parsed portable provenance still validates,
    model registration remains unchanged, and the ingested data entry has no
    `creation_info`. Test full S3 source reconstruction from access root plus
    relative origin and verify public export excludes the private origin.
22. Validate cleanup on copies first. Then run focused Villa manager tests,
    Atlas inference-bundle/model/config/export tests, `git diff --check`, and
    compile/import smoke tests. No real inference or public data-sync is
    required.
23. Keep commits reviewable: one Villa commit for manager protocol changes and
    one Atlas commit for lean ingestion/tests. Report remote and local one-off
    cleanup separately from source commits.

## Spec update

Replace the Atlas staging specification that currently defines immutable
identity as run UUID plus a complete path/size/SHA-256 upload manifest. Specify
marker-only rclone staging: `_INCOMPLETE` guards transfer, rclone owns resume,
and successful marker removal commits the staging prefix. State explicitly
that `inference.json` remains portable provenance and is not copied into Atlas
`creation_info`. Specify the lean existing Lasagna data entry and relative
origin/access-root resolution.

## Docs updates

Update `lasagna/docs/manager.md` and the Lasagna README where applicable:
remove upload-manifest, digest, collision, and per-file hashing claims; document
marker-only rclone retry behavior; explain relative Atlas origin paths; and
state that detailed provenance remains in `inference.json`, not Atlas data
entries.

## Changelog

Add one concise changelog entry recording the removal of the manager upload
manifest and the restoration of lean existing-schema Atlas Lasagna entries.
Record exact cleanup commands/results in `task_log.md`, not in the durable
specification.
