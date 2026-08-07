# Plan: provenance-driven Atlas model registration

## Approved Atlas identity contract

1. Keep the existing Atlas model shape minimal: `id`, `long_id`, `suffix`,
   `creation`, `properties`, and `data`. Keep the Atlas-wide
   `creation.process = "model_training"` convention and numeric model references.
2. Register Fiber output as Lasagna data with
   `properties.architecture = "fiber3d/unet"` and
   `properties.task = "lasagna"`.
3. Store the checkpoint path in `properties.snapshot`, relative to a configured
   snapshot root and including the run and `snapshots/` directory, for example
   `s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`. Add only
   `properties.snapshot_sha256` beyond the current model conventions.
4. Do not add a separate Atlas `run`, repository origin, inference commit,
   training step, precision, patch size, metrics, backend, or output-schema
   field. The same numeric model ID can later identify a Hugging Face model.

## Resolution and automatic registration

5. Add one resolver shared by manager validation/upload. Use carried
   `atlas_model_id` when present; otherwise match portable `model.sha256`
   against configured snapshot directories. Rehash candidate checkpoint bytes
   at decision time. Treat byte-identical copies with identical normalized
   metadata as one identity and reject conflicting metadata for a hash.
6. Resolve new Atlas identity from trusted snapshot metadata. Prefer embedded
   `atlas_model_id` and model creation UTC. For legacy snapshots, accept exactly
   one UTC `YYYYMMDD_HHMMSS` token in the actual run name. Query Atlas by the
   canonical 14-digit ID and reject collisions with another checkpoint hash;
   never invent an alternate timestamp.
7. Remove the normal upload-time `--model-id` and `--register-model`
   requirement. If the resolved model is absent, preflight prints the proposed
   minimal record and upload registers it automatically. If present, compare
   canonical/long ID, suffix where schema-required, task, architecture,
   snapshot path, snapshot SHA-256, and creation date/process.
8. Keep resolution and preflight read-only until upload ingestion. Stage only
   after successful preflight and keep registration/upload idempotent. Never
   permit two Atlas identities for one hash or one ID for different hashes.

## Inference provenance

9. At manager launch, resolve and carry the exact snapshot-relative path, hash,
   creation time, and Atlas identity through private and portable inference
   metadata. In the shared Fiber/Lasagna inference metadata writer, resolve the
   checked-out Villa Git commit used by the running code and store it as
   `inference.code_commit` in portable `inference.json`. This must also work for
   direct inference invocation; the manager is not the source of truth.
   Preserve the existing dirty-worktree indicator, accept a packaged build
   commit override, and use null rather than failing outside Git. Do not put the
   commit in the Atlas model.
10. Direct inference retains checkpoint-embedded identity and fails clearly at
    Atlas resolution when configured snapshot directories are required but no
    unique matching snapshot is available.
11. Update Fiber and Lasagna training/checkpoint guidance, and checkpoint
    writers where in scope, to store authoritative model creation UTC and
    optional registered Atlas ID, avoiding legacy timestamp recovery.

## Existing-run migration

12. Perform a one-off, non-shipped migration for the two work-in-progress runs.
    Rehash the matched snapshot, prepare updated private `metadata.json` and
    portable `artifacts/inference.json`, preserve originals during the update,
    replace each file atomically, cross-validate the pair, and roll back on
    failure. Do not add a persistent manager migration command or module.
13. Validate and then apply it to:
    - `PHerc0332-20251211183505-las-sd1-84afaf75`
    - `PHerc1299-20260309130042-las-sd1-aac02eb8`
    Verify and preserve their existing run value
    `s1a_128_1_single_8x8_20260801_084232`. Backfill resolved Atlas identity,
    creation fields, snapshot-relative path, and the known working Villa commit
    `70a63e29fbd2bec5a53aa86337511e887b250775` at
    `inference.code_commit`. Do not substitute migration-time `HEAD`. Refuse
    snapshot hash/run mismatches.

## Configuration

14. Set the existing manager config fields to
    `atlas_dir = "/home/hendrik/vesuvius-atlas"` and
    `upload_staging_s3 = "s3://philodemos/hendrik/lasagna"`, preserving all
    other keys. The manager appends `inference/<run_uuid>/`. Leave historical
    `paul/...` origins unchanged and never stage directly to a public bucket.

## Tests and validation

15. Add Atlas schema and round-trip tests for the minimal model record,
    `fiber3d/unet`, snapshot-relative paths and hashes, numeric model references,
    duplicate checkpoint copies, fresh hashing, embedded and legacy identities,
    missing/ambiguous metadata, ID collision, compatibility mismatch,
    automatic registration, and idempotency.
16. Add shared inference-writer and manager tests for canonical
    `inference.code_commit` capture under both managed and direct Fiber/Lasagna
    invocation, direct-inference fallback, bundle validity, and unchanged
    non-target fields for Fiber and Lasagna artifacts.
17. Validate the one-off transformation on copies of both runs, compare JSON,
    then update the real metadata with backups and cross-file validation. Do not
    upload unless separately requested.
18. Run focused manager and Atlas inference-bundle suites, compilation, browser
    typechecking/tests, full Atlas schema validation, and `git diff --check`.

## Spec update

Specify the minimal Atlas identity record, numeric model references,
`fiber3d/unet`, `model_training`, snapshot-relative path and SHA-256,
inference-only Git commit, automatic idempotent registration,
collision/mismatch rejection, one-off existing-run backfill, and Hendrik staging
root.

## Docs updates

Document automatic model resolution/registration, snapshot-directory lookup,
the minimal Atlas model fields, inference-only Git provenance, legacy failure
modes, and private staging layout.

## Changelog

Add Lasagna/Vesuvius and Atlas changelog entries for provenance-driven model
registration and completed-run migration.
