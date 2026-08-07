# Task log: provenance-driven Atlas model registration

## Findings

- Atlas currently has one Lasagna model,
  `20260419180421-lasagna`, with task `lasagna`, architecture `unet`, and no
  checkpoint hash or run property.
- Existing private Lasagna origins use `s3://philodemos/paul/lasagna/...`; the
  requested equivalent manager root is `s3://philodemos/hendrik/lasagna`.
- Both new completed runs carry checkpoint SHA-256
  `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17` and the
  correct actual run `s1a_128_1_single_8x8_20260801_084232` in both portable and
  private provenance. Their unresolved field is Atlas model identity.
- Atlas `ModelProperties` currently has architecture/task/snapshot/output
  fields but no run. Proposed inference models currently put run and SHA-256
  only in `creation.metadata`.
- Snapshot indexing already records run, hash, architecture, task,
  `model_creation_utc`, and optional `atlas_model_id`, providing the required
  join surface.

## Deviations

- No implementation or live metadata/config mutation was performed; the user
  requested a plan only.

## Validation

- Independent plan and implementation reviews completed and incorporated.
- Manager/provenance focused suite: 59 passed.
- Direct Lasagna provenance regression: 1 passed.
- Atlas inference-bundle suite: 4 passed.
- Both live bundles passed read-only `las_manager open-data validate` with the
  approved proposed model; no upload was performed.
- Python compilation and both repositories' `git diff --check` passed. Atlas
  browser type checking was not run because its `node_modules` are absent and
  dependency installation was not authorized.
- The tiny end-to-end Fiber CPU inference test was stopped after several
  minutes without output. Commit capture is covered by the shared helper test
  and the inference writer assertion remains in that end-to-end test.

## Plan review changes

- Clarified that existing inferences already contain the run; Atlas
  `Model.properties` is the missing destination.
- Added authoritative checkpoint rehashing, normalized duplicate-copy handling,
  precise model compatibility, canonical-ID collision failure, Atlas/browser
  schema updates, Atlas checkout configuration, and journaled migration
  rollback/recovery.
- Removed the unrequested manual model-ID override path.
- User approved a revised minimal Atlas layout: numeric model references,
  `fiber3d/unet`, `model_training`, snapshot-relative path and SHA-256 only.
  Removed the planned Atlas `run` and other descriptive provenance fields.
  The Villa commit is now inference metadata only and will be backfilled for
  the two existing runs.
- Independent review required commit capture to live in the shared inference
  metadata writer so direct runs are covered. The canonical field is
  `inference.code_commit`; historical migration is pinned to known-working
  commit `70a63e29fbd2bec5a53aa86337511e887b250775`, never migration-time `HEAD`.
- Removed the initially added persistent `open-data migrate` command after user
  clarification. The two work-in-progress bundles will be updated by a one-off
  validated operation; no migration product surface is shipped.
- Implementation review found and drove fixes for nested `fiber3d/unet` model
  reload, carried-ID rehash bypass, run-dropping snapshot roots, UTC offset
  normalization, non-Git inference behavior, the ignored registration flag,
  and first-use snapshot resolution without preflight cache mutation.

## Live updates

- Configured `/home/hendrik/vesuvius-atlas` and
  `s3://philodemos/hendrik/lasagna` in the user's manager config.
- Backfilled both completed bundles with model ID `20260801084232`, snapshot
  `s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`, checkpoint
  SHA-256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`,
  architecture `fiber3d/unet`, and inference commit
  `70a63e29fbd2bec5a53aa86337511e887b250775`.
