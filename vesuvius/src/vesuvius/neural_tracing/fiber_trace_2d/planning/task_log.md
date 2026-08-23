# Task Log: managed Fiberlet preprocessing and publication

## 2026-08-23

- Planning only; no implementation has been started.
- The active `fiber-lets3` branch contains the native whole-volume command but
  currently predates the tracked `lasagna/manager` sources. Current
  `origin/main` contains the manager implementation. The implementation must
  integrate main first rather than copying manager code or using untracked
  bytecode.
- Existing manager orchestration is reusable but still names its generic
  process state, completion validation, and upload APIs as inference. The plan
  extracts one shared managed-job path and keeps inference records compatible.
- The native final combined Zarr already contains compact anchors, prefixes,
  and routes. Its separate float anchor Zarr is a durable resume cache and does
  not need to be uploaded.
- The native dataset fingerprint currently includes absolute local manifest
  paths and persists those paths in metadata. That is acceptable for an
  experimental local cache but not for a portable/open-data artifact. The plan
  separates runtime locations from stable manifest-hash/source identity before
  publication support is enabled.
- Atlas has no existing Fiberlet data type. Existing `lasagna` represents dense
  neural prediction bundles and requires a trained model ID plus Lasagna
  manifest/OME-Zarr layout, so a combined sparse Fiberlet graph must not be
  forced through that contract. The plan adds one minimal derived volume data
  type and no fake model.
- Local Fiberlet processing may precede publication of its inputs. Atlas
  validation/upload will require the referenced Fiber and normal prediction
  identities to exist so the published dependency graph is resolvable.
- Existing untracked compiled Python extensions under `lasagna/` were observed
  and left untouched.
- Independent review identified missing resume semantics, native completion
  validation, bounded-memory sparse upload, passthrough ownership checks, and
  several places where existing manager/Atlas helpers must be extracted rather
  than copied. The plan now includes those corrections.
- The portable metadata contract was clarified: it is not merely a settings
  fingerprint checked against caller configuration. The final Zarr must store
  every effective setting needed by a reader to construct compatible decoders
  and interpret coordinates. Canonical global settings and stable source
  identities feed the fingerprints; runtime filesystem paths do not. Paths
  remain only in private manager execution records.
- Created `fiberlets-development-integration` for the broader unmerged
  Fiberlet work, merged current `origin/main`, and committed the integration as
  `33e029fd2`.
- Established a post-merge baseline: 61 manager tests passed and all 25 native
  Fiberlet storage tests passed.
- Replaced path-bearing native dataset metadata with schema-v2 structured
  source and processing contracts. Fingerprints are now canonically derived
  from scientific settings and stable source identities; the final dataset can
  be reopened from its own metadata, and whole-volume preprocessing performs a
  final fresh source-derived completeness check before reporting success.
- Added path-independence/source-sensitivity and self-describing-open coverage;
  the focused native suite now passes 26 test cases.
- Generalized the existing manager reservation, tmux launch, runner lifecycle,
  completion, reconciliation, listing, and upload paths rather than creating a
  second Fiberlet supervisor. Legacy inference records still default to the
  `inference` lifecycle phase.
- Added `fiberlet run`, `fiberlet resume`, and `fiberlet ls` with strict
  completed-run role/source/coordinate/crop validation, freshly hashed input
  manifests, a private resumable anchor cache, stable source context, bounded
  portable provenance, contextual completion, and manager-owned option guards.
- Added explicit `fiberlet_threads` configuration (portable default 32) and
  configured this host for 128 workers. Per-run `--threads` remains an explicit
  last-wins override.
- Fiberlet completion now validates the native schema-v2 header, array roots,
  processing contract, source run/hash identity, and fingerprint before a run
  can complete. A post-processing provenance failure returns exit status 1
  rather than leaving a failed record behind a successful runner exit.
- Generalized the existing marker-protected staging flow by artifact kind.
  Fiberlet Zarr payloads use rclone's streaming traversal below the distinct
  `fiberlets/<run_uuid>` namespace; inference retains its fixed inventory and
  `inference/<run_uuid>` behavior.
- Extracted shared Atlas portable-bundle/origin/idempotent-ingest helpers and
  added a minimal `fiberlets` volume entry. It requires already-registered
  Fiber and normal Lasagna predictions, creates no model, and plans only one
  canonical and one CC-licensed public copy with no derivation.
- Validation results: 75 manager/open-data/provenance tests passed; the native
  `vc_fiberlets`/storage targets built and all 26 focused storage tests passed;
  the new Atlas Fiberlet test plus all 5 existing inference-bundle tests passed.
- The wider Atlas `test_data_sync.py` run had 27 passes, 7 skips, and 4 existing
  expectation failures in unrelated flattened-OBJ and TIFXYZ routing tests.
  The new `entity.data` parameter source is used only by the Fiberlet rule, so
  those failures are not caused by the Fiberlet dispatch.
- No production whole-volume preprocessing/upload was run because it is large
  and operator-owned. The native, manager, upload, Atlas ingest, idempotency,
  and exact two-copy planning paths are covered with focused synthetic tests.
- Existing ignored compiled Python extensions in `lasagna/` and untracked Atlas
  sync-state files were left untouched.
