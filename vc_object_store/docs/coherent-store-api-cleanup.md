# Coherent store API cleanup

## Contract

Current and exact-version reads, graph queries, edits, geometry, extension data and artifacts must use shared resolution and validation. Physical placement (local, partitioned, referenced, packed) belongs in storage routing, not consumer-specific branches. History policy changes retention, not identity. Preserve binary identifiers, slim history-off storage, checked writes, tombstone precedence and immutable dependencies; do not introduce proxy records or per-object routing directories.

## Tasks

- [x] A. Packing: carry artifact content when repacking a sealed dependency closure; preserve references and atomic failure cleanup.
- [x] B. Inherited edge edits: replace manual edge recreation with shared overrides; continue versions and retain original metadata/extension descriptors.
- [x] C. Exact-version graph snapshots: retain full version identities rather than requiring archive UUIDs; support current-only states.
- [x] D. Version-reference reconstruction: preserve exact source identity even when a newer overlay shadows the selected current-only base version.
- [x] E. Relative source paths: resolve against the selected record's owning database, including packed locators, not the caller's database.
- [x] F. Collection-wide queries: one visible-state selection/graph/export path across overlays and shards, including filtering, pagination and deletion precedence.
- [x] G. Inherited relationship geometry edits: use the ordinary override/dependency/decision machinery instead of requiring removal and recreation.
- [x] Add regressions, run the approved PR suite, inspect the full diff and record results.

The earlier audit's final row (traversal and point-ID allocation) overlaps F. Shared endpoint lookup and retained point-ID allocation are already fixed; retain those tests. E (source-file paths) is a separate issue, not another name for traversal. B and G share relationship override infrastructure and should be implemented together.

## Baseline and evidence

Starting commit: `2fea874e0`. Six focused regressions pass, but additional probes reproduce: inherited edge version 5 becomes 1 after extension editing; graph reads reject a valid current-only version with an empty UUID; exact base-version reference reconstruction fails after an overlay edit; referenced relative source lookup fails from another directory; packed artifact lookup becomes unavailable. With a pinned scope, endpoint traversal finds two objects while the physical graph snapshot finds none.

Existing full-suite baseline: 51 native tests, 199 Python tests and 117 Spiral integration tests pass; optional SSH and real-data tests skip. Use `bash local/scripts/run_vcobj_pr_tests`; no full-volume benchmark is needed for this correctness cleanup.

## Progress

Ordinary `ObjectStore` transactions now attach the pinned reference scope; object/edge selection, graph construction, extension scans and JSON export share that scope. `ObjectCollection` delegates those operations rather than implementing separate selection/edit rules. One record-override helper preserves inherited versions and descriptors while leaving payloads referenced. Shared relationship binding resolution supports current-only and archived sources. Exact-version results carry their owning store identity. Packed database copies preserve and verify artifact sidecars, including repacking; relative file lookup uses the owning database.

Added `object_store_coherent_api` regression coverage for the seven findings, including all four source/overlay history-policy combinations. The follow-up audit extended shared routing to viewport geometry and surface proximity queries, corrected length cursor pagination across shards, and removed the spatial accelerator's local-only geometry-history foreign key. Existing import tests now exercise ordinary current and exact graph APIs for plain, packed and sharded stores. Collection-specific record/geometry read wrappers were removed in favor of the shared readers.

The container had lost the development packages required by the existing build; the user approved restoring them. No volume-cartographer build is required, and no full-volume workload is being run.

Storage contract remains unchanged: no new persistent tables, per-object directories, proxy records or geometry payload copies. An override in a spatial-enabled store preserves its optional spatial accelerator; history-off/spatial-off stores do not acquire one.

## Boundaries

Exact references resolve across stores; arbitrary collection-wide historical snapshots are not introduced. Each store retains its own publication timeline. Spatial queries still require spatial indexes rather than silently scanning unindexed geometry. Artifact content remains in the existing sidecar layout, which must accompany a moved packed file; packing does not copy unrelated external dataset files. Cross-store query performance and macOS execution are not benchmarked by this correctness suite.

## Verification

Final command: `bash local/scripts/run_vcobj_pr_tests` (Linux, RelWithDebInfo native build, Python 3.14). Results: 52 native tests passed and one optional SSH test skipped; 199 Python tests passed; 117 Spiral integration tests passed and one optional real-data test skipped. Native fixtures cover plain/packed/sharded stores and source/overlay history on/off combinations. `git diff --check` passed. The existing Spiral `PatchSamplingAtlas` nanobind shutdown leak diagnostic remains; it was present in the baseline and is outside this cleanup. No full-volume performance or macOS result is claimed.
