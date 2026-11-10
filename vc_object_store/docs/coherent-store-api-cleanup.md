# Coherent store API cleanup

## Current review fixes

Baseline: `3812cb17b`. C++ store plus VC3D adapters; existing CMake/approved PR runner, Linux and macOS portability, unchanged schema 21 and binary identifiers. These fixes must remove duplicated resolution, not introduce more placement-specific branches. Each fix is committed separately with regressions; the final run uses `bash local/scripts/run_vcobj_pr_tests`.

- [x] S. Reject inherited live edge UUID insertion; continue inherited deleted-edge versions through shared tombstone resolution. `test_coherent_store` passes for history on/off insertion rejection and inherited resurrection.
- [x] T. Centralize canonical opposite-endpoint lookup for traversal and attached documents. Coherent-store and transform tests pass, including uppercase attachment removal without retiring the owner; VC3D and source metadata use the same helper.
- [x] U. Preserve ownership on no-op edge update/repair; repair delegates to the ordinary update path instead of duplicating SQL/validation/stamping. Coherent-store and transform regressions pass, including no-op source identity/commit sequence and real endpoint edits.
- [x] V. Route geometry history to the owning record instead of the attachment database. Coherent-store regressions verify both source history policies through both overlay policies.
- [x] W. Carry selected metadata through historical geometry transforms and transform listings. Shared `objectAtGeometry` resolves retained metadata across referenced stores, and local/hosted readers use it. Exact transforms retain selected metadata and only retain write eligibility for the same complete current reference. Coherent, transforms, history-off view and hosted regressions pass after a clean native rebuild.
- [x] X. Resolve full exact surface references before deciding whether a VC3D surface is editable. SurfaceCatalog resolves the requested owner/generation/object/version even after current deletion, then compares the complete reference to the head. Added inherited/deleted/wrong-object/wrong-owner catalog regressions; production and regression sources pass syntax checks (GUI runtime not built).
- [x] Y. Materialize QuadSurface/stored fits from the selected revision, never current metadata plus old geometry. The shared grid materializer now serves raw and transformed adapters, including editing without validity masking; Spiral uses the same transform path instead of its own scaling loop. Native transform/grid tests pass; changed VC3D production and stored-fit regression sources pass syntax checks (GUI runtime not built).
- [ ] Z. Share CLI/Python historical export reference selection.
- [ ] AA. Cache referenced hosted records through the ordinary owner-qualified cache path.

Agent-facing rules are in the root `AGENTS.md` so they apply to consumers as well as the store implementation. The recurring failure is discarding a resolved owner/version downstream and reconstructing it from current/local state. Keep one resolved state throughout the operation; placement decisions belong only in shared storage routing.

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

Previous completed pass: `bash local/scripts/run_vcobj_pr_tests` (Linux, RelWithDebInfo native build, Python 3.14). Results: 52 native tests passed and one optional SSH test skipped; 199 Python tests passed; 117 Spiral integration tests passed and one optional real-data test skipped. Native fixtures cover plain/packed/sharded stores and source/overlay history on/off combinations. `git diff --check` passed. The existing Spiral `PatchSamplingAtlas` nanobind shutdown leak diagnostic remains; it was present in the baseline and is outside this cleanup. No full-volume performance or macOS result is claimed.

## Follow-up audit tasks

Baseline: `ce429d54a`. The local audit probe reproduced all five issues below. Fix through shared resolution, without proxy records or payload duplication, and add permanent regressions before closing each item.

- [x] H. Restore inherited archived geometry: create the ordinary record override before restoration; never silently retain the newer geometry or dereference a missing local ID.
- [x] I. Exact references in consumers: use the selected record's source identity in selections, fiber exports and other version consumers; support mixed-source saved selections.
- [x] J. UUID namespace: reject object/edge collisions across visible referenced stores, not only within the writable database.
- [x] K. Source identity lookup: resolve indexed source identities across referenced stores with overlay precedence instead of treating inherited objects as absent.
- [x] L. History policy changes: preserve borrowed geometry references rather than copying their payloads into the overlay.
- [x] Add targeted regressions, run the approved PR suite and record results for H–L.

Implemented: H shares override preparation for restored metadata (including relative file references) and selects default restore history from the visible owner. I uses shared reference construction, mixed-source selection artifacts and explicit owning-store snapshot queries for historical exports; hosted listings/downloads carry runtime owner identity too. J checks the shared reference scope without changing trusted initial batch imports. K reuses the indexed visible-row merger. L leaves immutable dependency geometry under its original retention policy. The hosted regression also exposed inherited `serviceOwner` options on read-only dependencies; one shared read-option helper now removes that invalid combination. No persistent tables, columns, proxy records or payload copies have been added.

Permanent regressions cover active/retired/default inherited restores, relative file references, mixed local/inherited selections over paged hosted transport, version-1 selection artifact compatibility, inherited historical geometry export, both cross-kind UUID collision directions, source collection/point duplicates across stores, override/deletion precedence, and repeated On→Off→On→Off history transitions retaining borrowed delta geometry with zero local geometry chunks.

Follow-up verification: `bash local/scripts/run_vcobj_pr_tests` completed successfully on Linux with 52 native tests passed (one optional SSH skip), 199 Python tests passed, and 117 Spiral integration tests passed (one optional real-data skip). `git diff --check` passed. The pre-existing Spiral nanobind shutdown leak diagnostic remains unchanged. No macOS or full-volume performance validation was run. Selection artifacts now encode version 2 while decoding existing version 1; database schema remains 21, with unchanged binary persistent identifiers.

## Consumer and deleted-state audit

Baseline: `9b756e7d3`. Fix the six findings through shared reference and routing APIs, preserving the schema and binary identifiers. Do not add per-object directories or copy referenced payloads.

- [x] M. Preserve complete source/version references in VC3D fiber loading/export, surface history, Spiral relationship comparison and Python Spiral input pins.
- [x] N. Repin edited inherited surfaces using the committed owner from the transaction receipt, not the old source owner.
- [x] O. Reject geometry re-import of deleted inherited identities through shared tombstone-aware lookup.
- [x] P. Resolve history of deleted inherited objects through the same owning-record lookup, without local-store fallback assumptions.
- [x] Q. Read explicitly selected retained geometry independently of whether its object is currently visible.
- [x] R. Add complete exact-reference surface reads and preserve source identity in Lasagna surface references, keeping existing reference syntax readable.
- [x] Add regressions, validate native/Python/consumer tests, inspect the diff and commit.

Audit probes reproduced incorrect owner references, failed repinning, a deleted inherited UUID being re-imported, two base history entries becoming zero through an overlay, and readable raw historical geometry being rejected by typed views. GUI call sites were code-traced; no GUI execution was claimed.

Implemented M–R: shared object-source resolution can return the winning tombstone's owner for history and identity checks, without falling through deletion. Typed views resolve explicit geometry independently of current visibility. Exact surface reads accept the ordinary complete object reference and share the same grid materialization as current reads. Consumers use record references, and changed pins use receipt ownership. Spiral selections/manifests preserve mixed-source references without changing content digests; legacy single-store pins and Lasagna's two-field surface version syntax remain readable. Database schema 21 and binary identifiers are unchanged; no tables, proxy records, directories or geometry payload copies were introduced.

Regressions exercise all four source/overlay history-policy combinations, inherited deletion/re-import, metadata overrides, retained typed geometry after deletion, wrong-object geometry rejection, exact surface reads, consumer pin/manifest round-trips and legacy reference compatibility.

M–R verification: `bash local/scripts/run_vcobj_pr_tests` passed on Linux (RelWithDebInfo, Python 3.14): 52 native tests, 200 Python store tests, 118 Spiral integration tests and 12 Lasagna stored-surface tests. Optional SSH and real-data tests skipped. Lasagna used the existing reader native build with this checkout's Python reader package; no dependencies were installed. All four modified production VC3D translation units passed `c++ -std=c++20 -fPIC -fsyntax-only` using the existing Qt6/OpenCV/dependency includes. `git diff --check` passed. The pre-existing Spiral nanobind shutdown leak diagnostic remains; no GUI runtime, macOS or full-volume performance result is claimed. Changes are committed in separate core-store, VC3D, Spiral and Lasagna blocks.
