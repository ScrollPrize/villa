# Coherent store API cleanup

Current reference/command contract: [one versioned reference representation](versioned-reference-contract.md) and [legacy-removal tracker](reference-review/legacy-removal.md). Earlier review reports below are dated evidence; their command-1/2 and partial-identity compatibility decisions are superseded. Current transactions accept format 3 with complete references only.

Selected-value consumer follow-up: [BV-01–07 implementation and validation](reference-review/bound-consumers/README.md). This consolidates loaded selected state and shared graph capture without changing schema 21 or its history/placement rules.

Subsequent [API fusion and runtime legacy removal](reference-review/api-fusion.md), independently [reviewed by Astra](reference-review/api-fusion-astra.md), supersedes the earlier selected-value Mapping interface, late metadata-override transform method and acceptance of ownerless saved inputs/receipts. Current usage is in [reading](reading.md); historical reports below are not compatibility promises.

Latest completed follow-up: [R4 fixes and validation](review-r4/fixes.md); previous [R3 fixes](review-r3-summary.md) and [21-finding tracker](coherent-store-fix-tasks.md). The dated review sections below retain their historical evidence.

## Open findings from review of `557a6f925`

These findings were originally code-traced during a read-only review. The implementation pass now addresses AH–AR and the additional inventory entries through shared APIs, with targeted regressions and a review of the resulting delta. Current evidence and explicit capability limits are recorded in [the consolidated implementation checklist](coherent-store-audit-inventory.md#implementation-pass); previous suite counts are not used as evidence for newly covered cases.

- [x] AH. Extension registration (`src/Extensions.cpp:203`): reject incompatible inherited definitions before committing registration, using the same collection-wide compatibility rules as scans and exact reads.
- [x] AI. Hosted transforms (`src/HostedSession.cpp:727`): construct registrations, identities and route choices from one coherent server state, not independent current reads guarded only by client polling.
- [x] AJ. Restoration (`src/Lifecycle.cpp:102`): resolve source availability independently of destination history policy, including both mixed-policy directions and current-only reactivation.
- [x] AK. Spiral point editing (`volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp:140`): replace surviving-draft maximum-ID allocation with the shared retained allocator; test deletion plus append in one save and separate saves, and after restore.
- [x] AL. Copy-with-NT (`volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp:874`, `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py:17`): carry complete references through queued jobs and returned provenance, including intervening overlay edits/deletion.
- [x] AM. Neural fiber fingerprints (`vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py:117`): include owning-store identity and bump the digest format; divergent copies must not collide.
- [x] AN. Editor guard (`src/ObjectStoreSession.cpp:1382`): compare the complete pinned reference in `requireCurrent`, not only generation/version.
- [x] AO. Umbilicus references (`src/VolumeUmbilicus.cpp:46`): construct references from the selected owner, not the attachment.
- [x] AP. Copy provenance (`src/ObjectCopy.cpp:30`): retain the selected source owner's identity in `copied_from`.
- [x] AQ. Relationship exact reads (`src/History.cpp:941`): replace the incomplete history-off result constructor with the shared complete endpoint-reference path. Current extension callers do not consume the missing fields; this is an API consistency gap.
- [x] AR. Metadata/approval no-ops (`src/Metadata.cpp:32,66`): compare before preparing an inherited override, while preserving required relationship validation. Existing rollback preserves ownership but still performs unnecessary work.

### Inventory-driven extension of this task list

The extensive follow-up review at the same baseline is recorded in [the consolidated audit inventory and checklist](coherent-store-audit-inventory.md). Its four per-file ledgers cover **444 unique files**, including wrapper/caller closure, with **zero pending dispositions** and no finding quotas. It added **26 entries** (`CORE-1–7`, `TRANS-1–5`, `CPP-1–9`, `PY-1–5`) to AH–AR: **37 addressed task entries in total**. Their canonical checkboxes and detailed evidence live in that inventory and its linked ledgers; do not maintain a second conflicting copy here.

The list explicitly distinguishes code-traced correctness/API issues, fail-closed capability gaps, duplication/maintenance work and stale documentation. No production code changed and no runtime verification was claimed during this review. Coverage of named paths is recorded; it is not a claim that an arbitrary source file or the entire repository is bug-free. Future fixes must migrate the affected sibling callers and add the documented regressions, followed by delta review against this fixed inventory.

### Why repeated reviews find more issues

Source-history check on 2026-11-10: `git blame` of the failing logic, inspection of the relevant commits, and ancestry checks against the already-fetched `origin/main` distinguish inherited upstream assumptions from branch additions. None of the eleven flagged implementations was introduced in the last fix batch (`8c8a8ea00..557a6f925`). This is source-level attribution, not a before/after runtime proof; shared-code changes can expose older assumptions.

| Findings | Origin of the relevant logic | Classification |
| --- | --- | --- |
| AI, AL, AM, AN, AO, AP | `7e6b82164` (upstream object-store hub rework; some surrounding plumbing is older) | Six inherited upstream paths not fully migrated to complete identities/coherent reads |
| AK | `c3d9da6fe` (upstream Spiral object-store consumer) | One inherited local point-ID allocator missed when migrating the fiber editor |
| AH, AJ | `c42feac81` (schema-21 branch implementation) | Earlier branch limitations: local-only schema registration and destination-history restore branching |
| AQ | `2fea874e0` (earlier shared-resolution cleanup) | Earlier cleanup left a second, incomplete relationship-result constructor |
| AR | `ce429d54a` (earlier query/edit cleanup) | Earlier cleanup inserted override preparation into a lookup called before no-op comparison; rollback protects correctness but not cost |

Therefore the current batch is seven inherited upstream paths plus four earlier branch paths, not eleven newly introduced regressions from the last fixes. Fixing an individual reported site did not close the corresponding invariant across all consumers. For example, fixing fiber ID allocation left the point-collection allocator untouched, and fixing surface references left umbilicus/copy/queued-job/cache references untouched.

The review process was incomplete: agents received broad areas plus a request for only their strongest two to four findings. They had no exhaustive assigned-file checklist or persisted negative-review coverage. The result was selective, search-order-dependent inspection, not a random statistical sample and not a repository-wide proof. Completed finding checkboxes and green tests establish only their covered cases. Earlier wording about finishing fixes must not be interpreted as closing the entire architecture audit.

### Finite review and closure procedure

Replace repeated open-ended scans with one coverage ledger at a fixed commit. The inventory review described above has now completed this protocol's discovery/disposition stage; implementation and regression closure remain open. No additional model/schema features are in scope.

1. Enumerate production vcobj APIs and consumers with tracked-file searches, including wrappers, caches, async protocols and hosted adapters. Save the concrete files/symbols assigned to each lane below. Keyword matches are discovery seeds, not proof of coverage; follow callers and wrappers too.
2. Assign each lane a bounded file/symbol inventory. Require a disposition for every entry: shared path used, justified storage-layer special case, covered defect ID, or unchecked. Do not cap the number of findings. Unchecked entries remain visible and cannot be called clean.
3. Reproduce AH–AR with failing tests before fixes; merge duplicates by root cause. Extend one shared API where necessary, then migrate every inventoried sibling caller before closing that invariant. Preserve legitimate validation and history-retention differences.
4. Test a reusable contract matrix: local versus referenced versus sharded/packed; local versus hosted transport; all four source/destination history-policy pairs where applicable; active/retired/deleted/overridden states; exact reads and no-op writes; sibling owners with equal generation/version counters. Concurrency and retained-ID tests are explicit additions, not inferred from ordinary CRUD coverage. Mark inapplicable combinations with a reason rather than silently skipping them.
5. Review the implementation diff for regressions separately from reviewing untouched inventory. Subsequent edits reopen the affected ledger entries and dependent callers, not the entire repository without a recorded reason.
6. Close this audit when all inventory entries have evidence-backed dispositions, all confirmed issues are fixed or explicitly accepted by the user, required contract tests and consumer checks pass, and the final delta review has no unresolved findings. This is an auditable completion condition, not a guarantee of zero undiscovered bugs. Do not require indefinitely repeated clean open-ended agent runs.

| Coverage lane | Scope to enumerate | Known open items | Current coverage |
| --- | --- | --- | --- |
| Identity construction and guards | Core readers, reference constructors, editor pins/receipts, provenance | AN, AO, AP | Partial; exact inventory/dispositions still required |
| Identity transport and caches | VC3D jobs, neural services, Spiral/Lasagna wrappers, serialized references and fingerprints | AL, AM | Partial; exact inventory/dispositions still required |
| History and lifecycle | Object/relationship exact reads, restoration, optional retention, endpoint bindings | AJ, AQ | Partial; exact inventory/dispositions still required |
| Mutation preparation | Metadata, approval, structural edits, overrides, no-op/revalidation paths | AK, AR | Partial; exact inventory/dispositions still required |
| Extension schema/data | Registration, compatibility, exact snapshots, overlay/shard scans and writes | AH | Partial; exact inventory/dispositions still required |
| Snapshot consistency | Hosted selections, transforms, cache invalidation, local parity | AI | Partial; exact inventory/dispositions still required |
| Placement and payloads | Shared geometry/artifact resolution, copy/export/packing, source paths, graph/spatial adapters | Earlier targeted fixes closed; no new item in this pass | Partial; absence of a new finding is not complete coverage |

Report progress as inventoried entries reviewed/total, confirmed issues open/closed, new regressions versus older omissions, and contract cases passing/missing. Historical discovery status at this stage was eleven open code-traced findings, before the full inventory and runtime checks. For the current pass, use [the fix tracker](coherent-store-fix-tasks.md); do not use number of agent runs or total existing test count as the completion metric.

## Latest review tasks

Baseline: `8c8a8ea00`. Resolve these six findings through shared APIs; keep schema 21, binary identities and history-off storage unchanged. Add regression tests, run the approved test runner and record any validation limitations.

- [x] AB. Compare complete selected object identities before transformed geometry writes; reject sibling-overlay versions with equal numeric counters.
- [x] AC. Make project copying preserve the resolved collection and its dependencies/artifacts rather than backing up only the root SQLite file.
- [x] AD. Resolve record-owned history policy centrally; project resource discovery and current-version helpers must not infer it from the attachment's policy.
- [x] AE. Resolve artifacts by complete object reference and carry historical Lasagna model references unchanged.
- [x] AF. Repin fiber edits and related endpoints through shared receipt-aware identity handling, preserving unchanged inherited ownership.
- [x] AG. Allocate structural control-point IDs through retained allocation history, not a GUI-only counter that can rewind after restore.
- [x] Run regressions and consumer checks, review the diff and record results.

Implementation: transformed writes compare the complete retained identity; artifacts and surfaces share `StoreReader::selectedObject`; receipts retain full references (including unchanged inputs) and consumers repin through that contract. Structural IDs use the same retained allocation data as validation, including restored metadata-only versions. Optional history follows the winning record's retention policy. Project copies use the collection snapshot/closure API, with common backup, locator relocation and publication cleanup shared with packing. A copied sharded base remains sealed behind a mutable overlay; an unpartitioned copy retains the existing direct-fork semantics. No persistent tables, proxy records or edit-time payload copies are added.

Validation: `bash local/scripts/run_vcobj_pr_tests` passed on Linux (RelWithDebInfo, Python 3.14): 52 native tests, 200 Python store tests, 118 Spiral tests and 13 Lasagna tests. Optional SSH and real-data tests skipped. `bash local/scripts/check_vcobj_consumers` passed syntax checks of the changed VC3D translation units; `bash local/scripts/run_vcobj_consumer_checks` passed both focused native consumer regressions. `git diff --check` passed. The first native run hit two timeouts and the hosted concurrent-listing scheduling assertion under eight-way test concurrency; two-way runs passed unchanged assertions. The new hosted Python fixture initially omitted its required author; correcting that fixture cleared its failure. The existing Spiral nanobind shutdown leak warning remains. No full GUI, macOS or full-volume performance validation is claimed.

Diff versus `8c8a8ea00`: production adds 452 lines and removes 191 (net +261); regression tests add 311 lines. This pass centralizes missing capabilities rather than reducing total code: collection copying/publication, complete receipts, retained-ID allocation and owner-aware optional history require new shared APIs, while consumer-specific SQL/selection/ID logic is removed. No new persistent tables, schema changes or payload copies during ordinary edits are introduced. Export intentionally copies its dependency closure and artifacts; unrelated external dataset files remain external.

## Current review fixes

Baseline: `3812cb17b`. C++ store plus VC3D adapters; existing CMake/approved PR runner, Linux and macOS portability, unchanged schema 21 and binary identifiers. These fixes must remove duplicated resolution, not introduce more placement-specific branches. Each fix is committed separately with regressions; the final run uses `bash local/scripts/run_vcobj_pr_tests`.

- [x] S. Reject inherited live edge UUID insertion; continue inherited deleted-edge versions through shared tombstone resolution. `test_coherent_store` passes for history on/off insertion rejection and inherited resurrection.
- [x] T. Centralize canonical opposite-endpoint lookup for traversal and attached documents. Coherent-store and transform tests pass, including uppercase attachment removal without retiring the owner; VC3D and source metadata use the same helper.
- [x] U. Preserve ownership on no-op edge update/repair; repair delegates to the ordinary update path instead of duplicating SQL/validation/stamping. Coherent-store and transform regressions pass, including no-op source identity/commit sequence and real endpoint edits.
- [x] V. Route geometry history to the owning record instead of the attachment database. Coherent-store regressions verify both source history policies through both overlay policies.
- [x] W. Carry selected metadata through historical geometry transforms and transform listings. Shared `objectAtGeometry` resolves retained metadata across referenced stores, and local/hosted readers use it. Exact transforms retain selected metadata and only retain write eligibility for the same complete current reference. Coherent, transforms, history-off view and hosted regressions pass after a clean native rebuild.
- [x] X. Resolve full exact surface references before deciding whether a VC3D surface is editable. SurfaceCatalog resolves the requested owner/generation/object/version even after current deletion, then compares the complete reference to the head. Inherited/deleted/wrong-object/wrong-owner catalog regressions pass in a targeted runtime build of the actual consumer sources.
- [x] Y. Materialize QuadSurface/stored fits from the selected revision, never current metadata plus old geometry. The shared grid materializer now serves raw and transformed adapters, including editing without validity masking; Spiral uses the same transform path instead of its own scaling loop. Native transform/grid tests and the existing stored-fit regression executable pass. Missing current fits are reported by the shared geometry loader instead of a preliminary consumer-specific presence query.
- [x] Z. Share CLI/Python historical export reference selection. Both use the selected record's reference constructor, not attachment ownership. Python/CLI JSON export parity passes for sharded imports and referenced geometry collections in both plain and packed stores; all 200 Python store tests pass.
- [x] AA. Cache referenced hosted records through the ordinary owner-qualified cache path. Removed the attachment-owner bypass; mixed local/referenced records prefetch and read from a fresh disk cache with the server stopped. The hosted-cache native regression passes.

Agent-facing rules are in the root `AGENTS.md` so they apply to consumers as well as the store implementation. The recurring failure is discarding a resolved owner/version downstream and reconstructing it from current/local state. Keep one resolved state throughout the operation; placement decisions belong only in shared storage routing.

S–AA validation: `bash local/scripts/run_vcobj_pr_tests` builds the native library and runs the native, Python store, Spiral and Lasagna suites. The focused consumer check, `bash local/scripts/run_vcobj_consumer_checks`, links the actual VC3D/core sources against that native build and runs the existing `test_spiral_stored_fits` executable plus `checkReferencedSurfaceSelections` from `test_surface_catalog`. It avoids rebuilding the entire GUI. All modified consumer translation units and both consumer regression sources also passed `bash local/scripts/check_vcobj_consumers` syntax checks. These local runners contain only build/environment configuration; regression implementations are checked in with their existing CMake test targets.

Final Linux results: 52 native tests passed (optional SSH skipped), 200 Python store tests passed, 118 Spiral tests passed (optional real-data test skipped), and 12 Lasagna tests passed. Both focused C++ consumer runtime checks passed. `git diff --check` passed. The previously documented Spiral nanobind shutdown leak warning remains unchanged.

Production diff versus `3812cb17b`: 204 lines added, 205 removed (net −1); regression tests add 176 net lines. This is primarily consolidation, not a large code-size reduction: removing duplicate adapters, repair logic and routing exceptions pays for the shared historical metadata API and its hosted transport. Schema 21, binary identifiers and history-off storage stay unchanged; no new persistent tables, payload copies or routing directories are introduced. The full GUI, macOS and full-volume performance were not tested.

## Contract

Use the [review contract](review-contract.md) to verify each rule across sibling callers before closing a finding.

Current and exact-version reads, graph queries, edits, geometry, extension data and artifacts must use shared resolution and validation. Physical placement (local, partitioned, referenced, packed) belongs in storage routing, not consumer-specific branches. History policy changes retention, not identity. Preserve binary identifiers, slim history-off storage, checked writes, tombstone precedence and immutable dependencies; do not introduce proxy records or per-object routing directories.

R4 implementation rules: loaded tool outputs retain both the complete pin/write generation and any loaded coordinate-mapping guard, including artifact publication; use shared checked request preparation, not a fresh current lookup. Preserve a transformed reader's explicit validity mask rather than reconstructing it from transformed coordinates. Apply shadowing before region object-limit admission or refill bounded pages with cumulative work budgets; discarded output still costs work. A foreign structural slice does not prove ownership of an already allocated target point ID; restore identity from that target's retained history. Optional report/export files must respect the same backing-store closure protection as geometry exports and publish atomically.

Additional regression-backed rules from [R3](review-r3-summary.md):

- An immutable endpoint reference does not make graph membership immutable. Guard loaded membership and key graph caches by observed store freshness; successful receipts advance freshness without consuming change-feed events. A no-op receipt must not erase the loaded guard.
- Editors sharing one selected surface coordinate their own guarded receipts through the shared edit state. Advance only state actually committed or reloaded: an unchanged endpoint in a receipt does not authorize advancing another editor's relationship-membership baseline. External changes require reload; failed saves preserve pending edits and veto navigation. Same UUID does not imply the same selected version.
- Evidence producers record the version actually used, and consumers validate it before interpreting results. Never stamp old results with a freshly read source identity. Current attachment reads and explicitly selected historical attachment reads have distinct semantics, implemented by shared document/graph APIs.
- Physical locator relocation belongs at the common database-copy boundary, including compaction. Source grouping uses the resolved owner's locator. Composite public writes use operation-wide savepoints, even inside a caller's transaction; tests catch failures and commit the outer transaction to verify atomicity.

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
