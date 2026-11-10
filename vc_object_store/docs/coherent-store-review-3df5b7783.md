# Post-cleanup vcobj review

This is the frozen pre-fix review. All 21 findings are now addressed; implementation and validation are recorded in [the follow-up tracker](coherent-store-fix-tasks.md).

Reviewed commit: `3df5b7783`. The initial review covered the implementation delta from `557a6f925`; the user's follow-up expanded it into a fresh full-inventory review, unchanged files included. Three agents reviewed shared core, transport/storage/extensions, and native consumers; the coordinator reviewed Python consumers/bindings and reconciled findings. This is a review, not another implementation pass. No finding quotas. Existing limitations and speculative concerns are not counted as defects. **Current total: 21 open findings (the original seven plus 14 additional cases); one P1 and 20 P2.**

## Initial delta-review findings

All seven are P2 correctness or API-consistency findings. Five have small runtime reproductions against the existing native build; the two hosted GUI interleavings are code-traced, not GUI-tested. The passing suites from the previous pass did not cover these scenarios.

| ID | Problem | Origin | Evidence |
| --- | --- | --- | --- |
| R2-1 | Hosted fiber loading can mix two registration snapshots within one fiber | New regression in `23f777ef6` | Code-traced |
| R2-2 | Chain renderer discards selected revisions and fetches current geometry/links separately | Pre-existing missed sibling of CPP-2 | Code-traced |
| R2-3 | Full-record track edit undoes relative dataset-path rebasing | Incomplete CORE-6 repair; failure existed before this delta | Runtime reproduced |
| R2-4 | Copying/packing into another directory breaks relative external-source locators | Earlier copy/packing omission; track variant exposed by new resolver | Runtime reproduced for both copy and pack |
| R2-5 | Generic checked-edit helper silently revalidates relationships when destination history is off | Pre-existing history-policy special case | Runtime reproduced under both policies |
| R2-6 | Unchanged full-record saves create versions/overrides through one API but not another | Pre-existing divergent upsert/collection implementations | Runtime reproduced |
| R2-7 | Iterative neural surface copies record the original input instead of the actual preceding input | Pre-existing missed iterative sibling of AL | Runtime reproduced |

### R2-1: one captured transform must serve the entire loaded fiber

`volume-cartographer/apps/VC3D/LineAnnotationController.cpp:15076–15100` captures transforms to derive `fiber.frame`, then asks `viewTransformedAt(head, ...)` to fetch another context for controls. The changed `storedFiberLinePoints` at lines 379–384 directly applies the first affine; width, directions and normals also use that first frame. A hosted registration update between calls can translate controls by 100 voxels while leaving line points in the earlier frame. Hosted `readStore` does not provide a database snapshot across its separate current calls.

The old line helper rejected a mapping mismatch; its removal creates the silent inconsistency. Do not restore redundant fetching as the fix: use the existing shared selected-object transform API with the captured explicit context for all components. Add a deterministic transform-change interleaving test around this consumer, not merely another test of one coherent transform response.

### R2-2: chain display needs the same selected graph path as ordinary rendering

`volume-cartographer/apps/VC3D/FiberCollectionController.cpp:1172–1181` uses listed metadata but separately fetches `currentGeometry(uuid)`. Lines 1204–1222 then fetch current edges, endpoint families and components. A hosted writer can publish doubled stored coordinates together with scale 2 between listing and geometry read: old scale 1 interprets the new points incorrectly. Later links can likewise belong to another revision.

Retain listing references, materialize their exact revisions with one captured transform context, and use shared selected graph/recorded endpoint closure for decorations. The regular renderer was migrated; this sibling was missed. Tests need the chain path with an intervening hosted commit.

### R2-3: full-record updates overwrite the correctly rebased track locator

`vc_object_store/src/internal/RecordSql.cpp:391–395` creates an inherited override but copies back only the rebased `storageRef` when the caller supplied the old unchanged value. The newly rebased `track.datasetRef` is overwritten later by the caller's original relative value.

Reproduction: a sealed base in `base/` has a track with `datasetRef="tracks"`; an overlay in `overlay/` performs a full-record rename. Before the rename, resolution gives `base/tracks`; afterward it gives `overlay/tracks`. The previous regression used `setMeta`, not full-record update. Preserve unchanged locator targets through the common record-update boundary, including track details; do not add a separate caller workaround. No actual track dataset is required to demonstrate the wrong locator.

### R2-4: physical relocation must preserve external source targets

`vc_object_store/src/ObjectCollection.cpp:632–654` copies database snapshots and rewrites collection-member locations, but does not preserve relative `objects.storage_ref` or `track.dataset_ref` targets. The packing path has the same omission. `SourcePaths.cpp:40–44` subsequently resolves those values beside the destination database/container.

Runtime reproduction created a real source fiber JSON and a track dataset directory. Both resolved before export. After `copyObjectCollection` and separately `packObjectCollection` into another directory, fiber resolution returned missing and the track locator pointed at a nonexistent destination-side directory. The original source data still existed. Unrelated external datasets should remain external, as documented, but their targets must not silently change. Centralize relocation of external source locators and consider retained versions as well as current records; do not copy the datasets or patch individual importers.

### R2-5: history policy still changes relationship semantics in the generic editor

`vc_object_store/src/StoreReader.cpp:206,219–220` chooses `KeepRecordedRelationshipOp` when the attachment retains history, but `RetainRelationshipOp` otherwise. The latter silently revalidates a relationship against changed endpoints. This contradicts the helper's promise to keep relationships recorded and the shared owner-aware retention contract.

Reproduction: sealed base has a relationship depending on endpoint metadata `quality=1`; edit that field to 2 through `buildCurrentEditRequest` in an overlay. Applicable relationship count is **0 with history on, 1 with history off**. The retained base endpoint is available under both policies. Use the shared keep/revalidate decision and owner retention rules: unsupported retention should require an explicit decision, not become automatic validation.

### R2-6: no-op full-record edits still depend on which API is used

`vc_object_store/src/ObjectCollection.cpp:791` has its own `editableObject` no-op comparison. Ordinary `ObjectStore::upsertObject` / `UpdateObjectOp` in `src/internal/RecordSql.cpp:385` lacks that comparison and prepares overrides/writes. Submitting the identical record changes version **2→3** and commit sequence **3→4** in the probe; inherited saves can create ownership overrides and discard approval unnecessarily. `writeCollection` also unconditionally emits a full-record update.

Move logical no-op detection to the common update boundary, retaining required relationship validation, and remove the collection-only comparator path. Test both APIs, inherited ownership/approval, and unchanged attachment saves.

### R2-7: chained neural outputs lose their actual input identity

`vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py:2004–2010` updates `source_version` only when loading the original `self.source`. Lines 2038–2040 always save that original object/reference as the output's `input` and `sources`. But `run` changes `current_input` to the previously generated surface at lines 2739–2748.

Runtime reproduction through `_StoredSurfaces`: load A, save B, load B, save C. C records A's reference as its input, not B's. This does not alter geometry but makes the recorded derivation wrong and prevents reconstructing the actual input chain. Keep the loaded complete reference for each current input; root ancestry may be recorded separately. `git blame` places the original-only guard/provenance in `7e6b82164`, so this is not a new regression from adding complete references. Test at least two iterations rather than only a single queued copy.

## Initial delta-review coverage and limits

The frozen production delta contains **88 files**: 50 in `vc_object_store`, 31 native-consumer files, and seven Python-consumer files. All 88 have review dispositions; no assigned delta files remain pending. The core and transport file lists overlap at shared boundaries; their union plus the coordinator's `derivation.py` review covers all 50 store files. This review follows relevant callers and shared boundaries; it is not a new claim of line-by-line coverage of the earlier 444-file inventory or every numerical/UI algorithm. No production or checked-in test code was changed; probes used self-cleaning temporary fixtures and existing build outputs. No dependencies were installed and no full-volume solve was run. **All seven findings remain open: this pass reports them, it does not fix them.**

Python delta disposition: `fit_session.py` and `spiral_service.py` retain the umbilicus owner through request/manifest/provenance; `object_store_inputs.py` uses shared validity and full overlap references; `surface_store.py` delegates exact reads and returns complete references; `fiber_snapping.py` explicitly chooses geometry-only derivation; `fiber_store.py` preserves catalog/geometry identities, canonicalization and source-change rejection; `infer_rowcol_triplet_wraps.py` has R2-7. The shared `derivation.py` policy and its callers were also checked. Scalar-frame restrictions and the neural command's explicitly local store requirement were not reported as new bugs.

The C++ lane covered all 31 changed production files, five changed test/build files and affected reader, geometry, region-request and edit-manager boundaries. Core and transport per-file dispositions are recorded with the review evidence below.

Additional non-blocking concern: resident surface indexing now clones/builds a complete index on some GUI paths. No latency was measured, so this is a profiling requirement, not an eighth confirmed bug. Stale discovery-era “eleven open”/“coverage pending” paragraphs in the old cleanup document should be labeled historical; the latest implementation inventory and this review are the current records.

Local review evidence: `local/review-core-round2.cpp`, `local/review-transport-round2.cpp`, and the corresponding lane Markdown reports. These are reviewer probes, not substitutes for permanent regression tests. Neural provenance was reproduced using the existing `_store_with_surface` fixture and the real `_StoredSurfaces.load/save` APIs, without running inference or replacing those APIs with mocks. Native GUI races remain code-traced. The core/transport checks do not establish macOS behavior, throughput or general concurrency safety.

Lane evidence and per-file dispositions: `local/review-core-round2.md`, `local/review-transport-round2.md`, `local/review-cpp-round2.md`. The core executable reproduced its three cases twice; the transport executable reproduced both copy and pack. `git diff --check` passed. No full-suite rerun was needed for this read-only review; previously green tests are not evidence against the newly reproduced cases.

Investigated but not counted: numeric-only artifact publication guards lack complete pin information, but no reachable equal-counter owner replacement was demonstrated in the current tool caller; native filesystem-only tools retain their explicit boundary; managed extension/artifact copying remains an explicit unsupported capability in the derived-geometry writer. These are not silently promoted to confirmed bugs.

Why another pass found these: R2-1 is an actual regression in the latest consumer simplification. The other six follow operations or siblings missed by the previous tests: full-record rather than field edits, relocating external sources rather than only collection/artifact payloads, the generic relationship builder rather than direct keep-recorded calls, chain rendering rather than ordinary region rendering, and multi-iteration rather than one-shot neural copying. Their source attribution and narrow counterexamples are recorded above; this is not seven new schema/design requirements.

## Full-inventory follow-up: additional findings

The user requested a fresh full scan, not merely the 88-file delta above. The earlier seven remain open and unchanged. The following cases were found in the expanded review, including untouched implementations and consumers. They are additional concrete repair sites, not newly requested features. Shared causes must be fixed through the common interfaces identified below rather than by adding consumer-specific storage branches. Lane-local identifiers are retained for tracing evidence; duplicate corroborations are not counted again.

| ID | Priority | Additional problem | Evidence | Lane |
| --- | --- | --- | --- | --- |
| R2-8 | P2 | Restoring retained delta geometry into a history-off overlay queries nonexistent history tables | Runtime | FULL-CORE-1 |
| R2-9 | P2 | Public geometry batch ingest leaves partial records after a caught failure despite its atomicity contract | Runtime | FULL-CORE-2 |
| R2-10 | P2 | Shared current typed view can bind version A's identity to version B's geometry | Code trace | FULL-CORE-3 |
| R2-11 | P2 | Correction editor drops loaded references and can overwrite concurrent edits/additions | Code trace | CPP-F1 |
| R2-12 | P2 | 3D fiber renderer drops listed references and mixes annotations with newer geometry | Code trace | CPP-F2 |
| R2-13 | P2 | GUI navigation and Python catalogs treat ingestion summaries as current geometry | Runtime for supported geometry edit and stale counts | CPP-F3 |
| R2-14 | P2 | Incomplete surface queries are silently turned into persisted winding annotations | Code trace | CPP-F4 |
| R2-15 | P2 | Sheet-strip cached catalog ordinals select the wrong fiber after catalog changes | Runtime | PY-F1 |
| R2-16 | P2 | Solved normal evidence is rendered against newer source geometry without rejection | Runtime | PY-F2 |
| R2-17 | P2 | Sheet-strip resume skips modified fibers because completion checks only UUID | Code trace | PY-F3 |
| R2-18 | P2 | Spiral materialization rejects valid traced fibers with empty controls and a dense line | Runtime | PY-F4 |
| R2-19 | P1 | Python submit bypasses the lifetime guard: a progress callback closing the store causes SIGSEGV | Runtime | BIND-F1 |
| R2-20 | P2 | Hosted head cache uses noncanonical UUID keys that canonical invalidation does not clear | Runtime | FULL-TRANSPORT canonical-head case |
| R2-21 | P2 | Spatial cache includes deleted objects when retired objects are requested; SQL excludes them | Runtime | FULL-SUPPORT spatial-cache case |

### R2-8: retention policy must not change which logical geometry can be restored

`vc_object_store/src/Lifecycle.cpp:124–133` restores through `materializeGeometry`; the delta path in `src/geometry/GeometryDelta.cpp:180–192` writes `geometry_current_chunks` and `geometry_materializations` even when the destination deliberately has no history tables. A sealed history-on source with sparse edits restored into a history-off overlay throws `prepare: no such table: geometry_current_chunks`. The selected source is resolvable; its encoding wrongly controls success. Use the common retention-independent publication path. Test historical delta source to history-off destination, not just a full snapshot. Detailed probe and attribution: [core ledger](full-review-core-3df5b7783.md#full-core-1--p2-retained-delta-restoration-still-assumes-destination-history-tables).

### R2-9: the public batch's rollback boundary is incomplete

`vc_object_store/src/geometry/GeometryIngestBatch.cpp:69–102` lacks a whole-operation savepoint although `objects/GeometryIngest.hpp:65–70` promises atomic creation within a caller transaction. Ingest a valid object then an invalid one; catch the error and commit unrelated caller work: two batch object records remain, including the partial invalid one. Higher-level checked submit currently masks this by abandoning the entire transaction. Fix the shared batch boundary, including relation import, and test caught late failures. [Core evidence](full-review-core-3df5b7783.md#full-core-2--p2-direct-public-geometry-batch-api-is-not-atomic-on-validation-failure).

### R2-10: shared typed current reads still select the head twice

`vc_object_store/src/geometry/GeometryViews.cpp:169–181` fetches current metadata and current geometry separately, then attaches the first reference to the second payload. Hosted `currentGeometry` fetches the current object again. An intervening cache invalidation can yield an A-labelled view containing B. Local SQLite read transactions hide this issue. Resolve the complete object once, then its exact descriptor. This shared API bug is distinct from consumers independently throwing away a reference. [Core trace and missing interleaving test](full-review-core-3df5b7783.md#full-core-3--p2-current-typed-views-independently-select-metadata-and-geometry-heads).

### R2-11: correction replacement must retain the original edit baseline

`volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp:29–48` converts selected collections to JSON without their references/mapping. `SegmentationModule_Session.cpp:392–425` retains only that JSON. On save, `vc_object_store/src/AttachedDocument.cpp:190–217` guards newly fetched heads and retires current collections missing from the old submitted set. It can overwrite another editor's changes or remove newly attached corrections. Preserve selected references, transforms and baseline membership through the shared edit API. [Native details](full-review-cpp-consumers-3df5b7783.md#cpp-f1--correction-editing-drops-the-loaded-guards).

### R2-12: 3D rendering must not reinterpret listed annotations against a newer line

`volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp:177–225` requests transformed geometry by UUID after receiving a complete listed record. It then uses the old family/tags/gap ranges with the newly selected points. A shortening hosted edit can trigger `Inferred gap outside fiber`; other edits silently mislabel geometry. Use the exact selected revision and captured transform. As with R2-2, the fix is to use the shared selected-graph/view infrastructure, not another renderer-specific resolver. [Native trace](full-review-cpp-consumers-3df5b7783.md#cpp-f2--3d-fiber-rendering-discards-the-listed-revision).

### R2-13: consumers must use actual geometry, not ingestion summaries

`FiberCollectionController.cpp:710–738,767–774` sizes the slider from saved FiberDetail counts but reads current geometry. The Python `tracers/shared/annotations/fiber_store.py` catalog has the same assumption for filtering and sampling. Core explicitly documents those fields as ingestion summaries. A normal checked structural edit changing 4 points to 2 leaves summary count 4; reading point 3 fails. Growing to 6 likewise leaves 4. Length/bbox summaries also remain unchanged. Use shared selected geometry shapes/derived measurements; do not repair this by duplicating summary maintenance. [Reproduction and callers](full-review-cpp-consumers-3df5b7783.md#cpp-f3--ingestion-counts-are-used-as-live-geometry-shape).

### R2-14: partial reads are insufficient evidence for numbered annotations

`SeedingWidget.cpp:1094–1100` ignores the completion/remote state returned by SurfaceCatalog. `finalizePathLabelWraps` at 935–949 persists and consecutively numbers whatever intersections arrived. Missing middle geometry or a budget stop therefore silently changes inferred winding steps. Require completion through the shared query/fetch path or refuse publication. This is persisted incorrect inference, not merely an incomplete display. [Native trace](full-review-cpp-consumers-3df5b7783.md#cpp-f4--partial-surface-intersections-become-persisted-winding-labels).

### R2-15: cached ordinals need a validated source catalog

`sheet_strip/build_index.py` stores catalog ordinals plus a digest, but `sheet_strip/data.py:87–96` applies them to a fresh catalog without checking that digest; training/scoring share this path. Runtime: index unit 0 originally A, retire A, reopen against that index: unit 0 now returns B without error. Validate the saved source selection or resolve stable selected references before using ordinal arrays. [Python evidence](full-review-python-consumers-3df5b7783.md#py-f1--p2-persisted-sheet-strip-catalog-ordinals-are-reused-against-a-different-catalog).

### R2-16: normal evidence needs revision binding, not just store UUID

`fiber_normal_sign_visualization.py:258–277` checks the constraint checksum and source store UUID, then loads current geometry at line 329 while retaining old normal positions at 361–377. A checked +100-voxel geometry edit was accepted: the exported fiber started at `(110,110,110)`, its arrow remained at `(10,10,10)`. A shared evidence/source binding must select the exact source or reject stale results. [Python reproduction and related CurveBank path](full-review-python-consumers-3df5b7783.md#py-f2--p2-solved-normal-evidence-is-combined-with-newer-fiber-geometry).

### R2-17: resume must compare the complete input identity

`sheet_strip/infer_all.py:121–162` skips completed UUIDs without recording/comparing their input revisions. Same UUID with changed geometry therefore reuses stale output after reopening. This is a separate persisted-completion path from R2-15's ordinal mapping. Use a common version-aware completion record and test same-UUID edits. [Python code trace](full-review-python-consumers-3df5b7783.md#py-f3--p2-sheet-strip-resume-treats-unchanged-uuid-as-unchanged-input).

### R2-18: valid dense-line fibers cannot require fabricated controls

`spiral-fitting/object_store_inputs.py:1400–1405` rejects a valid stored fiber with empty control arrays and a nonempty dense line. Runtime materialization failed with `has no control points`; the native store accepted the geometry and native rendering supports it. Missing required array descriptors are NOT claimed valid. Support the valid no-control representation through shared fiber-span handling, without inventing points. [Python reproduction](full-review-python-consumers-3df5b7783.md#py-f4--p2-spirals-fiber-adapter-rejects-valid-traced-fibers-without-controls).

### R2-19: every Python operation must retain its store lease

`vc_object_store/python/objects.cpp:1557–1562` bypasses `PyObjectStore::withStore`, releases the GIL and retains only a raw reference through preparation/submission. Its progress callback can call `close()`, which succeeds because no shared lease exists. Runtime isolated child printed `CLOSE SUCCEEDED DURING SUBMIT` and terminated with SIGSEGV. `start_history_generation` has the same unleased pattern, code-traced only. Use the existing lifetime wrapper, not a second close policy. [Full evidence, safe probe setup and attribution](full-review-python-consumers-3df5b7783.md#bind-f1--p1-python-submit-bypasses-the-shared-lifetime-guard-and-can-crash).

### R2-20: canonicalize once at the hosted cache boundary

`vc_object_store/src/HostedSession.cpp:488–502` builds a head-cache key from caller spelling, while invalidation at 358–363 receives canonical UUIDs. `session.readStore` exposes this raw reader path even though other session entry points normalize first. Runtime: uppercase/lowercase reads both return version 1; after a checked edit, uppercase still returns 1 while lowercase returns 2. Canonicalize through the shared UUID helper at the cache boundary and test both submit and poll invalidation. This is an existing missed entry path, not a new schema feature. Full transport evidence will be retained in its coverage ledger.

### R2-21: cached lifecycle filtering must match uncached selection

`vc_object_store/src/internal/SpatialCache.hpp:35–44,78` loads every non-active state into a retired set, then lets `includeRetired` admit all of them. Deletion markers retain bounds, so this includes deleted objects; the SQL path correctly excludes state 2 regardless of retirement selection. Runtime after a supported deletion: cached default returns 0 regions, cached `include_retired` returns 1, and reopening with cache disabled returns 0. Centralize the lifecycle predicate or exclude deleted objects when building the index, rather than filtering them differently in callers. Test deleted and retired records with cache on/off. This is a missed adaptation when persistent deletion markers were integrated, not a newly requested behavior. The transport ledger records the core support reviewer’s self-cleaning probe.

## Full-scan coverage and validation

The refreshed inventory contains **456 unique files, all dispositioned; zero pending paths**. It combines the previous 444-file inventory, a fresh repository-wide source/build search and followed wrappers/callers. Twelve additional paths were included, notably build integration, the shared derivation module and ResidentSurface declarations; one neural dataset hit was explicitly ruled out as an unrelated in-memory tifxyz representation. A final repeat of the discovery search found **zero unassigned non-test source/build matches**. Four bounded cross-lane evidence rows repeat files already counted elsewhere and do not inflate the 456 total.

| Review lane | Assigned files | Pending | Coverage record |
| --- | ---: | ---: | --- |
| Core records, versions, history, geometry, traversal, transforms, spatial APIs | 112 | 0 | [Per-file ledger](full-review-core-3df5b7783.md) |
| Storage, VFS, sharding, references, import/export, extensions, transport, bindings, detectors | 146 | 0 | [Per-file ledger](full-review-transport-storage-3df5b7783.md) |
| Native consumers, GUI, tools and their integration | 132 | 0 | [Per-file ledger](full-review-cpp-consumers-3df5b7783.md) |
| Python consumers and indirect workflow adapters | 66 | 0 | [Per-file ledger](full-review-python-consumers-3df5b7783.md) |
| **Total unique** | **456** | **0** | |

Discovery included all tracked store implementation/header/binding/build/script files plus repository-wide matches for `vc_object_store`, `vc::objects`, `.vcobj`, `ObjectStoreSession`, `StoreReader`, `PinnedObject`, `geometry_snapshot`, `read_surface_grid`, `current_object_revision`, `graph_read` and `vcobj:`. Source extensions included C/C++/CUDA, Python, Rust, JavaScript/TypeScript, shell and CMake. Followed indirect adapters are listed explicitly. Every row states inspected behavior or a reason it is build-only, numerical-only, test-only or unrelated; these exclusions are not silently described as reviewed store implementations. Relevant regression tests were inspected for missing cases, but this is not a fresh exhaustive review of every test file or proof of unrelated numerical/UI behavior.

Nine additional cases have runtime evidence, bringing the combined report to **14 runtime-supported and seven code-traced findings**. The new probes used the existing native build/bindings, tiny generated fixtures and automatic temporary-directory cleanup. The lifetime crash ran in an isolated child with core dumps disabled. No production code, test code, dependencies or real datasets were changed; no long solve/benchmark or full-suite rerun was performed. `git diff --check` and inventory completeness/duplicate checks pass. GUI scheduling cases remain code-traced; runtime-supported stale-summary evidence verifies the shared state mismatch, not a GUI event sequence. This scope is an exhaustive pass over the discovered relevant file inventory, not a guarantee that no undiscovered defects remain.

Nonblocking documentation inconsistencies are kept separate from correctness counts: `StoreConversion.hpp:11–12` still says a history-mode change mints fresh geometry tokens although the current implementation preserves logical geometry identities; the Python `enable_history` docstring repeats the reload-token guidance. MCP export documentation still names `objectRevision` where the bridge returns a complete `version`. Source conversion into a different directory also shares R2-4's relative-locator issue (code-traced additional caller, not another count).

Not counted: the suspected Python stale `expected_version` bypass was runtime-rejected correctly; native rendering accepts valid empty control arrays; numeric-only guards without a demonstrated reachable equal-counter owner replacement remain risks, not confirmed defects. Documented scalar-only detector frames and local-only CLI input boundaries are existing limitations, not accidental placement bypasses.

Why the count increased: the earlier follow-up explicitly covered only changed production files. This full pass includes unchanged adapters, public lower-level calls, cached versus uncached paths, hosted entry points and persisted workflow caches. That revealed actual missed cases, including a callback-driven crash, rather than 14 new requirements. Future fixes should use these recorded scenarios and per-file dispositions as their regression checklist, preserving the shared-API boundaries rather than repeating open-ended samples.
