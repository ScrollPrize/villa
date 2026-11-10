# Round 3 core review

Reviewed commit: `cf8406afc`. This is a read-only review; no production changes were made. Findings are not capped. The review follows `AGENTS.md`, `reading.md`, `coherent-store-api-cleanup.md`, and the preceding fix tracker.

## Findings

### C3-1 — First deletion of an inherited object is omitted from retained history and the change feed

Priority: P1. Location: `src/History.cpp:96–112`, `src/History.cpp:504`, `src/History.cpp:659–671`; mutation entry point `src/Records.cpp:93–117`.

Trigger: a history-enabled overlay deletes an inherited object without first making a local metadata/geometry revision. The object must be eligible for deletion (for example, a file-backed object with no incident edges).

`deleteObject` creates the same-key override and turns it into a tombstone. At publication, `head()` finds no local history head and its ordinary visible lookup stops at that tombstone. Consequently `p.parent` is empty. The deletion branch only creates a tombstone revision and calls `commit.object(...)` when that parent exists. The current object disappears, but its deletion has neither an `object_revisions` entry nor an `object_versions` entry/change-feed event. Consumers following the feed can retain the deleted object indefinitely.

Runtime verification used the existing `test_coherent_store.cpp` fixture with history enabled in both stores, deleted its unconnected inherited file object, committed, and queried the resulting overlay. Output: `deleted local history rows 0 object_versions 0 visible 0`.

Shared fix direction: preserve the resolved before-state through publication and publish a deletion even when its retained predecessor belongs to another store. Do not introduce a fabricated local baseline, duplicate referenced geometry, or use visible-current resolution to recover the before-state after deletion. Cover direct deletion as the first override, including change-feed and exact/history reads, for history-on and history-off sources.

Classification: missed collection/publication sibling. The local-only conditional predates the previous audit; the earlier tombstone fix preserved an existing local head but did not cover an inherited first deletion.

### C3-2 — Keeping an unchanged recorded relationship still materializes an inherited override

Priority: P2. Location: `src/History.cpp:1118–1120`; automatic caller `src/StoreReader.cpp:218–223`.

Trigger: call `keepRecordedRelationship()` on an unchanged inherited relationship in a history-enabled overlay. `buildCurrentEditRequest()` adds this operation for every undecided incident edge, so an ordinary unrelated or no-op save can trigger it.

After validation, the history-on path unconditionally calls `createRecordOverride` and stamps a content change. No relationship field or dependency needs to have changed. This changes the winning owner and version, advances the commit sequence, and can clear approval for an operation whose purpose is to retain existing evidence. The history-off path and recently fixed `updateEdge`/`repairEdgeGeometry`/`retainRelationship` paths do not do this.

Runtime verification with the existing referenced fixture: `keep sequence 3 -> 4 ownerChanged 1` for one keep operation and no other edit.

Shared fix direction: make dependency validation independent of materializing a record. Use the common relationship no-op decision before creating an override; keep required validation active. Add `keepRecordedRelationship` and the default edit-request builder to the same no-op matrix as update/repair/retain, in both history modes.

Classification: missed sibling of the preceding no-op cleanup, not a newly introduced fix regression. The last round added the history-off case but left this history-on branch intact.

### C3-3 — An unrelated registration changes volume-query length filtering and ordering

Priority: P2. Location: `src/Queries.cpp:199–214`, `src/Queries.cpp:244–277`; related raw-region filtering at `src/GeometryRegion.cpp:331–347`.

Trigger: a fiber's persisted detail length differs from its selected geometry length (for example after geometry edits, or when supplied as an ingestion annotation). Query it by volume and minimum length. Adding any registration anywhere in the transform graph changes which algorithm supplies length, even if that registration does not connect to the queried volume.

With no registrations, `volumeSelection` uses `fibers.length`/`tracks.length`, scaled by the frame. With any registration, it switches to `registeredVolumeSelection`, which measures selected geometry with `transformedLength`. Therefore identical geometry in the same coordinate mapping can appear/disappear or change pagination order merely because an unrelated registration was added. The native/volume region distinction has the same mixed summary-versus-selected-geometry policy and should be included in the cleanup.

Runtime verification through the existing Python binding: stored fiber summary length 100, two line points at `(1,1,1)` and `(11,1,1)` (length 10), volume `a`, minimum length 50. Selection returned 1 object. Adding an identity registration from unrelated `x` to `y` changed the same selection to 0 objects.

Shared fix direction: give selected-geometry length one consistent definition across selection, filtering, ordering and cursor construction. If an indexed cache is retained, maintain it from the same selected geometry; an unrelated transform registration must not choose different semantics. If import-summary filtering remains a separately supported feature, expose it explicitly rather than switching implicitly.

Classification: pre-existing missed sibling of selected-geometry/count cleanup; registration-dependent branching originates before the last fix round.

### C3-4 — Native composite creation/replacement helpers are not operation-atomic

Priority: P2. Locations: `src/SurfaceCreate.cpp:35–41`, `src/VolumeUmbilicus.cpp:59–67`.

Trigger: call one of these public native helpers within a caller-owned `WriteTxn`, catch a geometry validation error, then commit that transaction. The surface helper upserts its object before `putGeometry` validates the geometry. The umbilicus helper first retires the current selection, then upserts and encodes/writes the replacement. Neither wraps its entire operation in the shared savepoint boundary.

Runtime verification for `createSurface`: provide `GeometryData` with kind `surface` but no points array. The call throws `geometry: missing array points`. Catching this and committing leaves the new surface object present (`partial surface 1`). For `setVolumeUmbilicus`, the code path can additionally leave the old selection retired when replacement encoding fails. Session/request wrappers are atomic, so native and request paths disagree.

Shared fix direction: use the same composite-operation/savepoint boundary as the fixed geometry-ingestion batch helper, enclosing selection retirement, object mutation, and geometry publication. Keep inner APIs reusable; do not reproduce rollback logic in individual consumers. Parent review also owns the analogous `recordCatalogTransforms` batch helper.

Classification: missed sibling of the preceding batch atomicity fixes; no indication that the last fix round introduced it.

## Checked candidates that are not findings

- Generic sparse byte edits permit non-finite coordinate payloads while typed geometry APIs reject them. A probe confirmed this, but `tests/cpp/test_object_store_geometry_edits.cpp:121` explicitly documents preservation of non-finite bits by the generic byte API. This is not reported as a newly discovered corruption bug.
- Shared metadata/annotation chunks use content-addressed payload hashes independent of array names. Reusing the helper with another logical name does not create a different hash namespace.
- Geometry archives, current-only descriptors, borrowed payloads and delta parents now resolve through the common immutable-reference infrastructure. No new proxy or current-only-token defect was confirmed in those paths.
- Physical row ordering differs between some optimized single-database and composed traversal/query paths. I did not count that as a correctness finding without a public order guarantee; both use stable ordering for their own path. Consolidating these remains an architectural simplification opportunity, not an additional verified defect.

## Verification

- Used the existing built libraries and Python module under `local/vcobj-pr-tests/build` at the reviewed checkout; no dependency installation or benchmark was run.
- Python query probe ran with `AGENTS_AGENT_MODE=1 PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python /home/hendrik/.venv_las/bin/python`.
- Native probes compiled C++20 from standard input against the existing test libraries using the same include and link inputs as `ninja -C local/vcobj-pr-tests/build -t commands test_coherent_store`. The first reused the test's `Fixture`; the second exercised public `createSurface` and `editGeometry` APIs directly.
- All probe executables and databases were inside Python `TemporaryDirectory` contexts, automatically removed on success or failure. No persistent test source or production code was modified.
- These were focused behavioral probes, not a rerun of the full regression suite. They demonstrate the remaining defects rather than claiming suite-level coverage.

## Coverage and ownership

This list describes inspection coverage, not a claim that every possible input was tested. Implementations were read and traced; helper/header review focused on their contracts and resolver call sites. Root and other agents own the explicitly delegated surfaces below.

| Area | Files reviewed | Disposition |
|---|---|---|
| Identity, mutation, publication | `Database.cpp`, `Provisioning.cpp`, `Records.cpp`, `Metadata.cpp`, `Lifecycle.cpp`, `History.cpp`, `Versions.cpp`, `Transactions.cpp`, `Commands.cpp` | Full implementation pass; C3-1/C3-2; composite siblings checked separately. |
| Reader, selection, query, graph | `StoreReader.cpp`, `Selections.cpp`, `Queries.cpp`, `relationships/Edges.cpp`, `relationships/Traversal.cpp` | Selection/owner/tombstone/endpoint/publication paths reviewed; C3-3. |
| Extensions | `Extensions.cpp`, `ExtensionHistory.cpp`, `internal/Extensions.hpp`, `internal/ExtensionHistory.hpp` | Read/write/current and retained snapshot ownership, source pins, no-op behavior and transaction boundaries reviewed; no additional confirmed finding. |
| Geometry implementation | `geometry/Geometry.cpp`, `GeometryEdits.cpp`, `StructuralGeometry.cpp`, `SequenceMetadata.cpp`, `SharedGeometry.cpp`, `GeometryReference.cpp`, `GeometryDelta.cpp`, `GeometryReader.cpp`, `GeometryViews.cpp`, `TypedGeometry.cpp`, `DependencyIndex.cpp`, `Dependencies.hpp` | Current/exact/borrowed, history modes, payload/delta resolution, identities, sparse/structural writes and validation paths reviewed. |
| Geometry support | `GeometryInternal.hpp`, `ArrayLayout.hpp`, `GeometrySnapshot.hpp`, `GeometryToken.hpp`, `SequenceIdentity.hpp`, relevant read/reference headers | Helper/interface and caller review; no independent full fuzzing or numeric algorithm audit. |
| Spatial and surface helpers | `Spatial.cpp`, `GeometryRegion.cpp`, `SurfaceGrid.cpp`, `SurfaceProximity.cpp`, `SurfaceTileCache.cpp`, `SurfaceCreate.cpp`, `SurfaceDrafts.cpp`, `VolumeUmbilicus.cpp`, `spatial/SurfaceTiles.cpp`, `SurfaceIndex.cpp`, `VertexIndex.cpp`, `spatial/SurfaceGrid.cpp` | Store placement, selected geometry, validity and query path review; C3-3/C3-4. Numeric triangulation algorithms are outside this store-architecture audit. |
| Internal SQL and resolution | `internal/RecordResolver.hpp`, `ReferenceResolver.hpp`, `VisibleRows.hpp`, `RecordOverrides.cpp`, `RecordSql.cpp`, `EdgeSql.cpp`, `RecordJson.cpp`, `RecordFields.hpp`, `JsonFields.cpp`, `Anchors.cpp`, `Schema.cpp`, `Ownership.cpp`, `Publication.hpp`, `Versions.hpp`, `Sqlite.cpp` savepoint handling | Owner/key/visibility/override/history and transaction-boundary review. Low-level OS lock branches were read, not exercised on other platforms. |
| Import/history crossover | `Snapshots.cpp`, `StoreConversion.cpp`, `geometry/GeometryIngestBatch.cpp` | Operation-boundary and selected identity inspection; broader import/export/packing review belongs to root. |
| Transform and tool crossover | `VolumeTransforms.cpp`, `ToolObjects.cpp` | Read for relevant calls; root owns final findings, including request builders and transform batch atomicity. |

Explicitly delegated, not independently claimed as exhaustively reviewed here: `ObjectCollection.cpp`, packed/VFS and import/export implementations, `ObjectCopy.cpp`, `Artifacts.cpp`, `SourcePaths.cpp`, `ObjectStoreSession.cpp`, `HostedSession.cpp`, project/attachment/client/server/transport code, Python adapters, GUI/native consumers and linking workflow implementations. Root and the other review agents cover these. Internal initial-import helpers and low-level non-store numeric utilities are part of that delegated import or numeric surface. No unreviewed core area is silently represented as a test pass.
