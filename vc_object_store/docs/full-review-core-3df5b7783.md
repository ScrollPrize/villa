# Full vcobj review: core

Baseline: `3df5b7783`. Fresh full-scope review of all assigned relevant files, not only the recent diff. Previous findings R2-1–7 remain in [the combined findings](coherent-store-review-3df5b7783.md); corroborations are not counted again. Every assigned path requires a current disposition and inspected symbols/evidence. No finding quota. Follow additional callers and append rows if needed; report why pure numerical/build-only paths are excluded. Review only: no production fixes or commits.

## Coverage (112 assigned files)

| File | Disposition | Inspected behavior / evidence |
| --- | --- | --- |
| `vc_object_store/include/vc_object_store/geometry/ArrayBytes.hpp` | REVIEWED | scalar endian encoding/decoding, XYZ conversion and packed-item byte lookup. |
| `vc_object_store/include/vc_object_store/geometry/BoundsIndex.hpp` | REVIEWED | resident bbox hierarchy build/overlap/nearest; no store identity. |
| `vc_object_store/include/vc_object_store/geometry/Geometry.hpp` | DECLARATION | owned typed geometry, validity and shared area kernels; no store routing. |
| `vc_object_store/include/vc_object_store/geometry/SegmentClip.hpp` | REVIEWED | shared pure segment slab clipping; no persistence. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceGrid.hpp` | DECLARATION | resident float grid/validity and native tifxyz declaration; not a vcobj resolver. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceIndex.hpp` | DECLARATION | resident surface lookup/filters and shared bounds index declarations. |
| `vc_object_store/include/vc_object_store/geometry/TriangleDistance.hpp` | REVIEWED | shared numeric triangle nearest-point kernel; no persistence. |
| `vc_object_store/include/vc_object_store/geometry/VertexIndex.hpp` | DECLARATION | resident vertex index declaration; no store routing. |
| `vc_object_store/include/vc_object_store/objects/AttachedDocument.hpp` | DECLARATION | loaded reference/mapping, document and editable collection contracts. |
| `vc_object_store/include/vc_object_store/objects/Geometry.hpp` | DECLARATION | immutable descriptor, raw arrays and edit payload schemas; history-independent UUID identity. |
| `vc_object_store/include/vc_object_store/objects/GeometryCoordinates.hpp` | REVIEWED | canonical coordinates, authoritative metadata export and exact scale merge. |
| `vc_object_store/include/vc_object_store/objects/GeometryEdits.hpp` | DECLARATION | checked update/rollback contract and explicit relationship decisions. |
| `vc_object_store/include/vc_object_store/objects/GeometryRegion.hpp` | DECLARATION | bounded spatial query, selected references and mapping returned with visible runs. |
| `vc_object_store/include/vc_object_store/objects/ObjectCopy.hpp` | DECLARATION | current-state copy semantics, provenance and extension limitations. |
| `vc_object_store/include/vc_object_store/objects/ObjectStore.hpp` | DECLARATION | complete record/version, selection, lifecycle, geometry, graph, transaction and policy API declarations. |
| `vc_object_store/include/vc_object_store/objects/ResidentSurface.hpp` | DECLARATION | placement-free resident geometry query declaration. |
| `vc_object_store/include/vc_object_store/objects/Selections.hpp` | DECLARATION | owner-qualified binary durable selection format and capture restrictions. |
| `vc_object_store/include/vc_object_store/objects/StoreReader.hpp` | DECLARATION | exact/current selection, backend snapshot distinction, loaded-reference and allocation contract; known R2 keeper policy text. |
| `vc_object_store/include/vc_object_store/objects/SurfaceCreate.hpp` | DECLARATION | typed creation/provenance and checked request declarations. |
| `vc_object_store/include/vc_object_store/objects/SurfaceGrid.hpp` | DECLARATION | selected record plus geometry/mapping materialization declarations. |
| `vc_object_store/include/vc_object_store/objects/SurfaceTileCache.hpp` | DECLARATION | immutable revision keyed cross-connection cache API. |
| `vc_object_store/include/vc_object_store/objects/ToolObjects.hpp` | DECLARATION | loaded pins, coordinate provenance and artifact publication contracts. |
| `vc_object_store/include/vc_object_store/objects/TransactionJson.hpp` | DECLARATION | wire declarations for exact versions, queries, typed payloads and receipts; implementation transport-owned. |
| `vc_object_store/include/vc_object_store/objects/Transactions.hpp` | DECLARATION | complete ObjectVersion, checked expectations, receipts, typed ops and preparation contracts. |
| `vc_object_store/include/vc_object_store/objects/VolumeTransforms.hpp` | DECLARATION | shared mapping/route-choice/dependency API declarations. |
| `vc_object_store/include/vc_object_store/objects/VolumeUmbilicus.hpp` | DECLARATION | selected umbilicus result and checked replacement declarations. |
| `vc_object_store/include/vc_object_store/store/GeometryViews.hpp` | DECLARATION | read lifetimes, exact descriptor views, transform guards and inverse edit declarations. |
| `vc_object_store/src/AttachedDocument.cpp` | REVIEWED | readCollection/writeCollection; complete loaded reference and mapping guards, attach/replace/volume selection. |
| `vc_object_store/src/Database.cpp` | REVIEWED | create/open provisioning, sealed/read-only gate, reader pools, scoped callbacks. |
| `vc_object_store/src/GeometryRegion.cpp` | REVIEWED | geometryRegion frame recursion, VisibleObjectRows candidates, bounded tile/line reads and returned exact references. |
| `vc_object_store/src/History.cpp` | REVIEWED | publication, object/membership snapshots, owner-aware retained/current-only exact selection, relationship pins. |
| `vc_object_store/src/Lifecycle.cpp` | FINDING | FULL-CORE-1; restoreObject routes retained delta through policy-dependent materialization; retire/delete paths also read. |
| `vc_object_store/src/Metadata.cpp` | REVIEWED | setApproval/mutateMeta no-op before override, source/coordinate validation, journal policy. |
| `vc_object_store/src/ObjectCopy.cpp` | REVIEWED | readObjectCopy selected metadata/payload/annotations, extension refusal, direct/checked write wrappers. |
| `vc_object_store/src/Queries.cpp` | REVIEWED | objectSelectionQuery/filtering, scoped volume frames, tombstone winners, counts/tags/pagination. |
| `vc_object_store/src/Records.cpp` | REVIEWED | objectByUuid, typed lookup, lifecycle deletion checks; known R2 full-record no-op issue unchanged. |
| `vc_object_store/src/Selections.cpp` | REVIEWED | binary selection v1/v2 encoding, canonical keys, owner dictionary and transform snapshot. |
| `vc_object_store/src/Snapshots.cpp` | REVIEWED | metadata export/import, extension refusal, lifecycle and geometry closure checks, whole import savepoint. |
| `vc_object_store/src/SourcePaths.cpp` | REVIEWED | owner-root locator resolution, UUID source validation, export backing-file closure protection. |
| `vc_object_store/src/Spatial.cpp` | REVIEWED | geometry bounds/tile updates, override inheritance, area refresh and optional schema enforcement. |
| `vc_object_store/src/StoreReader.cpp` | REVIEWED | selectedObject exact/current selector, transform/read helpers, endpoint guards; known R2 history-dependent keeper. |
| `vc_object_store/src/SurfaceCreate.cpp` | REVIEWED | surfaceRecord/createSurface operations and checked session wrapper; ordinary typed creation. |
| `vc_object_store/src/SurfaceDrafts.cpp` | REVIEWED | resident draft geometry/index, validity, shared cut/nearest kernels; no persisted owner resolution. |
| `vc_object_store/src/SurfaceGrid.cpp` | REVIEWED | readSurfaceGrid selected ObjectRevision geometry and matching transformation metadata. |
| `vc_object_store/src/SurfaceProximity.cpp` | REVIEWED | scoped frame query, visible winner selection, shared triangle/tile kernel. |
| `vc_object_store/src/SurfaceTileCache.cpp` | REVIEWED | immutable revision/tile cache with mutex-protected LRU and byte capacity. |
| `vc_object_store/src/ToolObjects.cpp` | REVIEWED | noteRead/writable/coordinatesFor selected pins; artifact numeric handoff examined, no equal-counter reproduction established. |
| `vc_object_store/src/Transactions.cpp` | REVIEWED | read/write leases and scoped callbacks, savepoints, validation/publication/rollback ordering. |
| `vc_object_store/src/Versions.cpp` | REVIEWED | expected owner/lineage/version validation, receipts, history generation and paged change feed. |
| `vc_object_store/src/VolumeTransforms.cpp` | REVIEWED | registration/choice construction, scoped transforms, checkVolumeMapping; numeric legacy dependency helper inspected. |
| `vc_object_store/src/VolumeUmbilicus.cpp` | REVIEWED | selected exact umbilicus geometry, mapped reads, guarded replacement request. |
| `vc_object_store/src/geometry/ArrayLayout.hpp` | REVIEWED | geometryLayout range/tile checks and chunk dimensions; pure data-layout helper. |
| `vc_object_store/src/geometry/Dependencies.hpp` | REVIEWED | dependencyGeometry exact shared readers, stable IDs, fingerprints, point/channel/metadata dependency checks. |
| `vc_object_store/src/geometry/DependencyIndex.cpp` | REVIEWED | incident-edge structural candidate selection through shared topology; retained point validation. |
| `vc_object_store/src/geometry/DependencyIndex.hpp` | REVIEWED | dependency candidate/index declarations matched to implementation. |
| `vc_object_store/src/geometry/Geometry.cpp` | REVIEWED | whole/sparse/structural publication, checked dependencies, current descriptors and history retention, no-op logic. |
| `vc_object_store/src/geometry/GeometryDelta.cpp` | FINDING | FULL-CORE-1; materializeGeometry delta branch assumes history tables; codecs/read checks also inspected. |
| `vc_object_store/src/geometry/GeometryDelta.hpp` | REVIEWED | delta encoding/reconstruction/materialization declarations matched to implementation. |
| `vc_object_store/src/geometry/GeometryEdits.cpp` | REVIEWED | checked updateGeometryArrays, sparse no-op planning, sequence IDs/annotations and dependency decisions. |
| `vc_object_store/src/geometry/GeometryExport.cpp` | REVIEWED | exact native export and selected graph closure; safe staging/publication and target guards. |
| `vc_object_store/src/geometry/GeometryIngest.cpp` | REVIEWED | source prepare/registered ingest, direct transaction wrapper, checked source requests, cancellation and failures. |
| `vc_object_store/src/geometry/GeometryIngestBatch.cpp` | FINDING | FULL-CORE-2; whole-batch savepoint missing despite public atomic contract; identity checks and request guards inspected. |
| `vc_object_store/src/geometry/GeometryIngestBatch.hpp` | REVIEWED | batch/source-key internal declarations matched to caller contracts. |
| `vc_object_store/src/geometry/GeometryInternal.hpp` | REVIEWED | geometry schema/dtypes/finite values/hash/chunk layouts; same validation across publication forms. |
| `vc_object_store/src/geometry/GeometryReader.cpp` | REVIEWED | full/delta/current materialization reads, parent fallback, shared hash lookup, bounded reconstruction/cache. |
| `vc_object_store/src/geometry/GeometryReader.hpp` | REVIEWED | retained reader ownership, cache keys and logical geometry API. |
| `vc_object_store/src/geometry/GeometryReference.cpp` | REVIEWED | shared descriptor lookup current/history across scope; exact UUID and owner matching. |
| `vc_object_store/src/geometry/GeometryReference.hpp` | REVIEWED | resolved geometry descriptor/source contract. |
| `vc_object_store/src/geometry/GeometrySnapshot.hpp` | REVIEWED | typed scalar snapshots and bounded retained read adapter. |
| `vc_object_store/src/geometry/GeometryToken.hpp` | REVIEWED | ordinary immutable UUID geometry identities; no policy-specific token branch. |
| `vc_object_store/src/geometry/GeometryViews.cpp` | FINDING | FULL-CORE-3; current view reselects payload; exact views, inverse edits, mapping guards and typed adapters inspected. |
| `vc_object_store/src/geometry/InitialGeometry.hpp` | REVIEWED | initial descriptor creation and payload publication helpers. |
| `vc_object_store/src/geometry/SequenceIdentity.hpp` | REVIEWED | persistent stable sequence identity lookup/update helpers. |
| `vc_object_store/src/geometry/SequenceMetadata.cpp` | REVIEWED | shared sequence annotations, retained allocation over referenced history, metadata cache. |
| `vc_object_store/src/geometry/SequenceMetadata.hpp` | REVIEWED | sequence/annotation/retained-allocation contracts matched to implementation. |
| `vc_object_store/src/geometry/SharedGeometry.cpp` | REVIEWED | content-addressed payload source resolution, verified hash and metadata closure. |
| `vc_object_store/src/geometry/SharedGeometry.hpp` | REVIEWED | shared geometry payload/hash APIs matched to implementations. |
| `vc_object_store/src/geometry/SourceMetadata.cpp` | REVIEWED | nativeMetadata authoritative coordinate export, exact selected topology and annotation projection. |
| `vc_object_store/src/geometry/SourceMetadata.hpp` | REVIEWED | selected native export declaration. |
| `vc_object_store/src/geometry/StructuralGeometry.cpp` | REVIEWED | same-kind exact sequence references, retained point IDs, surface structural edits and dependency checks. |
| `vc_object_store/src/geometry/TypedGeometry.cpp` | REVIEWED | typed encode/decode schema, precision/validity, structural point-ID validation. |
| `vc_object_store/src/internal/Anchors.cpp` | REVIEWED | anchor validation, endpoint canonicalization, opposite-endpoint UUID semantics. |
| `vc_object_store/src/internal/Anchors.hpp` | REVIEWED | shared endpoint/anchor canonical APIs. |
| `vc_object_store/src/internal/CoordinateFilter.hpp` | REVIEWED | coordinate declaration predicates and bounds normalization shared by queries. |
| `vc_object_store/src/internal/Database.hpp` | REVIEWED | transaction-local policy/cache/scoped routing callbacks, writer/read lease ownership. |
| `vc_object_store/src/internal/EdgeSql.cpp` | REVIEWED | edge projection/hydration, UUID collision across scope, lifecycle validation, tombstone reinsertion and batching. |
| `vc_object_store/src/internal/EdgeSql.hpp` | REVIEWED | shared edge selection/insertion/candidate APIs. |
| `vc_object_store/src/internal/History.hpp` | REVIEWED | publication/selection/dependency validation internal declarations. |
| `vc_object_store/src/internal/JsonFields.cpp` | REVIEWED | record field parsing/normalization and default projection semantics. |
| `vc_object_store/src/internal/JsonFields.hpp` | REVIEWED | JSON field conversion declarations. |
| `vc_object_store/src/internal/RecordFields.hpp` | REVIEWED | shared type-specific detail fields and iteration. |
| `vc_object_store/src/internal/RecordJson.cpp` | REVIEWED | canonical full/projected records, fields/identity encoding, lifecycle handling. |
| `vc_object_store/src/internal/RecordJson.hpp` | REVIEWED | shared serialization contract. |
| `vc_object_store/src/internal/RecordOverrides.cpp` | REVIEWED | metadata/descriptor-only override, path rebasing, extension schema and spatial inheritance; no payload copies. |
| `vc_object_store/src/internal/RecordOverrides.hpp` | REVIEWED | createRecordOverride and prepareOverrideReferences contract. |
| `vc_object_store/src/internal/RecordResolver.hpp` | REVIEWED | winner/tombstone/exact-version/shared incident-edge lookup, canonical binary version reference. |
| `vc_object_store/src/internal/RecordSql.cpp` | REVIEWED | record projection/normalization/update/identity and geometry head; known R2 track rebase and no-op issues. |
| `vc_object_store/src/internal/RecordSql.hpp` | REVIEWED | shared record mutation/projection/explicit ID allocation declarations. |
| `vc_object_store/src/internal/ReferenceResolver.hpp` | REVIEWED | pinned source predicate and closure resolution contract. |
| `vc_object_store/src/internal/Spatial.hpp` | REVIEWED | shared bounds/inherited index/area update contract. |
| `vc_object_store/src/internal/SurfaceDrafts.hpp` | REVIEWED | immutable draft/tile placement and shared query kernel declarations. |
| `vc_object_store/src/internal/Versions.hpp` | REVIEWED | version/publication/receipt internal contracts. |
| `vc_object_store/src/internal/VisibleRows.hpp` | REVIEWED | merged physical queries with key-based winner/tombstone filter. |
| `vc_object_store/src/internal/VolumeFrames.hpp` | REVIEWED | scoped native/registered coordinate frame selection and mappings. |
| `vc_object_store/src/relationships/EdgeGeometry.cpp` | REVIEWED | current shared geometry applicability and checked repair against exact selected endpoints. |
| `vc_object_store/src/relationships/Edges.cpp` | REVIEWED | add/update/remove/no-op, explicit retention, scoped selected edges and lifecycle filtering. |
| `vc_object_store/src/relationships/Traversal.cpp` | REVIEWED | local indexed vs scoped traversal, exact selected graph closure and current graph construction. |
| `vc_object_store/src/spatial/SurfaceGrid.cpp` | REVIEWED | native tifxyz geometry/validity reader; file format input, not alternate vcobj owner resolution. |
| `vc_object_store/src/spatial/SurfaceIndex.cpp` | REVIEWED | resident triangle tile hierarchy, nearest kernel, deterministic filtering. |
| `vc_object_store/src/spatial/SurfaceTiles.cpp` | REVIEWED | shared cut/nearest/validity kernels and immutable tile reading. |
| `vc_object_store/src/spatial/SurfaceTiles.hpp` | REVIEWED | shared tile descriptors and query declarations. |
| `vc_object_store/src/spatial/VertexIndex.cpp` | REVIEWED | resident vertex index and nearest query; no database resolution. |

## Additional findings

### FULL-CORE-1 — P2: retained delta restoration still assumes destination history tables

- Locations: `src/Lifecycle.cpp:124–133` calls `materializeGeometry`; `src/geometry/GeometryDelta.cpp:180–192` unconditionally writes `geometry_current_chunks` and `geometry_materializations` for a delta descriptor. History-off destinations intentionally have neither table.
- Concrete case: create a history-on umbilicus, publish two sparse geometry edits, seal the base, create a history-off referencing overlay, then restore the first edited version by its complete reference. Source resolution succeeds, but restoration throws `prepare: no such table: geometry_current_chunks`. Restoring a full descriptor does not take this path.
- Consequence: the advertised cross-policy exact restore works only for some source representation encodings; history retention still changes whether selected geometry can be restored.
- Shared fix boundary: `materializeGeometry` must install the selected logical descriptor/current representation through a retention-independent geometry API. Do not introduce consumer-specific restore fallbacks or history tables into history-off stores.
- Attribution: missed earlier branch integration, not a new regression in `557a6f925..3df5b7783`. Delta-table materialization originates in upstream `ccdcd21c9b`; branch commits `c42feac819` and `2fea874e09` generalized descriptor resolution but left this branch conditional on unavailable storage. This is an additional concrete case of the earlier AJ restore invariant, not the already-listed R2 issues.
- Test gap: history-on sparse/delta source to history-off destination, historical selection differing from current geometry, followed by exact readback and current-state publication checks.

### FULL-CORE-2 — P2: direct public geometry batch API is not atomic on validation failure

- Locations: `src/geometry/GeometryIngestBatch.cpp:69–94` mutates each record/geometry and then imports relationships without a whole-operation savepoint; public wrapper at `:99–102` adds none. `include/vc_object_store/objects/GeometryIngest.hpp:65–70` explicitly promises atomic creation in the caller's transaction.
- Concrete case: in an ordinary write transaction, ingest two new umbilicus sources, the first valid and the second missing its required `points` array. Catch the thrown `geometry: missing array points`, then commit the caller's transaction. Two object records remain: the first complete and the second a geometry-less partial import.
- Consequence: callers that catch a rejected source and continue other work can persist part of a supposedly indivisible import; relationship validation failure after the object loop has the same missing rollback boundary. Checked-submit and directCommit wrappers happen to roll back their whole transaction, masking the lower-level public API defect.
- Shared fix boundary: protect the entire internal `detail::ingestGeometryBatch` operation with the shared savepoint, including records, payloads and topology, rather than relying on each caller to abandon its transaction.
- Attribution: missed pre-existing upstream API contract mismatch; mutation loop and direct wrapper originate in `39a8cf3a12`. Recent identity checks do not introduce the partial-write behavior.
- Test gap: invoke the public batch API inside an existing transaction, catch a late data/topology failure, commit unrelated caller work, verify no batch records/payloads/relations were published.

### FULL-CORE-3 — P2: current typed views independently select metadata and geometry heads

- Locations: `src/geometry/GeometryViews.cpp:169–181` reads `currentObject(id)`, separately reads `currentGeometry(id)`, then attaches the first result's complete reference to the latter payload. Followed transport boundary: `src/HostedSession.cpp:755–772` implements currentGeometry by calling currentObject again; `src/ObjectStoreSession.cpp:520–526` gives hosted callbacks no pinned server snapshot.
- Concrete code-traced case: a hosted `GeometrySnapshot(reader).fiber(id)` begins at version A; another session submit/poll invalidates the attachment head cache after the first object read; currentGeometry fetches version B. The returned view advertises A's reference while exposing B's coordinates. The single-UUID check does not detect this. Local read transactions are safe because both calls share their SQLite snapshot.
- Consequence: callers can carry an incorrect selected identity into caches, provenance or edit preconditions. This is the generic raw typed-view boundary, separate from the previously reported native mixed control/line transformation caller.
- Shared fix boundary: resolve current ObjectRevision once, then look up its exact `geometryRevision`, just as the explicitly selected transformed-view path already does. Preserve the associated complete reference without another moving-head selection.
- Attribution: pre-existing incomplete migration, not introduced in the reviewed delta. The split current geometry read originates in `7e6b82164d`; `4d6b2a6efa` selected the object and `5e6348921c` retained its full reference without tying payload selection to it.
- Test gap: a deterministic changing-head StoreReader or hosted cache invalidation between object and geometry reads. This case is code-traced, not claimed runtime-reproduced.

## Additional followed boundaries

| File | Disposition | Inspected boundary |
| --- | --- | --- |
| `vc_object_store/include/vc_object_store/objects/GeometryIngest.hpp` | FINDING | Public atomic-in-caller-transaction contract for FULL-CORE-2; source decoding and registered/source ingestion declarations also inspected. |
| `vc_object_store/src/ObjectCollection.cpp` | REVIEWED | Bounded follow-up: installCallbacks and collection transaction scope (roughly 210–330 and 380–470). Verified every shard sees all same-store partitions and dependencies; suspected history-off cross-shard endpoint failure ruled out. Whole-file review belongs to transport lane. |
| `vc_object_store/src/HostedSession.cpp` | FINDING | Bounded follow-up: HostedStoreReader currentObject/currentGeometry (755–772) confirms FULL-CORE-3's second moving-head lookup. Whole-file review belongs to transport lane. |
| `vc_object_store/src/ObjectStoreSession.cpp` | REVIEWED | Bounded follow-up: readStore (520–531), local transaction versus hosted callback lifetime. Whole-file review belongs to transport lane. |

## Evidence and limits

Record executed probes separately from code traces. No claim that earlier green tests validate newly found cases. Physical database access is legitimate inside the responsible storage layer; flag duplication or consumer bypass, not all SQL.

All 112 assigned files have fresh full-file dispositions above; no assigned file remains pending. Public declarations were checked against their implementations; pure numerical helpers were inspected for their shared use and marked accordingly, not treated as database resolvers. Additional bounded files are explicitly distinguished from full-file coverage. Previous R2 track rebasing, full-record no-op and history-dependent relationship keeping remain documented there and were not counted again.

Executed native probe: `local/review-core-full.cpp`, compiled against the existing `local/vcobj-pr-tests/build` libraries with C++20 and run as `./local/review-core-full`. It creates all database fixtures inside the existing test support RAII `TemporaryDirectory`, which deletes them on exit. No install, production/test edit or full suite rerun was done. Exact output:

```text
restore error: prepare: no such table: geometry_current_chunks
ingest error: geometry: missing array points
after failed batch committed objects=2
```

The raw-current-view hosted interleaving is code-traced only. No claim of exhaustive scheduling, portability or performance testing is made. Numeric-only tool artifact publication was examined but is not a finding here: no reachable same-number owner replacement through that consumer was demonstrated. Remaining whole-file transport and external-consumer coverage is owned by the coordinator's other ledgers, not silently attributed to this lane.
