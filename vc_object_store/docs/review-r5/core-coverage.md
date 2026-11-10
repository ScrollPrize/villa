# R5 core inspection coverage

Production baseline: `bea32d5d8`. Logical C++ core review by GPT-6.1 Sol. This records source inspection, not runtime reproduction or Astra adjudication. The seven numbered candidate files are proposals, not seven confirmed defects.

The R4 logical-core manifest remains 122 production paths: the tracked source/header diff to R5 added no files in this lane. I inspected logical read/write and resolver paths plus selected consumer entrypoints, keeping the physical storage, file formats, import/pack, transport, conversion and native/Python consumer audits in their independent lanes. Header entries are contract/inline-interface checks; kernel entries are checked at their store, validity and coordinate boundaries, not formally proved numerically.

| File | Disposition |
| --- | --- |
| `vc_object_store/include/vc_object_store/geometry/ArrayBytes.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/BoundsIndex.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/Geometry.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/SegmentClip.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceGrid.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceIndex.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/TriangleDistance.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/geometry/VertexIndex.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/include/vc_object_store/objects/AttachedDocument.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/Extensions.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/Geometry.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/GeometryCoordinates.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/GeometryEdits.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/GeometryRegion.hpp` | Budget, object limit, selected run and cancellation contract. |
| `vc_object_store/include/vc_object_store/objects/ObjectStore.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/ObjectStoreSession.hpp` | Attachment selection, copy/edit and durable-draft contracts; C-003–C-005 context. |
| `vc_object_store/include/vc_object_store/objects/ResidentSurface.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/Selections.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/StoreReader.hpp` | Shared read and mutation-preflight contracts; C-003 and C-005 context. |
| `vc_object_store/include/vc_object_store/objects/SurfaceCreate.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/SurfaceGrid.hpp` | Public selected and transformed surface read interfaces; C-001 context. |
| `vc_object_store/include/vc_object_store/objects/SurfaceTileCache.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/ToolObjects.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/TransactionJson.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/Transactions.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/VolumeTransforms.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/objects/VolumeUmbilicus.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/include/vc_object_store/store/GeometryViews.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/AttachedDocument.cpp` | Document layout/membership snapshots, attachments, exact graph materialization, selected frame/dependency identity. |
| `vc_object_store/src/Commands.cpp` | Transaction request/receipt codec, application order, expectation checks, rollback and request replay. |
| `vc_object_store/src/DraftCheckpoints.cpp` | Manifest staging, recovered draft and request identity; C-004. |
| `vc_object_store/src/DraftCheckpoints.hpp` | Checkpoint/pending state ownership and documented session-only keep-drafts flag; C-004 context. |
| `vc_object_store/src/ExtensionHistory.cpp` | Versioned owner resolution, selected snapshot contents and before-state metadata protections; C-003 distinguishes unversioned rows. |
| `vc_object_store/src/Extensions.cpp` | Registry detection, optional schemas and logical scan/replace dispatch; C-003. |
| `vc_object_store/src/GeometryRegion.cpp` | Visible candidate selection, cut/polyline output and budgets, frame merge and exclusion semantics. |
| `vc_object_store/src/History.cpp` | Publication, restore/tombstone predecessor, dependency metadata, endpoint snapshots, exact graph selection; C-005 and C-007. |
| `vc_object_store/src/Lifecycle.cpp` | Approval, retire, default and explicit restore, edge removals and current-state guards. |
| `vc_object_store/src/Metadata.cpp` | Reserved versus ordinary updates, no-op detection, current/retained version implications. |
| `vc_object_store/src/ObjectStoreSession.cpp` | Attachment barriers, hosted/current selection, artifacts/copy, fetch, draft recovery, paged spatial and observation workflows; C-003–C-005. |
| `vc_object_store/src/Queries.cpp` | Projection, selected geometry length, volume source frames and cursor handling; C-002. |
| `vc_object_store/src/Records.cpp` | Object reads, subtype/table handling, current winner and tombstone publication. |
| `vc_object_store/src/Selections.cpp` | Spatial/ordinary selection and normalized query inputs, projected list and saved selection boundaries. |
| `vc_object_store/src/Snapshots.cpp` | Selected read transaction and retained dependency snapshot behavior. |
| `vc_object_store/src/Spatial.cpp` | Current selected geometry bounds and region acceleration, override inheritance, optional index and update policy. |
| `vc_object_store/src/StoreReader.cpp` | Selected object/reference helpers, copy and edit compatibility, transform/length/attachment adapters; C-003, C-005. |
| `vc_object_store/src/SurfaceCreate.cpp` | Canonical geometry creation and optional transform/attachment staging. |
| `vc_object_store/src/SurfaceDrafts.cpp` | Scratch placement, update and nearest/region geometry boundaries; no confirmed routing defect. |
| `vc_object_store/src/SurfaceGrid.cpp` | Selected canonical and transformed surface wrapper paths; C-001 context. For the separate physical tifxyz reader see the storage lane. |
| `vc_object_store/src/SurfaceProximity.cpp` | Indexed search, draft handoff and target frame queries; incomplete hit identity left unconfirmed without a demonstrated consumer failure. |
| `vc_object_store/src/SurfaceTileCache.cpp` | Immutable geometry-revision cache keys, allocation/budget and retained tile construction. |
| `vc_object_store/src/ToolObjects.cpp` | Project/tool attachment navigation, check/reopen and loaded object behavior. |
| `vc_object_store/src/Transactions.cpp` | Read/write leasing, checked write context, publication/rollback and storage-owned mutex/pool behavior. |
| `vc_object_store/src/Versions.cpp` | Commit/cursor progression, numbered identities, receipt and manifest status; C-003. |
| `vc_object_store/src/VolumeTransforms.cpp` | Scoped registrations and choices, canonical conversion and version baseline. |
| `vc_object_store/src/VolumeUmbilicus.cpp` | Selected active umbilicus and placement through volume transforms. |
| `vc_object_store/src/catalog/Catalog.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/catalog/CatalogStore.cpp` | Placeholder creation, refresh, geometry-only remote fetch and restore identity; C-006. |
| `vc_object_store/src/geometry/ArrayLayout.hpp` | Shape, scalar count and surface channel validation helpers. |
| `vc_object_store/src/geometry/Dependencies.hpp` | Inline point/array applicability and revision dispatch; C-007. |
| `vc_object_store/src/geometry/DependencyIndex.cpp` | Incident-edge candidate reduction and identity-aware indexing boundary; no second endpoint directory. |
| `vc_object_store/src/geometry/DependencyIndex.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/Geometry.cpp` | Full, sparse, structural, and channel writes; affected edge decisions and shared geometry reference publication; C-007. |
| `vc_object_store/src/geometry/GeometryDelta.cpp` | Manifest/parent and chunk-delta replay/validation boundaries. |
| `vc_object_store/src/geometry/GeometryDelta.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/GeometryEdits.cpp` | Desired-to-structural/sparse sequence publication, previous state and guarded edit decisions. |
| `vc_object_store/src/geometry/GeometryInternal.hpp` | Manifest layout, chunk hash and descriptor validation, instrumentation. |
| `vc_object_store/src/geometry/GeometryReader.cpp` | Retained and batched revision resolution, chunk/read/cache pathways and selected validity; C-007 comparison. |
| `vc_object_store/src/geometry/GeometryReader.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/GeometryReference.cpp` | Logical geometry descriptors, current/history reference routes and payload retrieval. |
| `vc_object_store/src/geometry/GeometryReference.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/GeometrySnapshot.hpp` | Internal selected-array byte view contract. |
| `vc_object_store/src/geometry/GeometryToken.hpp` | Shared binary UUID geometry-reference parsing; no separate history-off token format. |
| `vc_object_store/src/geometry/GeometryViews.cpp` | Selected metadata/mapping, editable guard, coordinate transform and canonical surface inverse; C-001. |
| `vc_object_store/src/geometry/SequenceIdentity.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/SequenceMetadata.cpp` | Stable geometry ID/annotation manifests and cache scope; hash names checked as metrics labels. |
| `vc_object_store/src/geometry/SequenceMetadata.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/SharedGeometry.cpp` | Geometry payload persistence, semantic sharing and revision ownership boundaries. |
| `vc_object_store/src/geometry/SharedGeometry.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/SourceMetadata.cpp` | Source geometry identities and relations; ownership/metadata refresh path. |
| `vc_object_store/src/geometry/SourceMetadata.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/geometry/StructuralGeometry.cpp` | Allocation/identity after restore, array splice, grid origin and source metadata handling. |
| `vc_object_store/src/geometry/TypedGeometry.cpp` | Typed array encoding/decoding, stable IDs and surface validity contract. |
| `vc_object_store/src/internal/Anchors.cpp` | Canonical endpoints and shared dependency parsing/validation boundaries. |
| `vc_object_store/src/internal/Anchors.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/CoordinateFilter.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/EdgeSql.cpp` | Projected edge hydration, batched and ordinary insert, lifecycle and collection collision routing. |
| `vc_object_store/src/internal/EdgeSql.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/ExtensionHistory.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/Extensions.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/History.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/JsonFields.cpp` | Shared JSON field utility used by typed record codecs. |
| `vc_object_store/src/internal/JsonFields.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/Publication.hpp` | Publication boundary interface, physical filesystem details covered by storage lane. |
| `vc_object_store/src/internal/ReadRequest.hpp` | Local region budget/cancellation counter and transaction callback boundary. |
| `vc_object_store/src/internal/RecordFields.hpp` | Shared field layout and detail table descriptors used by codecs. |
| `vc_object_store/src/internal/RecordJson.cpp` | Common and typed detail codecs, projected identity field handling. |
| `vc_object_store/src/internal/RecordJson.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/RecordOverrides.cpp` | Override without copying geometry payloads, external locators and historical relocation. |
| `vc_object_store/src/internal/RecordOverrides.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/RecordResolver.hpp` | Binary and public identities, winner/tombstone and incident endpoint resolution. |
| `vc_object_store/src/internal/RecordSql.cpp` | Versions, prepared edits, no-op comparisons, annotations/tags, and publication; C-006 metadata replacement. |
| `vc_object_store/src/internal/RecordSql.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/ReferenceResolver.hpp` | Logical reference scope and physical predicate boundary. |
| `vc_object_store/src/internal/Spatial.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/SpatialCache.hpp` | Resident index/cache policy and current selected region hydration. |
| `vc_object_store/src/internal/SqlValues.hpp` | Internal batched SQL binding and statement-boundary contract. |
| `vc_object_store/src/internal/Sqlite.cpp` | Statement lifetime, active pool and savepoint rollback; prior C-006 retraction verified against actual unordered_multimap. |
| `vc_object_store/src/internal/Sqlite.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/SurfaceDrafts.hpp` | Canonical draft storage and transformed scratch geometry boundary. |
| `vc_object_store/src/internal/TransformedLength.hpp` | Shared selected-geometry polyline integration and unit conversion. |
| `vc_object_store/src/internal/Versions.hpp` | Declaration/interface reviewed against the associated logical implementation. |
| `vc_object_store/src/internal/VisibleRows.hpp` | Visible row merge and canonical object-key filtering. |
| `vc_object_store/src/internal/VolumeFrames.hpp` | Frame lookup over source closure, scaled bounds and transform handoff. |
| `vc_object_store/src/relationships/EdgeGeometry.cpp` | Relationship current applicability and selected bindings share dependency machinery; C-007 dependency context. |
| `vc_object_store/src/relationships/Edges.cpp` | Shared add/update/remove, local and logical selectors and dependency validation; C-007. |
| `vc_object_store/src/relationships/Traversal.cpp` | Local indexed versus referenced traversal, BFS and exact recorded relationship closure. |
| `vc_object_store/src/spatial/SurfaceGrid.cpp` | Physical tifxyz reader boundary and canonical mask handling; detailed external format I/O is delegated to storage. |
| `vc_object_store/src/spatial/SurfaceIndex.cpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/src/spatial/SurfaceTiles.cpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/src/spatial/SurfaceTiles.hpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |
| `vc_object_store/src/spatial/VertexIndex.cpp` | Geometry/index kernel boundary checked for validity and store-independent inputs; no physical store resolution here. |

## Not counted as new defects

- `keepDrafts` intentionally does not persist over restart; the documented outcome is to clear saved recovered drafts because the original editor is gone. C-004 instead concerns checkpoints that expose unstaged requests.
- The prior idle-statement leak report was withdrawn: the actual pool uses `std::unordered_multimap`, so duplicate SQL keys do not reject an insertion.
- The earlier exact graph closure suspicion did not establish a reachable broken lookup through immutable references. Historical write eligibility is distinct from exact read availability.
- Draft scale normalization and omission of retained history in editable current-only owners are documented policy. Read-only or history-off capabilities were not counted as errors solely for omitting unsupported historical operations.
- Region limits remain per attachment/draft group. Existing frame-local cancellation counters were not promoted to a defect without a consuming cumulative-counter contract.
- No builds, installs, tests or runtime probes were run in this lane. No GUI, macOS, real-volume or performance result is claimed.
