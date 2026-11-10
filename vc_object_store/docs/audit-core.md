# vcobj coverage: core

Review baseline: `557a6f925`. Scope: shared handling, identity/retention/placement correctness and duplication, not an exhaustive numerical/security audit. This inventory is generated from tracked core sources and repository-wide consumer search; add newly discovered wrapper/caller files with their discovery reason. No implementation edits during this review.

A disposition applies to the relevant vcobj symbols/paths, not every unrelated line in a large file. Allowed dispositions: PENDING, REVIEWED (name symbols/shared contract checked), FINDING (local ID and scenario), DECLARATION (corresponding implementation), EXCLUDED (specific reason). REVIEWED means no additional issue found in the stated path, not proof of correctness. No finding quota. Report all substantiated defects, combine duplicates, and keep unknowns visible. Code-traced and runtime-tested evidence must be distinguished.

## File ledger

| File | Disposition | Symbols / evidence / finding |
| --- | --- | --- |
| `vc_object_store/include/vc_object_store/geometry/ArrayBytes.hpp` | REVIEWED | copyScalars/scalarBytes/packedItem share endian-safe array decoding; no placement or object identity. |
| `vc_object_store/include/vc_object_store/geometry/BoundsIndex.hpp` | EXCLUDED | BoundsIndex build/query/nearest are resident AABB arithmetic; no persisted identity or store access. |
| `vc_object_store/include/vc_object_store/geometry/Geometry.hpp` | REVIEWED | Owned geometry declarations and validSurfacePoint shared validity contract; TypedGeometry.cpp implementation reviewed. |
| `vc_object_store/include/vc_object_store/geometry/SegmentClip.hpp` | EXCLUDED | clipSegment is standalone numerical slab clipping; no vcobj access. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceGrid.hpp` | REVIEWED | SurfaceGrid owns resident points; explicit validity mask preserves transformed-coordinate semantics; no store routing. |
| `vc_object_store/include/vc_object_store/geometry/SurfaceIndex.hpp` | DECLARATION | Resident SurfaceIndex implementation in spatial/SurfaceIndex.cpp; no persisted version selection. |
| `vc_object_store/include/vc_object_store/geometry/TriangleDistance.hpp` | EXCLUDED | closestPointOnTriangle generic numerical primitive; no store identity or routing. |
| `vc_object_store/include/vc_object_store/geometry/VertexIndex.hpp` | DECLARATION | Resident VertexIndex implementation in spatial/VertexIndex.cpp; row indices explicitly not object identity. |
| `vc_object_store/include/vc_object_store/objects/AttachedDocument.hpp` | DECLARATION | AttachedDocument.cpp attachedCollections/writeVolumeCollections; numeric expected API limitation cross-referenced by C++ consumer lane. |
| `vc_object_store/include/vc_object_store/objects/Geometry.hpp` | FINDING | CORE-7 stale GeometryRevision history-off token comment contradicts uniform UUID implementation. |
| `vc_object_store/include/vc_object_store/objects/GeometryCoordinates.hpp` | REVIEWED | geometryCoordinates/mergeGeometryCoordinates/geometryTransferMetadata central frame validation; volume IDs not object UUIDs. |
| `vc_object_store/include/vc_object_store/objects/GeometryEdits.hpp` | DECLARATION | updateGeometryArrays shared sparse/stable-ID implementation in GeometryEdits.cpp. |
| `vc_object_store/include/vc_object_store/objects/GeometryRegion.hpp` | DECLARATION | RegionOptions/SurfaceProximityQuery current-only spatial API; GeometryRegion.cpp/SurfaceProximity.cpp checked. CPP-5 exact-selection consumer gap. |
| `vc_object_store/include/vc_object_store/objects/ObjectCopy.hpp` | DECLARATION | readObjectCopy/objectCopyOperations contracts; implementation AP provenance issue. |
| `vc_object_store/include/vc_object_store/objects/ObjectStore.hpp` | FINDING | Complete object/version/graph APIs traced to implementations; CORE-7 coordinate comment contradicts arbitrary volume-ID support. |
| `vc_object_store/include/vc_object_store/objects/Selections.hpp` | DECLARATION | SelectionRecord v2 complete origins and exact transforms; Selections.cpp reviewed. |
| `vc_object_store/include/vc_object_store/objects/StoreReader.hpp` | DECLARATION | Exact selectedObject contract; hosted current calls explicitly not one snapshot; StoreReader.cpp and AI/GeometryViews inspected. |
| `vc_object_store/include/vc_object_store/objects/SurfaceCreate.hpp` | DECLARATION | createSurface/surfaceRecord shared SurfaceCreate.cpp. |
| `vc_object_store/include/vc_object_store/objects/SurfaceGrid.hpp` | DECLARATION | readSurfaceGrid ObjectRevision/ObjectVersion selection; SurfaceGrid.cpp reviewed. |
| `vc_object_store/include/vc_object_store/objects/SurfaceTileCache.hpp` | DECLARATION | Immutable geometry UUID cache with shared capacity; SurfaceTileCache.cpp reviewed. |
| `vc_object_store/include/vc_object_store/objects/ToolObjects.hpp` | DECLARATION | noteRead/coordinatesFor/publication contract; CORE-3 implementation violates retained pin metadata. |
| `vc_object_store/include/vc_object_store/objects/TransactionJson.hpp` | DECLARATION | Typed wire/selection/receipt codecs; implementations assigned transport lane, declarations inspected. |
| `vc_object_store/include/vc_object_store/objects/Transactions.hpp` | DECLARATION | Complete version references versus numeric write expectations, receipts/preparations; Versions/Transactions checked, transport lane handles submit. |
| `vc_object_store/include/vc_object_store/objects/VolumeTransforms.hpp` | DECLARATION | Shared transform registration/selection/checkVolumeMapping; VolumeTransforms.cpp reviewed. |
| `vc_object_store/include/vc_object_store/objects/VolumeUmbilicus.hpp` | DECLARATION | VolumeUmbilicus complete version contract; AO implementation mismatch. |
| `vc_object_store/include/vc_object_store/store/GeometryViews.hpp` | DECLARATION | Typed exact/current geometry views and inverse-edit validation; GeometryViews.cpp AI root cross-reference. |
| `vc_object_store/src/AttachedDocument.cpp` | REVIEWED | attachedCollections/writeCollection use shared reader and geometry operations; writeVolumeCollections preserves caller expectations. |
| `vc_object_store/src/Database.cpp` | REVIEWED | openImpl/schema validation distinguishes physical private import from editable access; version/feature guards. |
| `vc_object_store/src/GeometryRegion.cpp` | REVIEWED | RequestScope budgets all physical sources; VisibleObjectRows winners; source-local spatial IDs kept scoped. |
| `vc_object_store/src/History.cpp` | FINDING | AQ missing current-only relationship endpoint pins; CORE-4 exact membership supplementation tests attachment history. |
| `vc_object_store/src/Lifecycle.cpp` | FINDING | AJ: restoreObject source accessibility incorrectly governed by destination history flag; retire shared override/tombstones. |
| `vc_object_store/src/Metadata.cpp` | FINDING | AR: recordLocation promotes before semantic no-op comparison (savepoint rolls back); coordinates validation shared. |
| `vc_object_store/src/ObjectCopy.cpp` | FINDING | AP: copied_from owner uses attachment status; copy payload/annotations otherwise selected together. |
| `vc_object_store/src/Queries.cpp` | REVIEWED | selectObjects/spatialRegions merge physical winners before pagination; connectedTo resolved collection-wide. |
| `vc_object_store/src/Records.cpp` | REVIEWED | upsertObject delegation; objectByUuid shared resolution; deleteObject tombstone and incident-edge handling. |
| `vc_object_store/src/Selections.cpp` | REVIEWED | encode/decodeSelectionRecord preserve owner/generation; recordSelection uses selected record references. |
| `vc_object_store/src/Snapshots.cpp` | REVIEWED | exportMetadataSnapshot uses logical selections and rejects extension loss; import savepoint and shared object/edge writes. |
| `vc_object_store/src/SourcePaths.cpp` | REVIEWED | resolveStoragePath uses selected current owner and packed container; checks source identity. |
| `vc_object_store/src/Spatial.cpp` | REVIEWED | inheritSpatialBounds source-local IDs, shared GeometryReader; optional indexes not authoritative identity. |
| `vc_object_store/src/StoreReader.cpp` | REVIEWED | LocalStoreReader delegates shared reads; selectedObject exact reference path; numeric compatibility selection requires current owner. |
| `vc_object_store/src/SurfaceCreate.cpp` | REVIEWED | surfaceRecord/createSurface central metadata normalization and shared creation APIs. |
| `vc_object_store/src/SurfaceDrafts.cpp` | REVIEWED | Draft geometry has explicit coordinates/attachment; shared frameMapping and SurfaceCut/Nearest; no persistent identity claims. |
| `vc_object_store/src/SurfaceGrid.cpp` | REVIEWED | readSurfaceGrid exact ObjectVersion resolves selected metadata and geometry together. |
| `vc_object_store/src/SurfaceProximity.cpp` | REVIEWED | VisibleObjectRows and RetainedGeometryReader preserve winning source; scoped volume frames. |
| `vc_object_store/src/SurfaceTileCache.cpp` | REVIEWED | Immutable geometry UUID+rectangle cache key; no current-object fallback or backing-file keys. |
| `vc_object_store/src/ToolObjects.cpp` | FINDING | CORE-3 coordinatesFor replaces loaded pin metadata with current; publication pin reduction under consumer audit. |
| `vc_object_store/src/Transactions.cpp` | REVIEWED | snapshot pinning, collection resolver lifetime, checked writes, publication receipts and cache release. |
| `vc_object_store/src/Versions.cpp` | REVIEWED | checkTransactionContext, finishReceipt, object/relationship versions, feed cursors; receipts preserve inherited complete references. |
| `vc_object_store/src/VolumeTransforms.cpp` | REVIEWED | volumeTransforms reads scoped current registrations; selected geometry callers can supply metadata; mapping guards. |
| `vc_object_store/src/VolumeUmbilicus.cpp` | FINDING | AO: returned reference uses attachment owner rather than selected head owner. |
| `vc_object_store/src/geometry/ArrayLayout.hpp` | REVIEWED | geometryByteLength/requireArrayLayout/validateSurfaceChannel central shape validation. |
| `vc_object_store/src/geometry/Dependencies.hpp` | REVIEWED | dependencyGeometry and dependenciesChanged share stable-ID, point, metadata/channel dependency semantics. |
| `vc_object_store/src/geometry/DependencyIndex.cpp` | REVIEWED | structuralEdgeCandidates uses incidentEdges and dependency filtering; physical index is optional accelerator. |
| `vc_object_store/src/geometry/DependencyIndex.hpp` | DECLARATION | structuralEdgeCandidates in DependencyIndex.cpp. |
| `vc_object_store/src/geometry/Geometry.cpp` | FINDING | put/edit/channel paths compare before override; shared parent/payload resolution; CORE-5 structural publication lacks equivalent no-op guard. |
| `vc_object_store/src/geometry/GeometryDelta.cpp` | REVIEWED | decodeGeometryDelta shared reference lookup; hash/parent validation and materialization retained in storage layer. |
| `vc_object_store/src/geometry/GeometryDelta.hpp` | DECLARATION | load/decode/apply/materialize delta helpers; GeometryDelta.cpp reviewed. |
| `vc_object_store/src/geometry/GeometryEdits.cpp` | REVIEWED | updateGeometryArrays preserves source identities, sparse no-op checks and shared geometry writes. |
| `vc_object_store/src/geometry/GeometryExport.cpp` | FINDING | CORE-1 protected paths exclude dependency stores; CORE-2 exact current-only export requires current head. |
| `vc_object_store/src/geometry/GeometryIngest.cpp` | FINDING | CORE-6 track dataset references resolve against attachment directory rather than winning source. |
| `vc_object_store/src/geometry/GeometryIngestBatch.cpp` | REVIEWED | ingestGeometryBatch checks shared identity/tombstones and delegates relationship source resolution. |
| `vc_object_store/src/geometry/GeometryIngestBatch.hpp` | DECLARATION | sourceUuid/ingestGeometryBatch; GeometryIngestBatch.cpp reviewed. |
| `vc_object_store/src/geometry/GeometryInternal.hpp` | REVIEWED | geometrySchema/geometryManifest/hash/chunk layout are central raw array storage; raw geometry intentionally not typed-coordinate normalized. |
| `vc_object_store/src/geometry/GeometryReader.cpp` | REVIEWED | uniform resolveGeometryRecord/payload path; dependency cache lifetime; range/rectangle/batched reads. |
| `vc_object_store/src/geometry/GeometryReader.hpp` | DECLARATION | GeometryReader/RetainedGeometryReader/SurfaceTile shared storage paths implemented in GeometryReader.cpp. |
| `vc_object_store/src/geometry/GeometryReference.cpp` | REVIEWED | resolveGeometryRecord(s) searches archived/current descriptors through scope; validates UUID/checksum identity. |
| `vc_object_store/src/geometry/GeometryReference.hpp` | DECLARATION | resolveGeometryRecord(s) and source-private GeometryRecord; implementation reviewed. |
| `vc_object_store/src/geometry/GeometrySnapshot.hpp` | REVIEWED | Owned byte-array snapshot helpers and coordinate validation; no placement logic. |
| `vc_object_store/src/geometry/GeometryToken.hpp` | REVIEWED | canonicalGeometryToken and bind/read use uniform UUID identity regardless of history policy. |
| `vc_object_store/src/geometry/GeometryViews.cpp` | FINDING | AI root cross-reference: view/viewTransformedAt read current record and current geometry separately through hosted reader. |
| `vc_object_store/src/geometry/InitialGeometry.hpp` | DECLARATION | prepareInitialGeometry/writeInitialGeometry shared creation primitive in Geometry.cpp. |
| `vc_object_store/src/geometry/SequenceIdentity.hpp` | REVIEWED | sequenceIdArray central kind/coordinate-array to stable-ID mapping. |
| `vc_object_store/src/geometry/SequenceMetadata.cpp` | REVIEWED | mergeRetainedAllocations includes referenced current and archived roots; identity/annotation packing. |
| `vc_object_store/src/geometry/SequenceMetadata.hpp` | DECLARATION | MetadataSnapshot/mergeRetainedAllocations, SequenceMetadata.cpp reviewed. |
| `vc_object_store/src/geometry/SharedGeometry.cpp` | REVIEWED | payloadSource/read/visit and MetadataSnapshot resolve chunk hashes through shared scope. |
| `vc_object_store/src/geometry/SharedGeometry.hpp` | DECLARATION | Shared payload access/structural publication; SharedGeometry.cpp/Geometry.cpp (CORE-5). |
| `vc_object_store/src/geometry/SourceMetadata.cpp` | REVIEWED | nativeMetadata uses selected revision for payload annotations and exact relationship selection. |
| `vc_object_store/src/geometry/SourceMetadata.hpp` | DECLARATION | nativeMetadata selected revision contract; implementation reviewed. |
| `vc_object_store/src/geometry/StructuralGeometry.cpp` | FINDING | CORE-5 identity resize/empty splice still publish; retained stable-ID allocation and shared dependencies otherwise. |
| `vc_object_store/src/geometry/TypedGeometry.cpp` | REVIEWED | Owned geometry encode/decode/validation; rejects discarded arrays/attributes, no store routing. |
| `vc_object_store/src/internal/Anchors.cpp` | REVIEWED | canonicalizeEdge/otherEndpoint shared UUID normalization and dependency validation. |
| `vc_object_store/src/internal/Anchors.hpp` | DECLARATION | Canonical endpoint/anchor helper declarations; Anchors.cpp reviewed. |
| `vc_object_store/src/internal/CoordinateFilter.hpp` | REVIEWED | Normalized coordinate filter and SQL bindings share geometryCoordinates validation; internal source query layer. |
| `vc_object_store/src/internal/Database.hpp` | REVIEWED | StatementPool, Impl, transaction/request cache lifetimes and callbacks; public/physical connection boundaries. |
| `vc_object_store/src/internal/EdgeSql.cpp` | REVIEWED | addEdgeImpl collection-wide UUID collision/tombstone checks, shared endpoint lifecycle; row hydration. |
| `vc_object_store/src/internal/EdgeSql.hpp` | DECLARATION | Physical edge row serialization/selection helpers; EdgeSql.cpp and shared collision resolver reviewed. |
| `vc_object_store/src/internal/History.hpp` | DECLARATION | Shared history/publication helpers; History.cpp AQ/CORE-4 findings. |
| `vc_object_store/src/internal/JsonFields.cpp` | REVIEWED | bbox/scale parse helpers; no physical-placement logic. |
| `vc_object_store/src/internal/JsonFields.hpp` | DECLARATION | bboxFromJson/scaleFromJson; JsonFields.cpp reviewed. |
| `vc_object_store/src/internal/RecordFields.hpp` | REVIEWED | Common/detail field visitors centralize JSON and SQL serialization; owner identity attached by reader. |
| `vc_object_store/src/internal/RecordJson.cpp` | REVIEWED | common/object/edge JSON codecs and identity normalization; archive owner attached by resolver. |
| `vc_object_store/src/internal/RecordJson.hpp` | DECLARATION | Record serializers and recordContent transfer semantics; RecordJson.cpp reviewed. |
| `vc_object_store/src/internal/RecordOverrides.cpp` | FINDING | createRecordOverride/prepareOverrideReferences copy before-state descriptors, not payloads; CORE-6 dataset locator omission. |
| `vc_object_store/src/internal/RecordOverrides.hpp` | DECLARATION | createRecordOverride/prepareOverrideReferences; CORE-6 track path omission. |
| `vc_object_store/src/internal/RecordResolver.hpp` | REVIEWED | objectSource/edgeSource, resolveObject/resolveEdge, incidentEdges and binary ObjectVersion codecs; tombstones and winning owner retained. |
| `vc_object_store/src/internal/RecordSql.cpp` | REVIEWED | upsertObjectImpl shared override/identity checks; typed detail hydration; geometryHead uses winner resolver. |
| `vc_object_store/src/internal/RecordSql.hpp` | DECLARATION | Shared insertion/version/lifecycle/detail helpers; RecordSql.cpp reviewed. |
| `vc_object_store/src/internal/ReferenceResolver.hpp` | REVIEWED | resolveReference centralizes reference-source lookup without current-version fallback. |
| `vc_object_store/src/internal/Spatial.hpp` | DECLARATION | Bounds/index/request APIs; Spatial/GeometryRegion/SurfaceProximity implementations reviewed. |
| `vc_object_store/src/internal/SurfaceDrafts.hpp` | DECLARATION | Draft immutable tiles/frame placement; SurfaceDrafts.cpp implementation reviewed. |
| `vc_object_store/src/internal/Versions.hpp` | DECLARATION | Checked transaction/version/feed helpers; Versions.cpp reviewed. |
| `vc_object_store/src/internal/VisibleRows.hpp` | REVIEWED | VisibleObjectRows merges physical cursors using shared winner/tombstone resolver. |
| `vc_object_store/src/internal/VolumeFrames.hpp` | REVIEWED | frameMapping/scopedVolumeFrames/indexed frame traversal retain physical source scope; no consumer routing. |
| `vc_object_store/src/relationships/EdgeGeometry.cpp` | REVIEWED | repairEdgeGeometry validates expected heads then shared update; edgeGeometryState uses recorded source binding. |
| `vc_object_store/src/relationships/Edges.cpp` | REVIEWED | add/update/remove use shared endpoints, identity, tombstones; collection query filters winners before pagination. |
| `vc_object_store/src/relationships/Traversal.cpp` | REVIEWED | component/reachable/neighbours use shared selection for composed stores; exact graph carries selected revisions. |
| `vc_object_store/src/spatial/SurfaceGrid.cpp` | REVIEWED | readSurfaceGrid TIFF planes and shared validSurfacePoint; readSurfaceValidity shared mask conversion. |
| `vc_object_store/src/spatial/SurfaceIndex.cpp` | EXCLUDED | Resident rebuild/locateAll broad/narrow phase uses shared BoundsIndex/TriangleDistance; no persistent object identity. |
| `vc_object_store/src/spatial/SurfaceTiles.cpp` | EXCLUDED | surfaceTiles/SurfaceCut/SurfaceNearest operate passed immutable tile bytes and mapping; no database ownership branch. |
| `vc_object_store/src/spatial/SurfaceTiles.hpp` | DECLARATION | Shared spatial numerical declarations/templates; SurfaceTiles.cpp inspected. |
| `vc_object_store/src/spatial/VertexIndex.cpp` | EXCLUDED | Resident rebuild/nearest arithmetic and original indices; no persistent identity or store access. |

## Findings

All findings below are code-traced against `557a6f925`; no runtime reproduction was run during this read-only audit. Existing AH–AR IDs remain in `coherent-store-api-cleanup.md`, not new discoveries. The lane also traced the root of AI into GeometryViews; consumer-side instances remain owned by their respective audit lanes.

### CORE-1 — Native export can overwrite a referenced backing database

`src/geometry/GeometryExport.cpp:53–64`, `exportGeometry`, protects only `db.path()`, its ancestors, its sidecars and hard links to that one physical path. It does not enumerate the store's referenced/sharded/packed physical closure. With an overlay at `/output/root.vcobj` referring to `/base/base.vcobj`, exporting a fiber with `overwrite=true` to `/base/base.vcobj` passes these checks, stages a fiber JSON, and replaces the referenced database. The same missing closure protection applies to an ancestor directory containing a dependency. This is a data-loss risk, not just a failure to resolve geometry. Use the shared physical-closure path inventory/protection at publication; do not duplicate locator parsing in exporters. Regression: export to every dependency/container path, its hard link, sidecars and ancestors must fail without changing any bytes.

### CORE-2 — Exact native export incorrectly requires the version to remain the visible current head

`src/geometry/GeometryExport.cpp:85–91` rejects an explicitly selected version without `publishedAtUs` unless it equals `currentObjectRevision`. A sealed history-off base's exact version remains available through its reference after an overlay edits or deletes the current object, but exporting that exact surface/track/umbilicus fails with `object version has no publication time`. A geometry/metadata export does not require a historical graph timestamp. Select the supplied complete reference once and export its descriptor/payload independently of current visibility; explicitly separate any native relationship-neighborhood reconstruction requirement. Regression: exact export of a base current-only surface after overlay override and deletion, with the same bytes/metadata as before the override.

### CORE-3 — Tool-derived coordinates replace retained source metadata with current metadata

`src/ToolObjects.cpp:41–63`, `coordinatesFor`, remembers loaded pins in `read_`, but the unmapped branch fetches `reader.currentObject(sources.front())` to choose coordinates. Editing the source's coordinate metadata between loading it and creating a derived surface applies the newer frame to the old loaded geometry. The loaded complete pin is available and should select the frame. `StoreSurfaces::create` is an actual caller (consumer lane traced it). Resolve metadata from the recorded pin, using current only for genuinely unrecorded inputs; canonicalize source keys through the shared UUID helper. Regression: load source, change its metadata, create derived object, verify original frame is retained or a deliberate conflict is raised. Publication's reduction of pins to numeric versions is tracked with the C++ consumer/session audit, not duplicated here.

### CORE-4 — Exact historical graph membership changes with the attachment's history policy

`src/History.cpp:899–923`, `relationshipsAt(RevisionSelection)`, reconstructs historical memberships, then supplements/overwrites them with present incident edges when `value->uuid.empty() || !historyEnabled(db)`. Here `db` is the attachment, not the selected version's owner. A history-off overlay referencing a history-on sealed base therefore uses current base/overlay edge bindings while evaluating old base versions. An edge that was revised or removed after the selected versions can replace their historical membership and make the result inapplicable/absent. The same exact selection through a history-on attachment takes a different path. Determine current-only retention from the resolved selected state/source and never overwrite retained historical membership merely because the reader's attachment has history off. Regression: exact historical graph through both attachment policies, including subsequent edge revision/removal, must match the base exact query.

### CORE-5 — Identity structural edits still publish overrides and new versions

`src/geometry/StructuralGeometry.cpp:81–184,189–239` always calls `publishStructuralSnapshot`; `src/geometry/Geometry.cpp:523` unconditionally publishes a new geometry snapshot. An empty sequence splice (`eraseCount=0`, no parts) and a surface resize with origin `(0,0)` and unchanged dimensions are semantic no-ops, but they still promote inherited records and advance geometry/object identity, commit sequence and approval. Sparse geometry/channel/put paths already distinguish no-ops, so the structural path violates the shared no-op contract. Compare the planned resulting geometry, metadata and stable-ID state before creating an override; preserve required relationship validation. Regression: empty splice and identity resize on local/inherited objects under both history policies, asserting unchanged owner/version/geometry and no publication.

### CORE-6 — Track source ingestion still resolves dataset locators relative to the attachment

`src/geometry/GeometryIngest.cpp:151–152` handles tracks separately and resolves a relative `datasetRef` against `datasetRoot.value_or(db.path().parent_path())`. Existing-source matching at `288–292` similarly uses `target.storeDirectory`, which is populated from the attachment path at `370/395`. Other object kinds call the source-owner-aware shared `resolveStoragePath`. A referenced track whose dataset is beside its owning base fails to ingest from an overlay in another directory (or binds a different same-named dataset). `internal/RecordOverrides.cpp::prepareOverrideReferences` rebases only `storageKind == "file"`; a metadata override retains the same broken dataset-relative hint. Extend the shared source locator capability to typed track datasets and use it for ingestion, matching and overrides, including packed-store locations. Regression: relative external track dataset beside a referenced base, ingestion and metadata override through an overlay in another directory.

### CORE-7 — Public documentation still describes removed identity special cases

Documentation-only inconsistency: `include/vc_object_store/objects/Geometry.hpp`, `GeometryRevision` comment, says history-off tokens derive from object UUID and version, while the implementation now allocates independent geometry UUIDs uniformly. `include/vc_object_store/objects/ObjectStore.hpp`, metadata API comment, says volume UUIDs must be canonical UUID text, while `GeometryCoordinates.hpp` correctly accepts authoritative catalog IDs verbatim. Update these public contracts so callers/agents do not reconstruct tokens or reject valid volume IDs based on obsolete instructions. No runtime defect is inferred solely from these comments.

## Coverage limitations

All 111 assigned entries have a final disposition: 54 REVIEWED, 14 FINDING, 37 DECLARATION and 6 EXCLUDED; zero PENDING. The audit follows the relevant vcobj symbols rather than claiming unrelated numerical algorithms or complete security analysis. Numerical-only spatial primitives are explicitly excluded after checking that they have no store identity/placement logic. Declaration rows name the implementation inspected here or the responsible transport/consumer lane. No additional production file was added to this inventory: traced callers outside it were coordinated with the other lanes (StoreSurfaces/vc_obj2tifxyz, ViewerManager proximity, GUI point collections, hosted session and Python surface input).

Validation is code tracing only, not test execution or a proof of completeness. In particular, CORE-1–6 need the regression scenarios above. Existing AH, AI, AJ, AK, AL, AM, AN, AO, AP, AQ and AR are not counted as new findings; this lane directly reconfirmed AI/AJ/AO/AP/AQ/AR paths. The complete audit is reproducible from the baseline plus this per-file symbol ledger; subsequent code changes require targeted re-review rather than rerunning an unbounded sampled search.

## Implementation and regression evidence

The following evidence is from the subsequent fix pass, not the read-only baseline audit. CORE-1–7 are implemented through `validateExportDestination`, owner-based source-path resolution, reader-backed exact export/graph selection, selected ToolObjects metadata, and shared logical geometry comparison. Restoration and current-only relationship retention use the selected owner's capabilities. Complete write expectations and load-time mapping guards serve project, point-collection and geometry editors without placement-specific branches.

`test_coherent_store` covers all four source/overlay history-policy combinations; mixed-owner exact fiber export; cross-policy revalidated peers; metadata/approval no-op SQL write hooks; identity resize/empty splice; exact export after override/deletion; packed dependency and hardlink/sidecar/ancestor protection; symlinked artifact directories; relative track datasets; wrong-owner equal-counter writes; conflicting command guards; and retained coordinate metadata. `test_object_store_transforms` covers load-time registration changes without object-version changes. Existing initial-import and external-endpoint-history tests verify cross-shard and overlay-created edge resolution.

Review of the implementation delta found and fixed four integration cases before closure: mixed-owner export peers, conflicting numeric/full wire expectations, one-sided current-binding replacement of historical membership, and symlinked artifact protection. New regressions are in the same native suite. Final native run: 52 passed, one optional SSH skip. No schema, persistent routing structure or geometry payload copies were added. Full-stack test totals and code-size accounting live in the consolidated inventory.
