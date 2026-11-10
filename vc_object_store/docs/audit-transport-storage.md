# vcobj coverage: transport-storage

Review baseline: `557a6f925`. Scope: shared handling, identity/retention/placement correctness and duplication, not an exhaustive numerical/security audit. This inventory is generated from tracked core sources and repository-wide consumer search; add newly discovered wrapper/caller files with their discovery reason. No implementation edits during this review.

A disposition applies to the relevant vcobj symbols/paths, not every unrelated line in a large file. Allowed dispositions: PENDING, REVIEWED (name symbols/shared contract checked), FINDING (local ID and scenario), DECLARATION (corresponding implementation), EXCLUDED (specific reason). REVIEWED means no additional issue found in the stated path, not proof of correctness. No finding quota. Report all substantiated defects, combine duplicates, and keep unknowns visible. Code-traced and runtime-tested evidence must be distinguished.

## File ledger

| File | Disposition | Symbols / evidence / finding |
| --- | --- | --- |
| `vc_object_store/apps/support/ProcessMetrics.hpp` | EXCLUDED | peakRssBytes/process metrics only; no store identity, selection or payload access. |
| `vc_object_store/apps/support/SqliteMutexProfile.hpp` | EXCLUDED | optional benchmark SQLite mutex instrumentation, not production object resolution. |
| `vc_object_store/apps/vc_object_convert.cpp` | REVIEWED | CLI invokes explicit convertStore/changeHistoryMode; path/report creation is conversion boundary. |
| `vc_object_store/apps/vc_object_export.cpp` | FINDING | TRANS-2 version export insists on current object and cannot accept explicit owner, unlike shared exportGeometry. |
| `vc_object_store/apps/vc_object_import_afv.cpp` | REVIEWED | main/run import AFV/AFC via shared readers and ParallelImportStream, UUID routing and importInitialCollection; no per-endpoint copies; readback opt-in. |
| `vc_object_store/apps/vc_object_ingest.cpp` | REVIEWED | main delegates ingestGeometrySource/importSurfaceChannel; no direct store SQL. |
| `vc_object_store/apps/vc_object_links.cpp` | REVIEWED | main builds detector params then invokes shared detectors; no source path resolution for stored geometry. |
| `vc_object_store/apps/vc_object_server.cpp` | REVIEWED | main options/service lifecycle only; Server shared handlers. |
| `vc_object_store/benchmarks/prepared_import.cpp` | REVIEWED | prepare/native use shared AFV and parallel initial import APIs; reduced/inventory deliberately inspect independent benchmark schemas physically, not runtime readers. |
| `vc_object_store/include/vc_object_store/CtSheetNormal.hpp` | EXCLUDED | CT/lasagna sample API; external voxel sources, no vcobj identity. |
| `vc_object_store/include/vc_object_store/MemMap.hpp` | EXCLUDED | MappedFileRO raw source-file byte mapping, not a store resolver. |
| `vc_object_store/include/vc_object_store/PackedTrackStore.hpp` | EXCLUDED | external packed-track format descriptor, not packed vcobj. |
| `vc_object_store/include/vc_object_store/Progress.hpp` | EXCLUDED | progress callback/task accounting only. |
| `vc_object_store/include/vc_object_store/TerminalProgress.hpp` | EXCLUDED | terminal progress rendering only. |
| `vc_object_store/include/vc_object_store/Tiff.hpp` | EXCLUDED | external TIFF sample/image codecs, no vcobj reads. |
| `vc_object_store/include/vc_object_store/TrackCrossings.hpp` | EXCLUDED | geometric crossing kernel over supplied arrays; no store references. |
| `vc_object_store/include/vc_object_store/Uuid.hpp` | REVIEWED | canonicalUuid shared ASCII normalization/isUuid/newUuid; AFC/root consumers omit normalization (TRANS-3). |
| `vc_object_store/include/vc_object_store/client/HostedStore.hpp` | DECLARATION | HostedStore complete version requests, snapshot listings, artifact transport; HostedStore.cpp; historical relationship result gap TRANS-5. |
| `vc_object_store/include/vc_object_store/migration/ConvertStore.hpp` | DECLARATION | explicit upstream conversion and replacement contract; migration/ConvertStore.cpp. |
| `vc_object_store/include/vc_object_store/object_links/LinkObjects.hpp` | DECLARATION | checked detector preparation and marker publication; linking/LinkObjects.cpp. |
| `vc_object_store/include/vc_object_store/object_links/ObjectLinks.hpp` | FINDING | TRANS-4 OverlapEdgeAt supplies metadata and geometry separately; exact mixed-version materialization cache lacks complete selection identity. |
| `vc_object_store/include/vc_object_store/objects/AfvSource.hpp` | DECLARATION | AfvSource/AfcSource cursors; source format boundary implemented imports/AfvSource.cpp (TRANS-3). |
| `vc_object_store/include/vc_object_store/objects/Artifacts.hpp` | DECLARATION | artifact manifests/content references; shared Artifacts.cpp implementation. |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | DECLARATION | catalog placeholder/fetch/reset APIs; catalog/CatalogStore.cpp. |
| `vc_object_store/include/vc_object_store/objects/Extensions.hpp` | DECLARATION | ExtensionTable versioning/keySpace/source pins; implemented by Extensions.cpp/ExtensionHistory.cpp (AH). |
| `vc_object_store/include/vc_object_store/objects/FiberGraphSolveExtension.hpp` | DECLARATION | typed workflow row builders; imports/FiberGraphSolveExtension.cpp. |
| `vc_object_store/include/vc_object_store/objects/FiberImportExtension.hpp` | DECLARATION | compact extension provisioning and typed measurement rows; imports/FiberImportExtension.cpp. |
| `vc_object_store/include/vc_object_store/objects/FiberJson.hpp` | DECLARATION | version-pinned native fiber export; formats/FiberJson.cpp (TRANS-3 roots). |
| `vc_object_store/include/vc_object_store/objects/GeometryExport.hpp` | DECLARATION | complete ObjectVersion export contract; formats/GeometryOutput.cpp; wrappers gap TRANS-2. |
| `vc_object_store/include/vc_object_store/objects/GeometryIngest.hpp` | DECLARATION | source adapter contracts and checked ingest APIs; source formats reviewed here, GeometryIngest implementation core lane. |
| `vc_object_store/include/vc_object_store/objects/InitialImport.hpp` | DECLARATION | direct immutable parallel builder/routing contract; InitialImport.cpp. |
| `vc_object_store/include/vc_object_store/objects/ObjectCollection.hpp` | DECLARATION | logical UUID key, physical route/closure APIs; ObjectCollection.cpp. |
| `vc_object_store/include/vc_object_store/objects/ObjectStoreSession.hpp` | DECLARATION | pin/read/transport/draft/receipt contracts; ObjectStoreSession.cpp (AN), HostedSession.cpp (AI). |
| `vc_object_store/include/vc_object_store/objects/PackedStore.hpp` | DECLARATION | VFS locators/immutable builder/closure packing; PackedStore.cpp/PackedCollection.cpp. |
| `vc_object_store/include/vc_object_store/objects/Project.hpp` | DECLARATION | project/resource/version edits; Project.cpp (TRANS-1). |
| `vc_object_store/include/vc_object_store/objects/StoreConversion.hpp` | DECLARATION | explicit history-policy maintenance; StoreConversion.cpp. |
| `vc_object_store/include/vc_object_store/objects/UmbilicusFile.hpp` | DECLARATION | external frame import API; UmbilicusFile.cpp. |
| `vc_object_store/include/vc_object_store/objects/imports/ParallelImportStream.hpp` | REVIEWED | ParallelImportStream object/edge phase queues and UUID partitioning; independent readers/writers use shared initial-import callbacks. |
| `vc_object_store/include/vc_object_store/objects/imports/ShardQueue.hpp` | REVIEWED | bounded queue ownership/cancellation only; does not add per-object lookup structures. |
| `vc_object_store/include/vc_object_store/server/Server.hpp` | DECLARATION | hosted transport limits/workers only; Server.cpp delegates storage semantics. |
| `vc_object_store/python/AfvBindings.hpp` | REVIEWED | bindAfvSource delegates source parser; raw-array conversion only, no store identity resolution. |
| `vc_object_store/python/ExtensionBindings.hpp` | REVIEWED | bindExtensionApi/extension_rows_at_version preserve complete ObjectVersion/RelationshipVersion and delegate shared scanners; table codecs are transport-only. |
| `vc_object_store/python/GeometryRegionBindings.hpp` | REVIEWED | geometryRegionFromPy uses one withRead snapshot and shared geometryRegion; result projection only. |
| `vc_object_store/python/ObjectReadBindings.hpp` | REVIEWED | PyGraphReadView retains transaction/store and selected records; geometry revision arrays resolved through shared store APIs. |
| `vc_object_store/python/SessionBindings.hpp` | REVIEWED | bindObjectStoreSession delegates lifecycle/copy/import and events; does not re-resolve physical ownership. |
| `vc_object_store/python/StorageBindings.hpp` | REVIEWED | commonToPy retains version_store/version_generation; PyObjectStore leases/read/write and shared field codecs, no physical resolver. |
| `vc_object_store/python/TypedGeometryBindings.hpp` | REVIEWED | bindTypedGeometry delegates typed views/snapshots/transforms; geometry arrays/validity conversion does not re-read current metadata. |
| `vc_object_store/python/objects.cpp` | FINDING | TRANS-2 export_geometry/restore_object discard explicit owner selection and require current; TRANS-5 relationship revision transport lacks owner. Other wrappers delegate shared APIs/codecs. |
| `vc_object_store/python/vc_object_store/__init__.py` | REVIEWED | native binding re-export only. |
| `vc_object_store/python/vc_object_store/documents.py` | REVIEWED | attach_document/attached_document/point_collections use shared session APIs for local and hosted locations. |
| `vc_object_store/python/vc_object_store/locations.py` | REVIEWED | attached/open_store/HostedStoreHandle adapt transport only; complete exact-reference dictionaries passed unchanged. |
| `vc_object_store/python/vc_object_store/surfaces.py` | REVIEWED | read_surface supports complete reference independent of current owner; SurfaceGrid record preserves owner; create_surface delegates session creation. |
| `vc_object_store/scripts/import_afv.py` | REVIEWED | prepared/import_afv uses native AfvSource and checked upsert/put_geometry/extension APIs; atomic root publication safe for this newly created standalone store without external dependencies/artifacts. |
| `vc_object_store/src/Artifacts.cpp` | REVIEWED | resolveArtifact uses resolveReference; selected manifests and content hashes; ArtifactExport closure validation and shared history reconstruction. Physical sidecar I/O belongs here. |
| `vc_object_store/src/Commands.cpp` | REVIEWED | Executor dispatch/prepare/submit; writes delegate store APIs, checked versions+prepared read deps merged, receipts shared; no consumer SQL resolution. |
| `vc_object_store/src/CtSheetNormal.cpp` | EXCLUDED | CtSheetNormalSampler/LasagnaSheetNormalSampler/PooledLasagnaSheetNormalSampler read external voxel data via reader/mapped pool; numerical correctness outside this vcobj audit. |
| `vc_object_store/src/DraftCheckpoints.cpp` | REVIEWED | DraftDirectory load/write/storeRequest/loadRequest; base ObjectVersion serialized completely; keepDrafts intentionally session-only per header; payloads use transaction codec. |
| `vc_object_store/src/DraftCheckpoints.hpp` | DECLARATION | DraftMeta/PendingRecord/DraftDirectory; complete base pin and explicit recovery policy, implementation DraftCheckpoints.cpp. |
| `vc_object_store/src/ExtensionHistory.cpp` | FINDING | AH versionedTable takes first matching schema; replaceExtensionRows, readSnapshot and version reads use shared owner/geometry-payload resolution. |
| `vc_object_store/src/Extensions.cpp` | FINDING | AH createExtensionTable checks only local schema; scans/readers use shared sources and owner-version resolution. |
| `vc_object_store/src/FetchQueue.cpp` | REVIEWED | request/refetch/fetchNow/run; fetch scheduler only, store lookup/write delegated to session callback; no placement resolver. |
| `vc_object_store/src/FetchQueue.hpp` | DECLARATION | queue scheduler declarations, FetchQueue.cpp; no object resolver. |
| `vc_object_store/src/HostedSession.cpp` | FINDING | AI volumeTransforms mixes listing/head/choice snapshots. Exact-version/payload caches otherwise carry owner+lineage; history-off feed reset observed. |
| `vc_object_store/src/HostedSession.hpp` | DECLARATION | hosted attachment caches/events; HostedSession.cpp (AI). |
| `vc_object_store/src/InitialImport.cpp` | REVIEWED | importInitialCollection, appendObjectRefs/importEdges; independent shard staging, supplied UUID references, cancellation/publication; direct construction is explicit immutable-import boundary. |
| `vc_object_store/src/MemMap.cpp` | EXCLUDED | MappedFileRO OS raw mapping of external source arrays; no store selection. |
| `vc_object_store/src/ObjectCollection.cpp` | REVIEWED | uuidReadOrder/installCallbacks/load; UUID tombstones, sealed dependency pins, exact scope; copyObjectCollection uses shared snapshot/artifact closure; direct SQL is storage-layer routing/maintenance. |
| `vc_object_store/src/ObjectStoreSession.cpp` | FINDING | AN requireCurrent omits owner equality. pinOf, artifact selection, copy/durable receipt repinning use complete references; readStore remote snapshot limitation also AI. |
| `vc_object_store/src/PackedCollection.cpp` | REVIEWED | packObjectCollection imports sealed closure and relocates physical locators without object identity rewrites. |
| `vc_object_store/src/PackedStore.cpp` | REVIEWED | PackedFile VFS callbacks, importDatabase/finalize/compactPackedStore; immutable published members, shared snapshot backup and artifact export. Byte allocation/eviction implementation excluded from identity audit. |
| `vc_object_store/src/PackedTrackStore.cpp` | EXCLUDED | resolvePackedTrackStore/mapArray decode external track directories; distinct from packed vcobj. |
| `vc_object_store/src/Project.cpp` | FINDING | TRANS-1 setProjectName refreshes expected version after reading stale editable record; other resource/view operations carry read version and closure copy. |
| `vc_object_store/src/Provisioning.cpp` | REVIEWED | writeRecordVersion, registerStorageTables, forkStorageIdentity; physical table IDs separated from logical versions; unsupported identity-FK extension fork rejected explicitly. |
| `vc_object_store/src/StoreConversion.cpp` | REVIEWED | changeHistoryMode attaches collection resolver before current/history materialization; physical schema rewriting isolated to explicit maintenance, unknown extension programs retained/validated. |
| `vc_object_store/src/Tiff.cpp` | EXCLUDED | readTiffSamples/writeTiffSamples external TIFF codec; geometry export/import format boundary only. |
| `vc_object_store/src/TrackCrossings.cpp` | EXCLUDED | scan_crossing_events/consolidate_crossing_events arithmetic over supplied track arrays; no database access. |
| `vc_object_store/src/UmbilicusFile.cpp` | REVIEWED | external umbilicus JSON conversion and importVolumeUmbilicus delegates shared setVolumeUmbilicusRequest. |
| `vc_object_store/src/Uuid.cpp` | REVIEWED | nameUuid shared binary namespace parsing and RFC v5 hash; namespace canonicalization through parseUuid. |
| `vc_object_store/src/Wire.cpp` | FINDING | TRANS-5 relationshipJson/from lack owner identity; object revision/selection and exact input versions preserve owner; geometry uses immutable identity. |
| `vc_object_store/src/catalog/Catalog.hpp` | DECLARATION | catalog metadata/read adapters; CatalogStore.cpp. |
| `vc_object_store/src/catalog/CatalogStore.cpp` | REVIEWED | catalog registration conversion uses store object reads/commands and canonical volume binding; no consumer SQLite. |
| `vc_object_store/src/client/HostedStore.cpp` | REVIEWED | pageAll/CatchUp, recordSelection, objectHistory/objectsAtVersions; shared wire API and status-based pagination guards; exact input pins preserved. |
| `vc_object_store/src/formats/FiberInput.cpp` | REVIEWED | fiberDocument/fibers read external JSON into Prepared records; no current/history resolver. |
| `vc_object_store/src/formats/FiberInput.hpp` | REVIEWED | external fiber JSON preparation declarations/helpers; FiberInput.cpp and shared sourceObject path. |
| `vc_object_store/src/formats/FiberJson.cpp` | FINDING | TRANS-3 fiberNetworkJson roots compared without UUID canonicalization; exact exports otherwise preserve selected revision/geometry. |
| `vc_object_store/src/formats/FiberMetadata.hpp` | DECLARATION | fiber source metadata conversion; FiberInput/FiberJson formats. |
| `vc_object_store/src/formats/GeometryInput.cpp` | REVIEWED | readJson/deriveSourceUuid/sourceObject canonicalize external source identities and prepare metadata; filesystem access is the source-format boundary. |
| `vc_object_store/src/formats/GeometryInput.hpp` | DECLARATION | Prepared/readJson/sourceObject external source adapters; GeometryInput.cpp. |
| `vc_object_store/src/formats/GeometryOutput.cpp` | REVIEWED | exportGeometry resolves selected ObjectRevision, uses stored geometry and selected metadata; format output is explicit conversion boundary. |
| `vc_object_store/src/formats/GeometryOutput.hpp` | DECLARATION | export source-format helpers; GeometryOutput.cpp. |
| `vc_object_store/src/formats/PointCollectionInput.cpp` | REVIEWED | collections/collidingCollections external format conversion into Prepared records; fresh point IDs allocated for new imported objects only. |
| `vc_object_store/src/formats/RemoteSource.cpp` | REVIEWED | fetchRemoteSource/fetchRemoteSurface fetch external format staging via vc_reader; does not resolve stored vcobj payloads. |
| `vc_object_store/src/formats/RemoteSource.hpp` | DECLARATION | remote staging capability boundary; RemoteSource.cpp. |
| `vc_object_store/src/formats/SurfaceInput.cpp` | REVIEWED | surfaceGeometry/surface/surfaceChannel convert tifxyz metadata/channels to canonical geometry; source file access is format boundary. |
| `vc_object_store/src/formats/SurfaceInput.hpp` | REVIEWED | external surface channel discovery/format checks; source preparation, not current/history fallback. |
| `vc_object_store/src/formats/TrackInput.cpp` | REVIEWED | track reads packed external track arrays and prepares ingest geometry; no vcobj placement assumptions. |
| `vc_object_store/src/formats/TrackInput.hpp` | REVIEWED | track source identity/preparation helpers; external packed source format, not physical vcobj lookup. |
| `vc_object_store/src/formats/UmbilicusInput.cpp` | REVIEWED | umbilicus converts external JSON into Prepared geometry; binding delegated to shared ingest. |
| `vc_object_store/src/imports/AfvRecords.hpp` | REVIEWED | afvObject/constraint record conversion creates canonical UUID identities and explicit zero-length control arrays; source adapter only. |
| `vc_object_store/src/imports/AfvSource.cpp` | FINDING | TRANS-3 AfcSource compares unnormalized source provenance UUID against canonical AFV UUID. |
| `vc_object_store/src/imports/FiberGraphSolveExtension.cpp` | REVIEWED | fiberGraphSolveTables and typed row constructors; run/input pins and owner keys remain binary, no physical database assumptions. |
| `vc_object_store/src/imports/FiberImportExtension.cpp` | REVIEWED | fiberImportTables/provisioning and measurement/normal row construction; compact source identity metadata, typed extension rows. |
| `vc_object_store/src/imports/SourceIdentityIndex.cpp` | REVIEWED | lazy source_collections/source_points accelerator maintained from source_identity_dirty; sourcePoint/sourceCollection query shared VisibleObjectRows across references/shadows. Local SQL here is an implementation index, not a consumer resolver. |
| `vc_object_store/src/imports/SourceIdentityIndex.hpp` | DECLARATION | lazy local source-identity accelerator and shared visible-source lookup; SourceIdentityIndex.cpp. |
| `vc_object_store/src/imports/SourceRelationships.cpp` | REVIEWED | import source relationships resolve SourceIdentityIndex then shared addEdge/geometry ownership; import adapters only. |
| `vc_object_store/src/imports/SourceRelationships.hpp` | DECLARATION | source relationship adapters; SourceRelationships.cpp. |
| `vc_object_store/src/internal/Artifacts.hpp` | DECLARATION | resolved artifact API/export closure boundary; Artifacts.cpp. |
| `vc_object_store/src/internal/CollectionReads.hpp` | DECLARATION | attachCollectionReferenceResolver; implementation ObjectCollection.cpp installs shared routing for maintenance snapshots. |
| `vc_object_store/src/internal/DatabaseCopy.hpp` | REVIEWED | copySqliteSnapshot/relocateCollectionMembers; physical backup and locator rewrite are shared storage-maintenance boundary, not consumer resolution. |
| `vc_object_store/src/internal/Digest.hpp` | REVIEWED | binary SHA256 adapter; parseDigest duplicates parseHash but intentionally accepts uppercase inputs versus strict stored lowercase; non-routing codec. |
| `vc_object_store/src/internal/ExtensionHistory.hpp` | DECLARATION | managed metadata guard and selected snapshot APIs; ExtensionHistory.cpp (AH). |
| `vc_object_store/src/internal/Extensions.hpp` | DECLARATION | private typed SQL operations; shared Extensions.cpp. |
| `vc_object_store/src/internal/ImmutableImport.hpp` | DECLARATION | private build-only access; InitialImport/ObjectCollection implementation. |
| `vc_object_store/src/internal/InitialEdges.cpp` | REVIEWED | InitialEdgeInserter writes UUID endpoints without local proxy rows; direct SQL isolated to immutable batch construction. |
| `vc_object_store/src/internal/InitialEdges.hpp` | DECLARATION | trusted immutable constructor; InitialEdges.cpp. |
| `vc_object_store/src/internal/InitialObjects.cpp` | REVIEWED | insertInitialObjects/insertInitialObjectRefs direct immutable object/geometry/chunk inserts; legitimate batch construction boundary, UUID sharding leaves reference semantics unchanged. |
| `vc_object_store/src/internal/InitialObjects.hpp` | DECLARATION | insertInitialObjects/Refs; InitialObjects.cpp; stale comment still mentions removed reference-endpoint insertion. |
| `vc_object_store/src/internal/Ownership.cpp` | REVIEWED | StoreOwnership OS file lock/lease is physical store ownership boundary, not logical object lookup. |
| `vc_object_store/src/internal/Ownership.hpp` | DECLARATION | physical writable file lease; Ownership.cpp. |
| `vc_object_store/src/internal/Publication.hpp` | REVIEWED | publishNewFile/StagedDatabase; no-clobber publication, owned staging cleanup; filesystem-only boundary. |
| `vc_object_store/src/internal/ReadRequest.hpp` | REVIEWED | ReadRequest/ReadBudget bounded query work accounting; no local/referenced object resolution. |
| `vc_object_store/src/internal/Schema.cpp` | REVIEWED | createSchema: physical tables/indexes/retention triggers at schema boundary; no consumer placement resolver. |
| `vc_object_store/src/internal/Schema.hpp` | DECLARATION | schema DDL implementation Schema.cpp. |
| `vc_object_store/src/internal/SpatialCache.hpp` | REVIEWED | ResidentSpatialIndex/SpatialReadCache are explicitly per-connection accelerators; UUID filters canonicalized; unsupported queries fall to shared SQL. |
| `vc_object_store/src/internal/SqlValues.hpp` | REVIEWED | writeSqlValuesShared/writeSqlValues batch SQL boundary with variable limits; no identity-resolution branch. |
| `vc_object_store/src/internal/Sqlite.cpp` | REVIEWED | openConnection central packed/plain dispatch; shared UUID/hash binding, statement pooling, savepoint/write-cache rollback. |
| `vc_object_store/src/internal/Sqlite.hpp` | DECLARATION | central Statement/Savepoint/openConnection/UUID codec primitives; Sqlite.cpp. |
| `vc_object_store/src/internal/TransformedLength.hpp` | REVIEWED | transformedLength caches geometry identity plus transform matrix inside read snapshot; uses shared GeometryReader. |
| `vc_object_store/src/linking/Detectors.hpp` | DECLARATION | shared StoredGeometry detector snapshot contract; linking implementations. |
| `vc_object_store/src/linking/FiberSurface.cpp` | REVIEWED | fiber-surface candidate loader uses StoredGeometry/shared views; numerical detector excluded. |
| `vc_object_store/src/linking/LineSurface.cpp` | REVIEWED | detectLinesToSurfaces uses StoredGeometry; stripCacheKey includes exact selected coordinates/scale and external normal source, not moving current heads. |
| `vc_object_store/src/linking/LineSurface.hpp` | DECLARATION | LineData carries selected ObjectRecord plus transformed points; LineSurface.cpp. |
| `vc_object_store/src/linking/LinkCandidates.cpp` | REVIEWED | selectLinkCandidates expands shared selections/neighbourhood and calls shared region queries; no physical store assumption. |
| `vc_object_store/src/linking/LinkObjects.cpp` | REVIEWED | prepareLinkObjects/detection and request construction retain selected versions and shared geometry sources; expected versions validated at submit. |
| `vc_object_store/src/linking/PointCollectionSurface.cpp` | REVIEWED | loadCollections/loadSurfaces use selected geometry and annotations; detection delegates shared publish path. |
| `vc_object_store/src/linking/StoredGeometry.cpp` | REVIEWED | snapshotOf/load/loadGeometrySnapshot use shared geometry resolution; select/load current paths share one ReadTxn, named overlap reads carry explicit geometry. validate compares numeric versions plus loaded geometry identity (see coverage note). |
| `vc_object_store/src/linking/StoredGeometry.hpp` | DECLARATION | StoredGeometry snapshot selection/publication guard; StoredGeometry.cpp; exact overlap cache gap TRANS-4. |
| `vc_object_store/src/linking/SurfaceOverlap.cpp` | FINDING | TRANS-4 OverlapMaterializer::surface cache omits metadata version/coordinate binding for pinned historical batches. |
| `vc_object_store/src/linking/TrackSurface.cpp` | REVIEWED | withTracks loads selected ObjectRecord through StoredGeometry and shared coordinate scales; detector arithmetic excluded. |
| `vc_object_store/src/linking/TrackTrack.cpp` | REVIEWED | detectTracksToTracks shared StoredGeometry + crossing kernel; source-ID grouping is track-format contract, not store row IDs. |
| `vc_object_store/src/migration/ConvertStore.cpp` | REVIEWED | convertStoreImpl/copyToCurrentSchema/numberSchema17/copyArtifacts: explicit isolated legacy schema conversion; rejects unmapped tables/programs, preserves retained versions, validates staged copy before in-place replacement. |
| `vc_object_store/src/server/Server.cpp` | REVIEWED | hosted request handlers select/read/submit via shared ObjectStore and wire codecs; raw lane ownership/network staging belongs at transport boundary. |

## Findings

All findings below are code-traced, not runtime reproduced. Existing AH/AI/AN and related AQ remain tracked in `coherent-store-api-cleanup.md`, not counted as new findings here.

### TRANS-1 — Project renaming can overwrite concurrent metadata with a stale record (P2)

`src/Project.cpp:102–116`, `setProjectName`, reads `projectRecord` in one session callback, modifies its name, then separately calls `currentRevision` before submitting that original record. If another writer changes metadata/tags between those reads, the submitted record is stale but the expected version is the newer one, so the optimistic check permits lost updates. This is independent of local/referenced placement. Preserve the original complete selected reference with its record and submit against that reference, or express the rename through the shared checked metadata-edit helper. Regression: interleave a metadata edit between project selection and rename; either preserve both edits or report conflict, never overwrite the new metadata.

### TRANS-2 — Exact export/restore wrappers cannot name a previous owner (P2)

`apps/vc_object_export.cpp:40–46`, `python/objects.cpp:1246–1265` (`export_geometry`) and `:1696–1708` (`restore_object`) construct historical references from `currentObjectRevision(...)->reference(version,generation)`. They provide no explicit owner/store reference input. After an overlay overrides an inherited object, a retained version owned by the base cannot be requested through these wrappers even though the underlying C++ export/restore APIs support complete `ObjectVersion` references. Numeric version/generation is not sufficient to recover the old owner. Keep numeric convenience forms if clearly documented, but add the same explicit-reference selection used by surface/artifact APIs and funnel selection through one shared helper. Regression: retain a base version, override it in an overlay, then export/restore the explicit base reference through CLI/Python; no current-owner reconstruction should occur.

### TRANS-3 — Two adapter boundaries omit shared UUID canonicalization (P2)

`src/imports/AfvSource.cpp:195` compares AFC provenance `afv_uuid` text directly with a canonicalized supplied AFV UUID. `src/formats/FiberJson.cpp:255–275` seeds export traversal from raw `roots` and compares them to canonical selected-revision keys. Valid uppercase UUID spellings therefore produce a false source mismatch or unknown export-root error, unlike normal object selection. Canonicalize these identity inputs once using `util::canonicalUuid`; do not add lowercase-only private parsers. Regressions: mixed-case matching AFV/AFC source UUID and mixed-case root UUID in an otherwise valid fiber network export.

### TRANS-4 — Pinned overlap cache conflates geometry identity with transformed-state identity (P2)

`src/linking/SurfaceOverlap.cpp:968–993`, `OverlapMaterializer::surface`, keys cached `LoadedSurface` by object UUID plus geometry revision, then returns cached transformed data before examining `edge.object(uuid)` metadata. `materializePinned` accepts multiple `OverlapEdgeAt` records, each with its own object metadata and geometry revision. Two exact selections can share geometry but differ in coordinate binding/scale; within one batch, the second selection then reuses the first selection's transformed grid. The `OverlapEdgeAt` descriptor (`include/vc_object_store/object_links/ObjectLinks.hpp`) has no complete endpoint version to distinguish those states. Carry complete selected endpoint references through this interface and shared transform resolution; cache by that selection plus target frame/transform, not bare geometry. Regression: materialize two pinned overlaps at metadata-only coordinate revisions with unchanged geometry, in both input orders; each must match independent materialization.

### TRANS-5 — Historical relationship transport drops owning-store identity (P2)

`src/History.cpp:169–196`, `relationship`, resolves the owning source but populates only relationship numeric version and generation, not `record.versionStoreUuid`; `RelationshipRevision` has no alternative owner field. `src/Wire.cpp:296–318` also serializes/deserializes its record without the owner-version codec used by current edge queries. `python/objects.cpp:58–67` and all `relationships_at_versions` wrappers expose that incomplete result. Consequently a relationship selected from a referenced base cannot supply the complete `RelationshipVersion` required by `python/ExtensionBindings.hpp:182` (`extension_rows_at_version`) without guessing its owner. Resolve and retain the complete relationship reference at the shared history boundary and preserve it in all wire/binding codecs. This complements existing AQ (history-off missing endpoint fields), rather than duplicating it: this finding is historical relationship **own** identity, including history-on reads. Regression: inherited retained relationship, metadata override and versioned relation-extension rows, round-tripped local → wire → Python; exact row selection must resolve the original owner.

## Supplemental cross-lane evidence

| File | Disposition | Why followed / evidence |
| --- | --- | --- |
| `vc_object_store/src/History.cpp` | FINDING | Followed relationship wire producers; relationship at 169–196 establishes TRANS-5. Core lane owns the complete file review. |
| `vc_object_store/src/internal/RecordJson.cpp` | REVIEWED | Followed TRANS-5 serialization: objectToJson/edgeToJson encode stored record fields, not complete numbered owner identity; query codecs explicitly append it. Core lane owns complete inventory coverage. |
| `vc_object_store/src/internal/RecordSql.cpp` | REVIEWED | Followed parent question about wholesale imported metadata; rejectInitialExtensionMetadata and storage-reference checks reject unmanaged copying of managed descriptors. Core lane owns full file. |

## Coverage limitations

Original lane inventory: **142/142 dispositioned; zero PENDING**. Three supplemental cross-lane evidence files are listed separately, not counted as newly unreviewed inventory. Review covered vcobj-facing symbols, transitive wrappers and retained-state/placement contracts; it was not a numerical geometry-kernel, memory-safety, networking-security or performance benchmark audit. No builds/tests or implementation edits were performed; finding regressions remain work to do.

Low-level exceptions are intentional: schema/conversion/immutable construction own physical SQL; collection/packed modules own physical routing and byte layout; source-format adapters own external JSON/TIFF/track/AFV reads; the reduced benchmark owns its independent comparison schema. These are not consumer fallback resolvers. Numerical CT/normal/crossing kernels, OS memory mapping and progress/profiling helpers are explicitly excluded above.

Investigated but **not substantiated as defects**: (1) ArtifactExport opens physical source snapshots without the runtime collection callback; retained metadata parent chains are rooted locally because first inherited overrides save a full local state, so the suspected foreign-parent replay failure does not follow. (2) Copying arbitrary source metadata through upsert does not silently accept managed geometry/extension descriptors: the shared insertion path rejects invalid managed metadata; no new dangling-reference finding was inferred from generic JSON copying alone. (3) Draft `keepDrafts` recovery is intentionally not persisted, per its public documented recovery policy. (4) Strip cache identity includes selected point bytes and coordinate scale, so a separate object-version key is not necessary for unchanged transformed inputs. (5) StoredGeometry uses numeric version guards in one current read/publication workflow; no reachable equal-version owner replacement during that workflow was established, so this alone is not labeled a defect. The exact mixed-metadata materialization cache issue is separately documented as TRANS-4.

## Implementation follow-up (2026-11-10)

The ledger above is the original audit evidence, not a claim that each finding remains unfixed. The implementation pass adds no schema or physical-placement alternatives:

- TRANS-2: export CLI accepts a complete owner reference (`--store`, `--generation`, `--version`); Python export/restore accept `reference`/`source_reference`. Both use shared selection. Default current export remains a current collection export; explicit versions use the selected owner's snapshot. Shared snapshot and source-path operations are forwarded through hosted readers rather than reimplemented by consumers.
- TRANS-3: AFV/AFC identity comparison and fiber-export roots use the existing UUID canonicalizer. Python consumers use its native `canonical_uuid` binding too.
- TRANS-4: pinned overlaps carry complete endpoint references instead of independently supplied metadata and geometry. A shared parser replaces duplicate Python parsers; materialization resolves the selected states and its cache includes full owner/version identity. Mixed metadata-only revisions are tested in both input orders.
- TRANS-5/AQ: relationship wire/bindings preserve the relationship owner and complete selected endpoint references. Extension readers can directly use returned relationship identity; graph consumers need not parse internal revision tokens or reread current endpoints.
- AI: transform registrations, their selected revisions, and route choices are fetched in one server read. The attachment's existing invalidation/retry mechanism remains, tested against a delayed response from this new coherent endpoint and concurrent atomic registration updates.
- CPP-2 transport contract: region runs retain their selected object reference and actual display affine, so picking and inverse editing use the geometry that was drawn.
- AH: one `resolvedExtensionTable` validates all visible definitions for registration and both current/exact reads. Conflicting new registration fails before creating a local schema. Regression fixtures also manufacture an already-invalid physical overlay to verify old incompatible collections are rejected by current and exact reads.

Final validation through the approved argument-free PR runner: 52 native tests passed (optional SSH skipped), 201 Python store tests passed, 121 Spiral tests passed (one skipped), and 13 Lasagna tests passed. This includes the coherent-transform, region-mapping, source canonicalization and exact-reference tests. Export retains its prior RuntimeError and message for legacy numeric selection failures. The transform-cache proxy now intercepts the coherent endpoint; the listing stress test waits for actual bounded writer progress instead of assuming three creations within one second. The extension fixture checks early rejection and separately manufactures an invalid legacy collection for read guards. See the consolidated inventory for commands, final log, consumer checks and validation limits.
