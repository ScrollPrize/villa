# Full vcobj review: transport-storage

Baseline: `3df5b7783`. Fresh full-scope review of all assigned relevant files, not only the recent diff. Previous findings R2-1–7 remain in [the combined findings](coherent-store-review-3df5b7783.md); corroborations are not counted again. Every assigned path requires a current disposition and inspected symbols/evidence. No finding quota. Follow additional callers and append rows if needed; report why pure numerical/build-only paths are excluded. Review only: no production fixes or commits.

## Coverage (146 assigned files)

| File | Disposition | Inspected behavior / evidence |
| --- | --- | --- |
| `vc_object_store/CMakeLists.txt` | BUILD ONLY | All target/feature/dependency/install branches read; no runtime resolver or mutable store behavior. |
| `vc_object_store/apps/support/ProcessMetrics.hpp` | DIAGNOSTIC ONLY | Platform-guarded RSS bytes (macOS native bytes/Linux KiB conversion); no store handling. |
| `vc_object_store/apps/support/SqliteMutexProfile.hpp` | DIAGNOSTIC ONLY | Opt-in preconnection mutex instrumentation forwards original SQLite locks; per-thread counters/samples; no record or storage resolver. |
| `vc_object_store/apps/vc_object_convert.cpp` | REVIEWED | Offline converter argument checks and output separation; shared converter/policy entry points only. |
| `vc_object_store/apps/vc_object_export.cpp` | REVIEWED | Selected owner/generation/version built via StoreReader and passed unchanged to shared export; no local SQL. |
| `vc_object_store/apps/vc_object_import_afv.cpp` | REVIEWED | Readers share ParallelImportStream and InitialImport; dependency pins, typed source extensions, counters before publication and optional readback validated. |
| `vc_object_store/apps/vc_object_ingest.cpp` | REVIEWED | Admission frame checks; shared ingest/channel update APIs and partial failure status; no placement-specific logic. |
| `vc_object_store/apps/vc_object_links.cpp` | REVIEWED | Parameter dispatch to shared detectors; no local store SQL or identity synthesis. |
| `vc_object_store/apps/vc_object_server.cpp` | REVIEWED | Server options, service ownership and loopback/security contract; no independent record handling. |
| `vc_object_store/benchmarks/prepared_import.cpp` | REVIEWED DIAGNOSTIC | Full input preparation, matched prototype/native imports, shared ParallelImportStream/InitialImport, UUID shard routing, timing/readback/inventory/cleanup reviewed. Direct SQL explicitly restricted to prototype and diagnostic physical-table inventory, not production collection resolution. |
| `vc_object_store/include/vc_object_store/CtSheetNormal.hpp` | DECLARATION | CT/Zarr/pooled normal sampler contracts, invalid samples versus source failures, explicit scale and immutable source inputs; no record/version resolution. |
| `vc_object_store/include/vc_object_store/MemMap.hpp` | REVIEWED | Move-only read-only mapping ownership; views retain mapping until close/destruction; no independent store locator interpretation. |
| `vc_object_store/include/vc_object_store/PackedTrackStore.hpp` | DECLARATION | Explicit source-path resolver and immutable mapped VCTRK01 arrays; source ID is external track identity, not vcobj row identity. |
| `vc_object_store/include/vc_object_store/Progress.hpp` | REVIEWED | Synchronous callback lifetime and owning nested labels; threading responsibility documented; no store selection or payload access. |
| `vc_object_store/include/vc_object_store/TerminalProgress.hpp` | REVIEWED | Heartbeat/update/stop synchronization and callback lifetime reviewed; terminal presentation only, no placement/history assumptions. |
| `vc_object_store/include/vc_object_store/Tiff.hpp` | DECLARATION | Raw sample and image serialization APIs use explicit file inputs; no vcobj object or selected-version resolution. |
| `vc_object_store/include/vc_object_store/TrackCrossings.hpp` | DECLARATION | Array-only crossing kernel contracts, prevalidated source-index inputs and CSR outputs; store identity remains with caller. |
| `vc_object_store/include/vc_object_store/Uuid.hpp` | REVIEWED | ASCII canonical UUID validation, random v4 generation and v5 declarations; shared logical-key normalization, not placement-specific IDs. |
| `vc_object_store/include/vc_object_store/client/HostedStore.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/migration/ConvertStore.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/object_links/LinkObjects.hpp` | REVIEWED | Ordered prepare/checked-commit contract, per-store link marker, parsers and registered preparer. No alternate store resolver; numeric marker guard caveat considered with implementation. |
| `vc_object_store/include/vc_object_store/object_links/ObjectLinks.hpp` | REVIEWED | Detector selections/scopes, candidate publication, scalar CoordinateParams contract, immutable-voxel strip-cache contract, and complete ObjectVersion pinned overlap inputs reviewed. Scalar-frame detector support is explicitly limited; registration-graph transforms are not promised here. |
| `vc_object_store/include/vc_object_store/objects/AfvSource.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/Artifacts.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/Extensions.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/FiberGraphSolveExtension.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/FiberImportExtension.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/FiberJson.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/GeometryExport.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/GeometryIngest.hpp` | REVIEWED | Admission, repair and checked-request contracts; unsupported source relationship targeting explicitly documented. |
| `vc_object_store/include/vc_object_store/objects/InitialImport.hpp` | REVIEWED | Trusted immutable import and borrowed-record lifetime API; shard and finalization callbacks match implementation. |
| `vc_object_store/include/vc_object_store/objects/ObjectCollection.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/ObjectStoreSession.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/PackedStore.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/Project.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/StoreConversion.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/UmbilicusFile.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/include/vc_object_store/objects/imports/ParallelImportStream.hpp` | REVIEWED | Independent task stealing and bounded per-shard queues; object/edge completion counters, cancel/join ordering reviewed. |
| `vc_object_store/include/vc_object_store/objects/imports/ShardQueue.hpp` | REVIEWED | Batch queue memory bounds, cancellation wakeups, closure and first-error propagation reviewed. |
| `vc_object_store/include/vc_object_store/server/Server.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/python/AfvBindings.hpp` | REVIEWED | AfvSource typed iteration forwards native source decoder; no store owner/reference interpretation. |
| `vc_object_store/python/CMakeLists.txt` | BUILD ONLY | Binding target links common geometry/linking APIs; package copy/install list; no runtime interpretation. |
| `vc_object_store/python/ExtensionBindings.hpp` | REVIEWED | Typed rows and binary UUID conversion, shared schemas and complete object/relationship exact references; all mutations and scans delegate shared extension APIs. |
| `vc_object_store/python/GeometryRegionBindings.hpp` | REVIEWED | Copies mutable query options before GIL release, retains store lease, delegates shared region query; preserves completion and selected run identity. |
| `vc_object_store/python/ObjectReadBindings.hpp` | REVIEWED | Graph snapshot transaction/array ownership, selected object and relationship records, pinned geometry revision batches/ranges; thread confinement and closure guards. |
| `vc_object_store/python/SessionBindings.hpp` | REVIEWED | Attachment lifecycle and event/copy/import forwarding; storage semantics remain native session responsibilities. |
| `vc_object_store/python/StorageBindings.hpp` | REVIEWED | Shared record detail projection, anchor validation, geometry codecs, withStore/read/write leases and transaction lifecycle. BIND-F1 is a caller bypass of the provided lifetime guard. |
| `vc_object_store/python/TypedGeometryBindings.hpp` | REVIEWED | Typed input validation and array ownership; view/count/transform calls forward shared geometry API; valid empty control arrays accepted. |
| `vc_object_store/python/objects.cpp` | FINDING | FINDING BIND-F1: submit/start_history_generation bypass common store lease; progress callback closes store and crashes submit (runtime). Remaining session/local/hosted wrappers, selection, artifacts, checked writes, graph reads and complete refs freshly reviewed. |
| `vc_object_store/python/vc_object_store/__init__.py` | REVIEWED | Native binding export only. |
| `vc_object_store/python/vc_object_store/documents.py` | REVIEWED | Document and point-collection wrappers delegate common attachment/session APIs; no local physical layout handling. |
| `vc_object_store/python/vc_object_store/locations.py` | REVIEWED | Explicit local/hosted facade preserves shared API routing; selects exact refs via session. Supported method subset and hosted read_only limitations documented. |
| `vc_object_store/python/vc_object_store/surfaces.py` | REVIEWED | Exact reference selectors and surface/channel validity delegated native; selected record retained with geometry in SurfaceGrid; creation delegates common API. |
| `vc_object_store/scripts/import_afv.py` | REVIEWED | Read-only AfvSource; canonical source UUID5 naming, valid empty-control geometry, typed source extension, checked transactions and sibling staged publication. Mapping mirrors native AfvRecords.hpp; no confirmed behavioral divergence. |
| `vc_object_store/scripts/run_cpp_tests.sh` | TEST WORKFLOW ONLY | Argument-free build/test driver, scoped temporary fixtures with cleanup traps, persistent output outside tmp; inspected only, not executed. |
| `vc_object_store/src/Artifacts.cpp` | REVIEWED | Content addressed shared closure routing, hash-checked copying and retained historical manifests; publication and deletion cleanup inspected. |
| `vc_object_store/src/CMakeLists.txt` | BUILD ONLY | Library boundaries and source registration inspected; shared geometry/store/native adapters and offline converter kept separate; no consumer SQL. |
| `vc_object_store/src/Commands.cpp` | REVIEWED | Complete command variant encode/decode and prepare/replay inspected; full expected refs preserved, mutation delegates to common transaction APIs. |
| `vc_object_store/src/CtSheetNormal.cpp` | REVIEWED | Full CT estimator cache, nearest Lasagna decode and pooled source reader inspected; vc_reader handles Zarr, shared local decode handles both Lasagna backends. Numerical/input-format helper, no vcobj owner/version lookup; malformed pack fuzzing not performed. |
| `vc_object_store/src/DraftCheckpoints.cpp` | REVIEWED | Complete ObjectVersion persisted in drafts, atomic manifest-last snapshot, owned stale file cleanup and request payload recovery. |
| `vc_object_store/src/DraftCheckpoints.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/ExtensionHistory.cpp` | REVIEWED | Shared payload manifests, exact ObjectVersion/RelationshipVersion reads, no-op savepoint rollback, retained schema identity and override behavior. |
| `vc_object_store/src/Extensions.cpp` | REVIEWED | Schema registration/compatibility, typed row writes, current/retained owner snapshots and referenced source precedence; immutable and manual versioned paths inspected. |
| `vc_object_store/src/FetchQueue.cpp` | REVIEWED | Per-object background fetch work lifecycle, bounded request lifetime, explicit retry/refetch, cancellation/join behavior; no direct store SQL. |
| `vc_object_store/src/FetchQueue.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/HostedSession.cpp` | FINDING HOSTED-F1 | Complete hosted request/cache/poll implementation read. Raw caller UUID head key differs from canonical invalidation; runtime uppercase-read/canonical-edit probe retains stale v1 versus v2. |
| `vc_object_store/src/HostedSession.hpp` | DECLARATION | Hosted attachment identity, cache/fetch lifetime and resolver API read alongside implementation. |
| `vc_object_store/src/InitialImport.cpp` | REVIEWED | Private shard creation, disjoint numeric ID reservations, UUID routing, producer lifetime and aggregate checks before publication; packed/single/directory paths inspected. |
| `vc_object_store/src/MemMap.cpp` | REVIEWED | POSIX/Windows mapping setup, descriptor closure, empty-file rejection and RAII release; purely physical byte mapping, not a competing store resolver. |
| `vc_object_store/src/ObjectCollection.cpp` | FINDING R2-4 | Complete traversal, identity routing, pins, trusted import, copy and mutable overlay APIs read. Existing R2 copy-relative-locator finding corroborated, not additional. |
| `vc_object_store/src/ObjectStoreSession.cpp` | REVIEWED | Complete local/hosted session attach/read/submit/draft/repin/artifact logic read; readStore exposes hosted cache issue in HOSTED-F1. |
| `vc_object_store/src/PackedCollection.cpp` | FINDING R2-4 | Complete pack/copy closure and artifact transfer read. Existing R2 relative source-locator omission applies to packed destination. |
| `vc_object_store/src/PackedStore.cpp` | REVIEWED | Entire VFS builder/read/write/cache/flush/locking/truncate/journal/finalization and compact export read; immutable publication, member pins and sidecar export checked. |
| `vc_object_store/src/PackedTrackStore.cpp` | REVIEWED | DBM backing fingerprint resolution, direct packed-directory mode, header/endian/size checks, offsets and source-ID ordering; explicit source-format boundary independent of selected vcobj owner. |
| `vc_object_store/src/Project.cpp` | REVIEWED | Current-object references guard edits, project closure copy uses shared collection helper; history-optional ordering and view/resource CRUD. |
| `vc_object_store/src/Provisioning.cpp` | REVIEWED | Independent record version versus commit sequence, history/spatial policies, physical UUID/table directory registration and copy-fork identity updates. |
| `vc_object_store/src/StoreConversion.cpp` | REVIEWED | History policy staging, trigger suspension/restoration, borrowed geometry retention, extension chunk retention and atomic replacement fully read. |
| `vc_object_store/src/Tiff.cpp` | REVIEWED | Full strip/tile read-write, native sample conversion and decode validation inspected; physical image codec only. GeometryOutput supplies resolved paths; libtiff diagnostic-handler concurrency not stress-tested. |
| `vc_object_store/src/TrackCrossings.cpp` | REVIEWED | Entire validation, deterministic radix ordering, exact-voxel/tangent filtering, pair clustering/arclength and symmetric CSR construction read; array kernel has no record ownership/history or SQL path. |
| `vc_object_store/src/UmbilicusFile.cpp` | REVIEWED | Entire importer/frame validation/scale selection read; store mutation delegates to shared setVolumeUmbilicusRequest. |
| `vc_object_store/src/Uuid.cpp` | REVIEWED | Namespaced v5 SHA-1 implementation, thread-local EVP state, shared namespace parsing and fixed default namespace; no alternative logical-key scheme. |
| `vc_object_store/src/Wire.cpp` | REVIEWED | All wire structures/read queries/errors/identity/geometry conversions read; complete versions retained and unsupported query features rejected explicitly. |
| `vc_object_store/src/catalog/Catalog.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/catalog/CatalogStore.cpp` | REVIEWED | Catalog placeholder/fetch/restore/refresh paths read; source fetch uses shared codec, numeric fetch guards remain API limitation without confirmed owner collision. |
| `vc_object_store/src/client/HostedStore.cpp` | REVIEWED | Entire client: exact reads, paged/feed catch-up queries, transforms, byte ranges, staged request retries and artifact hash verification read. |
| `vc_object_store/src/formats/FiberInput.cpp` | REVIEWED | Canonical source UUIDs and recursive branch peer loading; explicit dangling repair policy and duplicate detection. |
| `vc_object_store/src/formats/FiberInput.hpp` | REVIEWED | Native coordinate array validation used by the shared JSON codec. |
| `vc_object_store/src/formats/FiberJson.cpp` | REVIEWED | Versioned JSON decoding, exact revision export via StoreReader, relationship selection and canonical root traversal. |
| `vc_object_store/src/formats/FiberMetadata.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/formats/GeometryInput.cpp` | REVIEWED | Common canonical UUID/source identity derivation and absolute locator construction. |
| `vc_object_store/src/formats/GeometryInput.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/formats/GeometryOutput.cpp` | REVIEWED | Geometry snapshot only; native export preserves revision identity, structural IDs, channels and transfer attributes. |
| `vc_object_store/src/formats/GeometryOutput.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/formats/PointCollectionInput.cpp` | REVIEWED | Document-scoped IDs/duplicates, explicit renumber admission; shared coordinate merge and structural geometry encoding. |
| `vc_object_store/src/formats/RemoteSource.cpp` | REVIEWED | Shared HTTP/catalog URL resolver; explicit five-file remote surface download scope, no object-store placement logic. |
| `vc_object_store/src/formats/RemoteSource.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/formats/SurfaceInput.cpp` | REVIEWED | Native surface/shared TIFF reader, coordinate frame/validity/auxiliary channels and source change fingerprint. |
| `vc_object_store/src/formats/SurfaceInput.hpp` | REVIEWED | Shared source completeness guard rejects unreadable TIFFs and interrupted saves. |
| `vc_object_store/src/formats/TrackInput.cpp` | REVIEWED | Packed source row to typed track geometry with canonical axis conversion and source metadata. |
| `vc_object_store/src/formats/TrackInput.hpp` | REVIEWED | Source-ID canonicalization and deterministic source-polyline measurements; no store lookup. |
| `vc_object_store/src/formats/UmbilicusInput.cpp` | REVIEWED | Shared typed encoder and coordinate declaration merge; source UUID and absolute locator. |
| `vc_object_store/src/imports/AfvRecords.hpp` | REVIEWED | Deterministic object UUID construction and line-point-only geometry; constraint extensions retain canonical orientation. |
| `vc_object_store/src/imports/AfvSource.cpp` | REVIEWED | Readonly source snapshots, typed bounds/geometry validation, workload partitions, deterministic UUIDs and AFC orientation/measurement import. |
| `vc_object_store/src/imports/FiberGraphSolveExtension.cpp` | REVIEWED | Typed stage/run/input/result schema, compound primary keys, probability/UUID/vector validity; based-on input pin recording. |
| `vc_object_store/src/imports/FiberImportExtension.cpp` | REVIEWED | Explicit concise provisioning, typed measurement columns and source orientation; unknown diagnostic extras retained as row payload. |
| `vc_object_store/src/imports/SourceIdentityIndex.cpp` | REVIEWED | Lazy point-identity accelerator invalidation and publication refresh; visible rows resolve across stores, duplicate IDs rejected. |
| `vc_object_store/src/imports/SourceIdentityIndex.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/imports/SourceRelationships.cpp` | REVIEWED | Canonical reciprocal branch reconciliation, shared geometry/endpoint insertion, indexed source point/collection links. |
| `vc_object_store/src/imports/SourceRelationships.hpp` | DECLARATION | Declaration and policy inspected alongside corresponding implementation; no placement-specific alternate path. |
| `vc_object_store/src/internal/Artifacts.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/internal/CollectionReads.hpp` | REVIEWED | Read lease routing helpers and provider callback lifetime inspected alongside ObjectCollection; physical routing stays in shared storage layer. |
| `vc_object_store/src/internal/DatabaseCopy.hpp` | REVIEWED | Backup helper and WAL snapshot lifecycle inspected; staging validation delegates to SQLite; known relocation issue is caller-owned logical-source rebasing. |
| `vc_object_store/src/internal/Digest.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/internal/ExtensionHistory.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/internal/Extensions.hpp` | REVIEWED | Complete declarations and documented policy read alongside implementation; exact/current identity, closure, retention and mutable lifetime contracts inspected. |
| `vc_object_store/src/internal/ImmutableImport.hpp` | DECLARATION | Private initial create/seal/manifest access; no writable reopening of published immutable stores. |
| `vc_object_store/src/internal/InitialEdges.cpp` | REVIEWED | InitialEdgeInserter reserves edge IDs, batches shared normalized edge rows, defers only nonunique lookup indexes and marks failed streams unusable. |
| `vc_object_store/src/internal/InitialEdges.hpp` | DECLARATION | Trusted initial edge construction explicitly excludes endpoint rediscovery; caller owns transaction. |
| `vc_object_store/src/internal/InitialObjects.cpp` | REVIEWED | insertInitialObjectRefs reserves IDs, rejects reserved extension descriptors, routes geometry through shared initial preparation; trusted immutable append policy is explicit. |
| `vc_object_store/src/internal/InitialObjects.hpp` | DECLARATION | Initial object append/savepoint contract; implementation InitialObjects.cpp. |
| `vc_object_store/src/internal/Ownership.cpp` | REVIEWED | Canonical-path service-exclusive versus standalone-shared ownership guard, platform-gated implementation; distributed/hard-link bypass not claimed. |
| `vc_object_store/src/internal/Ownership.hpp` | DECLARATION | Host-local lifetime ownership policy; implementation Ownership.cpp. |
| `vc_object_store/src/internal/Publication.hpp` | REVIEWED | publishNewFile no-replace hard-link publication; staging witness and sidecar cleanup retained on failures. |
| `vc_object_store/src/internal/ReadRequest.hpp` | REVIEWED | Scoped per-thread request budget/cancellation and metadata/geometry work accounting; no ownership inference or retained-state selection. |
| `vc_object_store/src/internal/Schema.cpp` | REVIEWED | Full schema construction, binary identity fields, state enums, feature manifest, conditional history/materialization tables and immutable-import index suppression inspected; history-off materialization mismatch already recorded as FULL-CORE-1, not recounted. |
| `vc_object_store/src/internal/Schema.hpp` | DECLARATION | Shared schema/chunk-table constructor declarations only. |
| `vc_object_store/src/internal/SpatialCache.hpp` | FINDING | New reproduced cache/SQL lifecycle divergence: load stores Deleted with Retired, then includeRetired permits deleted descriptors (lines35–44,78). Cached include_retired query returns1 deleted region versus0 with cache disabled. Followed Queries.cpp spatialRegions and Records.cpp deleteObject; parent supplied detailed finding. |
| `vc_object_store/src/internal/SqlValues.hpp` | REVIEWED | SQLite parameter-limited batched statements and borrowed binding lifetimes; transaction ownership explicitly stays with caller, no partial endpoint or placement resolver. |
| `vc_object_store/src/internal/Sqlite.cpp` | REVIEWED | Entire physical connection/VFS dispatch, binary UUID/hash conversions, pooled statements, read budgets and nested savepoint rollback inspected; physical DB access remains at shared storage boundary. |
| `vc_object_store/src/internal/Sqlite.hpp` | REVIEWED | Shared Statement/Savepoint ownership, bindings, connection declarations and canonical binary key helpers; no consumer-specific row IDs. |
| `vc_object_store/src/internal/TransformedLength.hpp` | REVIEWED | Exact supplied geometry revision plus coordinate mapping determine length; uses actual array shape and bounded revision/mapping-keyed cache, no current-object substitution or stale ingest-length fallback. |
| `vc_object_store/src/linking/Detectors.hpp` | REVIEWED | Internal detector declarations all consume shared StoredGeometry; no independent storage handling. |
| `vc_object_store/src/linking/FiberSurface.cpp` | REVIEWED | loadFiber/withFibers, endpoint-control span, empty-controls line handling, incremental neighborhood selection, cache selection and public wrappers. Geometry comes from native snapshot; no source-file reimport or physical owner assumption. |
| `vc_object_store/src/linking/LineSurface.cpp` | REVIEWED | SheetNormalSource, stripCacheKey/cachedStrip, detector loading, candidate anchors and publish wrapper; full exact geometric inputs and normal-source identity key disposable cache, atomic replacement cleans failed temporary. Native geometry through StoredGeometry; numerical strip/contact/growth code read, no extra store resolver. |
| `vc_object_store/src/linking/LineSurface.hpp` | REVIEWED | LineData retains selected record and source scale; loader/incremental interfaces feed common detector rather than private store reads. |
| `vc_object_store/src/linking/LinkCandidates.cpp` | REVIEWED | canonical edgeIdentity, scoped pending reconciliation, reviewed-edge preservation, shared edge geometry validation and commitStoredCandidates/publishDetection. Native read snapshot validates before checked publication; no direct physical-database SQL. |
| `vc_object_store/src/linking/LinkObjects.cpp` | REVIEWED | Strict plan parsers, detector dispatch, local/service normal/cache-path policy, prepareLinkObjects snapshot closure and atomic marker publication. Numeric expectedObjects guard considered alongside StoredGeometry revision validation; no reachable ordinary same-counter owner substitution established. |
| `vc_object_store/src/linking/PointCollectionSurface.cpp` | REVIEWED | Role filtering, selected geometry plus revision annotations, stable point IDs/source anchors, incremental scopes and publication. Uses shared StoredGeometry and spatial APIs; no owner/path resolver. Scalar frame limitation declared by CoordinateParams. |
| `vc_object_store/src/linking/StoredGeometry.cpp` | REVIEWED | select/load/snapshotOf/validate, exact named revision ownership, optional validity channel, point annotations, coordinate scales and spatial broad phase. Native reads hold one transaction; arbitrary-reader exact use supplies named revisions. Numeric metadata guard residual risk not counted without reachable equal-counter owner replacement; registered-transform support intentionally outside scalar detector contract. |
| `vc_object_store/src/linking/StoredGeometry.hpp` | REVIEWED | Shared detector snapshot/loader, loaded geometry revision tracking, numeric expectations, annotation and bounds interface. No consumer-specific backing SQL; exact revision helper used by pinned overlaps. |
| `vc_object_store/src/linking/SurfaceOverlap.cpp` | REVIEWED | Detection selection/load/publish, current-edge stale check, materializePinned objectsAtVersions and named geometry, full identity materializer cache, shared agreement traversal. Current mode holds native ReadTxn; selected mode never requires current live endpoint. Scalar geometryScaleToFrame limitation explicit; numerical fit/search sections read. |
| `vc_object_store/src/linking/TrackSurface.cpp` | REVIEWED | Native snapshot track selection/load and type/domain validation feed common line-surface detector; failure/cancellation preserves existing edges, no independent backing data lookup. |
| `vc_object_store/src/linking/TrackTrack.cpp` | REVIEWED | Selected dataset identity/family plus stored track arrays, crossing kernel adaptation, local point/support anchors, source-arclength metadata and checked publication. datasetRef is grouping only, not a path; public geometry loader resolves placement. |
| `vc_object_store/src/migration/ConvertStore.cpp` | REVIEWED | All upstream16–20 mapping, archive reconstruction, tombstone recovery, artifact validation and explicit rejection paths read. Separate-destination relative locators share R2-4 omission (code trace). |
| `vc_object_store/src/server/Server.cpp` | REVIEWED | All handlers/routing and lifecycle read; one shared read transaction per response; bounded upload/read/write lanes and prepared request replay. |

## Additional findings

### HOSTED-F1 / R2-20: current-head cache ignores canonical UUID identity (P2)

`src/HostedSession.cpp:25,358–362,488–497,755–757`: the cache keys a current head by the caller's raw UUID spelling, while receipt/feed invalidation supplies canonical lowercase UUIDs. Public `ObjectStoreSession::readStore` (`src/ObjectStoreSession.cpp:520–526`) exposes this path. Read uppercase UUID at version 1, submit a normal canonical-UUID metadata edit, then read both spellings: uppercase still returns version 1; lowercase returns version 2. Current geometry also follows the stale cached head. Canonicalize once in the shared hosted cache boundary, including query/requery keys, rather than fixing individual consumers. Missing regression: valid mixed-case UUID through public readStore, canonical edit/feed invalidation, then current metadata/geometry lookup. Pre-existing missed implementation (`git blame`: `7e6b82164d`), not a regression in the latest delta.

### SUPPORT-F1 / R2-21: cached spatial lifecycle differs from SQL (P2)

Core agent independently inspected `src/internal/SpatialCache.hpp:35–44,78` and reproduced: load records every `state<>0` UUID in one retired set; `includeRetired=true` then admits Deleted as well as Retired descriptors. A deleted two-point umbilicus remains in cached spatial results (1 region) while the cache-disabled SQL path returns 0; default includeRetired=false returns 0. Repair the shared cache lifecycle predicate to match the store selector and test Active/Retired/Deleted under both paths. Full reproduction/evidence is recorded with R2-21 in the coordinator's report.

### Other assigned-lane findings and corroborations

- Root's independent Python review reproduced BIND-F1 / R2-19: submit bypasses the shared handle lease; progress callback close succeeds and the child process segfaults. Its detailed reproduction belongs to the Python/coordinator report, not a second finding.
- Existing R2-4 relative-locator relocation failure was confirmed for copy and pack. The separate-output upstream converter also copies relative storage_ref/dataset_ref unchanged (code trace only), sharing that repair boundary.
- Numeric-only artifact publication/fetch/detector guards were inspected but no reachable same-counter owner replacement was established; do not count a hypothetical failure. They remain narrower public contracts than complete-reference editing.
- Documentation-only mismatch: StoreConversion.hpp says history transitions issue fresh geometry tokens, while current changeHistoryMode preserves current geometry token identities. Correct that comment when updating the related code.

## Evidence and limits

All 146 assigned rows now have fresh inspected dispositions, with assistance explicitly reflected in the Python/build, numerical/SQL helper and linking rows. No PENDING paths remain. Review covered relevant code in every assigned file, not just changed lines; no new production code or tests were edited. Numerical algorithms, build wiring and diagnostic benchmarks were read for their storage/API boundaries and marked accordingly, not asserted numerically correct by this audit. No finding quota was used.

Executed hosted probe: `local/review-hosted-canonical.cpp` against the existing native build (no rebuild/install). The fixture uses `TemporaryDirectory` and a caught exception boundary; service/session shut down and fixture files are automatically removed. Command: `/home/hendrik/villa3/local/review-hosted-canonical` (loopback permission granted). Output, exit 0:

```text
before upper=1 lower=1
after upper=1 lower=2
```

Compile command:

```sh
c++ -std=c++20 -O0 -g -Ivc_object_store/include -Ivc_object_store/tests/cpp -Ivc_reader/include -Ilocal/vcobj-pr-tests/build/src/reader/include local/review-hosted-canonical.cpp -Wl,--start-group local/vcobj-pr-tests/build/src/libvc_object_server.a local/vcobj-pr-tests/build/src/libvc_object_geometry.a local/vcobj-pr-tests/build/src/libvc_object_links.a local/vcobj-pr-tests/build/src/libvc_track_crossings.a local/vcobj-pr-tests/build/src/libvc_resident_geometry.a local/vcobj-pr-tests/build/src/libvc_objects.a local/vcobj-pr-tests/build/src/libvc_object_formats.a local/vcobj-pr-tests/build/src/libvc_geometry.a local/vcobj-pr-tests/build/src/reader/libvc_reader_catalog.a local/vcobj-pr-tests/build/src/reader/libvc_reader_sftp.a -Wl,--end-group -Llocal/vcobj-pr-tests/build/src/reader -Wl,-rpath,/home/hendrik/villa3/local/vcobj-pr-tests/build/src/reader -lvc_reader_native -lsqlite3 -lcurl -lz -lcrypto -lzstd -lpthread -lopencv_core -ltiff -lblosc -llz4 -fopenmp -o local/review-hosted-canonical
```

Earlier copy/pack reproduction and its exact compile/run command remain in `local/review-transport-round2.md`; it used existing native libraries and automatically cleaned fixtures. This is corroborating R2-4, not fresh case count. All other dispositions are code traces unless the row/report explicitly says runtime-proven. Existing green suites do not substitute for the missing new regression tests. Physical SQLite access inside the responsible storage/conversion layer is legitimate; this review flags consumer bypasses and divergent semantics, not SQL itself.
