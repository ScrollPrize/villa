# R5 physical/storage coverage

Baseline: `bea32d5d8`. This lane covers the 118 native physical/storage paths of the R4 inventory. Root separately owns the 15 CLI, benchmark and packaging paths appended to that inventory. No new tracked source paths appeared between R4's final fix commit and this frozen baseline. Caller closure was cross-checked against bindings, VC3D, Spiral and public native entry points; those consumers are accounted for in their respective lanes, with specific cross-layer proof noted below.

Inspected store-facing identity, routing, layout, publication, input validation, transaction and cache boundaries. The file dispositions identify the paths examined rather than claiming proof for all inputs or interleavings. No production or test code was changed, and no builds, installs or runtime probes were run in this lane. Detector numerical bodies (crossing radix/event computations, overlap fitting, normal estimation, strip marching/matching, projection/scoring) are explicitly outside this storage review.

Static candidates: S-001 (ownership alias), S-002 (registered remote identity), S-003 (cache publication), S-004 (immutable single-shard copy), S-006 (caller-owned candidate batch), S-008 (project resource sort), S-009 (resource identity), S-010 (hosted listing normalization). Adjacent root owns S-005 and S-007. C-003 covers single-object copy extension safety. These are candidates, not independently confirmed results.

Scope cross-check: source UUID and source position provisioning stay distinct from data rows; trusted initial edges intentionally do not rediscover remote endpoints; remote TIFF coordinate/mask-only semantics are documented; cached `null` payloads are not published; the SQL idle pool is an actual multimap; the installed zstd target rejected in R4 is not reopened. The R4 savepoint, source-point range, umbilicus dimension, AFV report and conditional Boost fixes were considered.

| File | Reviewed path / disposition |
| --- | --- |
| `vc_object_store/include/vc_object_store/CtSheetNormal.hpp` | Source reader contracts, missing data versus invalid samples, scale. |
| `vc_object_store/include/vc_object_store/MemMap.hpp` | Readonly mappings and move/close lifetime. |
| `vc_object_store/include/vc_object_store/PackedTrackStore.hpp` | Borrowed spans and immutable track file lifetime. |
| `vc_object_store/include/vc_object_store/Progress.hpp` | Shared synchronous progress callback contract. |
| `vc_object_store/include/vc_object_store/TerminalProgress.hpp` | Heartbeat and callback lock/lifetime. |
| `vc_object_store/include/vc_object_store/Tiff.hpp` | Exact TIFF layout and complete file probe contract. |
| `vc_object_store/include/vc_object_store/TrackCrossings.hpp` | Crossing input shape and ordering, not numerical kernels. |
| `vc_object_store/include/vc_object_store/Uuid.hpp` | Canonical, random and namespaced UUID interfaces. |
| `vc_object_store/include/vc_object_store/client/HostedStore.hpp` | Remote selection, exact reads, range transport and submits; S-010. |
| `vc_object_store/include/vc_object_store/migration/ConvertStore.hpp` | Explicit upstream conversion and in-place entry point; S-005 adjacent. |
| `vc_object_store/include/vc_object_store/object_links/LinkObjects.hpp` | Prepared detection transport and atomic commit interface. |
| `vc_object_store/include/vc_object_store/object_links/ObjectLinks.hpp` | Caller-owned and standalone candidate batches; S-006. |
| `vc_object_store/include/vc_object_store/objects/AfvSource.hpp` | Native AFV/AFC bounded snapshot, range and mapping contracts. |
| `vc_object_store/include/vc_object_store/objects/Artifacts.hpp` | Content-addressed artifact bytes and manifest semantics. |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | Placeholder identity, lazy fetch and published-cache contract; S-003. |
| `vc_object_store/include/vc_object_store/objects/FiberGraphSolveExtension.hpp` | Typed solver tables and run input references. |
| `vc_object_store/include/vc_object_store/objects/FiberImportExtension.hpp` | Typed source identity, provenance and raw endpoint order. |
| `vc_object_store/include/vc_object_store/objects/FiberJson.hpp` | Pinned version/network export contract. |
| `vc_object_store/include/vc_object_store/objects/GeometryExport.hpp` | Exact/current native export and byte budget. |
| `vc_object_store/include/vc_object_store/objects/GeometryIngest.hpp` | Registered versus direct source identity and failure reporting; S-002. |
| `vc_object_store/include/vc_object_store/objects/InitialImport.hpp` | Trusted immutable import phases, reservations and publish; S-004. |
| `vc_object_store/include/vc_object_store/objects/ObjectCollection.hpp` | Physical member versus logical resolution, copy contract; S-004. |
| `vc_object_store/include/vc_object_store/objects/ObjectCopy.hpp` | Selected geometry copy provenance and transaction ownership; C-003 adjacent. |
| `vc_object_store/include/vc_object_store/objects/PackedStore.hpp` | VFS lifetime, close-before-finalize and memory/persistence options. |
| `vc_object_store/include/vc_object_store/objects/Project.hpp` | Resource tuple identity, owner filters, ordering and project copy; S-008/S-009. |
| `vc_object_store/include/vc_object_store/objects/StoreConversion.hpp` | Offline retention policy transition and retained current geometry. |
| `vc_object_store/include/vc_object_store/objects/UmbilicusFile.hpp` | Legacy stamps, dimension checks and shared volume publication. |
| `vc_object_store/include/vc_object_store/objects/imports/ParallelImportStream.hpp` | Bounded stream, phase closure and abort. |
| `vc_object_store/include/vc_object_store/objects/imports/ShardQueue.hpp` | Queue capacity, cancellation and draining. |
| `vc_object_store/include/vc_object_store/server/Server.hpp` | Server exclusive ownership, lanes and payload bounds; S-001/S-007. |
| `vc_object_store/src/Artifacts.cpp` | Shared artifact manifest/hash/range resolver and sidecar preparation. |
| `vc_object_store/src/CMakeLists.txt` | Native targets and optional link/server reader dependency wiring. |
| `vc_object_store/src/CtSheetNormal.cpp` | Shared Zarr, pack layout and invalid versus missing data boundaries; numerical normal math excluded. |
| `vc_object_store/src/Database.cpp` | Manifest/features, ownership and pool setup; S-001/S-004. |
| `vc_object_store/src/FetchQueue.cpp` | Fetch coalescing, selected scope, shutdown and cancellation. |
| `vc_object_store/src/FetchQueue.hpp` | Fetch queue request keys and completion lifetime. |
| `vc_object_store/src/HostedSession.cpp` | Cache identity/envelopes, coalescing, polling, prefetch and selection delegation. |
| `vc_object_store/src/HostedSession.hpp` | Attachment reader ownership and transport API boundary. |
| `vc_object_store/src/InitialImport.cpp` | Shard construction/abort, manifest and publication; single-shard path in S-004. |
| `vc_object_store/src/MemMap.cpp` | Open/stat/map and error cleanup. |
| `vc_object_store/src/ObjectCollection.cpp` | Snapshot closure, reference routing, immutable/import commit and copy; S-004. |
| `vc_object_store/src/ObjectCopy.cpp` | Selected metadata and geometry, composite savepoint; C-003 adjacent. |
| `vc_object_store/src/PackedCollection.cpp` | Sealed closure export, member relocation and artifact preparation. |
| `vc_object_store/src/PackedStore.cpp` | Packed directory/extents, VFS lock callbacks, memory budget and finalization. |
| `vc_object_store/src/PackedTrackStore.cpp` | Mapped arrays, header/count/dtype validation and immutable views. |
| `vc_object_store/src/Project.cpp` | Session requests, current/history reads, copy closure and resource lookups; S-008/S-009. |
| `vc_object_store/src/Provisioning.cpp` | Physical identities, table directory, allocations and retained version counters. |
| `vc_object_store/src/SourcePaths.cpp` | Canonical path/alias resolution, locator rewrite and owning database scope. |
| `vc_object_store/src/StoreConversion.cpp` | Offline retention policy, payload materialization, shared dependency resolver and staged reopen. |
| `vc_object_store/src/Tiff.cpp` | TIFF layout, complete strip/tile probes, exact byte read/write boundaries. |
| `vc_object_store/src/TrackCrossings.cpp` | Packed crossing input/index validation; numerical radix/event bodies excluded. |
| `vc_object_store/src/UmbilicusFile.cpp` | Checked frame stamps and shared volume request; R4 dimension fix checked. |
| `vc_object_store/src/Uuid.cpp` | Namespaced UUID v5 and shared binary/string handling. |
| `vc_object_store/src/Wire.cpp` | Wire version identities, metadata spelling and typed response envelopes. |
| `vc_object_store/src/catalog/Catalog.hpp` | Internal catalog preparation/download boundary; S-003. |
| `vc_object_store/src/catalog/CatalogStore.cpp` | Catalog placeholders and geometry-only fetch comparator, cache publication; S-002/S-003. |
| `vc_object_store/src/client/HostedStore.cpp` | Hosted catch-up, canonical UUID boundaries, transport and prepared submit; S-010. |
| `vc_object_store/src/formats/FiberInput.cpp` | Native fiber source decoding and branch identity handling. |
| `vc_object_store/src/formats/FiberInput.hpp` | Finite point array decoder. |
| `vc_object_store/src/formats/FiberJson.cpp` | Exact fiber/network reads and topology endpoint versions. |
| `vc_object_store/src/formats/FiberMetadata.hpp` | Shared fiber metadata/branch export contract. |
| `vc_object_store/src/formats/GeometryInput.cpp` | Source UUID derivation and source-owned record construction. |
| `vc_object_store/src/formats/GeometryInput.hpp` | Geometry-only versus source-owning decoder interfaces. |
| `vc_object_store/src/formats/GeometryOutput.cpp` | Selected native points, masks, scale and channels to file. |
| `vc_object_store/src/formats/GeometryOutput.hpp` | Exact snapshot format writer interface. |
| `vc_object_store/src/formats/PointCollectionInput.cpp` | Point IDs, renumbering and range validation; R4 fix retained. |
| `vc_object_store/src/formats/RemoteSource.cpp` | Remote coordinate/mask staging only; auxiliary channel absence documented. |
| `vc_object_store/src/formats/RemoteSource.hpp` | Remote geometry source protocols and supplied files. |
| `vc_object_store/src/formats/SurfaceInput.cpp` | TIFF geometry-only versus source-owned UUID; S-002. |
| `vc_object_store/src/formats/SurfaceInput.hpp` | Authoritative surface save/layout validation boundary. |
| `vc_object_store/src/formats/TrackInput.cpp` | Packed track identity and typed source geometry. |
| `vc_object_store/src/formats/TrackInput.hpp` | Canonical track source ID and geometry-derived length helper. |
| `vc_object_store/src/formats/UmbilicusInput.cpp` | Native versioned umbilicus data and metadata import. |
| `vc_object_store/src/geometry/GeometryExport.cpp` | Selected object/revision/topology and destination staging/publication. |
| `vc_object_store/src/geometry/GeometryIngest.cpp` | Registered/direct/session ingestion, failures and selected identity; S-002. |
| `vc_object_store/src/geometry/GeometryIngestBatch.cpp` | Whole batch savepoint, endpoints and typed relationships. |
| `vc_object_store/src/geometry/GeometryIngestBatch.hpp` | Shared canonical source UUID and ingest boundary. |
| `vc_object_store/src/geometry/InitialGeometry.hpp` | Prepared geometry ownership and caller-owned savepoint. |
| `vc_object_store/src/imports/AfvRecords.hpp` | AFV/AFC into shared initial records and typed evidence. |
| `vc_object_store/src/imports/AfvSource.cpp` | Native cursor snapshots, source schemas/ranges and mappings. |
| `vc_object_store/src/imports/FiberGraphSolveExtension.cpp` | Typed solver run, sample, hypothesis, width, position and metric rows. |
| `vc_object_store/src/imports/FiberImportExtension.cpp` | Narrow provenance and typed native constraint/normal fields. |
| `vc_object_store/src/imports/SourceIdentityIndex.cpp` | Source point index publication and resolved source records. |
| `vc_object_store/src/imports/SourceIdentityIndex.hpp` | Source index maintenance contract. |
| `vc_object_store/src/imports/SourceRelationships.cpp` | Native topology reconciliation and shared endpoint anchors. |
| `vc_object_store/src/imports/SourceRelationships.hpp` | Native relationship endpoint boundary. |
| `vc_object_store/src/internal/Artifacts.hpp` | Artifact publication/verification and resolver helper contract. |
| `vc_object_store/src/internal/CollectionReads.hpp` | Resolved collection query source and closure access. |
| `vc_object_store/src/internal/Database.hpp` | Leased SQLite connections, shared resolvers and caches; R4 C6 actual multimap. |
| `vc_object_store/src/internal/DatabaseCopy.hpp` | Snapshot copy and shared locator rewriting primitives. |
| `vc_object_store/src/internal/Digest.hpp` | Shared hash normalization and typed digest binding. |
| `vc_object_store/src/internal/ImmutableImport.hpp` | Private construction versus published readonly initial state; S-004. |
| `vc_object_store/src/internal/InitialEdges.cpp` | Trusted edge insertion, deferred indexes, allocator and caller savepoint. |
| `vc_object_store/src/internal/InitialEdges.hpp` | Trusted initial edge inserter contract. |
| `vc_object_store/src/internal/InitialObjects.cpp` | Trusted object preparation, typed payload and reserved row IDs. |
| `vc_object_store/src/internal/InitialObjects.hpp` | Trusted object inserter caller-owned savepoint. |
| `vc_object_store/src/internal/Ownership.cpp` | Service/standalone ownership and canonical pathname; S-001. |
| `vc_object_store/src/internal/Ownership.hpp` | Host-local writable ownership contract; S-001. |
| `vc_object_store/src/internal/Publication.hpp` | Staged database/publish cleanup and sync helpers. |
| `vc_object_store/src/internal/ReadRequest.hpp` | Shared read budgets and cancellation scope. |
| `vc_object_store/src/internal/Schema.cpp` | Core binary layout and optional history/spatial schema. |
| `vc_object_store/src/internal/Schema.hpp` | Shared schema construction contract. |
| `vc_object_store/src/internal/Sqlite.cpp` | Physical open, statement pools, binds and savepoints; R4 C6 actual multimap. |
| `vc_object_store/src/internal/Sqlite.hpp` | Typed binds and nested savepoint interface. |
| `vc_object_store/src/linking/Detectors.hpp` | Detector dispatch over a pinned geometry snapshot. |
| `vc_object_store/src/linking/FiberSurface.cpp` | Pinned fiber inputs and cache keys; numerical strip detector excluded. |
| `vc_object_store/src/linking/LineSurface.cpp` | Geometry/volume cache provenance and publish paths; numerical strip kernels excluded. |
| `vc_object_store/src/linking/LineSurface.hpp` | Shared line/surface loader and publish API. |
| `vc_object_store/src/linking/LinkCandidates.cpp` | Common candidate identity, stale deletion and batch boundary; S-006. |
| `vc_object_store/src/linking/LinkObjects.cpp` | Prepare/commit/marker with retained evidence. |
| `vc_object_store/src/linking/PointCollectionSurface.cpp` | Selected point collections, annotations and coordinate snapshots; numerical projection/scoring excluded. |
| `vc_object_store/src/linking/StoredGeometry.cpp` | Selected geometry/paths/transforms and write guards. |
| `vc_object_store/src/linking/StoredGeometry.hpp` | Shared pinned geometry and cache declarations. |
| `vc_object_store/src/linking/SurfaceOverlap.cpp` | Exact/current materialization, metadata/frame and revision matching; RANSAC/fitting excluded. |
| `vc_object_store/src/linking/TrackSurface.cpp` | Shared pinned track/frame loading, numerical strip detector excluded. |
| `vc_object_store/src/linking/TrackTrack.cpp` | Track source grouping, stable input geometry and crossing publication; numerical crossing excluded. |
| `vc_object_store/src/migration/ConvertStore.cpp` | Upstream migration snapshots, format validation, shared relocation, publish and replacement; S-005 adjacent. Legacy history numbering examined only at conversion boundaries. |
| `vc_object_store/src/server/Server.cpp` | Server ownership, staged uploads, dispatch, ranges and checked submit; S-001 and adjacent S-007. |
