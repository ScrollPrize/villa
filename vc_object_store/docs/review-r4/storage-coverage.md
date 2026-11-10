# R4 physical/storage coverage

Baseline: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. This second native lane covers physical database/VFS/collection/import/export/client-server/linking code within `vc_object_store`, plus corresponding public/internal headers and source wiring. 118 file dispositions below. Logical readers/mutations/history/geometry engine and Python/bindings belong to the other lanes. No production changes, installs, builds or runtime probes were performed here.

The inspected scope is complete for the assigned files' storage/identity/reference/mapping/cache/transaction/publication/control paths. Numerical crossing/overlap/normal/tile algorithms unrelated to store state are explicitly excluded, not claimed as exhaustively mathematically reviewed. Tests used by adjacent lanes do not establish new scenarios here; candidates are code-traced only. No finding quota was imposed.

## Findings and rejected candidates

- S-002: public direct object-copy helper lacks a composite savepoint (P2; no in-repository production caller).
- S-004: stamped umbilicus dimensions narrow from int64 to int without checking supported range (P3).
- S-005: source point-ID renumber counter can execute signed overflow at INT64_MAX (P3).
- S-006: CLI report output can truncate its input or its newly imported object-store database (P1).
- S-007: installed static CMake dependency targets are incompletely provisioned (P2).
- S-003: withdrawn by author after root pointed out the shared publishPayload guard; null values never persist, so the initially proposed cache failure is not reachable. Original candidate and objection remain individually logged.
- S-001: root candidate rejected by the independent model's documented remote coordinate-only contract. Not counted as a confirmed storage defect here.

No new defect was established in key-based shard routing, sealed dependency discovery, normal current/exact collection reads, binary UUID/digest handling, known-extension copy preservation, snapshot-backed copies, retained external-locator relocation, packed checksum/directory validation, or import stream cancellation and phase ordering. This is a bounded code inspection, not a proof for every input or scheduler interleaving.

Public `writeObjectCopy` has no production repository caller; its finding is specifically API contract consistency, not claimed GUI data loss. Batch source imports intentionally trust supplied global UUID endpoints and do not reopen/validate shards: that omission is part of the requested fast-import design, not a finding. Source metadata, extension provisioning metadata and extension data rows were distinguished in inspection.

## File dispositions

| File | Disposition |
| --- | --- |
| `vc_object_store/include/vc_object_store/CtSheetNormal.hpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/include/vc_object_store/MemMap.hpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/include/vc_object_store/PackedTrackStore.hpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/include/vc_object_store/Progress.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/TerminalProgress.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/Tiff.hpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/include/vc_object_store/TrackCrossings.hpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/include/vc_object_store/Uuid.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/client/HostedStore.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/migration/ConvertStore.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/object_links/LinkObjects.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/object_links/ObjectLinks.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/AfvSource.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/Artifacts.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/FiberGraphSolveExtension.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/FiberImportExtension.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/FiberJson.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/GeometryExport.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/GeometryIngest.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/InitialImport.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/ObjectCollection.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/ObjectCopy.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/PackedStore.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/Project.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/StoreConversion.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/UmbilicusFile.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/imports/ParallelImportStream.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/objects/imports/ShardQueue.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/include/vc_object_store/server/Server.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/Artifacts.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/CMakeLists.txt` | Build/source wiring inspected; no runtime store resolver; unrelated build dependencies not installed. |
| `vc_object_store/src/CtSheetNormal.cpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/src/Database.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/FetchQueue.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/FetchQueue.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/HostedSession.cpp` | S-003 rejected after tracing publishPayload: null results are deliberately not persisted. Positive cache/reference/freshness paths reviewed. |
| `vc_object_store/src/HostedSession.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/InitialImport.cpp` | Trusted batch construction, UUID routing, phase ordering, failure propagation and publication/cleanup inspected; unchecked endpoint lookup is intentional trusted-import behavior. |
| `vc_object_store/src/MemMap.cpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/src/ObjectCollection.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/ObjectCopy.cpp` | S-002: public composite write lacks operation-wide rollback. No in-repository production caller. |
| `vc_object_store/src/PackedCollection.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/PackedStore.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/PackedTrackStore.cpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/src/Project.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/Provisioning.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/SourcePaths.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/StoreConversion.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/Tiff.cpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/src/TrackCrossings.cpp` | Input representation/bounds, shared-reader use, ownership and platform branches inspected; unrelated numerical algorithm internals excluded. |
| `vc_object_store/src/UmbilicusFile.cpp` | S-004: stamped dimension narrowing lacks representability check. |
| `vc_object_store/src/Uuid.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/Wire.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/catalog/Catalog.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/catalog/CatalogStore.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/client/HostedStore.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/FiberInput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/FiberInput.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/FiberJson.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/FiberMetadata.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/GeometryInput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/GeometryInput.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/GeometryOutput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/GeometryOutput.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/PointCollectionInput.cpp` | S-005: next-ID arithmetic can overflow at the int64 boundary. |
| `vc_object_store/src/formats/RemoteSource.cpp` | S-001 independently rejected as documented coordinate-only contract; remote transport staging and required-file handling reviewed. |
| `vc_object_store/src/formats/RemoteSource.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/SurfaceInput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/SurfaceInput.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/TrackInput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/formats/TrackInput.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/formats/UmbilicusInput.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/geometry/GeometryExport.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/geometry/GeometryIngest.cpp` | Batch-wide savepoint and source identity/coordinates checked; per-source reporting/partial progress is explicitly separate from atomic ingestGeometryBatch. |
| `vc_object_store/src/geometry/GeometryIngestBatch.cpp` | Batch-wide savepoint and source identity/coordinates checked; per-source reporting/partial progress is explicitly separate from atomic ingestGeometryBatch. |
| `vc_object_store/src/geometry/GeometryIngestBatch.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/geometry/InitialGeometry.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/imports/AfvRecords.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/imports/AfvSource.cpp` | Trusted batch construction, UUID routing, phase ordering, failure propagation and publication/cleanup inspected; unchecked endpoint lookup is intentional trusted-import behavior. |
| `vc_object_store/src/imports/FiberGraphSolveExtension.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/imports/FiberImportExtension.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/imports/SourceIdentityIndex.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/imports/SourceIdentityIndex.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/imports/SourceRelationships.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/imports/SourceRelationships.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Artifacts.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/CollectionReads.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Database.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/DatabaseCopy.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Digest.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/ImmutableImport.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/InitialEdges.cpp` | Trusted batch construction, UUID routing, phase ordering, failure propagation and publication/cleanup inspected; unchecked endpoint lookup is intentional trusted-import behavior. |
| `vc_object_store/src/internal/InitialEdges.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/InitialObjects.cpp` | Trusted batch construction, UUID routing, phase ordering, failure propagation and publication/cleanup inspected; unchecked endpoint lookup is intentional trusted-import behavior. |
| `vc_object_store/src/internal/InitialObjects.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Ownership.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/internal/Ownership.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Publication.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/ReadRequest.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Schema.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/internal/Schema.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/internal/Sqlite.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |
| `vc_object_store/src/internal/Sqlite.hpp` | Public/internal contract and declarations inspected with corresponding implementation; template queue/failure and progress synchronization inspected where implemented here. |
| `vc_object_store/src/linking/Detectors.hpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/FiberSurface.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/LineSurface.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/LineSurface.hpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/LinkCandidates.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/LinkObjects.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/PointCollectionSurface.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/StoredGeometry.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/StoredGeometry.hpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/SurfaceOverlap.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/TrackSurface.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/linking/TrackTrack.cpp` | Detector input selections, geometry references, candidate preparation/publication and cache/source identity inspected; unrelated numerical detector bodies excluded. |
| `vc_object_store/src/migration/ConvertStore.cpp` | Physical/source identity, sealed dependency pinning, shared resolution/copy/locator relocation and final publication inspected; schema-specific SQL is at storage boundary. |
| `vc_object_store/src/server/Server.cpp` | Storage/control-path review: common ownership/reference/publication APIs; no additional concrete defect established. |

## Finite residual CLI/build inventory

The root's final tracked-file complement added the following 15 CLI/support/benchmark/script/build entries to the 118 source/header entries above: **133 physical-lane dispositions total**. These are not numerical test bodies; regression tests remain consulted evidence, not exhaustively audited production. The script and benchmark implementation were read for source identity, destructive outputs, atomic publication, measurement and cleanup semantics. Two additional concrete candidates S-006/S-007 were logged separately and sent for independent-model adjudication.

| File | Disposition |
| --- | --- |
| `vc_object_store/apps/support/ProcessMetrics.hpp` | Platform-specific RSS units checked; metrics only, no store resolution or mutation. |
| `vc_object_store/apps/support/SqliteMutexProfile.hpp` | Opt-in forwarding profiler checked: startup-before-connections contract, native lock forwarding, thread-local counters, bounded stack samples; no lock-elision production alternative. |
| `vc_object_store/apps/vc_object_convert.cpp` | Calls common offline conversion/history maintenance; source/destination/report distinctness/existence guards inspected; report generation occurs after conversion, not claimed multi-file transactional. |
| `vc_object_store/apps/vc_object_export.cpp` | Current or complete explicit store/generation/version selection passed into common selected export and destination protection. |
| `vc_object_store/apps/vc_object_import_afv.cpp` | S-006: unchecked report stream can truncate protected source/output. UUID-sharded reader/writer pipeline, typed extension provenance, cancellation and aggregate checks reviewed; optional readback separated from import timing. |
| `vc_object_store/apps/vc_object_ingest.cpp` | Common format import/channel API calls, coordinate options and reported partial-success boundary inspected; no consumer-local resolver. |
| `vc_object_store/apps/vc_object_links.cpp` | CLI options passed to existing detector APIs; sequential detector publications are explicit separate operations; no alternative link engine. |
| `vc_object_store/apps/vc_object_server.cpp` | Thin server startup/configuration wrapper; default loopback/no-auth scope is explicitly documented; server implementation reviewed above. |
| `vc_object_store/benchmarks/prepared_import.cpp` | Preparation/shared AFV conversion, prototype-vs-native workload distinction, source limits, UUID routing, worker-summed vs wall timers, optional sampled readback and private-output cleanup inspected. Benchmark prototype is explicitly not supported production vcobj. |
| `vc_object_store/benchmarks/README.md` | Diagnostic sampling/timing/readback/cleanup boundaries checked against implementation; performance projections are labelled and source/worker sums distinguished. |
| `vc_object_store/scripts/import_afv.py` | Native AFV source reader + public Python write API; derived UUIDs/typed source extension, private staging and no-replace link publication inspected; no independent AFV codec. |
| `vc_object_store/scripts/run_cpp_tests.sh` | Argument-free existing runner inspected: bounded explicit output/fixture roots, cleanup traps, agent mode and targeted configure/test commands; not executed in this lane. |
| `vc_object_store/CMakeLists.txt` | Build target/source/options/export boundary inspected; S-007 concerns installed dependency closure, not source-tree tests. |
| `vc_object_store/cmake/vc_object_storeConfig.cmake.in` | S-007: zstd compatibility alias and conditional Boost dependency missing for exported static interfaces. |
| `vc_object_store/pyproject.toml` | Python build-component/install boundary and native source cache keys inspected; no setup/install run. |
