# Full vcobj review: cpp-consumers

Baseline: `3df5b7783`. Fresh full-scope review of all assigned relevant files, not only the recent diff. Previous findings R2-1–7 remain in [the combined findings](coherent-store-review-3df5b7783.md); corroborations are not counted again. Every assigned path requires a current disposition and inspected symbols/evidence. No finding quota. Follow additional callers and append rows if needed; report why pure numerical/build-only paths are excluded. Review only: no production fixes or commits.

## Coverage (132 assigned files)

| File | Disposition | Inspected behavior / evidence |
| --- | --- | --- |
| `spiral-fitting/cpp/CMakeLists.txt` | BUILD ONLY | Target/source registration for shared TrackCrossings; no runtime store handling. |
| `volume-cartographer/CMakeLists.txt` | BUILD ONLY | ObjectStore.cmake integration and package/runtime discovery; no record resolver. |
| `volume-cartographer/apps/CMakeLists.txt` | BUILD ONLY | Native targets link vc_objects; no runtime handling. |
| `volume-cartographer/apps/VC3D/CMakeLists.txt` | BUILD ONLY | GUI target links shared libraries/converter; no storage logic. |
| `volume-cartographer/apps/VC3D/CState.cpp` | CHECKED | Attachment/routing prepare/replace/detach; polling and draftActiveSurface use shared session; project boundary flush guards. |
| `volume-cartographer/apps/VC3D/CState.hpp` | CHECKED | Session, attachment routes, draft and cache ownership declarations. |
| `volume-cartographer/apps/VC3D/CWindow.cpp` | CHECKED | Catalog event full-reference comparisons, volume-switch exact reopen, current model artifacts and hosted status; lifecycle delegates shared sessions. |
| `volume-cartographer/apps/VC3D/CWindow.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp` | CHECKED | Lasagna current-model request and CLI orchestration delegate managers/catalog; no data decoding. |
| `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp` | CHECKED | File-tool boundary, shared StoreSurfaces arguments; exportedObjCopy uses session.materializeArtifacts; process completion selects current artifact intentionally. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.cpp` | FINDING CPP-F3; R2 CORROBORATION | selectFiber/navigate stale FiberDetail point counts; region path pinned graphReadViewAround; requestChains hosted snapshot issue already R2. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/FiberCollectionGenerator.hpp` | CHECKED | External generator request/event structures; source UUID and scale forwarded, no store decoding. |
| `volume-cartographer/apps/VC3D/FiberCollectionGeneratorDialog.cpp` | CHECKED | Output .vcobj path choice only; no database handling. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp` | CHECKED | setObjectStore/storeError/current surfaceHasModel use session location/artifacts; service HTTP state separate. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp` | CHECKED | Session weak ownership, service endpoint binding and current-model contract. |
| `volume-cartographer/apps/VC3D/LineAnnotationCompoundTest.inc` | OPTIONAL INTEGRATION TEST | Real compound volume fixture uses isolated project/store; primarily sampling/input-routing tests, not vcobj resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.cpp` | CHECKED; R2 CORROBORATION | Decode/line cache/peer closures, complete edit guards/mapping operations, structural-ID allocation, durable submission and receipt repinning reviewed; R2 hosted coherence remains; numeric-only owner compare unconfirmed risk. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationFiberCacheTest.inc` | TEST COVERAGE | Shared closure copy, export bytes/no-op persistence, peers/registration invalidation checked; does not inject hosted mutation within callback. |
| `volume-cartographer/apps/VC3D/LineAnnotationNearbyBenchmark.inc` | OPTIONAL BENCHMARK | Isolated project, hosted read-only load, sync/async numerical comparison and cancellation; no concurrent hosted geometry edit fixture. |
| `volume-cartographer/apps/VC3D/LineAnnotationObjectStoreTest.inc` | TEST COVERAGE | Native create/edit/history reset, point collections, mapped fibers, structural IDs, stale writes, branch edits and controls-only geometry; missing stale correction and changed-size navigator cases. |
| `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp` | CHECKED | StoredFiber retains optional complete pin, frame mapping, structural IDs and annotations; resident value type only. |
| `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp` | CHECKED | Merge/split/redirect are in-memory domain edits; compare shared coordinate identity/mapping; persistence delegated controller. |
| `volume-cartographer/apps/VC3D/MenuActionController.cpp` | CHECKED | Project create/reset/saveAs, attach/routing/import all delegate shared helpers; current import/export actions explicit. |
| `volume-cartographer/apps/VC3D/MenuActionController.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp` | CHECKED | materializeJsonDocuments hashes generated contents in owned QTemporaryDir; atomicWrite is transport materialization, not store interpretation. |
| `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp` | CHECKED | Shared background region scheduling, cancellation, completeness and prefetch; no physical storage interpretation. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp` | CHECKED | createSampleProjectFiles/createOpenDataSampleProject use ObjectStore::create and shared addCatalogSample; external resource setup separate. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp` | CHECKED | Creation/resource APIs carry attachment; no object placement implementation. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.cpp` | CHECKED | reload retains refs/mappings; flush guards edits/removals and repins receipt; events compare complete identities. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/ProjectResources.cpp` | CHECKED | planUmbilicusImport/importVolumeUmbilicus/annotationUmbilici/retireUmbilici use shared binding, geometry and explicit current-edit APIs. |
| `volume-cartographer/apps/VC3D/ProjectResources.hpp` | CHECKED | Import resource contracts; no alternate physical resolver. |
| `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp` | CHECKED | Asynchronous count cache checks attachment identity and request generation; shared session counts. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp` | CHECKED | rebuildStores/counts/status/routes and selectedStore use shared session and project refs; only physical paths shown in UI. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp` | CHECKED | UI store identity and count-cache declarations. |
| `volume-cartographer/apps/VC3D/SeedingWidget.cpp` | FINDING CPP-F4 | Relative winding annotation persists numbered intersections despite partial/remote geometry query; shared tool route otherwise used. |
| `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp` | CHECKED | Loaded surface mutations publish through guarded saver; command-line operations intentionally current source UUID via shared StoreSurfaces; hosted file-tool limitation explicit. |
| `volume-cartographer/apps/VC3D/SessionDrafts.cpp` | CHECKED | restore/save/discard draft orchestration delegates session APIs and QPointer-guarded callbacks. |
| `volume-cartographer/apps/VC3D/SessionDrafts.hpp` | CHECKED | Draft session ownership and callback declarations. |
| `volume-cartographer/apps/VC3D/SpiralPanel.cpp` | UI ONLY | Fit controls/commit result UI forward service requests; no record or geometry reads. |
| `volume-cartographer/apps/VC3D/SpiralPanel.hpp` | UI ONLY | Widget declarations and summary label; no store resolver. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp` | CHECKED | projectBlock reads volumeUmbilicus through shared API, forwards owner generation/version; resources use project reader. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp` | CHECKED | setObjectStore/status/head/changesSince/read position use session APIs; artifact-cache protocol independent of vcobj storage. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp` | CHECKED | StoreBinding tracks generation/commit and receipts; no per-consumer DB resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp` | CHECKED | Brush/PCL loaded references guard submit, retained point-ID allocator and receipt references; scalar frame protocol intentional. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp` | CHECKED | readObject selects current once for live-vs-fit comparison; exact geometry/annotations; relationshipsAt complete references. Scalar-frame limitation explicit in Spiral input protocol. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp` | CHECKED | Live comparison, not exact replay, documented; editable point collection retains full reference. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp` | CHECKED | listStoredFits/readStoredFit select complete geometry-owned record, exact channels and declared scalar frame; retired-fit filter intentional. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp` | CHECKED | Stored-fit geometry revision and read contracts. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp` | CHECKED | Fit-input async generation guard and complete reference serialization, reload/compare and save conflict handling. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/StoreImportController.cpp` | CHECKED | prepare/start validate shared canonical identity and binding; ingestGeometrySource shared; package change cancellation. |
| `volume-cartographer/apps/VC3D/StoreImportController.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp` | CHECKED | Current listing vs explicit exact open; shared selected geometry, pinOf, channel loader, region/nearest APIs, intentional hosted CLI refusal. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp` | CHECKED | Complete pinned open result and frame; listing mapping and file-tool restrictions explicit. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp` | CHECKED | History opens complete versions; restore/copy/current tags/binding edits delegate session and shared destination handling. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/SurfacePanelStoreTest.inc` | TEST COVERAGE | Catalog/current/history/editing level mapping, sample restore, draft cuts and registration panel tests; no partial region winding-write test. |
| `volume-cartographer/apps/VC3D/VCAppMain.cpp` | ENTRYPOINT ONLY | --attach-store and native test flags forward controller; no persistent record interpretation. |
| `volume-cartographer/apps/VC3D/ViewerManager.cpp` | CHECKED | locate uses resident SurfaceQuerySet and surface-selected frames; no physical resolver. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp` | CHECKED | parseAffine delegates affine structural validation to shared VolumeAffine. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp` | CHECKED DECLARATIONS | Relevant session ownership, pin/reference/frame, cache and edit API declarations checked against implementation; no independent resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp` | CHECKED | Session status/read-only project route descriptions, shared project creation and resource handlers; no schema resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp` | CHECKED | Import shares StoreImportController; export selects exact revision and exportGeometry; attach/detach/project resources delegate CState/helpers. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp` | CHECKED | Catalog row, writability and project attachment-reference serialization helpers; UUID is domain key, not local DB row. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp` | FINDING CPP-F2 | loadStore discards listed record reference before current geometry read; old range metadata may be applied to new points. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp` | CHECKED / CPP-F2 CALLER | startLoad/request revision cancellation follows asynchronously polled events; does not pin hosted read callback before first event. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp` | CHECKED | Displayed surface metadata selects published artifact; shared catalogReference/fetchPublishedFile; no fresh geometry lookup. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp` | CHECKED | Resident SurfaceQuerySet overlap path and cancellation/invalidation; no persistent object interpretation. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp` | CHECKED | Draft/stored cache distinction, exact geometry selection and shared region APIs; complete pin invalidation. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp` | CHECKED | Resident projection cache keyed geometry generation/plane basis; ViewerManager locate/shared region shape delegation. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp` | CHECKED | Autosave snapshot/draft sequencing, SurfaceStoreEdit origin matching, async publication guards; correction edit issue separate CPP-F1. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp` | FINDING CPP-F1 CALLER | Editing/draft base uses full pin; correction collection load/save retains JSON only; recovery validates complete base identity and coordinates. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp` | FINDING CPP-F1 | read/write strip selected collection refs/mappings via JSON adapters; replacement guards only fresh heads. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp` | FINDING CPP-F1 CONTRACT | Correction read/write API exposes JSON only, no loaded references or removal baseline. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp` | CHECKED | Selected pin, immutable geometry view, loaded transform equality plus atomic check, durable retry/receipt repin; no history-mode branch. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp` | CHECKED | Displacement copy carries complete selected reference, current geometry saved first; draft and correction update delegates module. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp` | CHECKED | Current model job setup and provenance source list; current source explicitly pinned store/lineage/version, shared session; source jobs intentionally current. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.cpp` | CHECKED | Resident manipulation; draft initial complete base and coordinate identity through session set/updateDraft; persistence delegated. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp` | CHECKED | Registration graph display and current dialog operations use shared registerVolumeTransform/chooseVolumeTransform. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp` | CHECKED DECLARATIONS | Store-facing ownership, selected identities, async/draft contracts and value types reviewed; implementation disposition applies. |
| `volume-cartographer/apps/src/FiberConstraintExtraction.cpp` | CHECKED | Single native read transaction keeps selected metadata/mappings/current geometry coherent; exact batch reads, completeness refusal; separate output DB workflow. |
| `volume-cartographer/apps/src/FiberConstraintGeometry.hpp` | CHECKED | Current geometry/array shape selection under supplied read transaction replaces stale ingest summary. |
| `volume-cartographer/apps/src/vc_gen_umbilicus.cpp` | FILE ADAPTER | Produces standard umbilicus file with explicit frame metadata; no vcobj row handling. |
| `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` | CHECKED | Live candidate discovery and per-surface pinned loads through shared StoreSurfaces/proximity API; creates derived objects. |
| `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp` | CHECKED | Live UUID selection then pinned per-source loads; derived output/artifacts through shared StoreSurfaces/ToolObjects. |
| `volume-cartographer/apps/src/vc_obj2tifxyz.cpp` | CHECKED | Optional store source loads through StoreSurfaces; output creates new shared object/provenance, separate raw OBJ/tifxyz file mode. |
| `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp` | CHECKED | Store output through StoreSurfaces::create; no direct store reads. |
| `volume-cartographer/apps/src/vc_render_tifxyz.cpp` | CHECKED | StoreSurfaces load in target volume; rejects competing manual affine; channel loader owns session after temporary wrapper dies. |
| `volume-cartographer/apps/src/vc_tifxyz2obj.cpp` | CHECKED | StoreSurfaces load, ToolObjects artifact publication; explicit scratch export separate from object resolution. |
| `volume-cartographer/cmake/ObjectStore.cmake` | BUILD ONLY | Installed/sibling package selection and shared targets; no bootstrap side effects/runtime storage branches. |
| `volume-cartographer/core/CMakeLists.txt` | BUILD ONLY | Core source registration/shared library linkage. |
| `volume-cartographer/core/include/vc/core/PointCollections.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/core/StoredCollection.hpp` | CHECKED | JSON adapter declarations, not persistence ownership. |
| `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp` | CHECKED | CLI binding/options parse shared voxel scales; explicit file-only boundary. |
| `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp` | NUMERIC ONLY | Surface validity/area math; no store source selection. |
| `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp` | CHECKED DECLARATIONS | Identity/source/geometry contracts inspected; shared store adapters or resident value structures, no alternate storage resolver. |
| `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp` | CHECKED | Raw fiber-file JSON UUID validation; not a vcobj identity resolver. |
| `volume-cartographer/core/src/FiberView.cpp` | NUMERIC ONLY | Polyline interval/clipping delegates geometric helpers; no record resolver. |
| `volume-cartographer/core/src/GridStore.cpp` | OUTSIDE VCOBJ STORAGE | Resident grid/mmap abstraction, not vcobj schema or version resolution. |
| `volume-cartographer/core/src/GrowSurface.cpp` | NUMERIC ONLY | Resident SurfaceIndex/VertexIndex/SurfaceQuerySet consumers; no persistent identity handling. |
| `volume-cartographer/core/src/PointCollections.cpp` | CHECKED | Resident collections JSON identity and ID handling; imported document parsed before replacement; persisted IDs subsequently shared allocator. |
| `volume-cartographer/core/src/ProjectCreation.cpp` | CHECKED | New/existing store validation with shared API; staged project file publication, rollback of newly created store only. |
| `volume-cartographer/core/src/ProjectDocument.cpp` | CHECKED | Project-only locations/route IDs; saveAs rebases references without copying databases. |
| `volume-cartographer/core/src/QuadSurface.cpp` | CHECKED | Native tifxyz identity/new-copy helpers and resident geometry; vcobj data loading delegated QuadSurfaceStore; no DB bypass. |
| `volume-cartographer/core/src/QuadSurfaceStore.cpp` | CHECKED | readSurfaceGrid selected overload, revision-bound channel loader and shared grid validity/edit operations; no physical SQL. |
| `volume-cartographer/core/src/StoreSurfaces.cpp` | CHECKED | Native-file transaction load/pin, guarded update, shared source-coordinate resolution; no hosted promise. |
| `volume-cartographer/core/src/StoredCollection.cpp` | CHECKED / CALLER GAP | JSON point adapter intentionally omits transient reference/mapping; SurfaceCorrections caller loses these (CPP-F1). |
| `volume-cartographer/core/src/SurfaceQuerySet.cpp` | CHECKED | Rebuild/locate delegates common SurfaceIndex with resident geometry identity mapping; no store resolution. |
| `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp` | NUMERIC ONLY | Surface validity/intersections; no store access. |
| `volume-cartographer/libs/flatboi/flatboi.cpp` | CHECKED | run_store loads transformed source, creates derived surface and publishes reports through shared wrappers. |
| `volume-cartographer/scripts/benchmark_cleanup.cpp` | NUMERIC ONLY | VertexIndex benchmark on in-memory geometry. |
| `volume-cartographer/scripts/build_macos.sh` | BUILD ONLY | Dependency discovery/build mounts/targets inspected for shared store; no runtime store consumer. |
| `volume-cartographer/scripts/ci.sh` | BUILD ONLY | Dependency discovery/build mounts/targets inspected for shared store; no runtime store consumer. |
| `volume-cartographer/tools/vc3d-mcp/vc3d_mcp/tools/catalog_volume.py` | CHECKED FORWARDER | Typed RPC argument/job forwarding only; native bridge owns import/export/routing. Export response documentation old objectRevision name is stale, not storage regression. |
| `volume-cartographer/tools/vc3d-mcp/vc3d_mcp/tools/stores.py` | CHECKED FORWARDER | Typed RPC argument/job forwarding only; native bridge owns import/export/routing. Export response documentation old objectRevision name is stale, not storage regression. |

## Additional findings

### CPP-F1 — Correction editing drops the loaded guards

`SurfaceCorrections.cpp:29–48` converts selected `AttachedCollection` values to plain collection JSON, then reconstructs them on save without their complete references or coordinate-mapping guards. `SegmentationModule_Session.cpp:392–425` retains only that JSON and the surface UUID. The shared `replaceAttachedCollections` implementation (`vc_object_store/src/AttachedDocument.cpp:190–217`) guards fresh heads instead of the earlier loaded revisions and removes currently attached collections missing from the submitted set. Two editors can therefore overwrite one another's correction edits or remove a newly attached collection. This is older code, not introduced by the last delta. Fix boundary: retain complete correction selection/baseline membership in the editing state and pass it through the shared replacement API, including expected removals. Missing test: edit/delete/add from a second session between correction load and save, plus registration change.

### CPP-F2 — 3D fiber rendering discards the listed revision

`fibers3d/Fiber3DData.cpp:177–225` takes a listed `ObjectRecord`, but at line 186 requests transformed geometry by UUID rather than its complete reference. It then applies the old record's family/tags/name and inferred-gap ranges to the fresh points. A hosted writer shortening the fiber between listing and geometry read can raise `Inferred gap outside fiber` at line 225; less obvious edits silently mix annotation and geometry states. `Fiber3DPanel` event cancellation does not prevent this before the next asynchronous poll; hosted `readStore` callbacks are not snapshot-isolated. Fix boundary: select the listed reference and carry that selected record plus one captured transform context through the shared view API. Missing test: interleaved hosted edit between list and geometry read. Older code; separate from the core current-geometry resolver finding and existing R2 chain rendering case.

### CPP-F3 — Ingestion counts are used as live geometry shape

`FiberCollectionController.cpp:710–738` builds the navigation slider and initial midpoint from `FiberDetail.linePointCount/controlPointCount`; `navigate` at 767–774 reads the selected revision's actual array. The public `ObjectStore.hpp:27–28` contract explicitly says those detail fields are ingestion summaries and `putGeometry` does not update them. A legitimate geometry-only edit shrinking 1000 samples to 100 leaves the initial slider midpoint at 499 and navigation fails; extension makes the tail inaccessible. Fix boundary: derive shape/count from the selected geometry through the shared typed geometry API, not duplicated summary maintenance in callers. The Python `fiber_store._catalog` (parent review, lines 59–66) likewise filters using stale point count and samples using stale length/bbox; this is one cross-consumer contract finding. The core reviewer reproduced the underlying contract with a self-cleaning fixture through normal checked `edit_geometry_sequence`: shrinking 4 points to 2 left summary count 4/length 3/old bbox while geometry became [2,3]; reading old point 3 raised `geometry point range out of bounds`. Growing to 6 likewise left count 4. Missing test: geometry-only sample-count change followed by GUI navigation and catalog extraction. Older code.

### CPP-F4 — Partial surface intersections become persisted winding labels

`SeedingWidget.cpp:1094–1100` consumes `SurfaceCatalog::region` runs without checking `complete` or pending remote geometry. `SurfaceCatalog.cpp:280–296` explicitly carries both, and default `RegionOptions` has output/processing/candidate budgets. `finalizePathLabelWraps` at 935–949 persists those intersections as a point collection and numbers them consecutively; points are added at 1178–1183. A missing middle catalog surface or truncated query therefore silently assigns the wrong relative winding steps to surviving points. This is not merely incomplete display. Fix boundary: complete/fetch/retry via the shared region-query workflow (or explicitly refuse the write) before creating the numbered collection. Missing test: remote placeholder or budget stop prevents annotation publication. Older code.

## Evidence and limits

All 132 assigned paths have dispositions. Relevant store-facing functions and reachable helpers were inspected, not unrelated tracing numerics/UI internals. Cross-lane helper evidence: `AttachedDocument.cpp`, `GeometryViews.cpp`, `ObjectStore.hpp`, and `GeometryRegion.hpp`; their full implementations remain assigned to the core reviewer. Existing R2-1/R2-2 are corroborated, not recounted. Valid fibers with empty control arrays retain their sampled lines in native views; absent arrays were not established as schema-valid and are not a native finding.

No tests/builds/probes were executed for this read-only pass. Test source coverage was inspected; no claim that earlier green suites validate the newly found scenarios. No production/test changes or commits. Hosted timing scenarios are code traces, not reproduced scheduler traces. Equal-counter owner replacement in `refreshChangedStoreFibers` remains an unconfirmed risk because ordinary overrides increment version; it is not counted. Resident SurfaceQuerySet rebuilding may be expensive but was not timed, so it is not a confirmed performance defect. Intentional current-state job/export actions, explicit native-file-only command tools, and Spiral's scalar coordinate protocol were not mistaken for exact-history readers. The MCP export docstring still names `objectRevision` while the bridge returns complete `version`; this is documentation drift, not another storage finding.
