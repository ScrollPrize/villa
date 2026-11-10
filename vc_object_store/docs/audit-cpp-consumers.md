# vcobj coverage: cpp-consumers

Review baseline: `557a6f925`. Scope: shared handling, identity/retention/placement correctness and duplication, not an exhaustive numerical/security audit. This inventory is generated from tracked core sources and repository-wide consumer search; add newly discovered wrapper/caller files with their discovery reason. No implementation edits during this review.

A disposition applies to the relevant vcobj symbols/paths, not every unrelated line in a large file. Allowed dispositions: PENDING, REVIEWED (name symbols/shared contract checked), FINDING (local ID and scenario), DECLARATION (corresponding implementation), EXCLUDED (specific reason). REVIEWED means no additional issue found in the stated path, not proof of correctness. No finding quota. Report all substantiated defects, combine duplicates, and keep unknowns visible. Code-traced and runtime-tested evidence must be distinguished.

## File ledger

| File | Disposition | Symbols / evidence / finding |
| --- | --- | --- |
| `volume-cartographer/apps/VC3D/CState.cpp` | REVIEWED | attach/detach/prepare/route/restoreStoreRoutes, session polling and drafts; shared session except explicit offline converter invocation. |
| `volume-cartographer/apps/VC3D/CState.hpp` | DECLARATION | Session attachment/route declarations and inline session accessor; CState.cpp traced. |
| `volume-cartographer/apps/VC3D/CWindow.cpp` | FINDING | CPP-3 current-surface refresh partial identity; source routing/frame-switch paths use SurfaceCatalog/session. |
| `volume-cartographer/apps/VC3D/CWindow.hpp` | DECLARATION | Frame switching declarations; CWindow.cpp traced (CPP-3). |
| `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp` | REVIEWED | onVisLasagnaObj uses shared catalog/model reference; service request intentionally uses current UUID. |
| `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp` | REVIEWED | exportedObjCopy uses shared materializeArtifacts; command outputs use StoreSurfaces APIs. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.cpp` | FINDING | CPP-2: navigate drops selected pin; region control/edge overlays reread current metadata after region read. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.hpp` | DECLARATION | Cached pin/catalog row definitions; navigation and overlays traced in CPP-2. |
| `volume-cartographer/apps/VC3D/FiberCollectionGenerator.hpp` | DECLARATION | Generator request/output preview protocol; no vcobj record resolver. |
| `volume-cartographer/apps/VC3D/FiberCollectionGeneratorDialog.cpp` | REVIEWED | UI generator request/output selection only; no record resolution or persistence. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp` | REVIEWED | setObjectStore passes session location; surfaceHasModel uses session artifacts; service identity checked. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp` | DECLARATION | Weak session and service-input protocol declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/LineAnnotationCompoundTest.inc` | EXCLUDED | VC_TESTING fixture/benchmark harness, not a production vcobj resolver; inclusion guards checked. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.cpp` | FINDING | CPP-3/4; current export, staged import, runtime IDs, structural edit and async nearby-query paths traced; separate retained geometry paths use GeometrySnapshot. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.hpp` | DECLARATION | Store pins, numerical-version caches and edit methods; CPP-3/4. |
| `volume-cartographer/apps/VC3D/LineAnnotationFiberCacheTest.inc` | EXCLUDED | VC_TESTING fixture/benchmark harness, not a production vcobj resolver; inclusion guards checked. |
| `volume-cartographer/apps/VC3D/LineAnnotationNearbyBenchmark.inc` | EXCLUDED | VC_TESTING fixture/benchmark harness, not a production vcobj resolver; inclusion guards checked. |
| `volume-cartographer/apps/VC3D/LineAnnotationObjectStoreTest.inc` | EXCLUDED | VC_TESTING fixture/benchmark harness, not a production vcobj resolver; inclusion guards checked. |
| `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp` | DECLARATION | StoredFiber carries complete optional PinnedObject and volume frame; structural edits checked. |
| `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp` | REVIEWED | planFiberMerge/Split preserve frame; compare typed coordinate identity before merge; output persisted by controller. |
| `volume-cartographer/apps/VC3D/MenuActionController.cpp` | REVIEWED | Project reset/create/attach/save, route menus and folder ingest delegate shared project/session/ingest; path logic only UI and source identity. |
| `volume-cartographer/apps/VC3D/MenuActionController.hpp` | DECLARATION | Menu/project/import declarations; implementation delegates shared store APIs. |
| `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp` | REVIEWED | materializeJsonDocuments: attachment+content keyed temporary JSON transport, shared atomic writer. |
| `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp` | REVIEWED | background/RegionRequests/requestRegion cancellation and stale-response token handling. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp` | REVIEWED | createSampleProjectFiles creates fresh store; createOpenDataSampleProject delegates addCatalogSample; resource URL logic not vcobj storage. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp` | DECLARATION | Sample creation declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.cpp` | FINDING | CPP-1: flush drops loaded expected revision; reload/noteRevisions separately reread version. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.hpp` | DECLARATION | Numeric revision cache and flush/reload declarations; CPP-1. |
| `volume-cartographer/apps/VC3D/ProjectResources.cpp` | REVIEWED | Resource import/read/retire uses VolumeUmbilicus/session shared APIs; provenance filename is source metadata. |
| `volume-cartographer/apps/VC3D/ProjectResources.hpp` | DECLARATION | Import/binding plan declarations; shared import implementation traced. |
| `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp` | REVIEWED | retain/request/start key async counts by complete attachment epoch and reject stale callbacks. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp` | REVIEWED | rebuildStores/counts/download/reveal use session APIs and attachment epochs; physical paths only for file UI. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp` | DECLARATION | Store UI/count/download declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp` | REVIEWED | storedSurface/storeForTools delegates SurfaceCatalog; explicit native-tool local-path capability limitation noted in CPP-6. |
| `volume-cartographer/apps/VC3D/SessionDrafts.cpp` | REVIEWED | start/tick/checkpoint/resolvePending delegate session recovery; async result uses request UUID and attachment. |
| `volume-cartographer/apps/VC3D/SessionDrafts.hpp` | DECLARATION | Draft lifecycle declarations; SessionDrafts.cpp traced. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp` | FINDING | Umbilicus protocol drops owner at284–288; cross-lane Python protocol finding, related AO. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp` | DECLARATION | Project input protocol declarations; sender covered by PY-4/AO. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp` | REVIEWED | Store binding/feed/input updates use session location/status; CommitResult parsed without owner but receiver intentionally reloads current UUID; PY-4 covers umbilicus protocol. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp` | DECLARATION | Store binding and CommitResult protocol; result receiver deliberately opens current UUID, not an exact pin. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp` | FINDING | AK retained point-ID allocation; CPP-3 numeric-only edit/receipt tracking. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp` | DECLARATION | Brush/PCL edit and receipt fields are numeric-only; CPP-3 and AK. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp` | FINDING | CPP-7 duplicated geometry decoding/adapter policy; same-volume scalar restriction is intentional fitter contract, not a transform correctness defect. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp` | DECLARATION | ManifestPin/FitInputs declarations; decoder/transform path CPP-7. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp` | REVIEWED | readStoredFit resolves objectAtGeometry/current once, validates frame and loads same revision with pinned lazy channels. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp` | DECLARATION | Fit list/exact read declarations; implementation carries selected revision. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp` | FINDING | CPP-3 numeric edit/receipt state; fitted/current input reads, async cancellation and draft state traced; commit result opens current UUID intentionally. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp` | DECLARATION | SavedObject and input/draft state declarations; CPP-3. |
| `volume-cartographer/apps/VC3D/StoreImportController.cpp` | FINDING | CPP-9 outer ingest exception loses fatalError; attachment/cancel/job lifecycle otherwise shared. |
| `volume-cartographer/apps/VC3D/StoreImportController.hpp` | DECLARATION | Ingest job/options/report declarations; exception-report path CPP-9. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp` | REVIEWED | open preserves complete revisions/view frame; listing/region use shared APIs. CPP-2 class of multi-read listing noted separately. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp` | DECLARATION | Row/toolInput and Opened pin/frame declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp` | FINDING | showHistory keeps owner; restoreRevision drops owner comparison (CPP-3); shared copy/metadata/filter operations inspected. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.hpp` | DECLARATION | History/current surface methods; CPP-3/5. |
| `volume-cartographer/apps/VC3D/SurfacePanelStoreTest.inc` | EXCLUDED | VC_TESTING fixture/benchmark harness, not a production vcobj resolver; inclusion guards checked. |
| `volume-cartographer/apps/VC3D/VCAppMain.cpp` | REVIEWED | Store CLI flag delegates CWindow/CState attachment; guarded test includes excluded. |
| `volume-cartographer/apps/VC3D/ViewerManager.cpp` | FINDING | CPP-5 locate queries current UUID then uses indices on possibly retained displayed surface. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp` | REVIEWED | parseAffine uses shared VolumeAffine validation; text parsing only. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp` | DECLARATION | Shared affine parsing declarations. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp` | REVIEWED | state describes attachment identity; project creation delegates shared createProject; no vcobj SQL. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp` | FINDING | CPP-6 export rejects hosted and reopens physical store because shared exporter lacks StoreReader/session overload; route/import APIs centralized. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp` | DECLARATION | storeReferenceJson delegates state; surfaceRowJson projects UI fields, not an exact pin. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp` | DECLARATION | Bridge route/job/export state declarations; handlers traced. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp` | REVIEWED | loadStore selects fibers, shared transformed views and annotations; snapshot local transaction, no physical store branching. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp` | DECLARATION | StoreSource retains attachment plus target volume; loader traced. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp` | REVIEWED | loadStore request/cancel tokens checked; event invalidation includes fibers and registration/route-choice changes. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp` | FINDING | CPP-5 current catalog published artifact selection for a possibly historical displayed surface. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp` | DECLARATION | Artifact option/cache declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp` | FINDING | CPP-5 overlap color queries discard selected target surfaces and query current UUIDs; CPP-8 class of cache invalidation applies to the unpinned row-wise result; shared query and GUI cancellation traced. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp` | FINDING | CPP-8 missing registration invalidation and stale in-flight cut result fence. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp` | DECLARATION | Region/cut cache state; CPP-8. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp` | REVIEWED | storeEditFor/saveSurface/storePin preserve editor-origin pin; shared SurfaceStoreEdit handles writes. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp` | DECLARATION | StoreEdit and pin resolver declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp` | FINDING | CPP-3 recovered draft partial identity; editCoordinates rereads current metadata when no view (CPP-2 family). |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp` | REVIEWED | attached collections and growth annotations use shared APIs; no local placement queries. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp` | DECLARATION | Correction save/load adapter declarations; implementation uses shared extension edits. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp` | REVIEWED | save uses shared canonicalSurface/quad_rect_edit, mapping guard; repin uses complete receipt reference. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp` | DECLARATION | Pinned edit/save declarations; implementation traced. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp` | FINDING | AL Copy-with-NT serialization loses owner; before/after growth annotation stores geometry revision. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp` | DECLARATION | Source pin and growth output protocol; existing AL. |
| `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp` | FINDING | linkedSurfacesOf/job submission transmits active revision generation/version without owner at1634–1637; cross-lane Python protocol finding. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp` | DECLARATION | DraftTarget retains optional complete base; implementation added below. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp` | REVIEWED | refresh/register/choose use shared volumeTransforms and registration APIs; no placement-specific resolution. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp` | DECLARATION | Shared transform cache and registration UI declarations. |
| `volume-cartographer/apps/src/FiberConstraintExtraction.cpp` | REVIEWED | extract fiber selection/volume transforms/bulk geometry reads share read transaction. |
| `volume-cartographer/apps/src/FiberConstraintGeometry.hpp` | REVIEWED | selectFiberConstraintGeometry selects shared currentGeometry under supplied transaction. |
| `volume-cartographer/apps/src/vc_gen_umbilicus.cpp` | EXCLUDED | Only comment refers to object-store umbilicus; produces external geometry. |
| `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` | REVIEWED | StoreSurfaces load/create and shared nearestSurfaces; derived frame issue tracked CORE-3. |
| `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp` | REVIEWED | Shared object selection, StoreSurfaces load/create/artifacts; CORE-3 at common layer. |
| `volume-cartographer/apps/src/vc_obj2tifxyz.cpp` | FINDING | Temporary source StoreSurfaces at206 discarded before create at236: CORE-3 lost source frame/pin. |
| `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp` | REVIEWED | New object through StoreSurfaces.create; no source revisions selected. |
| `volume-cartographer/apps/src/vc_render_tifxyz.cpp` | REVIEWED | StoreSurfaces.load; lazy channels retain shared session/revision. |
| `volume-cartographer/apps/src/vc_tifxyz2obj.cpp` | REVIEWED | StoreSurfaces load/artifacts; ToolObjects pin publication handled core lane. |
| `volume-cartographer/core/include/vc/core/PointCollections.hpp` | DECLARATION | In-memory collection model, shared UUID generation; persistence adapter in StoredCollection.cpp. |
| `volume-cartographer/core/include/vc/core/StoredCollection.hpp` | DECLARATION | storedCollection/collectionJson; implementation inspected. |
| `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp` | DECLARATION | Project creation option/result declarations; complete implementation traced. |
| `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp` | DECLARATION | AnnotationVolume extends shared VolumeVoxels; project/store descriptor API. |
| `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp` | DECLARATION | Store loader/grid/surface helper declarations; QuadSurfaceStore implementation traced. |
| `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp` | DECLARATION | StoreSurfaces load/create/update; implementation inspected. |
| `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp` | REVIEWED | StoreToolOptions add/from/take; shared geometry scale validation. |
| `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp` | EXCLUDED | Pure surface area/unit helpers, no vcobj persistence. |
| `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp` | DECLARATION | Resident surface spatial query declarations; implementation added below, not current-store rereads. |
| `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp` | EXCLUDED | Raw fiber JSON format validation; shared UUID validation, no store handling. |
| `volume-cartographer/core/src/FiberView.cpp` | EXCLUDED | Uses SegmentClip geometry utility only. |
| `volume-cartographer/core/src/GridStore.cpp` | EXCLUDED | Uses MemMap utility only for independent gridstore format. |
| `volume-cartographer/core/src/GrowSurface.cpp` | EXCLUDED | Uses VertexIndex geometry utility only. |
| `volume-cartographer/core/src/PointCollections.cpp` | EXCLUDED | PointCollection JSON/in-memory model, not vcobj database resolution. |
| `volume-cartographer/core/src/ProjectCreation.cpp` | REVIEWED | createProject creates/validates store via ObjectStore APIs; stages project JSON and cleans only newly created paths. |
| `volume-cartographer/core/src/ProjectDocument.cpp` | REVIEWED | Project schema/descriptor validation and saveAs path rebasing; copying store closures belongs shared Project APIs, not document serialization. |
| `volume-cartographer/core/src/QuadSurface.cpp` | REVIEWED | QuadSurface creation/saveAs shared UUID and geometry utilities; no store routing. |
| `volume-cartographer/core/src/QuadSurfaceStore.cpp` | REVIEWED | load_quad_from_store selected revision, lazy channel loader and surface edit adapters. |
| `volume-cartographer/core/src/StoreSurfaces.cpp` | FINDING | load/update use complete pins; create delegates ToolObjects::coordinatesFor: CORE-3. |
| `volume-cartographer/core/src/StoredCollection.cpp` | REVIEWED | storedCollection/collectionJson: value adapters preserve IDs; no placement resolution. |
| `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp` | EXCLUDED | Geometry/intersection utilities, no store handling. |
| `volume-cartographer/libs/flatboi/flatboi.cpp` | REVIEWED | run_store uses StoreSurfaces load/create/artifacts; CORE-3 common path. |
| `volume-cartographer/scripts/benchmark_cleanup.cpp` | EXCLUDED | VertexIndex performance benchmark, no database consumer. |
| `volume-cartographer/tools/vc3d-mcp/vc3d_mcp/tools/catalog_volume.py` | REVIEWED | RPC facade for project/catalog/volume operations; no direct vcobj reads. |
| `volume-cartographer/tools/vc3d-mcp/vc3d_mcp/tools/stores.py` | REVIEWED | RPC argument/result facade for objects/store/project operations; backend handlers inspected. |

## Additional files discovered by caller/include closure

| File | Disposition | Discovery / symbols checked |
| --- | --- | --- |
| `volume-cartographer/apps/VC3D/SpiralPanel.cpp` | REVIEWED | CommitResult receiver at1849–1870 emits only the UUID to openStoredFit; deliberate current lookup, not reconstruction of an incomplete exact pin. |
| `volume-cartographer/apps/VC3D/SpiralPanel.hpp` | DECLARATION | Commit/openStoredFit signal declarations corresponding to that receiver. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.cpp` | REVIEWED | beginSession/setDraft keep the complete optional DraftTarget base; spatial picking delegates ViewerManager (CPP-5). |
| `volume-cartographer/core/src/SurfaceQuerySet.cpp` | REVIEWED | SurfaceQuerySet operates on supplied resident geometry and SurfaceIndex; no fresh UUID lookup for that geometry. |
| `volume-cartographer/apps/VC3D/SeedingWidget.cpp` | REVIEWED | resolve toolStore and relative winding path intersections via SurfaceCatalog::region; current-store intersections are intentional. |
| `volume-cartographer/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp` | EXCLUDED | ObjectStoreRegions include used only for viewRegion geometric projection helper, no record read/write/cache. |
| `volume-cartographer/cmake/ObjectStore.cmake` | EXCLUDED | Build/link configuration only; no runtime data resolver. |
| `volume-cartographer/scripts/build_macos.sh` | EXCLUDED | Dependency/build setup only; no runtime data resolver. |
| `volume-cartographer/scripts/ci.sh` | EXCLUDED | Build/test invocation only; no runtime data resolver. |

## Findings

All findings below are code-traced, not runtime reproduced. Existing AH–AR IDs are referenced rather than duplicated.

### CPP-1 — Point-collection GUI edits lose their loaded revision

`PointCollectionSync.cpp:90–106` converts edited GUI values to `AttachedCollection` and calls `writeVolumeCollections` without its expected-version argument. That API selects the current head while preparing the write. Thus another editor's intervening edit can be overwritten instead of rejected. `reload` reads values then calls `noteRevisions` in a separate read, potentially recording a newer revision than the loaded values; those revisions are only used for event suppression, not writes. Keep a complete loaded `ObjectVersion` with each collection, submit it through the shared checked-edit API, and repin from the receipt rather than rereading current heads.

### CPP-2 — Fiber navigation/region decoration mixes selected and current versions

`FiberCollectionController.cpp:774–783` captures `selectedPin_` but resolves `GeometrySnapshot(reader).fiber(pin.object.objectUuid)` without its revision. An intervening edit can navigate using a different line (or invalid slider index) than the selected point count. `850–906` obtains `geometryRegion` first, then opens another read and uses `volumeMappings`/current records to transform control points of the retained `run.revisionUuid`, and attaches current branch anchors. A coordinate-frame or geometry edit between these reads produces mismatched line/controls/links. Pass the complete selected revisions through the shared geometry/relationship read APIs, including returned region provenance, instead of fresh UUID-only reads.

`LineAnnotationController.cpp:15086–15102` has the same issue at the typed-view boundary: it holds `head`, records a frame from that record's metadata, but calls `viewTransformedAt(target.volume, target.scale)` without `head`. Geometry-only resolution selects the latest metadata naming that geometry, not the exact metadata revision already read. A metadata-only coordinate edit between hosted reads can make the stored pin/frame disagree with the transformed control points; pass the existing selected state to the shared view overload.

### CPP-3 — Several GUI identity checks still compare partial versions

`SpiralStoreEdits.cpp:31–43,88–100,173–215` and `SpiralWorkspace.cpp:1129–1196,2043` retain only numeric versions for loaded brush patches/point collections and receipts, losing owner and lineage. `SurfacePanelController.cpp:707`, `CWindow.cpp:2735`, `LineAnnotationController.cpp:10834`, and `SegmentationModule_Session.cpp:475` compare generation/version without owner. `LineAnnotationController.cpp:15215,15394–15410` caches endpoint/fiber revisions as numbers. Distinct referenced branches can hold the same object UUID and equal counters but different content; these paths can suppress refresh/restoration or accept an edit/result against the wrong selected identity. Use complete `ObjectVersion`/`PinnedObject::reference()` everywhere and the shared freshness/check APIs. This is the consumer continuation of AN, not a replacement for fixing AN itself. AK separately covers Spiral's retained point-ID allocation.

`LineAnnotationController.cpp:2970–2977` also advances an auto-link working session to the latest pin based only on equal geometry tokens. Its intended case is the operation's own relationship-only edit, but a concurrent metadata/frame edit also leaves the geometry token unchanged. Adopt the operation's actual receipt, or explicitly compare the full selected geometry/frame dependencies before rebasing the session; geometry equality alone does not establish that its voxel-space control coordinates remain valid.

### CPP-4 — Fiber editor preempts shared relationship-retention policy

`LineAnnotationController.cpp:15702–15705` checks the attachment's `reader.status().historyEnabled` and rejects every incident non-`fiber_branch` relationship before emitting `KeepRecordedRelationshipOp`. It does not inspect the relationship owner or whether the edit changes the endpoint. A history-off overlay can reference a history-retaining base with an already retained relationship; even a no-op save hits this unconditional consumer policy. Remove the blanket GUI gate and let the shared relationship operation decide retention requirements from the selected endpoint/relationship owners and actual edit.

### CPP-5 — Surface picking ignores the displayed revision

`ViewerManager.cpp:638–662` accepts a concrete displayed `QuadSurface` but asks `SurfaceCatalog::nearest` for current geometry by UUID, then converts that result's grid indices with the supplied surface. `SurfacePanelController::showHistory` can display an older retained surface. If its current successor changed geometry or grid dimensions, picking the old surface uses the new surface's coordinates/indices. The shared proximity query should accept the displayed complete revision (and draft identity where applicable), or a common resident-geometry query should be used consistently; callers must not reinterpret current-grid results as belonging to an arbitrary displayed grid.

The same selected/current mismatch occurs in `SegmentationOverlayController.cpp:703–714,812–814`: the chosen overlay surfaces are reduced to UUIDs, then each output row queries current `SurfaceCatalog::nearest`, potentially matching another revision than the one shown and even different revisions across rows during a commit. `InkDetectionOverlayController.cpp:325–340` similarly obtains the published ink-file list from the current catalog record for a possibly historical displayed surface. A successor whose geometry/artifact changes can therefore supply the wrong image for the displayed grid. Both should use the selected geometry/artifact dependency identity, not only its UUID.

### CPP-6 — Native export remains an explicitly local-only capability

`agent_bridge/AgentBridgeHandlers_stores.cpp:270–274` rejects hosted stores, and `330–352` opens a new physical `ObjectStore` to export instead of using its attached reader. The shared `GeometryExport` API has ObjectStore/ReadTxn entry points but no StoreReader/session equivalent. This is a **capability gap, not silent corruption**: the error is explicit. A general reader-backed export API would allow the bridge to use the same selected snapshot regardless of attachment placement. Native CLI launch via `SurfaceCatalog::toolStore` has an analogous documented local-file restriction; it needs an explicit transport/materialization design rather than pretending a hosted endpoint is a filename.

### CPP-7 — Spiral GUI duplicates geometry interpretation

`SpiralStoreInputs.cpp:26–73,217–248` manually decodes little-endian geometry array bytes/dtypes/shapes into points and IDs, then applies scale, instead of obtaining typed point/fiber views from the shared geometry layer. Its relationship-family/adapter-metadata interpretation also parallels the Python fitter adapter. This is a **centralization/maintenance finding**, not evidence of presently wrong numerical output. In particular, the fitter deliberately accepts a single volume and scalar frame, so rejection of other registered volumes is not counted as an unsupported-transform bug here. Preserve that domain validation explicitly while moving typed geometry reads and the input-projection contract behind common interfaces/tests; do not silently broaden the model's supported inputs.

### CPP-8 — Surface overlay caches lack complete invalidation/result identity

`StoreSurfaceOverlayController.cpp:60–73` invalidates committed cache only for surface changes, unlike Fiber3DPanel's explicit volume-transform/route-choice invalidation. A registration edit changes mapped positions without touching surfaces; `setSurfaces` at94–100 returns early for the same UUID list, leaving old runs displayed until unrelated movement triggers a new read. At `318–346`, a plane-cut job captures sources/volume but its callback only checks that the viewer pointer still exists. `invalidate` at144–166 clears results without cancelling/versioning that cut job. After switching volume or attachment, the old job can install its former geometry into the new view. Use the existing shared tokenized region request mechanism (including transform dependencies) for cuts as well as ordinary regions. The row-wise overlap job in `SegmentationOverlayController` likewise needs selected revisions/dependency keys (CPP-5), not only local surface-map counters.

### CPP-9 — Import controller converts a fatal infrastructure failure into apparent success

`StoreImportController.cpp:150–159` catches exceptions from the shared ingest call and creates a report containing only `problems`. `AgentBridgeHandlers_stores.cpp:181–196` decides success from the absence of `fatalError`, cancellation and project replacement. Thus an unexpected session/network/attachment failure escaping ingest is reported as a successful zero/partial import with incidental problems, even though infrastructure failures are supposed to terminate the job. Use the shared fatal-report convention for this outer error path; preserve per-object recoverable problems separately. This is error-contract duplication, not a request to introduce a second ingest implementation.

## Coverage limitations

All 118 original inventory files have a disposition, plus 9 caller/search-closure files: 127 unique files total. Relevant vcobj paths were traced through wrappers, read/write adapters, async result handoffs and version caches; unrelated numerical algorithms were not reviewed. The inventory was cross-checked against `rg -l 'vc_object_store|ObjectStore|StoreReader|storePin'` in VC3D/core/scripts/cmake and followed to wrapper implementations not matched by that search. Standalone tests under `core/test`, `apps/VC3D/test` and `agent_bridge/test` are test-only and excluded from production-handler coverage; production-included `.inc` harnesses are individually dispositioned above. Other C++ entry points in the original inventory were traced through StoreSurfaces/ToolObjects; shared store implementation findings are cross-referenced to the core lane.

No production code was changed and no builds or runtime reproduction were performed. This is an exhaustive disposition of the enumerated relevant production paths, not a proof that all possible races or numerical faults are absent. Explicit capability restrictions (CPP-6) and duplication risk (CPP-7) are separated from demonstrated incorrect data/identity/error paths (CPP-1–5,8–9). The previous AH–AR audit findings remain separately tracked; the cross-language owner omissions reference the Python lane rather than duplicating it here.

## Remediation pass

The ledger above records the original review baseline, not the state after these fixes. Runtime verification is separated below from the original code-traced findings.

| Finding | Shared implementation used by consumers | Verification |
| --- | --- | --- |
| CPP-1 | PointCollectionSync carries each loaded AttachedCollection reference and mapping check into writeVolumeCollections; repins from the receipt, not a second head read. Unknown/revalidation and registration events refresh the loaded data. | Shared checked-write tests, syntax check of PointCollectionSync. |
| CPP-2 | Navigation and editor geometry views carry the selected ObjectRevision; region decorations use the captured run reference and affine rather than a fresh frame lookup. StoreReader.graphReadViewAround preserves recorded off-screen peers, keeps explicit visible pins fixed, and delegates applicability to graphReadViewAt. It chooses the newest recorded off-screen version within one owner/lineage and rejects conflicting owners instead of consulting current state. Source conversion retains its StoreSurfaces context through creation. | All 29 affected native translation units pass syntax checks. The recorded-closure regression passes, covering off-screen peers and explicit historical pins. |
| CPP-3 | Brush/PCL drafts, save results, input-update accounting, endpoint caches, restore checks and async edit checks retain complete ObjectVersion identities; all writes use expectObject or reference-bearing AttachedCollection. The request-aware expectRelationshipEndpoints also guards neighboring owners/lineages without replacing caller pins. Auto-link rebasing also compares frame/coordinate identity. | Final syntax and consumer suites pass after the request-aware endpoint migration. Spiral edit tests cover forged owner at equal counters and inherited edits with history on/off; saved-draft regression rejects equal counters from another owner. |
| CPP-4 | Removed the GUI's blanket attachment-history prohibition; KeepRecordedRelationshipOp alone applies the shared selected-owner retention policy. | Shared relationship tests; inherited brush edits in both history modes. |
| CPP-5 | Picking and overlap coloring query immutable displayed grids through SurfaceQuerySet; published ink options use the displayed record metadata. Surface-viewer cuts use ResidentSurface, a thin public wrapper around the existing stored/draft tile intersection kernel. | Resident-grid tests distinguish two versions of one UUID, freeze pointer-space origin, compare cut output to session-draft output and preserve the old grid after mutation. |
| CPP-6 | AgentBridge export uses the attachment's StoreReader and complete selected reference, including hosted stores; it never reopens a physical database file. | Shared reader export tests and AgentBridge syntax check. Native command-line tools that require a local filename still explicitly reject hosted stores; this is an intentional capability boundary, not a fallback resolver. |
| CPP-7 | Removed raw array decoding from SpiralStoreInputs in favor of typed GeometrySnapshot fiber/PCL views. | Spiral input tests pass. The small mirrored fitter metadata projection is an explicit language-boundary domain protocol, parity-tested with Python; it is not generic object-store policy. |
| CPP-8 | Surface overlays invalidate on registration/route changes; plane cuts reuse RegionRequests cancellation/identity fences and capture the displayed immutable grid. | RegionRequests regression rejects completed jobs after refresh/reset and rejects superseded tokens. |
| CPP-9 | Import infrastructure exceptions set fatalError, so the final result cannot report success after an aborted import. | StoreImportController syntax check. |
| AK | Spiral append IDs are assigned by StoreReader.assignGeometryPointIds under the checked snapshot, not by the maximum surviving draft ID. | Spiral edit test removes an earlier allocated ID and confirms it is never reused. |
| AL / PY-4 | Copy-with-NT and project umbilicus/surface requests preserve the complete owner reference across the language boundary. | C++ sender syntax checks plus Python protocol tests from the parallel lane. |

Local reproducible validation uses `bash local/scripts/check_vcobj_consumers` for the affected C++ translation units and `bash local/scripts/run_vcobj_consumer_checks` for the Release consumer harness. Checked-in regressions are wired through the existing `volume-cartographer/apps/VC3D/test/CMakeLists.txt` and `volume-cartographer/core/test/CMakeLists.txt`. No dependency installs or performance-claim benchmarks were performed in this pass.
