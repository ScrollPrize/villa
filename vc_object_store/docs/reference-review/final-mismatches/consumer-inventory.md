# Consumer coverage

Frozen baseline: `153c9a10f` with tracked working diff SHA-256 `286b96ae90b5960043810003370f6ec3a66110597251dbcdd39477b313501235`. The preceding nine PF fixes remain in that diff and are not changed by this review. Review documents and automatically cleaned diagnostics only.

The reproducible consumer screen below includes 128 production files (109 volume-cartographer, 9 Spiral, 5 Lasagna, 5 Vesuvius). The root followed matching identity, version, selection, geometry, artifact, guard and lifecycle calls, plus indirect wrappers. This is broad screening plus the detailed traces below, not a claim to have proved every line or every concurrency interleaving in those files. Core/native coverage and explicit omissions are in `astra-inventory.md`.

## Detailed invariant traces and dispositions

- Python store facade (`locations.py`, `surfaces.py`, `documents.py`, `selected.py`): most adaptation delegates native handling; numeric surface coercion is FM-P03. Deliberately limited hosted query argument support fails closed and is documented, not a second resolver. Immutable geometry payload reads, selected reference facades and application content digests are legitimate.
- Lasagna `stored_surfaces`, `fit_service`, `fit2tifxyz`, `tifxyz_io`, `approval_inpaint`, `fit`: selected references flow unchanged through native decoding, surfaces and artifacts. Path/type dispatch is application-boundary routing. Artifact publication helper is intentionally current/new-output selection; its observed producer creates the output UUID first. No automatic ToolObjects-style successful-write repinning is proposed.
- Spiral `object_store_inputs`, service request positions, fitting startup/status, `grow_track_graph`: FM-P01/P02. Selected input graphs and full-reference receipts use native paths. Application consumed-metadata/content hashing and model checkpoint format versions are not alternate object handles. Server minimum-position/receipt semantics are legitimate application rules, but basic input decoding should be shared.
- Vesuvius surface adapter, fiber catalog/training reader, generated-store writer and row/column output provenance: delegates surface reads; FM-P03 reaches the surface adapter. Fiber direct-volume policy is explicit and the PF fix is retained. Geometry-derived counts/lengths, complete-reference evidence digests, source-change checks, new-object absent guards and geometry-only immutable payload lookups remain appropriate. Training model behavior and neural tracing algorithms are excluded.
- Native `QuadSurfaceStore`, `StoreSurfaces`, ToolObjects and SurfaceCatalog: PF selected-view, stale-no-op and source-frame fixes retained. Channels read immutable selected geometry payloads. ToolObjects read-baseline semantics remain deliberate. ObjectReference-based surface edits and retained mapping guards flow through shared helpers.
- `SpiralStoreInputs`, native fit-input tests and Python manifest validator: FM-C02 ingress mismatch; current-head content comparison itself is intentional. Domain consumed-metadata projection is explicit and parity tested, not a duplicated physical store resolver.
- `StoredCollection`, `PointCollectionSync`, `SurfaceCorrections`, standalone `PointCollections` boundary: FM-C01. Checked collection writes otherwise retain references and mappings; structural allocation uses retained native allocation; UUID-based GUI membership adapters are not persistent store directories. No-op UI paths with no pending edits are distinct from an invoked checked save.
- `LineAnnotationController` selected loads, deferred line reads, cached geometry digests, graph baselines, save preparation and mapped control-point attributes: full pins/mapping retained; immutable payload reading plus the retained displayed frame is appropriate for deferred lines. UI editor state and pending submission grouping are deliberate version-handling adapters. No independent new defect established by the root trace.
- ProjectDocument/ProjectCreation/ProjectWorkspace/ProjectResources and registrations: project-file resources, application route selections and explicit current discovery are separate from vcobj object identity. Store copies route through shared closure APIs; umbilicus import/binding and coordinate registration delegate native handling. Attached resource-grid metadata collection is application-specific, not a second vcobj resolver.
- Remaining GUI overlays, catalog/history, segmentation/growth, queued draft jobs, import commands and CLI StoreSurfaces users: screened guard/reference/geometry/artifact uses and followed shared wrappers. Existing loaded pins, guarded drafts and geometry export delegation preserved. Rendering/geometry algorithms were not audited numerically. Build/bootstrap packaging matches are not runtime store policy. Legacy-named external geometry tools/files are not evidence of runtime old-vcobj-schema acceptance.

## Independent-review limitations

Astra completed the core/hosted/bindings trace and independent adjudication of FM-P01/P02. A second C++ lane identified FM-C02 before failing at its model usage limit; the additional Python lane also failed at that limit. The root completed the above source traces and runtime probes instead. No final independent-model approval is claimed for FM-P03 or FM-C01/C02. No production fix is included in this review.

## Screened files

- `lasagna/fit2tifxyz.py`
- `lasagna/fit_service.py`
- `lasagna/scripts/bootstrap_venv.py`
- `lasagna/setup.py`
- `lasagna/stored_surfaces.py`
- `spiral-fitting/config.py`
- `spiral-fitting/cpp/track_crossings.cpp`
- `spiral-fitting/fit_session.py`
- `spiral-fitting/fit_spiral.py`
- `spiral-fitting/grow_track_graph.py`
- `spiral-fitting/object_store_inputs.py`
- `spiral-fitting/runners/run_single.py`
- `spiral-fitting/service_commits.py`
- `spiral-fitting/spiral_service.py`
- `vesuvius/src/vesuvius/fiber_spline_generator/cli.py`
- `vesuvius/src/vesuvius/fiber_spline_generator/store.py`
- `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py`
- `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py`
- `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py`
- `volume-cartographer/apps/VC3D/CState.cpp`
- `volume-cartographer/apps/VC3D/CState.hpp`
- `volume-cartographer/apps/VC3D/CWindow.cpp`
- `volume-cartographer/apps/VC3D/CWindow.hpp`
- `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp`
- `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp`
- `volume-cartographer/apps/VC3D/FiberCollectionController.cpp`
- `volume-cartographer/apps/VC3D/FiberCollectionController.hpp`
- `volume-cartographer/apps/VC3D/FiberMapStaleness.hpp`
- `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp`
- `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp`
- `volume-cartographer/apps/VC3D/MenuActionController.cpp`
- `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp`
- `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp`
- `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp`
- `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp`
- `volume-cartographer/apps/VC3D/PointCollectionSync.cpp`
- `volume-cartographer/apps/VC3D/PointCollectionSync.hpp`
- `volume-cartographer/apps/VC3D/ProjectResources.cpp`
- `volume-cartographer/apps/VC3D/ProjectResources.hpp`
- `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp`
- `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp`
- `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp`
- `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp`
- `volume-cartographer/apps/VC3D/SegmentationCommandHandler.hpp`
- `volume-cartographer/apps/VC3D/SessionDrafts.cpp`
- `volume-cartographer/apps/VC3D/SessionDrafts.hpp`
- `volume-cartographer/apps/VC3D/SpiralPointCollectionEdit.cpp`
- `volume-cartographer/apps/VC3D/SpiralPointCollectionEdit.hpp`
- `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp`
- `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp`
- `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp`
- `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp`
- `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp`
- `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp`
- `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp`
- `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp`
- `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp`
- `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp`
- `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp`
- `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp`
- `volume-cartographer/apps/VC3D/StoreImportController.cpp`
- `volume-cartographer/apps/VC3D/StoreImportController.hpp`
- `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp`
- `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp`
- `volume-cartographer/apps/VC3D/SurfacePanelController.cpp`
- `volume-cartographer/apps/VC3D/SurfacePanelController.hpp`
- `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp`
- `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp`
- `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp`
- `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp`
- `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp`
- `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp`
- `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp`
- `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp`
- `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp`
- `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp`
- `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp`
- `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp`
- `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp`
- `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp`
- `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp`
- `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp`
- `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp`
- `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp`
- `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp`
- `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp`
- `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp`
- `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp`
- `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp`
- `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp`
- `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp`
- `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp`
- `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp`
- `volume-cartographer/apps/src/FiberConstraintExtraction.cpp`
- `volume-cartographer/apps/src/FiberConstraintGeometry.hpp`
- `volume-cartographer/apps/src/ObjAlphaCompRefinement.cpp`
- `volume-cartographer/apps/src/vc_gen_umbilicus.cpp`
- `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp`
- `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp`
- `volume-cartographer/apps/src/vc_obj2tifxyz.cpp`
- `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp`
- `volume-cartographer/apps/src/vc_render_tifxyz.cpp`
- `volume-cartographer/apps/src/vc_tifxyz2obj.cpp`
- `volume-cartographer/core/include/vc/core/PointCollections.hpp`
- `volume-cartographer/core/include/vc/core/StoredCollection.hpp`
- `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp`
- `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp`
- `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp`
- `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp`
- `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp`
- `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp`
- `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp`
- `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp`
- `volume-cartographer/core/src/FiberView.cpp`
- `volume-cartographer/core/src/GridStore.cpp`
- `volume-cartographer/core/src/GrowSurface.cpp`
- `volume-cartographer/core/src/PointCollections.cpp`
- `volume-cartographer/core/src/ProjectCreation.cpp`
- `volume-cartographer/core/src/ProjectDocument.cpp`
- `volume-cartographer/core/src/QuadSurface.cpp`
- `volume-cartographer/core/src/QuadSurfaceStore.cpp`
- `volume-cartographer/core/src/StoreSurfaces.cpp`
- `volume-cartographer/core/src/StoredCollection.cpp`
- `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp`

