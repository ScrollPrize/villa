# R4 native consumer coverage

Baseline: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. Read-only inspection; documentation only. Scope: C++ vcobj consumers outside `vc_object_store/`, including CLI wrapper/caller closure. Discovery: tracked C++/header/test-include files matching `vc_object_store|vc::objects|StoreSurfaces|StoreToolOptions|store_channel_loader|load_quad_from_store|SurfaceStoreEdit|PinnedObject|AttachedCollection`; four indirect editing/viewer callers added explicitly. 145 file dispositions below. There is no finding quota.

Inspection covers store-facing control flow and source/reference/coordinate/async write boundaries, not every unrelated numerical or rendering body. Tests are evidence only; no builds, dependency installation, GUI execution or temporary probes were run in this lane. Two concrete code-traced findings are individually logged as N-001/N-002; cross-model adjudication remains pending. No production edits.

## Family checks and exclusions

- Surface open/history/channel load uses complete selected states and shared geometry/channel readers. Resident overlay queries reuse cached immutable geometry and shared cancellation/completeness handling; no new defect established in the R3 delta.
- Fiber editor/cache/structural operations retain selected references and a loaded commit baseline. R3 conservative unrelated-commit rejection is intentional, not a new defect. Point/correction IDs use shared retained allocation.
- Ordinary point collections preserve loaded references and mapping checks; CState preflights pending writes before routing/volume changes. Correction selection preflight exists, but same-editor restart is missed (N-001).
- Spiral input documents intentionally support declared direct volume scale for point collections/fibers; absence of registration handling there is not reported as a stale-mapping bug. Brush batch API explicitly documents per-patch commits and the production UI calls it with one patch at a time; no atomicity defect claimed for its multi-item convenience signature.
- CLI current discovery followed by explicit current loading is not treated as an exact-snapshot contract. The actual editable CLI refinement path has a loaded identity mapping but fails to retain/check it at write time (N-002).
- Artifact transport, imports, project save/copy, source extraction and agent-bridge exports call shared session/export/closure APIs. No consumer-local SQLite vcobj resolver was found in this inventory. Legacy AFV/constraint SQLite input handling is not vcobj routing.
- Relationship consumed-identity comparison in Spiral currently uses numeric version and selected geometry rather than content hash; a same-version divergent-owner path within its same-root-store restriction was not established, so this is not promoted to a finding.

## File dispositions

| File | Disposition |
| --- | --- |
| `spiral-fitting/cpp/track_crossings.cpp` | Shared numerical/geometry algorithm consumer; no persistent identity routing. Store-facing interface reviewed, unrelated math excluded. |
| `volume-cartographer/apps/VC3D/CState.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/CState.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/CWindow.cpp` | N-001: correction restart path traced. R3 selection guards remain separate and do not cover restarting the same editor. |
| `volume-cartographer/apps/VC3D/CWindow.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationFiberCacheTest.inc` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/LineAnnotationObjectStoreTest.inc` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/MenuActionController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/ProjectResources.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ProjectResources.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SessionDrafts.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SessionDrafts.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/StoreImportController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/StoreImportController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/SurfacePanelStoreTest.inc` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/ViewerManager.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_editing.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp` | N-001: correction restart path traced. R3 selection guards remain separate and do not cover restarting the same editor. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Input.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp` | N-001: correction restart path traced. R3 selection guards remain separate and do not cover restarting the same editor. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/VC3D/test/test_fiber_3d_data.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_brush_patch.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_project_inputs.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_service_protocol.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_edits.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_inputs.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_stored_fits.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_spiral_updated_drafts.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_store_region_requests.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/test/test_surface_catalog.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/src/FiberConstraintExtraction.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/FiberConstraintGeometry.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/apps/src/ObjAlphaCompRefinement.cpp` | N-002: loaded mapping guard omitted on CLI update; producer/caller traced. |
| `volume-cartographer/apps/src/vc_gen_umbilicus.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_obj2tifxyz.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_render_tifxyz.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/apps/src/vc_tifxyz2obj.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/include/vc/core/PointCollections.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/StoredCollection.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp` | Shared numerical/geometry algorithm consumer; no persistent identity routing. Store-facing interface reviewed, unrelated math excluded. |
| `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp` | Shared numerical/geometry algorithm consumer; no persistent identity routing. Store-facing interface reviewed, unrelated math excluded. |
| `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp` | Declarations/data-model wrapper reviewed with implementation caller; no independent persistence resolver. |
| `volume-cartographer/core/src/FiberView.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/GridStore.cpp` | Excluded from vcobj semantics: uses vcobj MemMap utility only, not an object-store reader/writer. |
| `volume-cartographer/core/src/GrowSurface.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/PointCollections.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/ProjectCreation.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/ProjectDocument.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/QuadSurface.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/QuadSurfaceStore.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/StoreSurfaces.cpp` | N-002: loaded mapping guard omitted on CLI update; producer/caller traced. |
| `volume-cartographer/core/src/StoredCollection.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp` | Shared numerical/geometry algorithm consumer; no persistent identity routing. Store-facing interface reviewed, unrelated math excluded. |
| `volume-cartographer/core/test/support/TestProject.hpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_fiber_constraint_geometry.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_grow_seg_store_mode.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_ink_detection_overlay.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_lasagna_service_manager_stub.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_line_annotation_generated_views.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_mesh_tool_outputs.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_open_data_manifest.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_point_collections.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_point_index.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_quadsurface_save_paths.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_quadsurface_store.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_segmentation_lasagna_panel_ui.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_surface_query_set.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_tool_store_modes.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_viewer_manager_stub.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_viewer_manager_stub.hpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_viewer_overlay_surface_primitives.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_volume_live_s3.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/core/test/test_volume_pkg.cpp` | Test/support evidence inspected for adjacent contract coverage; not a production resolver; no execution in this review. |
| `volume-cartographer/libs/flatboi/flatboi.cpp` | Store-facing calls/reference flow reviewed; delegates physical routing to shared APIs. No additional concrete defect established. |
| `volume-cartographer/scripts/benchmark_cleanup.cpp` | Excluded from vcobj semantics: uses vcobj MemMap utility only, not an object-store reader/writer. |
