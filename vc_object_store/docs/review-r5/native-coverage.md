# R5 native consumer coverage

Baseline `bea32d5d8`. GPT-6.1 Sol source review; docs only. Refresh: `git grep -l -E 'vc_object_store|vc::objects|StoreSurfaces|StoreToolOptions|store_channel_loader|load_quad_from_store|SurfaceStoreEdit|PinnedObject|AttachedCollection' -- '*.cpp' '*.hpp' '*.inc' ':!vc_object_store/**'` yielded 141 tracked paths. Followed four indirect callers (`ViewerManager.cpp`, `AgentBridgeHandlers_editing.cpp`, `SegmentationModule_Input.cpp`, `SegmentationEditManager.cpp`), agreeing with the 145-path R4 seed. This pass traces store-facing control flow and relevant caller boundaries, not every unrelated numerical, rendering or test implementation. No test execution, builds or installs.

## Findings and deliberate exclusions

- [N-001](findings/N-001.md): the native `QuadSurfaceStore` adapter overrides shared transformed validity with the raw-file predicate. Separate from the core inverse-edit issue and the R4-fixed Python consumer.
- [N-002](findings/N-002.md): failed anchored correction saves do not veto close/switch through `pointCollectionsSaveRequested`; the R4 restart fix remains intact.
- `SurfaceProximityHit` does not carry the full record pin. The native supported callers use it as UUID discovery (picker/filter) or overlap count (seed growth), not to save edited geometry selected from the hit. No standalone exact-edit failure established.
- Conservative branch-membership conflict rejection, explicit per-patch brush commits and deliberate direct-volume-only point/fiber adapter inputs are existing contracts, not new bugs. Current-state export/restore/lifecycle convenience actions are not assumed to promise a UI-listing snapshot. AFV/constraint SQL is not a second vcobj resolver.
- Declarations and test-only sources are inventoried as such. Tests can be consulted as contract evidence, but no implication that each was executed or that all test internals were independently audited.

| File | Disposition |
| --- | --- |
| `spiral-fitting/cpp/track_crossings.cpp` | Shared numerical/geometry boundary; unrelated kernel body excluded. |
| `volume-cartographer/apps/VC3D/CState.cpp` | N-002: void correction-save request cannot veto closeAll; ordinary collection flush is checked. |
| `volume-cartographer/apps/VC3D/CState.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/CWindow.cpp` | N-002: close/switch prep and teardown ignore the anchored slot's false return; proximity picker is UUID discovery only. |
| `volume-cartographer/apps/VC3D/CWindow.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.cpp` | Selected navigation pin is retained through worker; region and transformed geometry use shared readers. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.cpp` | Loaded branch membership/commit, object refs, affine checks and receipt repinning traced; intentionally conservative guard retained. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/LineAnnotationFiberCacheTest.inc` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/LineAnnotationObjectStoreTest.inc` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/MenuActionController.cpp` | N-002 caller closure: open/reset routes reach CloseVolume; checked failure is required before state replacement. |
| `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.cpp` | Checked ordinary collection flush, complete references and loaded mapping guards traced; not the N-002 anchored slot. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/ProjectResources.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ProjectResources.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SessionDrafts.cpp` | Shared durable checkpoint and pending-receipt resolution; native wrapper owns UI scheduling only. |
| `volume-cartographer/apps/VC3D/SessionDrafts.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp` | Complete loaded point/brush references and shared point allocation; per-patch brush commits documented. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp` | Shared exact/current views; explicit direct binding for point/fiber input projection; no unsupported registration fallback. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp` | Current and geometry-only selectors explicitly distinguished; lazy channels use selected geometry revision. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/StoreImportController.cpp` | Captures destination attachment and frame before background shared ingestion; does not route via filesystem probing. |
| `volume-cartographer/apps/VC3D/StoreImportController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp` | Complete selected object and view passed into common loader; N-001 adapter caller; proximity is discovery only. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp` | Retained history selection/binding and proximity query controls; filters consume UUID discovery, not editable hit state. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/SurfacePanelStoreTest.inc` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/ViewerManager.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_editing.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp` | Shared ingestion/export; explicitly current export selection pins once inside worker. Pending UI-save entrypoint inherits N-002. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp` | Measured length selection and complete selected transformed view; scene-level bounds/budgets are consumer policy. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp` | Shared resident/stored region queries and bounded results; uses displayed selected geometry. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp` | N-002: saveCorrections connected as a bool-returning slot to a void signal; surface-selection guard separately checks result. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Input.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp` | R4 restart guard uses save result; N-002 remains in outer close protocol. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp` | Shared correction collection request preserves loaded baseline; N-002 lies in caller failure propagation. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp` | Shared checked mapping/request and durable receipt paths; loaded correction membership retained. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/VC3D/test/test_fiber_3d_data.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_brush_patch.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_project_inputs.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_service_protocol.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_edits.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_inputs.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_stored_fits.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_spiral_updated_drafts.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_store_region_requests.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/test/test_surface_catalog.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/src/FiberConstraintExtraction.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/FiberConstraintGeometry.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/apps/src/ObjAlphaCompRefinement.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_gen_umbilicus.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` | Shared store mode and overlap query; proximity results are counts, not editable geometry. |
| `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_obj2tifxyz.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_render_tifxyz.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/apps/src/vc_tifxyz2obj.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/include/vc/core/PointCollections.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/StoredCollection.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp` | Shared numerical/geometry boundary; unrelated kernel body excluded. |
| `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp` | Shared numerical/geometry boundary; unrelated kernel body excluded. |
| `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp` | Declaration/adapter boundary checked with caller; no separate physical routing found. |
| `volume-cartographer/core/src/FiberView.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/GridStore.cpp` | MemMap utility consumer only; no vcobj persistence logic. |
| `volume-cartographer/core/src/GrowSurface.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/PointCollections.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/ProjectCreation.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/ProjectDocument.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/QuadSurface.cpp` | N-001 evidence: resident grid validity can represent finite negative coordinates except the full sentinel. |
| `volume-cartographer/core/src/QuadSurfaceStore.cpp` | N-001: selected reader is shared, but raw tifxyz predicate is incorrectly re-applied in the target frame. |
| `volume-cartographer/core/src/StoreSurfaces.cpp` | R4 loaded pin+mapping preserved on writes/artifacts; N-001 read-side conversion delegated to QuadSurfaceStore. |
| `volume-cartographer/core/src/StoredCollection.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp` | Shared numerical/geometry boundary; unrelated kernel body excluded. |
| `volume-cartographer/core/test/support/TestProject.hpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_fiber_constraint_geometry.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_grow_seg_store_mode.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_ink_detection_overlay.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_lasagna_service_manager_stub.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_line_annotation_generated_views.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_mesh_tool_outputs.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_open_data_manifest.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_point_collections.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_point_index.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_quadsurface_save_paths.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_quadsurface_store.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_segmentation_lasagna_panel_ui.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_surface_query_set.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_tool_store_modes.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_viewer_manager_stub.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_viewer_manager_stub.hpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_viewer_overlay_surface_primitives.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_volume_live_s3.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/core/test/test_volume_pkg.cpp` | Test/support inventory; adjacent contract evidence only, no execution. |
| `volume-cartographer/libs/flatboi/flatboi.cpp` | Store-facing caller boundary reviewed; shared routing/selection used, no additional concrete defect established. |
| `volume-cartographer/scripts/benchmark_cleanup.cpp` | MemMap utility consumer only; no vcobj persistence logic. |

145 dispositions. Both findings are source-traced candidates; independent adjudication is recorded separately.
