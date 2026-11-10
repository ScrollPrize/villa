# R3 native-consumer review

Baseline: `cf8406afc`. Read-only review of native consumers outside `vc_object_store/`, including the last fix delta. Production code was not changed. Findings below are source-traced, **not runtime-reproduced** in this pass. No test/build/install commands were run; previous suite totals are not evidence for these cases.

## Results

Six distinct native-consumer issues remain. N1's stale correction-snapshot direction is introduced/exposed by the last correction-baseline fix; its reverse surface-pin problem predates that fix. N2–N6 are missed sibling paths, not evidence of six new regressions. No finding quota was used. These are not six added storage features: the remedies should preserve and share existing selected-state/edit contracts.

### N1 — Correction and surface editors invalidate each other's baseline during their own workflow

**Priority: high. Evidence: code trace. Classification: last-fix regression in correction save, existing reverse pin gap.**

- `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp:392` loads `_correctionsSnapshot` once. `:413` saves it through `SurfaceCorrections.cpp:42`.
- `vc_object_store/src/AttachedDocument.cpp:221` checks the loaded commit sequence, and `:246` checks the loaded owner reference. The correction adapter repins only that correction snapshot.
- `SegmentationModule.cpp:2740` (autosave) and `:2900` (ordinary save) publish the surface and repin only `SurfaceStoreEdit`; neither advances the correction snapshot.
- `segmentation/growth/SegmentationGrower.cpp:1147` saves the surface, `:1163` adds growth evidence, and `:1170`–`:1182` adjusts/saves correction anchors. Thus the next correction save necessarily presents an older commit, even without another user.
- Conversely adding/removing a correction attachment can publish the local owner surface through relationship endpoint bookkeeping, but `SurfaceStoreEdit::_pin` is not repinned from that receipt.
- `SegmentationModule_Session.cpp:425` catches the failed correction save and returns no success result; `SegmentationModule.cpp:598`–`:612` proceeds to switch/reload corrections, so unsaved correction edits may then be discarded.

This is not merely the documented conservative rejection of unrelated external commits. The same editor issues both dependent writes, yet has no coordinated baseline advancement. Before the R2-11 fix, corrections reconstructed current state on save, so the newly introduced strict correction snapshot did not have this self-conflict; reverting to that unsafe behavior is not a fix.

**Shared fix direction:** coordinate surface/correction/evidence saves with shared receipts or one transaction, updating all affected loaded baselines only for verified own changes. Do not simply refresh current heads and erase real external conflicts. Propagate save failure to navigation.

**Required regression:** one session loads both editors; surface save then correction save; correction attachment add/remove then surface save; growth plus correction-anchor shift; repeat with referenced owner and history off/on.

### N2 — Fiber saves reconstruct relationship membership from current state instead of the loaded graph

**Priority: high. Evidence: code trace. Classification: missed sibling of loaded-collection guard fixes.**

- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp:15470`–`:15507` guards loaded fiber objects, but peers outside the edited set are read afresh at `:15491`.
- `:15562`–`:15590` reads current incident edges. `:15640`–`:15696` merges loaded branch indices/state against those current edges; `:15703`–`:15716` removes any current branch missing from the loaded retained-branch set.
- `LineAnnotationStoredFiber.hpp` keeps branch UUID and control-point indices but not a selected relationship reference/membership baseline or loaded peer reference.
- There is no loaded relationship-membership guard or `expectedCommitSequence` in this save path. `expectRelationshipEndpoints` (`vc_object_store/src/StoreReader.cpp:152`) protects endpoint objects, not the edge selection.
- `vc_object_store/src/History.cpp:319`–`:329` dirties physically local endpoints of relationship changes; immutable inherited endpoints need not change. Therefore object guards do not cover the overlay-only branch case.

**Trigger/impact:** load inherited A, add a branch A–B in another overlay session, then save A from the old editor. Current edge discovery sees the new branch but the loaded branch list does not; save can remove the unseen branch while the inherited A/B object expectations still pass. Similar stale approval/anchor changes can be overwritten; fresh peer loading can apply old array indices to a new peer geometry.

**Shared fix direction:** retain the selected graph/edit baseline, including relationships/membership and peer identities, then use shared optimistic guards. Do not replace the save baseline with fresh peer/edge reads.

**Required regression:** inherited endpoints with concurrent branch add/delete/approval and peer point insertion/reordering; unchanged loaded fiber geometry must not bypass the graph guard.

### N3 — Orthogonal-plane overlays substitute current geometry for a displayed historical surface

**Priority: medium. Evidence: code trace. Classification: missed sibling of the resident/exact surface fix.**

- `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp:38` copies visible runs into a consumer type without the complete `objectReference`.
- `:240`/ `:328` derive UUID targets and query the catalog's current region.
- `:391`–`:400` chooses resident versus stored rendering only by whether the session has a draft. A displayed historical surface without a draft goes down the current-region path.
- `:508`–`:517` can combine those current run grid coordinates with normals from the active historical `QuadSurface`.
- In contrast `:584`–`:585`, the surface-viewer path correctly uses a resident query over the displayed grid.

**Trigger/impact:** open a retained surface version whose current geometry differs. XY/XZ/YZ overlay lines depict the head instead of the displayed version; offset lines can mix current cuts with historical normals.

**Shared fix direction:** both viewer types must query the same selected/resident displayed surface and retain the run's complete selection/mapping. Keep current catalog browsing separate from explicitly displayed selections.

**Required regression:** old grid versus moved/resized current grid, in orthogonal and surface viewers, with offset lines enabled.

### N4 — Ordinary point/correction editors still allocate persistent IDs from visible maximum

**Priority: medium. Evidence: code trace. Classification: missed sibling of fiber/Spiral retained-ID fixes.**

- A fresh `PointCollections` instance starts its local counter at 1 (`core/include/vc/core/PointCollections.hpp:152`); `volume-cartographer/core/src/PointCollections.cpp:735`–`:742` advances it using only currently installed points, not retained allocations. `:765` returns the next increment. Reloading within the same live instance can preserve its higher counter; reopening does not.
- `core/src/StoredCollection.cpp:17`–`:29` treats those IDs as persistent structural IDs.
- `apps/VC3D/PointCollectionSync.cpp:91`–`:125` and `segmentation/SurfaceCorrections.cpp:21`–`:26` serialize them unchanged, without the shared retained allocator.
- `vc_object_store/src/AttachedDocument.cpp:62`–`:117` preserves desired IDs on update; `src/geometry/StructuralGeometry.cpp:139` correctly rejects a previously used identity being inserted as a new point.

**Trigger/impact:** save IDs 1,2,3; restore a version with only 1,2; reopen the collection UI; adding a point chooses 3 again. Saving fails instead of assigning a fresh identity. Deleting the highest ID then reopening has the same failure. This is a usability/persistence failure, not proof that the shared validator permits silent corruption.

**Shared fix direction:** distinguish transient/new UI points from retained identities in the shared collection adapter; allocate new persistent IDs through the existing retained-allocation API and repin the UI from the receipt. Do not weaken structural validation or add another counter.

**Required regression:** ordinary and anchored corrections, deletion/reopen/append and history restore/reopen/append, including referenced collection owners.

### N5 — Coordinate-binding dialogs guard newly read heads rather than the selection used in the dialog

**Priority: medium. Evidence: code trace. Classification: older missed loaded-edit sibling.**

- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp:15763`–`:15783` reads the pinned fiber metadata/geometry and shows its bounds in the binding dialog, but `:15785` uses `editCurrent` and constructs the bind operation from the fresh head.
- `SurfacePanelController.cpp:758`–`:770` similarly displays catalog-row coordinates then binds the fresh head after the modal dialog.

**Trigger/impact:** another writer changes geometry or the coordinate declaration while the dialog is open. A scale chosen using the old bounds/units silently applies to the new state; the otherwise-correct fresh transaction guard cannot detect this user-observation conflict.

**Shared fix direction:** carry the selected reference and displayed frame across the dialog and submit a loaded-state guarded bind operation. Current-state actions not derived from displayed geometry can remain explicitly current.

**Required regression:** mutate geometry/coordinates between prompt acceptance and submit; reject and ask to reload rather than applying the stale interpretation.

### N6 — Historical surface activation still displays current/stale Lasagna attached results

**Priority: medium. Evidence: code trace. Classification: missed sibling of exact artifact/source selection.**

- `volume-cartographer/apps/VC3D/SurfacePanelController.cpp:670` opens a complete historical surface selection.
- `CWindow.cpp:2584`–`:2591` drops that selection and calls UUID-only `attachedDocument`.
- `vc_object_store/src/AttachedDocument.cpp:148` selects current attachments and current attached collection records.
- `CWindow.cpp:4854`–`:4875` and `:4952`–`:4967` only reload the result display when the UUID changes, so switching versions of the same UUID can retain results from the previous selection unchanged.

**Trigger/impact:** review a surface version before/after a Lasagna results update. The surface geometry is correctly historical while correction/results evidence is current or leftover from another version. This path must not imply that the displayed results correspond to the displayed geometry.

**Shared fix direction:** pass the selected owner reference through a shared exact attached-evidence read, with its existing endpoint/source provenance validation; reload by complete selection, not UUID. Where historical evidence cannot be resolved consistently, clear/show unavailability instead of current results.

**Required regression:** two surface versions with different attached result evidence, same-UUID version switch, and retained reference after overlay/deletion. Do not invent a collection-wide cross-store timestamp snapshot.

## Coverage and method

Discovery seed: `rg -l 'vc_object_store|vc::objects|vcobj|ObjectVersion|StoreReader' volume-cartographer spiral-fitting/cpp --glob '*.{cpp,h,hpp,inc}'`, plus the previous native ledger and indirect callers. It produced 150 paths, all given a disposition below. A repository-wide native search outside these directories found only the numerical `spiral-fitting/cpp/track_crossings.cpp` consumer. `vc_object_store/` core/transport code is another review lane; relevant shared helpers were traced here only to prove consumer behavior.

“TRACED” means the relevant paths/callers were followed, not every unrelated UI/numerical line. “STORE-BOUNDARY SCREEN” means all matched store boundaries and their immediate control flow were inspected; it is not a claim to full numerical/UI verification. Headers were checked against implementation. Tests are explicitly listed as coverage-gap inputs, not claimed freshly executed. Build scripts/docs are outside production runtime review; the existing read/cleanup contract and previous ledger were read.

| File | Disposition | Evidence/scope |
| --- | --- | --- |
| `spiral-fitting/cpp/track_crossings.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/apps/VC3D/CState.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/CState.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/CWindow.cpp` | TRACED | N6; traced surface activation, volume switching, attached result display, event invalidation. |
| `volume-cartographer/apps/VC3D/CWindow.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/CWindowContextMenu.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/CommandLineToolRunner.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.cpp` | TRACED | Rechecked selected geometry counts and graphReadViewAround fixes; region/navigate/cache. |
| `volume-cartographer/apps/VC3D/FiberCollectionController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/FiberCollectionGenerator.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/FiberCollectionGeneratorDialog.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/LasagnaServiceManager.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationCompoundTest.inc` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.cpp` | TRACED | N2, N5; traced selected geometry/mappings, branch reconstruction, object guards, receipt repinning, binding dialog, cache invalidation. |
| `volume-cartographer/apps/VC3D/LineAnnotationController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationFiberCacheTest.inc` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/LineAnnotationNearbyBenchmark.inc` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/LineAnnotationObjectStoreTest.inc` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/LineAnnotationStoredFiber.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/MenuActionController.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/MenuActionController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/ObjectStoreArtifacts.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/ObjectStoreRegions.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/OpenDataSampleProject.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.cpp` | TRACED | N4; traced load/flush/receipt mapping and StoredCollection adapter. |
| `volume-cartographer/apps/VC3D/PointCollectionSync.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/ProjectResources.cpp` | TRACED | Read current resource discovery, guarded import/retire delegation. |
| `volume-cartographer/apps/VC3D/ProjectResources.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/ProjectStoreCounts.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.cpp` | TRACED | Store-facing methods inspected; attachment/status/count/routing APIs; numerical/widget layout excluded. |
| `volume-cartographer/apps/VC3D/ProjectWorkspace.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/SessionDrafts.cpp` | TRACED | Read shared draft load/save/discard/callback lifecycle. |
| `volume-cartographer/apps/VC3D/SessionDrafts.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp` | TRACED | Read project/umbilicus complete references. |
| `volume-cartographer/apps/VC3D/SpiralProjectInputs.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/SpiralServiceManager.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.cpp` | TRACED | Read loaded guards/mappings, retained allocation and receipt repinning. |
| `volume-cartographer/apps/VC3D/SpiralStoreEdits.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.cpp` | TRACED | Read live-vs-saved comparison, selected geometry/annotations and relationships. |
| `volume-cartographer/apps/VC3D/SpiralStoreInputs.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.cpp` | TRACED | Read selected fit geometry/source metadata and frame conversion. |
| `volume-cartographer/apps/VC3D/SpiralStoredFits.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.cpp` | TRACED | Traced selected-input serialization, asynchronous comparison and edit publishing. |
| `volume-cartographer/apps/VC3D/SpiralWorkspace.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/StoreImportController.cpp` | TRACED | Read shared ingest and binding/identity handling. |
| `volume-cartographer/apps/VC3D/StoreImportController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.cpp` | TRACED | Traced exact/current open, selected geometry/mappings, region/proximity and shared readers. |
| `volume-cartographer/apps/VC3D/SurfaceCatalog.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp` | TRACED | N5, N6; traced exact history activation, bind dialog, restore/retire. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/SurfacePanelStoreTest.inc` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/VCAppMain.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/VolumeRegistrations.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_session.cpp` | TRACED | Store-facing status/project/resource methods inspected. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp` | TRACED | Traced shared import/export/attach paths, selected export reference and destination handling. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeInternal.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/agent_bridge/AgentBridgeServer.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.cpp` | TRACED | Rechecked R2 selected-reference fix and active loading path. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DData.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/fibers3d/Fiber3DPanel.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/overlays/InkDetectionOverlayController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/overlays/SegmentationOverlayController.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.cpp` | TRACED | N3; read both plane/surface rendering paths, target/run conversion, caches and drafts. |
| `volume-cartographer/apps/VC3D/overlays/StoreSurfaceOverlayController.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.cpp` | TRACED | N1; traced synchronous/asynchronous saves, activation, correction save calls. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SegmentationModule_Session.cpp` | TRACED | N1; traced correction snapshot lifetime, read/write/error handling and editing state. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.cpp` | TRACED | N1, N4; read collection adapters, write baseline and growth annotation creation. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceCorrections.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.cpp` | TRACED | N1; read guarded saves, coordinate checks and receipt repinning. |
| `volume-cartographer/apps/VC3D/segmentation/SurfaceStoreEdit.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.cpp` | TRACED | N1; traced pre/post-growth surface saves, annotation commit and correction-anchor save. |
| `volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrower.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/segmentation/tools/SegmentationEditManager.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/VC3D/test/test_command_line_tool_runner.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_fiber_3d_data.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_fiber_collection_generator.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_brush_patch.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_project_inputs.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_service_protocol.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_edits.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_store_inputs.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_stored_fits.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_spiral_updated_drafts.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_store_region_requests.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/test/test_surface_catalog.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/VC3D/viewer_controls/panels/VolumeRegistrationsPanel.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/apps/src/FiberConstraintExtraction.cpp` | TRACED | Read native read-transaction, mapping and selected batch geometry extraction. |
| `volume-cartographer/apps/src/FiberConstraintGeometry.hpp` | TRACED | Read selected geometry/array shape derivation. |
| `volume-cartographer/apps/src/vc_gen_umbilicus.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_grow_seg_from_segments.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_obj2tifxyz.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_obj2tifxyz_legacy.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_render_tifxyz.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/apps/src/vc_tifxyz2obj.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/core/include/vc/core/PointCollections.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/StoredCollection.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/types/ProjectCreation.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/util/QuadSurface.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreSurfaces.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/util/StoreToolOptions.hpp` | DECLARATIONS | Store-facing types/ownership declarations matched to implementation; no separate resolver. |
| `volume-cartographer/core/include/vc/core/util/SurfaceArea.hpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/include/vc/core/util/SurfaceQuerySet.hpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/include/vc/fiber_tracer/FiberJson.hpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/src/FiberView.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/src/GridStore.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/src/GrowSurface.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/src/PointCollections.cpp` | TRACED | N4; traced installCollections/getNextPointId/addPoint and serialization. |
| `volume-cartographer/core/src/ProjectCreation.cpp` | TRACED | Read store creation/open and atomic project setup boundaries. |
| `volume-cartographer/core/src/ProjectDocument.cpp` | TRACED | Read project serialization/identity handling; no direct store schema interpretation. |
| `volume-cartographer/core/src/QuadSurface.cpp` | STORE-BOUNDARY SCREEN | Current store-facing calls and surrounding control flow screened; shared session/store adapters used. Non-store UI/numerical bodies excluded. |
| `volume-cartographer/core/src/QuadSurfaceStore.cpp` | TRACED | Read selected metadata/geometry/frame loading, channel loaders and saves. |
| `volume-cartographer/core/src/StoreSurfaces.cpp` | TRACED | Read CLI shared selected reads, write/source/artifact contracts. |
| `volume-cartographer/core/src/StoredCollection.cpp` | TRACED | N4; read point-ID JSON conversion and creation mapping. |
| `volume-cartographer/core/src/fiber_tracer/FiberSideStripIntersections.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/core/test/support/TestProject.hpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_fiber_constraint_geometry.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_grow_seg_store_mode.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_ink_detection_overlay.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_lasagna_service_manager_stub.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_line_annotation_generated_views.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_mesh_tool_outputs.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_open_data_manifest.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_point_collections.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_point_index.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_quadsurface_save_paths.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_quadsurface_store.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_segmentation_lasagna_panel_ui.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_surface_query_set.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_tool_store_modes.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_viewer_manager_stub.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_viewer_manager_stub.hpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_viewer_overlay_surface_primitives.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_volume_live_s3.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_volume_pkg.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_volume_pkg_full.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/core/test/test_volume_pkg_more.cpp` | TEST/OPTIONAL | Fixture/reference discovery and regression-gap screen; not a fresh exhaustive assertion audit or test run. |
| `volume-cartographer/libs/flatboi/flatboi.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |
| `volume-cartographer/scripts/benchmark_cleanup.cpp` | NUMERIC/FILE UTILITY | Shared numerical, UUID, mmap or file-format utility use; no persistent object/version resolver. |

### Indirect consumers without discovery-keyword matches

- `apps/VC3D/SeedingWidget.cpp`: strict completeness before saved intersection annotations rechecked.
- `apps/VC3D/ViewerManager.cpp`: displayed/resident query delegation and selected frame handling screened.
- `apps/VC3D/overlays/ViewerOverlayControllerBase.cpp`: projection/cache keys screened; no persistent lookup.
- `apps/VC3D/segmentation/tools/SegmentationEditManager.cpp`: resident edits and shared draft publication screened.
- `apps/VC3D/SpiralPanel.cpp/.hpp`: UI forwarding only; no independent persistent resolver.
- Native CMake files from the prior 132-path inventory: build registration only; not rereviewed as runtime storage semantics.

## Explicit limits and non-findings

- No GUI runtime, concurrency injection, macOS run, packed/full-volume benchmark or full test suite was run in this review.
- No new raw SQLite consumer was found in this lane. The remaining defects mainly **discard a loaded selection or edit baseline while still calling shared APIs**; API usage alone is insufficient evidence of coherent state.
- Number-only equality in fiber cache invalidation remains a design-risk search hit, but an equal-number owner-switch repro was not established; it is not counted separately.
- Geometry-only transformed metadata selection in `SurfaceStoreEdit` was examined, but the current exact write guard prevents the suspected stale edit; no distinct confirmed failure is counted.
- CLI candidate discovery followed by explicitly current per-object load is not reported as an exact-snapshot bug without an exact-snapshot contract.
- The previous R2 3D-fiber pin, selected-count, partial-intersection, hosted-transform and graph-closure fixes were checked at their consumers; they are not re-counted as new findings.
