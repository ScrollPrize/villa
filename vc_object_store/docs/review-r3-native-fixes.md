# R3 native consumer fixes

Baseline `cf8406afc`. Implement N1–N6 from the native review through shared selected-state, retained-allocation and checked-write APIs. No schema, proxy row or payload-copy additions. Dependencies restored. Do not equate syntax compilation with GUI runtime coverage.

- [x] N1 coordinate surface/correction own commits under the surface editor's existing mutex; only checked own receipts advance the loaded membership baseline. Correction edits revalidate their attachment, growth evidence remains pinned evidence. Failed correction saves keep the loaded data, block navigation and preserve pending growth corrections.
- [x] N2 retain the commit baseline of the loaded fiber graph, including membership. Conflicting writes reject before mutation; other open panes do not silently acquire a newer graph baseline from a receipt. Repeated no-op saves retain the original baseline. This intentionally rejects unrelated external commits too; a more selective membership guard may improve that conservative UX later.
- [x] N3 cut plane-view intersections from displayed resident geometry when present, just as surface viewers already do. Both share the retained resident index and the existing asynchronous bounded cut request; scrolling does not rebuild the whole grid. Keep complete references on stored query runs; invalidate resident intersections on pointer changes and in-place edit signals. Incomplete queries report an error instead of silently installing truncated results.
- [x] N4 seed ordinary and correction UI point allocation from shared retained-ID allocation through `AttachedCollection::nextPointId`; no new persisted counter.
- [x] N5 retain the displayed record/reference across coordinate-binding dialogs and use it for the checked bind, rather than constructing a bind from a fresh head.
- [x] N6 provide exact selected-owner attached-document reads through shared recorded-graph selection; current UUID reads retain current-head semantics through shared decoding. Reload results for same-UUID selections too. Annotation-only document changes explicitly revalidate attachment evidence, not only spatially affected edges.

Build/test: existing `bash local/scripts/run_vcobj_consumer_checks`, `bash local/scripts/check_vcobj_consumers`, and targeted store tests through the approved runners. Tests and actual results are recorded below as work completes.

## Important boundary

A genuinely later geometry edit cannot retain old growth evidence when its source geometry was discarded in history-off mode. It must still reject rather than silently revalidate or drop the evidence. This is distinct from the fixed self-conflict: surface save → attach growth evidence → save correction anchors works, as does correction attachment add/remove → surface save. No implicit history retention or history-off evidence substitution was introduced.

## Regression coverage

`test_surface_catalog.cpp` exercises coordinated surface/correction writes and external conflicts under all local/referenced × history-on/off combinations, removed point ID allocation after reopen, and current/exact attached-document evidence. `LineAnnotationObjectStoreTest.inc` adds stale branch-approval and repeated-no-op guards. `SurfacePanelStoreTest.inc` exercises plane intersections from different displayed grids with the same UUID. These GUI test includes are also enabled in the reusable syntax checker; their execution requires the full VC3D test binary.

The surface panel now asks its editor guard before activation or deselection, leaving the previous selection intact on failed correction writes. Working-level changes preflight the same save. Same-UUID/different-view activation ends the old editor after saving, rather than carrying its correction baseline into the new selection. All `onActiveSegmentChanged` callers catch rejected transitions.

An unchanged correction UI makes no write request: the adapter returns when desired collections equal its loaded/successfully saved UI snapshot. It does not refresh any owner or commit guard. This lets a clean stale editor navigate away after external changes while genuinely changed UI still conflicts; regression coverage checks both behaviors in all four history/reference modes.

Verified: `bash local/scripts/run_vcobj_consumer_checks` passed all eight native consumer executables, including the four correction fixture modes, current/historical attached documents and a changed-coordinate-declaration bind rejection. No full interactive GUI runtime claim is made; the controller/plane regression includes are compiled through the reusable syntax checker.

Final `bash local/scripts/check_vcobj_consumers` passed all 33 translation units, including `SegmentationModule.cpp` and `VC_TESTING` controller regression includes. The final UI no-op adapter and regression were then recompiled and passed the eight-consumer runtime suite. `git diff --check` passed. Native/shared production delta: 335 added / 88 removed (net +247); regression delta: 160 added / 5 removed (net +155). The new code is loaded-editor baseline coordination, guarded navigation and shared resident query routing, not persistent schema overhead. Full interactive VC3D runtime remains untested in this pass.
