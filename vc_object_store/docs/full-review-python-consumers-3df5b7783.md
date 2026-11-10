# Full vcobj review: python-consumers

Baseline: `3df5b7783`. Fresh full-scope review of all assigned relevant files, not only the recent diff. Previous findings R2-1–7 remain in [the combined findings](coherent-store-review-3df5b7783.md); corroborations are not counted again. Every assigned path requires a current disposition and inspected symbols/evidence. No finding quota. Follow additional callers and append rows if needed; report why pure numerical/build-only paths are excluded. Review only: no production fixes or commits.

## Coverage (66 assigned files)

| File | Disposition | Inspected behavior / evidence |
| --- | --- | --- |
| `lasagna/approval_inpaint.py` | Reviewed | load_approval_surface retains one StoredSurface for XYZ, approval and displacement channels; no separate current lookup. |
| `lasagna/fit.py` | Reviewed | Surface/approval loading entrypoints pass store, volume and target scale through shared adapters; remaining fit operations use materialized arrays. |
| `lasagna/fit2tifxyz.py` | Reviewed | _store_surface, correction/atlas attachments, side-file artifact publication and owned-session cleanup use common store/surfaces/documents APIs; output geometry is newly created. |
| `lasagna/fit_service.py` | Reviewed | Stored surface/model selectors, correction collections and atlas/job-spec resolution delegate to stored_surfaces/documents. Complete references preserved when supplied; no proven additional concurrent-current defect. |
| `lasagna/scripts/bootstrap_venv.py` | Excluded | EXCLUDE: dependency discovery/installation commands only; inspected object-store install wiring, not executed. |
| `lasagna/setup.py` | Excluded | EXCLUDE: package/dependency declaration only; no object resolution or storage operations. |
| `lasagna/stored_surfaces.py` | Reviewed | Selector parsing, StoreSession artifact lookup and StoredSurface channel reads retain selected owner/reference and geometry; publication handles newly created outputs. |
| `lasagna/tifxyz_io.py` | Reviewed | Stored-surface load branch delegates common read_surface; file-format branch is an explicit different source type. |
| `spiral-fitting/config.py` | Declaration | DECLARATION: object-store selectors, location, positions and policy settings consumed by object_store_inputs; no independent storage path. |
| `spiral-fitting/cpp/track_crossings.cpp` | Reviewed | Numerical crossing wrapper delegates shared TrackCrossings implementation; no database lookup or storage placement. |
| `spiral-fitting/find_inconsistent_windings.py` | Reviewed | build_fit_inputs/main use normal fit input loading with refresh disabled and deliberate current-input analysis; no private SQL. |
| `spiral-fitting/fit_session.py` | Reviewed | Fit-input/config/checkpoint manifests transport pins; path-input descriptors explicitly exclude store-managed geometry. |
| `spiral-fitting/fit_spiral.py` | Reviewed | _load_object_store_inputs, _resume_input_pins and load_replacement_inputs retain complete InputPins; runtime algorithms consume frozen materialization. |
| `spiral-fitting/grow_track_graph.py` | Reviewed | New surfaces use create_surface; stored_seed_surfaces/retire_surfaces use public local-store selection and checked writes. CLI is explicitly local. |
| `spiral-fitting/object_store_inputs.py` | Finding / corroboration | FINDING PY-F4: materialize_fibers rejects valid empty controls with a nonempty dense line (runtime reproduced). Freshly inspected selection/pins, graph reads, exact geometry/overlap materialization, link requests/receipt repinning and refresh paths. |
| `spiral-fitting/pcl_views.py` | Reviewed | Pure transformations of already-materialized point/relationship collections; no storage resolution. |
| `spiral-fitting/runners/run_single.py` | Reviewed | Store CLI forwarding, UUID-based resume identity and overwrite protection use shared location classification; no direct database reading. |
| `spiral-fitting/service_commits.py` | Reviewed | find_committed_fit/write_commit/commit_in_place use public selection, create_surface and current revision; commit artifacts are new outputs, not historical copies. |
| `spiral-fitting/spiral_runtime.py` | Reviewed | Input preparation/replacement, checkpoint and adoption paths carry materialized FitInputs/pins; no physical storage routing. |
| `spiral-fitting/spiral_service.py` | Reviewed | Binding/open, admission/receipt checks, checkpoint pins, prepare_position and commit paths use shared store facade; location-specific startup validation is explicit. |
| `vc_object_store/python/vc_object_store/derivation.py` | Reviewed | Shared portable versus geometry_only metadata policy; rejects managed payload copying and records explicit selected source. No private payload resolution. |
| `vesuvius/src/vesuvius/fiber_spline_generator/__init__.py` | Declaration | DECLARATION: package documentation only. |
| `vesuvius/src/vesuvius/fiber_spline_generator/cli.py` | Reviewed | Preview/final publication delegates write_store with explicit L0 volume frame; other store variables refer to CT Zarr, not vcobj. |
| `vesuvius/src/vesuvius/fiber_spline_generator/store.py` | Reviewed | write_store uses typed Fiber and one checked creation transaction, self-cleaning staging and exclusive publication; newly generated geometry, not a store-copy API. |
| `vesuvius/src/vesuvius/neural_tracing/autoreg_mesh/dataset.py` | Excluded | EXCLUDE: keyword-discovery false positive; stored surface means an already-loaded tifxyz lattice, not vcobj. No vcobj API/selector. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_anchored_dataset.py` | Reviewed | Selection/export paths use shared digest/read_polylines/write_fibers; explicit derived stores and round-trip check, not placement-specific SQL. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_constraint_database_validation.py` | Reviewed | Direct SQLite is the separate legacy constraint DB, not vcobj. Source membership and bounded geometry checks use fiber_catalog/read_polylines. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_curve_bank.py` | Finding / corroboration | RELATED PY-F2: raw polylines paired with frozen normal curves by UUID only; no source-version binding in this helper. Checked sample/identity loading. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction.py` | Reviewed | Preparation verifies feature source and reads selected polylines via shared helper; direction optimization is outside storage scope. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction_review.py` | Reviewed | Delegates primary visualization to normal-sign exporter (PY-F2); selected/rejected geometry uses shared polyline reader. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_full_slab.py` | Reviewed | Input signatures use fibers_digest; subset and solve preparation delegated to common workflow components. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_constraint_score.py` | Reviewed | Combined test/GT geometry read/write through shared helpers; volume and duplicate-ID checks; constraints are independent workflow files. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_inpainting.py` | Reviewed | GT source fingerprint and immutable tensor/solution input signatures; no vcobj structural reads beyond shared digest. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py` | Reviewed | export_features checks before/after digest; verify_feature_source uses source identity/length checks. SQLite is separate constraint DB. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_joint_gt_eval.py` | Reviewed | Combined derived store creation and splitting use shared readers/writer; selected data stay workflow-owned, no vcobj SQL. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_json_solve.py` | Reviewed | Explicit legacy JSON conversion validates scan/frame and round-trip geometry through write_fibers/read_polylines; no pretending to copy store-managed attachments. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_min_flow_benchmark.py` | Reviewed | Seam geometry loads explicit volume and selected polylines through shared adapter; solver math is not storage handling. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_expansion.py` | Reviewed | Source signature, catalog selection, curve cache, round exports use shared adapters; caches scoped to immutable workflow input signature. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_pair.py` | Reviewed | Paired workflow uses selected-fiber digest and normal dispatch; no independent storage resolver. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_normal_sign_visualization.py` | Finding / corroboration | FINDING PY-F2: export validates store UUID but mixes current line geometry with old constraint positions/normals. Runtime reproduced. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_piece_split.py` | Reviewed | split_graph reads source curves via shared helpers; split objects here are solver tensor pieces, not persistent vcobj identity allocation. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_reference_comparison.py` | Reviewed | Projection signatures use source digest; samples use shared reader and piece IDs; no physical store assumptions. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_selected_dataset_eval.py` | Reviewed | Selected/source geometry equivalence and digest checks; all vcobj reads delegated, outputs are solver files. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_soft_winding.py` | Reviewed | Global prior sampling uses explicit volume/shared polyline reader; no private geometry reconstruction from vcobj tables. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_layout.py` | Reviewed | prepare/cache validation uses shared source fingerprints/catalog/polylines; direct SQLite refers only to separate constraint DB. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_reference_subset.py` | Reviewed | Checks parent input digest and cache signatures before source sampling; source curves via common read_polylines. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_surface.py` | Reviewed | Input hashes and CurveBank-based materialization checked; shared stale-normal binding concern belongs to PY-F2, not another finding. |
| `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` | Finding / corroboration | EXISTING R2-7 corroborated: chained load/save records original source. Complete exact input read and output derivation otherwise shared. |
| `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py` | Reviewed | read_surface/create_surface delegate shared surfaces API, retain complete reference in extra; only XYZ/ZYX/tifxyz adaptation. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/datasets.py` | Reviewed | open_fiber_source and build_mixed_dataset use FiberStore, shared source guards, split policies and manifest entries. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_bank.py` | Reviewed | Neighbor geometry, spatial records and provenance delegated to FiberStore; local KD trees built from guarded selected geometry. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_snapping.py` | Reviewed | snap_store carries source reference and explicit geometry_only derivation policy to shared writer; legacy JSON branch is distinct format. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/location_weights.py` | Reviewed | Cache key uses shared complete-reference digest; geometry comes from FiberStore callers. Non-store numerical/config cache design not claimed validated. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/build_location_weights.py` | Reviewed | Store input uses FiberStore geometry and UUID keys; cache publication delegated to location_weights, no physical store reads. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/snap_fibers.py` | Reviewed | Explicit source-format CLI dispatch to snap_store versus snap_json; shared adapter owns vcobj handling. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/build_index.py` | Finding / corroboration | FINDING PY-F1 producer: persists catalog ordinal in runs_fiber and digest in meta; downstream ignores digest. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/data.py` | Finding / corroboration | FINDING PY-F1: RunFibers applies cached ordinals to fresh catalog without validating recorded digest. Runtime A retired -> B selected. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py` | Finding / corroboration | FINDING PY-F1 consumer: Scorer also matches current catalog ordinal to old StripIndex. Inference geometry itself uses FiberStore. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer_all.py` | Finding / corroboration | FINDING PY-F3: resume done set contains UUID only; modified same-UUID inputs silently reuse old outputs. Fresh geometry uses FiberStore. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/train.py` | Finding / corroboration | FINDING PY-F1 consumer: make loader/RunFibers has no independent digest check. Rest is training on materialized inputs. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume.py` | Reviewed | Final vcobj output delegates volume_store.export_store; trace tile files/ID allocation are separate source data, not vcobj internals. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume_store.py` | Reviewed | Finished-tile adapter derives stable UUID5 keys then shared write_fibers staging; explicit volume binding. Replaces generated output, not editing historical store. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/train/dagger/collect.py` | Reviewed | FiberStore-specific supervision dispatch uses FiberBank; no separate storage/identity interpretation. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py` | Finding / corroboration | Full adapter inspected: selected graph reads, source-position guards, complete-reference digests, derived-write policy, spatial block reads. CPP-F3 also affects catalog use of stored summaries. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/build_fiber_pool.py` | Reviewed | FiberStore catalog/geometry used for immutable standalone pool output; UUID keys retained. No vcobj placement bypass. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py` | Reviewed | Parallel workers use guarded FiberStore (reopen per process); outputs UUID-tagged numerical matches. No SQLite/owner lookup. |

## Additional findings

### PY-F1 — P2: persisted sheet-strip catalog ordinals are reused against a different catalog

`sheet_strip/build_index.py:43–57` persists catalog ordinals in `runs_fiber.npy` and records `fibers_digest` in metadata. `sheet_strip/data.py:40–64,87–96` loads that metadata but applies the ordinals to a freshly opened FiberStore without validating the digest. `train.py:275–278` and `infer.py:264–272` use the same unchecked association. Retiring an earlier fiber changes the ordinal's meaning: an old index for A can silently sample B. Complete-reference validation already exists in the shared FiberStore but is not applied across this persisted-index boundary.

Runtime reproduction created A/B, recorded index unit 0 and the original digest, retired A, then constructed the real RunFibers against the old index. It returned B; the saved digest differed and no error was raised. Fix the common index/source binding: validate the recorded catalog identity or persist stable selected references and map them against the captured source. Test retirement/reordering and same-UUID geometry updates.

### PY-F2 — P2: solved normal evidence is combined with newer fiber geometry

`fiber_trace_3d/fiber_normal_sign_visualization.py:258–277` validates the solution's constraint-file checksum but checks its source fibers only by store UUID. At line 329 it loads current polylines, while lines 361–377 use old constraint coordinates/normals. Editing geometry within the same store is therefore accepted. CurveBank's current-geometry/frozen-normal association is another affected boundary, not a separately counted finding.

Runtime reproduction created valid solution/constraint fixtures, translated a source fiber by 100 voxels through a checked sparse geometry edit, then called the actual exporter. It succeeded with one fiber and one arrow: the OBJ started at `(110,110,110)` while the arrow remained at `(10,10,10)`. Reject stale evidence or read its exact recorded source through a shared evidence/source-selection helper; all output components must use that selection. No consumer-local placement resolver is needed.

### PY-F3 — P2: sheet-strip resume treats unchanged UUID as unchanged input

`sheet_strip/infer_all.py:121–139` populates its completed set from only the logged fiber UUID. Output/log provisioning at lines 147–162 does not retain a complete source reference or revision digest. After a same-UUID geometry edit, restarting with the same output directory skips the changed input and silently reuses stale output. FiberStore's live-session change detection cannot protect this reopen/resume boundary. Code-traced, not GPU-runtime-tested. Completion reuse must compare the selected input identity; add a same-UUID/new-geometry resume test. This differs from PY-F1's ordinal misassociation: the UUID is correct here, but its revision is not.

### PY-F4 — P2: Spiral's fiber adapter rejects valid traced fibers without controls

`spiral-fitting/object_store_inputs.py:1400–1405` rejects every fiber with fewer than one control point before considering its nonempty dense line. Runtime reproduction stored a valid Fiber geometry with `control_point_ids` shape `[0]`, `control_points` shape `[0,3]` and `line_points` shape `[2,3]`; the actual `materialize_fibers` raised `ObjectStoreInputError: fiber … has no control points`. The source passes native schema validation. Native rendering accepts this empty-controls case. The rejected missing-array variant is NOT claimed valid: required named arrays and nonempty control data are different requirements.

Handle the valid dense-line-only representation in the shared fiber-span/materialization boundary without inventing controls. Keep control/branch checks for data that actually has them; test both valid representations. This is an older consumer assumption, not a new schema requirement.

All four are missed existing consumer paths rather than fixes introduced by this documentation pass. Existing R2-7 is only corroborated. Stored fiber summary usage also corroborates the native lane's CPP-F3 and must not be counted twice.

### BIND-F1 — P1: Python submit bypasses the shared lifetime guard and can crash

During cross-lane completion, the coordinator also reviewed all native Python binding and Python package files; their per-file dispositions are in the transport ledger, not added to the 66-file count above. `vc_object_store/python/objects.cpp:1557–1562` releases the GIL and retains only a raw ObjectStore reference through prepare/submit. In contrast, `StorageBindings.hpp:351–380` provides the shared `withStore` lease and rejects close while leased. A Python progress callback can re-enter `close()`, which accepts closure because this submit path never took a lease. The native operation then uses a destroyed ObjectStore.

Runtime reproduction used the real two-surface `link_objects` request and progress callback calling the same store's `close()`. The callback printed `CLOSE SUCCEEDED DURING SUBMIT`; the isolated Python child exited with SIGSEGV (-11). Core dumps were disabled, parent-owned temporary fixtures were automatically cleaned, and no existing database was touched. `start_history_generation` at lines 1568–1570 has the same unleased GIL-release pattern (code-traced sibling, not separately reproduced). Route these operations through the common lifetime wrapper; test reentrant close and concurrent close without deliberately allowing use-after-free. The submit/progress path predates the latest review delta (`7e6b82164d`/`2144fdd3f3`); this is a missed caller, not a newly introduced change in this review.

A separate suspicion about `expected_version` being reduced to a geometry guard was tested and ruled out: after metadata advanced version 2 to 3, a geometry edit with expected version 2 correctly raised `is at version 3, not 2; reload before committing`. It is not a finding.

## Evidence and limits

Record executed probes separately from code traces. No claim that earlier green tests validate newly found cases. Physical database access is legitimate inside the responsible storage layer; flag duplication or consumer bypass, not all SQL.

All 66 assigned paths have fresh dispositions. Relevant vcobj entry points, their source/version propagation, persistent outputs/caches and shared adapter calls were inspected; unrelated numerical/training algorithms are not claimed validated. The extra `autoreg_mesh/dataset.py` discovery is an explicit false positive (in-memory tifxyz lattice, not vcobj). The three runtime probes used `/home/hendrik/.venv_las/bin/python`, existing native bindings under `local/vcobj-pr-tests/build/python`, and self-cleaning TemporaryDirectory fixtures under `local/`. No installs, full training, production edits or regression-test changes were performed. PY-F3 is code-traced only. Full-suite reruns would not replace the missing regression cases identified here.
