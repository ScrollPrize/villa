# R4 Python and binding coverage

Baseline: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. Read-only inspection; documentation only, no builds, dependency changes, test execution or fixes in this lane.

## Method and scope

Inspected the shared reading/cleanup contracts, all 14 tracked Python binding/package/build files, and the repository-wide Python direct-use/caller inventory below. Discovery used tracked-file searches for `vc_object_store|object_store_inputs|stored_surfaces|service_commits|surface_store|annotations.fiber_store`, followed producer/consumer and wrapper calls. This covers vcobj-specific control flow, not every unrelated numerical or training body. Test sources were consulted as existing evidence (especially R3 source-version, stored-surface and Spiral input tests); no claim that previous green tests reproduce new findings.

73 explicitly dispositioned files. No finding quota; two findings logged separately: [P-001](findings/P-001.md) is a code-traced transformed-mask correctness issue, [P-002](findings/P-002.md) is a stale documentation contract. Independent cross-model adjudication is pending.

| File | Disposition |
| --- | --- |
| `lasagna/approval_inpaint.py` | P-001: shared transformed-validity mask discarded/recomputed. Artifact publication issue delegated to shared-core lane. |
| `lasagna/fit2tifxyz.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `lasagna/fit_service.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `lasagna/stored_surfaces.py` | P-001: shared transformed-validity mask discarded/recomputed. Artifact publication issue delegated to shared-core lane. |
| `lasagna/tifxyz_io.py` | P-001: shared transformed-validity mask discarded/recomputed. Artifact publication issue delegated to shared-core lane. |
| `spiral-fitting/config.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/find_inconsistent_windings.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/fit_session.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/fit_spiral.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/flatten_spiral_checkpoint.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/grow_track_graph.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/object_store_inputs.py` | Selection/pin capture, frozen record/geometry reads, manifests, materialization boundaries reviewed; explicit same-volume restriction and legacy pin interpretation retained. |
| `spiral-fitting/pcl_views.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/runners/run_single.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/runners/run_sweep.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/service_commits.py` | New derived surface creation/idempotent lookup; checked shared creator. Service-local serialization is explicit, not cross-process deduplication. |
| `spiral-fitting/spiral_runtime.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/spiral_service.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vc_object_store/python/AfvBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/CMakeLists.txt` | Build wiring; no runtime resolution. |
| `vc_object_store/python/ExtensionBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/GeometryRegionBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/ObjectReadBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/SessionBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/StorageBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/TypedGeometryBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/objects.cpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/__init__.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/derivation.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/documents.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/locations.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/surfaces.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/scripts/import_afv.py` | Native importer argument/launch wrapper; source/import implementation belongs to storage lane. |
| `vesuvius/src/vesuvius/fiber_spline_generator/cli.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/fiber_spline_generator/store.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/autoreg_mesh/dataset.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_anchored_dataset.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_constraint_database_validation.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_curve_bank.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction_review.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_full_slab.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_constraint_score.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_inpainting.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_joint_gt_eval.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_json_solve.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_min_flow_benchmark.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_expansion.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_pair.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_normal_sign_visualization.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_piece_split.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_reference_comparison.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_selected_dataset_eval.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_layout.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_reference_subset.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_surface.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/datasets.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_bank.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_snapping.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/location_weights.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/build_location_weights.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/build_index.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/data.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer_all.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume_store.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/train/dagger/collect.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py` | Shared frozen-source/catalog/exact geometry/digest/derived-write boundaries reviewed; direct bound-volume interpretation intentional. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/build_fiber_pool.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |

## Deliberate exclusions and investigated non-findings

- Dependency/bootstrap/setup scripts only provision the module; no runtime object/geometry routing to inspect. No installs were run.
- Python tests/manual smoke scripts are test fixtures, not production consumers; they do not expand the production API inventory. No real-data or hosted service run was attempted.
- Numerical UV/flow/training code reached through the listed adapters consumes arrays, not stores; only source identity/read/write boundaries were inspected. Direct legacy constraint SQLite formats are not vcobj database bypasses.
- Read-only adapters deliberately limited to one directly declared volume do not acquire arbitrary transform support merely because the generic store supports it; this is a documented capability restriction, not a current-state fallback.
- Numeric convenience selectors and explicitly current APIs are not called exact-reference APIs. Missing an exact overload is not itself proof of stale data; the artifact publisher family was reported to the core reviewer for shared handling rather than duplicated here.
- An apparent duplicate `data_checksum` assignment was caused by overlapping terminal output ranges; reinspection of numbered source disproved it. No finding was logged.
- Schema-21 core source/owner semantics and physical storage resolution are delegated to the core/storage reviewers, not independently reimplemented or assumed valid solely from binding delegation.

## Validation needed for findings

P-001 needs an affine-target mask regression; no runtime reproduction is claimed. P-002 is a direct implementation/documentation comparison. Both files contain exact locations, reachable behavior and a shared fix direction. No production files were changed.
