# R5 Python and binding coverage

Frozen source: `bea32d5d8`. Sol review; candidates are provisional pending Astra adjudication.

The 73 R4 paths were reconciled against a new tracked Python search for shared store APIs, wrappers and alias imports. The ledger contains 76 unique dispositions: 62 direct Python keyword matches (including two shared module files), 12 additional binding/module/build files, and two indirect Spiral wrappers. Coverage is the combined Python lane (including the root reviewer’s shared binding/adapter tracing). The three additions are the Lasagna packaging/bootstrap paths and the aliased soft-winding fiber-store consumer.

This is a store-facing control-flow inspection, not a numerical-kernel or training audit. Tests were read as contract evidence; no tests, builds or dependency installs were run in this lane. Legacy constraint SQLite is a separate format and is not counted as a vcobj access bypass. Deliberate directly-declared-volume restrictions, explicitly current convenience APIs, and Python-specific ndarray/bytes adaptation are not standalone findings. The prior transformed-mask fix in Python is not the native N-001 problem.

Individual candidates: [P-001](findings/P-001.md) frozen UV evidence race; [P-002](findings/P-002.md) redundant re-export staging; [P-003](findings/P-003.md) duplicated anchor codec (architecture); [P-004](findings/P-004.md) saved UV layout versus freshly read source geometry; [P-005](findings/P-005.md) Spiral frozen pins versus live relationships.

| File | Disposition |
| --- | --- |
| `lasagna/approval_inpaint.py` | Explicit transformed validity survives the adapter in this revision; prior R4 mask issue not repeated. |
| `lasagna/fit_service.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `lasagna/fit2tifxyz.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `lasagna/scripts/bootstrap_venv.py` | Packaging/bootstrap only; no store runtime path or install executed. |
| `lasagna/setup.py` | Packaging dependency declaration only; no store runtime path. |
| `lasagna/stored_surfaces.py` | Explicit transformed validity survives the adapter in this revision; prior R4 mask issue not repeated. |
| `lasagna/tifxyz_io.py` | Explicit transformed validity survives the adapter in this revision; prior R4 mask issue not repeated. |
| `spiral-fitting/config.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/find_inconsistent_windings.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/fit_session.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/fit_spiral.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/flatten_spiral_checkpoint.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/grow_track_graph.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/object_store_inputs.py` | Frozen object identities and surfaces traced; P-005: old Spiral input pins do not freeze relationship membership or endpoint positions. |
| `spiral-fitting/pcl_views.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/runners/run_single.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/runners/run_sweep.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/service_commits.py` | New derived surface creation/idempotent lookup; checked shared creator. Service-local serialization is explicit, not cross-process deduplication. |
| `spiral-fitting/spiral_runtime.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `spiral-fitting/spiral_service.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vc_object_store/python/AfvBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/CMakeLists.txt` | Build wiring; no runtime resolution. |
| `vc_object_store/python/ExtensionBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/GeometryRegionBindings.hpp` | Region ownership/lifetime and bound geometry delegation traced; no independent finding. |
| `vc_object_store/python/ObjectReadBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/objects.cpp` | Binding-level version/lifetime conversions and native delegation traced; P-003: duplicate anchor semantic codec. |
| `vc_object_store/python/SessionBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/StorageBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/TypedGeometryBindings.hpp` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/__init__.py` | Binding/helper lifetime, identity, conversion and delegation reviewed; no independent confirmed defect. |
| `vc_object_store/python/vc_object_store/derivation.py` | Shared derivation keys and publication delegation traced; no independent finding. |
| `vc_object_store/python/vc_object_store/documents.py` | Shared staged document creation/write boundary traced; compare P-002 consumer duplication. |
| `vc_object_store/python/vc_object_store/locations.py` | Location parsing and shared snapshot selection boundaries traced; no independent finding. |
| `vc_object_store/python/vc_object_store/surfaces.py` | Shared transformed read/mask and exact/current selection boundaries traced; no independent finding. |
| `vc_object_store/scripts/import_afv.py` | Native importer argument/launch wrapper; source/import implementation belongs to storage lane. |
| `vesuvius/src/vesuvius/fiber_spline_generator/cli.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/fiber_spline_generator/store.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/autoreg_mesh/dataset.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_anchored_dataset.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_constraint_database_validation.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_curve_bank.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction_review.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction.py` | Signature and geometry selection traced; shares the check-to-read boundary family of P-001. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_full_slab.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_constraint_score.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_inpainting.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_joint_gt_eval.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_json_solve.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_min_flow_benchmark.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_expansion.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_pair.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_normal_sign_visualization.py` | Passes the selected snapshot to the shared polyline reader. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_piece_split.py` | Saved-piece derivation and source-evidence boundary traced; no independent resolver finding. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_reference_comparison.py` | P-004: fresh source geometry/digest can be mixed with a saved older layout. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_selected_dataset_eval.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_soft_winding.py` | Solver-prior polyline read followed to shared fiber adapter; same frozen-evidence family as P-001/P-004, not counted separately. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_layout.py` | P-001: saved signature is checked independently of the later current geometry read. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_reference_subset.py` | UV prepare/signature downstream boundary traced (P-001 family). |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_surface.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py` | Inference publication and shared transformed reads traced; no independent finding. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/datasets.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_bank.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_snapping.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/location_weights.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/build_location_weights.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/build_index.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/data.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer_all.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume_store.py` | P-002: tracing re-export publication path stages the same bytes twice. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/train/dagger/collect.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py` | Shared frozen-source/catalog/exact geometry/digest/derived-write boundaries reviewed; direct bound-volume interpretation intentional. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/build_fiber_pool.py` | Shared adapter call boundary reviewed; numerical/training bodies outside the vcobj inspection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py` | R3 source-evidence guards traced; no bypass of original source guard confirmed. |
