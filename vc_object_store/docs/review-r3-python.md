# Round 3: Python consumers and bindings review

Baseline: `cf8406afc`. Review only; no production code changes. This review covers unchanged consumers as well as the last fix round. It found **two confirmed correctness issues**, both missed evidence-provenance boundaries rather than newly introduced routing regressions. There was no finding quota.

## Confirmed findings

### PY-R3-1 — P2: old fiber/patch matches can be certified as matching new fiber geometry

- Producer: `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py:62`, `:85–87`.
- Consumer: `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/build_index.py:43–65`, `:80`.
- Downstream validation: `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/data.py` (`RunFibers` construction).

The match producer persists `fiber:<UUID>`, arc positions, patch coordinates and sampled XYZ, but no complete source revision or source snapshot identity. The index builder takes those previously generated matches, resolves their UUIDs against the **current** catalog, and records the **current** source digest in the new index. Its before/after guard only ensures that the source did not change during index building; it never checks that the matches were generated from that source state. Consequently the downstream digest guard added in R2-15 accepts an index that certifies obsolete matching evidence as current.

Trigger: generate matches, translate or otherwise edit a fiber under its existing UUID, then rebuild the sheet-strip index from the old match JSONL. Even a translation preserving length and sample count is sufficient. Training/inference then combines old patch locations with new fiber geometry without rejecting the mismatch.

Runtime reproduction used the actual `build_index.main`, real vcobj geometry edits and actual `RunFibers`; only the patch-catalog discovery was replaced with a minimal in-memory patch descriptor. A fiber at Y=20 was translated to Y=120 while its old match XYZ stayed at Y=20. The index builder succeeded, stamped the new digest, and `RunFibers` accepted it:

```text
OLD EVIDENCE DIGEST != INDEX DIGEST: True
CERTIFIED INDEX ACCEPTED. RUN Y: 20.0 CURRENT FIBER Y: 120.0
```

Shared fix direction: record the selected source identity when producing the match evidence and validate it when building the index, before interpreting UUIDs, ordinals or arc positions. Reuse the existing source identity/guard primitives; do not construct another per-consumer identity scheme. Evidence with no recorded source identity must be regenerated or explicitly treated as unverifiable, not silently given the current source digest.

Classification: missed older producer/consumer boundary adjacent to R2-15. The downstream guard itself is useful and should remain.

### PY-R3-2 — P2: H/V feature verification does not compare against the constraint DB's recorded source snapshot

- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py:60–72` (`verify_feature_source`).
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py:79–94` (`prepare`).

`verify_feature_source` checks freshly computed feature identity against the source store **as it exists now**, then compares UUIDs and lengths with the constraint database. It ignores the recorded `provenance.fiber_source` snapshot. A length-preserving geometry edit therefore passes when features are regenerated but constraints are not. The later database hash and stage-1 checks only establish that the old constraints and old stage-1 result agree with each other; they do not establish that the new features describe the geometry used to extract those constraints.

Trigger: extract constraints, translate or bend a fiber while preserving its length and UUID, regenerate axial features, then prepare H/V classification using the old constraints/stage-1 output. The solve mixes two different geometric states.

Runtime reproduction used a real vcobj fiber, a constraint DB with the old `fiber_source` identity, a real geometry edit translating the line by 100 voxels, and the actual `verify_feature_source` with a freshly computed feature digest. It accepted the source:

```text
SOURCE CHANGED: True
VERIFY ACCEPTED: {'fibers_digest': '<new digest>', 'fiber_lengths_checked': 1}
```

Shared fix direction: compare the constraint database's recorded source identity with the selected input before combining any derived results. Reuse the source guard already used by the R2-16 UV/split/evaluation fixes. UUID/length equality is not a substitute for source revision identity; it can remain only as a secondary consistency check if useful.

Classification: missed sibling of R2-16, not a new regression from that fix.

## Inventory and coverage

The repository-wide Python consumer search returned **60 files**. Five are declarative/manual-harness exclusions below; the other **55** received a fresh inspection of their vcobj/input/output/cache boundaries. In addition, all **13 source files** in `vc_object_store/python` (eight C++ binding files and five Python helpers) were reviewed. `python/CMakeLists.txt` is build configuration, not a store consumer. The inspection did not stop at the two findings or only examine changed files.

Discovery command:

```sh
rg -l 'vc_object_store|fiber_store|stored_surfaces|open_fiber_source' --glob '*.py' --glob '!**/tests/**' --glob '!**/test_*' --glob '!local/**' --glob '!**/build/**' --glob '!**/.venv/**' --glob '!**/docs/**'
```

### Bindings and shared Python helpers: 13 files

Reviewed native lifetime leases, GIL-release boundaries, conversion/identity codecs, selected geometry reads, mutation delegation, artifact/source references, hosted/local parity and exception handling:

- `vc_object_store/python/{objects.cpp,AfvBindings.hpp,ExtensionBindings.hpp,GeometryRegionBindings.hpp,ObjectReadBindings.hpp,SessionBindings.hpp,StorageBindings.hpp,TypedGeometryBindings.hpp}`.
- `vc_object_store/python/vc_object_store/{__init__.py,locations.py,documents.py,derivation.py,surfaces.py}`.

No additional confirmed finding in this group. In particular, store-level GIL-releasing calls now retain a shared store lease; the selected region input is copied before releasing the GIL; graph reads use the confined shared implementation rather than consumer SQL; exact geometry/artifact helpers carry the complete selected reference. Low-level numeric-version mutation APIs coexist with checked complete-reference transaction APIs; their mere presence is not counted as a new defect.

### Spiral: 6 runtime files

- `spiral-fitting/{fit_spiral.py,grow_track_graph.py,object_store_inputs.py,service_commits.py,spiral_service.py,runners/run_single.py}`.

Reviewed selection capture, retained exact reads, input pins and checkpoint manifests, artifacts/documents, link requests, refresh/repinning, status/cache identity, commit planning and store opens. Numerical fitting internals are not part of this storage review. No additional confirmed finding.

### Lasagna: 5 runtime files

- `lasagna/{approval_inpaint.py,fit2tifxyz.py,fit_service.py,stored_surfaces.py,tifxyz_io.py}`.

Reviewed stored-surface references, geometry/channel/artifact consistency, annotation loading, output publication and service callers. Numerical inpainting/rendering is outside scope. No additional confirmed finding.

### Import, fiber generator and surface inference: 5 files

- `vc_object_store/scripts/import_afv.py`.
- `vesuvius/src/vesuvius/fiber_spline_generator/{cli.py,store.py}`.
- `vesuvius/src/vesuvius/neural_tracing/inference/{infer_rowcol_triplet_wraps.py,surface_store.py}`.

Reviewed ingress delegation, metadata sanitization, source provenance, selected input/output handling and the previously fixed inference selector ordering. No additional confirmed finding.

### Fiber workflow consumers: 22 files

All paths below are relative to `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/`:

- `fiber_anchored_dataset.py`
- `fiber_constraint_database_validation.py`
- `fiber_curve_bank.py`
- `fiber_direction.py`
- `fiber_direction_review.py`
- `fiber_full_slab.py`
- `fiber_gt_constraint_score.py`
- `fiber_gt_inpainting.py`
- `fiber_hv_stage.py`
- `fiber_joint_gt_eval.py`
- `fiber_json_solve.py`
- `fiber_min_flow_benchmark.py`
- `fiber_neighbor_expansion.py`
- `fiber_neighbor_pair.py`
- `fiber_normal_sign_visualization.py`
- `fiber_piece_split.py`
- `fiber_reference_comparison.py`
- `fiber_selected_dataset_eval.py`
- `fiber_soft_winding.py`
- `fiber_uv_layout.py`
- `fiber_uv_reference_subset.py`
- `fiber_uv_surface.py`

Reviewed adapter entry points, source identity checks, derived-cache/result identities, constraint-database provenance and source geometry reuse. Solver/loss numerical internals were not re-audited. PY-R3-2 is the additional confirmed issue.

### Fiber/tracer adapters and training/evaluation consumers: 17 files

All paths below are relative to `vesuvius/src/vesuvius/neural_tracing/tracers/`:

- `shared/annotations/fiber_store.py`
- `fibers/data/{datasets.py,fiber_bank.py,fiber_snapping.py,location_weights.py}`
- `fibers/evaluation/protocol.py`
- `fibers/heading_model/data.py`
- `fibers/scripts/{build_location_weights.py,check_task_sampler.py}`
- `fibers/sheet_strip/{build_index.py,data.py,infer.py,infer_all.py}`
- `fibers/tracing/volume_store.py`
- `fibers/train/dagger/collect.py`
- `sheets/scripts/{build_fiber_pool.py,match_fibers_to_patches.py}`

Reviewed exact catalog/geometry loading, source guard propagation, derived materialized outputs, UUID-versus-ordinal persisted references, before/after digest checks and delegation to the shared adapter. Fully materialized standalone geometry datasets are not automatically stale merely because the original mutable source later changes; reconnection to current geometry is the important boundary. PY-R3-1 is the confirmed failure at that boundary.

### Explicit exclusions from the 60-file search: 5 files

- `lasagna/scripts/bootstrap_venv.py`, `lasagna/setup.py`: dependency/setup declarations, not store reads or writes; no installation run.
- `spiral-fitting/config.py`: configuration declaration, not a store resolver.
- `volume-cartographer/apps/VC3D/agent_bridge/test/{manual_add_method_smoke.py,object_store_smoke.py}`: manual GUI smoke harnesses, not production consumers; native/GUI coverage belongs to the parallel review.

Also excluded: generated build trees, local experiment runners, virtual environments, unrelated Zarr/object-store names, and training/numerical algorithm bodies without a vcobj boundary. Existing tests were inspected selectively, including the fiber-store version, normal-sign visualization and H/V feature tests; this was not a full test-suite run. Native storage, transport, import/export, relationship detection and GUI implementations are covered by the other review partitions, not claimed as independently reviewed here.

## Reproduction environment and limitations

Both probes ran with the already-built local bindings and existing environment:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python:vesuvius/src:spiral-fitting:local/vcobj-pr-tests/spiral-build:. /home/hendrik/.venv_las/bin/python -
```

Fixtures used `tempfile.TemporaryDirectory(dir='local', prefix='review-r3-')` and were automatically removed. No dependencies were installed, no real experiment data were modified, and neither the production implementation nor tests were changed. Linux local-store reproduction establishes the two deterministic evidence-mixing failures; this pass does not claim macOS runtime validation or full-volume performance results.

The common missing invariant is that a derived artifact must carry the identity of the input that produced it all the way through each subsequent stage. A guard against the current source at only the last stage cannot reconstruct identity that an earlier producer omitted.
